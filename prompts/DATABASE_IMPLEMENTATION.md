# YouTube Watch History — Implementation Guide

*This document is Part 2 of the database documentation, continuing from `DATABASE_FOUNDATION.md`. It explains each script that has been built, how they work, and how to use them.*

---

## 1. Script Overview

The database implementation consists of three scripts, executed in order:

| # | Script                   | Purpose                                      | Required |
|---|--------------------------|----------------------------------------------|----------|
| 1 | `create_schema.sql`      | Creates the database, tables, views, indexes | **Yes**  |
| 2 | `etl_yt.py`              | Loads real Google Takeout data into the DB    | **Yes**  |
| 3 | `populate_synthetic.py`  | Generates synthetic data for testing          | Optional |

---

## 2. `create_schema.sql` — Database Schema

### What It Does

This SQL script builds the entire `youtube_analytics` database from scratch. It creates the `yt` schema (isolating all project tables from the default `public` schema), seven tables following a star schema pattern, analytical indexes, and four views.

### How to Run

The script is executed in **two steps** inside pgAdmin 4:

1. **Step 1:** Connect to the default `postgres` database. Create the `youtube_analytics` database — either by running the `CREATE DATABASE` statement or by right-clicking Databases → Create → Database in the pgAdmin UI.
2. **Step 2:** Connect to `youtube_analytics`, open the Query Tool, and run everything from the `STEP 2 START` marker onward.

### Tables Created

#### Dimension Tables

**`yt.dim_channels`** — One row per unique YouTube channel.
- `channel_id` (SERIAL PK) — Auto-incremented surrogate key.
- `channel_name` (VARCHAR) — Channel display name from the `subtitles` field.
- `channel_url` (TEXT) — Full YouTube channel URL.
- `created_at` (TIMESTAMPTZ) — Row creation timestamp.
- Unique constraint on `(channel_name, channel_url)` prevents duplicates.

**`yt.dim_categories`** — YouTube's official video category taxonomy.
- `category_id` (INTEGER PK) — YouTube's own category IDs (1–44).
- `category_name` (VARCHAR) — Human-readable name (e.g., "Music", "Sports").
- Pre-populated with all 30 standard YouTube categories on schema creation.
- Will be linked to videos during Tier 2 API enrichment.

**`yt.dim_videos`** — One row per unique video ID.
- `video_id` (VARCHAR(20) PK) — Extracted from `titleUrl` (e.g., `hMEyBtsuAJE`).
- `title` (TEXT, nullable) — Video title with `"Watched "` prefix stripped. `NULL` for ghost videos.
- `title_url` (TEXT) — Full YouTube watch URL.
- `channel_id` (FK → dim_channels) — Which channel published the video.
- `status` (VARCHAR) — One of `active`, `unavailable`, `private`. Checked constraint.
- **Tier 2 enrichment fields** (all nullable, populated later via YouTube Data API):
  - `description`, `tags` (TEXT[]), `category_id` (FK → dim_categories), `duration_seconds`, `view_count`, `like_count`.
- **Tier 3 field:** `has_transcript` (BOOLEAN) — Flag for transcript availability.
- `enriched_at` (TIMESTAMPTZ) — `NULL` until API enrichment runs.
- `created_at`, `updated_at` (TIMESTAMPTZ) — Row metadata.

#### Fact Table

**`yt.fact_watch_events`** — One row per watch event. The core of all analytical queries.
- `event_id` (SERIAL PK) — Auto-incremented surrogate key.
- `video_id` (FK → dim_videos) — Which video was watched.
- `channel_id` (FK → dim_channels) — Denormalized from dim_videos for faster star-schema joins.
- `watched_at` (TIMESTAMPTZ) — UTC timestamp of the watch event.
- Unique constraint on `(video_id, watched_at)` — the deduplication key for quarterly uploads.

#### NLP Output Tables

**`yt.nlp_topics`** — Topics identified by NLP models.
- `topic_id` (SERIAL PK), `topic_label` (VARCHAR), `keywords` (TEXT[]).
- `model_name`, `model_version` — Track which model produced the topic.

**`yt.nlp_video_topics`** — Many-to-many junction table linking videos to topics.
- Composite PK on `(video_id, topic_id)`.
- `score` (FLOAT, 0–1) — Relevance score for the topic assignment.
- `ON DELETE CASCADE` on both foreign keys.

**`yt.nlp_embeddings`** — Dense vector embeddings per video.
- `video_id` (PK, FK → dim_videos).
- `embedding` (FLOAT[]) — Stored as array; designed for future migration to `pgvector`.
- `model_name`, `model_version` — Track the embedding model.

#### ETL Log Table

**`yt.etl_log`** — Created automatically by `etl_yt.py` on first run.
- Tracks every ingest run: file name, timestamp, record counts at each filtering stage, new vs. duplicate counts, and duration.
- Enables the application to compute "total videos ever watched" including removed ones.

### Indexes

Optimized for analytical read patterns:

| Index | Table | Column(s) | Rationale |
|-------|-------|-----------|-----------|
| `idx_watch_events_watched_at` | fact_watch_events | `watched_at` | Time-based queries (most common) |
| `idx_watch_events_video_id` | fact_watch_events | `video_id` | Video lookups and joins |
| `idx_watch_events_channel_id` | fact_watch_events | `channel_id` | Channel-based aggregations |
| `idx_videos_channel_id` | dim_videos | `channel_id` | Channel → video joins |
| `idx_videos_status` | dim_videos | `status` | Filter by active/unavailable |
| `idx_videos_category_id` | dim_videos | `category_id` | Category-based queries |
| `idx_video_topics_topic_id` | nlp_video_topics | `topic_id` | Topic → video lookups |

### Views

**`yt.v_watch_history`** — Primary dashboard view. Joins fact_watch_events with all dimensions, converts timestamps to Danish time (`Europe/Copenhagen`), and extracts time dimensions (year, month, day_of_week, hour). One query gives you everything needed for temporal analytics.

**`yt.v_sessions`** — Binge detection view. Uses `LAG()` window functions to compute the gap between consecutive watch events. A new session starts when the gap exceeds **45 minutes**. Outputs `session_id`, `session_video_count`, `session_start`, and `session_end` for each event.

**`yt.v_channel_loyalty`** — Channel loyalty metrics. Aggregates total views, unique videos watched, active days, and the first/last watch timestamp per channel.

**`yt.v_etl_history`** — ETL audit trail. Displays ingest run history with Danish timestamps, filtering counts, and performance metrics.

### Schema Amendment

After initial deployment, one amendment was made:

```sql
ALTER TABLE yt.dim_videos ALTER COLUMN title DROP NOT NULL;
```

This was required to support **ghost videos** — records where the video was deleted from YouTube after being watched. These retain their `video_id` and `title_url` but have `NULL` titles and channels, allowing potential future recovery via the YouTube Data API.

---

## 3. `etl_yt.py` — ETL Pipeline

### What It Does

Reads a Google Takeout YouTube watch history JSON export, classifies each record, cleans and transforms the data, and upserts it into the PostgreSQL database with full deduplication support.

### How to Run

```bash
# Prerequisites
pip install psycopg2-binary

# Update DB_CONFIG password in the script, then:
python etl_yt.py /path/to/watch-history.json

# Or, if you're already in the project folder:
python etl_yt.py watch-history.json
```

### Pipeline Stages

#### Stage 1: Load

- Reads the full JSON file into memory with `json.load()` (safe up to ~500MB; the real file is ~19MB).
- No streaming needed at current data volumes.

#### Stage 2: Filter & Classify

Every record is classified into exactly one of four types:

| Type | Detection Logic | Action |
|------|----------------|--------|
| **Ad** | `details` field contains `{"name": "From Google Ads"}` | Dropped entirely |
| **Removed** | No `titleUrl` field (title is `"Watched a video that has been removed"`) | Dropped, but counted in `etl_log.removed_dropped` for total-watched calculations |
| **Ghost** | Has `titleUrl`, but title after stripping `"Watched "` is a URL (no real title, no channel) | Kept with `NULL` title and `NULL` channel |
| **Clean** | Has `titleUrl`, real title, and channel info | Fully processed |

#### Stage 3: Extract & Clean

For each non-dropped record:

1. **Video ID extraction** — Parsed from `titleUrl` using `urllib.parse` (e.g., `watch?v=hMEyBtsuAJE` → `hMEyBtsuAJE`).
2. **Title cleaning** — `"Watched "` prefix stripped (8 characters). Ghost video titles set to `NULL`.
3. **Channel extraction** — `channel_name` and `channel_url` parsed from the nested `subtitles[0]` dict. `NULL` if missing.
4. **Timestamp** — Passed through as-is (ISO 8601 UTC string); PostgreSQL handles the `TIMESTAMPTZ` conversion.

#### Stage 4: Upsert

Executed in a single database transaction (all-or-nothing):

1. **Channels** — Pre-loads existing channels into memory, then inserts only new ones. Uses `ON CONFLICT (channel_name, channel_url) DO NOTHING`.
2. **Videos** — Pre-loads existing video IDs, then bulk-inserts new videos with `execute_values`. Ghost videos get `status = 'unavailable'`. Uses `ON CONFLICT (video_id) DO NOTHING`.
3. **Watch events** — Bulk-inserts all events with `execute_values`. Uses `ON CONFLICT (video_id, watched_at) DO NOTHING` for deduplication.

If any step fails, the entire transaction rolls back cleanly — no partial data.

#### Stage 5: Log & Report

- Inserts a row into `yt.etl_log` with full metrics.
- Prints a summary report including database totals and the "total ever watched" count (events in DB + removed videos across all ingest runs).

### Deduplication Behavior

Designed for the quarterly upload pattern where each new JSON contains the full history plus the latest 3 months:

- **First run on a file:** All records are new, zero duplicates.
- **Subsequent runs with overlapping data:** Only genuinely new events are inserted. The `UNIQUE(video_id, watched_at)` constraint catches every duplicate.
- **Re-running the same file:** Safe — all events are skipped as duplicates.

### Real-World Performance

Tested against actual Google Takeout exports:

| Run | File Size | Raw Records | After Filtering | New Events | Duplicates | Duration |
|-----|-----------|-------------|-----------------|------------|------------|----------|
| 1st | 19 MB     | 48,407      | 44,346          | 44,346     | 0          | 2.7s     |
| 2nd | —         | 39,700      | 34,830          | 20,078     | 14,515     | 1.5s     |
| 3rd | —         | 45,300      | 40,157          | 4,167      | 35,715     | 1.0s     |

---

## 4. `populate_synthetic.py` — Synthetic Data Generator (Optional)

### What It Does

Generates realistic synthetic YouTube watch history data and inserts it into the database. Used for **testing and validating the schema** before loading real data.

### How to Run

```bash
# Prerequisites
pip install psycopg2-binary faker numpy

# Update DB_CONFIG password in the script, then:
python populate_synthetic.py
```

### What It Generates

| Entity | Count | Notes |
|--------|-------|-------|
| Channels | 150 | Modeled on 8 content niches (boxing/MMA, tech, music, education, gaming, cooking, podcasts, sports) |
| Videos | 2,000 | Realistic titles generated from niche-specific templates |
| Watch events | ~8,000 | Includes binge session clusters and evening-weighted timestamps |
| NLP topics | 8 | One per niche, with realistic keyword lists |
| Video-topic links | ~2,500 | Primary topic + occasional secondary topic per video |

### Key Design Choices

- **Temporal realism** — Watch timestamps are weighted toward evenings (18:00–01:00) to mimic real behavior. 30% of events are generated as binge sessions (3–15 videos in quick succession).
- **Channel loyalty** — Top 10% of videos get 5× the selection weight, simulating re-watches and favorite content.
- **Content realism** — Title templates use domain-specific vocabulary (fighter names, programming languages, artists, etc.) that will produce meaningful TF-IDF and topic modeling results.

### When to Use

- **Before loading real data** — Validate that the schema, views, and queries work correctly.
- **During development** — Test dashboard queries and NLP pipelines without exposing personal data.
- **After schema changes** — Quick smoke test that inserts still work.

### Cleanup

To clear synthetic data before loading real data:

```sql
TRUNCATE TABLE yt.fact_watch_events CASCADE;
TRUNCATE TABLE yt.nlp_video_topics CASCADE;
TRUNCATE TABLE yt.nlp_embeddings CASCADE;
TRUNCATE TABLE yt.nlp_topics CASCADE;
TRUNCATE TABLE yt.dim_videos CASCADE;
TRUNCATE TABLE yt.dim_channels CASCADE;
-- dim_categories is kept (YouTube's standard categories)
```

---

## 5. Current Database State

After running the ETL pipeline on three overlapping exports, the database contains:

| Table | Row Count | Notes |
|-------|-----------|-------|
| `dim_channels` | ~13,500+ | Unique YouTube channels |
| `dim_videos` | ~41,000+ | Unique videos (includes ~2,500 ghost videos) |
| `dim_categories` | 30 | Pre-populated YouTube taxonomy |
| `fact_watch_events` | ~68,500+ | All watch events across all ingested files |
| `etl_log` | 3+ | One row per ingest run |

The `etl_log` tracks a cumulative **~700 removed videos** that were watched but later deleted from YouTube, bringing the "total ever watched" count to **~69,000+**.

---

## 6. File Structure

```
nlp-YouTube-Watch-History/
├── DATABASE_FOUNDATION.md      # Part 1: Architecture decisions & schema design rationale
├── DATABASE_IMPLEMENTATION.md  # Part 2: This file — script documentation
├── create_schema.sql           # Database DDL (tables, views, indexes)
├── etl_yt.py                   # Production ETL pipeline
├── populate_synthetic.py       # Optional synthetic data generator
├── watch-history.json          # Google Takeout export (not committed to git)
├── library.py                  # Legacy Python codebase (reference only)
└── functions.py                # Legacy helper functions (reference only)
```

---

## 7. Next Steps

With the database populated with real data, the following phases are planned:

1. **Tier 2 Enrichment** — Build a YouTube Data API batch script to populate `description`, `tags`, `category_id`, `duration_seconds`, `view_count`, and `like_count` on `dim_videos`.
2. **NLP Pipeline** — Topic modeling (NMF/LDA/BERTopic) on video titles, populating `nlp_topics` and `nlp_video_topics`.
3. **Embedding Pipeline** — Sentence-transformer embeddings for similarity search and recommendation, populating `nlp_embeddings`.
4. **FastAPI Backend** — REST + WebSocket API serving analytics from the database.
5. **React Frontend** — Interactive dashboard visualizing temporal patterns, channel loyalty, sessions, and topic evolution.
6. **Docker Compose** — Container orchestration for PostgreSQL + FastAPI + React.
7. **CI/CD** — GitHub Actions for automated testing and quarterly ingest automation.

---

*This document should be updated as new scripts are added or existing ones are modified.*
