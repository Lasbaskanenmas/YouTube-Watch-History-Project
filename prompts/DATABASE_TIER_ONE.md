# YouTube Watch History — Project Status & Design Analysis

*Part 3 of the database documentation. Covers the current state of the project after completing the Design Analysis & Gap Identification phase. Continues from `DATABASE_FOUNDATION.md` (Part 1: architecture decisions) and `DATABASE_IMPLEMENTATION.md` (Part 2: script documentation).*

---

## 1. Current Project State

The `youtube_analytics` database is fully operational on a local PostgreSQL 16 instance, managed via pgAdmin 4. Real YouTube watch history data has been loaded, deduplicated across multiple overlapping exports, and optimized for analytical queries through a layered system of views and materialized views.

### What Has Been Built

| Component | Status | Description |
|-----------|--------|-------------|
| PostgreSQL database | **Live** | `youtube_analytics` database with `yt` schema, running locally |
| Star schema | **Complete** | 4 dimension tables, 1 fact table, 3 NLP-ready tables |
| ETL pipeline | **Production-ready** | `etl_yt.py` — handles ad filtering, ghost videos, deduplication, logging |
| Materialized views | **Live** | 6 pre-computed analytics layers for dashboard performance |
| Regular views | **Live** | 5 analytical views including session detection and discovery tracking |
| Refresh pipeline | **Ready** | `refresh_views.sql` — single-command refresh of all materialized views |
| Synthetic data tool | **Available** | `populate_synthetic.py` — for testing and development |

### Database Contents

As of the latest ETL run:

| Table | Approximate Rows | Notes |
|-------|------------------|-------|
| `dim_channels` | ~13,500 | Unique YouTube channels |
| `dim_categories` | 30 | Pre-populated YouTube taxonomy |
| `dim_date` | ~4,000 | 2020-01-01 through 2030-12-31 |
| `dim_videos` | ~41,000 | Includes ~2,500 ghost videos with NULL title/channel |
| `fact_watch_events` | ~68,500 | All watch events across multiple ingested exports |
| `etl_log` | 3+ | One row per ingest run |
| `nlp_topics` | 0 | Awaiting NLP pipeline |
| `nlp_video_topics` | 0 | Awaiting NLP pipeline |
| `nlp_embeddings` | 0 | Awaiting NLP pipeline |

The ETL log tracks ~700 removed videos across all ingest runs, bringing the "total ever watched" count to approximately 69,000.

---

## 2. File Structure

```
nlp-YouTube-Watch-History/
├── DATABASE_FOUNDATION.md       # Part 1: Architecture decisions & schema rationale
├── DATABASE_IMPLEMENTATION.md   # Part 2: Script-level documentation
├── DATABASE_STATUS.md           # Part 3: This file — current state & gap analysis
├── create_schema.sql            # Complete DDL: tables, indexes, views, materialized views
├── etl_yt.py                    # Production ETL pipeline (Google Takeout → PostgreSQL)
├── refresh_views.sql            # Post-ETL materialized view refresh commands
├── populate_synthetic.py        # Optional synthetic data generator for testing
├── watch-history.json           # Google Takeout export (not committed to git)
├── yt_sample.json               # 1,000-record sample for development/testing
├── library.py                   # Legacy Python codebase (reference only)
└── functions.py                 # Legacy helper functions (reference only)
```

---

## 3. Schema Overview

### Tables (8)

**Dimension tables:**

| Table | Role | Key |
|-------|------|-----|
| `dim_channels` | One row per unique YouTube channel | `channel_id` (SERIAL) |
| `dim_categories` | YouTube's 30 standard video categories | `category_id` (YouTube's own IDs) |
| `dim_date` | Pre-computed date attributes (2020–2030) | `date_key` (DATE) |
| `dim_videos` | One row per unique video, with Tier 2/3 enrichment fields | `video_id` (VARCHAR, from YouTube URL) |

**Fact table:**

| Table | Role | Key |
|-------|------|-----|
| `fact_watch_events` | One row per watch event | `event_id` (SERIAL), deduplicated on `(video_id, watched_at)` |

**NLP output tables (empty, ready for pipeline):**

| Table | Role | Key |
|-------|------|-----|
| `nlp_topics` | Topic model results | `topic_id` (SERIAL) |
| `nlp_video_topics` | Many-to-many: videos ↔ topics with scores | Composite `(video_id, topic_id)` |
| `nlp_embeddings` | Dense vector embeddings per video | `video_id` (FK) |

**ETL infrastructure (auto-created by `etl_yt.py`):**

| Table | Role |
|-------|------|
| `etl_log` | Audit trail for every ingest run |

### Regular Views (5)

| View | Purpose |
|------|---------|
| `v_watch_history` | Primary dashboard view — full denormalized watch event with Danish time, date dimensions from `dim_date`, channel, and category |
| `v_sessions` | Session detection via 45-minute gap analysis using `LAG()` window functions |
| `v_channel_loyalty` | Channel loyalty metrics: total views, unique videos, active days, watch span |
| `v_etl_history` | ETL audit trail with Danish timestamps (depends on `etl_log` created by ETL script) |
| `v_discovery_events` | Flags first-time channel discoveries and first video watches via `ROW_NUMBER()` |

### Materialized Views (6)

| Materialized View | Purpose | Refresh |
|-------------------|---------|---------|
| `mv_daily_summary` | Daily activity metrics including zero-activity days from `dim_date` | After each ETL run |
| `mv_session_summary` | Pre-computed session-level metrics: video count, duration | After each ETL run |
| `mv_monthly_summary` | Monthly activity rollup: views, unique videos/channels, avg per active day | After each ETL run |
| `mv_channel_monthly` | Channel × month breakdown for trend-by-channel charts | After each ETL run |
| `mv_hourly_distribution` | Hourly viewing distribution by day-of-week and year (heatmap data) | After each ETL run |
| `mv_channel_funnel` | Channel loyalty funnel: Discovery → Casual → Regular → Loyal classification | After each ETL run |

All materialized views are refreshed via `refresh_views.sql`.

### Indexes (7)

Optimized for the analytical read patterns of a star schema:

| Index | Table | Column(s) |
|-------|-------|-----------|
| `idx_watch_events_watched_at` | `fact_watch_events` | `watched_at` |
| `idx_watch_events_video_id` | `fact_watch_events` | `video_id` |
| `idx_watch_events_channel_id` | `fact_watch_events` | `channel_id` |
| `idx_videos_channel_id` | `dim_videos` | `channel_id` |
| `idx_videos_status` | `dim_videos` | `status` |
| `idx_videos_category_id` | `dim_videos` | `category_id` |
| `idx_video_topics_topic_id` | `nlp_video_topics` | `topic_id` |

Plus unique indexes on each materialized view for `REFRESH CONCURRENTLY` support.

---

## 4. Design Analysis & Gap Identification

A systematic analysis was performed across four domains to identify what the schema can and cannot answer, and what optimizations would close the gaps.

### 4.1 Temporal Analysis

**Resolved gaps:**

- Weekend vs weekday analysis → solved by `dim_date.is_weekend`
- Quarterly and seasonal trends → solved by `dim_date.quarter` and `dim_date.season`
- Danish holiday patterns → solved by `dim_date.is_danish_holiday`
- Watch streaks (consecutive active days) → solved via `dim_date` + gap-and-island SQL pattern
- Rolling averages with zero-activity days included → solved by `mv_daily_summary` (LEFT JOIN from `dim_date`)
- Monthly trend comparisons → solved by `mv_monthly_summary`
- Weekly heatmaps (day × hour) → solved by `mv_hourly_distribution`
- Session duration → solved by `mv_session_summary`

**Key architectural decision:** `dim_date` was chosen over computed columns because it's the standard star schema pattern, pre-computes all attributes once, eliminates repeated `EXTRACT` calls in views, and is trivially extensible for future attributes.

**Remaining temporal gaps (require Tier 2 enrichment):**

- Attention span trends (watching shorter/longer videos over time) → needs `duration_seconds` populated
- Shorts vs long-form content distinction → needs `duration_seconds` for reliable classification

### 4.2 Aggregation Performance

**Resolved gaps:**

All high-frequency dashboard queries are now served by materialized views rather than full fact table scans:

| Dashboard Widget | Data Source |
|------------------|-------------|
| Monthly trend chart | `mv_monthly_summary` |
| Top channels per period | `mv_channel_monthly` |
| Weekly heatmap | `mv_hourly_distribution` |
| Videos per day sparkline | `mv_daily_summary` |
| Session length metrics | `mv_session_summary` |
| Channel funnel breakdown | `mv_channel_funnel` |

**Refresh strategy:** All materialized views are refreshed via `refresh_views.sql` after each quarterly ETL ingest. The `CONCURRENTLY` keyword allows reads during refresh.

### 4.3 Relationship & Entity Gaps

**Resolved gaps:**

- Channel loyalty funnel classification → `mv_channel_funnel` (Discovery / Casual / Regular / Loyal)
- First-time channel/video discovery tracking → `v_discovery_events`

**Deferred to post-NLP (structures designed, not yet implemented):**

| Materialized View | Purpose | Dependency |
|-------------------|---------|------------|
| `mv_video_rewatches` | Re-watch counts and time span per video | None — can build now, deferred by choice |
| `mv_channel_topics` | Channel-to-topic affinity | `nlp_video_topics` populated |
| `mv_channel_cooccurrence` | Channel co-occurrence within sessions | `nlp_video_topics` populated |
| `mv_session_transitions` | Channel/topic transition patterns within sessions | `nlp_video_topics` populated |

### 4.4 Domain-Specific (Stakeholder / Marketing) Gaps

**Resolved gaps:**

- Channel loyalty funnel → `mv_channel_funnel`
- Content discovery timing → `v_discovery_events`

**Deferred (require Tier 2 API enrichment):**

| Question | What's Needed |
|----------|---------------|
| Attention span trends over time | `dim_videos.duration_seconds` |
| Content gap analysis (missing categories) | `dim_videos.category_id` |
| Time-to-discovery (how fast you find new videos) | New `dim_videos.published_at` column + API data |
| Shorts vs long-form breakdown | `dim_videos.duration_seconds` |

**Planned schema tweak for Tier 2 readiness:**

```sql
ALTER TABLE yt.dim_videos ADD COLUMN published_at TIMESTAMPTZ;
```

This zero-cost addition prepares for the "time-to-discovery" metric once API enrichment populates it.

---

## 5. Operational Procedures

### Quarterly Data Ingest

When a new Google Takeout JSON arrives:

```bash
# 1. Run the ETL pipeline
python etl_yt.py watch-history.json

# 2. Refresh all materialized views
# (execute refresh_views.sql in pgAdmin Query Tool)
```

The ETL pipeline handles all deduplication automatically. The ETL log records every run for auditability.

### Rebuilding from Scratch

If the database needs to be recreated:

1. Create database `youtube_analytics` in pgAdmin (or via SQL)
2. Run `create_schema.sql` (everything from STEP 2 onward)
3. Run `etl_yt.py` with each watch history JSON file
4. Run `refresh_views.sql` to populate materialized views

Note: `v_etl_history` depends on `yt.etl_log`, which is auto-created on first ETL run. This view will not resolve until the ETL has executed at least once.

---

## 6. Next Steps

The database foundation and analytical layer are complete for the current data. The following phases are planned, in priority order:

### Phase 1: NLP Pipeline (Tier 1)

Build topic modeling and embedding pipelines using the data already in the database (video titles and channel names):

- TF-IDF + NMF topic modeling (baseline, matching legacy approach)
- BERTopic or transformer-based topic modeling (advanced)
- Sentence-transformer embeddings for similarity search
- Populate `nlp_topics`, `nlp_video_topics`, and `nlp_embeddings`
- Build the deferred materialized views (`mv_video_rewatches`, `mv_channel_topics`, `mv_channel_cooccurrence`, `mv_session_transitions`)
- Update `refresh_views.sql` with the new materialized views

### Phase 2: Tier 2 API Enrichment

Build a YouTube Data API batch script to enrich `dim_videos`:

- Populate `description`, `tags`, `category_id`, `duration_seconds`, `view_count`, `like_count`, and `published_at`
- Update `status` field for videos that are no longer available
- Respect daily API quota (~10,000 units/day; ~4–5 days for full enrichment)
- Unlock attention span trends, content gap analysis, Shorts vs long-form, and time-to-discovery metrics

### Phase 3: Tier 3 Transcript Enrichment (Stretch)

- Use `youtube-transcript-api` to pull auto-generated captions
- Store in a new `dim_transcripts` table or as TEXT on `dim_videos`
- Enable deep topic modeling on full video content, not just titles

### Phase 4: FastAPI Backend

- REST API serving analytics from the database
- WebSocket support for real-time dashboard updates
- Endpoint for quarterly ingest (upload JSON → ETL → refresh views)
- Python-native integration with NLP pipeline

### Phase 5: React Frontend

- Interactive dashboard with temporal trends, channel loyalty, session analysis
- Weekly heatmap visualization (from `mv_hourly_distribution`)
- Channel funnel breakdown (from `mv_channel_funnel`)
- Discovery timeline (from `v_discovery_events`)
- Topic evolution charts (post-NLP)

### Phase 6: Docker Compose & CI/CD

- Container orchestration: PostgreSQL + FastAPI + React
- GitHub Actions for automated testing on push
- Automated quarterly ingest pipeline
- Environment configuration management

---

## 7. Design Decisions Log

A running record of key architectural decisions made during this phase:

| Decision | Rationale |
|----------|-----------|
| Star schema over 3NF | Analytical workload (bulk writes, heavy reads) favors star schema's flat fact table and dimension joins |
| `dim_date` over computed columns | Standard dimensional modeling; pre-computes all attributes once; extensible for holidays, seasons, etc. |
| Materialized views over summary tables | Derived from source data (no manual maintenance); refreshed atomically; `CONCURRENTLY` allows reads during refresh |
| Ghost videos kept with NULL title/channel | Preserves video ID for potential future API recovery; doesn't pollute NLP pipelines (NULL titles excluded naturally) |
| Ads dropped pre-ingest (no ad table) | Simplifies schema; ad analysis was deprioritized by stakeholder decision |
| Session gap threshold: 45 minutes | Calibrated between "I paused the video" (30 min) and "I came back later" (60 min); adjustable in `v_sessions` |
| Channel funnel thresholds | Discovery (1 view), Casual (2–5), Regular (6–20 + 3 active days), Loyal (20+ views, 7+ days, 30+ day span) |
| `etl_log` created by ETL script, not schema | Keeps schema script independent of Python tooling; log table is infrastructure, not analytics |
| `published_at` deferred to Tier 2 | Zero value until API enrichment runs; column will be added via ALTER TABLE when ready |

---

*This document should be updated at the start of each new project phase to reflect the evolving state of the database and application.*
