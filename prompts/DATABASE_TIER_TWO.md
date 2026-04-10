# YouTube Watch History — Tier 2 Enrichment

*This document is Part 4 of the database documentation, continuing from `DATABASE_TIER_ONE.md`. It covers the YouTube Data API enrichment pipeline — what it does, how it works, and the state of the database after it has run.*

---

## 1. What Changed

Before Tier 2 enrichment, `dim_videos` held only what Google Takeout provided: a video ID, a title, a channel reference, and a watch URL. The following fields existed on the table but were entirely NULL:

`description`, `tags`, `category_id`, `duration_seconds`, `view_count`, `like_count`, `enriched_at`

After running `enrich_videos.py`, these fields are now populated for all recoverable active videos, making the table ready for the NLP pipeline.

---

## 2. `enrich_videos.py` — The Enrichment Script

### What It Does

Connects to the YouTube Data API v3, fetches metadata for every unenriched active video in batches of 50, and writes the results back to `dim_videos`. Videos that the API no longer returns are flipped to `status = 'unavailable'`. Every run is logged to `yt.enrichment_log`.

### Prerequisites

```bash
pip install psycopg2-binary python-dotenv requests
```

The script reads the API key from a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_key_here
```

The `.env` file must be listed in `.gitignore` and never committed to version control.

### How to Run

```bash
python enrich_videos.py
```

No arguments required. The script is fully self-contained and resumable — re-running it safely skips already-enriched videos.

### How to Get a YouTube Data API Key

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project
3. Navigate to **APIs & Services → Library**, search for `YouTube Data API v3`, click **Enable**
4. Navigate to **APIs & Services → Credentials → Create Credentials → API key**
5. Under **API restrictions**, restrict the key to `YouTube Data API v3` only
6. Copy the key into your `.env` file

The API is free. The default quota is **10,000 units per day**, which resets at midnight Pacific Time.

---

## 3. Pipeline Design

### Quota Management

Each call to `videos.list` costs **1 quota unit** regardless of how many video IDs are included (up to 50). The script is hard-capped at 10,000 units per day. With batches of 50, this allows up to 500,000 videos per day — far more than the dataset requires.

For the current dataset of ~57,000 active videos, enrichment completes in a single run using only ~1,145 units.

### Video Selection

The script targets only videos satisfying both conditions:

```sql
status = 'active' AND enriched_at IS NULL
```

Ghost videos (`status = 'unavailable'`, no title, no channel) are excluded — the API returns nothing for them anyway.

### Batch Processing

Video IDs are split into batches of 50 and sent to the `videos.list` endpoint with:

```
part=snippet,contentDetails,statistics
```

This single `part` combination returns all six enrichment fields in one call per batch.

### Duration Parsing

The API returns video duration in ISO 8601 format (e.g. `PT1H2M3S`). The script converts this to integer seconds using a regex parser, storing the result in `duration_seconds`. A value of `PT0S` is stored as `0`, not NULL.

### Handling Hidden Like Counts

YouTube allows creators to hide their like count. When `likeCount` is absent from the API response, the script stores `NULL` in `like_count` rather than `0`, preserving the distinction between "zero likes" and "likes hidden".

### Unavailable Video Detection

If a video ID is sent to the API but does not appear in the response, the video has been deleted or privated since it was watched. The script flips these to `status = 'unavailable'` immediately, preventing future runs from wasting quota on them.

### Resumability

The script commits to the database after every batch. If it is interrupted mid-run, all completed batches are preserved. Re-running the script picks up exactly where it left off.

---

## 4. New Table: `yt.enrichment_log`

Auto-created on first run. One row per enrichment run.

| Column | Type | Description |
|---|---|---|
| `log_id` | SERIAL PK | Auto-incremented run ID |
| `run_at` | TIMESTAMPTZ | When the run started (UTC) |
| `videos_targeted` | INTEGER | Active videos with `enriched_at IS NULL` at run start |
| `videos_enriched` | INTEGER | Videos successfully enriched by the API |
| `videos_flipped` | INTEGER | Videos flipped to `unavailable` (API returned nothing) |
| `api_units_used` | INTEGER | Total quota units consumed |
| `duration_seconds` | FLOAT | Wall-clock time for the run |

---

## 5. Database State After Tier 2 Enrichment

First and only enrichment run — April 5, 2026:

| Metric | Value |
|---|---|
| Videos targeted | 57,237 |
| Videos enriched | 54,840 |
| Videos flipped to unavailable | 2,397 |
| API units used | 1,145 / 10,000 |
| Run duration | 288.5s |

Current state of `dim_videos`:

| Status | Count | `enriched_at` |
|---|---|---|
| `active` | 54,840 | Populated |
| `unavailable` | 5,175 | NULL |

The 5,175 unavailable videos include the 2,397 flipped during enrichment plus ~2,500 ghost videos that were already unavailable before enrichment ran.

### Enriched Field Coverage (active videos only)

| Field | Populated | Coverage |
|---|---|---|
| `description` | 49,662 | 90.6% |
| `tags` | 41,200 | 75.1% |
| `category_id` | 54,840 | 100% |
| `duration_seconds` | 54,840 | 100% |
| `view_count` | 54,812 | 99.9% |
| `like_count` | 54,202 | 98.8% |

All `category_id` values are valid foreign keys into `dim_categories`. No negative view or like counts were found. Duration parsing produced zero NULLs.

---

## 6. Re-Running Enrichment

The script is designed for the quarterly ingest pattern. After loading a new Google Takeout export via `etl_yt.py`, newly inserted active videos will have `enriched_at IS NULL`. Running `enrich_videos.py` again will enrich only those new videos — already-enriched videos are untouched.

---

## 7. Next Steps

With `dim_videos` fully enriched, the database is ready for the **NLP Pipeline (Tier 3)**:

1. **Topic Modeling** — Run NMF, LDA, or BERTopic on video titles and descriptions to populate `yt.nlp_topics` and `yt.nlp_video_topics`.
2. **Embedding Pipeline** — Generate sentence-transformer embeddings per video for similarity search and recommendation, populating `yt.nlp_embeddings`.

The enriched fields that will be most valuable to the NLP pipeline are `description` (rich long-form text), `tags` (creator-defined keywords), and `category_id` (a strong prior for topic assignment).

---

*This document should be updated after each new enrichment run if coverage numbers change significantly.*
