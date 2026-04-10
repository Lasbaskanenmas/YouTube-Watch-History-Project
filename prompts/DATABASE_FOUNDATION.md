# YouTube Watch History — Database Foundation

## 1. Project Purpose

This project is a personal analytics platform built around YouTube watch history data exported from Google Takeout. Its primary goals are to:

- Demonstrate professional-grade skills in **NLP and deep learning** applied to real-world data.
- Showcase the ability to design, build, and maintain a **local PostgreSQL database** following **CI/CD principles**.
- Deliver a fully functional, real-time **web application** that surfaces deep learning–driven analytics on personal viewing behavior.

This is a portfolio project. The emphasis is on architectural decisions, production-quality engineering, and end-to-end delivery — not just model accuracy.

---

## 2. The Data

### Source

- **Google Takeout** YouTube watch history export.
- Format: dictionary-style JSON file, one object per watch event.
- Each observation has **9 raw attributes**:

| # | Attribute          | Description                                                                 |
|---|--------------------|-----------------------------------------------------------------------------|
| 1 | `header`           | Always "YouTube" — identifies the product.                                  |
| 2 | `title`            | Video title, prefixed with "Watched ".                                      |
| 3 | `titleUrl`         | Full YouTube URL containing the video ID (e.g., `watch?v=abc123`).          |
| 4 | `subtitles`        | Nested dict with `name` (channel name) and `url` (channel URL).            |
| 5 | `time`             | ISO 8601 timestamp in UTC (e.g., `2024-02-18T15:16:21.549Z`).              |
| 6 | `products`         | Always `[YouTube]` — identifies the product.                                |
| 7 | `activityControls` | Describes the activity type (e.g., "YouTube watch history").                |
| 8 | `description`      | Mostly NaN — rarely populated.                                              |
| 9 | `details`          | Nested field; contains `"From Google Ads"` for ad impressions.              |

### Volume

- First export contains **48,407 raw observations**.
- After ad removal, estimated **~35,000–42,000** real watch events.
- Approximately **12,487 unique channels** observed.

### Update Cadence

- A new JSON export is received from Google **every 3 months**.
- Each new file contains the **full history plus the latest 3 months** — meaning heavy duplication with prior exports.
- The ETL pipeline must **deduplicate** on ingest using a composite key of `video_id` (extracted from `titleUrl`) + `time`.

---

## 3. ETL Decisions

### Ad Filtering

- **Ads are dropped entirely** before any data enters the database.
- The sole reliable marker is the `details` column containing `"From Google Ads"`.
- No ad analysis will be performed — this decision simplifies the schema.

### Timezone Handling

- Raw timestamps arrive in **UTC**.
- PostgreSQL stores them as `TIMESTAMPTZ` (UTC internally).
- Queries convert to **Europe/Copenhagen** (`CET` in winter / `CEST` in summer) at the session level using `SET timezone = 'Europe/Copenhagen'`.
- This mirrors the approach in the legacy Python codebase (`library.py` → `convert_to_CEST`), which correctly used `Europe/Copenhagen` for Danish local time.

### Video ID Extraction

- The `titleUrl` field always contains a value in the export, even if the underlying video has since been removed from YouTube.
- The **video ID** is extracted from the URL (e.g., `hMEyBtsuAJE` from `https://www.youtube.com/watch?v=hMEyBtsuAJE`).
- This ID serves as the primary lookup key for YouTube Data API enrichment.
- A `status` field on the video record tracks whether the video is `active`, `unavailable`, or `private` — allowing the enrichment pipeline to skip dead links gracefully.

### Deduplication Strategy

- On each quarterly ingest, records are deduplicated using the composite natural key: **`video_id` + `watched_at` timestamp**.
- This prevents the same watch event from being inserted twice when overlapping exports are loaded.

---

## 4. Analytics Goals

The database and application are designed to answer five categories of questions:

### 4.1 Temporal Patterns
> "When do I watch the most? Has my behavior shifted over time?"
- Year-over-year trends, monthly/weekly/hourly distributions.
- Requires: `fact_watch_events.watched_at` with timezone-aware queries.

### 4.2 Content & Topic Patterns
> "What topics do I gravitate toward? Are there clusters of interest?"
- NLP topic modeling (NMF, LDA, or transformer-based) on video titles.
- Later enriched with descriptions, tags, and potentially transcripts.
- Requires: `dim_videos.title`, NLP output tables.

### 4.3 Channel Loyalty
> "Which creators do I return to most?"
- Repeat-visit analysis, channel ranking over time.
- Requires: `dim_channels` joined to `fact_watch_events`.

### 4.4 Session & Binge Behavior
> "Do I binge-watch? What does a typical session look like?"
- A **session** is defined as a sequence of watch events where the gap between consecutive videos is less than a configurable threshold (default: 30–60 minutes, to be calibrated).
- Implemented via SQL window functions (`LAG` over `watched_at`, partitioned by date).
- Requires: `fact_watch_events.watched_at` with dense timestamp data.

### 4.5 Interest Evolution & Prediction
> "How have my interests evolved year over year? What will I watch next?"
- Time-windowed topic distributions to show shifting interests.
- Recommendation/prediction via deep learning embeddings.
- Requires: NLP topic and embedding tables linked to videos and timestamps.

---

## 5. Text Data Tiers (NLP Pipeline)

The NLP strategy is designed in three tiers, allowing incremental enrichment:

| Tier | Source                    | Status       | Data                                          |
|------|---------------------------|--------------|-----------------------------------------------|
| 1    | Google Takeout export     | **Available** | Video titles, channel names                   |
| 2    | YouTube Data API          | **Planned**   | Descriptions, tags, category metadata         |
| 3    | `youtube-transcript-api`  | **Stretch**   | Auto-generated captions / transcripts         |

- **Tier 1** is the foundation — all NLP starts here.
- **Tier 2** requires a Google Cloud project with a YouTube Data API key. The free quota is ~10,000 units/day; enriching ~40K videos at ~3 units each takes roughly **4–5 days** of automated batch calls.
- **Tier 3** is feasible for most videos with auto-generated captions. The `youtube-transcript-api` Python library handles this without needing an API key.

---

## 6. Schema Design — Star Schema

### Why Star Schema over 3NF?

This system is **analytical (OLAP)**, not transactional (OLTP):

- Data is loaded in **bulk quarterly writes**, not frequent small transactions.
- The primary workload is **reads** — aggregations, joins, time series, NLP queries.
- 3NF optimizes for write consistency and update-anomaly prevention, which adds unnecessary JOIN complexity for our read-heavy pattern.
- A **star schema** keeps the fact table flat and scannable, normalizes only where redundancy is genuinely wasteful (channels, videos, categories), and extends cleanly as new dimensions are added.

### Conceptual Schema

```
                    ┌──────────────────┐
                    │  dim_categories  │
                    │  (from YT API)   │
                    └────────┬─────────┘
                             │
┌──────────────┐    ┌────────┴─────────┐    ┌──────────────────┐
│ dim_channels │────│ fact_watch_events │────│   dim_videos     │
│              │    │                  │    │ (title, metadata)│
└──────────────┘    └────────┬─────────┘    └──────┬───────────┘
                             │                     │
                    ┌────────┴─────────┐    ┌──────┴───────────┐
                    │  nlp_topics      │    │  nlp_embeddings  │
                    │  (topic model)   │    │  (deep learning) │
                    └──────────────────┘    └──────────────────┘
```

### Planned Tables

| Table                | Role             | Description                                                        |
|----------------------|------------------|--------------------------------------------------------------------|
| `fact_watch_events`  | Fact table       | One row per watch event. Foreign keys to dimensions. Core of all queries. |
| `dim_videos`         | Dimension        | One row per unique video ID. Holds title, URL, API-enriched metadata, status. |
| `dim_channels`       | Dimension        | One row per unique channel. Holds channel name and URL.            |
| `dim_categories`     | Dimension        | YouTube category taxonomy (from API). E.g., "Music", "Sports", "Education". |
| `nlp_topics`         | NLP output       | Topic model results linked to videos. Stores topic labels, keywords, scores. |
| `nlp_embeddings`     | NLP output       | Dense vector embeddings per video for similarity search and prediction. |

---

## 7. Technology Stack

| Layer      | Technology         | Rationale                                                              |
|------------|--------------------|------------------------------------------------------------------------|
| Database   | PostgreSQL         | Robust, supports `TIMESTAMPTZ`, full-text search, and `pgvector` for embeddings. |
| Backend    | FastAPI (Python)   | Native WebSocket support for real-time analytics. Python-native for seamless NLP integration. |
| Frontend   | React              | Component-driven, ideal for interactive dashboards.                    |
| NLP        | scikit-learn, transformers | TF-IDF, NMF, and transformer-based models for topic modeling and embeddings. |
| Containers | Docker Compose     | Orchestrates PostgreSQL + FastAPI + React locally.                     |
| CI/CD      | GitHub Actions     | Automated testing on push. Quarterly ingest via CLI or API endpoint.   |

---

## 8. Legacy Codebase Notes

The project inherits two Python files from an earlier version:

- **`library.py`** — Contains `YouTubeReader` (JSON loading, DataFrame conversion, ad removal, timezone conversion), `YouTubeWrangler` (time series, plotting, heatmaps), and `YouTubeTextStats` (TF-IDF, NMF topic modeling, word clouds).
- **`functions.py`** — Helper functions for subtitle splitting, weekly aggregation, and hourly grouping.

Key takeaways carried forward:

- The `Europe/Copenhagen` timezone conversion is correct for Danish time.
- Ad filtering via `details_name == 'From Google Ads'` is the established pattern.
- TF-IDF + NMF topic modeling on cleaned titles is the baseline NLP approach; the new system will extend this with deep learning methods.
- The subtitle-splitting logic confirms the nested dict structure `{'name': ..., 'url': ...}`.

---

*This document serves as the single source of truth for all architectural decisions made during the top-down design phase. All schema DDL, ETL pipelines, and application code should trace back to the decisions recorded here.*
