-- ============================================================================
-- YouTube Analytics — Database Schema
-- ============================================================================
-- Run this script in pgAdmin 4 in TWO steps:
--
-- STEP 1: Connect to the default 'postgres' database and run ONLY the
--         CREATE DATABASE statement below.
--
-- STEP 2: Disconnect, then connect to 'youtube_analytics' and run
--         EVERYTHING from "STEP 2 START" onward.
-- ============================================================================


-- ====================
-- STEP 1: Run this alone while connected to 'postgres'
-- ====================
--CREATE DATABASE youtube_analytics
--    ENCODING = 'UTF8'
--    LC_COLLATE = 'en_US.UTF-8'
--    LC_CTYPE = 'en_US.UTF-8';


-- ====================
-- STEP 2 START: Connect to 'youtube_analytics' and run everything below
-- ====================


-- ============================================================================
-- Schema
-- ============================================================================
-- All project tables live under the 'yt' schema to keep things isolated
-- from the default 'public' schema.
CREATE SCHEMA IF NOT EXISTS yt;


-- ============================================================================
-- Dimension Tables
-- ============================================================================

-- dim_channels: One row per unique YouTube channel (~12,487 expected)
CREATE TABLE yt.dim_channels (
    channel_id      SERIAL          PRIMARY KEY,
    channel_name    VARCHAR(255),
    channel_url     TEXT,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),

    -- Prevent duplicate channels
    UNIQUE(channel_name, channel_url)
);

COMMENT ON TABLE yt.dim_channels IS 'Dimension table for YouTube channels. Sourced from the subtitles field in Google Takeout export.';


-- dim_categories: YouTube category taxonomy (populated via YouTube Data API)
CREATE TABLE yt.dim_categories (
    category_id     INTEGER         PRIMARY KEY,    -- YouTube own category IDs (1-44)
    category_name   VARCHAR(100)    NOT NULL
);

COMMENT ON TABLE yt.dim_categories IS 'YouTube video category taxonomy. Populated from YouTube Data API enrichment (Tier 2).';

-- Pre-populate with YouTube standard categories
INSERT INTO yt.dim_categories (category_id, category_name) VALUES
    (1,  'Film & Animation'),
    (2,  'Autos & Vehicles'),
    (10, 'Music'),
    (15, 'Pets & Animals'),
    (17, 'Sports'),
    (18, 'Short Movies'),
    (19, 'Travel & Events'),
    (20, 'Gaming'),
    (21, 'Videoblogging'),
    (22, 'People & Blogs'),
    (23, 'Comedy'),
    (24, 'Entertainment'),
    (25, 'News & Politics'),
    (26, 'Howto & Style'),
    (27, 'Education'),
    (28, 'Science & Technology'),
    (29, 'Nonprofits & Activism'),
    (30, 'Movies'),
    (31, 'Anime/Animation'),
    (32, 'Action/Adventure'),
    (33, 'Classics'),
    (34, 'Comedy'),
    (35, 'Documentary'),
    (36, 'Drama'),
    (37, 'Family'),
    (38, 'Foreign'),
    (39, 'Horror'),
    (40, 'Sci-Fi/Fantasy'),
    (41, 'Thriller'),
    (42, 'Shorts'),
    (43, 'Shows'),
    (44, 'Trailers');

-- dim_date: Pre-computed date dimension for fast analytical queries
CREATE TABLE yt.dim_date (
    date_key        DATE            PRIMARY KEY,
    year            SMALLINT        NOT NULL,
    quarter         SMALLINT        NOT NULL,
    month           SMALLINT        NOT NULL,
    month_name      VARCHAR(10)     NOT NULL,
    week_of_year    SMALLINT        NOT NULL,
    day_of_month    SMALLINT        NOT NULL,
    day_of_week_num SMALLINT        NOT NULL,   -- 0=Sun, 1=Mon, ..., 6=Sat
    day_of_week     VARCHAR(10)     NOT NULL,
    is_weekend      BOOLEAN         NOT NULL,
    season          VARCHAR(10)     NOT NULL,    -- Based on Northern Hemisphere / Danish seasons
    is_danish_holiday BOOLEAN       DEFAULT FALSE
);

COMMENT ON TABLE yt.dim_date IS 'Date dimension table. Pre-computed date attributes from 2020 through 2030 for fast analytical joins without repeated EXTRACT calls.';

-- Populate dim_date: 2020-01-01 through 2030-12-31
INSERT INTO yt.dim_date (
    date_key, year, quarter, month, month_name, week_of_year,
    day_of_month, day_of_week_num, day_of_week, is_weekend, season, is_danish_holiday
)
SELECT
    d::DATE                                                          AS date_key,
    EXTRACT(YEAR FROM d)::SMALLINT                                   AS year,
    EXTRACT(QUARTER FROM d)::SMALLINT                                AS quarter,
    EXTRACT(MONTH FROM d)::SMALLINT                                  AS month,
    TRIM(TO_CHAR(d, 'Month'))                                        AS month_name,
    EXTRACT(WEEK FROM d)::SMALLINT                                   AS week_of_year,
    EXTRACT(DAY FROM d)::SMALLINT                                    AS day_of_month,
    EXTRACT(DOW FROM d)::SMALLINT                                    AS day_of_week_num,
    TRIM(TO_CHAR(d, 'Day'))                                          AS day_of_week,
    EXTRACT(DOW FROM d) IN (0, 6)                                    AS is_weekend,
    CASE
        WHEN EXTRACT(MONTH FROM d) IN (12, 1, 2)  THEN 'Winter'
        WHEN EXTRACT(MONTH FROM d) IN (3, 4, 5)   THEN 'Spring'
        WHEN EXTRACT(MONTH FROM d) IN (6, 7, 8)   THEN 'Summer'
        WHEN EXTRACT(MONTH FROM d) IN (9, 10, 11)  THEN 'Autumn'
    END                                                              AS season,
    -- Danish public holidays (fixed-date only; Easter-dependent ones need manual updates)
    (EXTRACT(MONTH FROM d), EXTRACT(DAY FROM d)) IN (
        (1, 1),     -- Nytårsdag
        (6, 5),     -- Grundlovsdag
        (12, 24),   -- Juleaften
        (12, 25),   -- 1. Juledag
        (12, 26),   -- 2. Juledag
        (12, 31)    -- Nytårsaften
    )                                                                AS is_danish_holiday
FROM generate_series('2020-01-01'::DATE, '2030-12-31'::DATE, '1 day'::INTERVAL) AS d;

-- dim_videos: One row per unique video ID
CREATE TABLE yt.dim_videos (
    video_id        VARCHAR(20)     PRIMARY KEY,    -- Extracted from titleUrl (e.g., 'hMEyBtsuAJE')
    title           TEXT            NULL,
    title_url       TEXT,
    channel_id      INTEGER         REFERENCES yt.dim_channels(channel_id),
    status          VARCHAR(20)     DEFAULT 'active'
                                    CHECK (status IN ('active', 'unavailable', 'private')),

    -- Tier 2: YouTube Data API enrichment fields
    description     TEXT,
    tags            TEXT[],
    category_id     INTEGER         REFERENCES yt.dim_categories(category_id),
    duration_seconds INTEGER,
    view_count      BIGINT,
    like_count      BIGINT,

    -- Tier 3: Transcript enrichment
    has_transcript  BOOLEAN         DEFAULT FALSE,

    -- Metadata
    enriched_at     TIMESTAMPTZ,                    -- NULL until API enrichment runs
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

COMMENT ON TABLE yt.dim_videos IS 'Dimension table for unique videos. Core fields from Takeout export; enrichment fields populated later via YouTube Data API (Tier 2).';


-- ============================================================================
-- Fact Table
-- ============================================================================

-- fact_watch_events: One row per watch event (the core of all queries)
CREATE TABLE yt.fact_watch_events (
    event_id        SERIAL          PRIMARY KEY,
    video_id        VARCHAR(20)     NOT NULL REFERENCES yt.dim_videos(video_id),
    channel_id      INTEGER         REFERENCES yt.dim_channels(channel_id),   -- Denormalized for fast star-schema joins
    watched_at      TIMESTAMPTZ     NOT NULL,

    -- Deduplication: same video + same timestamp = same event
    UNIQUE(video_id, watched_at)
);

COMMENT ON TABLE yt.fact_watch_events IS 'Fact table: one row per YouTube watch event. Deduplicated on (video_id, watched_at) to handle overlapping quarterly exports.';


-- ============================================================================
-- NLP Output Tables
-- ============================================================================

-- nlp_topics: Topic model output (NMF, LDA, or transformer-based)
CREATE TABLE yt.nlp_topics (
    topic_id        SERIAL          PRIMARY KEY,
    topic_label     VARCHAR(100),                   -- Human-readable label (e.g., 'Boxing & MMA')
    keywords        TEXT[]          NOT NULL,        -- Top words for this topic
    model_name      VARCHAR(100)    NOT NULL,        -- e.g., 'nmf_tfidf', 'bertopic'
    model_version   VARCHAR(50),                     -- e.g., 'v1.0', '2024-03-01'
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

COMMENT ON TABLE yt.nlp_topics IS 'Topics identified by NLP models. Each row is one topic with its keywords and source model metadata.';


-- nlp_video_topics: Many-to-many link between videos and topics (with score)
CREATE TABLE yt.nlp_video_topics (
    video_id        VARCHAR(20)     REFERENCES yt.dim_videos(video_id) ON DELETE CASCADE,
    topic_id        INTEGER         REFERENCES yt.nlp_topics(topic_id) ON DELETE CASCADE,
    score           FLOAT           NOT NULL CHECK (score >= 0 AND score <= 1),

    PRIMARY KEY(video_id, topic_id)
);

COMMENT ON TABLE yt.nlp_video_topics IS 'Junction table mapping videos to topics with relevance scores. Supports multi-topic assignment per video.';


-- nlp_embeddings: Dense vector embeddings per video
CREATE TABLE yt.nlp_embeddings (
    video_id        VARCHAR(20)     PRIMARY KEY REFERENCES yt.dim_videos(video_id) ON DELETE CASCADE,
    embedding       FLOAT[]         NOT NULL,       -- Dense vector; migrate to pgvector when ready
    model_name      VARCHAR(100)    NOT NULL,        -- e.g., 'sentence-transformers/all-MiniLM-L6-v2'
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

COMMENT ON TABLE yt.nlp_embeddings IS 'Dense vector embeddings per video for similarity search and recommendation. Stored as FLOAT[] — migrate to pgvector type for production similarity queries.';


-- ============================================================================
-- Indexes (optimized for analytical query patterns)
-- ============================================================================

-- Fact table: time-based queries are the most common
CREATE INDEX idx_watch_events_watched_at    ON yt.fact_watch_events (watched_at);
CREATE INDEX idx_watch_events_video_id      ON yt.fact_watch_events (video_id);
CREATE INDEX idx_watch_events_channel_id    ON yt.fact_watch_events (channel_id);

-- Videos: lookup by channel and enrichment status
CREATE INDEX idx_videos_channel_id          ON yt.dim_videos (channel_id);
CREATE INDEX idx_videos_status              ON yt.dim_videos (status);
CREATE INDEX idx_videos_category_id         ON yt.dim_videos (category_id);

-- NLP: topic lookups
CREATE INDEX idx_video_topics_topic_id      ON yt.nlp_video_topics (topic_id);


-- ============================================================================
-- Views
-- ============================================================================
-- View: Watch events with full denormalized context (Danish time)
CREATE OR REPLACE VIEW yt.v_watch_history AS
SELECT
    fe.event_id,
    fe.watched_at AT TIME ZONE 'Europe/Copenhagen'  AS watched_at_dk,
    dv.video_id,
    dv.title,
    dv.title_url,
    dv.status,
    dc.channel_name,
    dc.channel_url,
    dcat.category_name,
    -- Time dimensions from dim_date (pre-computed, no repeated EXTRACT calls)
    dd.year,
    dd.quarter,
    dd.month,
    dd.month_name,
    dd.week_of_year,
    dd.day_of_week,
    dd.day_of_week_num,
    dd.is_weekend,
    dd.season,
    dd.is_danish_holiday,
    -- Hour still needs EXTRACT (dim_date is date-level, not hour-level)
    EXTRACT(HOUR FROM fe.watched_at AT TIME ZONE 'Europe/Copenhagen') AS hour
FROM yt.fact_watch_events fe
JOIN yt.dim_videos dv           ON fe.video_id = dv.video_id
LEFT JOIN yt.dim_channels dc    ON fe.channel_id = dc.channel_id
LEFT JOIN yt.dim_categories dcat ON dv.category_id = dcat.category_id
LEFT JOIN yt.dim_date dd        ON dd.date_key = DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen');

COMMENT ON VIEW yt.v_watch_history IS 'Denormalized view of all watch events with Danish timezone conversion. Date attributes from dim_date; hour extracted at query time.';

-- View: Session detection using gap analysis
-- A new session starts when the gap between consecutive videos exceeds 45 minutes
CREATE OR REPLACE VIEW yt.v_sessions AS
WITH gaps AS (
    SELECT
        event_id,
        video_id,
        channel_id,
        watched_at,
        watched_at AT TIME ZONE 'Europe/Copenhagen' AS watched_at_dk,
        LAG(watched_at) OVER (ORDER BY watched_at)  AS prev_watched_at,
        EXTRACT(EPOCH FROM (
            watched_at - LAG(watched_at) OVER (ORDER BY watched_at)
        )) / 60.0 AS gap_minutes
    FROM yt.fact_watch_events
),
session_flags AS (
    SELECT
        *,
        CASE
            WHEN gap_minutes IS NULL OR gap_minutes > 45 THEN 1
            ELSE 0
        END AS new_session_flag
    FROM gaps
),
session_ids AS (
    SELECT
        *,
        SUM(new_session_flag) OVER (ORDER BY watched_at) AS session_id
    FROM session_flags
)
SELECT
    session_id,
    event_id,
    video_id,
    channel_id,
    watched_at,
    watched_at_dk,
    gap_minutes,
    COUNT(*) OVER (PARTITION BY session_id)  AS session_video_count,
    MIN(watched_at) OVER (PARTITION BY session_id) AS session_start,
    MAX(watched_at) OVER (PARTITION BY session_id) AS session_end
FROM session_ids;

COMMENT ON VIEW yt.v_sessions IS 'Session detection view. Groups consecutive watch events into sessions using a 45-minute gap threshold. Use session_id to aggregate binge behavior.';


-- View: Channel loyalty ranking
CREATE OR REPLACE VIEW yt.v_channel_loyalty AS
SELECT
    dc.channel_name,
    dc.channel_url,
    COUNT(*)                                                        AS total_views,
    COUNT(DISTINCT dv.video_id)                                     AS unique_videos,
    COUNT(DISTINCT DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')) AS active_days,
    MIN(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')             AS first_watched_dk,
    MAX(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')             AS last_watched_dk
FROM yt.fact_watch_events fe
JOIN yt.dim_channels dc     ON fe.channel_id = dc.channel_id
JOIN yt.dim_videos dv       ON fe.video_id = dv.video_id
GROUP BY dc.channel_id, dc.channel_name, dc.channel_url;

COMMENT ON VIEW yt.v_channel_loyalty IS 'Channel loyalty metrics: total views, unique videos, active days, and watch span per channel.';

-- View: ETL ingest history
CREATE OR REPLACE VIEW yt.v_etl_history AS
SELECT
    file_name,
    run_at AT TIME ZONE 'Europe/Copenhagen' AS run_at_dk,
    total_raw_records,
    ads_filtered,
    removed_dropped,
    new_events,
    duplicate_events,
    duration_seconds
FROM yt.etl_log
ORDER BY run_at DESC;

COMMENT ON VIEW yt.v_etl_history IS 'ETL audit trail. Displays ingest run history with Danish timestamps and filtering metrics.';

-- View: Flag first-time discoveries at the channel and video level
CREATE OR REPLACE VIEW yt.v_discovery_events AS
WITH ranked AS (
    SELECT
        fe.event_id,
        fe.video_id,
        fe.channel_id,
        fe.watched_at,
        dc.channel_name,
        ROW_NUMBER() OVER (
            PARTITION BY fe.channel_id ORDER BY fe.watched_at
        ) AS channel_watch_rank,
        ROW_NUMBER() OVER (
            PARTITION BY fe.video_id ORDER BY fe.watched_at
        ) AS video_watch_rank
    FROM yt.fact_watch_events fe
    LEFT JOIN yt.dim_channels dc ON fe.channel_id = dc.channel_id
)
SELECT
    *,
    channel_watch_rank = 1  AS is_channel_discovery,
    video_watch_rank = 1    AS is_first_watch
FROM ranked;

COMMENT ON VIEW yt.v_discovery_events IS 
    'Flags first-time channel discoveries and first video watches. Use is_channel_discovery to analyze when new channels enter the viewing repertoire.';

-- Materialized View: daily summary
CREATE MATERIALIZED VIEW yt.mv_daily_summary AS
SELECT
    dd.date_key,
    dd.year,
    dd.quarter,
    dd.month,
    dd.month_name,
    dd.day_of_week,
    dd.day_of_week_num,
    dd.is_weekend,
    dd.season,
    dd.is_danish_holiday,
    COALESCE(COUNT(fe.event_id), 0)             AS videos_watched,
    COALESCE(COUNT(DISTINCT fe.video_id), 0)    AS unique_videos,
    COALESCE(COUNT(DISTINCT fe.channel_id), 0)  AS unique_channels,
    MIN(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')  AS first_watch,
    MAX(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')  AS last_watch
FROM yt.dim_date dd
LEFT JOIN yt.fact_watch_events fe
    ON dd.date_key = DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')
WHERE dd.date_key <= CURRENT_DATE
GROUP BY dd.date_key, dd.year, dd.quarter, dd.month, dd.month_name,
         dd.day_of_week, dd.day_of_week_num, dd.is_weekend, dd.season,
         dd.is_danish_holiday
ORDER BY dd.date_key;

CREATE UNIQUE INDEX idx_mv_daily_summary_date ON yt.mv_daily_summary (date_key);

COMMENT ON MATERIALIZED VIEW yt.mv_daily_summary IS 'Pre-computed daily activity summary. Includes zero-activity days from dim_date. Refresh after each ETL run.';

-- Materialized View: monthly summary
CREATE MATERIALIZED VIEW yt.mv_monthly_summary AS
SELECT
    dd.year,
    dd.month,
    dd.month_name,
    dd.season,
    COUNT(fe.event_id)                  AS total_views,
    COUNT(DISTINCT fe.video_id)         AS unique_videos,
    COUNT(DISTINCT fe.channel_id)       AS unique_channels,
    COUNT(DISTINCT dd.date_key) 
        FILTER (WHERE fe.event_id IS NOT NULL)  AS active_days,
    ROUND(
        COUNT(fe.event_id)::NUMERIC / 
        NULLIF(COUNT(DISTINCT dd.date_key) 
            FILTER (WHERE fe.event_id IS NOT NULL), 0
        ), 1
    )                                   AS avg_videos_per_active_day
FROM yt.dim_date dd
LEFT JOIN yt.fact_watch_events fe
    ON dd.date_key = DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')
WHERE dd.date_key <= CURRENT_DATE
  AND dd.date_key >= '2020-01-01'
GROUP BY dd.year, dd.month, dd.month_name, dd.season
ORDER BY dd.year, dd.month;

CREATE UNIQUE INDEX idx_mv_monthly_summary 
    ON yt.mv_monthly_summary (year, month);

COMMENT ON MATERIALIZED VIEW yt.mv_monthly_summary IS 
    'Pre-computed monthly activity metrics. Core data source for trend dashboards.';

-- Materialized View: session summary
CREATE MATERIALIZED VIEW yt.mv_session_summary AS
SELECT
    session_id,
    COUNT(*)                                    AS video_count,
    MIN(watched_at_dk)                          AS session_start,
    MAX(watched_at_dk)                          AS session_end,
    EXTRACT(EPOCH FROM (MAX(watched_at) - MIN(watched_at))) / 60.0  AS duration_minutes
FROM yt.v_sessions
GROUP BY session_id;

CREATE UNIQUE INDEX idx_mv_session_summary_id ON yt.mv_session_summary (session_id);

COMMENT ON MATERIALIZED VIEW yt.mv_session_summary IS 'Pre-computed session-level metrics. One row per session with video count and duration.';

-- Materialized View: channel monthly activity
CREATE MATERIALIZED VIEW yt.mv_channel_monthly AS
SELECT
    dc.channel_id,
    dc.channel_name,
    dd.year,
    dd.month,
    dd.month_name,
    COUNT(*)                        AS views,
    COUNT(DISTINCT fe.video_id)     AS unique_videos
FROM yt.fact_watch_events fe
JOIN yt.dim_channels dc     ON fe.channel_id = dc.channel_id
JOIN yt.dim_date dd         ON dd.date_key = DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')
GROUP BY dc.channel_id, dc.channel_name, dd.year, dd.month, dd.month_name
ORDER BY dd.year, dd.month, views DESC;

CREATE UNIQUE INDEX idx_mv_channel_monthly 
    ON yt.mv_channel_monthly (channel_id, year, month);

COMMENT ON MATERIALIZED VIEW yt.mv_channel_monthly IS 
    'Channel activity per month. Powers channel trend charts and top-channels-by-period queries.';

-- Materialized View: hourly distribution by day of week
CREATE MATERIALIZED VIEW yt.mv_hourly_distribution AS
SELECT
    dd.year,
    dd.day_of_week_num,
    dd.day_of_week,
    dd.is_weekend,
    EXTRACT(HOUR FROM fe.watched_at AT TIME ZONE 'Europe/Copenhagen')::SMALLINT AS hour,
    COUNT(*) AS views
FROM yt.fact_watch_events fe
JOIN yt.dim_date dd ON dd.date_key = DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')
GROUP BY dd.year, dd.day_of_week_num, dd.day_of_week, dd.is_weekend,
         EXTRACT(HOUR FROM fe.watched_at AT TIME ZONE 'Europe/Copenhagen');

CREATE UNIQUE INDEX idx_mv_hourly_dist 
    ON yt.mv_hourly_distribution (year, day_of_week_num, hour);

COMMENT ON MATERIALIZED VIEW yt.mv_hourly_distribution IS 
    'Pre-computed hourly viewing distribution by day-of-week and year. Powers heatmap visualizations.';

-- Materialized View: classify every channel into a loyalty stage  
CREATE MATERIALIZED VIEW yt.mv_channel_funnel AS
WITH channel_stats AS (
    SELECT
        dc.channel_id,
        dc.channel_name,
        COUNT(*)                    AS total_views,
        COUNT(DISTINCT fe.video_id) AS unique_videos,
        COUNT(DISTINCT DATE(fe.watched_at AT TIME ZONE 'Europe/Copenhagen')) AS active_days,
        MIN(fe.watched_at)          AS first_watch,
        MAX(fe.watched_at)          AS last_watch,
        EXTRACT(EPOCH FROM (MAX(fe.watched_at) - MIN(fe.watched_at))) / 86400.0 AS span_days
    FROM yt.fact_watch_events fe
    JOIN yt.dim_channels dc ON fe.channel_id = dc.channel_id
    GROUP BY dc.channel_id, dc.channel_name
)
SELECT
    *,
    CASE
        WHEN total_views = 1                          THEN 'Discovery'
        WHEN total_views BETWEEN 2 AND 5              THEN 'Casual'
        WHEN total_views BETWEEN 6 AND 20 
             AND active_days >= 3                      THEN 'Regular'
        WHEN total_views > 20 
             AND active_days >= 7 
             AND span_days >= 30                       THEN 'Loyal'
        ELSE 'Casual'
    END AS funnel_stage
FROM channel_stats;

CREATE UNIQUE INDEX idx_mv_channel_funnel 
    ON yt.mv_channel_funnel (channel_id);

COMMENT ON MATERIALIZED VIEW yt.mv_channel_funnel IS 
    'Channel loyalty funnel: classifies each channel as Discovery, Casual, Regular, or Loyal based on viewing behavior.';

-- ============================================================================
-- Done
-- ============================================================================
-- Next step: Run the Python synthetic data script to populate these tables
-- with realistic test data before loading real Google Takeout exports.
