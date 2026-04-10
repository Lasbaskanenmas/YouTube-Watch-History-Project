--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: yt; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA yt;


ALTER SCHEMA yt OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: dim_categories; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.dim_categories (
    category_id integer NOT NULL,
    category_name character varying(100) NOT NULL
);


ALTER TABLE yt.dim_categories OWNER TO postgres;

--
-- Name: TABLE dim_categories; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.dim_categories IS 'YouTube video category taxonomy. Populated from YouTube Data API enrichment (Tier 2).';


--
-- Name: dim_channels; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.dim_channels (
    channel_id integer NOT NULL,
    channel_name character varying(255),
    channel_url text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE yt.dim_channels OWNER TO postgres;

--
-- Name: TABLE dim_channels; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.dim_channels IS 'Dimension table for YouTube channels. Sourced from the subtitles field in Google Takeout export.';


--
-- Name: dim_channels_channel_id_seq; Type: SEQUENCE; Schema: yt; Owner: postgres
--

CREATE SEQUENCE yt.dim_channels_channel_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE yt.dim_channels_channel_id_seq OWNER TO postgres;

--
-- Name: dim_channels_channel_id_seq; Type: SEQUENCE OWNED BY; Schema: yt; Owner: postgres
--

ALTER SEQUENCE yt.dim_channels_channel_id_seq OWNED BY yt.dim_channels.channel_id;


--
-- Name: dim_date; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.dim_date (
    date_key date NOT NULL,
    year smallint NOT NULL,
    quarter smallint NOT NULL,
    month smallint NOT NULL,
    month_name character varying(10) NOT NULL,
    week_of_year smallint NOT NULL,
    day_of_month smallint NOT NULL,
    day_of_week_num smallint NOT NULL,
    day_of_week character varying(10) NOT NULL,
    is_weekend boolean NOT NULL,
    season character varying(10) NOT NULL,
    is_danish_holiday boolean DEFAULT false
);


ALTER TABLE yt.dim_date OWNER TO postgres;

--
-- Name: TABLE dim_date; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.dim_date IS 'Date dimension table. Pre-computed date attributes from 2020 through 2030 for fast analytical joins without repeated EXTRACT calls.';


--
-- Name: dim_videos; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.dim_videos (
    video_id character varying(20) NOT NULL,
    title text,
    title_url text,
    channel_id integer,
    status character varying(20) DEFAULT 'active'::character varying,
    description text,
    tags text[],
    category_id integer,
    duration_seconds integer,
    view_count bigint,
    like_count bigint,
    has_transcript boolean DEFAULT false,
    enriched_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT dim_videos_status_check CHECK (((status)::text = ANY ((ARRAY['active'::character varying, 'unavailable'::character varying, 'private'::character varying])::text[])))
);


ALTER TABLE yt.dim_videos OWNER TO postgres;

--
-- Name: TABLE dim_videos; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.dim_videos IS 'Dimension table for unique videos. Core fields from Takeout export; enrichment fields populated later via YouTube Data API (Tier 2).';


--
-- Name: etl_log; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.etl_log (
    log_id integer NOT NULL,
    file_name text NOT NULL,
    run_at timestamp with time zone DEFAULT now(),
    total_raw_records integer NOT NULL,
    ads_filtered integer NOT NULL,
    removed_dropped integer NOT NULL,
    ghost_videos_kept integer NOT NULL,
    clean_records integer NOT NULL,
    new_channels integer DEFAULT 0,
    new_videos integer DEFAULT 0,
    new_events integer DEFAULT 0,
    duplicate_events integer DEFAULT 0,
    duration_seconds double precision
);


ALTER TABLE yt.etl_log OWNER TO postgres;

--
-- Name: TABLE etl_log; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.etl_log IS 'Audit log for ETL ingest runs. Tracks record counts and filtering at each stage for traceability.';


--
-- Name: etl_log_log_id_seq; Type: SEQUENCE; Schema: yt; Owner: postgres
--

CREATE SEQUENCE yt.etl_log_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE yt.etl_log_log_id_seq OWNER TO postgres;

--
-- Name: etl_log_log_id_seq; Type: SEQUENCE OWNED BY; Schema: yt; Owner: postgres
--

ALTER SEQUENCE yt.etl_log_log_id_seq OWNED BY yt.etl_log.log_id;


--
-- Name: fact_watch_events; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.fact_watch_events (
    event_id integer NOT NULL,
    video_id character varying(20) NOT NULL,
    channel_id integer,
    watched_at timestamp with time zone NOT NULL
);


ALTER TABLE yt.fact_watch_events OWNER TO postgres;

--
-- Name: TABLE fact_watch_events; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.fact_watch_events IS 'Fact table: one row per YouTube watch event. Deduplicated on (video_id, watched_at) to handle overlapping quarterly exports.';


--
-- Name: fact_watch_events_event_id_seq; Type: SEQUENCE; Schema: yt; Owner: postgres
--

CREATE SEQUENCE yt.fact_watch_events_event_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE yt.fact_watch_events_event_id_seq OWNER TO postgres;

--
-- Name: fact_watch_events_event_id_seq; Type: SEQUENCE OWNED BY; Schema: yt; Owner: postgres
--

ALTER SEQUENCE yt.fact_watch_events_event_id_seq OWNED BY yt.fact_watch_events.event_id;


--
-- Name: mv_channel_funnel; Type: MATERIALIZED VIEW; Schema: yt; Owner: postgres
--

CREATE MATERIALIZED VIEW yt.mv_channel_funnel AS
 WITH channel_stats AS (
         SELECT dc.channel_id,
            dc.channel_name,
            count(*) AS total_views,
            count(DISTINCT fe.video_id) AS unique_videos,
            count(DISTINCT date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text))) AS active_days,
            min(fe.watched_at) AS first_watch,
            max(fe.watched_at) AS last_watch,
            (EXTRACT(epoch FROM (max(fe.watched_at) - min(fe.watched_at))) / 86400.0) AS span_days
           FROM (yt.fact_watch_events fe
             JOIN yt.dim_channels dc ON ((fe.channel_id = dc.channel_id)))
          GROUP BY dc.channel_id, dc.channel_name
        )
 SELECT channel_id,
    channel_name,
    total_views,
    unique_videos,
    active_days,
    first_watch,
    last_watch,
    span_days,
        CASE
            WHEN (total_views = 1) THEN 'Discovery'::text
            WHEN ((total_views >= 2) AND (total_views <= 5)) THEN 'Casual'::text
            WHEN (((total_views >= 6) AND (total_views <= 20)) AND (active_days >= 3)) THEN 'Regular'::text
            WHEN ((total_views > 20) AND (active_days >= 7) AND (span_days >= (30)::numeric)) THEN 'Loyal'::text
            ELSE 'Casual'::text
        END AS funnel_stage
   FROM channel_stats
  WITH NO DATA;


ALTER MATERIALIZED VIEW yt.mv_channel_funnel OWNER TO postgres;

--
-- Name: MATERIALIZED VIEW mv_channel_funnel; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON MATERIALIZED VIEW yt.mv_channel_funnel IS 'Channel loyalty funnel: classifies each channel as Discovery, Casual, Regular, or Loyal based on viewing behavior.';


--
-- Name: mv_channel_monthly; Type: MATERIALIZED VIEW; Schema: yt; Owner: postgres
--

CREATE MATERIALIZED VIEW yt.mv_channel_monthly AS
 SELECT dc.channel_id,
    dc.channel_name,
    dd.year,
    dd.month,
    dd.month_name,
    count(*) AS views,
    count(DISTINCT fe.video_id) AS unique_videos
   FROM ((yt.fact_watch_events fe
     JOIN yt.dim_channels dc ON ((fe.channel_id = dc.channel_id)))
     JOIN yt.dim_date dd ON ((dd.date_key = date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))))
  GROUP BY dc.channel_id, dc.channel_name, dd.year, dd.month, dd.month_name
  ORDER BY dd.year, dd.month, (count(*)) DESC
  WITH NO DATA;


ALTER MATERIALIZED VIEW yt.mv_channel_monthly OWNER TO postgres;

--
-- Name: MATERIALIZED VIEW mv_channel_monthly; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON MATERIALIZED VIEW yt.mv_channel_monthly IS 'Channel activity per month. Powers channel trend charts and top-channels-by-period queries.';


--
-- Name: mv_daily_summary; Type: MATERIALIZED VIEW; Schema: yt; Owner: postgres
--

CREATE MATERIALIZED VIEW yt.mv_daily_summary AS
 SELECT dd.date_key,
    dd.year,
    dd.quarter,
    dd.month,
    dd.month_name,
    dd.day_of_week,
    dd.day_of_week_num,
    dd.is_weekend,
    dd.season,
    dd.is_danish_holiday,
    COALESCE(count(fe.event_id), (0)::bigint) AS videos_watched,
    COALESCE(count(DISTINCT fe.video_id), (0)::bigint) AS unique_videos,
    COALESCE(count(DISTINCT fe.channel_id), (0)::bigint) AS unique_channels,
    min((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)) AS first_watch,
    max((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)) AS last_watch
   FROM (yt.dim_date dd
     LEFT JOIN yt.fact_watch_events fe ON ((dd.date_key = date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))))
  WHERE (dd.date_key <= CURRENT_DATE)
  GROUP BY dd.date_key, dd.year, dd.quarter, dd.month, dd.month_name, dd.day_of_week, dd.day_of_week_num, dd.is_weekend, dd.season, dd.is_danish_holiday
  ORDER BY dd.date_key
  WITH NO DATA;


ALTER MATERIALIZED VIEW yt.mv_daily_summary OWNER TO postgres;

--
-- Name: MATERIALIZED VIEW mv_daily_summary; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON MATERIALIZED VIEW yt.mv_daily_summary IS 'Pre-computed daily activity summary. Includes zero-activity days from dim_date. Refresh after each ETL run.';


--
-- Name: mv_hourly_distribution; Type: MATERIALIZED VIEW; Schema: yt; Owner: postgres
--

CREATE MATERIALIZED VIEW yt.mv_hourly_distribution AS
 SELECT dd.year,
    dd.day_of_week_num,
    dd.day_of_week,
    dd.is_weekend,
    (EXTRACT(hour FROM (fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))::smallint AS hour,
    count(*) AS views
   FROM (yt.fact_watch_events fe
     JOIN yt.dim_date dd ON ((dd.date_key = date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))))
  GROUP BY dd.year, dd.day_of_week_num, dd.day_of_week, dd.is_weekend, (EXTRACT(hour FROM (fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))
  WITH NO DATA;


ALTER MATERIALIZED VIEW yt.mv_hourly_distribution OWNER TO postgres;

--
-- Name: MATERIALIZED VIEW mv_hourly_distribution; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON MATERIALIZED VIEW yt.mv_hourly_distribution IS 'Pre-computed hourly viewing distribution by day-of-week and year. Powers heatmap visualizations.';


--
-- Name: mv_monthly_summary; Type: MATERIALIZED VIEW; Schema: yt; Owner: postgres
--

CREATE MATERIALIZED VIEW yt.mv_monthly_summary AS
 SELECT dd.year,
    dd.month,
    dd.month_name,
    dd.season,
    count(fe.event_id) AS total_views,
    count(DISTINCT fe.video_id) AS unique_videos,
    count(DISTINCT fe.channel_id) AS unique_channels,
    count(DISTINCT dd.date_key) FILTER (WHERE (fe.event_id IS NOT NULL)) AS active_days,
    round(((count(fe.event_id))::numeric / (NULLIF(count(DISTINCT dd.date_key) FILTER (WHERE (fe.event_id IS NOT NULL)), 0))::numeric), 1) AS avg_videos_per_active_day
   FROM (yt.dim_date dd
     LEFT JOIN yt.fact_watch_events fe ON ((dd.date_key = date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))))
  WHERE ((dd.date_key <= CURRENT_DATE) AND (dd.date_key >= '2020-01-01'::date))
  GROUP BY dd.year, dd.month, dd.month_name, dd.season
  ORDER BY dd.year, dd.month
  WITH NO DATA;


ALTER MATERIALIZED VIEW yt.mv_monthly_summary OWNER TO postgres;

--
-- Name: MATERIALIZED VIEW mv_monthly_summary; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON MATERIALIZED VIEW yt.mv_monthly_summary IS 'Pre-computed monthly activity metrics. Core data source for trend dashboards.';


--
-- Name: v_sessions; Type: VIEW; Schema: yt; Owner: postgres
--

CREATE VIEW yt.v_sessions AS
 WITH gaps AS (
         SELECT fact_watch_events.event_id,
            fact_watch_events.video_id,
            fact_watch_events.channel_id,
            fact_watch_events.watched_at,
            (fact_watch_events.watched_at AT TIME ZONE 'Europe/Copenhagen'::text) AS watched_at_dk,
            lag(fact_watch_events.watched_at) OVER (ORDER BY fact_watch_events.watched_at) AS prev_watched_at,
            (EXTRACT(epoch FROM (fact_watch_events.watched_at - lag(fact_watch_events.watched_at) OVER (ORDER BY fact_watch_events.watched_at))) / 60.0) AS gap_minutes
           FROM yt.fact_watch_events
        ), session_flags AS (
         SELECT gaps.event_id,
            gaps.video_id,
            gaps.channel_id,
            gaps.watched_at,
            gaps.watched_at_dk,
            gaps.prev_watched_at,
            gaps.gap_minutes,
                CASE
                    WHEN ((gaps.gap_minutes IS NULL) OR (gaps.gap_minutes > (45)::numeric)) THEN 1
                    ELSE 0
                END AS new_session_flag
           FROM gaps
        ), session_ids AS (
         SELECT session_flags.event_id,
            session_flags.video_id,
            session_flags.channel_id,
            session_flags.watched_at,
            session_flags.watched_at_dk,
            session_flags.prev_watched_at,
            session_flags.gap_minutes,
            session_flags.new_session_flag,
            sum(session_flags.new_session_flag) OVER (ORDER BY session_flags.watched_at) AS session_id
           FROM session_flags
        )
 SELECT session_id,
    event_id,
    video_id,
    channel_id,
    watched_at,
    watched_at_dk,
    gap_minutes,
    count(*) OVER (PARTITION BY session_id) AS session_video_count,
    min(watched_at) OVER (PARTITION BY session_id) AS session_start,
    max(watched_at) OVER (PARTITION BY session_id) AS session_end
   FROM session_ids;


ALTER VIEW yt.v_sessions OWNER TO postgres;

--
-- Name: VIEW v_sessions; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON VIEW yt.v_sessions IS 'Session detection view. Groups consecutive watch events into sessions using a 45-minute gap threshold. Use session_id to aggregate binge behavior.';


--
-- Name: mv_session_summary; Type: MATERIALIZED VIEW; Schema: yt; Owner: postgres
--

CREATE MATERIALIZED VIEW yt.mv_session_summary AS
 SELECT session_id,
    count(*) AS video_count,
    min(watched_at_dk) AS session_start,
    max(watched_at_dk) AS session_end,
    (EXTRACT(epoch FROM (max(watched_at) - min(watched_at))) / 60.0) AS duration_minutes
   FROM yt.v_sessions
  GROUP BY session_id
  WITH NO DATA;


ALTER MATERIALIZED VIEW yt.mv_session_summary OWNER TO postgres;

--
-- Name: MATERIALIZED VIEW mv_session_summary; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON MATERIALIZED VIEW yt.mv_session_summary IS 'Pre-computed session-level metrics. One row per session with video count and duration.';


--
-- Name: nlp_embeddings; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.nlp_embeddings (
    video_id character varying(20) NOT NULL,
    embedding double precision[] NOT NULL,
    model_name character varying(100) NOT NULL,
    model_version character varying(50),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE yt.nlp_embeddings OWNER TO postgres;

--
-- Name: TABLE nlp_embeddings; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.nlp_embeddings IS 'Dense vector embeddings per video for similarity search and recommendation. Stored as FLOAT[] — migrate to pgvector type for production similarity queries.';


--
-- Name: nlp_topics; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.nlp_topics (
    topic_id integer NOT NULL,
    topic_label character varying(100),
    keywords text[] NOT NULL,
    model_name character varying(100) NOT NULL,
    model_version character varying(50),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE yt.nlp_topics OWNER TO postgres;

--
-- Name: TABLE nlp_topics; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.nlp_topics IS 'Topics identified by NLP models. Each row is one topic with its keywords and source model metadata.';


--
-- Name: nlp_topics_topic_id_seq; Type: SEQUENCE; Schema: yt; Owner: postgres
--

CREATE SEQUENCE yt.nlp_topics_topic_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE yt.nlp_topics_topic_id_seq OWNER TO postgres;

--
-- Name: nlp_topics_topic_id_seq; Type: SEQUENCE OWNED BY; Schema: yt; Owner: postgres
--

ALTER SEQUENCE yt.nlp_topics_topic_id_seq OWNED BY yt.nlp_topics.topic_id;


--
-- Name: nlp_video_topics; Type: TABLE; Schema: yt; Owner: postgres
--

CREATE TABLE yt.nlp_video_topics (
    video_id character varying(20) NOT NULL,
    topic_id integer NOT NULL,
    score double precision NOT NULL,
    CONSTRAINT nlp_video_topics_score_check CHECK (((score >= (0)::double precision) AND (score <= (1)::double precision)))
);


ALTER TABLE yt.nlp_video_topics OWNER TO postgres;

--
-- Name: TABLE nlp_video_topics; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON TABLE yt.nlp_video_topics IS 'Junction table mapping videos to topics with relevance scores. Supports multi-topic assignment per video.';


--
-- Name: v_channel_loyalty; Type: VIEW; Schema: yt; Owner: postgres
--

CREATE VIEW yt.v_channel_loyalty AS
 SELECT dc.channel_name,
    dc.channel_url,
    count(*) AS total_views,
    count(DISTINCT dv.video_id) AS unique_videos,
    count(DISTINCT date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text))) AS active_days,
    min((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)) AS first_watched_dk,
    max((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)) AS last_watched_dk
   FROM ((yt.fact_watch_events fe
     JOIN yt.dim_channels dc ON ((fe.channel_id = dc.channel_id)))
     JOIN yt.dim_videos dv ON (((fe.video_id)::text = (dv.video_id)::text)))
  GROUP BY dc.channel_id, dc.channel_name, dc.channel_url;


ALTER VIEW yt.v_channel_loyalty OWNER TO postgres;

--
-- Name: VIEW v_channel_loyalty; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON VIEW yt.v_channel_loyalty IS 'Channel loyalty metrics: total views, unique videos, active days, and watch span per channel.';


--
-- Name: v_discovery_events; Type: VIEW; Schema: yt; Owner: postgres
--

CREATE VIEW yt.v_discovery_events AS
 WITH ranked AS (
         SELECT fe.event_id,
            fe.video_id,
            fe.channel_id,
            fe.watched_at,
            dc.channel_name,
            row_number() OVER (PARTITION BY fe.channel_id ORDER BY fe.watched_at) AS channel_watch_rank,
            row_number() OVER (PARTITION BY fe.video_id ORDER BY fe.watched_at) AS video_watch_rank
           FROM (yt.fact_watch_events fe
             LEFT JOIN yt.dim_channels dc ON ((fe.channel_id = dc.channel_id)))
        )
 SELECT event_id,
    video_id,
    channel_id,
    watched_at,
    channel_name,
    channel_watch_rank,
    video_watch_rank,
    (channel_watch_rank = 1) AS is_channel_discovery,
    (video_watch_rank = 1) AS is_first_watch
   FROM ranked;


ALTER VIEW yt.v_discovery_events OWNER TO postgres;

--
-- Name: VIEW v_discovery_events; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON VIEW yt.v_discovery_events IS 'Flags first-time channel discoveries and first video watches. Use is_channel_discovery to analyze when new channels enter the viewing repertoire.';


--
-- Name: v_etl_history; Type: VIEW; Schema: yt; Owner: postgres
--

CREATE VIEW yt.v_etl_history AS
 SELECT file_name,
    (run_at AT TIME ZONE 'Europe/Copenhagen'::text) AS run_at_dk,
    total_raw_records,
    ads_filtered,
    removed_dropped,
    new_events,
    duplicate_events,
    duration_seconds
   FROM yt.etl_log
  ORDER BY run_at DESC;


ALTER VIEW yt.v_etl_history OWNER TO postgres;

--
-- Name: v_watch_history; Type: VIEW; Schema: yt; Owner: postgres
--

CREATE VIEW yt.v_watch_history AS
 SELECT fe.event_id,
    (fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text) AS watched_at_dk,
    dv.video_id,
    dv.title,
    dv.title_url,
    dv.status,
    dc.channel_name,
    dc.channel_url,
    dcat.category_name,
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
    EXTRACT(hour FROM (fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)) AS hour
   FROM ((((yt.fact_watch_events fe
     JOIN yt.dim_videos dv ON (((fe.video_id)::text = (dv.video_id)::text)))
     LEFT JOIN yt.dim_channels dc ON ((fe.channel_id = dc.channel_id)))
     LEFT JOIN yt.dim_categories dcat ON ((dv.category_id = dcat.category_id)))
     LEFT JOIN yt.dim_date dd ON ((dd.date_key = date((fe.watched_at AT TIME ZONE 'Europe/Copenhagen'::text)))));


ALTER VIEW yt.v_watch_history OWNER TO postgres;

--
-- Name: VIEW v_watch_history; Type: COMMENT; Schema: yt; Owner: postgres
--

COMMENT ON VIEW yt.v_watch_history IS 'Denormalized view of all watch events with Danish timezone conversion. Date attributes from dim_date; hour extracted at query time.';


--
-- Name: dim_channels channel_id; Type: DEFAULT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_channels ALTER COLUMN channel_id SET DEFAULT nextval('yt.dim_channels_channel_id_seq'::regclass);


--
-- Name: etl_log log_id; Type: DEFAULT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.etl_log ALTER COLUMN log_id SET DEFAULT nextval('yt.etl_log_log_id_seq'::regclass);


--
-- Name: fact_watch_events event_id; Type: DEFAULT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.fact_watch_events ALTER COLUMN event_id SET DEFAULT nextval('yt.fact_watch_events_event_id_seq'::regclass);


--
-- Name: nlp_topics topic_id; Type: DEFAULT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_topics ALTER COLUMN topic_id SET DEFAULT nextval('yt.nlp_topics_topic_id_seq'::regclass);


--
-- Name: dim_categories dim_categories_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_categories
    ADD CONSTRAINT dim_categories_pkey PRIMARY KEY (category_id);


--
-- Name: dim_channels dim_channels_channel_name_channel_url_key; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_channels
    ADD CONSTRAINT dim_channels_channel_name_channel_url_key UNIQUE (channel_name, channel_url);


--
-- Name: dim_channels dim_channels_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_channels
    ADD CONSTRAINT dim_channels_pkey PRIMARY KEY (channel_id);


--
-- Name: dim_date dim_date_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_date
    ADD CONSTRAINT dim_date_pkey PRIMARY KEY (date_key);


--
-- Name: dim_videos dim_videos_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_videos
    ADD CONSTRAINT dim_videos_pkey PRIMARY KEY (video_id);


--
-- Name: etl_log etl_log_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.etl_log
    ADD CONSTRAINT etl_log_pkey PRIMARY KEY (log_id);


--
-- Name: fact_watch_events fact_watch_events_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.fact_watch_events
    ADD CONSTRAINT fact_watch_events_pkey PRIMARY KEY (event_id);


--
-- Name: fact_watch_events fact_watch_events_video_id_watched_at_key; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.fact_watch_events
    ADD CONSTRAINT fact_watch_events_video_id_watched_at_key UNIQUE (video_id, watched_at);


--
-- Name: nlp_embeddings nlp_embeddings_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_embeddings
    ADD CONSTRAINT nlp_embeddings_pkey PRIMARY KEY (video_id);


--
-- Name: nlp_topics nlp_topics_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_topics
    ADD CONSTRAINT nlp_topics_pkey PRIMARY KEY (topic_id);


--
-- Name: nlp_video_topics nlp_video_topics_pkey; Type: CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_video_topics
    ADD CONSTRAINT nlp_video_topics_pkey PRIMARY KEY (video_id, topic_id);


--
-- Name: idx_mv_channel_funnel; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE UNIQUE INDEX idx_mv_channel_funnel ON yt.mv_channel_funnel USING btree (channel_id);


--
-- Name: idx_mv_channel_monthly; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE UNIQUE INDEX idx_mv_channel_monthly ON yt.mv_channel_monthly USING btree (channel_id, year, month);


--
-- Name: idx_mv_daily_summary_date; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE UNIQUE INDEX idx_mv_daily_summary_date ON yt.mv_daily_summary USING btree (date_key);


--
-- Name: idx_mv_hourly_dist; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE UNIQUE INDEX idx_mv_hourly_dist ON yt.mv_hourly_distribution USING btree (year, day_of_week_num, hour);


--
-- Name: idx_mv_monthly_summary; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE UNIQUE INDEX idx_mv_monthly_summary ON yt.mv_monthly_summary USING btree (year, month);


--
-- Name: idx_mv_session_summary_id; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE UNIQUE INDEX idx_mv_session_summary_id ON yt.mv_session_summary USING btree (session_id);


--
-- Name: idx_video_topics_topic_id; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_video_topics_topic_id ON yt.nlp_video_topics USING btree (topic_id);


--
-- Name: idx_videos_category_id; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_videos_category_id ON yt.dim_videos USING btree (category_id);


--
-- Name: idx_videos_channel_id; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_videos_channel_id ON yt.dim_videos USING btree (channel_id);


--
-- Name: idx_videos_status; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_videos_status ON yt.dim_videos USING btree (status);


--
-- Name: idx_watch_events_channel_id; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_watch_events_channel_id ON yt.fact_watch_events USING btree (channel_id);


--
-- Name: idx_watch_events_video_id; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_watch_events_video_id ON yt.fact_watch_events USING btree (video_id);


--
-- Name: idx_watch_events_watched_at; Type: INDEX; Schema: yt; Owner: postgres
--

CREATE INDEX idx_watch_events_watched_at ON yt.fact_watch_events USING btree (watched_at);


--
-- Name: dim_videos dim_videos_category_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_videos
    ADD CONSTRAINT dim_videos_category_id_fkey FOREIGN KEY (category_id) REFERENCES yt.dim_categories(category_id);


--
-- Name: dim_videos dim_videos_channel_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.dim_videos
    ADD CONSTRAINT dim_videos_channel_id_fkey FOREIGN KEY (channel_id) REFERENCES yt.dim_channels(channel_id);


--
-- Name: fact_watch_events fact_watch_events_channel_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.fact_watch_events
    ADD CONSTRAINT fact_watch_events_channel_id_fkey FOREIGN KEY (channel_id) REFERENCES yt.dim_channels(channel_id);


--
-- Name: fact_watch_events fact_watch_events_video_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.fact_watch_events
    ADD CONSTRAINT fact_watch_events_video_id_fkey FOREIGN KEY (video_id) REFERENCES yt.dim_videos(video_id);


--
-- Name: nlp_embeddings nlp_embeddings_video_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_embeddings
    ADD CONSTRAINT nlp_embeddings_video_id_fkey FOREIGN KEY (video_id) REFERENCES yt.dim_videos(video_id) ON DELETE CASCADE;


--
-- Name: nlp_video_topics nlp_video_topics_topic_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_video_topics
    ADD CONSTRAINT nlp_video_topics_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES yt.nlp_topics(topic_id) ON DELETE CASCADE;


--
-- Name: nlp_video_topics nlp_video_topics_video_id_fkey; Type: FK CONSTRAINT; Schema: yt; Owner: postgres
--

ALTER TABLE ONLY yt.nlp_video_topics
    ADD CONSTRAINT nlp_video_topics_video_id_fkey FOREIGN KEY (video_id) REFERENCES yt.dim_videos(video_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

