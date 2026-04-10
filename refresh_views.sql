-- ============================================================================
-- YouTube Analytics — Refresh Materialized Views
-- ============================================================================
-- Run this after every ETL ingest to update pre-computed analytics.
-- Usage: Execute in pgAdmin Query Tool on youtube_analytics
-- ============================================================================

REFRESH MATERIALIZED VIEW CONCURRENTLY yt.mv_daily_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY yt.mv_session_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY yt.mv_monthly_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY yt.mv_channel_monthly;
REFRESH MATERIALIZED VIEW CONCURRENTLY yt.mv_hourly_distribution;
REFRESH MATERIALIZED VIEW CONCURRENTLY yt.mv_channel_funnel;