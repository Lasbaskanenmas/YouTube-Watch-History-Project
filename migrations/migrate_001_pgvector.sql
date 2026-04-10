-- ============================================================================
-- Migration 001: pgvector — Convert nlp_embeddings to native vector type
-- ============================================================================
-- Run once, manually, after pgvector has been installed on the system.
-- Prerequisites:
--   - pgvector compiled and installed (see project README)
--   - Connected to youtube_analytics database in pgAdmin
-- Date: 2026-04-05
-- ============================================================================


-- Step 1: Enable the pgvector extension (safe to re-run)
CREATE EXTENSION IF NOT EXISTS vector;


-- Step 2: Drop the old FLOAT[] embedding column
ALTER TABLE yt.nlp_embeddings DROP COLUMN IF EXISTS embedding;


-- Step 3: Add embedding as a proper vector(384) column
-- 384 dimensions matches paraphrase-multilingual-MiniLM-L12-v2
ALTER TABLE yt.nlp_embeddings
    ADD COLUMN embedding vector(384);

COMMENT ON COLUMN yt.nlp_embeddings.embedding IS
    '384-dimensional sentence embedding from paraphrase-multilingual-MiniLM-L12-v2. '
    'NULL if embed_failed_at is set. Indexed via HNSW for cosine similarity search.';


-- Step 4: Add embed_failed_at column for permanent failure tracking
-- The resume query skips rows where this is NOT NULL, preventing
-- infinite retries on videos that consistently fail to embed.
ALTER TABLE yt.nlp_embeddings
    ADD COLUMN IF NOT EXISTS embed_failed_at TIMESTAMPTZ;

COMMENT ON COLUMN yt.nlp_embeddings.embed_failed_at IS
    'Set when embedding fails permanently for this video (e.g. no text, '
    'malformed Unicode, model error). Resume logic skips these rows.';


-- Step 5: HNSW index for fast approximate cosine similarity search
-- HNSW is preferred over IVFFlat at this scale (~54K vectors) because:
--   - No training step required
--   - Better recall at equivalent query speed
--   - Handles incremental inserts cleanly (quarterly ingest pattern)
CREATE INDEX IF NOT EXISTS idx_nlp_embeddings_hnsw
    ON yt.nlp_embeddings
    USING hnsw (embedding vector_cosine_ops);


-- ============================================================================
-- Verification
-- ============================================================================
-- Run this after the migration to confirm all changes landed correctly.
-- Expected output:
--   video_id      | character varying | varchar
--   embedding     | USER-DEFINED      | vector
--   embed_failed_at | timestamp with time zone | timestamptz
--   model_name    | character varying | varchar
--   model_version | character varying | varchar
--   created_at    | timestamp with...| timestamptz

SELECT
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_schema = 'yt'
AND   table_name   = 'nlp_embeddings'
ORDER BY ordinal_position;