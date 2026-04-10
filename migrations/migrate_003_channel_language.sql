-- ============================================================================
-- Migration 003: Channel Language — Add primary_language and language_source
-- ============================================================================
-- Adds two columns to dim_channels to support the two-tier language inference
-- system used by the topic modelling pipeline.
--
-- Tier 1 (semi-automatic): Top channels reviewed manually via CSV workflow.
--                          language_source = 'human'
-- Tier 2a (automatic):     Unicode diacritic detection on video titles.
--                          language_source = 'unicode'
-- Tier 2b (LLM-assisted):  Mistral 7B infers language from channel name.
--                          language_source = 'mistral'
--
-- NULL default is intentional — NULL means "not yet assessed".
-- The topic labelling pipeline treats NULL as English until inference runs.
-- This prevents false assumptions from polluting the audit trail.
--
-- Date: 2026-04-10
-- ============================================================================


-- Step 1: Add primary_language column
ALTER TABLE yt.dim_channels
    ADD COLUMN IF NOT EXISTS primary_language VARCHAR(10) DEFAULT NULL;

COMMENT ON COLUMN yt.dim_channels.primary_language IS
    'ISO 639-1 language code for the channel''s primary language. '
    'e.g. ''en'', ''da'', ''ka'', ''es'', ''de''. '
    'NULL means not yet assessed — treated as English by topic pipeline.';


-- Step 2: Add language_source column
ALTER TABLE yt.dim_channels
    ADD COLUMN IF NOT EXISTS language_source VARCHAR(20) DEFAULT NULL;

COMMENT ON COLUMN yt.dim_channels.language_source IS
    'How the language tag was assigned: '
    '''human'' = manually reviewed via CSV, '
    '''unicode'' = auto-detected via diacritic character patterns, '
    '''mistral'' = inferred by Mistral 7B from channel name. '
    'NULL when primary_language is NULL.';


-- Step 3: Add index for fast language-based filtering in topic pipeline
CREATE INDEX IF NOT EXISTS idx_channels_primary_language
    ON yt.dim_channels (primary_language);


-- ============================================================================
-- Verification
-- ============================================================================
SELECT
    column_name,
    data_type,
    column_default,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'yt'
AND   table_name   = 'dim_channels'
ORDER BY ordinal_position;
