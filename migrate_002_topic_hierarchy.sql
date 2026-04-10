-- ============================================================================
-- Migration 002: Topic Hierarchy — Link nlp_topics to dim_categories
-- ============================================================================
-- Adds category_id FK to nlp_topics so BERTopic granular topics (Level 2)
-- anchor to YouTube's official category taxonomy (Level 1).
-- Date: 2026-04-05
-- ============================================================================

ALTER TABLE yt.nlp_topics
    ADD COLUMN IF NOT EXISTS category_id INTEGER
        REFERENCES yt.dim_categories(category_id);

COMMENT ON COLUMN yt.nlp_topics.category_id IS
    'FK to dim_categories. Links each granular BERTopic topic (Level 2) to '
    'its parent YouTube category (Level 1). NULL until topic_model.py assigns it.';

-- Verification
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'yt'
AND table_name = 'nlp_topics'
ORDER BY ordinal_position;