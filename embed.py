"""
YouTube Analytics — Embedding Pipeline
=======================================
Generates dense vector embeddings for all active videos and writes them
to yt.nlp_embeddings using the paraphrase-multilingual-MiniLM-L12-v2
sentence-transformer model (384 dimensions, 50+ languages).

Text input per video (in order of richness):
    title + tags + cleaned description

Rules:
    - Only processes videos with status = 'active'.
    - Database-driven resume: skips videos already present in nlp_embeddings
      and videos flagged with embed_failed_at (permanent failures).
    - Commits every BATCH_SIZE videos — a crash loses at most one batch.
    - Errors on individual videos are logged to file and flagged in the DB;
      the run continues regardless.
    - Progress is displayed via tqdm with videos/sec and ETA.
    - All runs are logged to yt.embed_log.

Hardware:
    - Uses MPS (Apple Silicon GPU) when available.
    - Falls back to CPU automatically if MPS is unavailable.

Prerequisites:
    pip install psycopg2-binary pgvector sentence-transformers tqdm python-dotenv

Usage:
    python embed.py
"""

import os
import re
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    raise ImportError("Install psycopg2-binary: pip install psycopg2-binary")

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    raise ImportError("Install pgvector: pip install pgvector")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers")

try:
    import torch
except ImportError:
    raise ImportError("Install torch: pip install torch")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Install tqdm: pip install tqdm")

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("Install python-dotenv: pip install python-dotenv")


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

DB_CONFIG = {
    "dbname":   "youtube_analytics",
    "user":     "postgres",     # <-- Update if different
    "password": "qdw83nmm",
    "host":     "localhost",
    "port":     5432
}

MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_VERSION   = "v1.0"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 256       # Videos per DB commit cycle
ENCODE_BATCH    = 64        # Videos per model inference call (tune for memory)

# Log directory — relative to this script's location
LOG_DIR = Path(__file__).parent / "logs" / "ml_log"


# ============================================================================
# Embed Log Table (auto-created if missing)
# ============================================================================

CREATE_EMBED_LOG = """
CREATE TABLE IF NOT EXISTS yt.embed_log (
    log_id              SERIAL          PRIMARY KEY,
    run_at              TIMESTAMPTZ     DEFAULT NOW(),
    model_name          VARCHAR(100)    NOT NULL,
    model_version       VARCHAR(50),
    videos_targeted     INTEGER         NOT NULL,
    videos_embedded     INTEGER         NOT NULL DEFAULT 0,
    videos_failed       INTEGER         NOT NULL DEFAULT 0,
    videos_skipped      INTEGER         NOT NULL DEFAULT 0,
    duration_seconds    FLOAT
);

COMMENT ON TABLE yt.embed_log IS
    'Audit log for embedding pipeline runs. Tracks how many videos were '
    'embedded, failed, or skipped per run.';
"""

ADD_FAILED_AT_COLUMN = """
ALTER TABLE yt.nlp_embeddings
    ADD COLUMN IF NOT EXISTS embed_failed_at TIMESTAMPTZ;

COMMENT ON COLUMN yt.nlp_embeddings.embed_failed_at IS
    'Set when embedding fails permanently for this video. '
    'Resume logic skips rows where this is NOT NULL.';
"""


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(log_dir: Path) -> logging.Logger:
    """
    Configure a logger that writes to both the console and a per-run
    timestamped file in logs/ml_log/.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = log_dir / f"embed_{run_ts}.log"

    logger = logging.getLogger("embed")
    logger.setLevel(logging.DEBUG)

    # File handler — DEBUG and above (captures all failures in detail)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — INFO and above (keeps terminal clean)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


# ============================================================================
# Device Detection
# ============================================================================

def detect_device() -> str:
    """
    Return the best available device for sentence-transformers:
      - 'mps'  on Apple Silicon (Metal GPU)
      - 'cpu'  as universal fallback
    CUDA is not checked as this machine uses Apple Silicon.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


# ============================================================================
# Text Cleaning
# ============================================================================

# Patterns that add noise without semantic value
_RE_URL         = re.compile(r"https?://\S+|www\.\S+")
_RE_TIMESTAMP   = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_RE_HASHTAG     = re.compile(r"#\w+")
_RE_SUBSCRIBE   = re.compile(
    r"(subscribe|click the bell|hit the bell|turn on notifications"
    r"|follow us|join the channel|support us on patreon"
    r"|check out our merch|link in (the )?bio|link in description"
    r"|use code \w+|affiliate link|sponsored by|this video (is )?sponsored)",
    re.IGNORECASE
)
_RE_SOCIALS     = re.compile(
    r"(instagram|twitter|tiktok|facebook|twitch|discord|reddit"
    r"|linkedin|snapchat)\s*[:\-@]?\s*@?\w*",
    re.IGNORECASE
)
_RE_WHITESPACE  = re.compile(r"\s{2,}")


def clean_description(text: str | None) -> str:
    """
    Strip noise from a YouTube description, retaining only semantic content.

    Removes: URLs, timestamps, hashtags, boilerplate subscribe/social CTAs,
    sponsor mentions, and excessive whitespace.
    """
    if not text:
        return ""

    text = _RE_URL.sub(" ", text)
    text = _RE_TIMESTAMP.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)
    text = _RE_SUBSCRIBE.sub(" ", text)
    text = _RE_SOCIALS.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)

    # Truncate cleaned description to 512 chars — model handles up to 128
    # tokens, so long descriptions are redundant after cleaning anyway
    return text.strip()[:512]


def build_input_text(title: str | None,
                     tags: list[str] | None,
                     description: str | None) -> str:
    """
    Combine title, tags, and cleaned description into a single input string.

    Format:  "<title>. <tag1>, <tag2>. <cleaned description>"

    Rationale:
        - Title anchors the embedding (highest information density).
        - Tags add creator-defined keywords that titles omit.
        - Description adds conceptual depth after cleaning.
    """
    parts = []

    if title:
        parts.append(title.strip())

    if tags:
        # Join tags as a comma-separated phrase — readable by the model
        parts.append(", ".join(t.strip() for t in tags if t.strip()))

    cleaned = clean_description(description)
    if cleaned:
        parts.append(cleaned)

    return ". ".join(parts) if parts else ""


# ============================================================================
# Core Embedding Pipeline
# ============================================================================

def run_embedding():
    """
    Execute the full embedding pipeline:
    1. Detect hardware (MPS / CPU)
    2. Load the sentence-transformer model
    3. Fetch all unembedded active videos from the DB
    4. Generate embeddings in batches
    5. Write to nlp_embeddings with per-batch commits
    6. Flag individual failures with embed_failed_at
    7. Log the run to yt.embed_log
    """
    start_time  = time.time()
    logger      = setup_logger(LOG_DIR)

    # ----------------------------------------------------------------
    # Device + Model
    # ----------------------------------------------------------------
    device = detect_device()
    logger.info(f"Device:  {device.upper()}")
    logger.info(f"Model:   {MODEL_NAME}")
    logger.info(f"Dims:    {EMBEDDING_DIM}")
    logger.info("")

    logger.info("Loading sentence-transformer model...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("Model loaded.\n")

    # ----------------------------------------------------------------
    # Database connection
    # ----------------------------------------------------------------
    logger.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    register_vector(conn)
    cur = conn.cursor()

    try:
        cur.execute("SET search_path TO yt, public;")

        # Ensure support tables / columns exist
        cur.execute(CREATE_EMBED_LOG)
        cur.execute(ADD_FAILED_AT_COLUMN)
        conn.commit()

        # ----------------------------------------------------------------
        # STEP 1: Load unembedded active videos
        # ----------------------------------------------------------------
        logger.info("Loading unembedded active videos...")
        cur.execute("""
            SELECT
                dv.video_id,
                dv.title,
                dv.tags,
                dv.description
            FROM   yt.dim_videos dv
            WHERE  dv.status = 'active'
            AND    dv.video_id NOT IN (
                       SELECT video_id
                       FROM   yt.nlp_embeddings
                       WHERE  embed_failed_at IS NULL   -- exclude successes
                          OR  embed_failed_at IS NOT NULL  -- exclude perm failures
                   )
            ORDER  BY dv.video_id
        """)
        rows = cur.fetchall()
        videos_targeted = len(rows)

        if videos_targeted == 0:
            logger.info("No unembedded active videos found. Nothing to do.")
            _log_run(cur, conn, videos_targeted=0, videos_embedded=0,
                     videos_failed=0, videos_skipped=0,
                     duration=time.time() - start_time)
            return

        logger.info(f"Found {videos_targeted:,} videos to embed\n")

        # ----------------------------------------------------------------
        # STEP 2: Build input texts
        # ----------------------------------------------------------------
        video_ids   = [r[0] for r in rows]
        input_texts = [
            build_input_text(
                title=r[1],
                tags=r[2],
                description=r[3]
            )
            for r in rows
        ]

        # ----------------------------------------------------------------
        # STEP 3: Embed + write in BATCH_SIZE chunks
        # ----------------------------------------------------------------
        videos_embedded = 0
        videos_failed   = 0
        videos_skipped  = 0
        now             = datetime.now(timezone.utc)

        # Outer progress bar over DB commit batches
        n_batches = (videos_targeted + BATCH_SIZE - 1) // BATCH_SIZE

        with tqdm(total=videos_targeted,
                  unit="vid",
                  desc="Embedding",
                  dynamic_ncols=True) as pbar:

            for batch_idx in range(n_batches):
                start   = batch_idx * BATCH_SIZE
                end     = min(start + BATCH_SIZE, videos_targeted)

                b_ids   = video_ids[start:end]
                b_texts = input_texts[start:end]

                # ---- Skip videos with empty input text ----
                valid_mask  = [bool(t.strip()) for t in b_texts]
                valid_ids   = [i for i, v in zip(b_ids, valid_mask) if v]
                valid_texts = [t for t, v in zip(b_texts, valid_mask) if v]
                empty_ids   = [i for i, v in zip(b_ids, valid_mask) if not v]

                # Flag empty-text videos as permanently failed
                if empty_ids:
                    for vid in empty_ids:
                        logger.warning(f"SKIP (no text) | video_id={vid}")
                    try:
                        cur.execute("""
                            INSERT INTO yt.nlp_embeddings
                                (video_id, model_name, model_version, embed_failed_at)
                            VALUES (unnest(%s::text[]), %s, %s, NOW())
                            ON CONFLICT (video_id) DO UPDATE
                                SET embed_failed_at = NOW()
                        """, (empty_ids, MODEL_NAME, MODEL_VERSION))
                        conn.commit()
                    except Exception as db_err:
                        logger.error(f"DB error flagging empty videos: {db_err}")
                        conn.rollback()
                    videos_skipped += len(empty_ids)
                    pbar.update(len(empty_ids))

                if not valid_ids:
                    continue

                # ---- Generate embeddings ----
                try:
                    embeddings = model.encode(
                        valid_texts,
                        batch_size=ENCODE_BATCH,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True   # unit vectors → cosine = dot product
                    )
                except Exception as model_err:
                    # If the entire encode batch fails, flag all videos in it
                    logger.error(
                        f"Model error on batch {batch_idx + 1}: {model_err}"
                    )
                    for vid in valid_ids:
                        logger.debug(f"FAILED (model error) | video_id={vid}")
                    try:
                        cur.execute("""
                            INSERT INTO yt.nlp_embeddings
                                (video_id, model_name, model_version, embed_failed_at)
                            SELECT unnest(%s::text[]), %s, %s, NOW()
                            ON CONFLICT (video_id) DO UPDATE
                                SET embed_failed_at = NOW()
                        """, (valid_ids, MODEL_NAME, MODEL_VERSION))
                        conn.commit()
                    except Exception as db_err:
                        logger.error(f"DB error flagging model failures: {db_err}")
                        conn.rollback()
                    videos_failed += len(valid_ids)
                    pbar.update(len(valid_ids))
                    continue

                # ---- Write embeddings to DB ----
                insert_rows = []
                failed_ids  = []

                for vid, emb in zip(valid_ids, embeddings):
                    try:
                        insert_rows.append((vid, emb.tolist(), MODEL_NAME, MODEL_VERSION, now))
                    except Exception as parse_err:
                        logger.debug(f"FAILED (parse) | video_id={vid} | {parse_err}")
                        failed_ids.append(vid)

                # Bulk insert successes
                if insert_rows:
                    try:
                        execute_values(
                            cur,
                            """
                            INSERT INTO yt.nlp_embeddings
                                (video_id, embedding, model_name, model_version, created_at)
                            VALUES %s
                            ON CONFLICT (video_id) DO UPDATE SET
                                embedding     = EXCLUDED.embedding,
                                model_name    = EXCLUDED.model_name,
                                model_version = EXCLUDED.model_version,
                                created_at    = EXCLUDED.created_at,
                                embed_failed_at = NULL
                            """,
                            insert_rows,
                            template="(%s, %s::vector, %s, %s, %s)",
                            page_size=BATCH_SIZE
                        )
                        conn.commit()
                        videos_embedded += len(insert_rows)
                    except Exception as db_err:
                        logger.error(
                            f"DB insert error on batch {batch_idx + 1}: {db_err}"
                        )
                        conn.rollback()
                        # Re-flag all as failed so we don't silently lose them
                        failed_ids.extend([r[0] for r in insert_rows])
                        videos_failed += len(insert_rows)

                # Flag individual parse/DB failures
                if failed_ids:
                    for vid in failed_ids:
                        logger.warning(f"FAILED | video_id={vid}")
                    try:
                        cur.execute("""
                            INSERT INTO yt.nlp_embeddings
                                (video_id, model_name, model_version, embed_failed_at)
                            SELECT unnest(%s::text[]), %s, %s, NOW()
                            ON CONFLICT (video_id) DO UPDATE
                                SET embed_failed_at = NOW()
                        """, (failed_ids, MODEL_NAME, MODEL_VERSION))
                        conn.commit()
                    except Exception as db_err:
                        logger.error(f"DB error flagging failed videos: {db_err}")
                        conn.rollback()
                    videos_failed += len(failed_ids)

                pbar.update(len(b_ids))

        # ----------------------------------------------------------------
        # STEP 4: Log the run
        # ----------------------------------------------------------------
        duration = time.time() - start_time
        _log_run(
            cur, conn,
            videos_targeted=videos_targeted,
            videos_embedded=videos_embedded,
            videos_failed=videos_failed,
            videos_skipped=videos_skipped,
            duration=duration
        )

        # ----------------------------------------------------------------
        # STEP 5: Summary report
        # ----------------------------------------------------------------
        logger.info("")
        logger.info("=" * 60)
        logger.info("  EMBEDDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Duration:            {duration:.1f}s  ({duration/60:.1f} min)")
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  Videos targeted:     {videos_targeted:>8,}")
        logger.info(f"  Videos embedded:     {videos_embedded:>8,}")
        logger.info(f"  Videos failed:       {videos_failed:>8,}")
        logger.info(f"  Videos skipped:      {videos_skipped:>8,}")
        logger.info(f"  Model:               {MODEL_NAME}")
        logger.info(f"  Device:              {device.upper()}")
        if videos_failed > 0 or videos_skipped > 0:
            logger.info(f"  ─────────────────────────────────────")
            logger.info(f"  See log file for details on failures.")
        logger.info("=" * 60)

    except Exception as e:
        conn.rollback()
        logger.error(f"\nFATAL ERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()
        logger.info("Database connection closed.")


# ============================================================================
# Helpers
# ============================================================================

def _log_run(cur, conn, *, videos_targeted: int, videos_embedded: int,
             videos_failed: int, videos_skipped: int, duration: float):
    """Insert a summary row into yt.embed_log."""
    try:
        cur.execute(
            """
            INSERT INTO yt.embed_log
                (model_name, model_version, videos_targeted, videos_embedded,
                 videos_failed, videos_skipped, duration_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (MODEL_NAME, MODEL_VERSION, videos_targeted, videos_embedded,
             videos_failed, videos_skipped, round(duration, 2))
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Warning: could not write to embed_log: {e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    run_embedding()
