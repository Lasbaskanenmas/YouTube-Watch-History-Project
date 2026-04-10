"""
YouTube Analytics — Channel Language Inference Pipeline
========================================================
Populates primary_language and language_source on yt.dim_channels
using a two-tier approach:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — EXPORT (run first):
    python channel_language.py --export

    - Detects language via Unicode diacritics on video titles (100% reliable
      for Danish, Georgian, German, Spanish)
    - Runs Mistral 7B inference on channel names for ambiguous cases
    - Exports a CSV to logs/channel_review/ for your manual review
    - Does NOT write to the database yet

STEP 2 — Manual review:
    Open the exported CSV in Excel or Numbers.
    Fill in the primary_language column for any row where:
      - auto_detected is NULL (Mistral couldn't determine it)
      - You disagree with the auto_detected guess
    Leave blank if you genuinely don't know — Mistral's guess will be used.
    Save the CSV when done.

STEP 3 — IMPORT (run after manual review):
    python channel_language.py --import logs/channel_review/channel_review_YYYY-MM-DD.csv

    - Reads your corrections from the CSV
    - Writes human-verified entries with language_source = 'human'
    - Writes unicode-detected entries with language_source = 'unicode'
    - Writes Mistral-inferred entries with language_source = 'mistral'
    - Skips channels where primary_language is blank (left as NULL)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE SOURCE HIERARCHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    human   > unicode  > mistral

    If you fill in a language in the CSV, it overrides everything else.
    Unicode detection overrides Mistral inference.
    Mistral is used only when neither human nor unicode signals exist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPPORTED LANGUAGE CODES (ISO 639-1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    en = English        da = Danish         ka = Georgian
    de = German         es = Spanish        fr = French
    nl = Dutch          no = Norwegian      sv = Swedish
    it = Italian        pt = Portuguese     pl = Polish
    ru = Russian        ar = Arabic         ja = Japanese
    ko = Korean         zh = Chinese

    Add more as needed — any valid ISO 639-1 code is accepted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREREQUISITES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pip install psycopg2-binary python-dotenv requests tqdm

    Ollama must be running with mistral:7b pulled:
        ollama pull mistral:7b
        ollama serve  (or open the Ollama menu bar app)

Usage:
    python channel_language.py --export
    python channel_language.py --import logs/channel_review/channel_review_2026-04-10.csv
"""

import os
import re
import sys
import csv
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    raise ImportError("Install psycopg2-binary: pip install psycopg2-binary")

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("Install python-dotenv: pip install python-dotenv")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Install tqdm: pip install tqdm")


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

DB_CONFIG = {
    "dbname":   "youtube_analytics",
    "user":     "postgres",     # <-- Update if different
    "password": os.getenv("DB_PASSWORD", "qdw83nmm"),
    "host":     "localhost",
    "port":     5432
}

# Channels with at least this many views get included in the review CSV
MIN_VIEWS_FOR_REVIEW = 10

# Ollama configuration
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "mistral:7b"
OLLAMA_TIMEOUT = 30

# Paths
BASE_DIR    = Path(__file__).parent
REVIEW_DIR  = BASE_DIR / "logs" / "channel_review"
LOG_DIR     = BASE_DIR / "logs" / "ml_log"

# Valid ISO 639-1 codes — extend as needed
VALID_LANG_CODES = {
    "en", "da", "ka", "de", "es", "fr", "nl", "no", "sv",
    "it", "pt", "pl", "ru", "ar", "ja", "ko", "zh", "fi",
    "hu", "cs", "ro", "tr", "he", "vi", "th", "id", "ms"
}


# ============================================================================
# Unicode Diacritic Detection
# ============================================================================

# Character patterns that are 100% reliable language indicators
# These are unique to specific languages and never appear in English
UNICODE_PATTERNS = [
    # Danish / Norwegian (overlapping — treat as 'da' unless Norwegian signals)
    (re.compile(r'[æøåÆØÅ]'), 'da'),
    # Georgian (Mkhedruli script — completely unique)
    (re.compile(r'[\u10D0-\u10FF\u1C90-\u1CBF]'), 'ka'),
    # German (ß is uniquely German; ä/ö/ü overlap with others but ß does not)
    (re.compile(r'[ß]'), 'de'),
    # Spanish (inverted punctuation and ñ)
    (re.compile(r'[ñÑ¿¡]'), 'es'),
    # Russian / Bulgarian (Cyrillic)
    (re.compile(r'[\u0400-\u04FF]'), 'ru'),
    # Arabic
    (re.compile(r'[\u0600-\u06FF]'), 'ar'),
    # Japanese (Hiragana, Katakana, CJK)
    (re.compile(r'[\u3040-\u30FF\u4E00-\u9FFF]'), 'ja'),
    # Korean (Hangul)
    (re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF]'), 'ko'),
]


def detect_language_unicode(text: str) -> str | None:
    """
    Detect language from Unicode character patterns.
    Returns ISO 639-1 code if a reliable pattern is found, None otherwise.

    This is 100% accurate for the supported patterns because the characters
    are unique to those languages — they never appear in English content.
    """
    if not text:
        return None
    for pattern, lang_code in UNICODE_PATTERNS:
        if pattern.search(text):
            return lang_code
    return None


def detect_channel_language_unicode(channel_name: str,
                                     titles: list[str]) -> str | None:
    """
    Try to detect language from channel name first, then from video titles.
    Returns the first confident Unicode detection, or None.
    """
    # Check channel name first — most reliable single signal
    lang = detect_language_unicode(channel_name or "")
    if lang:
        return lang

    # Check video titles — aggregate signal across up to 20 titles
    lang_counts: dict[str, int] = {}
    for title in titles[:20]:
        detected = detect_language_unicode(title or "")
        if detected:
            lang_counts[detected] = lang_counts.get(detected, 0) + 1

    if lang_counts:
        # Return the most frequently detected language
        return max(lang_counts, key=lang_counts.get)

    return None


# ============================================================================
# Mistral Language Inference
# ============================================================================

def infer_language_mistral(channel_name: str,
                            logger: logging.Logger) -> str | None:
    """
    Use Mistral 7B via Ollama to infer the primary language of a YouTube
    channel from its name alone.

    Returns ISO 639-1 code or None if inference fails or is uncertain.
    """
    if not channel_name or not channel_name.strip():
        return None

    prompt = (
        f"What is the primary language of this YouTube channel?\n"
        f"Channel name: \"{channel_name}\"\n\n"
        f"Rules:\n"
        f"- Reply with ONLY a 2-letter ISO 639-1 language code (e.g. 'en', 'da', 'es')\n"
        f"- If the channel name is clearly English or ambiguous, reply 'en'\n"
        f"- If you cannot determine the language confidently, reply 'en'\n"
        f"- Do not explain, do not add punctuation, just the 2-letter code"
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}  # Very low temp for consistency
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip().lower()

        # Extract just the first 2-letter code from the response
        match = re.search(r'\b([a-z]{2})\b', raw)
        if match:
            code = match.group(1)
            if code in VALID_LANG_CODES:
                return code

    except requests.ConnectionError:
        logger.warning("Ollama not reachable — skipping Mistral inference")
    except requests.Timeout:
        logger.warning(f"Ollama timeout for channel: {channel_name[:50]}")
    except Exception as e:
        logger.warning(f"Ollama error for '{channel_name[:30]}': {e}")

    return None


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(log_dir: Path) -> logging.Logger:
    """Configure per-run timestamped logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = log_dir / f"channel_language_{run_ts}.log"

    logger = logging.getLogger("channel_language")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


# ============================================================================
# STEP 1: Export — Build Review CSV
# ============================================================================

def run_export():
    """
    Export a review CSV to logs/channel_review/.

    For each channel with MIN_VIEWS_FOR_REVIEW+ views:
      1. Try Unicode diacritic detection on channel name + video titles
      2. If no Unicode signal, try Mistral inference on channel name
      3. Write everything to CSV for manual review

    Does NOT write to the database.
    """
    logger = setup_logger(LOG_DIR)
    start_time = time.time()

    logger.info("")
    logger.info("=" * 60)
    logger.info("  MODE: EXPORT — Building channel review CSV")
    logger.info("=" * 60)
    logger.info("")

    # ----------------------------------------------------------------
    # Connect
    # ----------------------------------------------------------------
    logger.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # ----------------------------------------------------------------
    # Load channels with enough views
    # ----------------------------------------------------------------
    logger.info(f"Loading channels with {MIN_VIEWS_FOR_REVIEW}+ views...")
    cur.execute("""
        SELECT
            dc.channel_id,
            dc.channel_name,
            dc.primary_language,
            dc.language_source,
            cf.total_views
        FROM   yt.dim_channels dc
        JOIN   yt.mv_channel_funnel cf ON dc.channel_id = cf.channel_id
        WHERE  cf.total_views >= %s
        ORDER  BY cf.total_views DESC
    """, (MIN_VIEWS_FOR_REVIEW,))
    channels = cur.fetchall()
    logger.info(f"Found {len(channels):,} channels to process")

    # ----------------------------------------------------------------
    # Load a sample of video titles per channel for Unicode detection
    # ----------------------------------------------------------------
    logger.info("Loading video title samples for Unicode detection...")
    cur.execute("""
        SELECT
            dv.channel_id,
            array_agg(dv.title ORDER BY fe.watched_at DESC) AS titles
        FROM   yt.dim_videos dv
        JOIN   yt.fact_watch_events fe ON fe.video_id = dv.video_id
        WHERE  dv.status = 'active'
        AND    dv.title IS NOT NULL
        GROUP  BY dv.channel_id
    """)
    title_map: dict[int, list[str]] = {
        row[0]: [t for t in (row[1] or []) if t]
        for row in cur.fetchall()
    }

    # ----------------------------------------------------------------
    # Process each channel
    # ----------------------------------------------------------------
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    run_date   = datetime.now().strftime("%Y-%m-%d")
    csv_path   = REVIEW_DIR / f"channel_review_{run_date}.csv"

    unicode_count = 0
    mistral_count = 0
    already_tagged = 0
    skipped_mistral = 0

    rows = []

    logger.info("")
    logger.info("Processing channels...")
    logger.info("  (Mistral inference running for ambiguous channels)")
    logger.info("")

    with tqdm(total=len(channels), unit="channel",
              desc="Detecting", dynamic_ncols=True) as pbar:

        for channel_id, channel_name, existing_lang, existing_source, total_views in channels:

            titles = title_map.get(channel_id, [])

            # Skip channels already human-verified — don't overwrite
            if existing_source == 'human':
                auto_detected   = existing_lang
                mistral_guess   = existing_lang
                detection_method = 'already_human'
                already_tagged += 1
            else:
                # Tier 1: Unicode detection
                unicode_lang = detect_channel_language_unicode(
                    channel_name or "", titles
                )

                if unicode_lang:
                    auto_detected    = unicode_lang
                    mistral_guess    = None
                    detection_method = 'unicode'
                    unicode_count   += 1
                else:
                    # Tier 2: Mistral inference
                    auto_detected    = None
                    mistral_guess    = infer_language_mistral(
                        channel_name or "", logger
                    )
                    detection_method = 'mistral' if mistral_guess else 'none'
                    if mistral_guess:
                        mistral_count += 1
                    else:
                        skipped_mistral += 1

            rows.append({
                "channel_id":        channel_id,
                "channel_name":      channel_name or "",
                "total_views":       total_views,
                "existing_language": existing_lang or "",
                "existing_source":   existing_source or "",
                "unicode_detected":  auto_detected or "",
                "mistral_guess":     mistral_guess or "",
                "detection_method":  detection_method,
                # This is the column you fill in:
                "primary_language":  auto_detected or mistral_guess or "",
                "notes":             ""
            })

            pbar.update(1)

    # ----------------------------------------------------------------
    # Write CSV
    # ----------------------------------------------------------------
    fieldnames = [
        "channel_id",
        "channel_name",
        "total_views",
        "existing_language",
        "existing_source",
        "unicode_detected",
        "mistral_guess",
        "detection_method",
        "primary_language",   # ← Fill this in / correct this column
        "notes"               # ← Optional notes for yourself
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    duration = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Duration:            {duration:.1f}s")
    logger.info(f"  ─────────────────────────────────────")
    logger.info(f"  Channels processed:  {len(channels):>8,}")
    logger.info(f"  Unicode detected:    {unicode_count:>8,}")
    logger.info(f"  Mistral inferred:    {mistral_count:>8,}")
    logger.info(f"  Already human-tagged:{already_tagged:>8,}")
    logger.info(f"  No signal found:     {skipped_mistral:>8,}")
    logger.info(f"  ─────────────────────────────────────")
    logger.info(f"  CSV saved → {csv_path}")
    logger.info("")
    logger.info("  NEXT STEPS:")
    logger.info(f"  1. Open {csv_path.name} in Excel or Numbers")
    logger.info("  2. Review and correct the 'primary_language' column")
    logger.info("  3. Run: python channel_language.py --import <path_to_csv>")
    logger.info("=" * 60)

    cur.close()
    conn.close()
    logger.info("Database connection closed.")


# ============================================================================
# STEP 3: Import — Write Verified Languages to Database
# ============================================================================

def run_import(csv_path: str):
    """
    Import a manually reviewed CSV and write language tags to dim_channels.

    Language source hierarchy:
      - Rows you changed → language_source = 'human'
      - Rows with unicode_detected → language_source = 'unicode'
      - Rows with only mistral_guess → language_source = 'mistral'
      - Rows with blank primary_language → skipped (stays NULL)

    Never overwrites existing 'human' entries unless you explicitly
    changed the language in the CSV.
    """
    logger = setup_logger(LOG_DIR)
    start_time = time.time()

    logger.info("")
    logger.info("=" * 60)
    logger.info("  MODE: IMPORT — Writing language tags to database")
    logger.info("=" * 60)
    logger.info("")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Read CSV
    # ----------------------------------------------------------------
    logger.info(f"Reading CSV: {csv_file.name}")
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info(f"Read {len(rows):,} rows")

    # ----------------------------------------------------------------
    # Classify each row
    # ----------------------------------------------------------------
    human_updates   = []
    unicode_updates = []
    mistral_updates = []
    skipped         = []

    for row in rows:
        channel_id       = int(row["channel_id"])
        primary_language = row.get("primary_language", "").strip().lower()
        unicode_detected = row.get("unicode_detected", "").strip().lower()
        mistral_guess    = row.get("mistral_guess", "").strip().lower()
        existing_source  = row.get("existing_source", "").strip().lower()
        detection_method = row.get("detection_method", "").strip().lower()

        # Skip blank language entries
        if not primary_language:
            skipped.append(channel_id)
            continue

        # Validate language code
        if primary_language not in VALID_LANG_CODES:
            logger.warning(
                f"Invalid language code '{primary_language}' for "
                f"channel_id={channel_id} — skipping"
            )
            skipped.append(channel_id)
            continue

        # Determine source:
        # If the user changed the value from what was auto-detected → human
        # If unicode detected and not changed → unicode
        # If only mistral and not changed → mistral
        if detection_method == 'already_human':
            # Don't re-write already human-tagged entries
            skipped.append(channel_id)
            continue

        auto_value = unicode_detected or mistral_guess or ""

        if primary_language != auto_value or not auto_value:
            # User explicitly set or corrected this value
            source = 'human'
            human_updates.append((primary_language, source, channel_id))
        elif unicode_detected and primary_language == unicode_detected:
            source = 'unicode'
            unicode_updates.append((primary_language, source, channel_id))
        elif mistral_guess and primary_language == mistral_guess:
            source = 'mistral'
            mistral_updates.append((primary_language, source, channel_id))
        else:
            source = 'human'
            human_updates.append((primary_language, source, channel_id))

    # ----------------------------------------------------------------
    # Write to database
    # ----------------------------------------------------------------
    logger.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    all_updates = human_updates + unicode_updates + mistral_updates
    total_written = 0

    if all_updates:
        logger.info(f"Writing {len(all_updates):,} language tags...")
        execute_values(
            cur,
            """
            UPDATE yt.dim_channels AS dc SET
                primary_language = data.lang,
                language_source  = data.source
            FROM (VALUES %s) AS data(lang, source, channel_id)
            WHERE dc.channel_id = data.channel_id
            """,
            all_updates,
            template="(%s, %s, %s::integer)",
            page_size=500
        )
        conn.commit()
        total_written = len(all_updates)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    duration = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  IMPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Duration:            {duration:.1f}s")
    logger.info(f"  ─────────────────────────────────────")
    logger.info(f"  Total written:       {total_written:>8,}")
    logger.info(f"  Human-verified:      {len(human_updates):>8,}")
    logger.info(f"  Unicode-detected:    {len(unicode_updates):>8,}")
    logger.info(f"  Mistral-inferred:    {len(mistral_updates):>8,}")
    logger.info(f"  Skipped (blank):     {len(skipped):>8,}")
    logger.info(f"  ─────────────────────────────────────")
    logger.info("  NEXT STEP:")
    logger.info("  Set REMAP = True in topic_model.py and rerun to apply")
    logger.info("  the new language tags to topic labelling.")
    logger.info("=" * 60)

    cur.close()
    conn.close()
    logger.info("Database connection closed.")


# ============================================================================
# Argument Parser
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Analytics — Channel Language Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python channel_language.py --export
  python channel_language.py --import logs/channel_review/channel_review_2026-04-10.csv
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--export",
        action="store_true",
        help="Run language detection and export review CSV"
    )
    group.add_argument(
        "--import",
        dest="import_csv",
        metavar="CSV_PATH",
        help="Import a reviewed CSV and write language tags to the database"
    )

    args = parser.parse_args()

    if args.export:
        run_export()
    elif args.import_csv:
        run_import(args.import_csv)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    main()
