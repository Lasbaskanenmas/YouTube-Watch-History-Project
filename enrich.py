"""
YouTube Analytics — Tier 2 Enrichment
======================================
Fetches video metadata from the YouTube Data API v3 and populates
description, tags, category_id, duration_seconds, view_count, and
like_count on yt.dim_videos.

Rules:
    - Only processes videos with status = 'active' AND enriched_at IS NULL.
    - Respects the 10,000 unit/day quota (each batch of 50 costs 1 unit).
    - Videos the API no longer returns are flipped to status = 'unavailable'.
    - like_count is stored as NULL if the creator has hidden it.
    - All runs are logged to yt.enrichment_log.

Prerequisites:
    pip install psycopg2-binary python-dotenv requests

Usage:
    python enrich_videos.py

    The script reads YOUTUBE_API_KEY from a .env file in the same directory.
"""

import os
import re
import sys
import time
import requests
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


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

DB_CONFIG = {
    "dbname":   "youtube_analytics",
    "user":     "postgres",     # <-- Update if different
    "password": "qdw83nmm",     # <-- Update with your password
    "host":     "localhost",
    "port":     5432
}

QUOTA_LIMIT_PER_DAY = 10_000   # YouTube's free daily quota
BATCH_SIZE          = 50       # Max video IDs per API request
COST_PER_BATCH      = 1        # videos.list costs 1 unit per call
MAX_BATCHES         = QUOTA_LIMIT_PER_DAY // COST_PER_BATCH   # 10,000

YT_API_URL = "https://www.googleapis.com/youtube/v3/videos"


# ============================================================================
# Enrichment Log Table (auto-created if missing)
# ============================================================================

CREATE_ENRICHMENT_LOG = """
CREATE TABLE IF NOT EXISTS yt.enrichment_log (
    log_id              SERIAL          PRIMARY KEY,
    run_at              TIMESTAMPTZ     DEFAULT NOW(),
    videos_targeted     INTEGER         NOT NULL,
    videos_enriched     INTEGER         NOT NULL DEFAULT 0,
    videos_flipped      INTEGER         NOT NULL DEFAULT 0,
    api_units_used      INTEGER         NOT NULL DEFAULT 0,
    duration_seconds    FLOAT
);

COMMENT ON TABLE yt.enrichment_log IS
    'Audit log for Tier 2 enrichment runs. Tracks how many videos were '
    'enriched, flipped to unavailable, and how many API quota units were used.';
"""


# ============================================================================
# Parsing Helpers
# ============================================================================

def parse_iso_duration(iso_duration: str) -> int | None:
    """
    Convert an ISO 8601 duration string to total seconds.

    Examples:
        'PT1H2M3S' -> 3723
        'PT30S'    -> 30
        'PT0S'     -> 0
        'P1DT2H'   -> 93600
        None       -> None
    """
    if not iso_duration:
        return None

    pattern = re.compile(
        r'P(?:(?P<days>\d+)D)?'
        r'(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?'
    )
    match = pattern.fullmatch(iso_duration)
    if not match:
        return None

    days    = int(match.group('days')    or 0)
    hours   = int(match.group('hours')   or 0)
    minutes = int(match.group('minutes') or 0)
    seconds = int(match.group('seconds') or 0)

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def fetch_video_batch(video_ids: list[str], api_key: str) -> dict:
    """
    Fetch metadata for up to 50 video IDs from the YouTube Data API v3.

    Returns a dict mapping video_id -> parsed metadata dict.
    Videos not present in the response have been removed/privated.

    Costs: 1 quota unit per call regardless of batch size.
    """
    params = {
        "id":   ",".join(video_ids),
        "part": "snippet,contentDetails,statistics",
        "key":  api_key,
    }

    response = requests.get(YT_API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = {}
    for item in data.get("items", []):
        vid          = item["id"]
        snippet      = item.get("snippet", {})
        content      = item.get("contentDetails", {})
        stats        = item.get("statistics", {})

        # like_count is NULL if the creator has hidden it
        like_count_raw = stats.get("likeCount")
        like_count     = int(like_count_raw) if like_count_raw is not None else None

        results[vid] = {
            "description":      snippet.get("description") or None,
            "tags":             snippet.get("tags") or [],
            "category_id":      int(snippet["categoryId"]) if snippet.get("categoryId") else None,
            "duration_seconds": parse_iso_duration(content.get("duration")),
            "view_count":       int(stats["viewCount"])  if stats.get("viewCount")  else None,
            "like_count":       like_count,
        }

    return results


# ============================================================================
# Core Enrichment Pipeline
# ============================================================================

def run_enrichment():
    """
    Execute the full Tier 2 enrichment pipeline:
    1. Validate API key
    2. Load unenriched active videos from the DB
    3. Batch-fetch metadata from YouTube Data API
    4. Update dim_videos with enriched fields
    5. Flip missing videos to 'unavailable'
    6. Log the run to yt.enrichment_log
    """
    start_time = time.time()

    # ----------------------------------------------------------------
    # Validate API key
    # ----------------------------------------------------------------
    if not YOUTUBE_API_KEY:
        print("ERROR: YOUTUBE_API_KEY not found.")
        print("  Make sure your .env file contains: YOUTUBE_API_KEY=your_key_here")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Connect to database
    # ----------------------------------------------------------------
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur  = conn.cursor()

    try:
        cur.execute("SET search_path TO yt, public;")

        # Ensure enrichment log table exists
        cur.execute(CREATE_ENRICHMENT_LOG)
        conn.commit()

        # ----------------------------------------------------------------
        # STEP 1: Load unenriched active videos
        # ----------------------------------------------------------------
        print("Loading unenriched active videos...")
        cur.execute("""
            SELECT video_id
            FROM   yt.dim_videos
            WHERE  status      = 'active'
            AND    enriched_at IS NULL
            ORDER  BY video_id
        """)
        all_video_ids = [row[0] for row in cur.fetchall()]
        videos_targeted = len(all_video_ids)

        if videos_targeted == 0:
            print("  No unenriched active videos found. Nothing to do.")
            duration = time.time() - start_time
            cur.execute("""
                INSERT INTO yt.enrichment_log
                    (videos_targeted, videos_enriched, videos_flipped, api_units_used, duration_seconds)
                VALUES (%s, 0, 0, 0, %s)
            """, (0, round(duration, 2)))
            conn.commit()
            return

        print(f"  Found {videos_targeted:,} unenriched active videos")

        # Cap to daily quota limit
        max_videos  = MAX_BATCHES * BATCH_SIZE
        video_ids   = all_video_ids[:max_videos]
        if len(all_video_ids) > max_videos:
            print(f"  Quota cap: processing {max_videos:,} of {videos_targeted:,} videos today")
        else:
            print(f"  All {videos_targeted:,} videos fit within today's quota — processing all")

        # ----------------------------------------------------------------
        # STEP 2: Batch-fetch from YouTube API and update DB
        # ----------------------------------------------------------------
        batches           = [video_ids[i:i + BATCH_SIZE] for i in range(0, len(video_ids), BATCH_SIZE)]
        total_batches     = len(batches)
        api_units_used    = 0
        videos_enriched   = 0
        videos_flipped    = 0

        print(f"\nFetching metadata in {total_batches:,} batches of up to {BATCH_SIZE} videos...")
        print(f"  Quota cost: {total_batches:,} of {QUOTA_LIMIT_PER_DAY:,} units")
        print()

        for batch_num, batch in enumerate(batches, start=1):
            batch_set = set(batch)

            # Progress indicator every 100 batches
            if batch_num == 1 or batch_num % 100 == 0 or batch_num == total_batches:
                print(f"  Batch {batch_num:>5,} / {total_batches:,}  "
                      f"({api_units_used + 1} units used so far)")

            # ---- API call ----
            try:
                results = fetch_video_batch(batch, YOUTUBE_API_KEY)
                api_units_used += 1
            except requests.HTTPError as e:
                print(f"\n  API error on batch {batch_num}: {e}")
                print("  Stopping enrichment to avoid further errors.")
                break
            except requests.RequestException as e:
                print(f"\n  Network error on batch {batch_num}: {e}")
                print("  Stopping enrichment to avoid further errors.")
                break

            # ---- Update enriched videos ----
            enriched_ids   = set(results.keys())
            unavailable_ids = batch_set - enriched_ids  # Videos API didn't return

            if enriched_ids:
                update_data = []
                now = datetime.now(timezone.utc)
                for vid, meta in results.items():
                    update_data.append((
                        meta["description"],
                        meta["tags"],
                        meta["category_id"],
                        meta["duration_seconds"],
                        meta["view_count"],
                        meta["like_count"],
                        now,
                        vid,
                    ))

                execute_values(
                    cur,
                    """
                    UPDATE yt.dim_videos AS d SET
                        description      = data.description,
                        tags             = data.tags,
                        category_id      = data.category_id,
                        duration_seconds = data.duration_seconds,
                        view_count       = data.view_count,
                        like_count       = data.like_count,
                        enriched_at      = data.enriched_at,
                        updated_at       = data.enriched_at
                    FROM (VALUES %s) AS data(
                        description, tags, category_id, duration_seconds,
                        view_count, like_count, enriched_at, video_id
                    )
                    WHERE d.video_id = data.video_id
                    """,
                    update_data,
                    template="(%s, %s::text[], %s, %s, %s, %s, %s, %s)",
                    page_size=BATCH_SIZE,
                )
                videos_enriched += len(enriched_ids)

            # ---- Flip unavailable videos ----
            if unavailable_ids:
                cur.execute(
                    """
                    UPDATE yt.dim_videos
                    SET    status     = 'unavailable',
                           updated_at = NOW()
                    WHERE  video_id   = ANY(%s)
                    """,
                    (list(unavailable_ids),)
                )
                videos_flipped += len(unavailable_ids)

            # Commit each batch so progress is saved even if script is interrupted
            conn.commit()

        # ----------------------------------------------------------------
        # STEP 3: Log the run
        # ----------------------------------------------------------------
        duration = time.time() - start_time
        cur.execute(
            """
            INSERT INTO yt.enrichment_log
                (videos_targeted, videos_enriched, videos_flipped, api_units_used, duration_seconds)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (videos_targeted, videos_enriched, videos_flipped, api_units_used, round(duration, 2))
        )
        conn.commit()

        # ----------------------------------------------------------------
        # STEP 4: Summary report
        # ----------------------------------------------------------------
        remaining = videos_targeted - videos_enriched - videos_flipped
        print("\n" + "=" * 60)
        print("  ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"  Duration:            {duration:.1f}s")
        print(f"  ─────────────────────────────────────")
        print(f"  Videos targeted:     {videos_targeted:>8,}")
        print(f"  Videos enriched:     {videos_enriched:>8,}")
        print(f"  Flipped unavailable: {videos_flipped:>8,}")
        print(f"  API units used:      {api_units_used:>8,} / {QUOTA_LIMIT_PER_DAY:,}")
        if remaining > 0:
            print(f"  ─────────────────────────────────────")
            print(f"  Still unenriched:    {remaining:>8,}  (run again tomorrow)")
        print("=" * 60)

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()
        print("Database connection closed.")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    run_enrichment()
