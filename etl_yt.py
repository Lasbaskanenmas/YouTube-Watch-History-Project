"""
YouTube Analytics — ETL Pipeline
=================================
Loads a Google Takeout YouTube watch history JSON export into the
youtube_analytics PostgreSQL database.

Handles:
    - Ad filtering (details containing "From Google Ads")
    - Removed video detection (no titleUrl)
    - Ghost video handling (titleUrl present but no channel/title info)
    - Video ID extraction from titleUrl
    - Title cleaning ("Watched " prefix removal)
    - Channel extraction from nested subtitles field
    - Deduplication via ON CONFLICT on (video_id, watched_at)
    - ETL logging to yt.etl_log

Prerequisites:
    pip install psycopg2-binary

Usage:
    python etl_yt.py /path/to/watch-history.json
"""

import os
import re
import sys
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    raise ImportError("Install psycopg2-binary: pip install psycopg2-binary")


# ============================================================================
# Configuration
# ============================================================================
DB_CONFIG = {
    "dbname":   "youtube_analytics",
    "user":     "postgres",           # <-- Update if different
    "password": "qdw83nmm", # <-- Update with your password
    "host":     "localhost",
    "port":     5432
}


# ============================================================================
# Parsing Helpers
# ============================================================================

def extract_video_id(title_url: str) -> str | None:
    """
    Extract the video ID from a YouTube URL.
    e.g., 'https://www.youtube.com/watch?v=hMEyBtsuAJE' -> 'hMEyBtsuAJE'
    """
    try:
        parsed = urlparse(title_url)
        video_id = parse_qs(parsed.query).get('v', [None])[0]
        return video_id
    except Exception:
        return None


def is_ad(record: dict) -> bool:
    """Check if a record is a Google Ad."""
    details = record.get('details', [])
    if isinstance(details, list):
        return any(
            isinstance(d, dict) and d.get('name') == 'From Google Ads'
            for d in details
        )
    return False


def is_removed_video(record: dict) -> bool:
    """Check if a record is a removed video with no recoverable data."""
    return 'titleUrl' not in record or not record.get('titleUrl')


def is_ghost_video(record: dict) -> bool:
    """
    Check if a record is a 'ghost' video — has a titleUrl but the title
    is just the URL itself (no real title) and no channel info.
    """
    title = record.get('title', '')
    cleaned = title.replace('Watched ', '', 1).strip()
    # If the cleaned title looks like a URL, it's a ghost
    return cleaned.startswith('http://') or cleaned.startswith('https://')


def extract_channel(record: dict) -> tuple[str | None, str | None]:
    """
    Extract channel name and URL from the nested subtitles field.
    Returns (channel_name, channel_url) or (None, None).
    """
    subtitles = record.get('subtitles', [])
    if isinstance(subtitles, list) and subtitles:
        first = subtitles[0]
        if isinstance(first, dict):
            return first.get('name'), first.get('url')
    return None, None


def clean_title(record: dict) -> str | None:
    """
    Clean the title field:
    - Strip 'Watched ' prefix
    - Return None for ghost videos (title is just a URL)
    """
    title = record.get('title', '')

    # Strip the 'Watched ' prefix
    if title.startswith('Watched '):
        title = title[8:]  # len('Watched ') == 8

    # If the remaining title is a URL, it's a ghost video -> NULL
    if title.startswith('http://') or title.startswith('https://'):
        return None

    return title.strip() if title.strip() else None


# ============================================================================
# ETL Log Table (auto-created if missing)
# ============================================================================

CREATE_ETL_LOG = """
CREATE TABLE IF NOT EXISTS yt.etl_log (
    log_id              SERIAL          PRIMARY KEY,
    file_name           TEXT            NOT NULL,
    run_at              TIMESTAMPTZ     DEFAULT NOW(),
    total_raw_records   INTEGER         NOT NULL,
    ads_filtered        INTEGER         NOT NULL,
    removed_dropped     INTEGER         NOT NULL,
    ghost_videos_kept   INTEGER         NOT NULL,
    clean_records       INTEGER         NOT NULL,
    new_channels        INTEGER         DEFAULT 0,
    new_videos          INTEGER         DEFAULT 0,
    new_events          INTEGER         DEFAULT 0,
    duplicate_events    INTEGER         DEFAULT 0,
    duration_seconds    FLOAT
);

COMMENT ON TABLE yt.etl_log IS 'Audit log for ETL ingest runs. Tracks record counts and filtering at each stage for traceability.';
"""


# ============================================================================
# Core ETL Pipeline
# ============================================================================

def run_etl(file_path: str):
    """
    Execute the full ETL pipeline:
    1. Load JSON
    2. Filter & classify records
    3. Extract & clean fields
    4. Upsert into dim_channels, dim_videos, fact_watch_events
    5. Log the run to yt.etl_log
    """
    start_time = time.time()
    file_name = os.path.basename(file_path)

    # ----------------------------------------------------------------
    # STEP 1: Load JSON
    # ----------------------------------------------------------------
    print(f"Loading {file_name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total_raw = len(raw_data)
    print(f"  Loaded {total_raw:,} raw records")

    # ----------------------------------------------------------------
    # STEP 2: Filter & classify records
    # ----------------------------------------------------------------
    print("Filtering records...")
    ads = []
    removed = []
    ghosts = []
    clean = []

    for record in raw_data:
        if is_ad(record):
            ads.append(record)
        elif is_removed_video(record):
            removed.append(record)
        elif is_ghost_video(record):
            ghosts.append(record)
        else:
            clean.append(record)

    # Ghost videos are kept — they go into the database with NULL title/channel
    records_to_load = clean + ghosts

    print(f"  Ads filtered:        {len(ads):>6}")
    print(f"  Removed dropped:     {len(removed):>6}")
    print(f"  Ghost videos kept:   {len(ghosts):>6}")
    print(f"  Clean records:       {len(clean):>6}")
    print(f"  Total to load:       {len(records_to_load):>6}")

    # ----------------------------------------------------------------
    # STEP 3: Extract & clean fields
    # ----------------------------------------------------------------
    print("Extracting and cleaning fields...")
    parsed_records = []

    for record in records_to_load:
        video_id = extract_video_id(record.get('titleUrl', ''))
        if not video_id:
            continue  # Safety net: skip if we somehow can't extract an ID

        title = clean_title(record)
        title_url = record.get('titleUrl')
        watched_at = record.get('time')
        channel_name, channel_url = extract_channel(record)

        parsed_records.append({
            'video_id':     video_id,
            'title':        title,
            'title_url':    title_url,
            'watched_at':   watched_at,
            'channel_name': channel_name,
            'channel_url':  channel_url,
        })

    print(f"  Parsed {len(parsed_records):,} records successfully")

    # ----------------------------------------------------------------
    # STEP 4: Connect and upsert
    # ----------------------------------------------------------------
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute("SET search_path TO yt, public;")

        # Ensure the ETL log table exists
        cur.execute(CREATE_ETL_LOG)

        # ---- 4a: Upsert channels ----
        print("Upserting channels...")
        channel_id_map = {}

        # Collect unique channels
        unique_channels = {}
        for r in parsed_records:
            if r['channel_name'] and r['channel_url']:
                key = (r['channel_name'], r['channel_url'])
                unique_channels[key] = True

        # Pre-load existing channels into the map
        cur.execute("SELECT channel_id, channel_name, channel_url FROM dim_channels")
        for row in cur.fetchall():
            channel_id_map[(row[1], row[2])] = row[0]

        new_channels = 0
        for (ch_name, ch_url) in unique_channels:
            if (ch_name, ch_url) not in channel_id_map:
                cur.execute(
                    """INSERT INTO dim_channels (channel_name, channel_url)
                       VALUES (%s, %s)
                       ON CONFLICT (channel_name, channel_url) DO NOTHING
                       RETURNING channel_id""",
                    (ch_name, ch_url)
                )
                result = cur.fetchone()
                if result:
                    channel_id_map[(ch_name, ch_url)] = result[0]
                    new_channels += 1
                else:
                    # Race condition fallback: fetch existing
                    cur.execute(
                        "SELECT channel_id FROM dim_channels WHERE channel_name = %s AND channel_url = %s",
                        (ch_name, ch_url)
                    )
                    row = cur.fetchone()
                    if row:
                        channel_id_map[(ch_name, ch_url)] = row[0]

        print(f"  Channels: {len(unique_channels)} unique, {new_channels} new")

        # ---- 4b: Upsert videos ----
        print("Upserting videos...")
        new_videos = 0

        # Collect unique videos (first occurrence wins for title/channel)
        unique_videos = {}
        for r in parsed_records:
            vid = r['video_id']
            if vid not in unique_videos:
                unique_videos[vid] = r

        # Pre-load existing video IDs
        cur.execute("SELECT video_id FROM dim_videos")
        existing_video_ids = {row[0] for row in cur.fetchall()}

        video_insert_data = []
        for vid, r in unique_videos.items():
            if vid not in existing_video_ids:
                ch_key = (r['channel_name'], r['channel_url'])
                ch_id = channel_id_map.get(ch_key)

                # Determine status
                status = 'active' if r['title'] else 'unavailable'

                video_insert_data.append((
                    vid,
                    r['title'],         # Could be None for ghost videos
                    r['title_url'],
                    ch_id,              # Could be None for ghost videos
                    status
                ))

        if video_insert_data:
            execute_values(
                cur,
                """INSERT INTO dim_videos (video_id, title, title_url, channel_id, status)
                   VALUES %s
                   ON CONFLICT (video_id) DO NOTHING""",
                video_insert_data,
                page_size=1000
            )
            new_videos = len(video_insert_data)

        print(f"  Videos: {len(unique_videos)} unique, {new_videos} new")

        # ---- 4c: Insert watch events ----
        print("Inserting watch events...")

        # Count existing events before insert
        cur.execute("SELECT COUNT(*) FROM fact_watch_events")
        events_before = cur.fetchone()[0]

        event_data = []
        for r in parsed_records:
            ch_key = (r['channel_name'], r['channel_url'])
            ch_id = channel_id_map.get(ch_key)
            event_data.append((r['video_id'], ch_id, r['watched_at']))

        execute_values(
            cur,
            """INSERT INTO fact_watch_events (video_id, channel_id, watched_at)
               VALUES %s
               ON CONFLICT (video_id, watched_at) DO NOTHING""",
            event_data,
            page_size=1000
        )

        # Count after to determine new vs duplicate
        cur.execute("SELECT COUNT(*) FROM fact_watch_events")
        events_after = cur.fetchone()[0]
        new_events = events_after - events_before
        duplicate_events = len(event_data) - new_events

        print(f"  Events: {len(event_data)} submitted, {new_events} new, {duplicate_events} duplicates skipped")

        # ---- 4d: Log the ETL run ----
        duration = time.time() - start_time
        cur.execute(
            """INSERT INTO yt.etl_log (
                   file_name, total_raw_records, ads_filtered, removed_dropped,
                   ghost_videos_kept, clean_records, new_channels, new_videos,
                   new_events, duplicate_events, duration_seconds
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (file_name, total_raw, len(ads), len(removed), len(ghosts),
             len(clean), new_channels, new_videos, new_events, duplicate_events, round(duration, 2))
        )

        # ---- Commit ----
        conn.commit()

        # ----------------------------------------------------------------
        # STEP 5: Summary report
        # ----------------------------------------------------------------
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print("  ETL COMPLETE")
        print("=" * 60)
        print(f"  File:                {file_name}")
        print(f"  Duration:            {duration:.1f}s")
        print(f"  ─────────────────────────────────────")
        print(f"  Raw records:         {total_raw:>8,}")
        print(f"  Ads filtered:        {len(ads):>8,}")
        print(f"  Removed dropped:     {len(removed):>8,}")
        print(f"  Ghost videos kept:   {len(ghosts):>8,}")
        print(f"  Clean records:       {len(clean):>8,}")
        print(f"  ─────────────────────────────────────")
        print(f"  New channels:        {new_channels:>8,}")
        print(f"  New videos:          {new_videos:>8,}")
        print(f"  New watch events:    {new_events:>8,}")
        print(f"  Duplicates skipped:  {duplicate_events:>8,}")
        print(f"  ─────────────────────────────────────")

        # Total watched (including removed) for the application
        total_watched_ever = events_after + len(removed)
        cur2 = conn.cursor()
        cur2.execute("SELECT COALESCE(SUM(removed_dropped), 0) FROM yt.etl_log")
        total_removed_all_time = cur2.fetchone()[0]
        cur2.execute("SELECT COUNT(*) FROM fact_watch_events")
        total_events_in_db = cur2.fetchone()[0]
        cur2.close()

        print(f"\n  DATABASE TOTALS:")
        print(f"  Watch events in DB:  {total_events_in_db:>8,}")
        print(f"  Removed (all time):  {total_removed_all_time:>8,}")
        print(f"  Total ever watched:  {total_events_in_db + total_removed_all_time:>8,}")
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
    if len(sys.argv) < 2:
        print("Usage: python etl_yt.py /path/to/watch-history.json")
        print("\nExample:")
        print("  python etl_yt.py ~/Desktop/watch-history.json")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if not file_path.endswith('.json'):
        print(f"Warning: File does not have .json extension: {file_path}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)

    run_etl(file_path)
