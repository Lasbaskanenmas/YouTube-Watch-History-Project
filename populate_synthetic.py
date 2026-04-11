"""
YouTube Analytics — Synthetic Data Generator
=============================================
Populates the youtube_analytics database with realistic test data.

Prerequisites:
    pip install psycopg2-binary faker numpy

Usage:
    1. Ensure PostgreSQL is running and the schema has been created via create_schema.sql.
    2. Update the DB_CONFIG below with your local PostgreSQL credentials.
    3. Run: python populate_synthetic.py
"""

import os
import random
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    raise ImportError("Install psycopg2-binary: pip install psycopg2-binary")

try:
    from faker import Faker
except ImportError:
    raise ImportError("Install faker: pip install faker")


# ============================================================================
# Configuration
# ============================================================================
DB_CONFIG = {
    "dbname":   "youtube_analytics",
    "user":     "postgres",           # <-- Update if different
    "password": os.getenv("DB_PASSWORD", ""), # <-- Update with your password
    "host":     "localhost",
    "port":     5432
}

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

# Volume controls
NUM_CHANNELS        = 150       # Unique channels to generate
NUM_VIDEOS          = 2000      # Unique videos to generate
NUM_WATCH_EVENTS    = 8000      # Total watch events (some videos watched multiple times)
NUM_TOPICS          = 8         # NLP topics to generate
HISTORY_START       = datetime(2019, 1, 1, tzinfo=timezone.utc)
HISTORY_END         = datetime(2024, 12, 31, tzinfo=timezone.utc)


# ============================================================================
# Realistic Content Templates
# ============================================================================

# Channel name templates grouped by content niche
CHANNEL_TEMPLATES = {
    "boxing_mma": [
        "Top Rank Boxing", "DAZN Boxing", "UFC", "ESPN MMA", "Boxing Ring TV",
        "FightHub TV", "MMA Digest", "Beyond Kickboxing", "Boxing Legends TV",
        "The MMA Hour"
    ],
    "tech": [
        "Code With Arjun", "Fireship", "Traversy Media", "TechLead", "NetworkChuck",
        "Programming with Mosh", "The Coding Train", "CS Dojo", "Tech With Tim",
        "Joma Tech"
    ],
    "music": [
        "NPR Music", "Colors Studios", "KEXP", "Sofar Sounds", "Tiny Desk Concerts",
        "On The Radar Radio", "SBTV", "GRM Daily", "WORLDSTARHIPHOP", "Lyrical Lemonade"
    ],
    "education": [
        "3Blue1Brown", "Veritasium", "Kurzgesagt", "CrashCourse", "Khan Academy",
        "StatQuest", "MIT OpenCourseWare", "Numberphile", "TED-Ed", "Wendover Productions"
    ],
    "gaming": [
        "PewDiePie", "Markiplier", "jacksepticeye", "Dream", "MrBeast Gaming",
        "Valkyrae", "Sykkuno", "Pokimane", "Ludwig", "Disguised Toast"
    ],
    "cooking": [
        "Binging with Babish", "Joshua Weissman", "Adam Ragusea", "Pro Home Cooks",
        "Ethan Chlebowski", "Internet Shaquille", "Sorted Food", "Bon Appétit",
        "Nick DiGiovanni", "Sam The Cooking Guy"
    ],
    "podcasts": [
        "TMG Podcast Highlights", "JRE Clips", "Lex Fridman Podcast", "Diary of a CEO",
        "Flagrant", "Andrew Schulz", "H3 Podcast Highlights", "Theo Von",
        "Kill Tony", "PBD Podcast"
    ],
    "sports_other": [
        "Premier League", "NBA Highlights", "NFL", "Sky Sports Football",
        "F1 Official", "GolfTV", "Tennis TV", "BeIN Sports", "Olympic Channel",
        "B/R Football"
    ]
}

# Video title templates by niche
TITLE_TEMPLATES = {
    "boxing_mma": [
        "{fighter1} vs {fighter2} — Full Fight Highlights",
        "KNOCKOUT! {fighter1} Destroys {fighter2} in Round {round}",
        "{fighter1} Training Camp — Road to the Title",
        "Top 10 Knockouts of {year}",
        "Post-Fight Interview: {fighter1} on Beating {fighter2}",
        "{fighter1} vs {fighter2} — Official Weigh-In",
        "Boxing Breakdown: Why {fighter1} Will Beat {fighter2}",
        "UFC {num}: Best Moments from Fight Night"
    ],
    "tech": [
        "Git on MacOS (MacBook M1) — Full Setup Guide",
        "{lang} Tutorial for Beginners — {year} Edition",
        "I Built a {thing} in {hours} Hours",
        "Why {lang} is Taking Over in {year}",
        "{num} VS Code Extensions You NEED",
        "How I Became a Software Engineer Without a Degree",
        "Docker Explained in {minutes} Minutes",
        "System Design Interview: {thing}"
    ],
    "music": [
        "{artist} — {song} (Official Video)",
        "{artist} — Live at {venue}",
        "{artist} — {song} (Lyric Video)",
        "Best of {genre} — {year} Mix",
        "{artist} ft. {feature} — {song}",
        "{artist} — {album} Full Album",
        "Top {num} {genre} Songs of {year}",
        "{artist} — {song} (Acoustic Session)"
    ],
    "education": [
        "But What IS a Neural Network? — Chapter {num}",
        "The Essence of Linear Algebra — {topic}",
        "How {topic} Actually Works",
        "Why You Don't Understand {topic} (Yet)",
        "{topic} Explained in {minutes} Minutes",
        "The Map of {topic}",
        "How to Learn {topic} — A Roadmap",
        "What is {topic}? — Explained Simply"
    ],
    "gaming": [
        "{game} — Full Playthrough Part {num}",
        "I Played {game} for {hours} Hours Straight",
        "{game} — Best Moments Compilation",
        "Reacting to {game} Speedrun World Record",
        "{game} Tips and Tricks for Beginners",
        "This New {game} Update Changes EVERYTHING",
        "{game} — Ranked Grind to {rank}",
        "My {game} Settings for Maximum FPS"
    ],
    "cooking": [
        "The Perfect {dish} — Restaurant Quality at Home",
        "I Made {dish} Using Only {constraint}",
        "{num} Easy {meal} Recipes Under {minutes} Minutes",
        "Why Your {dish} Sucks (and How to Fix It)",
        "The Science Behind the Perfect {dish}",
        "Meal Prep — {num} Lunches for the Week",
        "I Tried Making {chef}'s Famous {dish}",
        "Street Food Tour: Best {dish} in {city}"
    ],
    "podcasts": [
        "{guest} on {topic} — Full Episode",
        "{guest}: '{quote}' — Clip",
        "The Truth About {topic} — {podcast}",
        "{guest} Tells His Craziest Story",
        "{podcast} — Best Moments #{num}",
        "{guest} on Why {topic} Matters",
        "Funniest Moments — {podcast} #{num}",
        "{guest} vs {guest2} Debate on {topic}"
    ],
    "sports_other": [
        "{team1} vs {team2} — Match Highlights",
        "Top {num} Goals of the {year} Season",
        "{player} — Best Moments {year}",
        "{team1} {score1}-{score2} {team2} — All Goals & Highlights",
        "Why {team1} Will Win the {trophy}",
        "{player} Interview After Historic Win",
        "Best Saves of the Week — Matchday {num}",
        "{league} Season {year} — Preview"
    ]
}

# Filler values for template substitution
FIGHTERS = ["Canelo", "Tyson Fury", "Usyk", "Crawford", "Spence", "Inoue", "Tank Davis",
            "Haney", "Lomachenko", "Beterbiev", "Bivol", "Joshua"]
ARTISTS = ["Kendrick Lamar", "Tyler the Creator", "Frank Ocean", "SZA", "Drake",
           "Bad Bunny", "Rosalía", "Dua Lipa", "The Weeknd", "Mac Miller"]
SONGS = ["Humble", "Earfquake", "Nights", "Kiss Me More", "Blinding Lights",
         "Dakiti", "Malamente", "Levitating", "Save Your Tears", "Self Care"]
GENRES = ["Hip-Hop", "R&B", "Jazz", "Lo-fi", "Indie", "Electronic", "Soul", "Afrobeats"]
LANGS = ["Python", "JavaScript", "TypeScript", "Rust", "Go", "SQL", "C++", "Swift"]
TOPICS_EDU = ["Calculus", "Machine Learning", "Quantum Mechanics", "Statistics",
              "Blockchain", "Bayesian Inference", "Graph Theory", "Entropy"]
GAMES = ["Elden Ring", "Baldur's Gate 3", "Minecraft", "Valorant", "GTA V",
         "Zelda", "FIFA 24", "Cyberpunk 2077", "Fortnite", "Counter-Strike 2"]
DISHES = ["Pasta Carbonara", "Ramen", "Sourdough Bread", "Steak", "Pizza",
          "Fried Rice", "Tacos", "Croissants", "Burger", "Pad Thai"]
TEAMS = ["Arsenal", "Man City", "Real Madrid", "Barcelona", "Bayern Munich",
         "Liverpool", "PSG", "Inter Milan", "Dortmund", "Atletico Madrid"]
PLAYERS = ["Haaland", "Mbappé", "Vinicius Jr", "Saka", "Bellingham",
           "Salah", "De Bruyne", "Messi", "Palmer", "Wirtz"]
GUESTS = ["Elon Musk", "David Goggins", "Andrew Huberman", "Naval Ravikant",
          "Alex Hormozi", "Gary Vee", "Jordan Peterson", "Sam Harris"]
PODCASTS = ["JRE", "Lex Fridman", "Diary of a CEO", "Flagrant", "PBD Podcast"]


def fill_template(template: str, niche: str) -> str:
    """Fill a title template with random contextual values."""
    replacements = {
        "{fighter1}": random.choice(FIGHTERS),
        "{fighter2}": random.choice(FIGHTERS),
        "{round}": str(random.randint(1, 12)),
        "{year}": str(random.randint(2019, 2024)),
        "{num}": str(random.randint(1, 50)),
        "{lang}": random.choice(LANGS),
        "{thing}": random.choice(["SaaS App", "Chat Bot", "API", "Game", "Portfolio Site"]),
        "{hours}": str(random.choice([12, 24, 48, 72, 100])),
        "{minutes}": str(random.choice([5, 10, 15, 20])),
        "{artist}": random.choice(ARTISTS),
        "{feature}": random.choice(ARTISTS),
        "{song}": random.choice(SONGS),
        "{genre}": random.choice(GENRES),
        "{album}": fake.catch_phrase(),
        "{venue}": random.choice(["Glastonbury", "Coachella", "NPR Tiny Desk", "BBC Radio 1"]),
        "{topic}": random.choice(TOPICS_EDU),
        "{game}": random.choice(GAMES),
        "{rank}": random.choice(["Diamond", "Immortal", "Champion", "Global Elite"]),
        "{dish}": random.choice(DISHES),
        "{meal}": random.choice(["Breakfast", "Lunch", "Dinner", "Snack"]),
        "{chef}": random.choice(["Gordon Ramsay", "Jamie Oliver", "Julia Child"]),
        "{city}": random.choice(["Tokyo", "Bangkok", "Mexico City", "Istanbul"]),
        "{constraint}": random.choice(["$5", "Gas Station ingredients", "one pan"]),
        "{team1}": random.choice(TEAMS),
        "{team2}": random.choice(TEAMS),
        "{player}": random.choice(PLAYERS),
        "{score1}": str(random.randint(0, 5)),
        "{score2}": str(random.randint(0, 4)),
        "{trophy}": random.choice(["Champions League", "Premier League", "World Cup"]),
        "{league}": random.choice(["Premier League", "La Liga", "Serie A", "Bundesliga"]),
        "{guest}": random.choice(GUESTS),
        "{guest2}": random.choice(GUESTS),
        "{quote}": random.choice(["Stay hard", "That changed my life", "I was broke"]),
        "{podcast}": random.choice(PODCASTS),
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def generate_video_id() -> str:
    """Generate a realistic-looking YouTube video ID (11 chars, base64-like)."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return ''.join(random.choice(chars) for _ in range(11))


def generate_youtube_url(video_id: str) -> str:
    """Generate a full YouTube URL from a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"


def generate_channel_url(channel_name: str) -> str:
    """Generate a realistic YouTube channel URL."""
    hash_part = hashlib.md5(channel_name.encode()).hexdigest()[:24]
    return f"https://www.youtube.com/channel/UC{hash_part}"


def generate_watch_timestamp() -> datetime:
    """
    Generate a realistic watch timestamp with patterns:
    - More watching in evenings (18:00-01:00) and weekends
    - Occasional binge sessions (clusters of events)
    """
    # Random date within range
    days_range = (HISTORY_END - HISTORY_START).days
    random_day = HISTORY_START + timedelta(days=random.randint(0, days_range))

    # Hour distribution: heavy evenings, moderate afternoons, light mornings
    hour_weights = [
        3, 2, 1, 1, 0.5, 0.5,     # 00-05: late night / very early
        1, 2, 3, 4, 5, 6,          # 06-11: morning ramp-up
        7, 7, 6, 6, 7, 8,          # 12-17: afternoon
        10, 12, 14, 15, 12, 8      # 18-23: peak evening
    ]
    hour = random.choices(range(24), weights=hour_weights, k=1)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return random_day.replace(hour=hour, minute=minute, second=second)


# ============================================================================
# Data Generation
# ============================================================================

def generate_channels() -> list[dict]:
    """Generate synthetic channel records."""
    channels = []
    all_channel_names = []
    for niche, names in CHANNEL_TEMPLATES.items():
        all_channel_names.extend(names)

    # Add some extra random channels to reach NUM_CHANNELS
    while len(all_channel_names) < NUM_CHANNELS:
        all_channel_names.append(f"{fake.first_name()} {fake.last_name()}")

    for name in all_channel_names[:NUM_CHANNELS]:
        channels.append({
            "channel_name": name,
            "channel_url": generate_channel_url(name)
        })
    return channels


def generate_videos(channels: list[dict]) -> list[dict]:
    """Generate synthetic video records linked to channels."""
    videos = []
    niches = list(TITLE_TEMPLATES.keys())

    # Map channels to niches (roughly)
    niche_channel_map = {}
    idx = 0
    for niche, names in CHANNEL_TEMPLATES.items():
        for name in names:
            niche_channel_map[name] = niche
        idx += len(names)

    for _ in range(NUM_VIDEOS):
        # Pick a random channel
        channel = random.choice(channels)
        channel_name = channel["channel_name"]

        # Determine niche from channel, or pick random
        niche = niche_channel_map.get(channel_name, random.choice(niches))

        # Generate video
        video_id = generate_video_id()
        template = random.choice(TITLE_TEMPLATES[niche])
        title = fill_template(template, niche)

        # Randomly assign a category (weighted toward niche-relevant ones)
        niche_category_map = {
            "boxing_mma": 17, "tech": 28, "music": 10, "education": 27,
            "gaming": 20, "cooking": 26, "podcasts": 24, "sports_other": 17
        }
        category_id = niche_category_map.get(niche, random.choice([22, 24, 27]))

        videos.append({
            "video_id": video_id,
            "title": f"Watched {title}",   # Matches Google Takeout format
            "title_url": generate_youtube_url(video_id),
            "channel_name": channel_name,
            "category_id": category_id,
            "status": random.choices(
                ["active", "unavailable", "private"],
                weights=[0.92, 0.06, 0.02], k=1
            )[0]
        })
    return videos


def generate_watch_events(videos: list[dict]) -> list[dict]:
    """
    Generate synthetic watch events with realistic patterns:
    - Some videos are watched once, others multiple times (re-watches)
    - Binge sessions: clusters of 3-15 videos within short time windows
    - Channel loyalty: top channels appear more frequently
    """
    events = []

    # Create a weighted video pool: some videos are "favorites" watched multiple times
    # Top 10% of videos get 5x the weight
    top_count = max(1, len(videos) // 10)
    top_videos = videos[:top_count]
    regular_videos = videos[top_count:]
    weighted_pool = top_videos * 5 + regular_videos

    # Generate events — mix of random and binge sessions
    remaining = NUM_WATCH_EVENTS
    while remaining > 0:
        if random.random() < 0.3 and remaining >= 3:
            # Binge session: 3-15 videos in quick succession
            session_length = min(random.randint(3, 15), remaining)
            session_start = generate_watch_timestamp()

            for i in range(session_length):
                video = random.choice(weighted_pool)
                watch_time = session_start + timedelta(minutes=random.randint(5, 25) * (i + 1))
                events.append({
                    "video_id": video["video_id"],
                    "channel_name": video["channel_name"],
                    "watched_at": watch_time
                })
            remaining -= session_length
        else:
            # Single random watch
            video = random.choice(weighted_pool)
            events.append({
                "video_id": video["video_id"],
                "channel_name": video["channel_name"],
                "watched_at": generate_watch_timestamp()
            })
            remaining -= 1

    return events


def generate_topics() -> list[dict]:
    """Generate synthetic NLP topic model output."""
    topics = [
        {"topic_label": "Boxing & MMA",        "keywords": ["fight", "knockout", "round", "boxing", "ufc", "mma", "title", "belt", "punch", "weigh"]},
        {"topic_label": "Programming & Tech",   "keywords": ["python", "code", "tutorial", "javascript", "developer", "api", "build", "react", "docker", "setup"]},
        {"topic_label": "Music & Artists",      "keywords": ["official", "video", "live", "album", "song", "ft", "lyrics", "session", "concert", "acoustic"]},
        {"topic_label": "Science & Education",  "keywords": ["explained", "learn", "neural", "network", "math", "science", "theory", "lecture", "course", "how"]},
        {"topic_label": "Gaming",               "keywords": ["gameplay", "playthrough", "tips", "ranked", "speedrun", "update", "best", "moments", "stream", "fps"]},
        {"topic_label": "Cooking & Food",       "keywords": ["recipe", "cook", "kitchen", "easy", "homemade", "meal", "prep", "restaurant", "street", "food"]},
        {"topic_label": "Podcasts & Interviews","keywords": ["episode", "interview", "clip", "podcast", "story", "truth", "debate", "guest", "conversation", "full"]},
        {"topic_label": "Football & Sports",    "keywords": ["goal", "highlights", "match", "season", "league", "champion", "player", "best", "premier", "transfer"]},
    ]
    return topics


# ============================================================================
# Database Insertion
# ============================================================================

def insert_data():
    """Generate all synthetic data and insert into PostgreSQL."""
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # Set search path
        cur.execute("SET search_path TO yt, public;")

        # ---- Generate data ----
        print("Generating synthetic data...")
        channels = generate_channels()
        videos = generate_videos(channels)
        watch_events = generate_watch_events(videos)
        topics = generate_topics()

        # ---- Insert channels ----
        print(f"Inserting {len(channels)} channels...")
        channel_id_map = {}
        for ch in channels:
            cur.execute(
                """INSERT INTO dim_channels (channel_name, channel_url)
                   VALUES (%s, %s)
                   ON CONFLICT (channel_name, channel_url) DO NOTHING
                   RETURNING channel_id""",
                (ch["channel_name"], ch["channel_url"])
            )
            result = cur.fetchone()
            if result:
                channel_id_map[ch["channel_name"]] = result[0]
            else:
                # Already existed — fetch the ID
                cur.execute(
                    "SELECT channel_id FROM dim_channels WHERE channel_name = %s AND channel_url = %s",
                    (ch["channel_name"], ch["channel_url"])
                )
                channel_id_map[ch["channel_name"]] = cur.fetchone()[0]

        # ---- Insert videos ----
        print(f"Inserting {len(videos)} videos...")
        for v in videos:
            ch_id = channel_id_map.get(v["channel_name"])
            cur.execute(
                """INSERT INTO dim_videos (video_id, title, title_url, channel_id, category_id, status)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (video_id) DO NOTHING""",
                (v["video_id"], v["title"], v["title_url"], ch_id, v["category_id"], v["status"])
            )

        # ---- Insert watch events ----
        print(f"Inserting {len(watch_events)} watch events...")
        event_data = []
        for e in watch_events:
            ch_id = channel_id_map.get(e["channel_name"])
            event_data.append((e["video_id"], ch_id, e["watched_at"]))

        execute_values(
            cur,
            """INSERT INTO fact_watch_events (video_id, channel_id, watched_at)
               VALUES %s
               ON CONFLICT (video_id, watched_at) DO NOTHING""",
            event_data,
            page_size=1000
        )

        # ---- Insert NLP topics ----
        print(f"Inserting {len(topics)} NLP topics...")
        topic_id_map = {}
        for t in topics:
            cur.execute(
                """INSERT INTO nlp_topics (topic_label, keywords, model_name, model_version)
                   VALUES (%s, %s, %s, %s)
                   RETURNING topic_id""",
                (t["topic_label"], t["keywords"], "nmf_tfidf", "v1.0_synthetic")
            )
            topic_id_map[t["topic_label"]] = cur.fetchone()[0]

        # ---- Assign topics to videos ----
        print("Assigning topics to videos...")
        niche_topic_map = {
            "boxing_mma": "Boxing & MMA",
            "tech": "Programming & Tech",
            "music": "Music & Artists",
            "education": "Science & Education",
            "gaming": "Gaming",
            "cooking": "Cooking & Food",
            "podcasts": "Podcasts & Interviews",
            "sports_other": "Football & Sports"
        }

        # Build a reverse lookup: channel_name -> niche
        channel_niche_map = {}
        for niche, names in CHANNEL_TEMPLATES.items():
            for name in names:
                channel_niche_map[name] = niche

        vt_data = []
        for v in videos:
            niche = channel_niche_map.get(v["channel_name"], random.choice(list(niche_topic_map.keys())))
            primary_topic = niche_topic_map[niche]
            primary_score = round(random.uniform(0.55, 0.95), 3)

            if primary_topic in topic_id_map:
                vt_data.append((v["video_id"], topic_id_map[primary_topic], primary_score))

            # Sometimes assign a secondary topic
            if random.random() < 0.3:
                secondary_topic_label = random.choice([t for t in topic_id_map if t != primary_topic])
                secondary_score = round(random.uniform(0.05, 0.35), 3)
                vt_data.append((v["video_id"], topic_id_map[secondary_topic_label], secondary_score))

        execute_values(
            cur,
            """INSERT INTO nlp_video_topics (video_id, topic_id, score)
               VALUES %s
               ON CONFLICT (video_id, topic_id) DO NOTHING""",
            vt_data,
            page_size=1000
        )

        # ---- Commit ----
        conn.commit()
        print("\n" + "=" * 60)
        print("SUCCESS: Synthetic data inserted!")
        print("=" * 60)

        # ---- Print summary ----
        summary_queries = [
            ("Channels",        "SELECT COUNT(*) FROM dim_channels"),
            ("Videos",          "SELECT COUNT(*) FROM dim_videos"),
            ("Watch Events",    "SELECT COUNT(*) FROM fact_watch_events"),
            ("NLP Topics",      "SELECT COUNT(*) FROM nlp_topics"),
            ("Video-Topic Links", "SELECT COUNT(*) FROM nlp_video_topics"),
        ]
        print("\nTable row counts:")
        for label, query in summary_queries:
            cur.execute(query)
            count = cur.fetchone()[0]
            print(f"  {label:.<30} {count:>6}")

        # Quick sanity check: show a few watch events
        print("\nSample watch events (Danish time):")
        cur.execute("""
            SELECT
                fe.watched_at AT TIME ZONE 'Europe/Copenhagen' AS watched_dk,
                dv.title,
                dc.channel_name
            FROM fact_watch_events fe
            JOIN dim_videos dv ON fe.video_id = dv.video_id
            LEFT JOIN dim_channels dc ON fe.channel_id = dc.channel_id
            ORDER BY fe.watched_at DESC
            LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"  {row[0]}  |  {row[1][:50]:.<55}  |  {row[2]}")

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()
        print("\nDatabase connection closed.")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    insert_data()
