
import sqlite3
from datetime import datetime

DB_NAME = "presentai_history.db"


def create_table():
    """Create the presentation history table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS presentation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            word_count INTEGER,
            speaking_speed REAL,
            filler_count INTEGER,
            practice_score INTEGER,
            transcript TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_session(
    word_count,
    speaking_speed,
    filler_count,
    practice_score,
    transcript
):
    """Save one presentation practice session."""
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO presentation_history (
            date,
            word_count,
            speaking_speed,
            filler_count,
            practice_score,
            transcript
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        int(word_count),
        float(speaking_speed),
        int(filler_count),
        int(practice_score),
        transcript
    ))

    conn.commit()
    conn.close()


def get_history():
    """Return saved sessions, newest first."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM presentation_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return [dict(row) for row in rows]
