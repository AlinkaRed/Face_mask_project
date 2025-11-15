import sqlite3
import os
import json
from dotenv import load_dotenv

load_dotenv()

def recreate_db() -> None:
    conn = sqlite3.connect(os.getenv('SQLITE_DB_PATH'))
    with conn:
        conn.execute("DROP TABLE IF EXISTS tg_updates")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tg_updates
            (
                id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL
            )
        """)
    conn.close()

def persist_updates(updates: list) -> None:
    conn = sqlite3.connect(os.getenv('SQLITE_DB_PATH'))
    with conn:
        data = []
        for update in updates:
            data.append(
                (json.dumps(update), )
            )
        conn.executemany(
            "INSERT INTO tg_updates (payload) VALUES (?)",
            data,
        )
    conn.close()