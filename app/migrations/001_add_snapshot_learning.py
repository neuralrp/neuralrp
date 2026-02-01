"""
Migration 001: Add Snapshot and Manual Mode Favorites with Learning

Run this script to add tables for snapshot favorites, manual mode support, and tag learning.
"""

import sqlite3
import os


def migrate(database_path: str = "app/data/neuralrp.db"):
    """
    Apply migration to add snapshot learning and manual mode tables.

    Args:
        database_path: Path to SQLite database file
    """
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    # Check if migration already applied
    c.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='schema_migrations'
    """)
    schema_migrations_exists = c.fetchone() is not None

    if schema_migrations_exists:
        c.execute("SELECT migration_name FROM schema_migrations WHERE migration_name = ?", ("001_add_snapshot_learning",))
        if c.fetchone():
            print("[MIGRATION] 001_add_snapshot_learning already applied")
            conn.close()
            return

    # Create migration tracking table if not exists
    if not schema_migrations_exists:
        c.execute("""
            CREATE TABLE schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT UNIQUE NOT NULL,
                applied_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        print("[MIGRATION] Created schema_migrations table")

    # Table 1: Snapshot Favorites (with source_type column)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sd_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            image_filename TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL,
            negative_prompt TEXT NOT NULL,
            scene_type TEXT,
            setting TEXT,
            mood TEXT,
            character_ref TEXT,
            tags TEXT,
            steps INTEGER,
            cfg_scale REAL,
            width INTEGER,
            height INTEGER,
            source_type TEXT DEFAULT 'snapshot',
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            note TEXT
        )
    """)
    print("[MIGRATION] Created sd_favorites table (with source_type column)")

    # Indexes for sd_favorites
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_sd_favorites_scene_type
        ON sd_favorites(scene_type)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_sd_favorites_source_type
        ON sd_favorites(source_type)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_sd_favorites_created
        ON sd_favorites(created_at DESC)
    """)
    print("[MIGRATION] Created indexes for sd_favorites")

    # Table 2: Danbooru Tag Favorites
    c.execute("""
        CREATE TABLE IF NOT EXISTS danbooru_tag_favorites (
            tag_text TEXT PRIMARY KEY,
            favorite_count INTEGER DEFAULT 0,
            last_used INTEGER DEFAULT (strftime('%s', 'now'))
        )
    """)
    print("[MIGRATION] Created danbooru_tag_favorites table")

    # Record migration
    c.execute("""
        INSERT INTO schema_migrations (migration_name)
        VALUES ('001_add_snapshot_learning')
    """)
    conn.commit()
    conn.close()

    print("[MIGRATION] Successfully applied 001_add_snapshot_learning")


if __name__ == "__main__":
    migrate()
