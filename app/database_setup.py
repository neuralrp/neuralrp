"""
NeuralRP Database Setup

One-time database initialization and schema management.
Run this to create all tables, indexes, and triggers.
Safe to run multiple times - uses IF NOT EXISTS for all operations.

Version tracking allows for future schema updates.
"""

import sqlite3
import os
import sys
from typing import Optional

# Try to import sqlite-vec for extension loading
sqlite_vec = None
SQLITE_VEC_AVAILABLE = False
try:
    import sqlite_vec as _sqlite_vec
    sqlite_vec = _sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    pass

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Schema version - increment when making schema changes
SCHEMA_VERSION = 6


def setup_database(database_path: str = "app/data/neuralrp.db") -> bool:
    """
    Initialize or update the NeuralRP database.
    
    Creates all tables, indexes, triggers, and virtual tables if they don't exist.
    Updates schema version tracking.
    
    Args:
        database_path: Path to SQLite database file
        
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        
        # Enable extension loading for sqlite-vec
        try:
            conn.enable_load_extension(True)
        except AttributeError:
            # Extension loading not available in this SQLite build
            pass
        
        # Enable foreign keys
        c.execute("PRAGMA foreign_keys = ON")
        
        # Create schema version table first
        _create_schema_version_table(c)
        
        # Get current schema version
        current_version = _get_schema_version(c)
        
        if current_version < SCHEMA_VERSION:
            print(f"[SETUP] Upgrading database from version {current_version} to {SCHEMA_VERSION}")
            _apply_migrations(c, current_version, SCHEMA_VERSION)
        else:
            print(f"[SETUP] Database already at version {SCHEMA_VERSION}")
        
        # Create all core tables
        _create_characters_table(c)
        _create_chats_table(c)
        _create_messages_table(c)
        _create_worlds_table(c)
        _create_world_entries_table(c)
        _create_character_tags_table(c)
        _create_world_tags_table(c)
        _create_chat_npcs_table(c)
        _create_relationship_states_table(c)
        _create_entities_table(c)
        _create_danbooru_tags_table(c)
        _create_sd_favorites_table(c)
        _create_change_log_table(c)
        _create_performance_metrics_table(c)
        _create_image_metadata_table(c)
        _create_danbooru_characters_table(c)
        
        # Create indexes
        _create_indexes(c)
        
        # Create triggers
        _create_triggers(c)
        
        # Create virtual tables (FTS5 and vec0)
        _create_virtual_tables(c, conn)
        
        # Update schema version
        _set_schema_version(c, SCHEMA_VERSION)
        
        conn.commit()
        conn.close()
        
        print(f"[SETUP] Database setup complete at version {SCHEMA_VERSION}")
        return True
        
    except Exception as e:
        print(f"[SETUP ERROR] Failed to setup database: {e}")
        return False


def _create_schema_version_table(c):
    """Create schema version tracking table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
    """)
    # Insert initial row if not exists
    c.execute("""
        INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 0)
    """)


def _get_schema_version(c) -> int:
    """Get current schema version."""
    c.execute("SELECT version FROM schema_version WHERE id = 1")
    result = c.fetchone()
    return result[0] if result else 0


def _set_schema_version(c, version: int):
    """Update schema version."""
    c.execute("""
        UPDATE schema_version 
        SET version = ?, updated_at = strftime('%s', 'now')
        WHERE id = 1
    """, (version,))


def _apply_migrations(c, from_version: int, to_version: int):
    """Apply any necessary migrations between versions."""
    
    # Migration 0 → 1 (v1.8.0 to v1.9.0)
    if from_version < 1:
        print("[MIGRATION] Applying v1.8.0 → v1.9.0 updates...")
        
        # Add snapshot_data column to messages table (if not exists)
        _add_column_if_not_exists(c, "messages", "snapshot_data", "TEXT")
        
        # Note: New tables (sd_favorites, danbooru_tags, etc.) are created automatically
        # by the table creation functions with IF NOT EXISTS
        
        print("[MIGRATION] v1.8.0 → v1.9.0 complete")
    
    # Migration 1 → 2 (v1.10.0: Danbooru Character Casting)
    if from_version < 2:
        print("[MIGRATION] Applying v1.10.0 updates...")

        # Add visual canon columns to characters table
        _add_column_if_not_exists(c, "characters", "visual_canon_id", "INTEGER")
        _add_column_if_not_exists(c, "characters", "visual_canon_tags", "TEXT")

        # Add visual canon columns to chat_npcs table
        _add_column_if_not_exists(c, "chat_npcs", "visual_canon_id", "INTEGER")
        _add_column_if_not_exists(c, "chat_npcs", "visual_canon_tags", "TEXT")

        # Note: New tables (danbooru_characters, vec_danbooru_characters) are created automatically
        # by _create_danbooru_characters_table() with IF NOT EXISTS

        print("[MIGRATION] v1.10.0 complete")

    # Migration 2 → 3 (v1.10.1: NPC Key Consolidation)
    if from_version < 3:
        print("[MIGRATION] Applying v1.10.1 updates...")

        # Consolidate NPCs from local_npcs to localnpcs
        _consolidate_npc_metadata_keys(c)

        print("[MIGRATION] v1.10.1 complete")

    # Migration 3 → 4 (v1.10.1: Remove Learning System)
    if from_version < 4:
        print("[MIGRATION] Applying v1.10.1 remove learning system...")

        # Drop danbooru_tag_favorites table (learning system no longer used)
        _drop_table_if_exists(c, "danbooru_tag_favorites")

        print("[MIGRATION] v1.10.1 remove learning system complete")

    # Migration 4 → 5 (v1.11.0: Ensure Visual Canon Columns)
    if from_version < 5:
        print("[MIGRATION] Applying v1.11.0 visual canon column consistency...")

        # Ensure visual canon columns exist in characters table (idempotent)
        _add_column_if_not_exists(c, "characters", "visual_canon_id", "INTEGER")
        _add_column_if_not_exists(c, "characters", "visual_canon_tags", "TEXT")

        # Ensure visual canon columns exist in chat_npcs table (idempotent)
        _add_column_if_not_exists(c, "chat_npcs", "visual_canon_id", "INTEGER")
        _add_column_if_not_exists(c, "chat_npcs", "visual_canon_tags", "TEXT")

        print("[MIGRATION] v1.11.0 visual canon column consistency complete")

    # Migration 5 → 6 (v1.12.0: Unify NPCs into characters table)
    if from_version < 6:
        print("[MIGRATION] Applying v1.12.0 NPC unification...")

        # Add chat_id column to characters table (for storing NPCs)
        _add_column_if_not_exists(c, "characters", "chat_id", "TEXT")

        # Create index on chat_id for performance
        try:
            c.execute("CREATE INDEX IF NOT EXISTS idx_characters_chat_id ON characters(chat_id)")
            print("[MIGRATION] Created idx_characters_chat_id index")
        except sqlite3.Error as e:
            print(f"[MIGRATION WARNING] Could not create index idx_characters_chat_id: {e}")

        # Create unique constraint on (chat_id, filename) for NPCs
        # Note: SQLite doesn't support adding UNIQUE constraints directly,
        # so we rely on application-level uniqueness checks

        print("[MIGRATION] v1.12.0 NPC unification complete")
  
def _add_column_if_not_exists(c, table: str, column: str, dtype: str):
    """Add a column to a table if it doesn't already exist."""
    try:
        # Check if column exists
        c.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in c.fetchall()]
        
        if column not in columns:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
            print(f"[MIGRATION] Added column '{column}' to '{table}'")
        else:
            print(f"[MIGRATION] Column '{column}' already exists in '{table}'")
    except sqlite3.Error as e:
        print(f"[MIGRATION WARNING] Could not add column {column} to {table}: {e}")


def _drop_table_if_exists(c, table: str):
    """Drop a table if it exists."""
    try:
        # Check if table exists
        c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if c.fetchone():
            c.execute(f"DROP TABLE {table}")
            print(f"[MIGRATION] Dropped table '{table}'")
        else:
            print(f"[MIGRATION] Table '{table}' does not exist, skipping")
    except sqlite3.Error as e:
        print(f"[MIGRATION WARNING] Could not drop table {table}: {e}")

def _consolidate_npc_metadata_keys(c):
    """
    Consolidate NPCs from local_npcs to localnpcs.
    
    Fixes split-key issue caused by migrate_npcs_to_db.py script that
    renamed localnpcs → local_npcs. This migration merges both keys
    back to single localnpcs key for consistency.
    """
    import json
    
    print("[MIGRATION] Consolidating NPC metadata keys (local_npcs → localnpcs)...")
    
    # Get all chats
    c.execute("SELECT id, metadata FROM chats")
    chats = c.fetchall()
    
    total_consolidated = 0
    
    for chat in chats:
        chat_id = chat[0]
        metadata = json.loads(chat[1]) if chat[1] else {}
        
        # Check if local_npcs exists
        if 'local_npcs' in metadata:
            local_npcs = metadata['local_npcs']
            localnpcs = metadata.get('localnpcs', {})
            
            # Merge: localnpcs takes precedence (newer data)
            merged_npcs = {**local_npcs, **localnpcs}
            
            # Update metadata to use only localnpcs
            metadata['localnpcs'] = merged_npcs
            del metadata['local_npcs']  # Remove old key
            
            # Update chat
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            c.execute(
                "UPDATE chats SET metadata = ? WHERE id = ?",
                (metadata_json, chat_id)
            )
            
            total_consolidated += len(local_npcs)
            print(f"[MIGRATION]   Consolidated {len(local_npcs)} NPCs for chat {chat_id}")
    
    if total_consolidated > 0:
        print(f"[MIGRATION] [SUCCESS] Consolidated {total_consolidated} NPCs across all chats")
    else:
        print("[MIGRATION] No NPC consolidation needed (all chats already use localnpcs)")

def _create_characters_table(c):
    """Create characters table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT UNIQUE,
            data TEXT,
            danbooru_tag TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
    """)


def _create_chats_table(c):
    """Create chats table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            branch_name TEXT,
            summary TEXT,
            metadata TEXT,
            created_at INTEGER,
            autosaved BOOLEAN DEFAULT 1
        )
    """)


def _create_messages_table(c):
    """Create messages table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            speaker TEXT,
            image_url TEXT,
            timestamp INTEGER,
            summarized BOOLEAN DEFAULT 0,
            snapshot_data TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    """)


def _create_worlds_table(c):
    """Create worlds table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS worlds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)


def _create_world_entries_table(c):
    """Create world entries table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS world_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            world_id INTEGER,
            uid TEXT,
            content TEXT,
            keys TEXT,
            secondary_keys TEXT,
            is_canon_law BOOLEAN,
            probability INTEGER,
            use_probability BOOLEAN,
            depth INTEGER,
            sort_order INTEGER,
            embedding_hash TEXT,
            last_embedded_at INTEGER,
            metadata TEXT,
            updated_at INTEGER,
            FOREIGN KEY (world_id) REFERENCES worlds(id) ON DELETE CASCADE
        )
    """)


def _create_character_tags_table(c):
    """Create character_tags junction table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS character_tags (
            char_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (char_id, tag),
            FOREIGN KEY (char_id) REFERENCES characters(filename) ON DELETE CASCADE
        )
    """)


def _create_world_tags_table(c):
    """Create world_tags junction table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS world_tags (
            world_name TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (world_name, tag),
            FOREIGN KEY (world_name) REFERENCES worlds(name) ON DELETE CASCADE
        )
    """)


def _create_chat_npcs_table(c):
    """Create chat_npcs table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_npcs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            name TEXT NOT NULL,
            data TEXT NOT NULL,
            created_from_text TEXT,
            created_at INTEGER,
            last_used_turn INTEGER,
            appearance_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            promoted BOOLEAN DEFAULT 0,
            promoted_at INTEGER,
            global_filename TEXT,
            visual_canon_id INTEGER,
            visual_canon_tags TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
            UNIQUE(chat_id, entity_id),
            UNIQUE(chat_id, name)
        )
    """)


def _create_relationship_states_table(c):
    """Create relationship_states table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS relationship_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            character_from TEXT NOT NULL,
            character_to TEXT NOT NULL,
            trust INTEGER DEFAULT 50,
            emotional_bond INTEGER DEFAULT 50,
            conflict INTEGER DEFAULT 50,
            power_dynamic INTEGER DEFAULT 50,
            fear_anxiety INTEGER DEFAULT 50,
            last_updated INTEGER,
            last_analyzed_message_id INTEGER,
            interaction_count INTEGER DEFAULT 0,
            history TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
            UNIQUE(chat_id, character_from, character_to)
        )
    """)


def _create_entities_table(c):
    """Create entities table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            first_seen INTEGER NOT NULL,
            last_seen INTEGER NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    """)


def _create_danbooru_tags_table(c):
    """Create danbooru_tags table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS danbooru_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_text TEXT UNIQUE NOT NULL,
            block_num INTEGER NOT NULL,
            frequency INTEGER DEFAULT 0,
            created_at INTEGER
        )
    """)


def _create_sd_favorites_table(c):
    """Create sd_favorites table."""
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


def _create_change_log_table(c):
    """Create change_log table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            old_data TEXT,
            new_data TEXT,
            timestamp INTEGER DEFAULT (strftime('%s', 'now')),
            undo_state TEXT DEFAULT 'active'
        )
    """)


def _create_performance_metrics_table(c):
    """Create performance_metrics table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_type TEXT,
            duration REAL,
            context_tokens INTEGER,
            timestamp INTEGER
        )
    """)


def _create_image_metadata_table(c):
    """Create image_metadata table."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS image_metadata (
            filename TEXT PRIMARY KEY,
            prompt TEXT,
            negative_prompt TEXT,
            steps INTEGER,
            cfg_scale REAL,
            width INTEGER,
            height INTEGER,
            sampler TEXT,
            scheduler TEXT,
            timestamp INTEGER
        )
    """)


def _create_danbooru_characters_table(c):
    """Create danbooru_characters table for Danbooru character casting."""
    c.execute("""
        CREATE TABLE IF NOT EXISTS danbooru_characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL CHECK(gender IN ('male', 'female', 'other', 'unknown')),
            core_tags TEXT NOT NULL,
            all_tags TEXT NOT NULL,
            image_link TEXT,
            source_id TEXT UNIQUE,
            created_at INTEGER
        )
    """)


def _create_indexes(c):
    """Create all database indexes."""
    indexes = [
        # Messages indexes
        ("idx_messages_chat_id", "CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)"),
        ("idx_messages_summarized", "CREATE INDEX IF NOT EXISTS idx_messages_summarized ON messages(chat_id, summarized)"),
        
        # World entries index
        ("idx_world_entries_world_id", "CREATE INDEX IF NOT EXISTS idx_world_entries_world_id ON world_entries(world_id)"),
        
        # Performance metrics index
        ("idx_performance_metrics_type", "CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(operation_type, timestamp)"),
        
        # Change log indexes
        ("idx_change_log_entity", "CREATE INDEX IF NOT EXISTS idx_change_log_entity ON change_log(entity_type, entity_id)"),
        ("idx_change_log_time", "CREATE INDEX IF NOT EXISTS idx_change_log_time ON change_log(timestamp DESC)"),
        
        # Relationship states indexes
        ("idx_relationship_states_chat", "CREATE INDEX IF NOT EXISTS idx_relationship_states_chat ON relationship_states(chat_id)"),
        ("idx_relationship_states_chars", "CREATE INDEX IF NOT EXISTS idx_relationship_states_chars ON relationship_states(character_from, character_to)"),
        
        # Entities indexes
        ("idx_entities_chat", "CREATE INDEX IF NOT EXISTS idx_entities_chat ON entities(chat_id)"),
        ("idx_entities_type", "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)"),
        
        # Chat NPCs indexes
        ("idx_chat_npcs_chat", "CREATE INDEX IF NOT EXISTS idx_chat_npcs_chat ON chat_npcs(chat_id)"),
        ("idx_chat_npcs_active", "CREATE INDEX IF NOT EXISTS idx_chat_npcs_active ON chat_npcs(chat_id, is_active)"),
        ("idx_chat_npcs_entity", "CREATE INDEX IF NOT EXISTS idx_chat_npcs_entity ON chat_npcs(entity_id)"),
        
        # Tag indexes
        ("idx_character_tags_tag", "CREATE INDEX IF NOT EXISTS idx_character_tags_tag ON character_tags(tag)"),
        ("idx_world_tags_tag", "CREATE INDEX IF NOT EXISTS idx_world_tags_tag ON world_tags(tag)"),
        
        # Danbooru tags indexes
        ("idx_danbooru_tags_block", "CREATE INDEX IF NOT EXISTS idx_danbooru_tags_block ON danbooru_tags(block_num)"),
        ("idx_danbooru_tags_text", "CREATE INDEX IF NOT EXISTS idx_danbooru_tags_text ON danbooru_tags(tag_text)"),
        
        # Danbooru characters indexes
        ("idx_danbooru_characters_gender", "CREATE INDEX IF NOT EXISTS idx_danbooru_characters_gender ON danbooru_characters(gender)"),
        ("idx_danbooru_characters_name", "CREATE INDEX IF NOT EXISTS idx_danbooru_characters_name ON danbooru_characters(name)"),
        
        # Favorites indexes
        ("idx_sd_favorites_scene_type", "CREATE INDEX IF NOT EXISTS idx_sd_favorites_scene_type ON sd_favorites(scene_type)"),
        ("idx_sd_favorites_source_type", "CREATE INDEX IF NOT EXISTS idx_sd_favorites_source_type ON sd_favorites(source_type)"),
        ("idx_sd_favorites_created", "CREATE INDEX IF NOT EXISTS idx_sd_favorites_created ON sd_favorites(created_at DESC)"),
    ]
    
    for name, sql in indexes:
        try:
            c.execute(sql)
        except sqlite3.Error as e:
            print(f"[SETUP WARNING] Could not create index {name}: {e}")


def _create_triggers(c):
    """Create FTS synchronization triggers."""
    # Drop existing triggers first to avoid errors
    triggers = ['messages_ai', 'messages_ad', 'messages_au']
    for trigger in triggers:
        c.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    
    # After insert trigger
    c.execute("""
        CREATE TRIGGER messages_ai AFTER INSERT ON messages
        BEGIN
            INSERT INTO messages_fts(message_id, chat_id, content, speaker, role, timestamp)
            VALUES (new.id, new.chat_id, new.content, new.speaker, new.role, new.timestamp);
        END
    """)
    
    # After delete trigger
    c.execute("""
        CREATE TRIGGER messages_ad AFTER DELETE ON messages
        BEGIN
            DELETE FROM messages_fts WHERE message_id = old.id;
        END
    """)
    
    # After update trigger
    c.execute("""
        CREATE TRIGGER messages_au AFTER UPDATE ON messages
        BEGIN
            DELETE FROM messages_fts WHERE message_id = old.id;
            INSERT INTO messages_fts(message_id, chat_id, content, speaker, role, timestamp)
            VALUES (new.id, new.chat_id, new.content, new.speaker, new.role, new.timestamp);
        END
    """)


def _create_virtual_tables(c, conn):
    """Create virtual tables (FTS5 and vec0)."""
    # FTS5 table for message search
    try:
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
            USING fts5(
                message_id UNINDEXED,
                chat_id UNINDEXED,
                content,
                speaker,
                role,
                timestamp UNINDEXED,
                tokenize='porter unicode61'
            )
        """)
    except sqlite3.Error as e:
        print(f"[SETUP WARNING] Could not create FTS5 table: {e}")
    
    # Load sqlite-vec extension before creating vec0 tables
    vec_loaded = False
    if SQLITE_VEC_AVAILABLE and sqlite_vec is not None:
        try:
            sqlite_vec.load(conn)
            vec_loaded = True
            print("[SETUP] sqlite-vec extension loaded successfully")
        except Exception as e:
            print(f"[SETUP WARNING] Could not load sqlite-vec extension: {e}")
    
    # vec0 table for world entries embeddings
    if vec_loaded:
        try:
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_world_entries 
                USING vec0(
                    entry_id INTEGER PRIMARY KEY,
                    world_name TEXT,
                    entry_uid TEXT,
                    embedding FLOAT[768]
                )
            """)
            print("[SETUP] vec_world_entries table created")
        except sqlite3.Error as e:
            print(f"[SETUP WARNING] Could not create vec_world_entries table: {e}")
    else:
        print("[SETUP] Skipping vec_world_entries table (sqlite-vec not available)")
    
    # vec0 table for danbooru tags embeddings
    if vec_loaded:
        try:
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_danbooru_tags 
                USING vec0(
                    embedding float[768]
                )
            """)
            print("[SETUP] vec_danbooru_tags table created")
        except sqlite3.Error as e:
            print(f"[SETUP WARNING] Could not create vec_danbooru_tags table: {e}")
    else:
        print("[SETUP] Skipping vec_danbooru_tags table (sqlite-vec not available)")
    
    # vec0 table for danbooru characters embeddings
    if vec_loaded:
        try:
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_danbooru_characters 
                USING vec0(
                    embedding float[768]
                )
            """)
            print("[SETUP] vec_danbooru_characters table created")
        except sqlite3.Error as e:
            print(f"[SETUP WARNING] Could not create vec_danbooru_characters table: {e}")
    else:
        print("[SETUP] Skipping vec_danbooru_characters table (sqlite-vec not available)")


if __name__ == "__main__":
    import sys
    
    # Allow custom database path from command line
    db_path = sys.argv[1] if len(sys.argv) > 1 else "app/data/neuralrp.db"
     
    print(f"[SETUP] Initializing NeuralRP database at: {db_path}")
    success = setup_database(db_path)
    
    if success:
        print("[SETUP] Database setup complete")
        sys.exit(0)
    else:
        print("[SETUP] Database setup failed")
        sys.exit(1)

