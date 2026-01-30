"""
NeuralRP Database Module
Centralized SQLite database operations for characters, world info, chats, and images.
"""

import sqlite3
import json
import time
import threading
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import numpy as np
import struct
from app.tag_manager import parse_tag_string

# Global database path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "app", "data", "neuralrp.db")

# Thread-local storage for connections
_thread_local = threading.local()


@contextmanager
def get_connection():
    """Get a thread-safe database connection with context manager."""
    if not hasattr(_thread_local, 'connection') or _thread_local.connection is None:
        _thread_local.connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _thread_local.connection.row_factory = sqlite3.Row

        # CRITICAL: Ensure data flushes to disk
        _thread_local.connection.execute("PRAGMA synchronous = FULL")  # Safest mode
        _thread_local.connection.execute("PRAGMA journal_mode = WAL")  # Better concurrency

        # Enable extension loading for sqlite-vec
        try:
            _thread_local.connection.enable_load_extension(True)
        except AttributeError:
            # Extension loading not available in this SQLite build
            pass

        # Enable foreign keys
        _thread_local.connection.execute("PRAGMA foreign_keys = ON")
    
    try:
        yield _thread_local.connection
    except Exception as e:
        _thread_local.connection.rollback()
        raise e



def init_db():
    """Initialize database tables if they don't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Characters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                filename TEXT UNIQUE,
                data TEXT,
                danbooru_tag TEXT,
                capsule TEXT,
                created_at INTEGER
            )
        """)
        
        # Worlds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS worlds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE
            )
        """)
        
        # World entries table
        cursor.execute("""
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
                FOREIGN KEY (world_id) REFERENCES worlds (id) ON DELETE CASCADE
            )
        """)
        
        # Chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                branch_name TEXT,
                summary TEXT,
                metadata TEXT,
                created_at INTEGER,
                autosaved BOOLEAN DEFAULT 1
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                role TEXT,
                content TEXT,
                speaker TEXT,
                image_url TEXT,
                timestamp INTEGER,
                summarized INTEGER DEFAULT 0,
                FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
            )
        """)
        
        # Add summarized column if it doesn't exist (for older databases)
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN summarized INTEGER DEFAULT 0")
            print("Added summarized column to messages table")
        except:
            pass  # Column already exists
        
        # Add updated_at column to characters table (v1.8.0+ for smart sync)
        try:
            cursor.execute("ALTER TABLE characters ADD COLUMN updated_at INTEGER")
            print("Added updated_at column to characters table")
        except:
            pass  # Column already exists
        
        # Add updated_at column to world_entries table (v1.8.0+ for smart sync)
        try:
            cursor.execute("ALTER TABLE world_entries ADD COLUMN updated_at INTEGER")
            print("Added updated_at column to world_entries table")
        except:
            pass  # Column already exists
        
        # Backfill updated_at for existing characters (set to created_at if null)
        try:
            cursor.execute("UPDATE characters SET updated_at = created_at WHERE updated_at IS NULL")
            conn.commit()
            print("Backfilled updated_at for existing characters")
        except Exception as e:
            print(f"Warning: Failed to backfill updated_at for characters: {e}")
        
        # Backfill updated_at for existing world entries (set to last_embedded_at or current time if null)
        try:
            cursor.execute("""
                UPDATE world_entries 
                SET updated_at = COALESCE(last_embedded_at, ?) 
                WHERE updated_at IS NULL
            """, (int(time.time()),))
            conn.commit()
            print("Backfilled updated_at for existing world entries")
        except Exception as e:
            print(f"Warning: Failed to backfill updated_at for world entries: {e}")
        
        # Drop capsule column from characters table (v1.8.2+ - capsules now chat-scoped)
        # Capsules are now stored in chat.metadata.characterCapsules instead
        try:
            cursor.execute("ALTER TABLE characters DROP COLUMN capsule")
            conn.commit()
            print("Dropped capsule column from characters table (capsules now chat-scoped)")
        except:
            pass  # Column doesn't exist or already dropped
        
        # Image metadata table
        cursor.execute("""
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
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT,
                duration REAL,
                context_tokens INTEGER,
                timestamp INTEGER
            )
        """)
        
        # Create indices if they don't exist
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_world_entries_world_id ON world_entries(world_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(operation_type, timestamp)
        """)
        
        # Change log for undo/redo support (v1.5.1 foundation)
        cursor.execute("""
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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_change_log_entity 
            ON change_log(entity_type, entity_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_change_log_time 
            ON change_log(timestamp DESC)
        """)
        
        # Relationship state tracking table (for relationship tracking feature)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationship_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                character_from TEXT NOT NULL,
                character_to TEXT NOT NULL,
                
                -- Five core dimensions (0-100 scale)
                trust INTEGER DEFAULT 50,
                emotional_bond INTEGER DEFAULT 50,
                conflict INTEGER DEFAULT 50,
                power_dynamic INTEGER DEFAULT 50,
                fear_anxiety INTEGER DEFAULT 50,
                
                -- Metadata
                last_updated INTEGER,
                last_analyzed_message_id INTEGER,
                interaction_count INTEGER DEFAULT 0,
                history TEXT,
                
                FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE,
                UNIQUE(chat_id, character_from, character_to)
            )
        """)
        
        # Performance indexes for relationship_states
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_relationship_states_chat
            ON relationship_states(chat_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_relationship_states_chars
            ON relationship_states(character_from, character_to)
        ''')
        
        # Chat-scoped NPC table (Phase 2.1)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_npcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                name TEXT NOT NULL,
                data TEXT NOT NULL,

                -- Metadata
                created_from_text TEXT,
                created_at INTEGER NOT NULL,
                last_used_turn INTEGER,
                appearance_count INTEGER DEFAULT 0,

                -- Flags
                is_active INTEGER DEFAULT 1,
                promoted BOOLEAN DEFAULT 0,
                promoted_at INTEGER,
                global_filename TEXT,

                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
                UNIQUE(chat_id, entity_id),
                UNIQUE(chat_id, name)
            )
        """)
        
        # Performance indexes for chat_npcs
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_npcs_chat
            ON chat_npcs(chat_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_npcs_active
            ON chat_npcs(chat_id, is_active)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_npcs_entity_id
            ON chat_npcs(entity_id)
        ''')
        
        # Entities table for unified entity ID management (v1.5.1+)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,  -- 'character', 'npc', 'user'
                name TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_chat 
            ON entities(chat_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type 
            ON entities(entity_type)
        """)
        
        # Character tags table (junction table for many-to-many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS character_tags (
                char_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (char_id, tag),
                FOREIGN KEY (char_id) REFERENCES characters(filename) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_character_tags_tag 
            ON character_tags(tag)
        """)
        
        # World tags table (junction table for many-to-many relationship)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_tags (
                world_name TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (world_name, tag),
                FOREIGN KEY (world_name) REFERENCES worlds(name) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_world_tags_tag 
            ON world_tags(tag)
        """)
        
        conn.commit()
        
        # ============================================================================  
        # TAG MIGRATION (v1.8.0+)
        # ============================================================================  
        
        def check_and_migrate_tags():
            """Migrate existing tags from character/world data to junction tables.
            
            One-time migration that runs when tag tables are empty.
            Extracts tags from character['data']['tags'] and world['tags'] arrays
            and saves them to character_tags/world_tags junction tables.
            """
            # Check if tags already exist in junction tables
            cursor.execute("SELECT COUNT(*) FROM character_tags")
            char_tag_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM world_tags")
            world_tag_count = cursor.fetchone()[0]
            
            # Skip migration if tags already exist
            if char_tag_count > 0 or world_tag_count > 0:
                return False
            
            print("[MIGRATION] Checking for tags to migrate...")
            
            # Migrate character tags
            char_migrated = 0
            cursor.execute("SELECT filename, data FROM characters")
            for row in cursor.fetchall():
                filename = row['filename']
                try:
                    char_data = json.loads(row['data'])
                    # Extract tags from top level (SillyTavern V2 format)
                    tags = char_data.get('tags', [])
                    if tags:
                        # Normalize tags
                        normalized_tags = []
                        for tag in tags:
                            parsed = parse_tag_string(tag)
                            normalized_tags.extend(parsed)
                        normalized_tags = [t for t in normalized_tags if t]
                        
                        # Add to junction table
                        for tag in normalized_tags:
                            cursor.execute("""
                                INSERT OR IGNORE INTO character_tags (char_id, tag)
                                VALUES (?, ?)
                            """, (filename, tag))
                        char_migrated += 1
                except Exception as e:
                    print(f"[MIGRATION] Failed to migrate tags for {filename}: {e}")
            
            # Migrate world tags
            world_migrated = 0
            cursor.execute("SELECT name FROM worlds")
            for row in cursor.fetchall():
                world_name = row['name']
                try:
                    # World data is stored differently - need to check JSON file
                    world_file = os.path.join(BASE_DIR, "app", "data", "worldinfo", f"{world_name}.json")
                    if os.path.exists(world_file):
                        with open(world_file, "r", encoding="utf-8") as f:
                            world_data = json.load(f)
                        # Extract tags from world['tags'] (top level)
                        tags = world_data.get('tags', [])
                        if tags:
                            # Normalize tags
                            normalized_tags = []
                            for tag in tags:
                                parsed = parse_tag_string(tag)
                                normalized_tags.extend(parsed)
                            normalized_tags = [t for t in normalized_tags if t]
                            
                            # Add to junction table
                            for tag in normalized_tags:
                                cursor.execute("""
                                    INSERT OR IGNORE INTO world_tags (world_name, tag)
                                    VALUES (?, ?)
                                """, (world_name, tag))
                            world_migrated += 1
                except Exception as e:
                    print(f"[MIGRATION] Failed to migrate tags for {world_name}: {e}")
            
            conn.commit()
            
            if char_migrated > 0 or world_migrated > 0:
                print(f"[MIGRATION] Migration complete: {char_migrated} character tags, {world_migrated} world tags")
            else:
                print("[MIGRATION] No tags found to migrate")
            
            return True
        
        # Run migration on startup
        check_and_migrate_tags()
        
        # ============================================================================  
        # CHARACTER OPERATIONS
        # ============================================================================

def db_get_all_characters() -> List[Dict[str, Any]]:
    """Get all characters from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, data, danbooru_tag, created_at, updated_at
            FROM characters
            ORDER BY created_at DESC
        """)
        
        characters = []
        for row in cursor.fetchall():
            char_data = json.loads(row['data'])
            char_data['_filename'] = row['filename']
            
            # Inject extensions if they exist
            if 'data' in char_data and 'extensions' not in char_data['data']:
                char_data['data']['extensions'] = {}
            
            # Add danbooru_tag to extensions
            if row['danbooru_tag']:
                char_data['data']['extensions']['danbooru_tag'] = row['danbooru_tag']
            
            characters.append(char_data)
        
        return characters


def db_get_character(filename: str) -> Optional[Dict[str, Any]]:
    """Get a specific character by filename."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, data, danbooru_tag, updated_at
            FROM characters
            WHERE filename = ?
        """, (filename,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        char_data = json.loads(row['data'])
        char_data['_filename'] = row['filename']
        
        # Inject extensions
        if 'data' in char_data and 'extensions' not in char_data['data']:
            char_data['data']['extensions'] = {}
        
        if row['danbooru_tag']:
            char_data['data']['extensions']['danbooru_tag'] = row['danbooru_tag']
        
        return char_data


def db_save_character(char_data: Dict[str, Any], filename: str) -> bool:
    """Save or update a character in the database."""
    try:
        # Get old state before saving (for change logging)
        old_char = db_get_character(filename)
        is_create = old_char is None
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract fields
            name = char_data.get('data', {}).get('name', 'Unknown')
            extensions = char_data.get('data', {}).get('extensions', {})
            danbooru_tag = extensions.get('danbooru_tag', '')
            
            # Remove _filename from data before saving
            save_data = char_data.copy()
            if '_filename' in save_data:
                del save_data['_filename']
            
            data_json = json.dumps(save_data, ensure_ascii=False)
            timestamp = int(time.time())
            
            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO characters 
                (filename, name, data, danbooru_tag, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (filename, name, data_json, danbooru_tag, timestamp, timestamp))
            
            conn.commit()
            
            # Log the change (characters are always significant)
            operation = 'CREATE' if is_create else 'UPDATE'
            log_change('character', filename, operation, old_char, save_data)
            
            # Automatic tag extraction (v1.8.0+)
            # Extract tags from character data and save to junction table
            tags = char_data.get('data', {}).get('tags', [])
            if tags:
                # Normalize tags
                normalized_tags = []
                for tag in tags:
                    parsed = parse_tag_string(tag)
                    normalized_tags.extend(parsed)
                normalized_tags = [t for t in normalized_tags if t]
                
                # Clear existing tags (in case of update)
                db_remove_character_tags(filename, [])
                
                # Add new tags
                if normalized_tags:
                    db_add_character_tags(filename, normalized_tags)
            
            return True
    except Exception as e:
        print(f"Error saving character to database: {e}")
        return False


def db_delete_character(filename: str) -> bool:
    """Delete a character from the database."""
    try:
        # Get old state before deleting (for change logging)
        old_char = db_get_character(filename)
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM characters WHERE filename = ?", (filename,))
            conn.commit()
            deleted = cursor.rowcount > 0
            
            # Log the deletion (characters are always significant)
            if deleted and old_char:
                log_change('character', filename, 'DELETE', old_char, None)
            
            return deleted
    except Exception as e:
        print(f"Error deleting character from database: {e}")
        return False


# ============================================================================
# TAG OPERATIONS
# ============================================================================

def db_add_character_tags(filename: str, tags: List[str]) -> bool:
    """Add tags to a character in the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            for tag in tags:
                cursor.execute("""
                    INSERT OR IGNORE INTO character_tags (char_id, tag)
                    VALUES (?, ?)
                """, (filename, tag.lower().strip()))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error adding character tags: {e}")
        return False


def db_remove_character_tags(filename: str, tags: List[str]) -> bool:
    """Remove specific tags from a character."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            normalized_tags = [tag.lower().strip() for tag in tags]
            if not normalized_tags:
                cursor.execute("DELETE FROM character_tags WHERE char_id = ?", (filename,))
            else:
                placeholders = ','.join(['?'] * len(normalized_tags))
                cursor.execute(f"""
                    DELETE FROM character_tags
                    WHERE char_id = ? AND tag IN ({placeholders})
                """, [filename] + normalized_tags)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error removing character tags: {e}")
        return False


def db_get_character_tags(filename: str) -> List[str]:
    """Get all tags for a specific character."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tag FROM character_tags
            WHERE char_id = ?
            ORDER BY tag
        """, (filename,))
        return [row['tag'] for row in cursor.fetchall()]


def db_get_all_character_tags(limit: Optional[int] = None) -> List[str]:
    """Get all unique character tags (alphabetical order)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if limit:
            cursor.execute("""
                SELECT DISTINCT tag
                FROM character_tags
                ORDER BY tag
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT DISTINCT tag
                FROM character_tags
                ORDER BY tag
            """)
        return [row['tag'] for row in cursor.fetchall()]


def db_get_popular_character_tags(limit: int = 5) -> List[Tuple[str, int]]:
    """Get top N tags by usage count."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tag, COUNT(*) as count
            FROM character_tags
            GROUP BY tag
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        return [(row['tag'], row['count']) for row in cursor.fetchall()]


def db_get_characters_by_tags(tags: List[str]) -> List[Dict[str, Any]]:
    """Get characters matching ALL specified tags (AND semantics)."""
    if not tags:
        return []
    
    normalized_tags = [tag.lower().strip() for tag in tags if tag.strip()]
    if not normalized_tags:
        return []
    
    placeholders = ','.join(['?'] * len(normalized_tags))
    query = f"""
        SELECT c.filename, c.data, c.danbooru_tag, c.capsule, c.created_at
        FROM characters c
        JOIN character_tags t ON t.char_id = c.filename
        WHERE t.tag IN ({placeholders})
        GROUP BY c.filename
        HAVING COUNT(DISTINCT t.tag) = ?
    """
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, normalized_tags + [len(normalized_tags)])
        
        characters = []
        for row in cursor.fetchall():
            char_data = json.loads(row['data'])
            char_data['_filename'] = row['filename']
            
            # Inject extensions if they exist
            if 'data' in char_data and 'extensions' not in char_data['data']:
                char_data['data']['extensions'] = {}
            
            # Add danbooru_tag to extensions
            if row['danbooru_tag']:
                char_data['data']['extensions']['danbooru_tag'] = row['danbooru_tag']
            
            characters.append(char_data)
        
        return characters


def db_add_world_tags(world_name: str, tags: List[str]) -> bool:
    """Add tags to a world in the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            for tag in tags:
                cursor.execute("""
                    INSERT OR IGNORE INTO world_tags (world_name, tag)
                    VALUES (?, ?)
                """, (world_name, tag.lower().strip()))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error adding world tags: {e}")
        return False


def db_remove_world_tags(world_name: str, tags: List[str]) -> bool:
    """Remove specific tags from a world."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            normalized_tags = [tag.lower().strip() for tag in tags]
            if not normalized_tags:
                cursor.execute("DELETE FROM world_tags WHERE world_name = ?", (world_name,))
            else:
                placeholders = ','.join(['?'] * len(normalized_tags))
                cursor.execute(f"""
                    DELETE FROM world_tags
                    WHERE world_name = ? AND tag IN ({placeholders})
                """, [world_name] + normalized_tags)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error removing world tags: {e}")
        return False


def db_get_world_tags(world_name: str) -> List[str]:
    """Get all tags for a specific world."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tag FROM world_tags
            WHERE world_name = ?
            ORDER BY tag
        """, (world_name,))
        return [row['tag'] for row in cursor.fetchall()]


def db_get_all_world_tags(limit: Optional[int] = None) -> List[str]:
    """Get all unique world tags (alphabetical order)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if limit:
            cursor.execute("""
                SELECT DISTINCT tag
                FROM world_tags
                ORDER BY tag
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT DISTINCT tag
                FROM world_tags
                ORDER BY tag
            """)
        return [row['tag'] for row in cursor.fetchall()]


def db_get_popular_world_tags(limit: int = 5) -> List[Tuple[str, int]]:
    """Get top N tags by usage count for worlds."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tag, COUNT(*) as count
            FROM world_tags
            GROUP BY tag
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        return [(row['tag'], row['count']) for row in cursor.fetchall()]


def db_get_worlds_by_tags(tags: List[str]) -> List[Dict[str, Any]]:
    """Get worlds matching ALL specified tags (AND semantics)."""
    if not tags:
        return []
    
    normalized_tags = [tag.lower().strip() for tag in tags if tag.strip()]
    if not normalized_tags:
        return []
    
    placeholders = ','.join(['?'] * len(normalized_tags))
    query = f"""
        SELECT w.id, w.name
        FROM worlds w
        JOIN world_tags t ON t.world_name = w.name
        WHERE t.tag IN ({placeholders})
        GROUP BY w.name
        HAVING COUNT(DISTINCT t.tag) = ?
    """
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, normalized_tags + [len(normalized_tags)])
        
        worlds = []
        for row in cursor.fetchall():
            world_id = row['id']
            world_name = row['name']
            
            # Get entries for this world
            cursor.execute("""
                SELECT uid, content, keys, secondary_keys, is_canon_law, 
                       probability, use_probability, depth, metadata, updated_at
                FROM world_entries
                WHERE world_id = ?
                ORDER BY sort_order, uid
            """, (world_id,))
            
            entries = {}
            for entry_row in cursor.fetchall():
                uid = entry_row['uid']
                
                # Parse keys
                keys = entry_row['keys'].split(',') if entry_row['keys'] else []
                secondary_keys = entry_row['secondary_keys'].split(',') if entry_row['secondary_keys'] else []
                
                # Parse metadata
                metadata = json.loads(entry_row['metadata']) if entry_row['metadata'] else {}
                
                entries[uid] = {
                    'uid': int(uid) if uid.isdigit() else uid,
                    'content': entry_row['content'],
                    'key': keys,
                    'keysecondary': secondary_keys,
                    'is_canon_law': bool(entry_row['is_canon_law']),
                    'probability': entry_row['probability'],
                    'useProbability': bool(entry_row['use_probability']),
                    'depth': entry_row['depth'],
                    **metadata
                }
            
            worlds.append({
                'name': world_name,
                'entries': entries
            })
        
        return worlds


# ============================================================================
# WORLD INFO OPERATIONS
# ============================================================================

def db_get_all_worlds() -> List[Dict[str, Any]]:
    """Get all world info including entries."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM worlds")
        
        worlds = []
        for row in cursor.fetchall():
            world_id = row['id']
            world_name = row['name']
            
            # Get entries for this world
            cursor.execute("""
                SELECT uid, content, keys, secondary_keys, is_canon_law, 
                       probability, use_probability, depth, metadata
                FROM world_entries
                WHERE world_id = ?
                ORDER BY sort_order, uid
            """, (world_id,))
            
            entries = {}
            for entry_row in cursor.fetchall():
                uid = entry_row['uid']
                
                # Parse keys
                keys = entry_row['keys'].split(',') if entry_row['keys'] else []
                secondary_keys = entry_row['secondary_keys'].split(',') if entry_row['secondary_keys'] else []
                
                # Parse metadata
                metadata = json.loads(entry_row['metadata']) if entry_row['metadata'] else {}
                
                entries[uid] = {
                    'uid': int(uid) if uid.isdigit() else uid,
                    'content': entry_row['content'],
                    'key': keys,
                    'keysecondary': secondary_keys,
                    'is_canon_law': bool(entry_row['is_canon_law']),
                    'probability': entry_row['probability'],
                    'useProbability': bool(entry_row['use_probability']),
                    'depth': entry_row['depth'],
                    **metadata  # Merge additional metadata
                }
            
            worlds.append({
                'name': world_name,
                'entries': entries
            })
        
        return worlds


def db_get_world(name: str) -> Optional[Dict[str, Any]]:
    """Get a specific world info by name."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM worlds WHERE name = ?", (name,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        world_id = row['id']
        
        # Get entries
        cursor.execute("""
            SELECT uid, content, keys, secondary_keys, is_canon_law,
                   probability, use_probability, depth, metadata, updated_at
            FROM world_entries
            WHERE world_id = ?
            ORDER BY sort_order, uid
        """, (world_id,))
        
        entries = {}
        for entry_row in cursor.fetchall():
            uid = entry_row['uid']
            
            # Parse keys
            keys = entry_row['keys'].split(',') if entry_row['keys'] else []
            secondary_keys = entry_row['secondary_keys'].split(',') if entry_row['secondary_keys'] else []
            
            # Parse metadata
            metadata = json.loads(entry_row['metadata']) if entry_row['metadata'] else {}
            
            entries[uid] = {
                'uid': int(uid) if uid.isdigit() else uid,
                'content': entry_row['content'],
                'key': keys,
                'keysecondary': secondary_keys,
                'is_canon_law': bool(entry_row['is_canon_law']),
                'probability': entry_row['probability'],
                'useProbability': bool(entry_row['use_probability']),
                'depth': entry_row['depth'],
                **metadata
            }
        
        return {'entries': entries}


def db_save_world(name: str, entries: Dict[str, Any], tags: Optional[List[str]] = None) -> bool:
    """Save or update a world info with all its entries.
    
    Note: This function deletes and re-inserts all entries. This automatically
    invalidates content hash (used for embedding cache invalidation) since
    db_get_world_content_hash() generates a hash based on current content.
    
    The semantic search engine uses this content hash to detect changes and
    will recompute embeddings when hash changes.
    
    Args:
        name: World name
        entries: Dictionary of world entries
        tags: Optional list of tags to save (v1.8.0+)
    """
    try:
        # Get old state before saving (for change logging)
        old_world = db_get_world(name)
        is_create = old_world is None
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert or get world
            cursor.execute("""
                INSERT OR IGNORE INTO worlds (name) VALUES (?)
            """, (name,))
            
            cursor.execute("SELECT id FROM worlds WHERE name = ?", (name,))
            world_id = cursor.fetchone()['id']
            
            # Delete existing entries for this world
            # This also invalidates any stale embeddings since the entry UIDs may change
            cursor.execute("DELETE FROM world_entries WHERE world_id = ?", (world_id,))
            
            # Delete any stale embeddings for this world
            try:
                cursor.execute("DELETE FROM vec_world_entries WHERE world_name = ?", (name,))
            except:
                pass  # sqlite-vec may not be available
            
            # Insert new entries
            entry_timestamp = int(time.time())
            for uid, entry in entries.items():
                keys = ','.join(entry.get('key', []))
                secondary_keys = ','.join(entry.get('keysecondary', []))
                
                # Extract standard fields
                content = entry.get('content', '')
                is_canon_law = entry.get('is_canon_law', False)
                probability = entry.get('probability', 100)
                use_probability = entry.get('useProbability', True)
                depth = entry.get('depth', 5)
                
                # Store remaining fields as metadata
                metadata_fields = {k: v for k, v in entry.items() if k not in [
                    'uid', 'content', 'key', 'keysecondary', 'is_canon_law',
                    'probability', 'useProbability', 'depth'
                ]}
                metadata_json = json.dumps(metadata_fields, ensure_ascii=False)
                
                cursor.execute("""
                    INSERT INTO world_entries 
                    (world_id, uid, content, keys, secondary_keys, is_canon_law,
                     probability, use_probability, depth, sort_order, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (world_id, uid, content, keys, secondary_keys, is_canon_law,
                      probability, use_probability, depth, int(uid) if uid.isdigit() else 999,
                      metadata_json, entry_timestamp))
            
            conn.commit()
            
            # Log the change (world info changes are always significant)
            operation = 'CREATE' if is_create else 'UPDATE'
            log_change('world_info', name, operation, old_world, {'entries': entries})
            
            # Automatic tag extraction (v1.8.0+)
            # Save tags to junction table if provided (used by import functions)
            if tags:
                # Normalize tags
                normalized_tags = []
                for tag in tags:
                    parsed = parse_tag_string(tag)
                    normalized_tags.extend(parsed)
                normalized_tags = [t for t in normalized_tags if t]
                
                # Clear existing tags (in case of update)
                db_remove_world_tags(name, [])
                
                # Add new tags
                if normalized_tags:
                    db_add_world_tags(name, normalized_tags)
            
            return True
    except Exception as e:
        print(f"Error saving world to database: {e}")
        return False


def db_delete_world(name: str) -> bool:
    """Delete a world and all its entries."""
    try:
        # Get old state before deleting (for change logging)
        old_world = db_get_world(name)
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM worlds WHERE name = ?", (name,))
            conn.commit()
            deleted = cursor.rowcount > 0
            
            # Log the deletion (world info changes are always significant)
            if deleted and old_world:
                log_change('world_info', name, 'DELETE', old_world, None)
            
            return deleted
    except Exception as e:
        print(f"Error deleting world from database: {e}")
        return False


def db_get_character_updated_at(filename: str) -> Optional[int]:
    """Get the updated_at timestamp for a character."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT updated_at FROM characters WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        return row['updated_at'] if row else None


def db_get_world_entry_timestamps(world_name: str) -> Dict[str, int]:
    """Get {uid: updated_at} for all entries in a world."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM worlds WHERE name = ?", (world_name,))
        row = cursor.fetchone()
        if not row:
            return {}
        
        world_id = row['id']
        cursor.execute("""
            SELECT uid, updated_at FROM world_entries WHERE world_id = ?
        """, (world_id,))
        
        return {entry_row['uid']: entry_row['updated_at'] for entry_row in cursor.fetchall()}


def db_update_world_entry(world_name: str, entry_uid: str, updates: Dict[str, Any]) -> bool:
    """Update specific fields in a world entry."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get world_id
            cursor.execute("SELECT id FROM worlds WHERE name = ?", (world_name,))
            row = cursor.fetchone()
            if not row:
                return False
            
            world_id = row['id']
            
            # Build update query dynamically
            update_fields = []
            values = []
            
            for field, value in updates.items():
                if field == 'content':
                    update_fields.append("content = ?")
                    values.append(value)
                elif field == 'key':
                    update_fields.append("keys = ?")
                    values.append(','.join(value) if isinstance(value, list) else value)
                elif field == 'is_canon_law':
                    update_fields.append("is_canon_law = ?")
                    values.append(1 if value else 0)
                elif field == 'probability':
                    update_fields.append("probability = ?")
                    values.append(int(value))
                elif field == 'useProbability':
                    update_fields.append("use_probability = ?")
                    values.append(1 if value else 0)
            
            if not update_fields:
                return False
            
            values.extend([world_id, entry_uid])
            
            query = f"UPDATE world_entries SET {', '.join(update_fields)} WHERE world_id = ? AND uid = ?"
            cursor.execute(query, values)
            
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating world entry: {e}")
        return False


# ============================================================================
# CHAT AND MESSAGE OPERATIONS
# ============================================================================

def db_get_all_chats() -> List[Dict[str, Any]]:
    """Get all chat sessions with metadata."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, branch_name, metadata, created_at
            FROM chats
            ORDER BY created_at DESC
        """)
        
        chats = []
        for row in cursor.fetchall():
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            chats.append({
                'id': row['id'],
                'branch_name': metadata.get('branch_name') or row['branch_name'],
                'created_at': row['created_at']
            })
        
        return chats


def db_get_chat(chat_id: str, include_summarized: bool = False) -> Optional[Dict[str, Any]]:
    """Get a specific chat with all its messages.
    
    Args:
        chat_id: Unique identifier for the chat
        include_summarized: If True, include archived/summarized messages. Default False.
    
    Returns:
        Chat data with messages, summary, and metadata, or None if not found
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, branch_name, summary, metadata, created_at
            FROM chats
            WHERE id = ?
        """, (chat_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Get messages - only active unless include_summarized is True
        if include_summarized:
            # Get all messages (both active and summarized)
            cursor.execute("""
                SELECT id, role, content, speaker, image_url, timestamp, summarized
                FROM messages
                WHERE chat_id = ?
                ORDER BY timestamp ASC
            """, (chat_id,))
        else:
            # Get only active (non-summarized) messages for normal use
            cursor.execute("""
                SELECT id, role, content, speaker, image_url, timestamp, summarized
                FROM messages
                WHERE chat_id = ? AND summarized = 0
                ORDER BY timestamp ASC
            """, (chat_id,))
        
        messages = []
        for msg_row in cursor.fetchall():
            messages.append({
                'id': msg_row['id'],
                'role': msg_row['role'],
                'content': msg_row['content'],
                'speaker': msg_row['speaker'],
                'image': msg_row['image_url'],
                'summarized': msg_row['summarized']
            })
        
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return {
            'messages': messages,
            'summary': row['summary'] or '',
            'activeCharacters': metadata.get('activeCharacters', []),
            'activeWI': metadata.get('activeWI'),
            'settings': metadata.get('settings', {}),
            'metadata': metadata
        }


def db_save_chat(chat_id: str, data: Dict[str, Any], autosaved: bool = True) -> Dict[str, int]:
    """Save or update a chat session with all messages using soft delete.
    
    Uses soft delete for archived messages: when messages are not included in the
    current save (e.g., after summarization), they are marked as summarized=1
    instead of being deleted. This preserves message history for search and analysis.
    
    Args:
        chat_id: Unique identifier for the chat
        data: Chat data including messages, summary, activeCharacters, etc.
        autosaved: Whether this is an autosaved chat (default: True)
    
    Returns:
        Dictionary mapping old message IDs to new database IDs for newly inserted messages
    """
    
    id_mapping = {}
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Load existing metadata when chat exists to preserve localnpcs
            cursor.execute("""
                SELECT id FROM chats WHERE id = ?
            """, (chat_id,))
            existing_chat = cursor.fetchone()
            
            if existing_chat:
                try:
                    cursor.execute("SELECT metadata FROM chats WHERE id = ?", (chat_id,))
                    row = cursor.fetchone()
                    if row and row['metadata']:
                        metadata = json.loads(row['metadata'])
                    else:
                        metadata = {}
                except (json.JSONDecodeError, TypeError) as e:
                    metadata = {}
            else:
                metadata = {}  # New chat

            # Update metadata with incoming data (preserve existing if not provided)
            metadata["activeCharacters"] = data.get("activeCharacters", metadata.get("activeCharacters", []))
            metadata["activeWI"] = data.get("activeWI", metadata.get("activeWI"))
            metadata["settings"] = data.get("settings", metadata.get("settings", {}))
            metadata["localnpcs"] = data.get("metadata", {}).get("localnpcs", metadata.get("localnpcs", {}))

            # Preserve backend-managed metadata (characterFirstTurns, characterCapsules)
            # These are set by /api/chat and should not be overwritten by frontend autosave
            incoming_metadata = data.get("metadata", {})
            if "characterFirstTurns" not in incoming_metadata:
                metadata["characterFirstTurns"] = metadata.get("characterFirstTurns", {})
            else:
                metadata["characterFirstTurns"] = incoming_metadata["characterFirstTurns"]

            if "characterCapsules" not in incoming_metadata:
                metadata["characterCapsules"] = metadata.get("characterCapsules", {})
            else:
                metadata["characterCapsules"] = incoming_metadata["characterCapsules"]

            # Critical: Ensure localnpcs exists (preserves NPCs from atomic creation)
            if "localnpcs" not in metadata:
                metadata["localnpcs"] = {}

            
            # Extract branch_name from metadata or data
            branch_name = metadata.get('branch_name') or data.get('branch_name')
            
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            summary = data.get('summary', '')
            timestamp = int(time.time())
            
            if existing_chat:
                # Update existing chat
                cursor.execute("""
                    UPDATE chats
                    SET branch_name = ?, summary = ?, metadata = ?, autosaved = ?
                    WHERE id = ?
                """, (branch_name, summary, metadata_json, autosaved, chat_id))
            else:
                # Insert new chat
                cursor.execute("""
                    INSERT INTO chats 
                    (id, branch_name, summary, metadata, created_at, autosaved)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chat_id, branch_name, summary, metadata_json, timestamp, autosaved))
            
            # Get all existing message IDs in database for this chat
            cursor.execute("""
                SELECT id, summarized FROM messages 
                WHERE chat_id = ?
            """, (chat_id,))
            
            existing_messages = {row['id']: row['summarized'] for row in cursor.fetchall()}
            
            # Get IDs from current save
            messages_to_save = data.get('messages', [])
            current_ids = set(m.get('id') for m in messages_to_save if m.get('id'))
            
            # Soft delete: Mark messages not in current save as summarized
            # This preserves history instead of deleting
            messages_to_archive = existing_messages.keys() - current_ids
            
            for msg_id in messages_to_archive:
                cursor.execute("""
                    UPDATE messages SET summarized = 1 WHERE id = ?
                """, (msg_id,))
            
 # Insert or update messages in current save
            for msg in messages_to_save:
                msg_id = msg.get('id')
                
                if msg_id and msg_id in existing_messages:
                    # Update existing message
                    cursor.execute("""
                        UPDATE messages
                        SET role = ?, content = ?, speaker = ?, image_url = ?, timestamp = ?, summarized = ?
                        WHERE id = ? AND chat_id = ?
                    """, (
                        msg.get('role', 'user'),
                        msg.get('content', ''),
                        msg.get('speaker'),
                        msg.get('image'),
                        msg.get('timestamp', int(time.time())),
                        0,  # Active messages are not summarized
                        msg_id,
                        chat_id
                    ))
                else:
                    # Insert new message - let database auto-generate ID
                    cursor.execute("""
                        INSERT INTO messages 
                        (chat_id, role, content, speaker, image_url, timestamp, summarized)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chat_id,
                        msg.get('role', 'user'),
                        msg.get('content', ''),
                        msg.get('speaker'),
                        msg.get('image'),
                        msg.get('timestamp', int(time.time())),
                        0  # Active messages are not summarized
                    ))
                    
                    # Track the ID mapping for newly inserted messages
                    if msg_id:
                        new_id = cursor.lastrowid
                        id_mapping[str(msg_id)] = new_id
            
            conn.commit()
            
            # Log if any messages were archived
            if messages_to_archive:
                print(f"Archived {len(messages_to_archive)} messages for chat {chat_id} (soft delete)")
            
            return id_mapping
    except Exception as e:
        print(f"Error saving chat to database (soft delete): {e}")
        return {}


def db_delete_chat(chat_id: str) -> bool:
    """Delete a chat and all its messages."""
    try:
        # Get old state before deleting (for change logging)
        # Chat deletion is a significant operation
        old_chat = db_get_chat(chat_id)
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            
            # Log the deletion (chat deletion is always significant)
            if deleted and old_chat:
                log_change('chat', chat_id, 'DELETE', old_chat, None)
            
            return deleted
    except Exception as e:
        print(f"Error deleting chat from database: {e}")
        return False


def db_cleanup_old_autosaved_chats(days: int = 7) -> int:
    """Delete autosaved chats older than specified days.
    
    Args:
        days: Number of days before a chat is considered old
    
    Returns:
        Number of chats deleted
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
            
            # Get list of chats to delete
            cursor.execute("""
                SELECT id FROM chats
                WHERE autosaved = 1 AND created_at < ?
            """, (cutoff_time,))
            
            chat_ids = [row['id'] for row in cursor.fetchall()]
            
            # Delete each chat
            deleted_count = 0
            for chat_id in chat_ids:
                if db_delete_chat(chat_id):
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old autosaved chats (older than {days} days)")
            
            return deleted_count
    except Exception as e:
        print(f"Error cleaning up old autosaved chats: {e}")
        return 0


def db_cleanup_empty_chats() -> int:
    """Delete chats with zero messages.
    
    Returns:
        Number of empty chats deleted
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Find chats with no messages
            cursor.execute("""
                SELECT c.id FROM chats c
                LEFT JOIN messages m ON c.id = m.chat_id
                WHERE m.chat_id IS NULL
            """)
            
            chat_ids = [row['id'] for row in cursor.fetchall()]
            
            # Delete each empty chat
            deleted_count = 0
            for chat_id in chat_ids:
                if db_delete_chat(chat_id):
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} empty chats")
            
            return deleted_count
    except Exception as e:
        print(f"Error cleaning up empty chats: {e}")
        return 0


# ============================================================================
# IMAGE METADATA OPERATIONS
# ============================================================================

def db_save_image_metadata(filename: str, params: Dict[str, Any]) -> bool:
    """Save image generation metadata."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO image_metadata
                (filename, prompt, negative_prompt, steps, cfg_scale, width, height,
                 sampler, scheduler, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                params.get('prompt', ''),
                params.get('negative_prompt', ''),
                params.get('steps', 20),
                params.get('cfg_scale', 7.0),
                params.get('width', 512),
                params.get('height', 512),
                params.get('sampler_name', 'Euler a'),
                params.get('scheduler', 'Automatic'),
                params.get('timestamp', int(time.time()))
            ))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving image metadata: {e}")
        return False


def db_get_image_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """Get image generation metadata."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT prompt, negative_prompt, steps, cfg_scale, width, height,
                   sampler, scheduler, timestamp
            FROM image_metadata
            WHERE filename = ?
        """, (filename,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'prompt': row['prompt'],
            'negative_prompt': row['negative_prompt'],
            'steps': row['steps'],
            'cfg_scale': row['cfg_scale'],
            'width': row['width'],
            'height': row['height'],
            'sampler_name': row['sampler'],
            'scheduler': row['scheduler'],
            'timestamp': row['timestamp']
        }


def db_get_all_image_metadata() -> Dict[str, Any]:
    """Get all image metadata as a dictionary."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, prompt, negative_prompt, steps, cfg_scale, 
                   width, height, sampler, scheduler, timestamp
            FROM image_metadata
            ORDER BY timestamp DESC
        """)
        
        images = {}
        for row in cursor.fetchall():
            images[row['filename']] = {
                'prompt': row['prompt'],
                'negative_prompt': row['negative_prompt'],
                'steps': row['steps'],
                'cfg_scale': row['cfg_scale'],
                'width': row['width'],
                'height': row['height'],
                'sampler_name': row['sampler'],
                'scheduler': row['scheduler'],
                'timestamp': row['timestamp']
            }
        
        return {'images': images}


# ============================================================================
# SQLITE-VEC EMBEDDING OPERATIONS (Disk-based, SIMD-accelerated)
# ============================================================================

def _serialize_float32(vector: np.ndarray) -> bytes:
    """Serialize a float32 numpy vector to bytes."""
    return vector.astype(np.float32).tobytes()


def _deserialize_float32(data: bytes, dimensions: int) -> np.ndarray:
    """Deserialize bytes back to float32 numpy vector."""
    return np.frombuffer(data, dtype=np.float32).reshape(dimensions)


def init_vec_table():
    """Initialize vec0 virtual table for embeddings if sqlite-vec is available."""
    try:
        import sqlite_vec
        
        with get_connection() as conn:
            # Load sqlite-vec extension
            sqlite_vec.load(conn)
            
            # Create virtual table for world entry embeddings if it doesn't exist
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_world_entries USING vec0(
                    entry_id INTEGER PRIMARY KEY,
                    world_name TEXT,
                    entry_uid TEXT,
                    embedding FLOAT[768]
                )
            """)
            conn.commit()
            return True
    except ImportError:
        return False
    except Exception as e:
        return False


def db_save_embedding(world_name: str, entry_uid: str, embedding: np.ndarray) -> bool:
    """Save an embedding vector to sqlite-vec (disk-based storage)."""
    try:
        import sqlite_vec
        
        # Validate embedding dimensions before saving
        if len(embedding) != EXPECTED_EMBEDDING_DIMENSIONS:
            return False
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            # Serialize the embedding
            embedding_bytes = _serialize_float32(embedding)
            
            # Insert or replace embedding
            conn.execute("""
                INSERT OR REPLACE INTO vec_world_entries (world_name, entry_uid, embedding)
                VALUES (?, ?, ?)
            """, (world_name, entry_uid, embedding_bytes))
            
            conn.commit()
            return True
    except ImportError:
        return False
    except Exception as e:
        return False


def db_delete_entry_embedding(world_name: str, entry_uid: str) -> bool:
    """Delete a specific embedding for a world entry from sqlite-vec."""
    try:
        import sqlite_vec
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM vec_world_entries 
                WHERE world_name = ? AND entry_uid = ?
            """, (world_name, entry_uid))
            
            conn.commit()
            deleted = cursor.rowcount > 0
            return deleted
    except ImportError:
        return False
    except Exception as e:
        return False


def db_get_embedding(world_name: str, entry_uid: str) -> Optional[np.ndarray]:
    """Get an embedding vector from sqlite-vec."""
    try:
        import sqlite_vec
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding FROM vec_world_entries
                WHERE world_name = ? AND entry_uid = ?
            """, (world_name, entry_uid))
            
            row = cursor.fetchone()
            if row and row['embedding']:
                # Deserialize the embedding (768 dimensions for all-mpnet-base-v2)
                return _deserialize_float32(row['embedding'], 768)
            
            return None
    except ImportError:
        return None
    except Exception as e:
        return None


# Expected embedding dimensions for all-mpnet-base-v2
EXPECTED_EMBEDDING_DIMENSIONS = 768


def db_search_similar_embeddings(world_name: str, query_embedding: np.ndarray, 
                                   k: int = 10, threshold: float = 0.3) -> List[Tuple[str, float]]:
    """
    Search for similar embeddings using sqlite-vec's SIMD-accelerated KNN search.
    Returns list of (entry_uid, similarity_score) tuples.
    """
    try:
        import sqlite_vec
        
        # Validate embedding dimensions
        if len(query_embedding) != EXPECTED_EMBEDDING_DIMENSIONS:
            return []
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            # Serialize query embedding
            query_bytes = _serialize_float32(query_embedding)
            
            cursor = conn.cursor()
            # Use vec_distance_cosine for cosine similarity (1 - cosine_distance = similarity)
            # Note: Cannot use column aliases in WHERE clause, so we use a subquery
            cursor.execute("""
                SELECT entry_uid, similarity
                FROM (
                    SELECT entry_uid, 
                           (1 - vec_distance_cosine(embedding, ?)) as similarity
                    FROM vec_world_entries
                    WHERE world_name = ?
                ) AS subquery
                WHERE similarity >= ?
                ORDER BY similarity DESC
                LIMIT ?
            """, (query_bytes, world_name, threshold, k))
            
            results = []
            for row in cursor.fetchall():
                results.append((row['entry_uid'], row['similarity']))
            
            return results
    except ImportError:
        return []
    except Exception as e:
        return []


def db_delete_world_embeddings(world_name: str) -> bool:
    """Delete all embeddings for a specific world."""
    try:
        import sqlite_vec
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            conn.execute("""
                DELETE FROM vec_world_entries WHERE world_name = ?
            """, (world_name,))
            
            conn.commit()
            return True
    except ImportError:
        return False
    except Exception as e:
        return False


def db_count_embeddings(world_name: Optional[str] = None) -> int:
    """Count embeddings in the database, optionally filtered by world."""
    try:
        import sqlite_vec
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            cursor = conn.cursor()
            if world_name:
                cursor.execute("""
                    SELECT COUNT(*) as count FROM vec_world_entries WHERE world_name = ?
                """, (world_name,))
            else:
                cursor.execute("SELECT COUNT(*) as count FROM vec_world_entries")
            
            row = cursor.fetchone()
            return row['count'] if row else 0
    except ImportError:
        return 0
    except Exception as e:
        return 0


# ============================================================================
# PERFORMANCE METRICS OPERATIONS
# ============================================================================

def db_save_performance_metric(operation_type: str, duration: float, context_tokens: int = 0) -> bool:
    """Save a performance metric to the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics 
                (operation_type, duration, context_tokens, timestamp)
                VALUES (?, ?, ?, ?)
            """, (operation_type, duration, context_tokens, int(time.time())))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving performance metric: {e}")
        return False


def db_get_recent_performance_metrics(operation_type: str, limit: int = 10) -> List[float]:
    """Get recent performance metrics for an operation type."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT duration
            FROM performance_metrics
            WHERE operation_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (operation_type, limit))
        
        return [row['duration'] for row in cursor.fetchall()]


def db_get_median_performance(operation_type: str, limit: int = 10) -> Optional[float]:
    """Get median performance for an operation type from recent metrics."""
    durations = db_get_recent_performance_metrics(operation_type, limit)
    if not durations:
        return None
    
    # Calculate median
    sorted_durations = sorted(durations)
    n = len(sorted_durations)
    if n % 2 == 0:
        return (sorted_durations[n // 2 - 1] + sorted_durations[n // 2]) / 2
    else:
        return sorted_durations[n // 2]


def db_cleanup_old_metrics(days: int = 7) -> bool:
    """Clean up performance metrics older than specified days."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
            cursor.execute("""
                DELETE FROM performance_metrics
                WHERE timestamp < ?
            """, (cutoff_time,))
            conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old performance metrics")
            return True
    except Exception as e:
        print(f"Error cleaning up performance metrics: {e}")
        return False


# ============================================================================
# CHANGE LOG OPERATIONS (v1.5.1 - Undo/Redo Foundation)
# ============================================================================

# Significant chat operations that warrant logging (not every message save)
SIGNIFICANT_CHAT_OPERATIONS = {
    'DELETE',
    'CREATE_BRANCH',
    'RENAME',
    'CLEAR',
    'ADD_CHARACTER',
    'REMOVE_CHARACTER',
    'MANUAL_SUMMARIZE'
}


def should_log_chat_change(operation: str) -> bool:
    """Determine if a chat operation is significant enough to log.
    
    We don't log every message save (too frequent), only structural changes
    that users might want to undo.
    """
    return operation in SIGNIFICANT_CHAT_OPERATIONS


def log_change(entity_type: str, entity_id: str, operation: str, 
               old_data: Any = None, new_data: Any = None) -> bool:
    """Log a change for undo/redo support.
    
    Args:
        entity_type: 'character', 'world_info', or 'chat'
        entity_id: Character filename, world name, or chat ID
        operation: 'CREATE', 'UPDATE', 'DELETE', or significant chat operations
        old_data: JSON-serializable snapshot before change (None for CREATE)
        new_data: JSON-serializable snapshot after change (None for DELETE)
    
    Returns:
        True if logged successfully, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO change_log (entity_type, entity_id, operation, old_data, new_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entity_type, 
                entity_id, 
                operation,
                json.dumps(old_data, ensure_ascii=False) if old_data else None,
                json.dumps(new_data, ensure_ascii=False) if new_data else None
            ))
            conn.commit()
            print(f"Change logged: {operation} {entity_type}/{entity_id}")
            return True
    except Exception as e:
        print(f"Error logging change: {e}")
        return False


def get_recent_changes(entity_type: Optional[str] = None, 
                       entity_id: Optional[str] = None,
                       limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent changes for debugging/undo.
    
    Args:
        entity_type: Filter by type ('character', 'world_info', 'chat') or None for all
        entity_id: Filter by specific entity ID or None for all of type
        limit: Maximum number of changes to return
    
    Returns:
        List of change records with id, entity_type, entity_id, operation,
        old_data (parsed), new_data (parsed), and timestamp
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        if entity_type and entity_id:
            cursor.execute("""
                SELECT id, entity_type, entity_id, operation, old_data, new_data, timestamp
                FROM change_log 
                WHERE entity_type = ? AND entity_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (entity_type, entity_id, limit))
        elif entity_type:
            cursor.execute("""
                SELECT id, entity_type, entity_id, operation, old_data, new_data, timestamp
                FROM change_log 
                WHERE entity_type = ? 
                ORDER BY timestamp DESC LIMIT ?
            """, (entity_type, limit))
        else:
            cursor.execute("""
                SELECT id, entity_type, entity_id, operation, old_data, new_data, timestamp
                FROM change_log 
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'entity_type': row['entity_type'],
                'entity_id': row['entity_id'],
                'operation': row['operation'],
                'old_data': json.loads(row['old_data']) if row['old_data'] else None,
                'new_data': json.loads(row['new_data']) if row['new_data'] else None,
                'timestamp': row['timestamp']
            })
        
        return results


def get_last_change(entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent change for a specific entity.
    
    Useful for implementing undo functionality.
    """
    changes = get_recent_changes(entity_type, entity_id, limit=1)
    return changes[0] if changes else None


def db_cleanup_old_changes(days: int = 30) -> bool:
    """Clean up change log entries older than specified days.
    
    Default is 30 days - a reasonable undo window without unbounded growth.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
            cursor.execute("""
                DELETE FROM change_log
                WHERE timestamp < ?
            """, (cutoff_time,))
            conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old change log entries")
            return True
    except Exception as e:
        print(f"Error cleaning up change log: {e}")
        return False


def mark_change_undone(change_id: int) -> bool:
    """Mark a change log entry as undone to prevent re-undo.
    
    Args:
        change_id: ID of the change to mark as undone
    
    Returns:
        True if marked successfully, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE change_log
                SET undo_state = 'undone'
                WHERE id = ?
            """, (change_id,))
            conn.commit()
            marked = cursor.rowcount > 0
            if marked:
                print(f"Marked change {change_id} as undone")
            return marked
    except Exception as e:
        print(f"Error marking change as undone: {e}")
        return False


def undo_last_delete(entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
    """Undo the last DELETE operation for an entity.
    
    Finds the most recent DELETE operation for the specified entity and restores
    it from the old_data snapshot.
    
    Args:
        entity_type: Type of entity ('character', 'world_info', or 'chat')
        entity_id: ID of the entity (filename, name, or chat_id)
    
    Returns:
        Dictionary with restoration details or None if no DELETE found
    """
    try:
        # Find most recent DELETE operation that hasn't been undone
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, old_data, timestamp
                FROM change_log
                WHERE entity_type = ? 
                  AND entity_id = ? 
                  AND operation = 'DELETE' 
                  AND undo_state = 'active'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (entity_type, entity_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            change_id = row['id']
            old_data = json.loads(row['old_data']) if row['old_data'] else None
            
            if not old_data:
                return None
            
            # Restore based on entity type
            if entity_type == 'character':
                # Restore character from old_data
                filename = entity_id
                if db_save_character(old_data, filename):
                    mark_change_undone(change_id)
                    return {
                        'success': True,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'restored_name': old_data.get('data', {}).get('name', 'Unknown'),
                        'change_id': change_id
                    }
            
            elif entity_type == 'world_info':
                # Restore world info from old_data
                world_name = entity_id
                entries = old_data.get('entries', {})
                if db_save_world(world_name, entries):
                    mark_change_undone(change_id)
                    return {
                        'success': True,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'restored_name': world_name,
                        'change_id': change_id
                    }
            
            elif entity_type == 'chat':
                # Restore chat from old_data
                chat_id = entity_id
                if db_save_chat(chat_id, old_data):
                    mark_change_undone(change_id)
                    return {
                        'success': True,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'restored_name': chat_id,
                        'change_id': change_id
                    }
            
            return None
            
    except Exception as e:
        print(f"Error undoing last delete: {e}")
        return None


def restore_version(change_id: int) -> Optional[Dict[str, Any]]:
    """Restore an entity to the state it was in at a specific change.
    
    Args:
        change_id: ID of the change to restore to
    
    Returns:
        Dictionary with restoration details or None if change not found
    """
    try:
        # Get the change record
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_type, entity_id, operation, old_data, new_data, timestamp
                FROM change_log
                WHERE id = ?
            """, (change_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            entity_type = row['entity_type']
            entity_id = row['entity_id']
            operation = row['operation']
            restore_data = json.loads(row['old_data']) if row['old_data'] else None
            
            if not restore_data:
                return None
            
            # Restore based on entity type and operation
            if entity_type == 'character':
                filename = entity_id
                if db_save_character(restore_data, filename):
                    # Log new change
                    log_change('character', filename, 'RESTORE', None, restore_data)
                    return {
                        'success': True,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'restored_name': restore_data.get('data', {}).get('name', 'Unknown'),
                        'change_id': change_id
                    }
            
            elif entity_type == 'world_info':
                world_name = entity_id
                entries = restore_data.get('entries', {})
                if db_save_world(world_name, entries):
                    # Log new change
                    log_change('world_info', world_name, 'RESTORE', None, restore_data)
                    return {
                        'success': True,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'restored_name': world_name,
                        'change_id': change_id
                    }
            
            elif entity_type == 'chat':
                chat_id = entity_id
                if db_save_chat(chat_id, restore_data):
                    # Log new change
                    log_change('chat', chat_id, 'RESTORE', None, restore_data)
                    return {
                        'success': True,
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'restored_name': chat_id,
                        'change_id': change_id
                    }
            
            return None
            
    except Exception as e:
        print(f"Error restoring version: {e}")
        return None


# ============================================================================
# WORLD INFO HASH OPERATIONS
# ============================================================================

def db_get_world_entry_hash(world_name: str, entry_uid: str) -> Optional[str]:
    """Get the content hash for a specific world entry."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Get world_id
        cursor.execute("SELECT id FROM worlds WHERE name = ?", (world_name,))
        row = cursor.fetchone()
        if not row:
            return None
        
        world_id = row['id']
        
        # Get embedding hash
        cursor.execute("""
            SELECT embedding_hash
            FROM world_entries
            WHERE world_id = ? AND uid = ?
        """, (world_id, entry_uid))
        
        row = cursor.fetchone()
        if row and row['embedding_hash']:
            return row['embedding_hash']
        
        return None


def db_update_world_entry_hash(world_name: str, entry_uid: str, content_hash: str) -> bool:
    """Update the content hash for a world entry."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get world_id
            cursor.execute("SELECT id FROM worlds WHERE name = ?", (world_name,))
            row = cursor.fetchone()
            if not row:
                return False
            
            world_id = row['id']
            
            # Update hash and timestamp
            cursor.execute("""
                UPDATE world_entries
                SET embedding_hash = ?, last_embedded_at = ?
                WHERE world_id = ? AND uid = ?
            """, (content_hash, int(time.time()), world_id, entry_uid))
            
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating world entry hash: {e}")
        return False


def db_get_world_content_hash(world_name: str) -> str:
    """Generate a hash representing the entire world's content for cache invalidation."""
    import hashlib
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Get world_id
        cursor.execute("SELECT id FROM worlds WHERE name = ?", (world_name,))
        row = cursor.fetchone()
        if not row:
            return "empty"
        
        world_id = row['id']
        
        # Get all entries ordered by uid for consistent hashing
        cursor.execute("""
            SELECT uid, content, keys, secondary_keys
            FROM world_entries
            WHERE world_id = ?
            ORDER BY uid
        """, (world_id,))
        
        # Build hash string
        hash_content = ""
        for row in cursor.fetchall():
            hash_content += f"{row['uid']}|{row['content']}|{row['keys']}|{row['secondary_keys']}|"
        
        if not hash_content:
            return "empty"
        
        return hashlib.md5(hash_content.encode('utf-8')).hexdigest()


# ============================================================================
# SOFT DELETE HELPER FUNCTIONS
# ============================================================================

def db_cleanup_old_summarized_messages(days: int = 90) -> int:
    """Permanently delete summarized messages older than specified days.
    
    This is optional cleanup for users who want to reclaim disk space.
    
    Args:
        days: Number of days before a summarized message is deleted
    
    Returns:
        Number of messages deleted
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
            
            cursor.execute("""
                DELETE FROM messages 
                WHERE summarized = 1 AND timestamp < ?
            """, (cutoff_time,))
            
            conn.commit()
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old summarized messages (older than {days} days)")
            
            return deleted_count
    except Exception as e:
        print(f"Error cleaning up old summarized messages: {e}")
        return 0


def db_get_summarized_message_count() -> int:
    """Get count of summarized messages in database.
    
    Returns:
        Number of summarized messages
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM messages WHERE summarized = 1")
            row = cursor.fetchone()
            return row['count'] if row else 0
    except Exception as e:
        print(f"Error counting summarized messages: {e}")
        return 0


def db_search_messages_with_summarized(query: str, chat_id: Optional[str] = None,
                                    speaker: Optional[str] = None,
                                    include_summarized: bool = True,
                                    limit: int = 50) -> List[Dict[str, Any]]:
    """Search messages across active and optionally summarized messages.
    
    Args:
        query: Search query string
        chat_id: Optional filter by specific chat
        speaker: Optional filter by speaker name
        include_summarized: If True, include summarized messages in search
        limit: Maximum number of results to return
    
    Returns:
        List of message dictionaries with match metadata
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Build WHERE clause
            conditions = []
            params = []
            
            # Include summarized or not
            if include_summarized:
                # Search all messages (active + summarized)
                conditions.append("1=1")  # Always true
            else:
                # Search only active messages
                conditions.append("summarized = 0")
            
            if chat_id:
                conditions.append("chat_id = ?")
                params.append(chat_id)
            
            if speaker:
                conditions.append("speaker = ?")
                params.append(speaker)
            
            where_clause = " AND ".join(conditions)
            params.append(f"%{query}%")
            params.append(limit)
            
            cursor.execute(f"""
                SELECT m.id, m.chat_id, m.role, m.content, m.speaker, 
                       m.timestamp, m.summarized, c.branch_name
                FROM messages m
                LEFT JOIN chats c ON m.chat_id = c.id
                WHERE {where_clause} AND m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'chat_id': row['chat_id'],
                    'role': row['role'],
                    'content': row['content'],
                    'speaker': row['speaker'],
                    'timestamp': row['timestamp'],
                    'summarized': row['summarized'],
                    'branch_name': row['branch_name']
                })
            
            return results
    except Exception as e:
        print(f"Error searching messages (soft delete): {e}")
        return []


# ============================================================================
# FTS5 FULL-TEXT SEARCH OPERATIONS
# ============================================================================

def init_fts_table() -> bool:
    """Initialize FTS5 virtual table for message search."""
    try:
        with get_connection() as conn:
            # Create FTS5 virtual table for messages
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id UNINDEXED,
                    chat_id UNINDEXED,
                    content,
                    speaker,
                    role,
                    timestamp UNINDEXED,
                    tokenize='porter unicode61'
                )
            """)
            
            # Create triggers to keep FTS5 in sync with messages table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages
                BEGIN
                    INSERT INTO messages_fts (message_id, chat_id, content, speaker, role, timestamp)
                    VALUES (NEW.id, NEW.chat_id, NEW.content, NEW.speaker, NEW.role, NEW.timestamp)
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages
                BEGIN
                    DELETE FROM messages_fts WHERE message_id = OLD.id
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages
                BEGIN
                    DELETE FROM messages_fts WHERE message_id = OLD.id
                    INSERT INTO messages_fts (message_id, chat_id, content, speaker, role, timestamp)
                    VALUES (NEW.id, NEW.chat_id, NEW.content, NEW.speaker, NEW.role, NEW.timestamp)
                END
            """)
            
            conn.commit()
            print("FTS5 search table initialized successfully")
            return True
    except Exception as e:
        print(f"Error initializing FTS5 table: {e}")
        return False


def migrate_populate_fts() -> None:
    """One-time migration to populate FTS5 index with existing messages.
    
    Should be called on startup if FTS5 table is empty.
    """
    try:
        with get_connection() as conn:
            # Check if FTS5 table has any data
            cursor = conn.execute("SELECT COUNT(*) as count FROM messages_fts").fetchone()
            if cursor['count'] > 0:
                print("FTS5 table already populated, skipping migration")
                return
            
            # Populate FTS5 with all existing messages
            cursor.execute("""
                INSERT INTO messages_fts (message_id, chat_id, content, speaker, role, timestamp)
                SELECT id, chat_id, content, speaker, role, timestamp
                FROM messages
            """)
            conn.commit()
            
            # Get count of migrated messages
            migrated_count = conn.execute("SELECT COUNT(*) as count FROM messages_fts").fetchone()['count']
            print(f"FTS5 migration complete: {migrated_count} messages indexed")
    except Exception as e:
        print(f"Error populating FTS5 index: {e}")


def db_search_messages(query: str, chat_id: Optional[str] = None,
                       speaker: Optional[str] = None,
                       start_timestamp: Optional[int] = None,
                       end_timestamp: Optional[int] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
    """Search messages using FTS5 full-text search.
    
    Args:
        query: Search query string (FTS5 syntax supported)
        chat_id: Optional filter by specific chat
        speaker: Optional filter by speaker name
        start_timestamp: Optional filter by date range (Unix timestamp)
        end_timestamp: Optional filter by date range (Unix timestamp)
        limit: Maximum number of results to return (default: 50)
    
    Returns:
        List of message dictionaries with search rank
    
    FTS5 query syntax:
    - Simple: "flame sword" (contains both words)
    - Phrase: '"flame sword"' (exact phrase match)
    - Boolean: "flame AND sword", "flame OR sword"
    - Exclude: "flame NOT ice"
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Build WHERE clause with optional filters
            conditions = ["messages_fts MATCH ?"]
            params = [query]
            
            if chat_id:
                conditions.append("chat_id = ?")
                params.append(chat_id)
            
            if speaker:
                conditions.append("speaker = ?")
                params.append(speaker)
            
            if start_timestamp:
                conditions.append("timestamp >= ?")
                params.append(start_timestamp)
            
            if end_timestamp:
                conditions.append("timestamp <= ?")
                params.append(end_timestamp)
            
            where_clause = " AND ".join(conditions)
            params.append(limit)
            
            # Execute search with rank for relevance ordering
            cursor.execute(f"""
                SELECT m.id, m.chat_id, m.role, m.content, m.speaker, 
                       m.timestamp, m.summarized, c.branch_name, fts.rank
                FROM messages_fts fts
                JOIN messages m ON fts.message_id = m.id
                LEFT JOIN chats c ON m.chat_id = c.id
                WHERE {where_clause}
                ORDER BY fts.rank, m.timestamp DESC
                LIMIT ?
            """, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'chat_id': row['chat_id'],
                    'role': row['role'],
                    'content': row['content'],
                    'speaker': row['speaker'],
                    'timestamp': row['timestamp'],
                    'summarized': row['summarized'],
                    'branch_name': row['branch_name'],
                    'rank': row['rank']  # FTS5 relevance score
                })
            
            return results
    except Exception as e:
        print(f"Error searching messages with FTS5: {e}")
        return []


def db_get_message_context(message_id: int, context_size: int = 2) -> Optional[Dict[str, Any]]:
    """Get a message with surrounding context.
    
    Args:
        message_id: ID of target message
        context_size: Number of messages before and after (default: 2)
    
    Returns:
        Dictionary with:
        - 'target': Target message
        - 'before': List of messages before target
        - 'after': List of messages after target
        - 'chat_id': Chat ID for navigation
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get target message info first
            cursor.execute("""
                SELECT id, chat_id, role, content, speaker, timestamp, summarized
                FROM messages
                WHERE id = ?
            """, (message_id,))
            
            target = cursor.fetchone()
            if not target:
                return None
            
            # Get chat_id for context query
            chat_id = target['chat_id']
            
            # Get messages before target (exclude target)
            cursor.execute("""
                SELECT id, role, content, speaker, timestamp, summarized
                FROM messages
                WHERE chat_id = ? AND id < ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (chat_id, message_id, context_size))
            
            before_messages = []
            for row in cursor.fetchall():
                before_messages.append({
                    'id': row['id'],
                    'role': row['role'],
                    'content': row['content'],
                    'speaker': row['speaker'],
                    'timestamp': row['timestamp'],
                    'summarized': row['summarized']
                })
            
            # Reverse to chronological order (oldest first)
            before_messages.reverse()
            
            # Get messages after target (exclude target)
            cursor.execute("""
                SELECT id, role, content, speaker, timestamp, summarized
                FROM messages
                WHERE chat_id = ? AND id > ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (chat_id, message_id, context_size))
            
            after_messages = []
            for row in cursor.fetchall():
                after_messages.append({
                    'id': row['id'],
                    'role': row['role'],
                    'content': row['content'],
                    'speaker': row['speaker'],
                    'timestamp': row['timestamp'],
                    'summarized': row['summarized']
                })
            
            return {
                'target': {
                    'id': target['id'],
                    'role': target['role'],
                    'content': target['content'],
                    'speaker': target['speaker'],
                    'timestamp': target['timestamp'],
                    'summarized': target['summarized']
                },
                'before': before_messages,
                'after': after_messages,
                'chat_id': chat_id
            }
    except Exception as e:
        print(f"Error getting message context: {e}")
        return None


def db_get_available_speakers() -> List[str]:
    """Get all unique speaker names from all messages.
    
    Returns:
        List of unique speaker names sorted alphabetically
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all unique, non-null speakers
            cursor.execute("""
                SELECT DISTINCT speaker
                FROM messages
                WHERE speaker IS NOT NULL AND speaker != ''
                ORDER BY speaker ASC
            """)
            
            speakers = [row['speaker'] for row in cursor.fetchall()]
            return speakers
    except Exception as e:
        print(f"Error getting available speakers: {e}")
        return []


# ============================================================================
# DATABASE HEALTH CHECK
# ============================================================================

def verify_database_health() -> bool:
    """Run on startup to catch corruption early.
    
    Performs SQLite integrity check and verifies core tables exist.
    Returns True if healthy, False if issues detected.
    """
    try:
        with get_connection() as conn:
            # SQLite integrity check (fast on small DBs)
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] != "ok":
                print(f"⚠️  Database corruption detected: {result[0]}")
                print("   Consider running migrate_to_sqlite.py to rebuild from JSON backup")
                return False
            
            # Quick check core tables exist
            cursor = conn.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name IN 
                ('characters', 'worlds', 'world_entries', 'chats', 'messages')
            """)
            table_count = cursor.fetchone()[0]
            if table_count != 5:
                print(f"⚠️  Missing core tables ({table_count}/5 found) - database may need reinitialization")
                return False
            
            print("Database integrity verified")
            return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False


# ============================================================================
# RELATIONSHIP STATE OPERATIONS
# ============================================================================

def db_get_relationship_state(chat_id: str, entity_from: str, entity_to: str) -> Optional[Dict[str, Any]]:
    """Get relationship state for a specific directional pair using entity IDs.
    
    Args:
        chat_id: Chat ID
        entity_from: Entity ID for source entity
        entity_to: Entity ID for target entity
    
    Returns:
        Relationship state dict or None if not found
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM relationship_states
            WHERE chat_id = ? AND character_from = ? AND character_to = ?
        ''', (chat_id, entity_from, entity_to))
        row = cursor.fetchone()
        return dict(row) if row else None


def db_get_relationship_state_by_names(chat_id: str, name_from: str, name_to: str) -> Optional[Dict[str, Any]]:
    """Get relationship state by character names (converts to entity IDs internally).
    
    Args:
        chat_id: Chat ID
        name_from: Character name for source
        name_to: Character name for target
    
    Returns:
        Relationship state dict or None if not found
    """
    entity_from = db_get_entity_id_by_name(chat_id, name_from, 'character')
    entity_to = db_get_entity_id_by_name(chat_id, name_to, 'character')
    
    if not entity_from or not entity_to:
        return None
    
    return db_get_relationship_state(chat_id, entity_from, entity_to)


def db_get_all_relationship_states(chat_id: str) -> List[Dict[str, Any]]:
    """Get all relationship states for a chat."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM relationship_states WHERE chat_id = ?
            ORDER BY last_updated DESC
        ''', (chat_id,))
        return [dict(row) for row in cursor.fetchall()]


def db_update_relationship_state(chat_id: str, character_from: str, character_to: str, 
                             scores: dict, last_message_id: int) -> bool:
    """Update or insert relationship state with history tracking."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            existing = db_get_relationship_state(chat_id, character_from, character_to)
            
            # Update history (keep last 20 snapshots)
            history = []
            if existing and existing.get('history'):
                history = json.loads(existing['history'])
            
            history.append({
                'message_id': last_message_id,
                'timestamp': time.time(),
                'trust': scores['trust'],
                'emotional_bond': scores['emotional_bond'],
                'conflict': scores['conflict'],
                'power_dynamic': scores['power_dynamic'],
                'fear_anxiety': scores['fear_anxiety']
            })
            history = history[-20:]  # Keep only last 20
            
            if existing:
                cursor.execute('''
                    UPDATE relationship_states
                    SET trust = ?, emotional_bond = ?, conflict = ?, 
                        power_dynamic = ?, fear_anxiety = ?,
                        last_updated = ?, last_analyzed_message_id = ?,
                        interaction_count = interaction_count + 1,
                        history = ?
                    WHERE chat_id = ? AND character_from = ? AND character_to = ?
                ''', (
                    scores['trust'], scores['emotional_bond'], scores['conflict'],
                    scores['power_dynamic'], scores['fear_anxiety'],
                    time.time(), last_message_id, json.dumps(history),
                    chat_id, character_from, character_to
                ))
            else:
                cursor.execute('''
                    INSERT INTO relationship_states
                    (chat_id, character_from, character_to, trust, emotional_bond, 
                     conflict, power_dynamic, fear_anxiety, last_updated, 
                     last_analyzed_message_id, interaction_count, history)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ''', (
                    chat_id, character_from, character_to,
                    scores['trust'], scores['emotional_bond'], scores['conflict'],
                    scores['power_dynamic'], scores['fear_anxiety'],
                    time.time(), last_message_id, json.dumps(history)
                ))
            
            conn.commit()
            return True
    except Exception as e:
        return False


def db_copy_relationship_states_for_branch(origin_chat_id: str, branch_chat_id: str, 
                                       fork_message_id: int) -> int:
    """Copy relationship states to branch at fork-point state."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT character_from, character_to, 
                       trust, emotional_bond, conflict, power_dynamic, fear_anxiety,
                       history
                FROM relationship_states
                WHERE chat_id = ?
            ''', (origin_chat_id,))
            
            origin_states = cursor.fetchall()
            copied_count = 0
            
            for state in origin_states:
                history = json.loads(state['history']) if state['history'] else []
                
                # Find snapshot at or before fork point
                fork_point_state = None
                for snapshot in history:
                    if snapshot.get('message_id', 0) <= fork_message_id:
                        fork_point_state = snapshot
                    else:
                        break
                
                # Use fork point state or current state
                if fork_point_state:
                    values = (
                        fork_point_state['trust'],
                        fork_point_state['emotional_bond'],
                        fork_point_state['conflict'],
                        fork_point_state['power_dynamic'],
                        fork_point_state['fear_anxiety']
                    )
                else:
                    values = (
                        state['trust'], state['emotional_bond'], state['conflict'],
                        state['power_dynamic'], state['fear_anxiety']
                    )
                
                new_history = [{
                    'message_id': fork_message_id,
                    'timestamp': time.time(),
                    'trust': values[0],
                    'emotional_bond': values[1],
                    'conflict': values[2],
                    'power_dynamic': values[3],
                    'fear_anxiety': values[4],
                    'note': 'Branch point'
                }]
                
                cursor.execute('''
                    INSERT INTO relationship_states
                    (chat_id, character_from, character_to, 
                     trust, emotional_bond, conflict, power_dynamic, fear_anxiety,
                     last_updated, last_analyzed_message_id, interaction_count, history)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ''', (
                    branch_chat_id, state['character_from'], state['character_to'],
                    *values, time.time(), fork_message_id, json.dumps(new_history)
                ))
                copied_count +=1
            
            conn.commit()
            return copied_count
    except Exception as e:
        return 0


def db_copy_relationship_states_with_mapping(
    origin_chat_id: str, 
    branch_chat_id: str,
    fork_message_id: int,
    entity_mapping: Dict[str, str]
) -> int:
    """
    Copy relationship states to branch with entity ID remapping.
    
    Args:
        origin_chat_id: Original chat ID
        branch_chat_id: New branch chat ID
        fork_message_id: Message ID where fork occurred
        entity_mapping: Maps old_entity_id -> new_entity_id for NPCs
    
    Returns:
        Number of relationships copied
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all relationships from origin chat
            cursor.execute('''
                SELECT character_from, character_to, 
                       trust, emotional_bond, conflict, power_dynamic, fear_anxiety,
                       history, interaction_count
                FROM relationship_states
                WHERE chat_id = ?
            ''', (origin_chat_id,))
            
            origin_states = cursor.fetchall()
            copied_count = 0
            
            for state in origin_states:
                # Remap entity IDs using mapping (NPCs get new IDs, characters stay same)
                from_entity = entity_mapping.get(state['character_from'], 
                                                 state['character_from'])
                to_entity = entity_mapping.get(state['character_to'], 
                                               state['character_to'])
                
                # Parse history to find state at fork point
                history = json.loads(state['history']) if state['history'] else []
                fork_point_snapshot = None
                
                for snapshot in history:
                    if snapshot.get('message_id', 0) <= fork_message_id:
                        fork_point_snapshot = snapshot
                    else:
                        break  # Stop at first snapshot after fork
                
                # Use fork point state or current state
                if fork_point_snapshot:
                    values = (
                        fork_point_snapshot['trust'],
                        fork_point_snapshot['emotional_bond'],
                        fork_point_snapshot['conflict'],
                        fork_point_snapshot['power_dynamic'],
                        fork_point_snapshot['fear_anxiety']
                    )
                else:
                    # No history, use current values
                    values = (
                        state['trust'],
                        state['emotional_bond'],
                        state['conflict'],
                        state['power_dynamic'],
                        state['fear_anxiety']
                    )
                
                # Create new history starting at fork point
                new_history = [{
                    'message_id': fork_message_id,
                    'timestamp': time.time(),
                    'trust': values[0],
                    'emotional_bond': values[1],
                    'conflict': values[2],
                    'power_dynamic': values[3],
                    'fear_anxiety': values[4],
                    'note': 'Branch fork point'
                }]
                
                # Insert relationship state for branch
                cursor.execute('''
                    INSERT INTO relationship_states
                    (chat_id, character_from, character_to, 
                     trust, emotional_bond, conflict, power_dynamic, fear_anxiety,
                     last_updated, last_analyzed_message_id, interaction_count, history)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ''', (
                    branch_chat_id, from_entity, to_entity,
                    *values,
                    time.time(), fork_message_id, json.dumps(new_history)
                ))
                
                copied_count += 1
            
            conn.commit()
            return copied_count
            
    except Exception as e:
        return 0


# ============================================================================
# ENTITY OPERATIONS (v1.5.1+ - NPC and Entity Management)
# ============================================================================

def generate_entity_id() -> str:
    """Generate a unique ID for entities."""
    import uuid
    return uuid.uuid4().hex[:12]


def db_entity_table_exists() -> bool:
    """Check if entities table exists.
    
    Returns:
        True if entities table exists, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='entities'
            """)
            return cursor.fetchone() is not None
    except Exception as e:
        return False


def db_create_entity(chat_id: str, entity_id: str, name: str, 
                     entity_type: str = 'npc') -> bool:
    """Create or update an entity record.
    
    Args:
        chat_id: Chat ID (None for global entities)
        entity_id: Unique entity ID (e.g., 'npc_abc123', 'char_xyz789')
        name: Entity name
        entity_type: Entity type ('character', 'npc', 'user')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        timestamp = int(time.time())
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO entities 
                (entity_id, entity_type, name, chat_id, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entity_id, entity_type, name, chat_id, timestamp, timestamp))
            conn.commit()
            return True
    except Exception as e:
        return False


def db_get_or_create_entity(chat_id: str, name: str, 
                             entity_type: str = 'npc') -> str:
    """Get existing entity or create new one, returns entity_id.
    
    Args:
        chat_id: Chat ID
        name: Entity name
        entity_type: Entity type ('character', 'npc', 'user')
    
    Returns:
        Entity ID string
    """
    # Graceful degradation: if table doesn't exist, use simple ID
    if not db_entity_table_exists():
        return f"{entity_type}_{name.replace(' ', '_').lower()}"
    
    # Try to find existing entity
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_id FROM entities
            WHERE chat_id = ? AND name = ? AND entity_type = ?
        """, (chat_id, name, entity_type))
        row = cursor.fetchone()
        if row:
            # Update last_seen
            cursor.execute("""
                UPDATE entities SET last_seen = ?
                WHERE entity_id = ?
            """, (int(time.time()), row['entity_id']))
            conn.commit()
            return row['entity_id']
    
    # Create new entity
    entity_id = f"{entity_type}_{generate_entity_id()}"
    db_create_entity(chat_id, entity_id, name, entity_type)
    return entity_id


def db_get_entity_info(entity_id: str) -> Optional[Dict[str, Any]]:
    """Get entity information.
    
    Args:
        entity_id: Unique entity ID
    
    Returns:
        Entity info dictionary or None if not found
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_id, entity_type, name, chat_id, first_seen, last_seen
            FROM entities WHERE entity_id = ?
        """, (entity_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def db_get_chat_entities(chat_id: str) -> List[Dict[str, Any]]:
    """Get all entities for a chat.
    
    Args:
        chat_id: Chat ID
    
    Returns:
        List of entity dictionaries
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_id, entity_type, name, first_seen, last_seen
            FROM entities WHERE chat_id = ?
            ORDER BY first_seen ASC
        """, (chat_id,))
        return [dict(row) for row in cursor.fetchall()]


def db_get_entity_id_by_name(chat_id: str, name: str, 
                             entity_type: str = 'npc') -> Optional[str]:
    """Get entity ID by name and type.
    
    Args:
        chat_id: Chat ID
        name: Entity name
        entity_type: Entity type ('character', 'npc', 'user')
    
    Returns:
        Entity ID string or None if not found
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_id FROM entities
            WHERE chat_id = ? AND name = ? AND entity_type = ?
        """, (chat_id, name, entity_type))
        row = cursor.fetchone()
        return row['entity_id'] if row else None


from typing import Dict, Any

def db_remap_entities_for_branch(
    origin_chat_id: str,
    branch_chat_id: str,
    localnpcs: Dict[str, Any],
) -> Dict[str, str]:
    """
    Create new entity IDs for NPCs in a forked branch.

    Args:
        origin_chat_id: Original chat ID
        branch_chat_id: New branch chat ID
        localnpcs: NPCs from chat metadata (will be modified in place)

    Returns:
        entity_mapping: Dict mapping old_entity_id -> new_entity_id
    """
    entity_mapping: Dict[str, str] = {}
    timestamp = int(time.time())

    with get_connection() as conn:
        cursor = conn.cursor()

        for old_npc_id, npc_data in list(localnpcs.items()):
            # Generate new entity ID for this branch
            new_entity_id = f"npc_{branch_chat_id}_{int(time.time() * 1000)}"

            # Register new entity in entities table
            cursor.execute(
                """
                INSERT INTO entities 
                (entity_id, entity_type, name, chat_id, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_entity_id, "npc", npc_data["name"], branch_chat_id, timestamp, timestamp),
            )

            # Update NPC data with new entity ID
            npc_data["entity_id"] = new_entity_id

            # Track mapping for relationship state copying
            entity_mapping[old_npc_id] = new_entity_id

        conn.commit()

    return entity_mapping



# ============================================================================
# NPC OPERATIONS (Phase 2.2 - Chat-scoped NPC Management)
# ============================================================================

def db_create_npc(chat_id: str, npc_data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """Create a new NPC. Returns (success, entity_id, error_message).
    
    FIX #1: Explicit transaction commit handling
    FIX #2: Comprehensive error logging at each step
    FIX #3: Entity ID uniqueness using UUID for guaranteed uniqueness
    """
    try:
        name = npc_data.get("name", "Unknown NPC")
        
        # FIX #3: Generate entity_id using UUID for guaranteed uniqueness
        # This prevents collisions when multiple NPCs are created rapidly
        import uuid
        entity_id = f"npc_{uuid.uuid4().hex}"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Verify entity_id uniqueness before insertion
            cursor.execute(
                "SELECT entity_id, name FROM chat_npcs WHERE chat_id = ? AND entity_id = ?",
                (chat_id, entity_id)
            )
            existing_by_id = cursor.fetchone()
            
            if existing_by_id:
                return False, None, "Entity ID already exists"
            
            # Also check for name uniqueness
            cursor.execute(
                "SELECT entity_id, name FROM chat_npcs WHERE chat_id = ? AND name = ?",
                (chat_id, name)
            )
            existing_by_name = cursor.fetchone()
            
            if existing_by_name:
                return False, None, "NPC name already exists in this chat"
            
            try:
                cursor.execute(
                    """
                    INSERT INTO chat_npcs 
                        (chat_id, entity_id, name, data, created_from_text, created_at, 
                         appearance_count, last_used_turn, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        entity_id,
                        name,
                        json.dumps(npc_data),
                        npc_data.get("created_from_text", ""),
                        int(time.time()),
                        0,          # appearance_count
                        None,       # last_used_turn
                        1,          # is_active
                    ),
                )
                
                # Verify INSERT succeeded
                affected_rows = cursor.rowcount
                
                if affected_rows == 0:
                    return False, None, "Database insertion failed (no rows affected)"
                
            except sqlite3.IntegrityError as e:
                return False, None, f"Database integrity error: {str(e)}"

            except Exception as e:
                return False, None, f"Database error during insertion: {str(e)}"

            # Register in entities
            try:
                success = db_create_entity(
                    chat_id=chat_id,
                    entity_id=entity_id,
                    name=name,
                    entity_type="npc",
                )
                
                if not success:
                    pass  # NPC is already in chat_npcs, don't fail
                    
            except Exception as e:
                pass  # NPC is already in chat_npcs, don't fail

            conn.commit()
            
            return True, entity_id, None

    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"



def db_get_chat_npcs(chat_id: str) -> list:
    """Get all NPCs for a specific chat"""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT entity_id, chat_id, name, data, is_active, created_at
                FROM chat_npcs 
                WHERE chat_id = ?
            """, (chat_id,))
            rows = cursor.fetchall()
            
            npcs = []
            for row in rows:
                npcs.append({
                    "entityid": row[0],
                    "entity_id": row[0],
                    "chat_id": row[1],
                    "name": row[2],
                    "data": json.loads(row[3]) if row[3] else {},
                    "isactive": bool(row[4]),
                    "is_active": bool(row[4]),
                    "promoted": False,
                    "globalfilename": None,
                    "createdat": row[5]
                })
            
            return npcs
        except Exception as e:
            return []
       

def db_get_npc_by_id(entity_id: str, chat_id: str = None) -> Optional[Dict]:
    """Get NPC by entity_id."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            if chat_id:
                cursor.execute('''
                    SELECT * FROM chat_npcs 
                    WHERE entity_id = ? AND chat_id = ?
                ''', (entity_id, chat_id))
            else:
                cursor.execute('''
                    SELECT * FROM chat_npcs WHERE entity_id = ?
                ''', (entity_id,))
            
            row = cursor.fetchone()
            if row:
                npc = dict(row)
                npc['data'] = json.loads(npc['data'])
                return npc
            return None
    except Exception as e:
        return None


def db_update_npc(chat_id: str, npc_id: str, npc_data: Dict) -> bool:
    """Update an existing NPC's data."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # First check if NPC exists
            cursor.execute("""
                SELECT entity_id FROM chat_npcs 
                WHERE chat_id = ? AND entity_id = ?
            """, (chat_id, npc_id))
            
            if not cursor.fetchone():
                print(f"[NPC] {npc_id} not found in chat {chat_id} for update")
                return False

            # Extract name from data if present
            name = npc_data.get("name", "")

            # Update the NPC
            cursor.execute("""
                UPDATE chat_npcs 
                SET data = ?, name = ?
                WHERE chat_id = ? AND entity_id = ?
            """, (json.dumps(npc_data), name, chat_id, npc_id))

            conn.commit()

            # Optional: log change for undo
            log_change("npc", npc_id, "UPDATE", {"chat_id": chat_id, "data": npc_data})

            return cursor.rowcount > 0
    except Exception as e:
        print(f"[NPC] Error updating NPC: {e}")
        return False

def db_increment_npc_appearance(chat_id: str, entity_id: str) -> None:
    """Increment appearance count when NPC speaks."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE chat_npcs 
                SET appearance_count = appearance_count + 1,
                    last_used_turn = ?
                WHERE chat_id = ? AND entity_id = ?
            ''', (time.time(), chat_id, entity_id))
            conn.commit()
    except Exception as e:
        print(f"[NPC] Error incrementing NPC appearance: {e}")


def db_set_npc_active(chat_id: str, entity_id: str, is_active: bool) -> bool:
    """Set NPC active/inactive status."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE chat_npcs SET is_active = ?
                WHERE chat_id = ? AND entity_id = ?
            ''', (1 if is_active else 0, chat_id, entity_id))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[NPC] Error setting active status: {e}")
        return False


def db_delete_npc(chat_id: str, entity_id: str) -> bool:
    """Permanently delete NPC."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM chat_npcs WHERE chat_id = ? AND entity_id = ?
            ''', (chat_id, entity_id))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[NPC] Error deleting NPC: {e}")
        return False


def db_create_npc_with_entity_id(chat_id: str, entity_id: str, npc_data: Dict) -> Tuple[bool, Optional[str]]:
    """
    Create NPC in database with a specific entity_id.
    Used when syncing metadata-only NPCs to the database.
    
    Args:
        chat_id: Chat ID
        entity_id: Existing entity ID to use (preserves relationships)
        npc_data: Character data dictionary
    
    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        name = npc_data.get("name", "Unknown NPC")
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if already exists
            cursor.execute(
                "SELECT entity_id FROM chat_npcs WHERE chat_id = ? AND entity_id = ?",
                (chat_id, entity_id)
            )
            if cursor.fetchone():
                return db_update_npc(chat_id, entity_id, npc_data), None
            
            # Insert with the provided entity_id
            cursor.execute(
                """
                INSERT INTO chat_npcs
                    (chat_id, entity_id, name, data, created_from_text, created_at,
                     appearance_count, last_used_turn, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    entity_id,
                    name,
                    json.dumps(npc_data),
                    npc_data.get("created_from_text", ""),
                    int(time.time()),
                    0,          # appearance_count
                    None,       # last_used_turn
                    1,          # is_active
                ),
            )
            
            # Register in entities table
            cursor.execute(
                """
                INSERT OR REPLACE INTO entities 
                (entity_id, entity_type, name, chat_id, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    "npc",
                    name,
                    chat_id,
                    int(time.time()),
                    int(time.time())
                )
            )
            
            conn.commit()
            return True, None
            
    except Exception as e:
        return False, str(e)


def db_promote_npc_to_character(chat_id: str, entity_id: str) -> Optional[str]:
    """
    Promote NPC to permanent character.
    Returns filename of created character, or None on failure.
    Also stores promotion history in character's extensions.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Get NPC data
            cursor.execute('''
                SELECT data, name FROM chat_npcs 
                WHERE chat_id = ? AND entity_id = ?
            ''', (chat_id, entity_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            npc_data = json.loads(row['data'])
            npc_name = row['name']
            
            # Generate unique filename
            filename = f"{npc_name.lower().replace(' ', '_')}.json"
            char_path = os.path.join(BASE_DIR, "app", "data", "characters", filename)
            
            # Check if character already exists
            counter = 1
            while os.path.exists(char_path):
                filename = f"{npc_name.lower().replace(' ', '_')}_{counter}.json"
                char_path = os.path.join(BASE_DIR, "app", "data", "characters", filename)
                counter += 1
            
            # Add promotion history to character extensions
            if 'extensions' not in npc_data:
                npc_data['extensions'] = {}
            
            npc_data['extensions']['promotion_history'] = {
                'origin_chat': chat_id,
                'promoted_at': int(time.time()),
                'original_entity_id': entity_id  # ✅ Added: Track original NPC ID
            }
            
            # Save as permanent character (reuse existing save logic)
            success = db_save_character(npc_data, filename)
            
            if success:
                # ✅ UPDATED: Mark as promoted + inactive + track global filename
                cursor.execute('''
                    UPDATE chat_npcs 
                    SET is_active = 0,
                        promoted = 1,
                        promoted_at = ?,
                        global_filename = ?
                    WHERE chat_id = ? AND entity_id = ?
                ''', (int(time.time()), filename, chat_id, entity_id))
                conn.commit()
                
                print(f"[NPC] Promoted '{npc_name}' to character: {filename}")
                return filename
            
            return None
    except Exception as e:
        print(f"[NPC] Error promoting NPC: {e}")
        return None

def db_create_npc_and_update_chat(chat_id: str, npc_data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Create NPC and update chat metadata in a single atomic transaction.
    
    This fixes a connection caching issue where NPCs created in the chat_npcs 
    table aren't immediately visible because db_create_npc() uses one connection 
    and db_save_chat() uses a separate connection that hasn't refreshed.
    
    Args:
        chat_id: Chat ID
        npc_data: Character data dictionary
    
    Returns:
        (success: bool, entity_id: str, error_message: str)
    """
    try:
        name = npc_data.get("name", "Unknown NPC")
        
        # Generate unique entity_id
        entity_id = f"npc_{chat_id}_{int(time.time())}_{hash(name + str(time.time())) % 100000}"
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Start transaction explicitly
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                # Step 1: Check for existing NPC by entity_id or name
                cursor.execute(
                    "SELECT entity_id, name FROM chat_npcs WHERE chat_id = ? AND entity_id = ?",
                    (chat_id, entity_id)
                )
                existing_by_id = cursor.fetchone()
                
                if existing_by_id:
                    conn.rollback()
                    return False, None, "Entity ID already exists"
                
                cursor.execute(
                    "SELECT entity_id, name FROM chat_npcs WHERE chat_id = ? AND name = ?",
                    (chat_id, name)
                )
                existing_by_name = cursor.fetchone()
                
                if existing_by_name:
                    conn.rollback()
                    return False, None, "NPC name already exists in this chat"
                
                # Insert NPC into chat_npcs table
                cursor.execute(
                    """
                    INSERT INTO chat_npcs 
                        (chat_id, entity_id, name, data, created_from_text, created_at, 
                         appearance_count, last_used_turn, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        entity_id,
                        name,
                        json.dumps(npc_data),
                        npc_data.get("created_from_text", ""),
                        int(time.time()),
                        0,          # appearance_count
                        None,       # last_used_turn
                        1,          # is_active
                    ),
                )
                
                # Register in entities table
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO entities 
                    (entity_id, entity_type, name, chat_id, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        "npc",
                        name,
                        chat_id,
                        int(time.time()),
                        int(time.time())
                    )
                )
                
                # Load existing chat metadata
                cursor.execute(
                    """
                    SELECT branch_name, summary, metadata, created_at
                    FROM chats
                    WHERE id = ?
                    """,
                    (chat_id,)
                )
                
                chat_row = cursor.fetchone()
                if not chat_row:
                    conn.rollback()
                    return False, None, f"Chat '{chat_id}' not found"
                
                # Parse existing metadata
                existing_metadata = json.loads(chat_row['metadata']) if chat_row['metadata'] else {}
                branch_name = existing_metadata.get('branch_name') or chat_row['branch_name']
                summary = chat_row['summary'] or ''
                created_at = chat_row['created_at']
                
                # Ensure localnpcs exists, preserving existing
                if "localnpcs" not in existing_metadata:
                    existing_metadata["localnpcs"] = {}
                
                # Add new NPC to metadata (preserving existing localnpcs)
                existing_metadata["localnpcs"][entity_id] = {
                    'name': name,
                    'data': npc_data,
                    'is_active': True,
                    'promoted': False,
                    'created_at': int(time.time())
                }
                
                # Get current active characters
                active_chars = existing_metadata.get('activeCharacters', [])
                
                if entity_id not in active_chars:
                    active_chars.append(entity_id)
                
                existing_metadata["activeCharacters"] = active_chars
                existing_metadata["activeWI"] = existing_metadata.get('activeWI')
                existing_metadata["settings"] = existing_metadata.get('settings', {})
                
                # Update chat record with new metadata
                metadata_json = json.dumps(existing_metadata, ensure_ascii=False)
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chats 
                    (id, branch_name, summary, metadata, created_at, autosaved)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        branch_name,
                        summary,
                        metadata_json,
                        created_at,
                        True  # autosaved
                    )
                )
                
                # Commit transaction
                conn.commit()
                
                # Force WAL checkpoint to ensure other connections can see the changes
                cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")
                
                return True, entity_id, None
                
            except Exception as e:
                # Rollback on any error
                conn.rollback()
                return False, None, f"Atomic transaction failed: {str(e)}"
                
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


def db_copy_npcs_for_branch(origin_chat_id: str, branch_chat_id: str) -> int:
    """
    Copy NPCs to branch when forking.
    Preserves entity_ids for relationship continuity and registers them in entities.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT entity_id, name, data, created_from_text, appearance_count
                FROM chat_npcs WHERE chat_id = ? AND is_active = 1
            ''', (origin_chat_id,))
            
            npcs = cursor.fetchall()
            copied_count = 0
            
            for npc in npcs:
                cursor.execute('''
                    INSERT INTO chat_npcs 
                    (chat_id, entity_id, name, data, created_from_text, 
                     created_at, appearance_count, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ''', (
                    branch_chat_id,
                    npc['entity_id'],   # preserve entity_id
                    npc['name'],
                    npc['data'],
                    npc['created_from_text'],
                    time.time(),
                    npc['appearance_count'],
                ))

                # Register entity for the branch chat
                db_create_entity(
                    chat_id=branch_chat_id,
                    entity_id=npc['entity_id'],
                    name=npc['name'],
                    entity_type='npc',
                )

                copied_count += 1
            
            conn.commit()
            print(f"[FORK] Copied {copied_count} NPCs to branch {branch_chat_id}")
            return copied_count
    except Exception as e:
        print(f"[FORK] Error copying NPCs: {e}")
        return 0


# ============================================================================
# FILTERED RELATIONSHIP CONTEXT (v1.6.1 - Adaptive Tier 3)
# ============================================================================

def get_relationship_context_filtered(
    chat_id: str,
    current_text: str,
    relationship_states: Dict[str, Dict[str, float]],
    relationship_templates: Dict[str, Dict[Tuple[int, int], List[str]]]
) -> str:
    """
    Generate filtered relationship context for prompt injection using adaptive Tier 3.
    
    Only includes dimensions that are BOTH:
    1. Deviating from neutral (score > 15 points from 50)
    2. Semantically relevant to current conversation (similarity > 0.35)
    
    This reduces prompt bloat by excluding neutral or irrelevant relationships.
    
    Args:
        chat_id: Chat ID
        current_text: Current turn's message text
        relationship_states: Dict of {from_entity: {dimension: score}}
        relationship_templates: Template dictionary (same format as get_relationship_context in main.py)
    
    Returns:
        Formatted relationship context string or empty string if no relevant relationships
    """
    import random
    
    if not current_text or not relationship_states:
        return ""
    
    lines = []
    
    # Filter: Only include dimensions that deviate from neutral (>15 points from 50)
    # AND are semantically relevant to current conversation
    # Note: This filtering is handled by adaptive_tracker.get_relevant_dimensions()
    # in the relationship_tracker module
    
    # For each relationship pair that deviates from neutral
    for from_entity, dimensions in relationship_states.items():
        # Skip if no dimensions deviate from neutral
        non_neutral_dims = [
            dim for dim, score in dimensions.items()
            if abs(score - 50) > 15
        ]
        
        if not non_neutral_dims:
            continue
        
        # Only include top 2 most extreme dimensions
        non_neutral_dims.sort(key=lambda x: abs(dimensions[x[1]] - 50), reverse=True)
        top_two = non_neutral_dims[:2]
        
        # Generate templates for top two dimensions
        for dim in top_two:
            score = dimensions[dim]
            templates = relationship_templates.get(dim, {})
            
            # Find matching template range
            for (low, high), template_list in templates.items():
                if low <= score <= high and template_list:
                    if templates:
                        template = random.choice(templates)
                        lines.append(template.format(from_=from_entity, to=to_entity))
            
        if lines:
            lines.append(".")
    
    if lines:
        return "### Relationship Context:\n" + "\n".join(lines)
    
    return ""

# Initialize database on module import (only once at the end)
init_db()
init_vec_table()
