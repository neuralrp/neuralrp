"""
NeuralRP Database Module
Centralized SQLite database operations for characters, world info, chats, and images.
"""

import sqlite3
import json
import time
import threading
import os
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import numpy as np
import struct

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
                created_at INTEGER
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
                FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
            )
        """)
        
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
        
        conn.commit()
        print("Database initialized successfully")


# ============================================================================
# CHARACTER OPERATIONS
# ============================================================================

def db_get_all_characters() -> List[Dict[str, Any]]:
    """Get all characters from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, data, danbooru_tag, capsule, created_at
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
            
            # Add danbooru_tag and capsule to extensions
            if row['danbooru_tag']:
                char_data['data']['extensions']['danbooru_tag'] = row['danbooru_tag']
            if row['capsule']:
                char_data['data']['extensions']['multi_char_summary'] = row['capsule']
            
            characters.append(char_data)
        
        return characters


def db_get_character(filename: str) -> Optional[Dict[str, Any]]:
    """Get a specific character by filename."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT filename, data, danbooru_tag, capsule
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
        if row['capsule']:
            char_data['data']['extensions']['multi_char_summary'] = row['capsule']
        
        return char_data


def db_save_character(char_data: Dict[str, Any], filename: str) -> bool:
    """Save or update a character in the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract fields
            name = char_data.get('data', {}).get('name', 'Unknown')
            extensions = char_data.get('data', {}).get('extensions', {})
            danbooru_tag = extensions.get('danbooru_tag', '')
            capsule = extensions.get('multi_char_summary', '')
            
            # Remove _filename from data before saving
            save_data = char_data.copy()
            if '_filename' in save_data:
                del save_data['_filename']
            
            data_json = json.dumps(save_data, ensure_ascii=False)
            timestamp = int(time.time())
            
            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO characters 
                (filename, name, data, danbooru_tag, capsule, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (filename, name, data_json, danbooru_tag, capsule, timestamp))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving character to database: {e}")
        return False


def db_delete_character(filename: str) -> bool:
    """Delete a character from the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM characters WHERE filename = ?", (filename,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting character from database: {e}")
        return False


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
                **metadata
            }
        
        return {'entries': entries}


def db_save_world(name: str, entries: Dict[str, Any]) -> bool:
    """Save or update a world info with all its entries.
    
    Note: This function deletes and re-inserts all entries. This automatically
    invalidates the content hash (used for embedding cache invalidation) since
    db_get_world_content_hash() generates a hash based on current content.
    
    The semantic search engine uses this content hash to detect changes and
    will recompute embeddings when the hash changes.
    """
    try:
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
                     probability, use_probability, depth, sort_order, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (world_id, uid, content, keys, secondary_keys, is_canon_law,
                      probability, use_probability, depth, int(uid) if uid.isdigit() else 999,
                      metadata_json))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving world to database: {e}")
        return False


def db_delete_world(name: str) -> bool:
    """Delete a world and all its entries."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM worlds WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting world from database: {e}")
        return False


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


def db_get_chat(chat_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific chat with all its messages."""
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
        
        # Get messages
        cursor.execute("""
            SELECT id, role, content, speaker, image_url, timestamp
            FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp ASC
        """, (chat_id,))
        
        messages = []
        for msg_row in cursor.fetchall():
            messages.append({
                'id': msg_row['id'],
                'role': msg_row['role'],
                'content': msg_row['content'],
                'speaker': msg_row['speaker'],
                'image': msg_row['image_url']
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


def db_save_chat(chat_id: str, data: Dict[str, Any]) -> bool:
    """Save or update a chat session with all messages."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare metadata
            metadata = data.get('metadata', {})
            metadata['activeCharacters'] = data.get('activeCharacters', [])
            metadata['activeWI'] = data.get('activeWI')
            metadata['settings'] = data.get('settings', {})
            
            # Extract branch_name from metadata or data
            branch_name = metadata.get('branch_name') or data.get('branch_name')
            
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            summary = data.get('summary', '')
            timestamp = int(time.time())
            
            # Insert or replace chat
            cursor.execute("""
                INSERT OR REPLACE INTO chats 
                (id, branch_name, summary, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (chat_id, branch_name, summary, metadata_json, timestamp))
            
            # Delete existing messages
            cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            
            # Insert messages
            for msg in data.get('messages', []):
                cursor.execute("""
                    INSERT INTO messages 
                    (chat_id, role, content, speaker, image_url, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    chat_id,
                    msg.get('role', 'user'),
                    msg.get('content', ''),
                    msg.get('speaker'),
                    msg.get('image'),
                    msg.get('timestamp', int(time.time()))
                ))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error saving chat to database: {e}")
        return False


def db_delete_chat(chat_id: str) -> bool:
    """Delete a chat and all its messages."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting chat from database: {e}")
        return False


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
            print("sqlite-vec vector table initialized successfully")
            return True
    except ImportError:
        print("WARNING: sqlite-vec not installed. Vector search will use fallback mode.")
        return False
    except Exception as e:
        print(f"Error initializing sqlite-vec table: {e}")
        return False


def db_save_embedding(world_name: str, entry_uid: str, embedding: np.ndarray) -> bool:
    """Save an embedding vector to sqlite-vec (disk-based storage)."""
    try:
        import sqlite_vec
        
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
        print("WARNING: sqlite-vec not available for saving embeddings")
        return False
    except Exception as e:
        print(f"Error saving embedding to sqlite-vec: {e}")
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
        print(f"Error getting embedding from sqlite-vec: {e}")
        return None


def db_search_similar_embeddings(world_name: str, query_embedding: np.ndarray, 
                                   k: int = 10, threshold: float = 0.3) -> List[Tuple[str, float]]:
    """
    Search for similar embeddings using sqlite-vec's SIMD-accelerated KNN search.
    Returns list of (entry_uid, similarity_score) tuples.
    """
    try:
        import sqlite_vec
        
        with get_connection() as conn:
            sqlite_vec.load(conn)
            
            # Serialize query embedding
            query_bytes = _serialize_float32(query_embedding)
            
            cursor = conn.cursor()
            # Use vec_distance_cosine for cosine similarity (1 - cosine_distance = similarity)
            cursor.execute("""
                SELECT entry_uid, 
                       (1 - vec_distance_cosine(embedding, ?)) as similarity
                FROM vec_world_entries
                WHERE world_name = ? AND similarity >= ?
                ORDER BY similarity DESC
                LIMIT ?
            """, (query_bytes, world_name, threshold, k))
            
            results = []
            for row in cursor.fetchall():
                results.append((row['entry_uid'], row['similarity']))
            
            return results
    except ImportError:
        print("WARNING: sqlite-vec not available for similarity search")
        return []
    except Exception as e:
        print(f"Error searching embeddings with sqlite-vec: {e}")
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
        print(f"Error deleting world embeddings: {e}")
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
        print(f"Error counting embeddings: {e}")
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


# Initialize database on module import
init_db()
init_vec_table()
