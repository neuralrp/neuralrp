"""
Migration Script: Move NPCs from metadata to database
=====================================================

This script migrates all NPCs from chat metadata (local_npcs key) to the 
chat_npcs database table. This fixes the issue where NPCs exist in metadata
but not in the database, causing them to appear as "unknown" and preventing
NPC operations (promote, delete, toggle) from working correctly.
"""

import sqlite3
import json
import sys
import os
from datetime import datetime

# Database path
DB_PATH = r'C:\Users\fosbo\neuralrp\app\data\neuralrp.db'

def generate_entity_id(chat_id: str, name: str) -> str:
    """Generate a unique entity ID for an NPC."""
    import time
    import hashlib
    unique_str = f"{chat_id}_{name}_{time.time()}"
    hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:8]
    return f"npc_{int(time.time())}_{hash_val}"

def migrate_npcs_from_metadata():
    """
    Migrate all NPCs from metadata to database.
    
    For each NPC in metadata (local_npcs):
    1. Check if it already exists in database
    2. If not, add it to chat_npcs table
    3. Update entities table
    4. Keep metadata synced
    """
    
    print(f"[MIGRATION] Starting at {datetime.now()}")
    print(f"[MIGRATION] Database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get all chats
        cursor.execute("SELECT id, metadata FROM chats")
        chats = cursor.fetchall()
        
        total_npcs_migrated = 0
        total_chats_processed = 0
        
        for chat in chats:
            chat_id = chat['id']
            metadata = json.loads(chat['metadata']) if chat['metadata'] else {}
            local_npcs = metadata.get('local_npcs', {})
            localnpcs_old = metadata.get('localnpcs', {})  # Old key name
            
            # Merge old and new key names
            all_npcs = {}
            all_npcs.update(localnpcs_old)
            all_npcs.update(local_npcs)
            
            if not all_npcs:
                continue
            
            total_chats_processed += 1
            print(f"\n[MIGRATION] Processing chat: {chat_id}")
            print(f"[MIGRATION]   Found {len(all_npcs)} NPCs in metadata")
            
            # Get existing NPCs in database for this chat
            cursor.execute(
                "SELECT entity_id, name FROM chat_npcs WHERE chat_id = ?",
                (chat_id,)
            )
            existing_db_npcs = {row['entity_id']: row['name'] for row in cursor.fetchall()}
            
            print(f"[MIGRATION]   Found {len(existing_db_npcs)} NPCs in database")
            
            migrated_in_chat = 0
            
            # For each NPC in metadata, check if it exists in database
            for npc_id, npc_data in all_npcs.items():
                npc_name = npc_data.get('name', 'Unknown')
                npc_data_dict = npc_data.get('data', npc_data)  # Handle both formats
                
                # Check if already in database
                if npc_id in existing_db_npcs:
                    print(f"[MIGRATION]   [OK] {npc_name} already in database (skipping)")
                    continue
                
                print(f"[MIGRATION]   [MIGRATING] {npc_name} ({npc_id})")
                
                # Prepare data for database
                data_json = json.dumps(npc_data_dict) if isinstance(npc_data_dict, dict) else str(npc_data_dict)
                created_from_text = npc_data.get('created_from_text', '')
                created_at = npc_data.get('created_at', int(datetime.now().timestamp()))
                is_active = npc_data.get('is_active', True)
                promoted = npc_data.get('promoted', False)
                
                # Insert into chat_npcs table
                cursor.execute("""
                    INSERT INTO chat_npcs 
                    (chat_id, entity_id, name, data, created_from_text, created_at, 
                     appearance_count, last_used_turn, is_active, promoted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chat_id,
                    npc_id,
                    npc_name,
                    data_json,
                    created_from_text,
                    created_at,
                    0,  # appearance_count
                    None,  # last_used_turn
                    1 if is_active else 0,
                    1 if promoted else 0
                ))
                
                # Register in entities table
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (entity_id, entity_type, name, chat_id, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    npc_id,
                    "npc",
                    npc_name,
                    chat_id,
                    created_at,
                    created_at
                ))
                
                migrated_in_chat += 1
                print(f"[MIGRATION]     [OK] Added to database")
            
            if migrated_in_chat > 0:
                # Normalize metadata: ensure we use 'local_npcs' key
                if 'localnpcs' in metadata and 'local_npcs' not in metadata:
                    metadata['local_npcs'] = metadata.pop('localnpcs')
                    print(f"[MIGRATION]   [OK] Renamed 'localnpcs' to 'local_npcs'")
                
                # Update metadata to reflect migration
                # (keep it for backwards compatibility, but now it's synced with DB)
                metadata['local_npcs'] = all_npcs
                metadata_json = json.dumps(metadata, ensure_ascii=False)
                
                # Update chat record
                cursor.execute("""
                    UPDATE chats SET metadata = ? WHERE id = ?
                """, (metadata_json, chat_id))
                
                print(f"[MIGRATION]   [OK] Migrated {migrated_in_chat} NPCs from chat {chat_id}")
                total_npcs_migrated += migrated_in_chat
            else:
                print(f"[MIGRATION]   No new NPCs to migrate for chat {chat_id}")
        
        # Commit all changes
        conn.commit()
        
        print(f"\n[MIGRATION] [SUCCESS] Migration completed successfully!")
        print(f"[MIGRATION]   Chats processed: {total_chats_processed}")
        print(f"[MIGRATION]   NPCs migrated: {total_npcs_migrated}")
        print(f"[MIGRATION]   Completed at: {datetime.now()}")
        
        # Verify migration
        cursor.execute("SELECT COUNT(*) FROM chat_npcs")
        total_in_db = cursor.fetchone()[0]
        print(f"[MIGRATION]   Total NPCs in database: {total_in_db}")
        
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"\n[MIGRATION] [ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = migrate_npcs_from_metadata()
    sys.exit(0 if success else 1)
