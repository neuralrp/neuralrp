"""
Soft Delete Migration Script
Adds 'summarized' column to messages table for archiving instead of deleting.
This preserves message history for search and analysis while keeping the interface clean.
"""

import sqlite3
import os

# Database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app", "data", "neuralrp.db")

def migrate_soft_delete():
    """Add summarized column to messages table and create index."""
    print("Starting soft delete migration...")
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Check if summarized column already exists
        cursor.execute("PRAGMA table_info(messages)")
        columns = [row['name'] for row in cursor.fetchall()]
        
        if 'summarized' in columns:
            print("✓ 'summarized' column already exists in messages table")
        else:
            # Add summarized column
            cursor.execute("ALTER TABLE messages ADD COLUMN summarized BOOLEAN DEFAULT 0")
            print("✓ Added 'summarized' column to messages table")
        
        # Check if index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_summarized'")
        index_exists = cursor.fetchone() is not None
        
        if index_exists:
            print("✓ Index 'idx_messages_summarized' already exists")
        else:
            # Create index for performance
            cursor.execute("CREATE INDEX idx_messages_summarized ON messages(chat_id, summarized)")
            print("✓ Created index 'idx_messages_summarized' for optimized queries")
        
        # Verify migration
        cursor.execute("PRAGMA table_info(messages)")
        columns = [row['name'] for row in cursor.fetchall()]
        
        if 'summarized' not in columns:
            print("✗ Migration failed: 'summarized' column not found")
            conn.rollback()
            return False
        
        conn.commit()
        print("\n✓ Soft delete migration completed successfully!")
        print("\nWhat this means:")
        print("- Messages are now archived (summarized=1) instead of deleted")
        print("- Search works across entire session history")
        print("- Message IDs persist after summarization")
        print("- Performance impact: Negligible (indexed queries)")
        print("\nNext steps:")
        print("- Modify db_save_chat() to mark messages as summarized")
        print("- Modify db_get_chat() to support include_summarized parameter")
        print("- Update /api/chats/{name} endpoint")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_soft_delete()
