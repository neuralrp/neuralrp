"""
Danbooru Characters Import Script

Imports Danbooru characters from Excel file into SQLite database.
Generates embeddings using all-mpnet-base-v2 for semantic search.

Usage: python app/import_danbooru_characters.py
"""

import logging
import pandas as pd
import sqlite3
import numpy as np
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DB_PATH

# Updated to use lowercase book1.xlsx
EXCEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "danbooru", "book1.xlsx")


def get_excel_mtime():
    """Get the modification time of the Excel file."""
    try:
        return os.path.getmtime(EXCEL_PATH)
    except Exception as e:
        logger.error(f"Failed to get Excel mtime: {type(e).__name__}: {e}")
        return 0


def get_last_import_time():
    """Get the last import timestamp from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = 'danbooru_last_import'")
        row = cursor.fetchone()
        conn.close()
        return float(row[0]) if row else 0
    except Exception as e:
        logger.error(f"Failed to get last import time: {type(e).__name__}: {e}")
        return 0


def set_last_import_time(mtime):
    """Set the last import timestamp in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('danbooru_last_import', ?)
    """, (str(mtime),))
    conn.commit()
    conn.close()


def detect_gender_from_tags(all_tags):
    """Detect gender from tags. Returns 'female', 'male', or 'unknown'."""
    tags_lower = all_tags.lower()
    if '1girl' in tags_lower or 'female' in tags_lower:
        return 'female'
    elif '1boy' in tags_lower or 'male' in tags_lower:
        return 'male'
    return 'unknown'


def load_excel_data(excel_file):
    """Load Danbooru characters from Excel file with flexible format."""
    print(f"[IMPORT] Loading Excel file: {excel_file}")
    df = pd.read_excel(excel_file, header=None)  # No header row
    print(f"[IMPORT] Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def import_characters(df, db_path=DB_PATH):
    """Import characters into SQLite database with flexible column format."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = ON")
    
    # Load sqlite-vec extension for embedding inserts
    vec_loaded = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        print("[IMPORT] sqlite-vec extension loaded")
        vec_loaded = True
    except Exception as e:
        print(f"[IMPORT FATAL] Could not load sqlite-vec: {e}")
        print("[IMPORT FATAL] Embeddings are required for this application.")
        sys.exit(1)
    
    # Load embedding model (local only to avoid network calls)
    print("[IMPORT] Loading embedding model (all-mpnet-base-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2', local_files_only=True)
        print("[IMPORT] Embedding model loaded successfully")
    except ImportError as e:
        print(f"[IMPORT FATAL] sentence-transformers not available: {e}")
        print("[IMPORT FATAL] Embeddings are required for this application.")
        sys.exit(1)
    
    imported = 0
    skipped = 0
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            # Column 0 is the character name/tag (required)
            name = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
            name = name.strip()
            
            if not name:
                skipped += 1
                continue
            
            # All columns including character name (column 0) are tags for matching
            all_tags = [name]  # Start with character tag from column 0
            for col_idx in range(1, len(row)):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    tag = str(val).strip()
                    if tag and tag not in all_tags:
                        all_tags.append(tag)
            
            all_tags_str = ', '.join(all_tags)
            
            # Detect gender from tags
            gender = detect_gender_from_tags(all_tags_str)
            
            # Generate embedding from all tags
            embedding_text = f"{gender} {all_tags_str}"
            embedding = model.encode(embedding_text).astype(np.float32)
            
            # Insert into danbooru_characters
            c.execute("""
                INSERT OR REPLACE INTO danbooru_characters
                (name, gender, core_tags, all_tags, image_link, source_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, gender, all_tags_str, all_tags_str, '', name, int(time.time())))
            
            char_id = c.lastrowid
            
            # Insert embedding into vec_danbooru_characters
            c.execute("""
                INSERT OR REPLACE INTO vec_danbooru_characters
                (rowid, embedding)
                VALUES (?, ?)
            """, (char_id, embedding.tobytes()))
            
            imported += 1
            
            # Progress logging
            if imported % 100 == 0:
                conn.commit()
                print(f"[IMPORT] Progress: {imported} characters imported...")
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Only show first 5 errors
                print(f"[IMPORT ERROR] Failed to import row {idx}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print(f"[IMPORT] Complete!")
    print(f"[IMPORT]   Imported: {imported} characters")
    print(f"[IMPORT]   Skipped: {skipped} characters (missing name)")
    print(f"[IMPORT]   Errors:  {errors} characters")
    print(f"[IMPORT] Total:   {imported + skipped} characters processed")
    
    return imported > 0


def check_if_import_needed():
    """
    Check if Danbooru import is needed.
    Returns (needs_import, reason) tuple.
    """
    if not os.path.exists(EXCEL_PATH):
        return (False, "Excel file not found")
    
    excel_mtime = get_excel_mtime()
    last_import = get_last_import_time()
    
    # Check if Excel file is newer than last import
    if excel_mtime > last_import:
        return (True, f"Excel file modified since last import ({excel_mtime} > {last_import})")
    
    # Check if we have any characters in database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM danbooru_characters")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            return (True, "No characters in database")
        else:
            return (False, f"Already imported {count} characters")
    except Exception as e:
        logger.error(f"Failed to check import status: {type(e).__name__}: {e}")
        return (True, "Database error, assuming import needed")


def clear_existing_data():
    """Clear all existing Danbooru data from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM danbooru_characters")
        deleted = c.rowcount
        conn.commit()
        conn.close()
        return deleted
    except Exception as e:
        print(f"[IMPORT WARNING] Could not clear existing data: {e}")
        return 0


if __name__ == "__main__":
    # Handle --check-only flag (used by launcher.bat)
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        needs_import, reason = check_if_import_needed()
        if needs_import:
            print(f"[IMPORT CHECK] Import needed: {reason}")
            sys.exit(1)  # Exit 1 = import needed
        else:
            print(f"[IMPORT CHECK] Import not needed: {reason}")
            sys.exit(0)  # Exit 0 = no import needed
    
    # Check for --force flag to force re-import
    force_reimport = len(sys.argv) > 1 and sys.argv[1] == "--force"
    
    print("="*60)
    print("Danbooru Characters Import")
    print("="*60)
    
    # Check if import is needed
    needs_import, reason = check_if_import_needed()
    
    if not needs_import and not force_reimport:
        print(f"[IMPORT] No import needed: {reason}")
        print("[IMPORT] Use --force to force re-import.")
        sys.exit(0)
    
    if force_reimport:
        print("[IMPORT] Force re-import requested.")
        deleted = clear_existing_data()
        if deleted > 0:
            print(f"[IMPORT] Cleared {deleted} existing characters.")
    
    print(f"[IMPORT] Reason: {reason}")
    
    # Check if Excel file exists
    if not os.path.exists(EXCEL_PATH):
        print(f"[IMPORT WARNING] Excel file not found: {EXCEL_PATH}")
        print("[IMPORT WARNING] Skipping Danbooru character import.")
        print("[IMPORT WARNING] Tag generation will not work without imported characters.")
        sys.exit(0)  # Exit gracefully, don't block startup
    
    # Load Excel data
    df = load_excel_data(EXCEL_PATH)
    
    if df.empty:
        print("[IMPORT ERROR] Excel file is empty or contains no data.")
        sys.exit(1)
    
    # Import to database
    print("\n[IMPORT] Starting import...")
    success = import_characters(df)
    
    if success:
        # Record import time
        set_last_import_time(get_excel_mtime())
        print(f"[IMPORT] Import timestamp recorded: {get_excel_mtime()}")
    
    print("\n[IMPORT] Done!")
