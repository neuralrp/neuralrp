"""
NeuralRP SQLite Migration Script
Migrates existing JSON data to SQLite database while maintaining backward compatibility.
"""

import os
import json
import shutil
from datetime import datetime
from app.database import (
    db_save_character, db_get_character,
    db_save_world, db_get_world,
    db_save_chat, db_get_chat,
    db_save_image_metadata, db_get_image_metadata
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backup_json")
CHAR_DIR = os.path.join(DATA_DIR, "characters")
WORLD_DIR = os.path.join(DATA_DIR, "worldinfo")
CHAT_DIR = os.path.join(DATA_DIR, "chats")
IMAGE_DIR = os.path.join(BASE_DIR, "app", "images")

# Statistics
stats = {
    "characters": {"migrated": 0, "skipped": 0, "errors": 0},
    "worlds": {"migrated": 0, "skipped": 0, "errors": 0},
    "chats": {"migrated": 0, "skipped": 0, "errors": 0},
    "images": {"migrated": 0, "skipped": 0, "errors": 0}
}


def create_backup():
    """Create backup of all JSON files before migration."""
    print("\n" + "="*70)
    print("CREATING BACKUP OF JSON FILES")
    print("="*70)
    
    if os.path.exists(BACKUP_DIR):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_backup = f"{BACKUP_DIR}_{timestamp}"
        print(f"Previous backup found, renaming to: {old_backup}")
        shutil.move(BACKUP_DIR, old_backup)
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Backup directory structure
    for dirname in ["characters", "worldinfo", "chats"]:
        os.makedirs(os.path.join(BACKUP_DIR, dirname), exist_ok=True)
    
    # Copy files
    backup_count = 0
    
    # Characters
    if os.path.exists(CHAR_DIR):
        for filename in os.listdir(CHAR_DIR):
            if filename.endswith(".json"):
                src = os.path.join(CHAR_DIR, filename)
                dst = os.path.join(BACKUP_DIR, "characters", filename)
                shutil.copy2(src, dst)
                backup_count += 1
    
    # World info
    if os.path.exists(WORLD_DIR):
        for filename in os.listdir(WORLD_DIR):
            if filename.endswith(".json"):
                src = os.path.join(WORLD_DIR, filename)
                dst = os.path.join(BACKUP_DIR, "worldinfo", filename)
                shutil.copy2(src, dst)
                backup_count += 1
    
    # Chats
    if os.path.exists(CHAT_DIR):
        for filename in os.listdir(CHAT_DIR):
            if filename.endswith(".json"):
                src = os.path.join(CHAT_DIR, filename)
                dst = os.path.join(BACKUP_DIR, "chats", filename)
                shutil.copy2(src, dst)
                backup_count += 1
    
    # Image metadata
    image_meta_path = os.path.join(IMAGE_DIR, "image_metadata.json")
    if os.path.exists(image_meta_path):
        shutil.copy2(image_meta_path, os.path.join(BACKUP_DIR, "image_metadata.json"))
        backup_count += 1
    
    print(f"✓ Backed up {backup_count} files to: {BACKUP_DIR}")


def migrate_characters():
    """Migrate character JSON files to database."""
    print("\n" + "="*70)
    print("MIGRATING CHARACTERS")
    print("="*70)
    
    if not os.path.exists(CHAR_DIR):
        print("⚠ Characters directory not found")
        return
    
    for filename in os.listdir(CHAR_DIR):
        if not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(CHAR_DIR, filename)
        
        try:
            # Check if already in database
            existing = db_get_character(filename)
            if existing:
                print(f"⊙ {filename} - Already in database, skipping")
                stats["characters"]["skipped"] += 1
                continue
            
            # Load from JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                char_data = json.load(f)
            
            # Save to database
            if db_save_character(char_data, filename):
                print(f"✓ {filename} - Migrated successfully")
                stats["characters"]["migrated"] += 1
            else:
                print(f"✗ {filename} - Failed to save to database")
                stats["characters"]["errors"] += 1
        
        except Exception as e:
            print(f"✗ {filename} - Error: {e}")
            stats["characters"]["errors"] += 1


def migrate_world_info():
    """Migrate world info JSON files to database."""
    print("\n" + "="*70)
    print("MIGRATING WORLD INFO")
    print("="*70)
    
    if not os.path.exists(WORLD_DIR):
        print("⚠ World info directory not found")
        return
    
    for filename in os.listdir(WORLD_DIR):
        if not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(WORLD_DIR, filename)
        world_name = filename.replace(".json", "")
        
        try:
            # Check if already in database
            existing = db_get_world(world_name)
            if existing:
                print(f"⊙ {world_name} - Already in database, skipping")
                stats["worlds"]["skipped"] += 1
                continue
            
            # Load from JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                world_data = json.load(f)
            
            # Save to database
            entries = world_data.get("entries", {})
            if db_save_world(world_name, entries):
                entry_count = len(entries)
                print(f"✓ {world_name} - Migrated with {entry_count} entries")
                stats["worlds"]["migrated"] += 1
            else:
                print(f"✗ {world_name} - Failed to save to database")
                stats["worlds"]["errors"] += 1
        
        except Exception as e:
            print(f"✗ {world_name} - Error: {e}")
            stats["worlds"]["errors"] += 1


def migrate_chats():
    """Migrate chat JSON files to database."""
    print("\n" + "="*70)
    print("MIGRATING CHATS")
    print("="*70)
    
    if not os.path.exists(CHAT_DIR):
        print("⚠ Chats directory not found")
        return
    
    for filename in os.listdir(CHAT_DIR):
        if not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(CHAT_DIR, filename)
        chat_id = filename.replace(".json", "")
        
        try:
            # Check if already in database
            existing = db_get_chat(chat_id)
            if existing:
                print(f"⊙ {chat_id} - Already in database, skipping")
                stats["chats"]["skipped"] += 1
                continue
            
            # Load from JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            # Save to database
            if db_save_chat(chat_id, chat_data):
                message_count = len(chat_data.get("messages", []))
                print(f"✓ {chat_id} - Migrated with {message_count} messages")
                stats["chats"]["migrated"] += 1
            else:
                print(f"✗ {chat_id} - Failed to save to database")
                stats["chats"]["errors"] += 1
        
        except Exception as e:
            print(f"✗ {chat_id} - Error: {e}")
            stats["chats"]["errors"] += 1


def migrate_image_metadata():
    """Migrate image metadata JSON to database."""
    print("\n" + "="*70)
    print("MIGRATING IMAGE METADATA")
    print("="*70)
    
    metadata_path = os.path.join(IMAGE_DIR, "image_metadata.json")
    
    if not os.path.exists(metadata_path):
        print("⚠ image_metadata.json not found")
        return
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        images = metadata.get("images", {})
        
        for filename, params in images.items():
            try:
                # Check if already in database
                existing = db_get_image_metadata(filename)
                if existing:
                    stats["images"]["skipped"] += 1
                    continue
                
                # Save to database
                if db_save_image_metadata(filename, params):
                    stats["images"]["migrated"] += 1
                else:
                    stats["images"]["errors"] += 1
            
            except Exception as e:
                print(f"✗ {filename} - Error: {e}")
                stats["images"]["errors"] += 1
        
        total = stats["images"]["migrated"] + stats["images"]["skipped"]
        print(f"✓ Processed {total} image metadata entries")
        print(f"  - Migrated: {stats['images']['migrated']}")
        print(f"  - Skipped: {stats['images']['skipped']}")
        if stats["images"]["errors"] > 0:
            print(f"  - Errors: {stats['images']['errors']}")
    
    except Exception as e:
        print(f"✗ Failed to load image_metadata.json: {e}")
        stats["images"]["errors"] += 1


def print_summary():
    """Print migration summary statistics."""
    print("\n" + "="*70)
    print("MIGRATION SUMMARY")
    print("="*70)
    
    categories = [
        ("Characters", stats["characters"]),
        ("World Info", stats["worlds"]),
        ("Chats", stats["chats"]),
        ("Image Metadata", stats["images"])
    ]
    
    for name, category_stats in categories:
        total = category_stats["migrated"] + category_stats["skipped"] + category_stats["errors"]
        if total > 0:
            print(f"\n{name}:")
            print(f"  Total processed: {total}")
            print(f"  ✓ Migrated:      {category_stats['migrated']}")
            print(f"  ⊙ Skipped:       {category_stats['skipped']}")
            if category_stats["errors"] > 0:
                print(f"  ✗ Errors:        {category_stats['errors']}")
    
    total_migrated = sum(s["migrated"] for s in stats.values())
    total_errors = sum(s["errors"] for s in stats.values())
    
    print("\n" + "="*70)
    if total_errors == 0:
        print(f"✓ MIGRATION COMPLETE - {total_migrated} items migrated successfully")
    else:
        print(f"⚠ MIGRATION COMPLETE WITH WARNINGS - {total_migrated} items migrated, {total_errors} errors")
    print("="*70)
    
    print(f"\nBackup location: {BACKUP_DIR}")
    print("\nNOTE: JSON files remain intact for SillyTavern compatibility.")
    print("The app will now use SQLite as the primary data source.")


def main():
    """Run the complete migration process."""
    print("\n" + "="*70)
    print("NEURALRP SQLITE MIGRATION")
    print("="*70)
    print("\nThis script will migrate your JSON data to SQLite database.")
    print("Your original JSON files will be backed up and remain intact.")
    
    response = input("\nProceed with migration? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Migration cancelled.")
        return
    
    # Run migration steps
    create_backup()
    migrate_characters()
    migrate_world_info()
    migrate_chats()
    migrate_image_metadata()
    print_summary()
    
    print("\n✓ Migration script completed successfully!")
    print("\nNext steps:")
    print("1. Review the summary above for any errors")
    print("2. Your JSON files are backed up in: " + BACKUP_DIR)
    print("3. The app will now use the SQLite database for all operations")
    print("4. JSON character files will continue to be auto-exported for compatibility")


if __name__ == "__main__":
    main()
