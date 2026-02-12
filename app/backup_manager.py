"""
NeuralRP Backup Manager

Handles automatic database backups with rotation and compression.
"""

import os
import shutil
import gzip
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
DB_PATH = os.path.join(DATA_DIR, "neuralrp.db")


def ensure_backup_dir() -> None:
    """Ensure backup directory exists."""
    os.makedirs(BACKUP_DIR, exist_ok=True)


def create_backup(backup_type: str = "manual", compress: bool = True) -> Optional[str]:
    """
    Create a database backup.

    Args:
        backup_type: Type of backup ('daily', 'weekly', 'manual', 'migration')
        compress: Whether to compress backup with gzip

    Returns:
        Path to created backup file, or None if failed
    """
    ensure_backup_dir()

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"[BACKUP] Database not found at {DB_PATH}")
        return None

    # Generate backup filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if compress:
        filename = f"neuralrp_{timestamp}_{backup_type}.db.gz"
    else:
        filename = f"neuralrp_{timestamp}_{backup_type}.db"

    backup_path = os.path.join(BACKUP_DIR, filename)

    try:
        # Create backup using SQLite API (ensures clean backup)
        conn = sqlite3.connect(DB_PATH)

        # Checkpoint WAL before backup (ensures all changes are in main DB)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Close connection before copying (ensures file is not locked)
        conn.close()

        # Copy database file
        if compress:
            # Compress with gzip
            with open(DB_PATH, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # Simple copy
            shutil.copy2(DB_PATH, backup_path)

        # Get file sizes
        db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
        backup_size_mb = os.path.getsize(backup_path) / (1024 * 1024)
        compression_ratio = (1 - (backup_size_mb / db_size_mb)) * 100 if compress else 0

        print(f"[BACKUP] Created {backup_type} backup: {filename}")
        print(f"[BACKUP]   DB size: {db_size_mb:.1f} MB, Backup size: {backup_size_mb:.1f} MB")
        if compress:
            print(f"[BACKUP]   Compression: {compression_ratio:.1f}%")

        return backup_path

    except Exception as e:
        print(f"[BACKUP ERROR] Failed to create backup: {e}")
        return None


def rotate_backups(retention_days: int = 30) -> int:
    """
    Delete old backups outside retention window.
    Keep only the most recent backup from each day within retention window.

    Args:
        retention_days: Number of days to keep backups

    Returns:
        Number of backups deleted
    """
    ensure_backup_dir()

    cutoff_date = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0

    # Group backups by date (keep most recent per day)
    backups_by_date = {}

    for filename in os.listdir(BACKUP_DIR):
        if not filename.startswith("neuralrp_") or not (filename.endswith(".db") or filename.endswith(".db.gz")):
            continue

        backup_path = os.path.join(BACKUP_DIR, filename)

        # Extract timestamp from filename
        # Format: neuralrp_YYYY-MM-DD_HHMMSS_type.db.gz
        try:
            parts = filename.replace(".db", "").replace(".gz", "").split("_")
            date_str = parts[1]  # YYYY-MM-DD
            timestamp_str = parts[2]  # HHMMSS

            backup_date = datetime.strptime(f"{date_str}_{timestamp_str}", "%Y-%m-%d_%H%M%S")

            # Delete backups older than retention window
            if backup_date < cutoff_date:
                os.remove(backup_path)
                deleted_count += 1
                print(f"[BACKUP] Deleted old backup: {filename}")
                continue

            # Keep only most recent backup per day
            date_key = backup_date.date()
            if date_key in backups_by_date:
                # Compare timestamps, keep newer one
                existing_path = backups_by_date[date_key]
                if backup_date > datetime.fromtimestamp(os.path.getmtime(existing_path)):
                    os.remove(existing_path)
                    deleted_count += 1
                    print(f"[BACKUP] Deleted duplicate backup (older): {os.path.basename(existing_path)}")
                    backups_by_date[date_key] = backup_path
                else:
                    os.remove(backup_path)
                    deleted_count += 1
                    print(f"[BACKUP] Deleted duplicate backup (older): {filename}")
            else:
                backups_by_date[date_key] = backup_path

        except (ValueError, IndexError) as e:
            print(f"[BACKUP WARNING] Could not parse filename {filename}: {e}")
            continue

    if deleted_count > 0:
        print(f"[BACKUP] Rotation complete: deleted {deleted_count} old backups")
    else:
        print(f"[BACKUP] Rotation complete: no old backups to delete")

    return deleted_count


def list_backups() -> List[dict]:
    """
    List all backups with metadata.

    Returns:
        List of backup info dictionaries
    """
    ensure_backup_dir()

    backups = []

    for filename in os.listdir(BACKUP_DIR):
        if not filename.startswith("neuralrp_") or not (filename.endswith(".db") or filename.endswith(".db.gz")):
            continue

        backup_path = os.path.join(BACKUP_DIR, filename)

        try:
            parts = filename.replace(".db", "").replace(".gz", "").split("_")
            date_str = parts[1]  # YYYY-MM-DD
            timestamp_str = parts[2]  # HHMMSS
            backup_type = parts[3] if len(parts) > 3 else "unknown"

            backup_date = datetime.strptime(f"{date_str}_{timestamp_str}", "%Y-%m-%d_%H%M%S")
            size_mb = os.path.getsize(backup_path) / (1024 * 1024)
            is_compressed = filename.endswith(".gz")

            backups.append({
                "filename": filename,
                "path": backup_path,
                "date": backup_date.strftime("%Y-%m-%d %H:%M:%S"),
                "type": backup_type,
                "size_mb": round(size_mb, 2),
                "compressed": is_compressed
            })

        except (ValueError, IndexError):
            continue

    # Sort by date (newest first)
    backups.sort(key=lambda x: x["date"], reverse=True)

    return backups


def restore_backup(backup_path: str) -> bool:
    """
    Restore database from backup.

    WARNING: This replaces the current database!

    Args:
        backup_path: Path to backup file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Verify backup exists
        if not os.path.exists(backup_path):
            print(f"[BACKUP ERROR] Backup not found: {backup_path}")
            return False

        # Decompress if needed
        temp_db_path = backup_path
        if backup_path.endswith(".gz"):
            temp_db_path = backup_path[:-3]  # Remove .gz
            with gzip.open(backup_path, 'rb') as f_in:
                with open(temp_db_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # Create emergency backup of current DB before restore
        current_backup = create_backup("before_restore", compress=False)
        if current_backup:
            print(f"[BACKUP] Emergency backup created: {os.path.basename(current_backup)}")

        # Replace database
        shutil.copy2(temp_db_path, DB_PATH)

        # Cleanup temporary decompressed file
        if backup_path.endswith(".gz"):
            os.remove(temp_db_path)

        print(f"[BACKUP] Database restored from: {os.path.basename(backup_path)}")
        return True

    except Exception as e:
        print(f"[BACKUP ERROR] Failed to restore backup: {e}")
        return False
