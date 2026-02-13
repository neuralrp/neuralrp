"""
Tests for database operations in app/database.py
"""

import pytest
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatabaseConnection:
    """Tests for database connection handling."""

    def test_thread_local_storage(self):
        """Test that thread-local storage is properly configured."""
        from app.database import _thread_local
        assert hasattr(_thread_local, 'connection')

    def test_connection_context_manager(self):
        """Test get_connection context manager yields a connection."""
        from app.database import get_connection
        with get_connection() as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

    def test_connection_has_row_factory(self):
        """Test that connection uses Row factory."""
        from app.database import get_connection
        with get_connection() as conn:
            assert conn.row_factory == sqlite3.Row


class TestDatabaseSchema:
    """Tests for database schema operations."""

    def test_characters_table_exists(self):
        """Test that characters table is created."""
        from app.database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='characters'")
        result = cursor.fetchone()
        conn.close()
        assert result is not None

    def test_worlds_table_exists(self):
        """Test that worlds table is created."""
        from app.database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='worlds'")
        result = cursor.fetchone()
        conn.close()
        assert result is not None

    def test_messages_table_exists(self):
        """Test that messages table is created."""
        from app.database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        result = cursor.fetchone()
        conn.close()
        assert result is not None


class TestDatabaseCRUD:
    """Tests for basic CRUD operations using the real database."""

    def test_get_nonexistent_character_returns_none(self):
        """Test that getting nonexistent character returns None."""
        from app.database import db_get_character
        
        result = db_get_character("nonexistent_character_xyz_123")
        assert result is None

    def test_get_nonexistent_world_returns_none(self):
        """Test that getting nonexistent world returns None."""
        from app.database import db_get_world
        
        result = db_get_world("nonexistent_world_xyz_789")
        assert result is None

    def test_character_save_and_get_roundtrip(self):
        """Test saving and retrieving a character in a transaction."""
        from app.database import db_save_character, db_get_character, db_delete_character
        
        test_filename = "__test_char_roundtrip__.json"
        char_data = {
            "data": {
                "name": "Test Character Roundtrip",
                "description": "A test character for roundtrip testing"
            }
        }
        
        try:
            db_save_character(char_data, test_filename)
            retrieved = db_get_character(test_filename)
            
            assert retrieved is not None
            assert retrieved["data"]["name"] == "Test Character Roundtrip"
        finally:
            db_delete_character(test_filename)

    def test_get_all_chats(self):
        """Test getting all chats returns a list."""
        from app.database import db_get_all_chats
        
        chats = db_get_all_chats()
        assert isinstance(chats, list)

    def test_get_all_characters(self):
        """Test getting all characters returns a list."""
        from app.database import db_get_all_characters
        
        characters = db_get_all_characters()
        assert isinstance(characters, list)

    def test_get_all_worlds(self):
        """Test getting all worlds returns a list."""
        from app.database import db_get_all_worlds
        
        worlds = db_get_all_worlds()
        assert isinstance(worlds, list)


class TestDatabaseIntegrity:
    """Tests for database integrity and constraints."""

    def test_database_has_required_tables(self):
        """Test that all required tables exist."""
        from app.database import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        required_tables = {
            'characters', 'worlds', 'world_entries', 
            'messages', 'image_metadata', 'change_log'
        }
        
        for table in required_tables:
            assert table in tables, f"Required table '{table}' not found"

    def test_database_foreign_keys_enabled(self):
        """Test that foreign keys are enabled."""
        from app.database import get_connection
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()
            assert result[0] == 1
