"""
Tests for world info operations
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWorldInfoRetrieval:
    """Tests for world info retrieval."""

    def test_get_nonexistent_world_returns_none(self):
        """Test that getting nonexistent world returns None."""
        from app.database import db_get_world
        
        result = db_get_world("nonexistent_world_xyz_999")
        assert result is None

    def test_get_all_worlds_returns_list(self):
        """Test that getting all worlds returns a list."""
        from app.database import db_get_all_worlds
        
        result = db_get_all_worlds()
        assert isinstance(result, (list, tuple))


class TestWorldInfoKeywordMatching:
    """Tests for world info keyword matching logic."""

    def test_preprocess_world_info_basic(self):
        """Test basic world info preprocessing."""
        from main import preprocess_world_info
        
        world_info = {
            "test_world": {
                "entries": {}
            }
        }
        
        result = preprocess_world_info(world_info)
        
        assert "test_world" in result

    def test_cached_world_entries_with_empty_text(self):
        """Test cached world entries with empty text."""
        from main import get_cached_world_entries
        
        world_info = {}
        
        result = get_cached_world_entries(world_info, "", max_entries=10)
        
        assert isinstance(result, (list, tuple))


class TestWorldInfoSemanticSearch:
    """Tests for semantic search in world info."""

    def test_semantic_search_engine_init(self):
        """Test that semantic search engine initializes."""
        from main import semantic_search_engine
        
        assert semantic_search_engine is not None


class TestWorldInfoEntries:
    """Tests for world info entry operations."""

    def test_get_world_entry_timestamps(self):
        """Test getting world entry timestamps."""
        from app.database import db_get_world_entry_timestamps
        
        result = db_get_world_entry_timestamps("nonexistent")
        
        assert isinstance(result, dict)

    def test_get_all_worlds_returns_list(self):
        """Test that getting all worlds returns a list."""
        from app.database import db_get_all_worlds
        
        result = db_get_all_worlds()
        assert isinstance(result, (list, tuple))

    def test_world_info_with_empty_entries(self):
        """Test world info structure with empty entries."""
        from app.database import db_save_world, db_get_world
        
        test_world = "__test_empty_world__"
        
        try:
            db_save_world(test_world, {}, {})
            result = db_get_world(test_world)
            
            assert result is not None
        finally:
            from app.database import get_connection
            with get_connection() as conn:
                conn.execute("DELETE FROM worlds WHERE name = ?", (test_world,))
                conn.commit()


class TestWorldInfoKeywordMatching:
    """Tests for world info keyword matching logic."""

    def test_preprocess_world_info_basic(self):
        """Test basic world info preprocessing."""
        from main import preprocess_world_info
        
        world_info = {
            "test_world": {
                "entries": {}
            }
        }
        
        result = preprocess_world_info(world_info)
        
        assert "test_world" in result

    def test_cached_world_entries_with_empty_text(self):
        """Test cached world entries with empty text."""
        from main import get_cached_world_entries
        
        world_info = {}
        
        result = get_cached_world_entries(world_info, "", max_entries=10)
        
        assert isinstance(result, (list, tuple))


class TestWorldInfoSemanticSearch:
    """Tests for semantic search in world info."""

    def test_semantic_search_engine_init(self):
        """Test that semantic search engine initializes."""
        from main import semantic_search_engine
        
        assert semantic_search_engine is not None


class TestWorldInfoEntries:
    """Tests for world info entry operations."""

    def test_get_world_entry_timestamps(self):
        """Test getting world entry timestamps."""
        from app.database import db_get_world_entry_timestamps
        
        result = db_get_world_entry_timestamps("nonexistent")
        
        assert isinstance(result, dict)
