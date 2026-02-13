"""
Tests for cast change detection
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCastChangeDetection:
    """Tests for detect_cast_change function."""

    def test_no_previous_cast(self):
        """Test when there's no previous cast metadata."""
        from main import detect_cast_change
        
        current_chars = [{"_filename": "alice.json"}]
        current_npcs = {}
        previous = {}
        
        changed, departed, arrived, _ = detect_cast_change(current_chars, current_npcs, "narrator", previous)
        
        assert changed is True

    def test_empty_current_cast(self):
        """Test when current cast is empty but previous had characters."""
        from main import detect_cast_change
        
        current_chars = []
        current_npcs = {}
        previous = {"previous_active_cast": ["alice.json"]}
        
        changed, departed, arrived, _ = detect_cast_change(current_chars, current_npcs, "narrator", previous)
        
        assert changed is True
        assert "alice.json" in departed

    def test_unchanged_cast(self):
        """Test when cast hasn't changed."""
        from main import detect_cast_change
        
        current_chars = [
            {"_filename": "alice.json"},
            {"_filename": "bob.json"}
        ]
        current_npcs = {}
        previous = {"previous_active_cast": ["alice.json", "bob.json"]}
        
        changed, departed, arrived, _ = detect_cast_change(current_chars, current_npcs, "narrator", previous)
        
        assert changed is False

    def test_focus_change_not_cast_change(self):
        """Test that focus change doesn't trigger cast change."""
        from main import detect_cast_change
        
        current_chars = [
            {"_filename": "alice.json"},
            {"_filename": "bob.json"}
        ]
        current_npcs = {}
        previous = {
            "previous_active_cast": ["alice.json", "bob.json"],
            "previous_focus_character": "alice.json"
        }
        
        changed, departed, arrived, _ = detect_cast_change(current_chars, current_npcs, "focus:bob", previous)
        
        assert changed is False


class TestSceneUpdateBlock:
    """Tests for build_scene_update_block function."""

    def test_both_departed_and_arrived(self):
        """Test scene update with both departures and arrivals."""
        from main import build_scene_update_block
        
        departed = {"alice.json", "bob.json"}
        arrived = {"charlie.json"}
        entity_to_name = {
            "alice.json": "Alice",
            "bob.json": "Bob",
            "charlie.json": "Charlie"
        }
        
        result = build_scene_update_block(departed, arrived, entity_to_name)
        
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result

    def test_empty_changes(self):
        """Test scene update with no changes."""
        from main import build_scene_update_block
        
        result = build_scene_update_block(set(), set(), {})
        
        assert result == ""

    def test_unknown_entity_name(self):
        """Test scene update when entity name mapping is missing."""
        from main import build_scene_update_block
        
        departed = {"unknown.json"}
        arrived = set()
        entity_to_name = {}
        
        result = build_scene_update_block(departed, arrived, entity_to_name)
        
        assert "unknown.json" in result
