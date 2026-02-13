"""
Tests for prompt building functions in main.py
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildSceneCastBlock:
    """Tests for build_scene_cast_block function."""

    def test_single_character(self):
        """Test SCENE CAST with single character."""
        from main import build_scene_cast_block
        
        characters = [{
            "_filename": "alice.json",
            "data": {"name": "Alice"},
            "is_active": True
        }]
        
        capsules = {"alice.json": "Alice is brave."}
        
        result = build_scene_cast_block(characters, capsules)
        
        assert "Alice is brave" in result
        assert "SCENE CAST" in result

    def test_multiple_characters(self):
        """Test SCENE CAST with multiple characters."""
        from main import build_scene_cast_block
        
        characters = [
            {"_filename": "alice.json", "data": {"name": "Alice"}, "is_active": True},
            {"_filename": "bob.json", "data": {"name": "Bob"}, "is_active": True}
        ]
        
        capsules = {"alice.json": "Alice is brave.", "bob.json": "Bob is clever."}
        
        result = build_scene_cast_block(characters, capsules)
        
        assert "Alice is brave" in result
        assert "Bob is clever" in result

    def test_excludes_specified_characters(self):
        """Test excluding characters from SCENE CAST."""
        from main import build_scene_cast_block
        
        characters = [
            {"_filename": "alice.json", "data": {"name": "Alice"}, "is_active": True},
            {"_filename": "bob.json", "data": {"name": "Bob"}, "is_active": True}
        ]
        
        capsules = {"alice.json": "Alice is brave.", "bob.json": "Bob is clever."}
        
        result = build_scene_cast_block(characters, capsules, exclude=["alice.json"])
        
        assert "Alice is brave" not in result
        assert "Bob is clever" in result

    def test_empty_list(self):
        """Test with empty character list."""
        from main import build_scene_cast_block
        
        result = build_scene_cast_block([], {})
        assert result == ""

    def test_uses_fallback_when_no_capsule(self):
        """Test fallback capsule when capsule is not provided."""
        from main import build_scene_cast_block
        
        characters = [{
            "_filename": "alice.json",
            "data": {
                "name": "Alice",
                "description": "A brave adventurer",
                "personality": "Brave and bold"
            },
            "is_active": True
        }]
        
        result = build_scene_cast_block(characters, {})
        
        assert "Alice" in result
        assert "SCENE CAST" in result


class TestBuildSceneUpdateBlock:
    """Tests for build_scene_update_block function."""

    def test_character_arrived(self):
        """Test SCENE UPDATE when character joins."""
        from main import build_scene_update_block
        
        arrived = {"alice.json"}
        entity_to_name = {"alice.json": "Alice"}
        
        result = build_scene_update_block(set(), arrived, entity_to_name)
        
        assert "Alice" in result
        assert "entered" in result.lower() or "arrived" in result.lower()

    def test_character_departed(self):
        """Test SCENE UPDATE when character leaves."""
        from main import build_scene_update_block
        
        departed = {"bob.json"}
        entity_to_name = {"bob.json": "Bob"}
        
        result = build_scene_update_block(departed, set(), entity_to_name)
        
        assert "Bob" in result

    def test_multiple_changes(self):
        """Test SCENE UPDATE with multiple changes."""
        from main import build_scene_update_block
        
        departed = {"alice.json"}
        arrived = {"charlie.json"}
        entity_to_name = {"alice.json": "Alice", "charlie.json": "Charlie"}
        
        result = build_scene_update_block(departed, arrived, entity_to_name)
        
        assert "Alice" in result
        assert "Charlie" in result


class TestBuildFallbackCapsule:
    """Tests for build_fallback_capsule function."""

    def test_basic_capsule(self):
        """Test basic capsule generation."""
        from main import build_fallback_capsule
        
        data = {
            "description": "A brave knight who protects the realm.",
            "personality": "Bold and honorable"
        }
        
        result = build_fallback_capsule("Alice", data)
        
        assert "Alice" in result or "brave" in result.lower() or "knight" in result.lower()

    def test_minimal_data(self):
        """Test capsule with minimal data."""
        from main import build_fallback_capsule
        
        data = {}
        
        result = build_fallback_capsule("Test", data)
        
        assert "No description available" in result


class TestSplitMessagesByWindow:
    """Tests for split_messages_by_window function."""

    def test_short_messages(self):
        """Test with messages below window limit."""
        from main import split_messages_by_window
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
        
        messages = [
            MockMessage(role="user", content="Hello"),
            MockMessage(role="assistant", content="Hi there")
        ]
        
        recent, older = split_messages_by_window(messages, max_exchanges=5)
        
        assert len(recent) == 2
        assert len(older) == 0

    def test_messages_above_window(self):
        """Test with messages above window limit."""
        from main import split_messages_by_window
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
        
        messages = [
            MockMessage(role="user", content=f"Message {i}") for i in range(10)
        ]
        
        recent, older = split_messages_by_window(messages, max_exchanges=5)
        
        assert len(recent) == 5
        assert len(older) == 5
