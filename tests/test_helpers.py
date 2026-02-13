"""
Tests for helper functions in main.py
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_character_name, get_entity_id, build_entity_name_mapping


class TestGetCharacterName:
    """Tests for get_character_name function."""

    def test_standard_character(self):
        """Test extracting name from standard character object."""
        char = {
            "data": {
                "name": "Alice"
            }
        }
        assert get_character_name(char) == "Alice"

    def test_character_with_surname(self):
        """Test extracting full name including surname."""
        char = {
            "data": {
                "name": "Alice Smith"
            }
        }
        assert get_character_name(char) == "Alice Smith"

    def test_character_with_nickname(self):
        """Test character with nickname format."""
        char = {
            "data": {
                "name": "Alice (Ali)"
            }
        }
        assert get_character_name(char) == "Alice (Ali)"

    def test_empty_character(self):
        """Test with empty character object."""
        char = {}
        assert get_character_name(char) == "Unknown"

    def test_missing_data_key(self):
        """Test when data key is missing."""
        char = {"other": "value"}
        assert get_character_name(char) == "Unknown"

    def test_none_character(self):
        """Test with None character."""
        assert get_character_name(None) == "Unknown"


class TestGetEntityId:
    """Tests for get_entity_id function."""

    def test_global_character(self):
        """Test extracting entity ID from global character."""
        char = {
            "_filename": "alice.json"
        }
        assert get_entity_id(char) == "alice.json"

    def test_local_npc(self):
        """Test extracting entity ID from local NPC."""
        char = {
            "entity_id": "npc_alice_123"
        }
        assert get_entity_id(char) == "npc_alice_123"

    def test_preference_for_filename(self):
        """Test that _filename takes precedence over entity_id."""
        char = {
            "_filename": "alice.json",
            "entity_id": "npc_alice_123"
        }
        assert get_entity_id(char) == "alice.json"

    def test_empty_character(self):
        """Test with empty character object."""
        char = {}
        assert get_entity_id(char) == "Unknown"

    def test_none_character(self):
        """Test with None character."""
        assert get_entity_id(None) == "Unknown"


class TestBuildEntityNameMapping:
    """Tests for build_entity_name_mapping function."""

    def test_single_character(self):
        """Test mapping with single character."""
        characters = [{
            "_filename": "alice.json",
            "data": {"name": "Alice"}
        }]
        result = build_entity_name_mapping(characters, {})
        assert result == {"alice.json": "Alice"}

    def test_multiple_characters(self):
        """Test mapping with multiple characters."""
        characters = [
            {"_filename": "alice.json", "data": {"name": "Alice"}},
            {"_filename": "bob.json", "data": {"name": "Bob"}}
        ]
        result = build_entity_name_mapping(characters, {})
        assert result == {"alice.json": "Alice", "bob.json": "Bob"}

    def test_with_npcs(self):
        """Test mapping includes NPCs."""
        characters = [{"_filename": "alice.json", "data": {"name": "Alice"}}]
        npcs = {
            "npc_merchant": {
                "name": "Merchant",
                "active": True
            }
        }
        result = build_entity_name_mapping(characters, npcs)
        assert result == {"alice.json": "Alice", "npc_merchant": "Merchant"}

    def test_empty_inputs(self):
        """Test with empty character and NPC lists."""
        result = build_entity_name_mapping([], {})
        assert result == {}

    def test_npc_without_name(self):
        """Test NPC missing name field."""
        characters = []
        npcs = {"npc_1": {"active": True}}
        result = build_entity_name_mapping(characters, npcs)
        assert result == {"npc_1": "Unknown NPC"}
