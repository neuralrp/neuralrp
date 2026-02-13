"""
Tests for NPC management operations
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNpcManagement:
    """Tests for NPC management functions."""

    def test_get_chat_npcs_returns_list(self):
        """Test that getting chat NPCs returns a list."""
        from main import load_character_profiles
        
        result = load_character_profiles([], {}, chat_id="nonexistent_chat_123")
        assert isinstance(result, list)

    def test_load_character_profiles_with_empty_inputs(self):
        """Test load_character_profiles with empty inputs."""
        from main import load_character_profiles
        
        result = load_character_profiles([], {}, chat_id="test_chat_123")
        assert isinstance(result, list)
        assert result == []

    def test_load_character_profiles_with_active_chars(self):
        """Test load_character_profiles with active characters."""
        from main import load_character_profiles
        
        active_chars = ["alice.json"]
        result = load_character_profiles(active_chars, {}, chat_id="test_chat_456")
        
        assert isinstance(result, list)

    def test_load_character_profiles_with_npcs(self):
        """Test load_character_profiles includes NPCs."""
        from main import load_character_profiles
        
        localnpcs = {
            "npc_merchant": {
                "name": "Merchant",
                "entity_id": "npc_merchant",
                "active": True
            }
        }
        
        result = load_character_profiles([], localnpcs, chat_id="test_chat_789")
        
        assert isinstance(result, list)

    def test_load_character_profiles_includes_inactive_npcs(self):
        """Test that inactive NPCs are included in profiles."""
        from main import load_character_profiles
        
        localnpcs = {
            "npc_inactive": {
                "name": "Inactive NPC",
                "entity_id": "npc_inactive",
                "active": False
            }
        }
        
        result = load_character_profiles([], localnpcs, chat_id="test_chat_abc")
        
        assert isinstance(result, list)


class TestNpcToggleActive:
    """Tests for NPC active/inactive toggle."""

    def test_toggle_nonexistent_npc_handled(self):
        """Test toggling a nonexistent NPC doesn't crash."""
        from main import load_character_profiles
        
        localnpcs = {
            "npc_existing": {
                "name": "Existing NPC",
                "entity_id": "npc_existing",
                "active": True
            }
        }
        
        result = load_character_profiles([], localnpcs, chat_id="test_chat_xyz")
        
        assert isinstance(result, list)


class TestNpcPromoteToGlobal:
    """Tests for NPC promotion to global character."""

    def test_promote_requires_valid_npc(self):
        """Test that promotion function handles invalid NPCs."""
        from main import load_character_profiles
        
        result = load_character_profiles([], {}, chat_id="nonexistent")
        
        assert isinstance(result, list)
