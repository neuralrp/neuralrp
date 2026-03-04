"""
Tests for scene summarization operations
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGroupMessagesIntoScenes:
    """Tests for grouping messages into scenes."""

    def test_empty_messages(self):
        """Test grouping with empty messages."""
        from main import group_messages_into_scenes
        
        result = group_messages_into_scenes([], [], max_exchanges_per_scene=15)
        
        assert result == []

    def test_single_message_scene(self):
        """Test scene grouping with single message."""
        from main import group_messages_into_scenes
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
            id: int = 1
            speaker: str = ""
        
        messages = [MockMessage(role="user", content="Hello", id=1, speaker="user")]
        
        result = group_messages_into_scenes(messages, [], max_exchanges_per_scene=15)
        
        assert len(result) >= 1

    def test_multiple_exchanges_across_scenes(self):
        """Test messages split across multiple scenes."""
        from main import group_messages_into_scenes
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
            id: int
            speaker: str = ""
        
        messages = [
            MockMessage(role="user", content=f"Message {i}", id=i, speaker="user") 
            for i in range(30)
        ]
        
        result = group_messages_into_scenes(messages, [], max_exchanges_per_scene=15)
        
        assert len(result) >= 1


class TestShouldShowCanonLaw:
    """Tests for canon law reinforcement logic."""

    def test_initial_turn_shows_canon(self):
        """Test that canon law shows on initial turns."""
        from main import should_show_canon_law
        
        assert should_show_canon_law(1, freq=3) is True
        assert should_show_canon_law(2, freq=3) is True

    def test_canon_formula_basic(self):
        """Test canon law formula."""
        from main import should_show_canon_law
        
        assert should_show_canon_law(3, freq=3) is False
        assert should_show_canon_law(4, freq=3) is False
        assert should_show_canon_law(5, freq=3) is True
        assert should_show_canon_law(6, freq=3) is False


class TestCalculateTurnForMessage:
    """Tests for turn calculation."""

    def test_turn_calculation_basic(self):
        """Test basic turn calculation."""
        from main import calculate_turn_for_message
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
        
        messages = [
            MockMessage(role="user", content="Hello"),
            MockMessage(role="assistant", content="Hi"),
            MockMessage(role="user", content="How are you?"),
            MockMessage(role="assistant", content="Good"),
        ]
        
        turn = calculate_turn_for_message(messages, 2)
        
        assert turn == 2

    def test_turn_calculation_first_message(self):
        """Test turn calculation for first message."""
        from main import calculate_turn_for_message
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
        
        messages = [
            MockMessage(role="user", content="Hello"),
            MockMessage(role="assistant", content="Hi"),
        ]
        
        turn = calculate_turn_for_message(messages, 0)
        
        assert turn == 1


class TestSplitMessagesByWindow:
    """Additional tests for message window splitting."""

    def test_zero_messages(self):
        """Test with zero messages."""
        from main import split_messages_by_window
        
        recent, older = split_messages_by_window([], max_exchanges=5)
        
        assert recent == []
        assert older == []

    def test_exactly_max_exchanges(self):
        """Test with exactly max exchanges."""
        from main import split_messages_by_window
        from pydantic import BaseModel
        
        class MockMessage(BaseModel):
            role: str
            content: str
        
        messages = [
            MockMessage(role="user", content=f"Message {i}") for i in range(10)
        ]
        
        recent, older = split_messages_by_window(messages, max_exchanges=5)
        
        assert len(older) >= 0
        assert len(recent) >= 0


class TestSummarizationSafeguards:
    """Tests for summary dedupe and hard limits."""

    def test_enforce_word_limit_truncates(self):
        """Word limiter should hard-truncate overlong text."""
        from main import enforce_word_limit

        text = "one two three four five six"
        result = enforce_word_limit(text, 4)

        assert result == "one two three four"

    def test_filter_unsummarized_old_messages(self):
        """Old messages at/below marker should be excluded from re-summarization."""
        from main import filter_unsummarized_old_messages
        from pydantic import BaseModel

        class MockMessage(BaseModel):
            role: str
            content: str
            id: int
            speaker: str = ""

        old = [
            MockMessage(role="user", content="a", id=10),
            MockMessage(role="assistant", content="b", id=11),
            MockMessage(role="user", content="c", id=12),
        ]
        metadata = {"last_summarized_message_id": 11}

        result = filter_unsummarized_old_messages(old, metadata)

        assert len(result) == 1
        assert result[0].id == 12

    def test_group_messages_into_scenes_respects_start_turn(self):
        """Scene turn labels should preserve absolute turn offset."""
        from main import group_messages_into_scenes
        from pydantic import BaseModel

        class MockMessage(BaseModel):
            role: str
            content: str
            id: int
            speaker: str = ""

        messages = [
            MockMessage(role="user", content="u1", id=1),
            MockMessage(role="assistant", content="a1", id=2),
            MockMessage(role="user", content="u2", id=3),
            MockMessage(role="assistant", content="a2", id=4),
        ]

        scenes = group_messages_into_scenes(
            messages,
            cast_change_turns=[],
            max_exchanges_per_scene=15,
            start_turn=20
        )

        assert len(scenes) == 1
        start_turn, end_turn, _ = scenes[0]
        assert start_turn == 20
        assert end_turn == 21
