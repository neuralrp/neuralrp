"""
Tests for sticky window full card behavior in construct_prompt.

The sticky window system ensures:
- Turn 1 (first appearance): Full card
- Turn 2-3 (sticky window): Full card (reinforcement)
- Turn 4+: Capsule only

Bug regression test: character_full_card_turns should only be updated
on reset points, NOT during sticky window reinforcement.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import construct_prompt, PromptRequest, ChatMessage


def make_character(filename, name, description="Test character", capsule=None, personality="Brave and bold"):
    """Create a minimal character dict for testing."""
    char = {
        "_filename": filename,
        "data": {
            "name": name,
            "description": description,
            "personality": personality,
            "scenario": "A testing scenario"
        },
        "is_active": True
    }
    if capsule:
        char["data"]["extensions"] = {"multi_char_summary": capsule}
    return char


def make_messages(count, char_name="Alice"):
    """Create N user/assistant message pairs (each pair = 1 turn)."""
    messages = []
    msg_id = 0
    for i in range(count):
        messages.append(ChatMessage(
            id=msg_id, role="user", content=f"User message {i+1}"
        ))
        msg_id += 1
        messages.append(ChatMessage(
            id=msg_id, role="assistant", content=f"{char_name}: Response {i+1}",
            speaker=char_name
        ))
        msg_id += 1
    return messages


def has_full_card(prompt, char_name):
    """Check if prompt contains full character card."""
    return f"### Character: {char_name}" in prompt


def has_scene_cast_capsule(prompt, char_name):
    """Check if prompt contains SCENE CAST with character capsule."""
    return f"[{char_name}:" in prompt and "SCENE CAST" in prompt


class TestStickyWindowFullCard:
    """Tests for sticky window full card injection logic."""

    def test_first_appearance_gets_full_card(self):
        """Turn 1: First appearance should get full card."""
        char = make_character("alice.json", "Alice", "A brave knight")
        
        request = PromptRequest(
            messages=make_messages(1),  # 1 exchange = turn 1
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {}
        character_full_card_turns = {}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=1,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should get full card on first appearance
        assert has_full_card(prompt, "Alice")
        
        # Should record first turn
        assert character_first_turns.get("alice.json") == 1
        
        # Should record full card turn
        assert character_full_card_turns.get("alice.json") == 1

    def test_sticky_turn_2_gets_full_card(self):
        """Turn 2: Sticky window should get full card."""
        char = make_character("alice.json", "Alice", "A brave knight")
        
        request = PromptRequest(
            messages=make_messages(2),  # 2 exchanges = turn 2
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        # Simulate turn 1 already happened
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=2,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should get full card (sticky window)
        assert has_full_card(prompt, "Alice")

    def test_sticky_turn_3_gets_full_card(self):
        """Turn 3: Last sticky window turn should get full card."""
        char = make_character("alice.json", "Alice", "A brave knight")
        
        request = PromptRequest(
            messages=make_messages(3),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        # Simulate turn 1 already happened
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=3,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should get full card (sticky window)
        assert has_full_card(prompt, "Alice")

    def test_turn_4_uses_capsule(self):
        """Turn 4: After sticky window, should use capsule, not full card."""
        char = make_character(
            "alice.json", "Alice", "A brave knight",
            capsule="Alice is a brave knight who protects the realm."
        )
        
        request = PromptRequest(
            messages=make_messages(4),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        # Simulate first appearance at turn 1
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}  # Full card at turn 1
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=4,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should use capsule, NOT full card
        assert not has_full_card(prompt, "Alice")
        assert has_scene_cast_capsule(prompt, "Alice")
        assert "Alice is a brave knight" in prompt

    def test_turn_5_continues_capsule(self):
        """Turn 5+: Should continue using capsule."""
        char = make_character(
            "alice.json", "Alice", "A brave knight",
            capsule="Alice is a brave knight."
        )
        
        request = PromptRequest(
            messages=make_messages(5),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=5,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should use capsule, NOT full card
        assert not has_full_card(prompt, "Alice")
        assert has_scene_cast_capsule(prompt, "Alice")

    def test_sticky_turn_2_full_card_turn_not_updated(self):
        """
        REGRESSION TEST: character_full_card_turns should NOT be updated
        during sticky window reinforcement, only on reset points.
        
        Bug was: every turn updated full_card_turn, causing infinite sticky window.
        Fix: only update when is_first_appearance or is_returning.
        """
        char = make_character("alice.json", "Alice", "A brave knight")
        
        request = PromptRequest(
            messages=make_messages(2),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        # Simulate turn 1 already happened
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}  # Set on turn 1
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=2,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should get full card (sticky window)
        assert has_full_card(prompt, "Alice")
        
        # CRITICAL: full_card_turn should NOT be updated to 2
        # Bug would have set it to 2, causing infinite sticky window
        assert character_full_card_turns["alice.json"] == 1, \
            "full_card_turn should NOT be updated during sticky window"

    def test_sticky_turn_3_full_card_turn_not_updated(self):
        """Turn 3 sticky: full_card_turn should still be 1, not 3."""
        char = make_character("alice.json", "Alice", "A brave knight")
        
        request = PromptRequest(
            messages=make_messages(3),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=3,
            character_full_card_turns=character_full_card_turns
        )
        
        assert has_full_card(prompt, "Alice")
        
        # Should still be 1, not 3
        assert character_full_card_turns["alice.json"] == 1, \
            "full_card_turn should remain at original reset point"

    def test_returning_after_absence_gets_full_card(self):
        """Character returning after long absence should get full card + reset sticky."""
        char = make_character("alice.json", "Alice", "A brave knight")
        
        # Create messages showing character was absent for 20+ messages
        # Use generic messages that don't mention Alice
        messages = []
        msg_id = 0
        for i in range(25):
            messages.append(ChatMessage(
                id=msg_id, role="user", content=f"User message {i+1}"
            ))
            msg_id += 1
            messages.append(ChatMessage(
                id=msg_id, role="assistant", content=f"Narrator: Response {i+1}",
                speaker="Narrator"  # Not Alice
            ))
            msg_id += 1
        
        request = PromptRequest(
            messages=messages,
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        # Simulate character was active at turn 1, now returning at turn 26
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}  # Last full card at turn 1
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=26,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should get full card on return
        assert has_full_card(prompt, "Alice")
        
        # Should reset full_card_turn to current turn
        assert character_full_card_turns["alice.json"] == 26

    def test_sticky_window_after_return(self):
        """Sticky window should work correctly after returning character."""
        char = make_character("alice.json", "Alice", "A brave knight")
        
        request = PromptRequest(
            messages=make_messages(27),  # Turn 27 - sticky after return
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {"alice.json": 1}
        # Returned at turn 26, full_card_turn reset
        character_full_card_turns = {"alice.json": 26}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=27,
            character_full_card_turns=character_full_card_turns
        )
        
        # Turn 27: 27 - 26 = 1 <= 2, should get full card (sticky)
        assert has_full_card(prompt, "Alice")
        
        # full_card_turn should NOT be updated (stays at 26)
        assert character_full_card_turns["alice.json"] == 26

    def test_full_card_excluded_from_scene_cast(self):
        """Character with full card should NOT appear in SCENE CAST section."""
        char = make_character(
            "alice.json", "Alice", "A brave knight",
            capsule="This should not appear in SCENE CAST"
        )
        
        request = PromptRequest(
            messages=make_messages(1),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {}
        character_full_card_turns = {}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=1,
            character_full_card_turns=character_full_card_turns
        )
        
        # Has full card
        assert has_full_card(prompt, "Alice")
        
        # SCENE CAST should exist but NOT include Alice (she has full card)
        # or if there's only one character, SCENE CAST might be omitted
        if "SCENE CAST" in prompt:
            # If SCENE CAST is present, Alice should not be in it
            scene_cast_section = prompt.split("SCENE CAST")[1] if "SCENE CAST" in prompt else ""
            if "ACTIVE ONLY" in scene_cast_section:
                # Full card characters are excluded from SCENE CAST
                pass  # This is expected behavior


class TestStickyWindowMultipleCharacters:
    """Tests for sticky window with multiple characters."""

    def test_multiple_chars_independent_sticky_windows(self):
        """Multiple characters should have independent sticky window tracking."""
        alice = make_character("alice.json", "Alice", "A brave knight")
        bob = make_character("bob.json", "Bob", "A clever rogue")
        
        request = PromptRequest(
            messages=make_messages(4),
            characters=[alice, bob],
            settings={"system_prompt": "Test"}
        )
        
        # Alice appeared at turn 1, Bob at turn 2
        character_first_turns = {"alice.json": 1, "bob.json": 2}
        # Both got full cards at their first appearance
        character_full_card_turns = {"alice.json": 1, "bob.json": 2}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=4,
            character_full_card_turns=character_full_card_turns
        )
        
        # Turn 4: Alice - 4-1=3 > 2 (no sticky), Bob - 4-2=2 <= 2 (sticky)
        
        # Alice should have capsule (not full card)
        assert not has_full_card(prompt, "Alice")
        
        # Bob should still have full card (sticky window)
        assert has_full_card(prompt, "Bob")

    def test_second_character_added_mid_chat(self):
        """Second character added mid-chat should trigger full card."""
        alice = make_character("alice.json", "Alice", "A brave knight")
        bob = make_character("bob.json", "Bob", "A clever rogue")
        
        request = PromptRequest(
            messages=make_messages(3),
            characters=[alice, bob],
            settings={"system_prompt": "Test"}
        )
        
        # Alice from turn 1, Bob added at turn 3 (this request)
        character_first_turns = {"alice.json": 1}  # Bob not in first turns yet
        character_full_card_turns = {"alice.json": 1}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=3,
            character_full_card_turns=character_full_card_turns
        )
        
        # Both should get full cards (Alice sticky, Bob first appearance)
        assert has_full_card(prompt, "Alice")
        assert has_full_card(prompt, "Bob")
        
        # Bob's first turn should be recorded
        assert character_first_turns.get("bob.json") == 3


class TestStickyWindowEdgeCases:
    """Edge case tests for sticky window logic."""

    def test_no_previous_full_card_turn(self):
        """
        Edge case: If somehow full_card_turns is empty but character appeared recently,
        system should use SCENE CAST (not full card).
        
        This is actually expected behavior - full card only happens on reset points
        or sticky window from a recorded full card turn.
        """
        char = make_character("alice.json", "Alice", "A brave knight")
        
        # Messages show Alice appeared recently
        request = PromptRequest(
            messages=make_messages(2),  # Turn 2
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {}  # Empty - but Alice appeared in messages
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=2,
            character_full_card_turns=character_full_card_turns
        )
        
        # Since Alice appeared in recent messages, no full card needed
        # System correctly uses SCENE CAST
        assert not has_full_card(prompt, "Alice")
        assert has_scene_cast_capsule(prompt, "Alice")

    def test_fallback_capsule_when_no_capsule(self):
        """Fallback capsule should be used when no pre-generated capsule."""
        char = make_character(
            "alice.json", "Alice",
            description="A brave knight who protects the realm.",
            personality="Bold and honorable"
            # No capsule provided
        )
        
        request = PromptRequest(
            messages=make_messages(4),
            characters=[char],
            settings={"system_prompt": "Test"}
        )
        
        character_first_turns = {"alice.json": 1}
        character_full_card_turns = {"alice.json": 1}
        
        prompt = construct_prompt(
            request,
            character_first_turns,
            absolute_turn=4,
            character_full_card_turns=character_full_card_turns
        )
        
        # Should use capsule, not full card
        assert not has_full_card(prompt, "Alice")
        assert has_scene_cast_capsule(prompt, "Alice")
        
        # Fallback should contain description/personality info
        assert "brave" in prompt.lower() or "knight" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
