#!/usr/bin/env python3
"""
Test script to verify that Canon Law entries are inserted on the first turn.
"""

import sys
import os
import json
from main import construct_prompt, PromptRequest, ChatMessage

def test_canon_law_first_turn():
    """Test that Canon Law entries are inserted on the first turn."""

    # Create a minimal World Info with a Canon Law entry
    world_info = {
        "entries": {
            "0": {
                "uid": 0,
                "key": ["test canon law", "canon", "law"],
                "keysecondary": [],
                "comment": "",
                "content": "[Test Canon Law: This should always be included on first turn]",
                "constant": False,
                "selective": True,
                "selectiveLogic": 0,
                "addMemo": True,
                "order": 100,
                "position": 4,
                "disable": False,
                "excludeRecursion": False,
                "probability": 100,
                "useProbability": True,
                "displayIndex": 0,
                "depth": 5,
                "is_canon_law": True
            },
            "1": {
                "uid": 1,
                "key": ["regular lore", "lore"],
                "keysecondary": [],
                "comment": "",
                "content": "[Regular Lore: This is regular lore that shouldn't affect canon law insertion]",
                "constant": False,
                "selective": True,
                "selectiveLogic": 0,
                "addMemo": True,
                "order": 100,
                "position": 4,
                "disable": False,
                "excludeRecursion": False,
                "probability": 100,
                "useProbability": True,
                "displayIndex": 1,
                "depth": 5
            }
        }
    }

    # Test Case 1: First turn with no previous messages
    print("=== Test Case 1: First Turn (No Previous Messages) ===")
    messages = [
        ChatMessage(role="user", content="Hello, let's start the story!", id=1)
    ]

    request = PromptRequest(
        messages=messages,
        characters=[],
        world_info=world_info,
        settings={"world_info_reinforce_freq": 5},  # Default frequency
        summary="",
        mode="narrator"
    )

    prompt = construct_prompt(request)
    print(f"Number of messages: {len(request.messages)}")
    print(f"Prompt contains 'Canon Law': {'### Canon Law (World Rules):' in prompt}")
    print(f"Prompt contains canon law content: {'[Test Canon Law: This should always be included on first turn]' in prompt}")

    if "### Canon Law (World Rules):" in prompt and "[Test Canon Law: This should always be included on first turn]" in prompt:
        print("✅ PASS: Canon Law inserted on first turn")
    else:
        print("❌ FAIL: Canon Law NOT inserted on first turn")
        print("Generated prompt:")
        print(prompt)

    print("\n" + "="*60 + "\n")

    # Test Case 2: Second turn (should still work with frequency=1)
    print("=== Test Case 2: Second Turn (with frequency=1) ===")
    messages = [
        ChatMessage(role="user", content="Hello, let's start the story!", id=1),
        ChatMessage(role="assistant", content="The story begins...", id=2, speaker="Narrator")
    ]

    request = PromptRequest(
        messages=messages,
        characters=[],
        world_info=world_info,
        settings={"world_info_reinforce_freq": 1},  # Every turn
        summary="",
        mode="narrator"
    )

    prompt = construct_prompt(request)
    print(f"Number of messages: {len(request.messages)}")
    print(f"Prompt contains 'Canon Law': {'### Canon Law (World Rules):' in prompt}")

    if "### Canon Law (World Rules):" in prompt:
        print("✅ PASS: Canon Law inserted on second turn with frequency=1")
    else:
        print("❌ FAIL: Canon Law NOT inserted on second turn with frequency=1")

    print("\n" + "="*60 + "\n")

    # Test Case 3: Turn 4 (should NOT be inserted if reinforced on turn 3 with freq=5)
    print("=== Test Case 3: Turn 4 (Should NOT be inserted if reinforced on turn 3) ===")
    messages = [
        ChatMessage(role="user", content="Hello, let's start the story!", id=1),
        ChatMessage(role="assistant", content="The story begins...", id=2, speaker="Narrator"),
        ChatMessage(role="user", content="What happens next?", id=3),
        ChatMessage(role="assistant", content="The adventure continues...", id=4, speaker="Narrator")
    ]

    request = PromptRequest(
        messages=messages,
        characters=[],
        world_info=world_info,
        settings={"world_info_reinforce_freq": 5},  # Default frequency
        summary="",
        mode="narrator"
    )

    prompt = construct_prompt(request)
    print(f"Number of messages: {len(request.messages)}")
    print(f"Prompt contains 'Canon Law': {'### Canon Law (World Rules):' in prompt}")

    if "### Canon Law (World Rules):" not in prompt:
        print("✅ PASS: Canon Law correctly NOT inserted (reinforced recently)")
    else:
        print("❌ FAIL: Canon Law incorrectly inserted when it should have been skipped")

if __name__ == "__main__":
    test_canon_law_first_turn()