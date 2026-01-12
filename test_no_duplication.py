#!/usr/bin/env python3
"""
Test to verify that Canon Law entries are NEVER duplicated in the same prompt.
"""

import sys
import os
from main import construct_prompt, PromptRequest, ChatMessage

def test_no_canon_duplication():
    """Test that Canon Law is never duplicated in a single prompt."""

    # Create a minimal World Info with a Canon Law entry
    world_info = {
        "entries": {
            "0": {
                "uid": 0,
                "key": ["test canon law"],
                "keysecondary": [],
                "content": "[Test Canon Law: This is a unique canon law entry that should appear exactly once]",
                "is_canon_law": True,
                "constant": False,
                "selective": True,
                "displayIndex": 0,
                "depth": 5
            }
        }
    }

    print("=== Canon Law Duplication Test ===\n")

    # Test multiple scenarios where duplication could occur
    test_cases = [
        ("Turn 3 with freq=3", 3, 3),
        ("Turn 5 with freq=5", 5, 5),
        ("Turn 6 with freq=3", 6, 3),
        ("Turn 10 with freq=1", 10, 1),
    ]

    all_passed = True

    for test_name, num_messages, freq in test_cases:
        # Build message list
        messages = []
        for i in range(num_messages):
            if i % 2 == 0:
                messages.append(ChatMessage(role="user", content=f"Message {i+1}", id=i+1))
            else:
                messages.append(ChatMessage(role="assistant", content=f"Response {i+1}", id=i+1, speaker="Narrator"))

        request = PromptRequest(
            messages=messages,
            characters=[],
            world_info=world_info,
            settings={"world_info_reinforce_freq": freq},
            summary="",
            mode="narrator"
        )

        prompt = construct_prompt(request)
        
        # Count occurrences of the unique canon law content
        canon_content = "[Test Canon Law: This is a unique canon law entry that should appear exactly once]"
        count = prompt.count(canon_content)
        
        print(f"Test: {test_name}")
        print(f"  Messages: {num_messages}, Freq: {freq}")
        print(f"  Canon law occurrences: {count}")
        
        if count == 1:
            print(f"  ✅ PASS: Canon law appears exactly once (no duplication)\n")
        elif count == 0:
            print(f"  ❌ FAIL: Canon law missing from prompt!\n")
            all_passed = False
        else:
            print(f"  ❌ FAIL: Canon law appears {count} times (duplication detected!)\n")
            all_passed = False
            # Print the prompt for debugging
            print("Generated prompt:")
            print(prompt)
            print("\n" + "="*60 + "\n")

    if all_passed:
        print("="*60)
        print("✅ ALL TESTS PASSED: No canon law duplication detected!")
        print("="*60)
    else:
        print("="*60)
        print("❌ SOME TESTS FAILED: Check output above")
        print("="*60)

if __name__ == "__main__":
    test_no_canon_duplication()
