#!/usr/bin/env python3
"""
Test to debug why "breeder's program" doesn't trigger "breeders" key.
"""

import re
from main import get_cached_world_entries
import json

# Load the actual Palace world info
with open("app/data/worldinfo/The_Palace_plist_worldinfo.json", "r") as f:
    palace_world_info = json.load(f)

print("=== Testing Breeder Matching ===\n")

# Test text variations
test_texts = [
    "the breeder's program",
    "breeders program",
    "breeder program",
    "the breeders are working",
]

for text in test_texts:
    print(f"Testing: '{text}'")
    triggered_lore, canon_entries = get_cached_world_entries(
        palace_world_info,
        text,
        max_entries=10,
        semantic_threshold=0.25,
        is_initial_turn=False
    )
    
    # Check if "Breeders Initiative" was triggered
    breeder_triggered = any("Breeders Initiative" in lore for lore in triggered_lore)
    print(f"  Breeders Initiative triggered: {breeder_triggered}")
    if triggered_lore:
        print(f"  Triggered entries: {len(triggered_lore)}")
        for lore in triggered_lore:
            if "Breeder" in lore:
                print(f"    - {lore[:80]}...")
    print()

# Also test the normalization directly
print("\n=== Testing Normalization Logic ===\n")

def normalize_for_matching(s: str) -> str:
    """Normalize text for matching: remove apostrophes, quotes, hyphens, and other punctuation"""
    return re.sub(r"['\"\-\.,:;!?()]+", "", s)

test_key = "breeders"
test_texts_norm = ["breeder's program", "breeders program", "breeder program"]

for text in test_texts_norm:
    text_normalized = normalize_for_matching(text)
    key_normalized = normalize_for_matching(test_key)
    
    # Create pattern as in the code
    key_pattern = re.escape(key_normalized).replace(r"\ ", r"\s+")
    pattern = r"\b" + key_pattern + r"s?\b"
    
    print(f"Key: '{test_key}' -> Normalized: '{key_normalized}'")
    print(f"Text: '{text}' -> Normalized: '{text_normalized}'")
    print(f"Pattern: {pattern}")
    match = re.search(pattern, text_normalized)
    print(f"Match: {match is not None}")
    if match:
        print(f"  Matched: '{match.group()}'")
    print()
