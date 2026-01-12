#!/usr/bin/env python3
"""
Test keyword matching directly (bypassing semantic search).
"""

import re

def normalize_for_matching(s: str) -> str:
    """Normalize text for matching: remove apostrophes, quotes, hyphens, and other punctuation"""
    return re.sub(r"['\"\-\.,:;!?()]+", "", s)

def key_matches_text_OLD(key: str, text: str) -> bool:
    """OLD VERSION - has the bug"""
    key_normalized = normalize_for_matching(key.strip().lower())
    key_pattern = re.escape(key_normalized).replace(r"\ ", r"\s+")
    pattern = r"\b" + key_pattern + r"s?\b"
    text_normalized = normalize_for_matching(text)
    return re.search(pattern, text_normalized) is not None

def key_matches_text_NEW(key: str, text: str) -> bool:
    """NEW VERSION - should fix the bug"""
    key_normalized = normalize_for_matching(key.strip().lower())
    
    # If key ends in 's', remove it and make it optional
    if key_normalized.endswith('s') and len(key_normalized) > 3:
        key_base = key_normalized[:-1]
        key_pattern = re.escape(key_base).replace(r"\ ", r"\s+")
        pattern = r"\b" + key_pattern + r"s?\b"
    else:
        key_pattern = re.escape(key_normalized).replace(r"\ ", r"\s+")
        pattern = r"\b" + key_pattern + r"s?\b"
    
    text_normalized = normalize_for_matching(text)
    match = re.search(pattern, text_normalized)
    return match is not None

# Test cases
test_cases = [
    ("breeders", "breeder's program", "Should match: breeders key -> breeder's text"),
    ("breeders", "breeders program", "Should match: breeders key -> breeders text"),
    ("breeders", "breeder program", "Should match: breeders key -> breeder text"),
    ("breeder", "breeder's program", "Should match: breeder key -> breeder's text"),
    ("breeder", "breeders program", "Should match: breeder key -> breeders text"),
]

print("=== Testing OLD vs NEW Keyword Matching ===\n")

for key, text, description in test_cases:
    old_match = key_matches_text_OLD(key, text)
    new_match = key_matches_text_NEW(key, text)
    
    print(f"{description}")
    print(f"  Key: '{key}', Text: '{text}'")
    print(f"  OLD: {old_match}, NEW: {new_match}")
    if old_match != new_match:
        print(f"  ✅ FIXED!")
    else:
        print(f"  {'✅ PASS' if new_match else '⚠️  Still fails'}")
    print()
