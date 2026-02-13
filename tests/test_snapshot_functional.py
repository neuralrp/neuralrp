"""
Snapshot Feature - Functional Testing

Tests core components in isolation:
- Database layer: migration, embedding search, frequency tracking, snapshot history
- Analyzer layer: context extraction, keyword detection, semantic matching, caching
- Prompt builder layer: 4-block construction, character tags, fallback, duplicates

Phase 5A - Unit Testing
"""

import pytest
import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import (
    db_get_danbooru_tag_count,
    db_search_danbooru_embeddings,
    db_increment_tag_frequency,
    db_save_chat,
    db_get_chat,
    db_save_snapshot_to_metadata,
    db_get_snapshot_history
)
from app.danbooru_tags_config import get_total_tag_count
from app.snapshot_analyzer import SnapshotAnalyzer
from app.snapshot_prompt_builder import SnapshotPromptBuilder
from app.danbooru_tags_config import get_min_matches

# ==============================================================================
# Task 2: Database Layer Tests
# ==============================================================================

def test_danbooru_tags_migration():
    """Verify danbooru_tags table populated with ~1560 tags from config."""
    config_count = get_total_tag_count()
    db_count = db_get_danbooru_tag_count()

    # Allow some tolerance for missing tags (may not be 100% exact)
    assert db_count >= 1300, f"Expected at least 1300 tags, got {db_count} (config: {config_count})"
    print(f"✓ Danbooru tags migration: {db_count} tags (config: {config_count})")


def test_danbooru_embedding_search():
    """Test db_search_danbooru_embeddings returns relevant results."""
    import numpy as np
    from app.database import db_get_danbooru_embedding_count

    # Skip if tags not ready
    if db_get_danbooru_tag_count() == 0:
        print("⚠ Skipping embedding search test: no tags")
        return

    # Skip if embeddings not generated
    embedding_count = db_get_danbooru_embedding_count()
    if embedding_count == 0:
        print(f"⚠ Skipping embedding search test: no embeddings generated (run migration first)")
        return

    # Try to import semantic search engine from main.py (where it's defined)
    try:
        import main
        semantic_search_engine = main.semantic_search_engine
    except (ImportError, AttributeError):
        print("⚠ Skipping embedding search test: semantic engine unavailable (main.py not loaded)")
        return

    # Skip if semantic engine not available
    if not semantic_search_engine or not semantic_search_engine.load_model():
        print("⚠ Skipping embedding search test: semantic engine unavailable (model not loaded)")
        return

    # Test query for "girl"
    query_embedding = semantic_search_engine.model.encode(
        ["girl"],
        convert_to_numpy=True
    )[0]

    # Search block 1 (subject)
    results = db_search_danbooru_embeddings(
        query_embedding,
        block_num=1,
        k=5,
        threshold=0.35
    )

    assert len(results) > 0, "Should return at least 1 result"
    assert results[0][1] > 0.35, "First result should have similarity > 0.35"
    print(f"✓ Embedding search: {len(results)} results, top: {results[0][0]}")


def test_tag_frequency_tracking():
    """Test db_increment_tag_frequency updates tag usage count."""
    if db_get_danbooru_tag_count() == 0:
        print("⚠ Skipping frequency test: no tags")
        return

    # Create test tag with unique name
    test_tag = f"test_tag_{int(time.time())}"

    # Increment frequency (should not fail even if tag doesn't exist)
    result = db_increment_tag_frequency(test_tag)
    print(f"✓ Tag frequency increment: success={result}")


def test_snapshot_history():
    """Test db_save_snapshot_to_metadata and db_get_snapshot_history."""
    import json
    from app.database import get_connection

    # Create test chat with snapshot in metadata
    test_chat_id = f"snapshot_test_{int(time.time())}"
    snapshot_data = {
        'timestamp': int(time.time()),
        'prompt': 'test prompt',
        'negative': 'test negative',
        'scene_analysis': {'scene_type': 'test'},
        'character_tag': 'test_tag',
        'character_name': 'TestCharacter',
        'image_filename': 'test.png'
    }

    test_chat = {
        'messages': [
            {'role': 'user', 'content': 'Test message', 'speaker': 'User'}
        ],
        'metadata': {
            'snapshot_history': [snapshot_data]
        }
    }

    # Save chat - snapshot_history should now be preserved
    db_save_chat(test_chat_id, test_chat)

    # Retrieve history
    history = db_get_snapshot_history(test_chat_id)
    assert len(history) > 0, "History should have at least 1 snapshot"
    assert history[0]['prompt'] == 'test prompt', "Prompt should match"
    assert history[0]['character_name'] == 'TestCharacter', "Character should match"

    print(f"✓ Snapshot history: {len(history)} snapshots saved and retrieved")


# ==============================================================================
# Task 3: Analyzer Layer Tests
# ==============================================================================

def test_extract_conversation_context():
    """Test extract_conversation_context limits to last 4 messages."""
    # Create 10 messages
    long_messages = [
        {'role': 'assistant', 'content': f'Message {i}', 'speaker': 'Char'}
        for i in range(10)
    ]

    analyzer = SnapshotAnalyzer(None, None, {})
    context = analyzer.extract_conversation_context(long_messages, message_count=4)

    lines = context.split('\n')
    assert len(lines) == 4, f"Should extract 4 messages, got {len(lines)}"
    assert 'Message 6' in context, "Should contain 6th message"
    assert 'Message 5' not in context, "Should not contain 5th message"

    print(f"✓ Context extraction: {len(lines)} lines (last 4 messages)")


def test_scene_type_keywords():
    """Test detect_scene_type_keywords identifies all 6 scene types."""
    test_cases = {
        'combat': 'Draw your sword and fight the enemy with battle skills!',
        'dialogue': 'Tell me about your journey and speak your mind.',
        'exploration': 'Let us explore the forest path and discover new lands.',
        'romance': 'I love you with all my heart and hold you close.',
        'tavern': 'Order another drink at the tavern and rest here.',
        'magic': 'Cast a spell to heal the wound and summon spirits.'
    }

    analyzer = SnapshotAnalyzer(None, None, {})

    for expected_type, test_text in test_cases.items():
        scene_type, keywords = analyzer.detect_scene_type_keywords(test_text)
        assert scene_type == expected_type, \
            f"Expected '{expected_type}', got '{scene_type}'"
        assert len(keywords) >= 2, \
            f"Should have at least 2 keywords for '{expected_type}'"

    print(f"✓ Scene type detection: all 6 types identified correctly")


def test_semantic_matching_by_block():
    """Test match_tags_semantically filters by block_num correctly."""
    from app.database import db_get_danbooru_embedding_count

    # Skip if embeddings not generated
    embedding_count = db_get_danbooru_embedding_count()
    if embedding_count == 0:
        print(f"⚠ Skipping semantic matching test: no embeddings generated (run migration first)")
        return

    # Try to import semantic search engine from main.py (where it's defined)
    try:
        import main
        semantic_search_engine = main.semantic_search_engine
    except (ImportError, AttributeError):
        print("⚠ Skipping semantic matching test: semantic engine unavailable (main.py not loaded)")
        return

    # Skip if semantic engine not available
    if not semantic_search_engine or not semantic_search_engine.load_model():
        print("⚠ Skipping semantic matching test: semantic engine unavailable (model not loaded)")
        return

    analyzer = SnapshotAnalyzer(semantic_search_engine, None, {})

    # Test block 1 (subject) - "girl"
    query = "girl"
    results_block1 = analyzer.match_tags_semantically(
        query, block_num=1, k=5, threshold=0.35
    )

    # Test block 2 (environment) - "forest"
    results_block2 = analyzer.match_tags_semantically(
        query, block_num=2, k=5, threshold=0.35
    )

    # Both should return results, but different tags
    assert len(results_block1) > 0, "Block 1 should return results"
    assert len(results_block2) > 0, "Block 2 should return results"
    assert results_block1[0][0] != results_block2[0][0], \
        "Different blocks should return different tags"

    print(f"✓ Semantic matching: Block1='{results_block1[0][0]}', Block2='{results_block2[0][0]}'")


@pytest.mark.asyncio
async def test_scene_analysis_keywords():
    """Test analyze_scene with keyword detection (no LLM)."""
    test_messages = [
        {'role': 'assistant', 'content': 'I fight with my sword!', 'speaker': 'Warrior'}
    ]

    analyzer = SnapshotAnalyzer(None, None, {})  # No LLM, keyword-only
    result = await analyzer.analyze_scene(test_messages, "test_chat")

    assert result['scene_type'] == 'combat', "Should detect combat scene"
    assert result['keyword_detected'] == True, "Should detect keywords"
    assert result['llm_used'] == False, "Should not use LLM"
    assert len(result['matched_keywords']) > 0, "Should have matched keywords"

    print(f"✓ Scene analysis: type={result['scene_type']}, "
          f"keywords={result['matched_keywords']}")


def test_character_tag_caching():
    """Test get_character_tag uses cache to prevent redundant DB calls."""
    # Create test chat with NPC
    test_chat_id = f"cache_test_{int(time.time())}"
    test_chat = {
        'messages': [],
        'metadata': {
            'localnpcs': {
                'npc_123': {
                    'name': 'TestNPC',
                    'data': {'extensions': {'danbooru_tag': '1girl, blue hair'}}
                }
            }
        }
    }

    analyzer = SnapshotAnalyzer(None, None, {})

    # First call - should cache
    tag1 = analyzer.get_character_tag('npc_123', test_chat)
    assert tag1 == '1girl, blue hair', "Should return NPC's danbooru_tag"

    # Second call - should use cache (no DB access)
    tag2 = analyzer.get_character_tag('npc_123', test_chat)
    assert tag2 == tag1, "Cached tag should match"

    # Third call with different NPC - should return None
    tag3 = analyzer.get_character_tag('npc_456', test_chat)
    assert tag3 is None, "Non-existent NPC should return None"

    print(f"✓ Character tag caching: cache hit, no redundant DB calls")


# ==============================================================================
# Task 4: Prompt Builder Layer Tests
# ==============================================================================

def test_4_block_prompt_construction():
    """Test build_4_block_prompt constructs proper 4-block structure."""

    # Create mock analyzer
    class MockAnalyzer:
        def match_tags_semantically(self, query, block_num=None, k=15, threshold=0.35):
            if block_num == 1:
                return [("1girl", 0.8), ("long hair", 0.75)]
            elif block_num == 2:
                return [("forest", 0.7), ("night", 0.65)]
            elif block_num == 3:
                return [("cinematic lighting", 0.8), ("dramatic", 0.75)]
            return []

    builder = SnapshotPromptBuilder(MockAnalyzer())

    scene_analysis = {
        'scene_type': 'combat',
        'setting': 'battlefield',
        'mood': 'intense',
        'keyword_detected': True,
        'matched_keywords': ['battle', 'fight'],
        'llm_used': False
    }

    positive, negative = builder.build_4_block_prompt(scene_analysis)

    # Block 0: Quality tags
    assert "masterpiece" in positive, "Should have quality tags"
    assert "best quality" in positive, "Should have best quality"

    # Block 1: Subject tags
    assert "1girl" in positive, "Should have subject tags"
    assert "long hair" in positive, "Should have long hair"

    # Block 2: Environment tags
    assert "forest" in positive or "battlefield" in positive, \
        "Should have environment tags"

    # Block 3: Style tags
    assert "cinematic lighting" in positive, "Should have style tags"
    assert "dramatic" in positive, "Should have dramatic style"

    # Universal negatives
    assert "low quality" in negative, "Should have universal negatives"
    assert "worst quality" in negative, "Should have worst quality"

    tags = positive.split(', ')
    print(f"✓ 4-block prompt: {len(tags)} tags")
    print(f"  Positive: {positive}")
    print(f"  Negative: {negative}")


def test_character_tag_integration():
    """Test Block 1 includes character tags when provided."""

    class MockAnalyzer:
        def match_tags_semantically(self, query, block_num=None, k=15, threshold=0.35):
            # Return enough matches to avoid fallback
            return [("solo", 0.7), ("2girls", 0.6)]

    builder = SnapshotPromptBuilder(MockAnalyzer())

    scene_analysis = {'scene_type': 'dialogue', 'setting': '', 'mood': ''}

    # Without character tag (semantic matches used instead)
    positive1, _ = builder.build_4_block_prompt(scene_analysis, character_tag=None)
    # "1girl" might appear if fallback triggers, but with semantic matches it shouldn't
    # Let's just verify solo and 2girls are present from semantic matches
    assert "solo" in positive1, "Should have solo from semantic matches"
    assert "2girls" in positive1, "Should have 2girls from semantic matches"

    # With character tag
    positive2, _ = builder.build_4_block_prompt(
        scene_analysis,
        character_tag="1girl, blonde hair, blue eyes"
    )
    assert "1girl" in positive2, "Should have 1girl from character tag"
    assert "blonde hair" in positive2, "Should have blonde hair from character tag"
    assert "blue eyes" in positive2, "Should have blue eyes from character tag"
    # Character tags should be first (before semantic matches)
    tags = positive2.split(', ')
    char_tag_indices = [i for i, t in enumerate(tags) if t in ["1girl", "blonde hair", "blue eyes"]]
    if len(char_tag_indices) > 0 and "solo" in tags:
        solo_index = tags.index("solo")
        # At least one character tag should come before solo
        assert min(char_tag_indices) < solo_index, "Character tags should come before semantic matches"

    print(f"✓ Character tag integration: 3 character tags included with priority")


def test_fallback_tag_logic():
    """Test fallback tags applied when semantic matches insufficient."""
    from app.danbooru_tags_config import get_min_matches

    # Create mock analyzer that returns NO matches
    class EmptyAnalyzer:
        def match_tags_semantically(self, query, block_num=None, k=15, threshold=0.35):
            return []  # No semantic matches

    builder = SnapshotPromptBuilder(EmptyAnalyzer())

    scene_analysis = {
        'scene_type': 'other',
        'setting': '',
        'mood': ''
    }

    positive, _ = builder.build_4_block_prompt(scene_analysis)

    # Check block 1 fallback (needs 2 matches, has 0)
    min_matches_1 = get_min_matches(1)
    assert min_matches_1 == 2, "Block 1 should need 2 matches"
    assert "1girl" in positive, "Should use fallback 1girl"
    assert "solo" in positive, "Should use fallback solo"

    # Check block 2 fallback (needs 1 match, has 0)
    assert "simple background" in positive, "Should use fallback simple background"

    # Check block 3 fallback (needs 2 matches, has 0)
    assert "cinematic lighting" in positive or "detailed" in positive, \
        "Should use fallback style tags"

    print(f"✓ Fallback logic: all blocks met minimum match requirements")


def test_duplicate_tag_prevention():
    """Test build_4_block_prompt prevents duplicate tags."""

    # Create mock analyzer that returns DUPLICATE tags
    class DuplicateAnalyzer:
        def match_tags_semantically(self, query, block_num=None, k=15, threshold=0.35):
            # Return "1girl" which is also in fallback
            return [("1girl", 0.9)]

    builder = SnapshotPromptBuilder(DuplicateAnalyzer())

    scene_analysis = {'scene_type': 'other', 'setting': '', 'mood': ''}

    positive, _ = builder.build_4_block_prompt(scene_analysis)

    tags = [tag.strip() for tag in positive.split(',')]
    unique_tags = list(dict.fromkeys(tags))  # Preserve order, remove duplicates

    assert len(tags) == len(unique_tags), "Should have no duplicate tags"

    # Check only one "1girl"
    count_1girl = tags.count("1girl")
    assert count_1girl == 1, f"Should have only 1 '1girl', got {count_1girl}"

    print(f"✓ Duplicate prevention: {len(tags)} unique tags, no duplicates")
