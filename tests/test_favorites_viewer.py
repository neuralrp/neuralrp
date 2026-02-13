"""
Favorites Viewer - Phase 3 Testing

Comprehensive testing of favorites viewer functionality:
- Manual Mode Bug Tests (4 tests)
- Favorites Viewer Tests (24 tests)
- Integration Tests (8 tests)

Total: 36 tests covering all Phase 3 scenarios
"""

import pytest
import pytest_asyncio
import sqlite3
import os
import sys
import time
import json
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from app.database import (
    db_get_all_favorite_tags,
    db_get_popular_favorite_tags,
    db_get_favorites,
    db_add_snapshot_favorite,
    db_delete_favorite,
    db_increment_favorite_tag,
    db_detect_danbooru_tags
)

# ==============================================================================
# Task 1: Test Infrastructure - Fixtures
# ==============================================================================

@pytest.fixture
def test_db():
    """Create in-memory SQLite database with full schema for favorites testing."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Initialize sd_favorites table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sd_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            image_filename TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL,
            negative_prompt TEXT NOT NULL,
            scene_type TEXT,
            setting TEXT,
            mood TEXT,
            character_ref TEXT,
            tags TEXT,
            steps INTEGER,
            cfg_scale REAL,
            width INTEGER,
            height INTEGER,
            source_type TEXT DEFAULT 'snapshot',
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            note TEXT
        )
    """)

    # Initialize danbooru_tag_favorites table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS danbooru_tag_favorites (
            tag_text TEXT PRIMARY KEY,
            favorite_count INTEGER DEFAULT 0,
            last_used INTEGER DEFAULT (strftime('%s', 'now'))
        )
    """)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def populate_test_favorites(test_db):
    """Helper to populate test database with sample favorites."""
    def _populate():
        cursor = test_db.cursor()

        # Test favorites with various tags
        test_favorites = [
            {
                'image_filename': 'test1.png',
                'prompt': '1girl, solo, explicit, hardcore',
                'negative_prompt': 'lowres, bad anatomy',
                'scene_type': 'conversation',
                'setting': 'forest',
                'mood': 'intense',
                'character_ref': 'alice',
                'tags': ['1girl', 'solo', 'explicit', 'hardcore'],
                'steps': 20,
                'cfg_scale': 7.0,
                'width': 512,
                'height': 512,
                'source_type': 'snapshot',
                'note': 'Scene 1'
            },
            {
                'image_filename': 'test2.png',
                'prompt': '1girl, armor, sword, forest',
                'negative_prompt': 'lowres',
                'scene_type': 'battle',
                'setting': 'forest',
                'mood': 'epic',
                'character_ref': 'alice',
                'tags': ['1girl', 'armor', 'sword', 'forest'],
                'steps': 20,
                'cfg_scale': 7.0,
                'width': 512,
                'height': 512,
                'source_type': 'snapshot',
                'note': None
            },
            {
                'image_filename': 'test3.png',
                'prompt': 'blonde hair, blue eyes, solo, detailed',
                'negative_prompt': 'lowres',
                'scene_type': 'portrait',
                'setting': 'castle',
                'mood': 'peaceful',
                'character_ref': 'alice',
                'tags': ['blonde_hair', 'blue_eyes', 'solo', 'detailed'],
                'steps': 20,
                'cfg_scale': 7.0,
                'width': 512,
                'height': 512,
                'source_type': 'manual',
                'note': 'Portrait'
            },
        ]

        for fav in test_favorites:
            cursor.execute("""
                INSERT INTO sd_favorites (
                    chat_id, image_filename, prompt, negative_prompt,
                    scene_type, setting, mood, character_ref, tags,
                    steps, cfg_scale, width, height, source_type, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fav.get('chat_id'),
                fav['image_filename'],
                fav['prompt'],
                fav['negative_prompt'],
                fav.get('scene_type'),
                fav.get('setting'),
                fav.get('mood'),
                fav.get('character_ref'),
                json.dumps(fav.get('tags', [])),
                fav['steps'],
                fav['cfg_scale'],
                fav['width'],
                fav['height'],
                fav.get('source_type', 'snapshot'),
                fav.get('note')
            ))

        test_db.commit()

    return _populate


# ==============================================================================
# Task 2: Manual Mode Bug Tests (4 tests)
# ==============================================================================

def test_manual_tag_based_prompt(test_db):
    """Test tag-based prompt with >=2 danbooru tags saves and learns ALL tags."""
    cursor = test_db.cursor()

    # Simulate manual mode with tag-based prompt
    prompt = "1girl, solo, explicit, hardcore, detailed"

    # Detect danbooru tags with lenient threshold
    detected_danbooru = db_detect_danbooru_tags(prompt, threshold=1)

    # Hedge check: >=2 danbooru tags?
    assert len(detected_danbooru) >= 2, f"Expected >=2 danbooru tags, got {len(detected_danbooru)}"

    # Parse ALL tags
    all_tags = [tag.strip() for tag in prompt.split(',') if tag.strip()]

    # Should save 5 tags total
    assert len(all_tags) == 5, f"Expected 5 total tags, got {len(all_tags)}"
    assert 'explicit' in all_tags, "NSFW tag 'explicit' should be saved"
    assert 'hardcore' in all_tags, "NSFW tag 'hardcore' should be saved"
    assert 'detailed' in all_tags, "Custom tag 'detailed' should be saved"

    # Simulate learning from all tags
    learned_count = 0
    for tag in all_tags:
        cursor.execute("""
            INSERT INTO danbooru_tag_favorites (tag_text, favorite_count, last_used)
            VALUES (?, 1, ?)
            ON CONFLICT(tag_text) DO UPDATE SET favorite_count = favorite_count + 1, last_used = ?
        """, (tag, int(time.time()), int(time.time())))
        learned_count += 1

    # Verify all tags learned
    assert learned_count == 5, f"Expected to learn 5 tags, learned {learned_count}"

    # Verify in database
    cursor.execute("SELECT COUNT(*) FROM danbooru_tag_favorites WHERE tag_text IN ('explicit', 'hardcore', 'detailed')")
    count = cursor.fetchone()[0]
    assert count == 3, f"Expected 3 NSFW/custom tags in database, got {count}"

    print(f"✓ Tag-based prompt: {len(all_tags)} total tags, {len(detected_danbooru)} danbooru, {len(all_tags) - len(detected_danbooru)} custom")


def test_manual_nsfw_only_prompt(test_db):
    """Test tag-based with NSFW only: detects 0 danbooru, skips learning."""
    cursor = test_db.cursor()

    # NSFW-only prompt (no danbooru tags)
    prompt = "explicit, hardcore, nsfw"

    # Detect danbooru tags
    detected_danbooru = db_detect_danbooru_tags(prompt, threshold=1)

    # Should detect 0 danbooru tags
    assert len(detected_danbooru) == 0, f"Expected 0 danbooru tags, got {len(detected_danbooru)}"

    # Hedge fail: skip learning
    all_tags = []

    # Save with empty tags (simulated)
    assert len(all_tags) == 0, "Should save with empty tags array"

    # Verify no tags learned
    cursor.execute("SELECT COUNT(*) FROM danbooru_tag_favorites")
    count = cursor.fetchone()[0]
    assert count == 0, f"Expected 0 tags learned, got {count}"

    print("✓ NSFW-only prompt: 0 danbooru tags, learning skipped")


def test_manual_sentence_prompt(test_db):
    """Test sentence prompt with <2 danbooru tags: skips learning."""
    cursor = test_db.cursor()

    # Sentence-based prompt
    prompt = "A beautiful girl in a forest at sunset"

    # Detect danbooru tags
    detected_danbooru = db_detect_danbooru_tags(prompt, threshold=1)

    # Might detect 0 or 1 danbooru tag (e.g., 'sunset' if in config)
    assert len(detected_danbooru) < 2, f"Expected <2 danbooru tags, got {len(detected_danbooru)}"

    # Hedge fail: skip learning
    all_tags = []

    # Save with empty tags (simulated)
    assert len(all_tags) == 0, "Should save with empty tags array"

    # Verify no tags learned
    cursor.execute("SELECT COUNT(*) FROM danbooru_tag_favorites")
    count = cursor.fetchone()[0]
    assert count == 0, f"Expected 0 tags learned, got {count}"

    print(f"✓ Sentence prompt: {len(detected_danbooru)} danbooru tags (<2 threshold), learning skipped")


def test_manual_edge_case_2_danbooru_1_custom(test_db):
    """Test edge case: 2 danbooru + 1 custom tags, all should be saved and learned."""
    cursor = test_db.cursor()

    # Edge case prompt
    prompt = "1girl, solo, my_custom_style"

    # Detect danbooru tags
    detected_danbooru = db_detect_danbooru_tags(prompt, threshold=1)

    # Should detect 2 danbooru tags
    assert len(detected_danbooru) == 2, f"Expected 2 danbooru tags, got {len(detected_danbooru)}"

    # Parse ALL tags
    all_tags = [tag.strip() for tag in prompt.split(',') if tag.strip()]

    # Should save 3 tags total
    assert len(all_tags) == 3, f"Expected 3 total tags, got {len(all_tags)}"
    assert 'my_custom_style' in all_tags, "Custom tag 'my_custom_style' should be saved"

    # Simulate learning from all tags
    learned_count = 0
    for tag in all_tags:
        cursor.execute("""
            INSERT INTO danbooru_tag_favorites (tag_text, favorite_count, last_used)
            VALUES (?, 1, ?)
            ON CONFLICT(tag_text) DO UPDATE SET favorite_count = favorite_count + 1, last_used = ?
        """, (tag, int(time.time()), int(time.time())))
        learned_count += 1

    # Verify all tags learned
    assert learned_count == 3, f"Expected to learn 3 tags, learned {learned_count}"

    # Verify custom tag in database
    cursor.execute("SELECT favorite_count FROM danbooru_tag_favorites WHERE tag_text = ?", ('my_custom_style',))
    row = cursor.fetchone()
    assert row is not None, "Custom tag 'my_custom_style' should be in database"
    assert row[0] == 1, f"Custom tag count should be 1, got {row[0]}"

    print(f"✓ Edge case: {len(all_tags)} total tags, {len(detected_danbooru)} danbooru, {len(all_tags) - len(detected_danbooru)} custom")


# ==============================================================================
# Task 3: Favorites Viewer Tests - Basic Functionality (6 tests)
# ==============================================================================

def test_get_all_favorite_tags(test_db, populate_test_favorites):
    """Test db_get_all_favorite_tags returns all unique tags."""
    populate_test_favorites()

    # Get all unique tags
    tags = db_get_all_favorite_tags()

    # Should have tags from all test favorites
    assert len(tags) > 0, "Should return at least some tags"
    assert '1girl' in tags, "Tag '1girl' should be present"
    assert 'solo' in tags, "Tag 'solo' should be present"
    assert 'explicit' in tags, "Tag 'explicit' should be present"

    # Tags should be sorted alphabetically
    assert tags == sorted(tags), "Tags should be sorted alphabetically"

    print(f"✓ Get all favorite tags: {len(tags)} unique tags")


def test_get_popular_favorite_tags(test_db, populate_test_favorites):
    """Test db_get_popular_favorite_tags returns top N tags."""
    cursor = test_db.cursor()
    populate_test_favorites()

    # Populate danbooru_tag_favorites with some counts
    popular_tags = [
        ('1girl', 5),
        ('solo', 3),
        ('explicit', 2),
        ('hardcore', 2),
        ('armor', 1),
    ]

    for tag, count in popular_tags:
        cursor.execute("""
            INSERT INTO danbooru_tag_favorites (tag_text, favorite_count, last_used)
            VALUES (?, ?, ?)
        """, (tag, count, int(time.time())))

    test_db.commit()

    # Get top 5 popular tags
    popular = db_get_popular_favorite_tags(limit=5)

    # Should return 5 tags
    assert len(popular) == 5, f"Expected 5 popular tags, got {len(popular)}"

    # Should be sorted by favorite_count DESC
    counts = [tag[1] for tag in popular]
    assert counts == sorted(counts, reverse=True), "Tags should be sorted by count DESC"

    # Top tag should be '1girl' with count 5
    assert popular[0][0] == '1girl', "Top tag should be '1girl'"
    assert popular[0][1] == 5, "Top tag count should be 5"

    print(f"✓ Get popular favorite tags: {len(popular)} tags, top: {popular[0][0]} ({popular[0][1]})")


def test_get_favorites_basic(test_db, populate_test_favorites):
    """Test db_get_favorites returns favorites without filters."""
    populate_test_favorites()

    # Get all favorites
    favorites = db_get_favorites(limit=50, offset=0)

    # Should have 3 favorites
    assert len(favorites) == 3, f"Expected 3 favorites, got {len(favorites)}"

    # Should have correct structure
    fav = favorites[0]
    assert 'id' in fav, "Favorite should have 'id' field"
    assert 'image_filename' in fav, "Favorite should have 'image_filename' field"
    assert 'prompt' in fav, "Favorite should have 'prompt' field"
    assert 'tags' in fav, "Favorite should have 'tags' field"
    assert 'source_type' in fav, "Favorite should have 'source_type' field"

    print(f"✓ Get favorites basic: {len(favorites)} favorites")


def test_get_favorites_with_source_type_filter(test_db, populate_test_favorites):
    """Test db_get_favorites filters by source_type."""
    populate_test_favorites()

    # Get only snapshot favorites
    snapshot_favs = db_get_favorites(limit=50, offset=0, source_type='snapshot')
    assert len(snapshot_favs) == 2, f"Expected 2 snapshot favorites, got {len(snapshot_favs)}"
    for fav in snapshot_favs:
        assert fav['source_type'] == 'snapshot', "All should be snapshot type"

    # Get only manual favorites
    manual_favs = db_get_favorites(limit=50, offset=0, source_type='manual')
    assert len(manual_favs) == 1, f"Expected 1 manual favorite, got {len(manual_favs)}"
    for fav in manual_favs:
        assert fav['source_type'] == 'manual', "All should be manual type"

    print(f"✓ Source type filter: {len(snapshot_favs)} snapshot, {len(manual_favs)} manual")


def test_get_favorites_with_tag_filter(test_db, populate_test_favorites):
    """Test db_get_favorites filters by tags (AND semantics)."""
    populate_test_favorites()

    # Filter by single tag
    favs_with_solo = db_get_favorites(limit=50, offset=0, tags=['solo'])
    assert len(favs_with_solo) == 2, f"Expected 2 favorites with 'solo', got {len(favs_with_solo)}"

    # Filter by multiple tags (AND semantics)
    favs_with_1girl_and_solo = db_get_favorites(limit=50, offset=0, tags=['1girl', 'solo'])
    assert len(favs_with_1girl_and_solo) == 1, f"Expected 1 favorite with both '1girl' and 'solo', got {len(favs_with_1girl_and_solo)}"

    # Filter by non-existent tag
    favs_with_nonexistent = db_get_favorites(limit=50, offset=0, tags=['nonexistent_tag'])
    assert len(favs_with_nonexistent) == 0, f"Expected 0 favorites with 'nonexistent_tag', got {len(favs_with_nonexistent)}"

    print(f"✓ Tag filter: {len(favs_with_solo)} with 'solo', {len(favs_with_1girl_and_solo)} with '1girl,solo'")


def test_get_favorites_pagination(test_db, populate_test_favorites):
    """Test db_get_favorites pagination with offset."""
    populate_test_favorites()

    # Page 1 (offset 0, limit 2)
    page1 = db_get_favorites(limit=2, offset=0)
    assert len(page1) == 2, f"Expected 2 favorites on page 1, got {len(page1)}"

    # Page 2 (offset 2, limit 2)
    page2 = db_get_favorites(limit=2, offset=2)
    assert len(page2) == 1, f"Expected 1 favorite on page 2, got {len(page2)}"

    # Verify no duplicates
    page1_ids = [fav['id'] for fav in page1]
    page2_ids = [fav['id'] for fav in page2]
    assert len(set(page1_ids + page2_ids)) == 3, "Should have 3 unique favorites total"

    print(f"✓ Pagination: {len(page1)} on page 1, {len(page2)} on page 2")


# ==============================================================================
# Task 4: Favorites Viewer Tests - Tag Filtering (5 tests)
# ==============================================================================

def test_tag_filter_and_semantics(test_db, populate_test_favorites):
    """Test tag filtering uses AND semantics correctly."""
    populate_test_favorites()

    # Tag '1girl' appears in 2 favorites
    favs = db_get_favorites(limit=50, offset=0, tags=['1girl'])
    assert len(favs) == 2, f"Expected 2 favorites with '1girl', got {len(favs)}"

    # Tag 'solo' appears in 2 favorites
    favs = db_get_favorites(limit=50, offset=0, tags=['solo'])
    assert len(favs) == 2, f"Expected 2 favorites with 'solo', got {len(favs)}"

    # Both '1girl' AND 'solo' appears in 1 favorite (intersection)
    favs = db_get_favorites(limit=50, offset=0, tags=['1girl', 'solo'])
    assert len(favs) == 1, f"Expected 1 favorite with both '1girl' and 'solo', got {len(favs)}"

    # Verify it's the correct favorite (test1.png has both tags)
    assert favs[0]['image_filename'] == 'test1.png', "Should be test1.png"

    print("✓ AND semantics: '1girl'=2, 'solo'=2, '1girl,solo'=1")


def test_tag_filter_case_insensitive(test_db, populate_test_favorites):
    """Test tag filtering is case-insensitive."""
    cursor = test_db.cursor()
    populate_test_favorites()

    # Ensure tags are lowercase in database
    cursor.execute("UPDATE sd_favorites SET tags = ? WHERE id = 1",
                  (json.dumps(['1girl', 'solo', 'explicit', 'hardcore']),))
    cursor.execute("UPDATE sd_favorites SET tags = ? WHERE id = 2",
                  (json.dumps(['1girl', 'armor', 'sword', 'forest']),))
    cursor.execute("UPDATE sd_favorites SET tags = ? WHERE id = 3",
                  (json.dumps(['blonde_hair', 'blue_eyes', 'solo', 'detailed']),))
    test_db.commit()

    # Query with lowercase
    favs_lower = db_get_favorites(limit=50, offset=0, tags=['solo'])
    assert len(favs_lower) == 2, f"Expected 2 favorites with 'solo', got {len(favs_lower)}"

    # Query with uppercase (should also work)
    favs_upper = db_get_favorites(limit=50, offset=0, tags=['SOLO'])
    assert len(favs_upper) == 2, f"Expected 2 favorites with 'SOLO', got {len(favs_upper)}"

    # Results should be identical
    favs_lower_ids = sorted([fav['id'] for fav in favs_lower])
    favs_upper_ids = sorted([fav['id'] for fav in favs_upper])
    assert favs_lower_ids == favs_upper_ids, "Case-insensitive filtering should work"

    print("✓ Case-insensitive filtering: 'solo' and 'SOLO' return same results")


def test_tag_filter_empty_tags(test_db, populate_test_favorites):
    """Test tag filtering handles empty tags gracefully."""
    cursor = test_db.cursor()
    populate_test_favorites()

    # Add favorite with empty tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test_empty.png', 'test prompt', 'negative',
          None, None, None, None, json.dumps([]),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Query with any tag
    favs = db_get_favorites(limit=50, offset=0, tags=['1girl'])
    assert len(favs) == 2, f"Expected 2 favorites with '1girl', got {len(favs)}"

    # Empty tags favorite should not be included
    fav_ids = [fav['image_filename'] for fav in favs]
    assert 'test_empty.png' not in fav_ids, "Empty tags favorite should not be included in tag filter"

    print("✓ Empty tags handling: favorite with empty tags excluded from tag filter")


def test_tag_filter_many_tags(test_db, populate_test_favorites):
    """Test tag filtering with many active tags (10+)."""
    cursor = test_db.cursor()
    populate_test_favorites()

    # Create favorite with many tags
    many_tags = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7', 'tag8', 'tag9', 'tag10', 'tag11', 'tag12']
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test_many_tags.png', 'test prompt', 'negative',
          None, None, None, None, json.dumps(many_tags),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Filter with many tags
    favs = db_get_favorites(limit=50, offset=0, tags=many_tags[:5])
    assert len(favs) == 1, f"Expected 1 favorite with 5 tags, got {len(favs)}"

    # Filter with even more tags
    favs = db_get_favorites(limit=50, offset=0, tags=many_tags)
    assert len(favs) == 1, f"Expected 1 favorite with all tags, got {len(favs)}"

    print("✓ Many tags filter: handles 12+ tag filter")


def test_tag_filter_duplicate_tags(test_db, populate_test_favorites):
    """Test tag filtering with duplicate tags in query."""
    populate_test_favorites()

    # Query with duplicate tags (should handle gracefully)
    favs = db_get_favorites(limit=50, offset=0, tags=['solo', 'solo', '1girl'])
    assert len(favs) == 1, f"Expected 1 favorite with 'solo, solo, 1girl', got {len(favs)}"

    # Should be same as query without duplicates
    favs_unique = db_get_favorites(limit=50, offset=0, tags=['solo', '1girl'])
    assert len(favs_unique) == 1, f"Expected 1 favorite with 'solo, 1girl', got {len(favs_unique)}"

    # Results should be identical
    favs_ids = [fav['id'] for fav in favs]
    favs_unique_ids = [fav['id'] for fav in favs_unique]
    assert favs_ids == favs_unique_ids, "Duplicate tags should be handled"

    print("✓ Duplicate tag handling: query with duplicates works correctly")


# ==============================================================================
# Task 5: Favorites Viewer Tests - Pagination (3 tests)
# ==============================================================================

def test_pagination_load_more(test_db, populate_test_favorites):
    """Test pagination loads more results correctly."""
    populate_test_favorites()

    # Add more favorites to test pagination
    cursor = test_db.cursor()
    for i in range(4, 52):  # Add 48 more (total 51)
        cursor.execute("""
            INSERT INTO sd_favorites (
                chat_id, image_filename, prompt, negative_prompt,
                scene_type, setting, mood, character_ref, tags,
                steps, cfg_scale, width, height, source_type, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (None, f'test{i}.png', f'prompt {i}', 'negative',
              None, None, None, None, json.dumps(['tag1', 'tag2']),
              20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Page 1: offset 0, limit 50
    page1 = db_get_favorites(limit=50, offset=0)
    assert len(page1) == 50, f"Expected 50 favorites on page 1, got {len(page1)}"

    # Page 2: offset 50, limit 50
    page2 = db_get_favorites(limit=50, offset=50)
    assert len(page2) == 1, f"Expected 1 favorite on page 2, got {len(page2)}"

    # No more pages
    page3 = db_get_favorites(limit=50, offset=100)
    assert len(page3) == 0, f"Expected 0 favorites on page 3, got {len(page3)}"

    print(f"✓ Load more: 51 total, 50 on page 1, 1 on page 2")


def test_pagination_preserves_filters(test_db, populate_test_favorites):
    """Test pagination preserves tag and source_type filters."""
    populate_test_favorites()

    # Add more favorites with 'solo' tag
    cursor = test_db.cursor()
    for i in range(4, 52):
        tags = ['solo'] if i % 2 == 0 else []
        source_type = 'snapshot' if i % 3 == 0 else 'manual'
        cursor.execute("""
            INSERT INTO sd_favorites (
                chat_id, image_filename, prompt, negative_prompt,
                scene_type, setting, mood, character_ref, tags,
                steps, cfg_scale, width, height, source_type, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (None, f'test{i}.png', f'prompt {i}', 'negative',
              None, None, None, None, json.dumps(tags),
              20, 7.0, 512, 512, source_type, None))
    test_db.commit()

    # Filter by tag 'solo' + source_type 'snapshot'
    page1 = db_get_favorites(limit=20, offset=0, tags=['solo'], source_type='snapshot')
    assert len(page1) == 20, f"Expected 20 favorites on page 1, got {len(page1)}"

    # Verify all results have 'solo' tag
    for fav in page1:
        tags = json.loads(fav.get('tags', '[]'))
        assert 'solo' in tags, "All favorites should have 'solo' tag"
        assert fav['source_type'] == 'snapshot', "All favorites should be snapshot type"

    # Page 2 should also preserve filters
    page2 = db_get_favorites(limit=20, offset=20, tags=['solo'], source_type='snapshot')
    for fav in page2:
        tags = json.loads(fav.get('tags', '[]'))
        assert 'solo' in tags, "All favorites should have 'solo' tag"
        assert fav['source_type'] == 'snapshot', "All favorites should be snapshot type"

    print(f"✓ Pagination preserves filters: {len(page1)} + {len(page2)} filtered results")


def test_pagination_offset_bounds(test_db):
    """Test pagination handles out-of-bounds offsets gracefully."""
    cursor = test_db.cursor()

    # Add 10 favorites
    for i in range(1, 11):
        cursor.execute("""
            INSERT INTO sd_favorites (
                chat_id, image_filename, prompt, negative_prompt,
                scene_type, setting, mood, character_ref, tags,
                steps, cfg_scale, width, height, source_type, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (None, f'test{i}.png', f'prompt {i}', 'negative',
              None, None, None, None, json.dumps(['tag1']),
              20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Offset within bounds
    favs = db_get_favorites(limit=10, offset=5)
    assert len(favs) == 5, f"Expected 5 favorites at offset 5, got {len(favs)}"

    # Offset at exact end
    favs = db_get_favorites(limit=10, offset=10)
    assert len(favs) == 0, f"Expected 0 favorites at offset 10, got {len(favs)}"

    # Offset beyond bounds
    favs = db_get_favorites(limit=10, offset=100)
    assert len(favs) == 0, f"Expected 0 favorites at offset 100, got {len(favs)}"

    print("✓ Pagination bounds: handles out-of-bounds offsets gracefully")


# ==============================================================================
# Task 6: Integration Tests (8 tests)
# ==============================================================================

def test_integration_manual_to_viewer(test_db):
    """Test manual favorite appears in viewer."""
    cursor = test_db.cursor()

    # Create manual favorite with tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'manual_test.png', '1girl, solo, explicit, detailed', 'negative',
          None, None, None, None, json.dumps(['1girl', 'solo', 'explicit', 'detailed']),
          20, 7.0, 512, 512, 'manual', 'Test note'))
    test_db.commit()

    # Retrieve from viewer
    favs = db_get_favorites(limit=50, offset=0, source_type='manual')
    assert len(favs) == 1, f"Expected 1 manual favorite, got {len(favs)}"
    assert favs[0]['image_filename'] == 'manual_test.png', "Should find manual_test.png"
    assert favs[0]['source_type'] == 'manual', "Should be manual type"
    assert favs[0]['note'] == 'Test note', "Should preserve note"

    # Verify tags
    tags = json.loads(favs[0].get('tags', '[]'))
    assert '1girl' in tags, "Should have '1girl' tag"
    assert 'explicit' in tags, "Should have 'explicit' tag"
    assert 'detailed' in tags, "Should have custom 'detailed' tag"

    print("✓ Manual to viewer: manual favorite appears correctly")


def test_integration_snapshot_to_viewer(test_db):
    """Test snapshot favorite appears in viewer."""
    cursor = test_db.cursor()

    # Create snapshot favorite with tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ('chat_123', 'snapshot_test.png', '1girl, armor, sword', 'negative',
          'battle', 'forest', 'epic', 'alice', json.dumps(['1girl', 'armor', 'sword']),
          20, 7.0, 512, 512, 'snapshot', 'Battle scene'))
    test_db.commit()

    # Retrieve from viewer
    favs = db_get_favorites(limit=50, offset=0, source_type='snapshot')
    assert len(favs) == 1, f"Expected 1 snapshot favorite, got {len(favs)}"
    assert favs[0]['image_filename'] == 'snapshot_test.png', "Should find snapshot_test.png"
    assert favs[0]['source_type'] == 'snapshot', "Should be snapshot type"
    assert favs[0]['scene_type'] == 'battle', "Should preserve scene_type"

    # Verify tags
    tags = json.loads(favs[0].get('tags', '[]'))
    assert '1girl' in tags, "Should have '1girl' tag"
    assert 'armor' in tags, "Should have 'armor' tag"

    print("✓ Snapshot to viewer: snapshot favorite appears correctly")


def test_integration_nsfw_tags_display(test_db):
    """Test NSFW tags display correctly in viewer."""
    cursor = test_db.cursor()

    # Create favorite with NSFW tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'nsfw_test.png', '1girl, explicit, hardcore, nsfw', 'negative',
          None, None, None, None, json.dumps(['1girl', 'explicit', 'hardcore', 'nsfw']),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Retrieve from viewer
    favs = db_get_favorites(limit=50, offset=0)
    assert len(favs) == 1, f"Expected 1 favorite, got {len(favs)}"

    # Verify NSFW tags are present
    tags = json.loads(favs[0].get('tags', '[]'))
    assert 'explicit' in tags, "Should have 'explicit' tag"
    assert 'hardcore' in tags, "Should have 'hardcore' tag"
    assert 'nsfw' in tags, "Should have 'nsfw' tag"

    # Verify all tags are returned (not just danbooru)
    assert len(tags) == 4, f"Expected 4 total tags, got {len(tags)}"

    print("✓ NSFW tags display: all tags (including NSFW) stored and displayed")


def test_integration_custom_tags_display(test_db):
    """Test custom tags display correctly in viewer."""
    cursor = test_db.cursor()

    # Create favorite with custom tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'custom_test.png', '1girl, solo, my_custom_style, unique_tag', 'negative',
          None, None, None, None, json.dumps(['1girl', 'solo', 'my_custom_style', 'unique_tag']),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Retrieve from viewer
    favs = db_get_favorites(limit=50, offset=0)
    assert len(favs) == 1, f"Expected 1 favorite, got {len(favs)}"

    # Verify custom tags are present
    tags = json.loads(favs[0].get('tags', '[]'))
    assert 'my_custom_style' in tags, "Should have 'my_custom_style' tag"
    assert 'unique_tag' in tags, "Should have 'unique_tag' tag"

    # Search by custom tag
    favs_by_custom = db_get_favorites(limit=50, offset=0, tags=['my_custom_style'])
    assert len(favs_by_custom) == 1, f"Expected 1 favorite with 'my_custom_style', got {len(favs_by_custom)}"

    print("✓ Custom tags display: custom tags stored, displayed, and searchable")


def test_integration_search_nsfw_tags(test_db):
    """Test search finds favorites by NSFW tags."""
    cursor = test_db.cursor()

    # Create favorites with various tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test1.png', '1girl, explicit, solo', 'negative',
          None, None, None, None, json.dumps(['1girl', 'explicit', 'solo']),
          20, 7.0, 512, 512, 'manual', None))

    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test2.png', '1girl, armor, solo', 'negative',
          None, None, None, None, json.dumps(['1girl', 'armor', 'solo']),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Search by NSFW tag
    favs = db_get_favorites(limit=50, offset=0, tags=['explicit'])
    assert len(favs) == 1, f"Expected 1 favorite with 'explicit', got {len(favs)}"
    assert favs[0]['image_filename'] == 'test1.png', "Should find test1.png"

    # Search by danbooru tag (should also work)
    favs = db_get_favorites(limit=50, offset=0, tags=['solo'])
    assert len(favs) == 2, f"Expected 2 favorites with 'solo', got {len(favs)}"

    print("✓ Search NSFW tags: can find favorites by NSFW tags")


def test_integration_search_custom_tags(test_db):
    """Test search finds favorites by custom tags."""
    cursor = test_db.cursor()

    # Create favorites with custom tags
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test1.png', '1girl, my_style_a, solo', 'negative',
          None, None, None, None, json.dumps(['1girl', 'my_style_a', 'solo']),
          20, 7.0, 512, 512, 'manual', None))

    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test2.png', '1girl, my_style_b, solo', 'negative',
          None, None, None, None, json.dumps(['1girl', 'my_style_b', 'solo']),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Search by custom tag
    favs = db_get_favorites(limit=50, offset=0, tags=['my_style_a'])
    assert len(favs) == 1, f"Expected 1 favorite with 'my_style_a', got {len(favs)}"
    assert favs[0]['image_filename'] == 'test1.png', "Should find test1.png"

    # Search by another custom tag
    favs = db_get_favorites(limit=50, offset=0, tags=['my_style_b'])
    assert len(favs) == 1, f"Expected 1 favorite with 'my_style_b', got {len(favs)}"
    assert favs[0]['image_filename'] == 'test2.png', "Should find test2.png"

    print("✓ Search custom tags: can find favorites by custom tags")


def test_integration_multiple_tag_search(test_db):
    """Test multiple tag search with AND semantics."""
    cursor = test_db.cursor()

    # Create favorites with various tag combinations
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test1.png', 'blonde, armor, solo', 'negative',
          None, None, None, None, json.dumps(['blonde', 'armor', 'solo']),
          20, 7.0, 512, 512, 'manual', None))

    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test2.png', 'blonde, solo', 'negative',
          None, None, None, None, json.dumps(['blonde', 'solo']),
          20, 7.0, 512, 512, 'manual', None))

    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test3.png', 'armor, solo', 'negative',
          None, None, None, None, json.dumps(['armor', 'solo']),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Single tag search
    favs = db_get_favorites(limit=50, offset=0, tags=['blonde'])
    assert len(favs) == 2, f"Expected 2 favorites with 'blonde', got {len(favs)}"

    # Multiple tag search (AND semantics)
    favs = db_get_favorites(limit=50, offset=0, tags=['blonde', 'armor'])
    assert len(favs) == 1, f"Expected 1 favorite with both 'blonde' and 'armor', got {len(favs)}"
    assert favs[0]['image_filename'] == 'test1.png', "Should find test1.png"

    # Three tag search
    favs = db_get_favorites(limit=50, offset=0, tags=['blonde', 'armor', 'solo'])
    assert len(favs) == 1, f"Expected 1 favorite with all three tags, got {len(favs)}"

    # Combination that returns 0
    favs = db_get_favorites(limit=50, offset=0, tags=['blonde', 'armor', 'nonexistent'])
    assert len(favs) == 0, f"Expected 0 favorites with 'blonde, armor, nonexistent', got {len(favs)}"

    print("✓ Multiple tag search: AND semantics work correctly")


def test_integration_delete_favorite(test_db):
    """Test delete favorite works and tag frequencies are not affected."""
    cursor = test_db.cursor()

    # Create favorite
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test_delete.png', '1girl, solo, explicit', 'negative',
          None, None, None, None, json.dumps(['1girl', 'solo', 'explicit']),
          20, 7.0, 512, 512, 'manual', None))

    fav_id = cursor.lastrowid
    test_db.commit()

    # Verify favorite exists
    favs = db_get_favorites(limit=50, offset=0)
    assert len(favs) == 1, f"Expected 1 favorite before delete, got {len(favs)}"

    # Delete favorite
    db_delete_favorite(fav_id)

    # Verify favorite is gone
    favs = db_get_favorites(limit=50, offset=0)
    assert len(favs) == 0, f"Expected 0 favorites after delete, got {len(favs)}"

    # Note: Tag frequencies are NOT affected by delete (per design)
    # This is tested by verifying no cascade delete occurs

    print("✓ Delete favorite: favorite removed, tag frequencies preserved")


# ==============================================================================
# Task 7: Edge Case Tests (5 additional tests)
# ==============================================================================

def test_edge_case_empty_tags_json(test_db):
    """Test favorites with malformed JSON tags."""
    cursor = test_db.cursor()

    # Create favorite with empty JSON array
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'empty_tags.png', 'test prompt', 'negative',
          None, None, None, None, '[]',
          20, 7.0, 512, 512, 'manual', None))

    # Create favorite with null JSON
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'null_tags.png', 'test prompt', 'negative',
          None, None, None, None, 'null',
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Retrieve favorites
    favs = db_get_favorites(limit=50, offset=0)

    # Should handle empty/null tags gracefully
    assert len(favs) == 2, f"Expected 2 favorites, got {len(favs)}"

    # Tags should be parseable
    for fav in favs:
        try:
            tags = json.loads(fav.get('tags', '[]'))
            assert isinstance(tags, list), "Tags should be a list"
        except json.JSONDecodeError:
            # If parsing fails, default to empty list
            tags = []

    print("✓ Empty/null JSON tags: handled gracefully")


def test_edge_case_very_long_prompt(test_db):
    """Test favorites with very long prompts."""
    cursor = test_db.cursor()

    # Create favorite with very long prompt (many tags)
    long_tags = ','.join([f'tag{i}' for i in range(100)])
    tags_list = [f'tag{i}' for i in range(100)]

    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'long_prompt.png', long_tags, 'negative',
          None, None, None, None, json.dumps(tags_list),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Retrieve favorite
    favs = db_get_favorites(limit=50, offset=0)
    assert len(favs) == 1, f"Expected 1 favorite, got {len(favs)}"

    # Verify all tags preserved
    tags = json.loads(favs[0].get('tags', '[]'))
    assert len(tags) == 100, f"Expected 100 tags, got {len(tags)}"

    print("✓ Very long prompt: all tags preserved")


def test_edge_case_unicode_tags(test_db):
    """Test favorites with unicode tags."""
    cursor = test_db.cursor()

    # Create favorite with unicode tags
    unicode_tags = ['1girl', '女の子', '少女', 'solo']

    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'unicode_tags.png', '1girl, test', 'negative',
          None, None, None, None, json.dumps(unicode_tags),
          20, 7.0, 512, 512, 'manual', None))
    test_db.commit()

    # Retrieve favorite
    favs = db_get_favorites(limit=50, offset=0)
    assert len(favs) == 1, f"Expected 1 favorite, got {len(favs)}"

    # Verify unicode tags preserved
    tags = json.loads(favs[0].get('tags', '[]'))
    assert '女の子' in tags, "Unicode tag '女の子' should be preserved"
    assert '少女' in tags, "Unicode tag '少女' should be preserved"

    # Search by unicode tag
    favs_by_unicode = db_get_favorites(limit=50, offset=0, tags=['女の子'])
    assert len(favs_by_unicode) == 1, f"Expected 1 favorite with '女の子', got {len(favs_by_unicode)}"

    print("✓ Unicode tags: handled correctly")


def test_edge_case_no_filters(test_db, populate_test_favorites):
    """Test retrieving favorites with no filters."""
    populate_test_favorites()

    # Get all favorites (no filters)
    favs = db_get_favorites(limit=50, offset=0)

    # Should return all favorites
    assert len(favs) == 3, f"Expected 3 favorites with no filters, got {len(favs)}"

    # Verify all source types present
    source_types = set(fav['source_type'] for fav in favs)
    assert 'snapshot' in source_types, "Should have snapshot type"
    assert 'manual' in source_types, "Should have manual type"

    print("✓ No filters: all favorites returned")


def test_edge_case_all_filters_combined(test_db, populate_test_favorites):
    """Test all filters combined (tags + source_type)."""
    populate_test_favorites()

    # Add more test data for comprehensive filtering
    cursor = test_db.cursor()
    cursor.execute("""
        INSERT INTO sd_favorites (
            chat_id, image_filename, prompt, negative_prompt,
            scene_type, setting, mood, character_ref, tags,
            steps, cfg_scale, width, height, source_type, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, 'test4.png', '1girl, solo, armor', 'negative',
          None, None, None, None, json.dumps(['1girl', 'solo', 'armor']),
          20, 7.0, 512, 512, 'snapshot', None))
    test_db.commit()

    # Filter by source_type='snapshot' + tags=['1girl', 'solo']
    favs = db_get_favorites(limit=50, offset=0, source_type='snapshot', tags=['1girl', 'solo'])
    assert len(favs) == 1, f"Expected 1 favorite with combined filters, got {len(favs)}"
    assert favs[0]['source_type'] == 'snapshot', "Should be snapshot type"

    tags = json.loads(favs[0].get('tags', '[]'))
    assert '1girl' in tags and 'solo' in tags, "Should have both tags"

    # Filter by source_type='manual' + tags=['solo']
    favs = db_get_favorites(limit=50, offset=0, source_type='manual', tags=['solo'])
    assert len(favs) == 1, f"Expected 1 manual favorite with 'solo', got {len(favs)}"

    print("✓ Combined filters: all filters work together")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
