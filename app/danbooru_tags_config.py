"""
Danbooru Tag Configuration for Simplified Snapshot Feature

This file contains minimal configuration for the simplified snapshot system.
Only quality tags and universal negatives are kept - everything else is now
extracted directly from LLM as natural language phrases.

Block Structure (SIMPLIFIED):
- Block 0: Quality/Masterpiece tags (hardwired, always used)
- All other blocks: REMOVED (replaced by LLM JSON extraction)

Example usage:
>>> from app.danbooru_tags_config import BLOCK_0, get_universal_negatives
>>> BLOCK_0[:3]
['masterpiece', 'best quality', 'high quality']
>>> get_universal_negatives()
['low quality', 'worst quality', 'bad anatomy']
"""

from typing import List

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# BLOCK 0: QUALITY/MASTERPIECE (keep minimal set)
# ============================================================================

BLOCK_0: List[str] = [
    # Core quality (always use first 3)
    "masterpiece",
    "best quality",
    "high quality"
]

# ============================================================================
# UNIVERSAL NEGATIVE TAGS
# ============================================================================

UNIVERSAL_NEGATIVES: List[str] = [
    "low quality",
    "worst quality",
    "bad anatomy"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_universal_negatives() -> List[str]:
    """Get universal negative tags."""
    return UNIVERSAL_NEGATIVES.copy()

# ============================================================================
# DEPRECATED FUNCTIONS
# These are kept for API compatibility but return minimal/empty values
# They will be removed in future versions
# ============================================================================

def get_all_tags() -> dict:
    """
    DEPRECATED: Old 4-block system removed.
    Returns only Block 0 (quality tags) for compatibility.
    """
    return {0: BLOCK_0}

def get_block_tags(block_num: int) -> List[str]:
    """DEPRECATED: Use direct LLM extraction instead."""
    if block_num == 0:
        return BLOCK_0
    return []

def get_block_target(block_num: int) -> int:
    """DEPRECATED: Targets removed with simplified system."""
    return 3

def get_all_targets() -> dict:
    """DEPRECATED: Old target system removed."""
    return {0: 3}

def get_fallback_tags(block_num: int) -> List[str]:
    """DEPRECATED: Fallback system removed."""
    return []

def get_min_matches(block_num: int) -> int:
    """DEPRECATED: Matching system removed."""
    return 0

def get_total_tag_count() -> int:
    """DEPRECATED: Only quality tags remain."""
    return len(BLOCK_0)

def get_tags_as_tuples() -> List[tuple]:
    """DEPRECATED: Only quality tags remain."""
    return [(tag, 0) for tag in BLOCK_0]

def get_interaction_tags() -> List[str]:
    """DEPRECATED: Action detection removed, use LLM extraction."""
    return []
