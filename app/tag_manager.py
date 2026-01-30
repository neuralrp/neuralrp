"""
Tag Manager for character/world organization

Provides tag normalization utilities. Database operations are handled in database.py.
"""

from typing import List


def normalize_tag(tag: str) -> str:
    """
    Normalize a single tag string.
    
    Normalization rules:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    
    Args:
        tag: Raw tag string
        
    Returns:
        Normalized tag string, or empty string if input is None/empty after stripping
    """
    if not tag:
        return ""
    
    # Lowercase and strip
    normalized = tag.lower().strip()
    
    # Collapse multiple spaces to single space
    normalized = " ".join(normalized.split())
    
    return normalized


def parse_tag_string(tag_string: str) -> List[str]:
    """
    Parse comma-separated tag string into normalized list.
    
    Args:
        tag_string: Comma-separated tags (e.g., "campaign, NSFW, fandom")
        
    Returns:
        List of normalized tags (lowercase, trimmed, no duplicates)
        
    Example:
        >>> parse_tag_string("campaign, NSFW,  WIP  ,fandom")
        ['campaign', 'nsfw', 'wip', 'fandom']
    """
    if not tag_string:
        return []
    
    # Split by comma
    raw_tags = tag_string.split(",")
    
    # Normalize and filter empty tags
    normalized_tags = []
    seen_tags = set()
    
    for tag in raw_tags:
        normalized = normalize_tag(tag)
        if normalized and normalized not in seen_tags:
            normalized_tags.append(normalized)
            seen_tags.add(normalized)
    
    return normalized_tags
