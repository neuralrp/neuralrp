"""
Snapshot Prompt Builder - Simplified Version

Builds SD prompts using JSON scene extraction instead of complex semantic matching.

New Architecture:
1. Extract scene JSON from LLM (location, action, activity, dress, expression)
2. Combine directly with character tags and quality tags
3. No semantic search, no tag tables, no scoring

Final Prompt Structure:
[Block 0: Quality] + [Character Tags (Count + Appearance)] + [Action, Activity, Expression] + [Dress] + [Location] + [User Tags]

Example:
"masterpiece, best quality, high quality, 1girl, solo, blonde hair, blue eyes, sitting, at basketball game, smiling, leather armor, tavern interior"
"""

from typing import List, Dict, Tuple, Optional, Any

from app.danbooru_tags_config import (
    get_universal_negatives,
    BLOCK_0
)


class SnapshotPromptBuilder:
    """Builds simplified SD prompts from JSON scene extraction."""
    
    def __init__(self):
        """Initialize prompt builder (no dependencies needed anymore)."""
        pass
    
    def build_simple_prompt(self,
                         scene_json: Dict[str, str],
                         character_tags: List[str],
                         user_tags: List[str],
                         character_count_tags: str) -> Tuple[str, str]:
        """
        Build simplified prompt using JSON extraction.

        Args:
            scene_json: {'location': str, 'action': str, 'activity': str, 'dress': str, 'expression': str}
            character_tags: List of danbooru tags from character visual_canon
            user_tags: List of danbooru tags from user settings
            character_count_tags: String like "1girl" or "2girls, 1boy"

        Returns:
            (positive_prompt, negative_prompt)

        Example:
        >>> scene_json = {'location': 'tavern interior', 'action': 'sitting', 'activity': 'drinking beer', 'dress': 'leather armor', 'expression': 'smiling'}
        >>> character_tags = ['blonde hair', 'blue eyes']
        >>> user_tags = []
        >>> character_count_tags = '1girl, solo'
        >>> builder.build_simple_prompt(scene_json, character_tags, user_tags, character_count_tags)
        'masterpiece, best quality, high quality, 1girl, solo, blonde hair, blue eyes, sitting, drinking beer, smiling, leather armor, tavern interior'
        """
        parts = []

        # Block 0: Quality (hardwired - first 3 tags only)
        parts.extend(BLOCK_0[:3])  # ["masterpiece", "best quality", "high quality"]

        # Character tags (count + appearance combined)
        if character_count_tags:
            parts.append(character_count_tags)
        parts.extend(character_tags[:20])

        # Action + Activity + Expression (from LLM JSON)
        action_text = scene_json.get('action', '')
        activity_text = scene_json.get('activity', '')
        expression_text = scene_json.get('expression', '')

        # Build action+activity+expression combination
        action_parts = []
        if action_text:
            action_parts.append(action_text)
        if activity_text:
            action_parts.append(activity_text)
        if expression_text:
            action_parts.append(expression_text)  # Expression after action+activity

        if action_parts:
            parts.append(', '.join(action_parts))

        # Dress (from LLM JSON)
        if scene_json.get('dress'):
            parts.append(scene_json['dress'])

        # Location (from LLM JSON) - force "at" prefix
        if scene_json.get('location'):
            location = scene_json['location'].strip()
            parts.append(f"at {location}")

        # User tags
        parts.extend(user_tags[:5])
        
        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            lower = part.lower()
            if lower not in seen:
                seen.add(lower)
                unique_parts.append(part)
        
        # Join with commas (standard SD format)
        positive_prompt = ', '.join(unique_parts)
        
        # Generate negative prompt (universal only)
        negative_prompt = ', '.join(get_universal_negatives())
        
        print(f"[SNAPSHOT] Final prompt: {positive_prompt}")
        
        return positive_prompt, negative_prompt
    
    # DEPRECATED METHODS - Kept for API compatibility, but not used
    # These will be removed in future versions
    
    def build_4_block_prompt(self, *args, **kwargs) -> Tuple[str, str]:
        """
        DEPRECATED: Use build_simple_prompt() instead.
        
        This method is kept for backward compatibility but now delegates
        to the simplified prompt builder.
        """
        print("[SNAPSHOT] Warning: build_4_block_prompt() is deprecated, use build_simple_prompt()")
        
        # Try to convert old-style parameters to new format
        if 'scene_analysis' in kwargs:
            scene_analysis = kwargs['scene_analysis']
            scene_json = scene_analysis.get('scene_json', {})
        else:
            scene_json = {}
        
        character_tags = kwargs.get('active_chars_data', [])
        user_tags = kwargs.get('user_data', [])
        character_count_tags = kwargs.get('character_count_tags', '')
        
        return self.build_simple_prompt(scene_json, character_tags, user_tags, character_count_tags)
    
    def _build_block_1(self, *args, **kwargs) -> List[str]:
        """DEPRECATED - Not used in simplified system."""
        return []
    
    def _build_block_2_semantic(self, *args, **kwargs) -> List[str]:
        """DEPRECATED - Not used in simplified system."""
        return []
    
    def _build_block_3_semantic(self, *args, **kwargs) -> List[str]:
        """DEPRECATED - Not used in simplified system."""
        return []
    
    def _build_negative_prompt(self) -> str:
        """DEPRECATED - Use get_universal_negatives() directly."""
        return ', '.join(get_universal_negatives())
    
    def _calculate_tag_score(self, *args, **kwargs) -> float:
        """DEPRECATED - Scoring removed in simplified system."""
        return 1.0
    
    def _select_tags_weighted(self, *args, **kwargs) -> List[str]:
        """DEPRECATED - Weighted selection removed in simplified system."""
        return []
