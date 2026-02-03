"""
Snapshot Scene Analyzer - Simplified Version

Analyzes conversation context and extracts scene description via LLM as JSON.
Uses progressive fallback chain when LLM fails.

EXAMPLE:
Input: "Alice and Bob are sitting in a tavern. Alice is drinking a beer."
Output: {
    "location": "cozy tavern interior",
    "action": "sitting",
    "dress": ""
}

FALLBACK CHAIN:
1. LLM JSON extraction (up to 3 retry attempts)
2. Simple pattern extraction (key:value or line-based)
3. Basic keyword matching (small curated lists)
4. Empty scene (fail-safe)
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Any


class SnapshotAnalyzer:
    """Analyzes conversation context for snapshot generation using LLM JSON extraction."""

    def __init__(self, semantic_search_engine=None, http_client=None, config=None):
        """
        Initialize the snapshot analyzer.
        
        Args:
            semantic_search_engine: DEPRECATED - kept for API compatibility
            http_client: httpx.AsyncClient for LLM requests (optional)
            config: Application config dict with kobold_url (optional)
        """
        self.semantic_search_engine = semantic_search_engine  # Kept for compatibility, not used
        self.http_client = http_client
        self.config = config or {}

    def _strip_character_names(self, text: str, character_names: List[str]) -> str:
        """
        Strip character names from text, replacing with generic 'Character'.
        
        Args:
            text: Original text content
            character_names: List of character names to strip
            
        Returns:
            Text with character names replaced by 'Character'
        """
        if not character_names or not text:
            return text
        
        result = text
        for name in character_names:
            if not name:
                continue
            # Replace whole word matches only (case-insensitive)
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(name) + r'\b'
            result = re.sub(pattern, 'Character', result, flags=re.IGNORECASE)
        
        return result
    
    def extract_conversation_context(self, messages: List[Dict],
                                     message_count: int = 2,
                                     character_names: Optional[List[str]] = None) -> str:
        """
        Extract last N messages and format for analysis.
        
        Args:
            messages: Message list from chat
            message_count: Number of recent messages to extract (default: 2 = 1 turn)
            character_names: List of character names to strip from content (prevents name leakage)
        
        Returns:
            Formatted conversation text with character names replaced by 'Character'
        """
        if not messages:
            return ""

        # Get last message_count messages
        recent_messages = messages[-message_count:]

        # Format messages
        context_lines = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            speaker = msg.get('speaker', role)

            # Truncate long messages to prevent prompt bloat
            if len(content) > 500:
                content = content[:500] + "..."
            
            # Strip character names from content (prevents name leakage to LLM)
            if character_names:
                content = self._strip_character_names(content, character_names)
                # Also strip from speaker name to anonymize
                speaker = self._strip_character_names(speaker, character_names)
                if speaker != msg.get('speaker', role) and speaker != role:
                    # Speaker was a character name, replace with 'Character'
                    speaker = 'Character'

            if role == 'user':
                context_lines.append(f"User: {content}")
            elif role == 'assistant':
                context_lines.append(f"{speaker}: {content}")

        return '\n'.join(context_lines)

    async def extract_scene_json(self, conversation_context: str, max_retries: int = 3) -> Dict[str, str]:
        """
        Extract scene description as JSON with progressive retry strategy.
        
        Priority hierarchy for actions:
        1. Physical interactions with others: hugging another, embracing another, holding hands, dancing together, fighting, attacking, carrying another
        2. Physical actions (solo): sitting, standing, walking, running, jumping, drinking, eating, sleeping, casting spell, fighting stance
        3. Speaking or gestures: talking, whispering, shouting, hand gestures (ONLY if no physical action)
        4. Generic: "standing" or "sitting" (last resort)
        
        CONSTRAINTS:
        - NO proper names: Use generic terms only (e.g., "hugging another", not "Alice hugging Bob")
        - SINGLE action only: Describe ONE action (e.g., "sitting", not "sitting and drinking")
        - Third-person: Use "he/she/they", NEVER use "you" or "your"
        
        Args:
            conversation_context: Last 2 turns of conversation
            max_retries: Number of LLM attempts (default: 3)
        
        Returns:
            {'location': str, 'action': str, 'dress': str}
            Returns empty dict on all retries failed.
        """
        prompts = [
            # Attempt 1: Standard with full instructions
            f"""Summarize the last two messages as JSON with these keys:
- "location": where the scene is happening (short phrase, 3-5 words). Generic terms only - NO proper names.
- "action": what the main character is doing (single action, 2-3 words). ONE action only - NOT multiple actions. Third-person only.
- "dress": how the character is dressed (short phrase, 2-3 words). Leave empty if no clothing description available.

ACTION PRIORITY (use this order):
1. Physical interactions: hugging another, embracing another, holding hands, dancing together, fighting, attacking, carrying another
2. Physical actions (solo): sitting, standing, walking, running, jumping, drinking, eating, sleeping, casting spell, fighting stance
3. Speaking or gestures: talking, whispering, shouting (ONLY if no physical action)
4. Generic: "standing" or "sitting" (last resort)

CONSTRAINTS - FOLLOW STRICTLY:
- NO proper names: Use "another" or generic terms (e.g., "hugging another", NOT "Alice hugging Bob")
- SINGLE action only: One verb (e.g., "sitting", NOT "sitting and drinking")
- Third-person: Use "he/she/they" context, NEVER use "you" or "your"
- Empty dress: Leave blank "" if clothing info is unknown

Reply with ONLY valid JSON, no extra text.

Recent conversation:
{conversation_context[:1000]}

Assistant:""",
            
            # Attempt 2: More strict
            f"""MUST reply with ONLY valid JSON. No explanations, no extra text.

Keys:
- "location": where the scene is happening (3-5 words, NO proper names)
- "action": ONE single action only (2-3 words, third-person, NEVER "you")
- "dress": clothing description or leave empty "" if unknown

Action priority: interactions → solo actions → speaking → standing.

Recent conversation:
{conversation_context[:1000]}

Assistant:""",
            
            # Attempt 3: Ultra-strict
            f"""Your ENTIRE response must be ONLY this format:
{{{{"location": "...", "action": "...", "dress": ""}}}}

RULES:
- location: 3-5 words, generic only, NO proper names like "Alice's" or "The Golden Dragon"
- action: ONE single action word (e.g., "sitting", "hugging another"), NOT "sitting and drinking"
- dress: clothing description OR empty string "" if unknown
- NEVER use "you" or "your"

Do not write ANYTHING else.

Recent conversation:
{conversation_context[:1000]}"""
        ]
        
        for attempt in range(max_retries):
            # Check if we have LLM available
            if self.http_client is None or not self.config.get('kobold_url'):
                print("[SNAPSHOT] LLM unavailable, skipping JSON extraction")
                return {}
                
            try:
                prompt = prompts[min(attempt, len(prompts) - 1)]
                
                response = await self.http_client.post(
                    f"{self.config['kobold_url']}/api/v1/generate",
                    json={
                        "prompt": prompt,
                        "max_length": 150,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "stop_sequence": ["###", "\n\n", "Assistant:"]
                    },
                    timeout=30.0
                )
                
                result = response.json().get('results', [{}])[0].get('text', '').strip()
                print(f"[SNAPSHOT DEBUG] Attempt {attempt + 1}: '{result[:100]}...'")
                
                # Parse JSON
                parsed = json.loads(result)
                
                # Validate required fields
                if all(k in parsed for k in ['location', 'action', 'dress']):
                    print(f"[SNAPSHOT] JSON extraction succeeded on attempt {attempt + 1}")
                    return parsed
                else:
                    print(f"[SNAPSHOT] Attempt {attempt + 1}: Missing required fields: {list(parsed.keys())}")
                    
            except json.JSONDecodeError as e:
                print(f"[SNAPSHOT] Attempt {attempt + 1}: JSON parsing failed - {e}")
            except Exception as e:
                print(f"[SNAPSHOT] Attempt {attempt + 1}: Error - {e}")
        
        print(f"[SNAPSHOT] JSON extraction failed after {max_retries} attempts, trying fallback")
        return {}

    def _extract_from_patterns(self, text: str) -> Dict[str, str]:
        """
        Fallback: Extract from key:value or line-based patterns.
        
        Handles:
        - Location: tavern
        - Action: drinking
        - Dress: leather armor
        
        Or:
        location: tavern
        action: drinking
        dress: leather armor
        """
        text_lower = text.lower()
        result = {'location': '', 'action': '', 'dress': ''}
        
        # Pattern 1: Key: Value format
        patterns = {
            'location': r'(?:location|where)[\s:]+([^\n]+)',
            'action': r'(?:action|doing)[\s:]+([^\n]+)',
            'dress': r'(?:dress|wearing|outfit)[\s:]+([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                result[key] = match.group(1).strip()[:50]  # Limit length
        
        # Pattern 2: Line-based (each line is a value)
        lines = text.strip().split('\n')
        if len(lines) >= 3:
            # Assume order: location, action, dress
            if not result['location']:
                result['location'] = lines[0].strip()[:50]
            if not result['action']:
                result['action'] = lines[1].strip()[:50]
            if not result['dress']:
                result['dress'] = lines[2].strip()[:50]
        
        # Check if we got at least one value
        if any(result.values()):
            print(f"[SNAPSHOT] Pattern extraction: {result}")
            return result
        
        return {}

    def _extract_from_keywords(self, text: str) -> Dict[str, str]:
        """
        Fallback: Simple keyword matching.
        
        Uses small curated lists, not semantic search.
        """
        text_lower = text.lower()
        result = {'location': '', 'action': '', 'dress': ''}
        
        # Small curated lists
        LOCATION_KEYWORDS = [
            'tavern', 'inn', 'bar', 'castle', 'palace', 'forest', 'woods', 'cave',
            'room', 'bedroom', 'kitchen', 'hall', 'street', 'road', 'river',
            'beach', 'mountain', 'garden', 'temple', 'church', 'library',
            'throne room', 'dungeon', 'market', 'shop'
        ]
        
        ACTION_KEYWORDS = [
            'sitting', 'standing', 'walking', 'running', 'jumping', 'drinking', 'eating',
            'fighting', 'attacking', 'defending', 'hugging', 'kissing', 'embracing',
            'holding hands', 'dancing', 'sleeping', 'casting spell', 'talking',
            'hugging another', 'embracing another', 'dancing together'
        ]
        
        DRESS_KEYWORDS = [
            'armor', 'dress', 'casual', 'formal', 'robe', 'cloak', 'shirt',
            'uniform', 'leather', 'naked', 'clothing', 'outfit'
        ]
        
        # Check each category
        for loc in LOCATION_KEYWORDS:
            if loc in text_lower:
                result['location'] = loc
                break
        
        for act in ACTION_KEYWORDS:
            if act in text_lower:
                result['action'] = act
                break
        
        for dress in DRESS_KEYWORDS:
            if dress in text_lower:
                result['dress'] = dress
                break
        
        # Fallback action: standing if none found
        if not result['action']:
            result['action'] = 'standing'
        
        if any(result.values()):
            print(f"[SNAPSHOT] Keyword fallback: {result}")
            return result
        
        return {}

    async def analyze_scene(self, messages: List[Dict], chat_id: str,
                           character_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simplified scene analysis: extract JSON from LLM with fallback chain.
        
        Args:
            messages: Chat messages
            chat_id: Chat ID for context
            character_names: List of character names to strip from context (prevents name leakage)
        
        Returns:
            {
                'scene_json': {'location': str, 'action': str, 'dress': str},
                'recent_context': str,
                'error': str  # Empty on success, error message otherwise
            }
        """
        recent_context = self.extract_conversation_context(
            messages, 
            message_count=2,
            character_names=character_names
        )
        
        if not recent_context:
            return {
                'scene_json': {},
                'recent_context': '',
                'error': 'No conversation context'
            }
        
        # Chain 1: JSON extraction (up to 3 retries)
        scene_json = await self.extract_scene_json(recent_context)
        
        if not scene_json:
            # Chain 2: Simple pattern extraction
            print("[SNAPSHOT] JSON failed, trying pattern extraction")
            scene_json = self._extract_from_patterns(recent_context)
        
        if not scene_json:
            # Chain 3: Keyword fallback
            print("[SNAPSHOT] Pattern failed, trying keyword fallback")
            scene_json = self._extract_from_keywords(recent_context)
        
        if not scene_json or not any(scene_json.values()):
            # Chain 4: Empty scene (fail-safe)
            print("[SNAPSHOT] All fallbacks failed, using empty scene")
            scene_json = {'location': '', 'action': '', 'dress': ''}
        
        return {
            'scene_json': scene_json,
            'recent_context': recent_context,
            'error': ''
        }

    def get_character_tag(self, char_ref: str, chat_data: Dict) -> Optional[str]:
        """
        Get danbooru tag for a character (DEPRECATED - kept for compatibility).
        
        This function is kept for API compatibility but should not be used
        in the new simplified system.
        
        Args:
            char_ref: Character reference (filename or npc_xxx)
            chat_data: Chat data dict with metadata
        
        Returns:
            Danbooru tag string or None
        """
        return None

    def clear_cache(self):
        """Clear the character tag cache (DEPRECATED - kept for compatibility)."""
        pass
