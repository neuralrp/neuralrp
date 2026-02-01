"""
Snapshot Scene Analyzer

Analyzes conversation context to determine scene type, setting, action, and mood.
Uses hybrid approach:
1. Keyword detection (fast path, always runs first)
2. LLM summary (async, optional enhancement)
3. Semantic tag matching (via all-mpnet-base-v2)

Falls back to keyword-only detection if LLM is unavailable.
"""

import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Predefined scene type keywords (minimum required for keyword detection)
SCENE_KEYWORDS: Dict[str, List[str]] = {
    'combat': [
        'battle', 'fight', 'sword', 'attack', 'defend', 'weapon',
        'war', 'kill', 'die', 'wound', 'blood', 'shield',
        'strike', 'slash', 'stab', 'punch', 'kick', 'block',
        'parry', 'dodge', 'enemy', 'foe', 'opponent', 'duel'
    ],
    'dialogue': [
        'speak', 'say', 'tell', 'ask', 'reply', 'conversation',
        'discuss', 'argue', 'whisper', 'shout', 'call', 'talk',
        'respond', 'answer', 'question', 'explain', 'describe',
        'narrate', 'mention', 'comment', 'remark', 'state'
    ],
    'exploration': [
        'explore', 'travel', 'walk', 'journey', 'path', 'road',
        'forest', 'mountain', 'discover', 'search', 'find', 'map',
        'venture', 'trek', 'hike', 'wander', 'roam', 'navigate',
        'investigate', 'examine', 'inspect', 'scout', 'survey'
    ],
    'romance': [
        'love', 'kiss', 'hug', 'hold', 'embrace', 'romantic',
        'heart', 'emotion', 'feeling', 'care', 'affection', 'date',
        'blush', 'tender', 'gentle', 'intimate', 'passion', 'desire',
        'adore', 'cherish', 'devotion', 'longing', 'yearning'
    ],
    'tavern': [
        'tavern', 'inn', 'drink', 'ale', 'bar', 'food', 'meal',
        'rest', 'sleep', 'room', 'bed', 'warmth', 'fireplace',
        'mug', 'wine', 'beer', 'toast', 'cheers', 'patron',
        'bartender', 'innkeeper', 'lodging', 'accommodation'
    ],
    'magic': [
        'spell', 'magic', 'fireball', 'heal', 'curse', 'summon',
        'mana', 'wizard', 'sorcery', 'enchant', 'potion', 'wand',
        'incantation', 'ritual', 'arcane', 'mystical', 'rune',
        'conjure', 'transmute', 'teleport', 'illusion', 'ward'
    ]
}


class SnapshotAnalyzer:
    """Analyzes conversation context for snapshot generation."""

    def __init__(self, semantic_search_engine, http_client=None, config=None):
        """
        Initialize the snapshot analyzer.

        Args:
            semantic_search_engine: The SemanticSearchEngine instance for embeddings
            http_client: httpx.AsyncClient for LLM requests (optional)
            config: Application config dict with kobold_url (optional)
        """
        self.semantic_search_engine = semantic_search_engine
        self.http_client = http_client
        self.config = config or {}
        self._char_tag_cache: Dict[str, Optional[str]] = {}  # Cache for character tags

    def extract_conversation_context(self, messages: List[Dict],
                                     message_count: int = 4) -> str:
        """
        Extract last N messages and format for analysis.

        Args:
            messages: Message list from chat
            message_count: Number of recent messages to extract (default: 4 = 2 turns)

        Returns:
            Formatted conversation text
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

            if role == 'user':
                context_lines.append(f"User: {content}")
            elif role == 'assistant':
                context_lines.append(f"{speaker}: {content}")

        return '\n'.join(context_lines)

    def detect_scene_type_keywords(self, conversation_text: str) -> Tuple[str, List[str]]:
        """
        Detect scene type via keyword matching (fast path).

        Returns:
            Tuple of (scene_type, matched_keywords)
        """
        text_lower = conversation_text.lower()

        best_scene = 'other'
        best_matches: List[str] = []
        best_count = 0

        # Check each scene type's keywords and count matches
        for scene_type, keywords in SCENE_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if len(matches) > best_count:
                best_count = len(matches)
                best_scene = scene_type
                best_matches = matches

        # Require at least 2 keyword matches to be confident
        if best_count < 2:
            return 'other', []

        return best_scene, best_matches

    async def summarize_scene_via_llm(self, conversation_context: str) -> Dict[str, str]:
        """
        Use LLM to summarize scene (last 2 turns).

        Returns:
            {'scene_type': str, 'setting': str, 'action': str, 'mood': str}
            Returns empty dict if LLM is unavailable.
        """
        # Check if we have the required components
        if self.http_client is None or not self.config.get('kobold_url'):
            print("[SNAPSHOT] LLM unavailable, skipping scene summary")
            return {}

        # Simplified prompt for faster response
        prompt = f"""Analyze this roleplay scene briefly.

{conversation_context[:800]}

Reply with ONLY this JSON (no explanation):
{{"scene_type":"combat|dialogue|exploration|romance|magic|other","setting":"brief place","mood":"emotion"}}"""

        try:
            response = await self.http_client.post(
                f"{self.config['kobold_url']}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_length": 80,
                    "temperature": 0.3,
                    "top_p": 0.9
                },
                timeout=10.0
            )

            result = response.json().get('results', [{}])[0].get('text', '')

            # Parse JSON from response (handle potential markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return {
                        'scene_type': parsed.get('scene_type', 'other'),
                        'setting': parsed.get('setting', ''),
                        'mood': parsed.get('mood', '')
                    }
                except json.JSONDecodeError:
                    pass

            return {}

        except Exception as e:
            print(f"[SNAPSHOT] LLM summary failed: {e}")
            return {}

    def match_tags_semantically(self, query_text: str,
                               block_num: Optional[int] = None,
                               k: int = 15,
                               threshold: float = 0.35) -> List[Tuple[str, float]]:
        """
        Match danbooru tags via semantic search.

        Args:
            query_text: Text to search for
            block_num: Filter by block (optional)
            k: Maximum number of results
            threshold: Similarity threshold (default: 0.35)

        Returns:
            List of (tag_text, similarity_score) tuples
        """
        if not self.semantic_search_engine or not self.semantic_search_engine.load_model():
            print("[SNAPSHOT] Semantic search unavailable")
            return []

        if not query_text or not query_text.strip():
            return []

        # Generate embedding for query
        try:
            query_embedding = self.semantic_search_engine.model.encode(
                [query_text], convert_to_numpy=True
            )[0]
        except Exception as e:
            print(f"[SNAPSHOT] Failed to generate embedding: {e}")
            return []

        # Import database function (avoid circular import by importing here)
        from app.database import db_search_danbooru_embeddings

        # Search danbooru tags
        results = db_search_danbooru_embeddings(
            query_embedding,
            block_num=block_num,
            k=k,
            threshold=threshold
        )

        return results

    async def analyze_scene(self, messages: List[Dict], chat_id: str) -> Dict[str, Any]:
        """
        Complete scene analysis: keyword detection + LLM summary (if available).

        Falls back to keyword-only if LLM unavailable.

        Returns:
            {
                'scene_type': str,
                'setting': str,
                'mood': str,
                'keyword_detected': bool,
                'matched_keywords': List[str],
                'llm_used': bool
            }
        """
        # Step 1: Extract conversation context (last 2 turns = 4 messages)
        conversation_text = self.extract_conversation_context(messages, message_count=4)

        if not conversation_text:
            return {
                'scene_type': 'other',
                'setting': '',
                'mood': '',
                'keyword_detected': False,
                'matched_keywords': [],
                'llm_used': False
            }

        # Step 2: Keyword detection (fast path, always runs)
        scene_type, matched_keywords = self.detect_scene_type_keywords(conversation_text)
        keyword_detected = scene_type != 'other'

        # Step 3: LLM summary (optional enhancement)
        llm_summary = await self.summarize_scene_via_llm(conversation_text)
        llm_used = bool(llm_summary)

        # Step 4: Combine results
        # Keyword detection takes priority for scene_type (more reliable)
        # LLM provides setting and mood details
        final_scene_type = scene_type if keyword_detected else llm_summary.get('scene_type', 'other')
        final_setting = llm_summary.get('setting', '')
        final_mood = llm_summary.get('mood', '')

        # If no setting from LLM, infer from scene type
        if not final_setting and keyword_detected:
            setting_hints = {
                'combat': 'battlefield',
                'dialogue': 'conversation scene',
                'exploration': 'outdoor adventure',
                'romance': 'intimate setting',
                'tavern': 'tavern interior',
                'magic': 'magical atmosphere'
            }
            final_setting = setting_hints.get(scene_type, '')

        # If no mood from LLM, infer from scene type
        if not final_mood and keyword_detected:
            mood_hints = {
                'combat': 'intense',
                'dialogue': 'conversational',
                'exploration': 'adventurous',
                'romance': 'romantic',
                'tavern': 'relaxed',
                'magic': 'mystical'
            }
            final_mood = mood_hints.get(scene_type, '')

        result = {
            'scene_type': final_scene_type,
            'setting': final_setting,
            'mood': final_mood,
            'keyword_detected': keyword_detected,
            'matched_keywords': matched_keywords,
            'llm_used': llm_used
        }

        print(f"[SNAPSHOT] Scene analysis: type={final_scene_type}, "
              f"keyword={keyword_detected}, llm={llm_used}")

        return result

    def get_character_tag(self, char_ref: str, chat_data: Dict) -> Optional[str]:
        """
        Get danbooru tag for a character with caching.

        Args:
            char_ref: Character reference (filename or npc_xxx)
            chat_data: Chat data dict with metadata

        Returns:
            Danbooru tag string or None
        """
        # Check cache first
        if char_ref in self._char_tag_cache:
            return self._char_tag_cache[char_ref]

        danbooru_tag = None

        if char_ref.startswith('npc_'):
            # Load NPC from chat metadata
            npcs = chat_data.get('metadata', {}).get('localnpcs', {})
            npc_data = npcs.get(char_ref)
            if npc_data:
                danbooru_tag = npc_data.get('data', {}).get('extensions', {}).get('danbooru_tag', '')
        else:
            # Load global character (import here to avoid circular import)
            from app.database import db_get_character
            char_data = db_get_character(char_ref)
            if char_data:
                danbooru_tag = char_data.get('data', {}).get('extensions', {}).get('danbooru_tag', '')

        # Cache result (even if None)
        self._char_tag_cache[char_ref] = danbooru_tag or None

        return danbooru_tag or None

    def clear_cache(self):
        """Clear the character tag cache."""
        self._char_tag_cache.clear()
