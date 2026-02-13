"""
Snapshot Scene Analyzer - Primary Character Focused

Analyzes conversation context and extracts scene description via LLM as JSON.
Focused on a single primary character with 20-message context window.

EXAMPLE:
Input: "Alice and Bob are sitting in a tavern. Alice is drinking a beer."
Output (primary character=Alice):
{
    "location": "cozy tavern interior",
    "action": "drinking",
    "activity": "at tavern",
    "dress": "casual clothes",
    "expression": "smiling"
}

EXTRACTION LOGIC:
1. LLM JSON extraction (3 retry attempts) from last 20 messages
2. Character card injection for context (description + personality only)
3. Primary character focus (determined by mode: auto/focus:name/narrator)
4. Extracts 5 fields: location, action, activity, dress, expression
5. Empty scene on LLM failure (no semantic search fallback)
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None


class SnapshotAnalyzer:
    """Analyzes conversation context for snapshot generation using LLM JSON extraction."""

    def __init__(self, semantic_search_engine=None, http_client=None, config=None):
        """
        Initialize the snapshot analyzer.
        
        Args:
            semantic_search_engine: SemanticSearchEngine instance for history search
            http_client: httpx.AsyncClient for LLM requests (optional)
            config: Application config dict with kobold_url (optional)
        """
        self.semantic_search_engine = semantic_search_engine
        self.http_client = http_client
        self.config = config or {}

    def _strip_character_names(self, text: str, character_names: List[str]) -> str:
        """
        Strip character names from text, replacing with 'another' or 'another's'.
        
        Uses 'another' instead of 'character' because:
        - LLM prompt examples already use "another" (e.g., "hugging another")
        - "another" is a common danbooru tag understood by Pony/Illustrious
        - Prevents confusion with generic word "character"
        
        Preserves possessive forms: "Rebecca's" → "another's", "Rebecca" → "another"
        
        Args:
            text: Original text content
            character_names: List of character names to strip
            
        Returns:
            Text with character names replaced by 'another' (preserving possessive)
        """
        if not character_names or not text:
            return text
        
        result = text
        for name in character_names:
            if not name:
                continue
            
            if name.endswith("'s"):
                # Name already has possessive form (e.g., "Saint Peter's")
                # Match name as-is, followed by word boundary
                pattern = r'\b' + re.escape(name) + r'\b'
            else:
                # Name without possessive (e.g., "James")
                # Match with optional possessive suffix (handles "James", "James's", "James'")
                # Use negative lookahead to prevent matching mid-word, instead of \b which fails after apostrophe
                pattern = r'\b' + re.escape(name) + r"(?:'s|')?(?!\w)"
            
            def replace_match(match):
                matched_text = match.group(0)
                if matched_text.endswith("'s") or matched_text.endswith("'"):
                    return "another's"
                return "another"
            
            result = re.sub(pattern, replace_match, result, flags=re.IGNORECASE)
        
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
            Formatted conversation text with character names replaced by 'another'
        """
        if not messages:
            return ""

        # Get last message_count messages
        recent_messages = messages[-message_count:]

        # Format messages
        context_lines = []
        for i, msg in enumerate(recent_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            speaker = msg.get('speaker', role)

            # Truncate long messages to prevent prompt bloat
            # EXCEPT the most recent message (last one in recent_messages) - keep in full
            is_most_recent = (i == len(recent_messages) - 1)
            if len(content) > 350 and not is_most_recent:
                content = content[:350] + "..."

            # Strip character names from content (prevents name leakage to LLM)
            if character_names:
                content = self._strip_character_names(content, character_names)
                # Also strip from speaker name to anonymize
                speaker = self._strip_character_names(speaker, character_names)
                if speaker != msg.get('speaker', role) and speaker != role:
                    # Speaker was a character name, replace with 'another'
                    speaker = 'another'

            if role == 'user':
                context_lines.append(f"User: {content}")
            elif role == 'assistant':
                context_lines.append(f"{speaker}: {content}")

        return '\n'.join(context_lines)

    async def extract_character_scene_json(
        self,
        messages: List[Dict],
        primary_character: Optional[str],
        primary_character_card: Optional[str],
        message_window: int = 20
    ) -> Dict[str, str]:
        """
        Extract scene description focused on primary character using LLM.

        Args:
            messages: All chat messages
            primary_character: Name of primary character to focus on
            primary_character_card: Character card data for context (description + personality only)
            message_window: Number of recent messages to analyze (default: 20)

        Returns:
            {'location': str, 'action': str, 'activity': str, 'dress': str, 'expression': str}
            Returns empty dict on LLM failure.
        """
        # Extract last N messages for analysis
        recent_messages = messages[-message_window:] if len(messages) > message_window else messages

        # Format conversation for LLM
        conversation_text = self.extract_conversation_context(
            recent_messages,
            message_count=message_window,
            character_names=[primary_character] if primary_character else None
        )

        if not conversation_text:
            return {}

        # Build character context section
        character_context = ""
        if primary_character and primary_character_card:
            # Use generic header to avoid leaking character name to LLM
            character_context = f"\n\n[CHARACTER PROFILE - PRIMARY CHARACTER]\n{primary_character_card}\n"

        # Construct prompt with character focus
        prompt = f"""Analyze the conversation and extract scene information as JSON.

{character_context}
{"Focus on primary character" if primary_character else "No character focus"}

CONTEXT: You will receive 20 recent messages:
- Messages 1-19: Truncated to 350 characters each (may be incomplete)
- Message 20: Full text (most recent turn)

MISSION: Intelligently infer scene information from context clues across all messages. You are ALLOWED to make educated guesses when information is incomplete.

OUTPUT STYLE: Generate danbooru/booru-style tags (short phrases used in anime image generation models like PonyXL/Illustrious). Use lowercase, avoid filler words like "is", "the", "a". Examples: "smile" not "smiling", "sitting" not "is sitting".

Extract:
1. "location": Where is the scene happening? (EXACTLY 3-5 words, NO proper names)
   GUESS from context clues across all 20 messages (activities, descriptions, dialogue hints)
   Examples: "tavern interior", "dark forest", "stone bridge", "cozy bedroom"
   WRONG: "Golden Dragon Tavern" (proper name), "Alice's bedroom" (proper name)

2. "action": What is the {"primary character" if primary_character else "main character"} doing RIGHT NOW in the MOST RECENT TURN? (EXACTLY 2-3 words, NO proper names)
   CRITICAL: DETERMINE from FINAL MESSAGE ONLY (message 20, full text)
   Action priority: Physical action with another → Physical action with self → Emotional reaction → Passive pose
   Examples: "hugging another", "waving hand", "crying", "standing", "fistfight"
   WRONG: "fighting with Bob" (proper name), "Alice is drinking" (proper name)

3. "activity": What is the most helpful 1-4 word phrase for image generation that describes the current scene? (1-4 words, NO proper names, or "" if no context)
    CRITICAL: This should be optimized for Stable Diffusion generation - the single most useful phrase to understand the scene.
    - Consider the broader context that makes the immediate action coherent for SD
    - If action is "trips" and location is "at park", use "walking with another" (frames the action)
    - If action is "doing dishes" and context is tavern, use "working" or "tavern scene"
    - Let the LLM determine what context is most helpful, even if it overlaps with other fields
    Examples: "walking with another", "playing volleyball at beach", "watching romantic sunset", "taking off shoes"
    WRONG: "trips" (that's the action field), "payment negotiation" (contradicts action), "reading Alice's book" (proper name)

4. "dress": What is the {"primary character" if primary_character else "the character"} wearing RIGHT NOW? (EXACTLY 2-3 words, NO proper names, or "nude" if NSFW)
     CRITICAL: Determine from MOST RECENT context in the 20 messages - prioritize later messages (especially messages 15-20)
     If outfit changed recently, use the NEWEST description, NOT earlier outfits from messages 1-10
     GUESS from context clues (activity, location, genre) - make reasonable inferences if not explicitly stated
     Examples: "armor", "casual", "swimsuit", "school uniform", "dress", "jeans"
     WRONG: "Alice's dress" (proper name), "Bob's helmet" (proper name), or using an old outfit that changed 10 messages ago
     Use "nude" if nudity mentioned or strongly implied by activity/location
     Use "" (empty string) only if absolutely no clues available

5. "expression": What is the facial expression of the {"primary character" if primary_character else "the character"}? (EXACTLY 1-2 words, NO proper names)
     CRITICAL: Consider the character's PERSONALITY TRAITS from the profile above.
     - If character is shy/timid, use subtle expressions ("nervous", "embarrassed")
     - If character is bold/cheerful, use stronger expressions ("smile", "excited")
     - Match expression to both current situation AND personality
     DETERMINE from dialogue, context, and emotional cues in the conversation
     Examples: "surprised", "frown", "smile", "angry", "neutral", "blush", "closed eyes"
     WRONG: "Alice looks surprised" (proper name), "happy Bob" (proper name)
     Use "neutral" if no emotional cues are available or if unclear
     Use "" (empty string) only if character is not present in scene

IMPORTANT:
- ACTION: Determine from FINAL MESSAGE ONLY (message 20, full text)
- ACTIVITY/LOCATION/EXPRESSION: Infer from context across all 20 messages, you are encouraged to make educated guesses
- DRESS: Prioritize MOST RECENT messages (especially messages 15-20), NOT earlier outfits from messages 1-10
- NO proper names in any field: Use generic terms only (e.g., "fighting with another", NOT "fighting with Bob")
- EXACTLY 2-3 words per field (or 3-5 for location, 1-4 for activity, 1-2 for expression), no exceptions
- Use context clues to infer location, activity, dress, and expression when not explicitly stated
- Third-person only, NO "you" or "your"
- Location is shared by all characters

Reply with ONLY valid JSON, no extra text.

Conversation:
{conversation_text}

Assistant:"""

        # Generate response with retry logic
        for attempt in range(3):
            # Check if we have LLM available
            if self.http_client is None or not self.config.get('kobold', {}).get('url'):
                print("[SNAPSHOT] LLM unavailable, skipping JSON extraction")
                return {}

            try:
                response = await self.http_client.post(
                    f"{self.config.get('kobold', {}).get('url', '')}/api/v1/generate",
                    json={
                        "prompt": prompt,
                        "max_length": 75,
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
                if all(k in parsed for k in ['location', 'action', 'activity', 'dress', 'expression']):
                    print(f"[SNAPSHOT] Character-scoped JSON extraction succeeded on attempt {attempt + 1}")
                    return parsed
                else:
                    print(f"[SNAPSHOT] Attempt {attempt + 1}: Missing required fields: {list(parsed.keys())}")

            except json.JSONDecodeError as e:
                print(f"[SNAPSHOT] Attempt {attempt + 1}: JSON parsing failed - {e}")
            except Exception as e:
                print(f"[SNAPSHOT] Attempt {attempt + 1}: Error - {e}")

        print(f"[SNAPSHOT] Character-scoped JSON extraction failed after 3 attempts")
        return {}

    async def analyze_scene(self, messages: List[Dict], chat_id: str,
                           character_names: Optional[List[str]] = None,
                           primary_character: Optional[str] = None,
                           primary_character_card: Optional[str] = None) -> Dict[str, Any]:
        """
        Simplified scene analysis using character-scoped LLM extraction.

        Args:
            messages: Chat messages
            chat_id: Chat ID for context
            character_names: List of character names to strip from context
            primary_character: Name of primary character to focus on
            primary_character_card: Character card data (description + personality)

        Returns:
            {
                'scene_json': {'location': str, 'action': str, 'activity': str, 'dress': str, 'expression': str},
                'recent_context': str,
                'error': str  # Empty on success, error message otherwise
            }
        """
        # Use character-scoped extraction with 20-message window
        scene_json = await self.extract_character_scene_json(
            messages,
            primary_character,
            primary_character_card,
            message_window=20
        )

        # Extract recent context for display (last 2 messages)
        recent_context = self.extract_conversation_context(
            messages[-2:] if len(messages) >= 2 else messages,
            message_count=2,
            character_names=character_names
        )

        if not scene_json:
            print("[SNAPSHOT] Character-scoped extraction failed, returning empty scene")
            scene_json = {'location': '', 'action': '', 'activity': '', 'dress': '', 'expression': ''}

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

    async def extract_location_dress_for_cache(
        self,
        messages: List[Dict],
        character_names: Optional[List[str]] = None,
        primary_character: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extract location and dress for caching during summarization.
        
        This is a lightweight extraction used to cache scene state that can be
        reused when generating snapshots, avoiding the need to analyze 20+ messages.
        
        Args:
            messages: Chat messages to analyze
            character_names: Names to strip from context
            primary_character: Primary character to focus on
            
        Returns:
            {'location': str, 'dress': str} - may be empty strings if extraction fails
        """
        if not messages:
            return {'location': '', 'dress': ''}
        
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        conversation_text = self.extract_conversation_context(
            recent_messages,
            message_count=10,
            character_names=character_names
        )
        
        if not conversation_text:
            return {'location': '', 'dress': ''}
        
        character_context = ""
        if primary_character:
            character_context = f"\n\n[PRIMARY CHARACTER: {primary_character}]\n"
        
        prompt = f"""Extract scene information as JSON. Focus on location and clothing.

{character_context}
MISSION: Extract location and dress from the conversation.

OUTPUT STYLE: Danbooru/booru-style tags (short lowercase phrases). Examples: "tavern interior", "armor", "casual clothes".

Extract exactly 2 fields:
1. "location": Where is this scene? 3-5 words. Examples: "tavern interior", "dark forest", "cozy bedroom". Use generic terms, no proper names.
2. "dress": What is the character wearing? 2-3 words. Examples: "armor", "casual", "swimsuit", "dress", "jeans". Use "nude" if nudity implied. Empty string if unclear.

Reply with ONLY valid JSON like: {{"location": "tavern interior", "dress": "armor"}}

Conversation:
{conversation_text}

Assistant:"""
        
        if self.http_client is None or not self.config.get('kobold', {}).get('url'):
            return {'location': '', 'dress': ''}
        
        try:
            response = await self.http_client.post(
                f"{self.config.get('kobold', {}).get('url', '')}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_length": 50,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop_sequence": ["###", "\n\n", "Assistant:"]
                },
                timeout=30.0
            )
            
            result = response.json().get('results', [{}])[0].get('text', '').strip()
            parsed = json.loads(result)
            
            location = parsed.get('location', '').strip()
            dress = parsed.get('dress', '').strip()
            
            print(f"[SNAPSHOT CACHE] Extracted - location: '{location}', dress: '{dress}'")
            
            return {'location': location, 'dress': dress}
            
        except json.JSONDecodeError as e:
            print(f"[SNAPSHOT CACHE] JSON parse failed: {e}")
        except Exception as e:
            print(f"[SNAPSHOT CACHE] Extraction failed: {e}")
        
        return {'location': '', 'dress': ''}

    async def extract_action_only(
        self,
        messages: List[Dict],
        character_names: Optional[List[str]] = None,
        primary_character: Optional[str] = None
    ) -> str:
        """
        Extract action only from last 2 messages for fast snapshot generation.
        
        Args:
            messages: Chat messages to analyze
            character_names: Names to strip from context
            primary_character: Primary character to focus on
            
        Returns:
            Action string (e.g., "hugging another", "standing")
        """
        if not messages or len(messages) < 2:
            return ''
        
        recent_messages = messages[-2:]
        
        conversation_text = self.extract_conversation_context(
            recent_messages,
            message_count=2,
            character_names=character_names
        )
        
        if not conversation_text:
            return ''
        
        character_context = ""
        if primary_character:
            character_context = f"\n\n[PRIMARY CHARACTER: {primary_character}]\n"
        
        prompt = f"""Extract the current action as JSON.

{character_context}
MISSION: Extract what the primary character is doing RIGHT NOW.

OUTPUT STYLE: Danbooru-style tags (short lowercase phrases).

Extract exactly 1 field:
"action": What is the character doing? 2-3 words. Examples: "hugging another", "standing", "sitting", "walking", "fighting". Use generic terms, no proper names.

Reply with ONLY valid JSON like: {{"action": "hugging another"}}

Conversation:
{conversation_text}

Assistant:"""
        
        if self.http_client is None or not self.config.get('kobold', {}).get('url'):
            return ''
        
        try:
            response = await self.http_client.post(
                f"{self.config.get('kobold', {}).get('url', '')}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_length": 30,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop_sequence": ["###", "\n\n", "Assistant:"]
                },
                timeout=30.0
            )
            
            result = response.json().get('results', [{}])[0].get('text', '').strip()
            parsed = json.loads(result)
            
            action = parsed.get('action', '').strip()
            
            print(f"[SNAPSHOT ACTION] Extracted action: '{action}'")
            
            return action
            
        except json.JSONDecodeError as e:
            print(f"[SNAPSHOT ACTION] JSON parse failed: {e}")
        except Exception as e:
            print(f"[SNAPSHOT ACTION] Extraction failed: {e}")
        
        return ''
