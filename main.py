# Simple FastAPI app with NPC support
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from collections import deque
from statistics import median
import httpx
import base64
import json
import logging
import os
import random
import re
import time
import asyncio
from typing import Dict, List, Optional, Any, Literal, Tuple, Set
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

# Import the updated NPC functions with defensive lookup
from app.database import (
    # Core connection and initialization
    init_db, get_connection,
    # Character operations
    db_get_character, db_get_all_characters, db_save_character, db_delete_character,
    db_get_character_updated_at,
    # World info operations
    db_get_world, db_get_all_worlds, db_save_world, db_delete_world,
    db_update_world_entry, db_get_world_entry_timestamps,
    db_get_world_content_hash,
    # Tag operations
    db_get_character_tags, db_get_all_character_tags, db_get_popular_character_tags,
    db_get_characters_by_tags, db_add_character_tags, db_remove_character_tags,
    db_get_world_tags, db_get_all_world_tags, db_get_popular_world_tags,
    db_get_worlds_by_tags, db_add_world_tags, db_remove_world_tags,
    # Chat and message operations
    db_get_chat, db_get_all_chats, db_save_chat, db_delete_chat,
    db_get_message_context, db_cleanup_old_autosaved_chats, db_cleanup_empty_chats, db_create_empty_chat,
    db_get_chat_npcs, db_create_npc_and_update_chat,
    db_get_npc_by_id, db_update_npc, db_create_npc_with_entity_id,
    db_delete_npc, db_set_npc_active,
    # Image metadata operations
    db_save_image_metadata, db_get_image_metadata, db_get_all_image_metadata,
    # sqlite-vec embedding operations
    db_save_embedding, db_get_embedding, db_delete_entry_embedding,
    db_search_similar_embeddings, db_delete_world_embeddings, db_count_embeddings,
    # Danbooru tags operations (Snapshot feature)
    db_migrate_danbooru_tags, db_search_danbooru_embeddings, db_increment_tag_frequency,
    # Snapshot favorites operations
    db_add_snapshot_favorite, db_get_favorites, db_delete_favorite, db_get_favorite_by_image,
    db_get_all_favorite_tags, db_get_popular_favorite_tags,
    # Performance metrics
    db_save_performance_metric, db_get_recent_performance_metrics,
    db_get_median_performance, db_cleanup_old_metrics,
    # Undo/Redo functions
    get_recent_changes, get_last_change, undo_last_delete, restore_version,
    # Search functions (FTS5 full-text search)
    db_search_messages, db_get_message_context, db_get_available_speakers,
    migrate_populate_fts,
    # Health check function
    verify_database_health,
    # Autosave cleanup functions
    db_cleanup_old_autosaved_chats, db_cleanup_empty_chats,
    # Relationship tracker functions
    db_get_relationship_state, db_get_all_relationship_states, db_update_relationship_state,
    # Branch fork functions
    db_remap_entities_for_branch, db_fork_chat_transaction,
    # Change log cleanup
    db_cleanup_old_changes,
)

from app.tag_manager import parse_tag_string
from app.snapshot_analyzer import SnapshotAnalyzer
from app.snapshot_prompt_builder import SnapshotPromptBuilder
from app.danbooru_tag_generator import parse_physical_body_plist

# NPC helper functions are now integrated in load_character_profiles()


def verify_sqlite_vec_extension():
    """
    Verify that sqlite-vec extension is properly loaded and functional.
    This is critical for embeddings-based features (Danbooru search, world info, etc.)
    
    Returns True if working, exits with code 1 if not.
    """
    try:
        import sqlite3
        import sqlite_vec
        import numpy as np
        
        # Create a test in-memory database
        conn = sqlite3.connect(":memory:")
        
        # Enable and load extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        # Test vec0 functionality by creating a test table and running a distance query
        conn.execute("""
            CREATE VIRTUAL TABLE test_vec USING vec0(embedding float[3])
        """)
        
        # Insert test data
        test_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        conn.execute("INSERT INTO test_vec (rowid, embedding) VALUES (?, ?)", (1, test_embedding))
        
        # Test distance function
        cursor = conn.execute("SELECT vec_distance_cosine(embedding, ?) FROM test_vec", (test_embedding,))
        result = cursor.fetchone()
        
        # Cleanup
        conn.execute("DROP TABLE test_vec")
        conn.close()
        
        if result and result[0] is not None:
            print("[STARTUP] sqlite-vec extension verified and working")
            return True
        else:
            raise RuntimeError("vec_distance_cosine returned None")
            
    except Exception as e:
        print("="*70)
        print("FATAL ERROR: sqlite-vec extension is not working properly")
        print("="*70)
        print()
        print("The sqlite-vec extension is required for NeuralRP to function.")
        print("Without it, critical features like semantic search will not work.")
        print()
        print("Error details:", str(e))
        print()
        print("To fix this issue:")
        print("  1. Ensure sqlite-vec is installed: pip install sqlite-vec")
        print("  2. Your Python installation must support extension loading")
        print("  3. Try reinstalling: pip uninstall sqlite-vec && pip install sqlite-vec")
        print()
        print("="*70)
        import sys
        sys.exit(1)


# Verify critical extension before starting
verify_sqlite_vec_extension()


# Removed orphaned code - this block references undefined variables and is not used

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SD Presets for context-aware optimization
SD_PRESETS = {
    "normal": {
        "steps": 25,
        "width": 640,
        "height": 512,
        "threshold": 0
    },
    "light": {
        "steps": 20,
        "width": 512,
        "height": 448,
        "threshold": 10000
    },
    "emergency": {
        "steps": 15,
        "width": 384,
        "height": 384,
        "threshold": 15000
    }
}

# Resource Manager for operation queuing
class ResourceManager:
    def __init__(self):
        self.active_llm = False
        self.active_sd = False
        self.llm_queue = deque()
        self.sd_queue = deque()
        self.lock = asyncio.Lock()
        self.performance_mode_enabled = True
    
    async def execute_llm(self, operation, op_type="heavy"):
        """Execute LLM operation with queuing based on operation type"""
        if not self.performance_mode_enabled:
            return await operation()
        
        async with self.lock:
            # Light operations can interleave with heavy SD
            if op_type == "light" or not self.active_sd:
                self.active_llm = True
            else:
                # Queue heavy LLM if SD is running
                future = asyncio.Future()
                self.llm_queue.append(future)
        
        # Wait outside the lock to avoid deadlock
        if op_type == "heavy" and self.active_sd:
            await future
        
        async with self.lock:
            self.active_llm = True
        
        try:
            result = await operation()
            return result
        finally:
            async with self.lock:
                self.active_llm = False
                # Start next LLM if any queued
                if self.llm_queue and not self.active_sd:
                    next_future = self.llm_queue.popleft()
                    next_future.set_result(None)
    
    async def execute_sd(self, operation):
        """Execute SD operation with queuing"""
        if not self.performance_mode_enabled:
            return await operation()
        
        async with self.lock:
            # Wait if LLM is running (SD is always heavy)
            if self.active_llm:
                future = asyncio.Future()
                self.sd_queue.append(future)
        
        # Wait outside the lock to avoid deadlock
        if self.active_llm:
            await future
        
        async with self.lock:
            self.active_sd = True
        
        try:
            result = await operation()
            return result
        finally:
            async with self.lock:
                self.active_sd = False
                # Start next SD if any queued
                if self.sd_queue:
                    next_future = self.sd_queue.popleft()
                    next_future.set_result(None)
                # Resume LLM queue if paused
                if self.llm_queue:
                    next_future = self.llm_queue.popleft()
                    next_future.set_result(None)
    
    def get_status(self):
        """Get current resource status"""
        return {
            "llm": "running" if self.active_llm else ("queued" if self.llm_queue else "idle"),
            "sd": "running" if self.active_sd else ("queued" if self.sd_queue else "idle"),
            "llm_queue_length": len(self.llm_queue),
            "sd_queue_length": len(self.sd_queue)
        }

# Performance Tracker with rolling medians and database persistence
class PerformanceTracker:
    def __init__(self, max_samples=10):
        self.llm_times = deque(maxlen=max_samples)
        self.sd_times = deque(maxlen=max_samples)
        self.max_samples = max_samples
        self._load_from_db()
    
    def _load_from_db(self):
        """Load recent metrics from database on startup"""
        # Load LLM metrics
        llm_metrics = db_get_recent_performance_metrics("llm", self.max_samples)
        for duration in reversed(llm_metrics):  # Reverse to maintain chronological order
            self.llm_times.append(duration)
        
        # Load SD metrics
        sd_metrics = db_get_recent_performance_metrics("sd", self.max_samples)
        for duration in reversed(sd_metrics):
            self.sd_times.append(duration)
        
        if llm_metrics or sd_metrics:
            print(f"Loaded {len(llm_metrics)} LLM and {len(sd_metrics)} SD metrics from database")
    
    def record_llm(self, duration, context_tokens=0):
        """Record LLM operation duration and persist to database"""
        self.llm_times.append(duration)
        # Persist to database
        db_save_performance_metric("llm", duration, context_tokens)
    
    def record_sd(self, duration, context_tokens=0):
        """Record SD operation duration and persist to database"""
        self.sd_times.append(duration)
        # Persist to database
        db_save_performance_metric("sd", duration, context_tokens)
    
    def get_median_llm(self):
        """Get median LLM time from in-memory cache or database"""
        if self.llm_times:
            return median(list(self.llm_times))
        # Fallback to database if in-memory is empty
        return db_get_median_performance("llm", self.max_samples)
    
    def get_median_sd(self):
        """Get median SD time from in-memory cache or database"""
        if self.sd_times:
            return median(list(self.sd_times))
        # Fallback to database if in-memory is empty
        return db_get_median_performance("sd", self.max_samples)
    
    def detect_contention(self, sd_duration, context_tokens):
        """Detect if SD is experiencing contention"""
        median_sd = self.get_median_sd()
        if median_sd is None:
            return False
        # If SD took > 3x median time and context is large
        return sd_duration > median_sd * 3 and context_tokens > 8000

# Smart Hint Engine
class SmartHintEngine:
    def __init__(self):
        self.shown_hints = set()
    
    def generate_hint(self, performance_tracker, context_tokens, sd_duration=None):
        """Generate contextual performance hints"""
        hints = []
        
        # Check for SD slow-down due to large context
        if sd_duration and performance_tracker.detect_contention(sd_duration, context_tokens):
            hint_key = "sd_slow_context"
            if hint_key not in self.shown_hints:
                hints.append({
                    "id": hint_key,
                    "message": "Images are slow because the story is very long—consider a smaller model or shorter context for smoother images.",
                    "severity": "warning"
                })
                self.shown_hints.add(hint_key)
        
        # Check for very large context in general
        if context_tokens > 12000:
            hint_key = "large_context"
            if hint_key not in self.shown_hints:
                hints.append({
                    "id": hint_key,
                    "message": "Story context is very long. Consider summarizing or creating a branch to maintain performance.",
                    "severity": "info"
                })
                self.shown_hints.add(hint_key)
        
        # Check for SD degradation
        if sd_duration and performance_tracker.get_median_sd():
            median_sd = performance_tracker.get_median_sd()
            if sd_duration > median_sd * 2:
                hint_key = "sd_degradation"
                if hint_key not in self.shown_hints:
                    hints.append({
                        "id": hint_key,
                        "message": "Image generation is slower than usual. The system may be under heavy load.",
                        "severity": "info"
                    })
                    self.shown_hints.add(hint_key)
        
        return hints
    
    def dismiss_hint(self, hint_id):
        """Allow user to dismiss a hint"""
        self.shown_hints.discard(hint_id)

# Global instances
resource_manager = ResourceManager()
performance_tracker = PerformanceTracker()
hint_engine = SmartHintEngine()

# Configuration
CONFIG = {
    "kobold_url": "http://127.0.0.1:5001",
    "sd_url": "http://127.0.0.1:7861",
    "system_prompt": "Write a highly detailed, creative, and immersive response. Stay in character at all times.",
    "performance_mode_enabled": True,
    "max_context": 8192,
    "summarize_threshold": 0.80
}

# Multi-Character Mode Instruction Templates
NARRATOR_INSTRUCTION = """You are the narrator of an interactive story.

Write in third person, describing actions, thoughts, and scenes cinematically.

When characters speak, always use the format:
Name: "dialogue line"

Any character may speak when it makes sense for the scene (answering questions addressed to them, reacting emotionally, or interjecting in conversations).

Keep attribution clear so the reader always knows who is talking.

Do not write as the user. Do not break character or comment about being an AI.

Keep the reply to a reasonable length (no more than 4–6 short paragraphs of narration plus dialogue).

Describe the user's character's actions only if they explicitly wrote them; otherwise, wait for user input before moving them.

Characters only know what they've witnessed or been told in the scene (no metagaming)."""

FOCUSED_TEMPLATE = """In this reply, you are speaking only as {CharacterName}, in first person.

Write {CharacterName}'s thoughts, feelings, and spoken words.

Do not write narration outside {CharacterName}'s perspective.

Do not write lines or actions for any other characters.

Respond directly to what just happened in the previous message, in {CharacterName}'s own voice and style.

{CharacterName} only knows what they've witnessed or been told in the scene.

Keep the reply under 3–5 paragraphs."""

SCENE_CAPSULE_PROMPT = """### System: Summarize the following conversation into a scene capsule.

Requirements:
- Use neutral, factual, past-tense prose
- Include key events and plot developments
- Note relationship changes between characters
- Record any world information discovered
- Explicitly mention when characters enter or exit

Do NOT:
- Mimic character voices or speech patterns
- Include direct dialogue quotes
- Add stylistic flourishes

{departed_note}

{canon_echo}

### Conversation (Turns {start_turn}-{end_turn}):
{conversation_text}

### Scene Capsule:
Scene (Turns {start_turn}-{end_turn}):"""

def clean_llm_response(text: str) -> str:
    """Remove any reinforcement markers that might have leaked into the LLM response."""
    # Pattern to match [REINFORCEMENT: ...] and [WORLD REINFORCEMENT: ...] blocks
    # These can span multiple lines if the LLM continued generating them
    import re
    
    # Remove complete reinforcement blocks (including multi-line)
    text = re.sub(r'\[REINFORCEMENT:.*?(?:\]|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[WORLD REINFORCEMENT:.*?(?:\]|$)', '', text, flags=re.DOTALL)
    
    # Also remove partial/incomplete reinforcement starts at the end
    text = re.sub(r'\[REINFORCEMENT:.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\[WORLD REINFORCEMENT:.*$', '', text, flags=re.DOTALL)
    
    # Remove any standalone opening brackets that might be reinforcement starts
    # Only if they're at the end and look like the start of a marker
    text = re.sub(r'\n?\[(?:REINFORCEMENT|WORLD)?\s*$', '', text, flags=re.IGNORECASE)
    
    # Remove generation parameter metadata that may leak into response
    # Matches patterns like "max_length": 400, "temperature": 0.85, "stop_sequence": [...]
    text = re.sub(r'"(?:max_length|temperature|stop_sequence|top_p|top_k|repetition_penalty)".*$', '', text, flags=re.DOTALL)
    
    # Clean up any resulting extra whitespace
    text = text.strip()
    
    return text

def get_continue_hint(mode: str, last_user_message: str = "") -> str:
    """Generate a context-aware continue hint for narrator mode."""
    if mode == "narrator" or not mode.startswith("focus:"):
        if last_user_message:
            truncated = last_user_message[:100] + "..." if len(last_user_message) > 100 else last_user_message
            return f"[Continue the scene. The user just said/did: \"{truncated}\". Decide how the characters react.]"
        return "[Continue the scene from the last message. As narrator, decide which characters speak or act next based on what just happened.]"
    # Focused mode - no continue hint needed
    return ""

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app/data")
IMAGE_DIR = os.path.join(BASE_DIR, "app/images")

for d in ["characters", "worldinfo", "chats"]:
    os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Mount static files and images
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Service Status Tracking
class ServiceStatus(BaseModel):
    status: Literal["connected", "disconnected", "testing"]
    details: str = ""
    latency_ms: int = 0

# Global service status tracking
service_status = {
    "kobold": ServiceStatus(status="disconnected"),
    "sd": ServiceStatus(status="disconnected")
}

# Models
class ChatMessage(BaseModel):
    role: str
    content: str
    id: int
    speaker: Optional[str] = None
    image: Optional[str] = None

class PromptRequest(BaseModel):
    messages: List[ChatMessage]
    characters: List[Dict[str, Any]] = []
    world_info: Optional[Dict[str, Any]] = None
    settings: Dict[str, Any] = {}
    summary: Optional[str] = ""
    mode: Optional[str] = "narrator"  # "narrator" or "focus:{CharacterName}"
    metadata: Optional[Dict[str, Any]] = None
    chat_id: Optional[str] = None  # For autosave tracking

class SDParams(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    sampler_name: str = "Euler a"
    scheduler: str = "Automatic"
    context_tokens: Optional[int] = 0  # Context length for SD optimization
    chat_id: Optional[str] = None  # For loading NPCs in manual generation

class InpaintRequest(BaseModel):
    image: str  # Base64 encoded image
    mask: str   # Base64 encoded mask
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    denoising_strength: float = 0.75
    cfg_scale: float = 8.0
    steps: int = 20
    sampler_name: str = "DPM++ 2M"
    mask_blur: int = 4

class CardGenRequest(BaseModel):
    char_name: Optional[str] = "" 
    field_type: str
    context: str
    source_mode: Optional[str] = "chat" # "chat" or "manual"
    chat_id: Optional[str] = None  # Chat ID for NPC creation
    save_as: Optional[str] = "global" # "global" (default) or "local_npc"
    gender: Optional[str] = ""

async def generate_dialogue_for_edit(
    char_name: str,
    personality: str = "",
    description: str = "",
    scenario: str = "",
    current_example: str = ""
) -> str:
    """Generate example dialogue using existing prompt format from NPC generation.

    Reuses DIALOGUE_EXAMPLES format that generates 2-3 dialogue exchanges
    with <START> markers, {{char}}/{{user}} placeholders, and *actions*/"speech".

    Args:
        char_name: Character name for placeholder
        personality: Personality traits for voice reference
        description: Physical description and appearance for context
        scenario: Character's scenario for context
        current_example: Existing dialogue text (if any) to use as style reference

    Returns:
        Generated dialogue exchanges in DIALOGUE_EXAMPLES format
    """
    system = "You are a creative character designer. Generate detailed, engaging character information in a structured format."

    prompt = f"""Create example dialogue for this character:

Name: {char_name}
"""
    if personality:
        prompt += f"Personality: {personality}\n"
    if description:
        prompt += f"Description: {description}\n"
    if scenario:
        prompt += f"Scenario: {scenario}\n"
    if current_example:
        prompt += f"\nCurrent Example (use as reference for style):\n{current_example}\n"

    prompt += """
Generate 2-3 dialogue exchanges in this exact format:

DIALOGUE_EXAMPLES:
<START>
{{user}}: [ask a relevant question]
{{char}}: *[action]* "[response in character]"

<START>
{{user}}: [ask another relevant question]
{{char}}: *[action]* "[response in character]"

Output ONLY DIALOGUE_EXAMPLES section, nothing else."""

    result = await call_llm_helper(system, prompt, 600)
    return result.strip()

async def generate_personality_for_edit(
    char_name: str,
    current_personality: str,
    description: str = "",
    scenario: str = ""
) -> str:
    """Generate or expand personality field using PList format.

    Args:
        char_name: Character name for PList label
        current_personality: Existing personality traits (if any)
        description: Physical description and appearance for context
        scenario: Character scenario for context

    Returns:
        PList-formatted personality string
    """
    system = "You are an expert at analyzing characters and writing roleplay personality traits."

    prompt = f"""Generate personality traits for {char_name}:
"""
    if current_personality:
        prompt += f"Current Personality: {current_personality}\n"
    if description:
        prompt += f"Description: {description}\n"
    if scenario:
        prompt += f"Scenario: {scenario}\n"

    prompt += """
Generate or expand personality traits in PList format:
[{char_name}'s Personality= "trait1", "trait2", "trait3", ...]

Include 4-6 traits. If current personality is provided, expand with additional compatible traits.
Output ONLY personality line, nothing else."""

    result = await call_llm_helper(system, prompt, 300)
    return result.strip()

async def generate_scenario_for_edit(
    char_name: str,
    current_scenario: str,
    personality: str = "",
    description: str = ""
) -> str:
    """Generate or expand scenario field.

    Args:
        char_name: Character name
        current_scenario: Existing scenario (if any)
        personality: Personality traits for consistency
        description: Physical description and appearance for context

    Returns:
        Single sentence scenario string
    """
    system = "You are an expert at writing roleplay scenarios."

    prompt = f"""Generate a scenario for {char_name}:
"""
    if current_scenario:
        prompt += f"Current Scenario: {current_scenario}\n"
    if personality:
        prompt += f"Personality: {personality}\n"
    if description:
        prompt += f"Description: {description}\n"

    prompt += """
Generate or expand scenario as a single sentence describing the situation.
Make it engaging and relevant to character's traits.
If current scenario is provided, improve or expand it.
Output ONLY the scenario sentence, nothing else."""

    result = await call_llm_helper(system, prompt, 200)
    return result.strip()

async def generate_first_message_for_edit(
    char_name: str,
    current_first_msg: str,
    personality: str = "",
    scenario: str = "",
    description: str = ""
) -> str:
    """Generate or expand first message.

    Args:
        char_name: Character name for placeholder
        current_first_msg: Existing first message (if any)
        personality: Personality traits for voice
        scenario: Scenario context
        description: Physical description and appearance for context

    Returns:
        First message string with *actions* and "speech"
    """
    system = "You are an expert at writing engaging roleplay opening messages."

    prompt = f"""Generate a first message for {char_name}:
"""
    if current_first_msg:
        prompt += f"Current First Message: {current_first_msg}\n"
    if personality:
        prompt += f"Personality: {personality}\n"
    if scenario:
        prompt += f"Scenario: {scenario}\n"
    if description:
        prompt += f"Description: {description}\n"

    prompt += """
Generate or expand first message as 2-3 sentences that introduce the character and establish the scenario.
Include actions in asterisks *like this* and dialogue in quotes.
If current first message is provided, improve or expand it.
Output ONLY the first message, nothing else."""

    result = await call_llm_helper(system, prompt, 300)
    return result.strip()

async def generate_body_for_edit(
    char_name: str,
    current_body: str,
    personality: str = "",
    description: str = "",
    scenario: str = ""
) -> str:
    """Generate or expand physical description using PList format.

    Args:
        char_name: Character name for PList label
        current_body: Existing physical description (if any)
        personality: Personality traits for consistency
        description: Physical description and appearance for context
        scenario: Scenario context

    Returns:
        PList-formatted physical description string
    """
    system = "You are an expert at analyzing characters and writing physical descriptions."

    prompt = f"""Generate physical description for {char_name}:
"""
    if current_body:
        prompt += f"Current Body: {current_body}\n"
    if personality:
        prompt += f"Personality: {personality}\n"
    if description:
        prompt += f"Description: {description}\n"
    if scenario:
        prompt += f"Scenario: {scenario}\n"

    prompt += """
Generate or expand physical body description in PList format:
[{char_name}'s body= "feature1", "feature2", "feature3", ...]

Include appearance, clothing, distinctive features. If current body is provided, expand with additional details.
Output ONLY the body line, nothing else."""

    result = await call_llm_helper(system, prompt, 300)
    return result.strip()

class WorldGenRequest(BaseModel):
    world_name: str
    section: str # history, locations, creatures, factions
    tone: str # neutral (simplified - only neutral option now)
    context: str
    source_mode: Optional[str] = "chat" # "chat" or "manual"

class WorldSaveRequest(BaseModel):
    world_name: str
    plist_text: str

class WorldAddEntryRequest(BaseModel):
    world_name: str
    entry_data: dict
    tags: Optional[List[str]] = None

# Editing Models
class CharacterEditRequest(BaseModel):
    filename: str
    field: str  # personality, body, dialogue, genre, tags, scenario, first_message, mes_example
    new_value: str
    context: Optional[str] = ""  # Optional context for AI-assisted editing

class CharacterEditFieldRequest(BaseModel):
    filename: str
    field: str
    context: Optional[str] = ""  # Context for AI generation
    source_mode: Optional[str] = "manual"  # "chat" or "manual"

class WorldEditRequest(BaseModel):
    world_name: str
    entry_uid: str
    field: str  # content, key, is_canon_law, probability
    new_value: Any

class WorldEditEntryRequest(BaseModel):
    world_name: str
    entry_uid: Optional[str]
    section: str  # history, locations, creatures, factions
    tone: str  
    context: Optional[str] = ""
    source_mode: Optional[str] = "manual"

# Character name helper for consistency with relationship tracking
def get_character_name(character_obj: Any) -> str:
    """
    Extract character name consistently from any character reference.

    CRITICAL: This is the ONLY approved way to get character names.
    DO NOT extract first name only or modify the name string.
    This ensures relationship tracker and other features work correctly.

    Args:
        character_obj: Can be:
            - Dict with 'data.name' (from db_get_character)
            - Dict with 'name' (from character card)
            - String (already a name)

    Returns:
        Full character name, never empty
    """
    if isinstance(character_obj, str):
        # Already a name string
        return character_obj.strip() or "Unknown"

    if isinstance(character_obj, dict):
        # Standard format: {'data': {'name': '...'}}
        if 'data' in character_obj:
            name = character_obj['data'].get('name', 'Unknown')
            if name and isinstance(name, str):
                return name.strip()

        # Fallback: {'name': '...'}
        name = character_obj.get('name', 'Unknown')
        if name and isinstance(name, str):
            return name.strip()

    return "Unknown"


def get_entity_id(character_obj: Any) -> str:
    """
    Extract entity ID consistently from any character reference.
    Used for relationship tracking to ensure proper entity identification.

    Args:
        character_obj: Can be:
            - Dict with '_filename' (global character)
            - Dict with 'entity_id' (local NPC)
            - String (already an entity ID)

    Returns:
        Entity ID string or 'Unknown' if not found
    """
    if isinstance(character_obj, str):
        # Already an entity ID string
        return character_obj.strip() or "Unknown"

    if isinstance(character_obj, dict):
        # Global character: use _filename (e.g., 'alice.json')
        if '_filename' in character_obj:
            return character_obj['_filename']

        # Local NPC: use entity_id (e.g., 'npc_123456_789')
        if 'entity_id' in character_obj:
            return character_obj['entity_id']

        # Fallback: try to derive from name
        return get_character_name(character_obj)

    return "Unknown"


def build_entity_name_mapping(
    characters: List[Dict],
    npcs: Dict
    ) -> Dict[str, str]:
    """
    Build mapping from entity_id to display name.
    
    Used by build_scene_update_block() to convert entity IDs to readable names.
    
    Returns:
        {entity_id: display_name, ...}
    """
    from urllib.parse import unquote
    
    mapping = {}
    
    # Global characters
    for char in characters:
        entity_id = char.get('_filename')
        name = get_character_name(char)
        if entity_id and name:
            mapping[entity_id] = name
            mapping[unquote(entity_id)] = name
    
    # Local NPCs
    for npc_id, npc in npcs.items():
        entity_id = npc.get('entity_id') or npc_id
        name = npc.get('name', 'Unknown NPC')
        if entity_id:
            mapping[entity_id] = name
            mapping[unquote(entity_id)] = name
    
    return mapping


def detect_cast_change(
    current_characters: List[Dict],
    current_npcs: Dict,
    mode: str,
    previous_metadata: Dict
) -> Tuple[bool, Set[str], Set[str], Dict]:
    """
    Detect if cast changed since last turn.
    
    Cast change definition:
        ALL modes (NARRATOR and FOCUS): active_set changed only
        
    Focus changes within same cast do NOT trigger cast change.
    
    Returns:
        (changed: bool, departed: Set[str], arrived: Set[str], updated_metadata: Dict)
    """
    from urllib.parse import unquote
    
    # Build current active set (entity IDs)
    current_set = set()
    for char in current_characters:
        if char.get('is_active', True):
            entity_id = char.get('_filename')
            if entity_id:
                current_set.add(entity_id)
    for npc_id, npc in current_npcs.items():
        if npc.get('is_active', True):
            entity_id = unquote(npc.get('entity_id') or npc_id)
            if entity_id:
                current_set.add(entity_id)
    
    # Get previous state from provided metadata
    previous_set = set(unquote(id) for id in previous_metadata.get('previous_active_cast', []))
    previous_focus = previous_metadata.get('previous_focus_character')
    
    # Determine current focus (for tracking, NOT for cast change detection)
    current_focus = None
    if mode.startswith('focus:'):
        focus_name = mode.split(':', 1)[1]
        
        # Search global characters
        for char in current_characters:
            if get_character_name(char) == focus_name:
                current_focus = char.get('_filename')
                break
        
        # Search NPCs if not found
        if not current_focus:
            for npc_id, npc in current_npcs.items():
                if npc.get('name') == focus_name:
                    current_focus = npc.get('entity_id') or npc_id
                    break
    
    # Cast change = active set changed (focus change alone doesn't count)
    cast_changed = (current_set != previous_set)
    
    # Calculate departed and arrived
    departed = previous_set - current_set
    arrived = current_set - previous_set
    
    # Build updated metadata dict (don't save here - pure function)
    updated_metadata = {
        'previous_active_cast': list(current_set),
        'previous_focus_character': current_focus  # Track for future use
    }
    
    return (cast_changed, departed, arrived, updated_metadata)


def build_scene_update_block(
    departed: Set[str],
    arrived: Set[str],
    entity_to_name: Dict[str, str]
) -> str:
    """
    Build SCENE UPDATE block when cast changes.
    Only emitted on turn where change is detected.
    """
    if not departed and not arrived:
        return ""
    
    lines = ["SCENE UPDATE"]
    
    for entity_id in departed:
        name = entity_to_name.get(entity_id, entity_id)
        lines.append(f"- {name} has left the scene.")
    
    for entity_id in arrived:
        name = entity_to_name.get(entity_id, entity_id)
        lines.append(f"- {name} has entered the scene.")
    
    lines.append("- Adjust your portrayal to reflect only the currently active characters listed in SCENE CAST below.")
    
    return "\n".join(lines)

# Token counting helper
async def get_token_count(text: str):
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{CONFIG['kobold_url']}/api/extra/tokencount",
                json={"prompt": text},
                timeout=10.0
            )
            return res.json().get("value", 0)
        except:
            # Fallback to rough estimate if API fails
            return len(text) // 4

# World info optimization: LRU caching to avoid reprocessing
from collections import OrderedDict

# Semantic Search Engine
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import threading
import gc
import torch

class LRUCache:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()  # Use RLock for reentrant operations
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                # Remove existing to update position
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def size(self):
        with self.lock:
            return len(self.cache)
    
    def get_stats(self):
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "usage_percent": (len(self.cache) / self.max_size) * 100 if self.max_size > 0 else 0
            }
    
    def __contains__(self, key):
        """Enable 'in' operator for LRUCache"""
        with self.lock:
            return key in self.cache
    
    def __setitem__(self, key, value):
        """Enable item assignment for LRUCache"""
        self.put(key, value)
    
    def __getitem__(self, key):
        """Enable item access for LRUCache"""
        return self.get(key)

# Optimized cache size - reduced from 1000 to 300 since query repetition is less common than expected
# Cache stores processed search results (not embeddings which are in sqlite-vec)
WORLD_INFO_CACHE = LRUCache(max_size=300)

# Recent Edits Tracking System (v1.7.3)
# In-memory tracking for one-time flash notifications
# Completely separate from reinforcement intervals - appears once then disappears


# Semantic Search Engine
def get_npc_timestamp(entity_id: str, chat_id: str) -> int:
    """Get NPC's last update timestamp from chat metadata."""
    chat = db_get_chat(chat_id)
    if chat and "metadata" in chat:
        npcs = chat["metadata"].get("localnpcs", {})
        npc = npcs.get(entity_id, {})
        return npc.get("updated_at", 0)
    return 0


class SemanticSearchEngine:
    GENERIC_KEYS = {
        # structural/generic
        "program", "system", "policy", "event", "room", "rooms", "implementation", "annual",
        "subgroup", "collection", "type", "has", "used", "for", "the", "and", "with",
        # location/role buckets
        "city", "town", "village", "room", "rooms", "hub", "center", "holding",
        # time/temporal buckets
        "time", "era", "age",
        # creature/class buckets
        "creature", "race", "species", "archetype",
        # faction buckets
        "faction", "guild", "house", "clique", "resistance", "ruling",
    }
    
    @staticmethod
    def parse_keys_with_types(keys):
        """
        Parse keys and categorize as phrase (quoted) or word (unquoted).
        
        Returns:
            dict with 'quoted' and 'unquoted' lists, and 'is_quoted' flags per key
        """
        quoted = []
        unquoted = []
        is_quoted_flags = []
        
        for key in keys:
            stripped = key.strip()
            
            # Check for matching quotes (single or double)
            if (stripped.startswith('"') and stripped.endswith('"')) or \
               (stripped.startswith("'") and stripped.endswith("'")):
                # Remove quotes and add to quoted
                inner = stripped[1:-1]
                if inner:  # Skip empty quotes
                    quoted.append(inner)
                    is_quoted_flags.append(True)
            else:
                # Unquoted key
                if stripped:  # Skip empty strings
                    unquoted.append(stripped)
                    is_quoted_flags.append(False)
        
        # Return categorized keys and flags
        return {
            'quoted': quoted,
            'unquoted': unquoted,
            'is_quoted_flags': is_quoted_flags,
            'all_normalized': [k.strip().lower() for k in keys if k.strip()]
        }
 
    def __init__(self):
        self.model = None
        self.embeddings_cache = {}
        self.world_info_cache = {}
        self.lock = threading.RLock()  # Use RLock for reentrant operations
        self.is_loading = False
        self.model_name = "all-mpnet-base-v2"
        self.device = None
        self.gpu_memory_limit = None
        self._initialized = False
        self._last_cleanup_time = 0
        self._cleanup_interval = 300  # Clean up every 5 minutes
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        self.unload_model()
    
    def __enter__(self):
        """Context manager entry - ensure model is loaded"""
        if not self.load_model():
            raise RuntimeError("Failed to load semantic search model")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - optional cleanup"""
        pass
    
    def cleanup_resources(self):
        """Periodic cleanup of resources to prevent memory leaks"""
        current_time = time.time()
        if current_time - self._last_cleanup_time < self._cleanup_interval:
            return
        
        with self.lock:
            # Clean up embeddings cache if it's getting too large
            if len(self.embeddings_cache) > 5:  # Keep only 5 most recent
                # Sort by insertion order and keep only the 5 most recent
                items = list(self.embeddings_cache.items())
                self.embeddings_cache = dict(items[-5:])
            
            # Force garbage collection
            gc.collect()
            
            # Clean up GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._last_cleanup_time = current_time
    
    def _detect_device(self):
        """Detect and configure optimal device for model"""
        if self.device is not None:
            return self.device
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            free_memory = gpu_memory - reserved_memory
            
            # Reserve some memory for other operations (20%)
            available_memory = free_memory * 0.8
            
            # Model typically needs ~400MB for all-mpnet-base-v2
            if available_memory > 500 * 1024 * 1024:  # 500MB threshold
                self.device = "cuda"
                self.gpu_memory_limit = available_memory
                print(f"Using GPU with {available_memory / 1024 / 1024:.1f}MB available memory")
            else:
                self.device = "cpu"
                print(f"GPU memory insufficient ({available_memory / 1024 / 1024:.1f}MB), using CPU")
        else:
            self.device = "cpu"
            print("CUDA not available, using CPU")
        
        return self.device
    
    def unload_model(self):
        """Explicitly unload model and free memory"""
        with self.lock:
            if self.model is not None:
                print("Unloading semantic search model...")
                del self.model
                self.model = None
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Model unloaded and memory freed")
    
    def load_model(self):
        """Load the sentence transformer model with proper race condition handling and device detection"""
        with self.lock:
            if self.model is not None:
                return True
            
            # Use atomic operation to prevent race conditions
            if self.is_loading:
                return False
            
            self.is_loading = True
        
        try:
            print("Loading semantic search model...")
            
            # Detect optimal device
            device = self._detect_device()
            
            # Load model with explicit device assignment (local only to avoid network calls)
            self.model = SentenceTransformer(self.model_name, device=device, local_files_only=True)
            
            # Verify model is on correct device
            if hasattr(self.model, 'device'):
                print(f"Model loaded on {self.model.device}")
            
            print("Semantic search model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load semantic search model: {e}")
            with self.lock:
                self.is_loading = False
            return False
        finally:
            with self.lock:
                self.is_loading = False
    
    def get_world_info_hash(self, world_info):
        """Generate a hash for the world info to detect changes"""
        if not world_info or "entries" not in world_info:
            return "empty"
        
        # Create a hash of content + keys to detect changes (keys drive embeddings)
        content = ""
        for entry in world_info.get("entries", {}).values():
            content += entry.get("content", "") + "|"
            content += ",".join(entry.get("key", [])) + "|"
            content += ",".join(entry.get("keysecondary", [])) + "|"
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_entry_embeddings(self, world_info):
        """Get or compute embeddings for all world info entries using sqlite-vec for persistence"""
        if not world_info or "entries" not in world_info:
            return {}
        
        world_name = world_info.get("name", "")
        if not world_name:
            # Fallback to hash-based identification for unnamed world info
            world_name = self.get_world_info_hash(world_info)
        
        # Load model if not loaded
        if not self.load_model():
            return {}
        
        # Try to load embeddings from sqlite-vec first
        embeddings_from_db = {}
        missing_uids = []
        
        for uid, entry in world_info["entries"].items():
            # Try to get embedding from sqlite-vec
            embedding = db_get_embedding(world_name, uid)
            if embedding is not None:
                embeddings_from_db[uid] = embedding
            else:
                missing_uids.append(uid)
        
        # If all embeddings are in database, return them
        if not missing_uids:
            print(f"Loaded {len(embeddings_from_db)} embeddings from sqlite-vec for world '{world_name}'")
            return embeddings_from_db
        
        # Compute missing embeddings
        print(f"Computing {len(missing_uids)} new embeddings for world '{world_name}'...")
        
        contents = []
        uids_to_compute = []
        
        for uid in missing_uids:
            entry = world_info["entries"][uid]
            all_keys = entry.get("key", [])
            
            # Parse keys into quoted and unquoted categories
            parsed = SemanticSearchEngine.parse_keys_with_types(all_keys)
            
            # Only embed unquoted keys (semantic search)
            primary_list = [k for k in parsed['unquoted'] 
                           if k.lower() not in self.GENERIC_KEYS]
            keys = ", ".join(primary_list)
            
            if keys:
                prefixed_content = f"Keywords: {keys}"
                contents.append(prefixed_content)
                uids_to_compute.append(uid)
            
            # Store key type metadata for later matching
            if 'entries' not in world_info.get('_metadata', {}):
                world_info.setdefault('_metadata', {})['entries'] = {}
            world_info['_metadata']['entries'][uid] = parsed
        
        if not contents:
            return embeddings_from_db
        
        # Compute embeddings
        try:
            embeddings = self.model.encode(contents, convert_to_numpy=True, show_progress_bar=False)
            
            # Save new embeddings to sqlite-vec and add to result
            for i, uid in enumerate(uids_to_compute):
                embedding = embeddings[i]
                # Save to sqlite-vec for persistence
                db_save_embedding(world_name, uid, embedding)
                embeddings_from_db[uid] = embedding
            
            print(f"Computed and saved {len(uids_to_compute)} new embeddings to sqlite-vec")
            return embeddings_from_db
            
        except Exception as e:
            print(f"Failed to compute embeddings: {e}")
            return embeddings_from_db
    
    def search_semantic(self, world_info, query_text, max_entries=10, similarity_threshold=0.3, is_initial_turn=False):
        """Perform semantic search on world info entries using sqlite-vec SIMD-accelerated search"""
        if not world_info or "entries" not in world_info:
            return [], []

        world_name = world_info.get("name", "")
        if not world_name:
            world_name = self.get_world_info_hash(world_info)

        # Load model if not loaded
        if not self.load_model():
            return [], []

        # Compute query embedding
        try:
            query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]

            # Adjust threshold for initial turns
            effective_threshold = similarity_threshold
            if is_initial_turn:
                effective_threshold = max(0.35, similarity_threshold)  # Higher threshold for initial turns

            # Try sqlite-vec SIMD-accelerated search first
            try:
                # Get more results than needed to allow for filtering
                search_results = db_search_similar_embeddings(
                    world_name, 
                    query_embedding, 
                    k=max_entries * 3 if max_entries > 0 else 100,
                    threshold=effective_threshold
                )
                
                if search_results:
                    # Process results from sqlite-vec
                    triggered_lore = []
                    canon_entries = []
                    
                    for uid, similarity in search_results:
                        if uid not in world_info["entries"]:
                            continue
                        
                        entry = world_info["entries"][uid]
                        
                        # Always include canon law entries
                        if entry.get("is_canon_law"):
                            canon_entries.append(entry.get("content", ""))
                            continue
                        
                        # Add to results with similarity score
                        triggered_lore.append((entry.get("content", ""), similarity, uid))
                        
                        # Stop if we've reached max_entries
                        if max_entries > 0 and len(triggered_lore) >= max_entries:
                            break
                    
                    print(f"sqlite-vec search: {len(triggered_lore)} semantic matches (threshold: {effective_threshold})")
                    return triggered_lore, canon_entries
            
            except Exception as vec_error:
                print(f"sqlite-vec search failed, falling back to numpy: {vec_error}")
            
            # Fallback to numpy-based search if sqlite-vec fails
            embeddings = self.get_entry_embeddings(world_info)
            if not embeddings:
                return [], []

            # Calculate similarities using numpy
            similarities = []
            for uid, entry_embedding in embeddings.items():
                # Compute cosine similarity: dot(a,b) / (norm(a) * norm(b))
                similarity = np.dot(query_embedding, entry_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding)
                )
                similarities.append((uid, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Filter by threshold and return entries WITH similarity scores
            triggered_lore = []  # List of (content, similarity, uid) tuples
            canon_entries = []

            for uid, similarity in similarities:
                if similarity < effective_threshold:
                    continue

                entry = world_info["entries"][uid]

                # Always include canon law entries
                if entry.get("is_canon_law"):
                    canon_entries.append(entry.get("content", ""))
                    continue

                # Add to results with similarity score
                triggered_lore.append((entry.get("content", ""), similarity, uid))

                # Stop if we've reached max_entries
                if max_entries > 0 and len(triggered_lore) >= max_entries:
                    break

            print(f"numpy fallback search: {len(triggered_lore)} semantic matches")
            return triggered_lore, canon_entries

        except Exception as e:
            print(f"Semantic search failed: {e}")
            return [], []

# Global semantic search engine instance
semantic_search_engine = SemanticSearchEngine()

# ============================================================================
# SNAPSHOT FEATURE (v1.9.0)
# ============================================================================

snapshot_analyzer = None
prompt_builder = None
snapshot_http_client = None

# ============================================================================
# ADAPTIVE RELATIONSHIP TRACKING ENGINE (v1.6.1)
# ============================================================================

# Import adaptive tracker
from app.relationship_tracker import (
    AdaptiveRelationshipTracker,
    initialize_adaptive_tracker,
    adaptive_tracker
)

# Legacy RelationshipAnalyzer kept for reference, but not used
# The AdaptiveRelationshipTracker in app/relationship_tracker.py is the new implementation
relationship_analyzer = None  # Deprecated, use adaptive_tracker instead


# ============================================================================
# RELATIONSHIP TEMPLATES
# ============================================================================

# Template definitions for natural language injection
RELATIONSHIP_TEMPLATES = {
    'trust': {
        (0, 20): ["{from_} deeply distrusts {to}", "{from_} views {to} with complete suspicion"],
        (21, 40): ["{from_} is wary of {to}", "{from_} has doubts about {to}'s intentions"],
        (41, 60): [],  # Neutral - don't inject
        (61, 80): ["{from_} trusts {to}", "{from_} has faith in {to}"],
        (81, 100): ["{from_} trusts {to} completely", "{from_} would trust {to} with their life"]
    },
    'emotional_bond': {
        (0, 20): ["{from_} is repulsed by {to}", "{from_} actively dislikes {to}"],
        (21, 40): ["{from_} is indifferent to {to}", "{from_} feels little connection to {to}"],
        (41, 60): [],  # Neutral - don't inject
        (61, 80): ["{from_} cares deeply for {to}", "{from_} has strong feelings for {to}"],
        (81, 100): ["{from_} is deeply in love with {to}", "{from_} adores {to}"]
    },
    'conflict': {
        (0, 20): [],  # Low conflict - don't inject
        (21, 40): ["{from_} has minor disagreements with {to}", "{from_} occasionally argues with {to}"],
        (41, 60): ["{from_} has noticeable tension with {to}", "{from_} is in active conflict with {to}"],
        (61, 80): ["{from_} is in active conflict with {to}", "{from_} openly opposes {to}"],
        (81, 100): ["{from_} views {to} as an enemy", "intense hostility exists between {from_} and {to}"]
    },
    'power_dynamic': {
        (0, 20): ["{from_} is completely submissive to {to}", "{from_} defers to {to} in all things"],
        (21, 40): ["{from_} often defers to {to}", "{from_} follows {to}'s lead"],
        (41, 60): [],  # Equal - don't inject
        (61, 80): ["{from_} often leads {to}", "{to} tends to take charge in situations"],
        (81, 100): ["{from_} completely dominates {to}", "{from_} commands {to} without question"]
    },
    'fear_anxiety': {
        (0, 20): [],  # No fear - don't inject
        (21, 40): ["{from_} is slightly nervous around {to}", "{from_} feels a bit uneasy near {to}"],
        (41, 60): ["{from_} is noticeably anxious around {to}", "{from_} shows signs of nervousness with {to}"],
        (61, 80): ["{from_} fears {to}", "{from_} is intimidated by {to}"],
        (81, 100): ["{from_} is terrified of {to}", "{from_} is completely paralyzed by fear around {to}"]
    }
}


def get_relationship_context(chat_id: str, character_objs: list, user_name: str,
                            recent_messages: list) -> str:
    """
    Generate compact relationship context for prompt injection.
    Uses Tier 3 semantic filtering to only include dimensions relevant to current conversation.
    Returns 0-75 tokens depending on active interactions.

    CRITICAL: character_objs should be full character objects (not just names).
    This function extracts entity IDs and names for proper relationship tracking.
    """
    import random

    # Extract entity IDs and names from character objects
    entity_map = {}  # entity_id -> name
    name_map = {}    # name -> entity_id
    for char_obj in character_objs:
        entity_id = get_entity_id(char_obj)
        name = get_character_name(char_obj)
        entity_map[entity_id] = name
        name_map[name] = entity_id

    # Identify active speakers in last 5 messages
    # Fix: Access Pydantic model attributes directly (m.speaker, m.role)
    active_speakers = set(m.speaker or m.role for m in recent_messages[-5:])

    # Find active characters (by entity ID or name)
    active_characters = []
    for entity_id, name in entity_map.items():
        if entity_id in active_speakers or name in active_speakers:
            active_characters.append((entity_id, name))

    user_is_active = any(
        m.role == 'user' or m.speaker == user_name
        for m in recent_messages[-5:]
    )

    # Need at least 2 entities (char+char or char+user)
    if not ((len(active_characters) >= 1 and user_is_active) or len(active_characters) >= 2):
        return ""

    # Get current turn text for semantic relevance filtering
    # Use the most recent message's content
    current_text = ""
    if recent_messages:
        current_text = recent_messages[-1].content or ""

    lines = []

    # Build relationship states dictionary for all active entity pairs
    # Format: {from_entity: {dimension: score}}
    relationship_states = {}

    # Character-to-character relationships
    for from_id, from_name in active_characters:
        for to_id, to_name in active_characters:
            if from_id == to_id:
                continue

            state = db_get_relationship_state(chat_id, from_id, to_id)
            if state:
                entity_key = f"{from_name}→{to_name}"
                relationship_states[entity_key] = {
                    'trust': state['trust'],
                    'emotional_bond': state['emotional_bond'],
                    'conflict': state['conflict'],
                    'power_dynamic': state['power_dynamic'],
                    'fear_anxiety': state['fear_anxiety']
                }

    # Character-to-user relationships
    if user_is_active and active_characters:
        user_name_or_default = user_name or "User"
        for from_id, from_name in active_characters:
            state = db_get_relationship_state(chat_id, from_id, user_name_or_default)
            if state:
                entity_key = f"{from_name}→{user_name_or_default}"
                relationship_states[entity_key] = {
                    'trust': state['trust'],
                    'emotional_bond': state['emotional_bond'],
                    'conflict': state['conflict'],
                    'power_dynamic': state['power_dynamic'],
                    'fear_anxiety': state['fear_anxiety']
                }
    
    # Use adaptive tracker's Tier 3 semantic filtering if available
    if adaptive_tracker and relationship_states:
        try:
            relevant_dimensions = adaptive_tracker.get_relevant_dimensions(
                current_text=current_text,
                relationship_states=relationship_states
            )
            
            # Generate templates only for relevant dimensions
            for entity_key, relevant_dims in relevant_dimensions.items():
                # Parse entity_key (format: "From→To")
                parts = entity_key.split('→')
                if len(parts) != 2:
                    continue
                char_from, char_to = parts[0], parts[1]
                
                # Get full state for score lookup
                state = relationship_states.get(entity_key)
                if not state:
                    continue
                
                # Generate templates for each relevant dimension
                lines_before = len(lines)
                for dim in relevant_dims:
                    score = state[dim]
                    # Clamp score to valid range to prevent lookup failures
                    score = max(0, min(100, score))
                    for (low, high), templates in RELATIONSHIP_TEMPLATES[dim].items():
                        if low <= score <= high:
                            if templates:
                                template = random.choice(templates)
                                lines.append(template.format(from_=char_from, to=char_to))
                            break  # Found matching range, stop searching
                
                # Only add period if this iteration added content
                if len(lines) > lines_before:
                    lines.append(".")
            
            if lines:
                return "### Relationship Context:\n" + "\n".join(lines)
            
            # If adaptive filtering returned no relevant dimensions, return empty
            return ""
        
        except Exception as e:
            print(f"[ADAPTIVE_TRACKER] Tier 3 filtering failed, falling back: {e}")
            # Fall through to legacy behavior
    
    # Fallback: Legacy behavior without semantic filtering
    # For each active character, describe notable feelings
    # Note: active_characters contains tuples of (entity_id, name)
    for from_id, from_name in active_characters:
        # Check toward other characters
        for to_id, to_name in active_characters:
            if from_id == to_id:
                continue
            
            # Get relationship state using entity IDs
            state = db_get_relationship_state(chat_id, from_id, to_id)
            if not state:
                continue
            
            # Find dimensions far from neutral (50 ± 15)
            notable = []
            dimensions = ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']
            
            for dim in dimensions:
                score = state[dim]
                # Clamp score to valid range to prevent lookup failures
                score = max(0, min(100, score))
                if abs(score - 50) > 15:
                    notable.append((dim, score))
            
            # Sort by extremity, take top 2
            notable.sort(key=lambda x: abs(x[1] - 50), reverse=True)
            top_two = notable[:2]
            
            # Generate templates for top two dimensions
            lines_before = len(lines)
            for dim, score in top_two:
                for (low, high), templates in RELATIONSHIP_TEMPLATES[dim].items():
                    if low <= score <= high:
                        if templates:
                            template = random.choice(templates)
                            lines.append(template.format(from_=from_name, to=to_name))
                        break  # Found matching range, stop searching
            
            # Only add period if this iteration added content
            if len(lines) > lines_before:
                lines.append(".")
    
    # Also check toward user if user is active (fallback only)
    if user_is_active and active_characters:
        user_name_or_default = user_name or "User"
        
        for from_id, from_name in active_characters:
            # Get relationship state using entity ID
            state = db_get_relationship_state(chat_id, from_id, user_name_or_default)
            if not state:
                continue
            
            # Find dimensions far from neutral
            notable = []
            dimensions = ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']
            
            for dim in dimensions:
                score = state[dim]
                # Clamp score to valid range to prevent lookup failures
                score = max(0, min(100, score))
                if abs(score - 50) > 15:
                    notable.append((dim, score))
            
            # Sort by extremity, take top 2
            notable.sort(key=lambda x: abs(x[1] - 50), reverse=True)
            top_two = notable[:2]
            
            # Generate templates for top two dimensions
            lines_before = len(lines)
            for dim, score in top_two:
                for (low, high), templates in RELATIONSHIP_TEMPLATES[dim].items():
                    if low <= score <= high:
                        if templates:
                            template = random.choice(templates)
                            lines.append(template.format(from_=from_name, to=user_name_or_default))
                        break  # Found matching range, stop searching
            
            # Only add period if this iteration added content
            if len(lines) > lines_before:
                lines.append(".")
    
    if lines:
        return "### Relationship Context:\n" + "\n".join(lines)
    
    return ""


# Global variable to store cleanup task
cleanup_task = None


async def analyze_and_update_relationships(chat_id: str, messages: list,
                                      character_objs: list, user_name: str = None):
    """
    Analyze relationships using semantic embeddings.
    Triggered at summarization boundary (every 10 messages).

    CRITICAL: character_objs should be full character objects (not just names).
    This function extracts entity IDs and names for proper relationship tracking.
    
    Note: Semantic scoring can be disabled by setting environment variable:
    NEURALRP_DISABLE_SEMANTIC_SCORING=1
    """
    import os
    USE_SEMANTIC_SCORING = os.environ.get('NEURALRP_DISABLE_SEMANTIC_SCORING', '0') != '1'

    if not USE_SEMANTIC_SCORING:
        print("[RELATIONSHIP] Semantic scoring disabled via environment variable, skipping")
        return

    if len(character_objs) < 1:
        return

    if adaptive_tracker is None:
        print("[RELATIONSHIP] Adaptive tracker not initialized, skipping")
        return

    # Extract entity IDs and names from character objects
    entity_map = {}  # entity_id -> name
    name_map = {}    # name -> entity_id
    for char_obj in character_objs:
        entity_id = get_entity_id(char_obj)
        name = get_character_name(char_obj)
        entity_map[entity_id] = name
        name_map[name] = entity_id

    # Build entity list for relationship analysis
    all_entity_ids = list(entity_map.keys())
    user_entity_id = None

    # Add user to entity list if provided
    if user_name:
        user_entity_id = "user_default"  # Standardized entity ID format
        all_entity_ids.append(user_entity_id)
        entity_map[user_entity_id] = user_entity_id
        name_map[user_entity_id] = user_name  # Keep display name mapping

    recent_speakers = set(m.speaker or m.role for m in messages[-10:])

    for char_from_id in all_entity_ids:
        char_from_name = entity_map.get(char_from_id, char_from_id)

        if char_from_name not in recent_speakers and char_from_id not in recent_speakers:
            continue

        for char_to_id in all_entity_ids:
            if char_from_id == char_to_id:
                continue

            char_to_name = entity_map.get(char_to_id, char_to_id)

            if char_to_name not in recent_speakers and char_to_id not in recent_speakers:
                continue

            current_state = db_get_relationship_state(chat_id, char_from_id, char_to_id)

            if not current_state:
                current_state = {
                    'trust': 50, 'emotional_bond': 50, 'conflict': 50,
                    'power_dynamic': 50, 'fear_anxiety': 50
                }

            relationship_messages = []
            for msg in messages[-10:]:
                speaker = msg.speaker or msg.role
                content = msg.content or ''

                # Check if speaker is the character_from entity
                if char_from_id in speaker or char_from_name in speaker:
                    # Check if content mentions character_to entity
                    if char_to_id.lower() in content.lower() or char_to_name.lower() in content.lower():
                        relationship_messages.append(content)

                    # Check for pronoun references
                    if any(word in content.lower() for word in ['he', 'she', 'they', 'you']):
                        relationship_messages.append(content)

            if not relationship_messages:
                continue

            try:
                new_scores = adaptive_tracker.analyze_conversation_scores(
                    messages=relationship_messages,
                    current_state=current_state
                )

                if new_scores is None:
                    continue

                if any(abs(new_scores[dim] - current_state[dim]) > 5
                       for dim in ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']):

                    db_update_relationship_state(
                        chat_id=chat_id,
                        character_from=char_from_id,
                        character_to=char_to_id,
                        scores=new_scores,
                        last_message_id=messages[-1].id if messages else 0
                    )

                    print(f"[RELATIONSHIP] {char_from_name}→{char_to_name}: "
                          f"trust={new_scores['trust']}, bond={new_scores['emotional_bond']}, "
                          f"conflict={new_scores['conflict']}")

            except Exception as e:
                print(f"[RELATIONSHIP] Scoring failed for {char_from_name}→{char_to_name}: {e}")
                continue


# Global variable to store cleanup task
cleanup_task = None

def store_image_metadata(filename: str, params: SDParams):
    """Store image generation parameters in database and JSON file for compatibility."""
    # Save to database (primary source)
    params_dict = {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "steps": params.steps,
        "cfg_scale": params.cfg_scale,
        "width": params.width,
        "height": params.height,
        "sampler_name": params.sampler_name,
        "scheduler": params.scheduler,
        "timestamp": int(time.time())
    }
    db_save_image_metadata(filename, params_dict)
    
    # Also export to JSON file for backward compatibility
    metadata_file = os.path.join(IMAGE_DIR, "image_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {"images": {}}
    
    metadata["images"][filename] = params_dict
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_image_metadata():
    """Load image metadata from database."""
    return db_get_all_image_metadata()

# Periodic cleanup task for semantic search engine and change log
async def periodic_cleanup():
    """Periodically clean up semantic search engine resources and old data"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            semantic_search_engine.cleanup_resources()
            
            # Clean up old change log entries every 24 hours (check every 5 min, run once per day)
            # We use a simple counter to avoid checking too frequently
            if not hasattr(periodic_cleanup, '_last_change_cleanup'):
                periodic_cleanup._last_change_cleanup = 0
            
            periodic_cleanup._last_change_cleanup += 1
            if periodic_cleanup._last_change_cleanup >= 288:  # 288 * 5 min = 24 hours
                db_cleanup_old_changes(30)  # Keep 30 days of history
                db_cleanup_old_metrics(7)   # Keep 7 days of performance metrics
                periodic_cleanup._last_change_cleanup = 0
                print("Periodic cleanup: Cleaned old change logs and metrics")
        except Exception as e:
            print(f"Periodic cleanup failed: {e}")

# Helper functions for JSON sync
def normalize_world_name(filename: str) -> str:
    """Remove SillyTavern suffixes from world filename.
    
    'exampleworld_plist_worldinfo.json' → 'exampleworld'
    """
    name = filename.replace(".json", "")
    for suffix in ["_plist", "_worldinfo", "_json"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def find_world_json_path(world_name: str) -> Optional[str]:
    """Find JSON file path for a world, handling SillyTavern suffixes."""
    wi_dir = os.path.join(DATA_DIR, "worldinfo")
    
    if not os.path.exists(wi_dir):
        return None
    
    # Try exact match first
    exact_path = os.path.join(wi_dir, f"{world_name}.json")
    if os.path.exists(exact_path):
        return exact_path
    
    # Try with suffixes
    for suffix in ["_worldinfo", "_plist", "_json"]:
        path = os.path.join(wi_dir, f"{world_name}{suffix}.json")
        if os.path.exists(path):
            return path
    
    return None


def entries_content_equal(entry_a: Dict, entry_b: Dict) -> bool:
    """Compare two world entries for content equality (ignoring metadata)."""
    return (
        entry_a.get("content") == entry_b.get("content") and
        entry_a.get("key") == entry_b.get("key") and
        entry_a.get("keysecondary") == entry_b.get("keysecondary") and
        entry_a.get("is_canon_law") == entry_b.get("is_canon_law")
    )


def sync_character_from_json(filename: str) -> Dict[str, Any]:
    """Sync a character from JSON file to database.
    
    Used after editing character JSON to update database copy.
    Uses timestamp-based conflict resolution: newer wins.
    
    Args:
        filename: Character filename (e.g., "Alice.json")
    
    Returns:
        {"success": True/False, "action": "created"|"updated"|"unchanged", "message": str}
    """
    try:
        json_path = os.path.join(DATA_DIR, "characters", filename)
        if not os.path.exists(json_path):
            return {"success": False, "action": "error", "message": "JSON file not found"}
        
        # Load JSON file
        json_mtime = int(os.path.getmtime(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        # Check database
        db_char = db_get_character(filename)
        
        if db_char is None:
            # New character - just save it
            db_save_character(json_data, filename)
            return {"success": True, "action": "created", "message": "Character imported to database"}
        
        # Character exists - check timestamps
        db_updated = db_get_character_updated_at(filename) or 0
        
        if json_mtime > db_updated:
            # JSON is newer - update database
            db_save_character(json_data, filename)
            return {"success": True, "action": "updated", "message": "Database updated from JSON (newer)"}
        else:
            # Database is newer or same - no action needed
            return {"success": True, "action": "unchanged", "message": "Database version is current"}
    
    except Exception as e:
        return {"success": False, "action": "error", "message": str(e)}


def sync_world_from_json(world_name: str) -> Dict[str, int]:
    """Smart sync a world from JSON file to database.
    
    Entry-level granularity with timestamp-based conflict resolution:
    - New entries in JSON → ADD to database
    - Entries only in DB → KEEP (user additions via UI)
    - Same UID, different content → NEWER WINS
    
    Args:
        world_name: World name (without .json extension)
    
    Returns:
        {"added": N, "updated": N, "unchanged": N, "kept": N}
    """
    result = {"added": 0, "updated": 0, "unchanged": 0, "kept": 0}
    
    # Find JSON file (handle SillyTavern suffixes)
    json_path = find_world_json_path(world_name)
    if not json_path or not os.path.exists(json_path):
        return result
    
    json_mtime = int(os.path.getmtime(json_path))
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    json_entries = json_data.get("entries", json_data) if isinstance(json_data, dict) else {}
    json_tags = json_data.get("tags", [])
    
    # Get database state
    db_world = db_get_world(world_name)
    
    if db_world is None:
        # New world - import entirely
        if db_save_world(world_name, json_entries, json_tags):
            result["added"] = len(json_entries)
        return result
    
    db_entries = db_world.get("entries", {})
    db_timestamps = db_get_world_entry_timestamps(world_name)
    
    # Track which entries we've processed
    processed_uids = set()
    
    # Process JSON entries
    for uid, json_entry in json_entries.items():
        processed_uids.add(uid)
        
        if uid not in db_entries:
            # New entry from JSON
            result["added"] += 1
        else:
            db_entry = db_entries[uid]
            if entries_content_equal(json_entry, db_entry):
                result["unchanged"] += 1
            else:
                # Content differs - check timestamps
                db_updated = db_timestamps.get(uid, 0)
                if json_mtime > db_updated:
                    result["updated"] += 1
                else:
                    result["kept"] += 1  # DB version is newer, keep it
    
    # Entries only in DB (user additions) - keep them
    for uid in db_entries:
        if uid not in processed_uids:
            result["kept"] += 1
    
    # If any changes, rebuild and save
    if result["added"] > 0 or result["updated"] > 0:
        # Merge: start with DB entries, overlay JSON entries where JSON is newer
        merged_entries = dict(db_entries)  # Keep DB entries
        for uid, json_entry in json_entries.items():
            if uid not in db_entries:
                merged_entries[uid] = json_entry  # Add new
            elif uid in db_entries:
                db_updated = db_timestamps.get(uid, 0)
                if json_mtime > db_updated:
                    merged_entries[uid] = json_entry  # JSON is newer
                # else: keep db version (already in merged_entries)
        
        db_save_world(world_name, merged_entries, json_tags)
    
    return result


def normalize_character_v2(char_data: dict) -> dict:
    """Ensure character card conforms to SillyTavern V2 spec.
    Adds missing fields with proper defaults.
    
    Args:
        char_data: Character dictionary (may or may not have spec wrapper)
    
    Returns:
        Normalized character data with all V2 fields present.
    """
    # Ensure spec wrapper
    if 'spec' not in char_data:
        char_data['spec'] = 'chara_card_v2'
    if 'spec_version' not in char_data:
        char_data['spec_version'] = '2.0'
    
    data = char_data.setdefault('data', {})
    
    # Required string fields (empty string default)
    string_fields = [
        'name', 'description', 'personality', 'scenario', 
        'first_mes', 'mes_example', 'creator_notes', 
        'system_prompt', 'post_history_instructions',
        'creator', 'character_version'
    ]
    for field in string_fields:
        if field not in data or data[field] is None:
            data[field] = ''
    
    # Required array fields
    if 'alternate_greetings' not in data or data['alternate_greetings'] is None:
        data['alternate_greetings'] = []
    
    if 'tags' not in data or data['tags'] is None:
        data['tags'] = []
    
    # Required null field
    if 'character_book' not in data:
        data['character_book'] = None
    
    # Extensions (preserve existing, add defaults for missing)
    extensions = data.setdefault('extensions', {})
    if 'multi_char_summary' not in extensions:
        extensions['multi_char_summary'] = ''
    if 'depth_prompt' not in extensions:
        extensions['depth_prompt'] = {'depth': 4, 'prompt': ''}
    elif 'prompt' not in extensions['depth_prompt']:
        extensions['depth_prompt']['prompt'] = ''
    if 'gender' not in extensions:
        extensions['gender'] = ''
    
    return char_data


async def import_characters_json_files_async(force: bool = False) -> int:
    """Import character JSON files to database with capsule auto-generation.
    
    Args:
        force: If True, reimport even if character exists (overwrites DB)
    """
    import_count = 0
    char_dir = os.path.join(DATA_DIR, "characters")
    
    if not os.path.exists(char_dir):
        return 0
    
    for f in os.listdir(char_dir):
        if f.endswith(".json") and f != ".gitkeep":
            file_path = os.path.join(char_dir, f)
            try:
                existing = db_get_character(f)
                if existing is None or force:
                    with open(file_path, "r", encoding="utf-8") as cf:
                        char_data = json.load(cf)
                    
                    # Normalize to V2 spec
                    char_data = normalize_character_v2(char_data)
                    
                    if db_save_character(char_data, f):
                        # Auto-generate capsule for newly imported character
                        try:
                            capsule = await generate_capsule_for_character(
                                char_name=char_data.get('name', ''),
                                description=char_data.get('description', ''),
                                personality=char_data.get('personality', ''),
                                scenario=char_data.get('scenario', ''),
                                mes_example=char_data.get('mes_example', ''),
                                gender=char_data.get('extensions', {}).get('gender', '')
                            )
                            # Update character data with generated capsule
                            if 'extensions' not in char_data:
                                char_data['extensions'] = {}
                            char_data['extensions']['multi_char_summary'] = capsule
                            # Re-save with capsule
                            db_save_character(char_data, f)
                            print(f"  [CAPSULE] Auto-generated capsule for {f}")
                        except Exception as e:
                            print(f"  [CAPSULE] Failed to generate capsule for {f}: {e}")
                        
                        import_count += 1
                        print(f"Imported character: {f}")
            except Exception as e:
                print(f"Failed to import character {f}: {e}")
    
    if import_count > 0:
        print(f"Character import complete: {import_count} characters")
    
    return import_count


def import_characters_json_files(force: bool = False) -> int:
    """Sync wrapper for async import (for startup compatibility).
    
    Note: Capsules are auto-generated during import for new characters.
    """
    import_count = 0
    char_dir = os.path.join(DATA_DIR, "characters")
    
    if not os.path.exists(char_dir):
        return 0
    
    for f in os.listdir(char_dir):
        if f.endswith(".json") and f != ".gitkeep":
            file_path = os.path.join(char_dir, f)
            try:
                existing = db_get_character(f)
                if existing is None or force:
                    with open(file_path, "r", encoding="utf-8") as cf:
                        char_data = json.load(cf)
                    
                    # Normalize to V2 spec
                    char_data = normalize_character_v2(char_data)
                    
                    if db_save_character(char_data, f):
                        import_count += 1
                        print(f"Imported character: {f}")
            except Exception as e:
                print(f"Failed to import character {f}: {e}")
    
    if import_count > 0:
        print(f"Character import complete: {import_count} characters")
    
    return import_count


def import_world_info_json_files(force: bool = False) -> int:
    """Import world info JSON files to database.
    
    Args:
        force: If True, reimport even if world exists (overwrites DB)
    
    Returns:
        Number of worlds imported
    """
    import_count = 0
    wi_dir = os.path.join(DATA_DIR, "worldinfo")
    
    if not os.path.exists(wi_dir):
        return 0
    
    for f in os.listdir(wi_dir):
        if f.endswith(".json") and f != ".gitkeep":
            name = normalize_world_name(f)
            file_path = os.path.join(wi_dir, f)
            try:
                existing = db_get_world(name)
                if existing is None or force:
                    with open(file_path, "r", encoding="utf-8") as wf:
                        world_data = json.load(wf)
                    entries = world_data.get("entries", world_data) if isinstance(world_data, dict) else {}
                    tags = world_data.get("tags", [])
                    if entries and db_save_world(name, entries, tags):
                        import_count += 1
                        print(f"Imported world info: {name}")
            except Exception as e:
                print(f"Failed to import world info {f}: {e}")
    
    if import_count > 0:
        print(f"World info import complete: {import_count} worlds")
    
    return import_count


# Auto-import JSON files on startup
def auto_import_json_files() -> Dict[str, int]:
    """Scan JSON folders and import any new files not already in database.

    Returns:
        Dict with counts: {"characters": N, "worlds": N, "chats": N}
    """
    result = {
        "characters": import_characters_json_files(),
        "worlds": import_world_info_json_files(),
        "chats": import_chats_json_files()
    }
    
    total = sum(result.values())
    if total > 0:
        print(f"Auto-import complete: {result['characters']} characters, {result['worlds']} worlds, {result['chats']} chats")
    else:
        print("Auto-import: No new JSON files to import")
    
    return result
 
def import_chats_json_files():
    """Import only chat JSON files not already in database."""
    import_count = 0
    
    chat_dir = os.path.join(DATA_DIR, "chats")
    if os.path.exists(chat_dir):
        for f in os.listdir(chat_dir):
            if f.endswith(".json") and f != ".gitkeep":
                name = f.replace(".json", "")
                file_path = os.path.join(chat_dir, f)
                try:
                    # Check if already in database
                    existing = db_get_chat(name)
                    if existing is None:
                        # Load and import
                        with open(file_path, "r", encoding="utf-8") as cf:
                            chat_data = json.load(cf)
                        if db_save_chat(name, chat_data):
                            import_count += 1
                            print(f"Imported chat: {name}")
                except Exception as e:
                    print(f"Failed to import chat {f}: {e}")
    
    if import_count > 0:
        print(f"Chat import complete: {import_count} chats")
    else:
        print("Chat import: No new JSON files to import")
    
    return import_count


async def check_ai_services():
    """Check if KoboldCpp and Stable Diffusion are running on startup"""
    import httpx
    
    kobold_url = CONFIG.get("kobold_url", "http://127.0.0.1:5001")
    sd_url = CONFIG.get("sd_url", "http://127.0.0.1:7861")
    
    # Check KoboldCpp
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{kobold_url}/api/v1/model")
            if response.status_code == 200:
                print(f"  [OK] KoboldCpp found at {kobold_url}")
            else:
                print(f"  [WARN] KoboldCpp responded but may not be ready")
    except Exception:
        print(f"  [ERROR] KoboldCpp not found at {kobold_url}")
        print(f"     [TIP] Start KoboldCpp with your model before chatting")
        print(f"     [TIP] Download: https://github.com/LostRuins/koboldcpp")
    
    # Check Stable Diffusion
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{sd_url}/sdapi/v1/samplers")
            if response.status_code == 200:
                print(f"  [OK] Stable Diffusion found at {sd_url}")
            else:
                print(f"  [WARN] Stable Diffusion responded but may not be ready")
    except Exception:
        print(f"  [ERROR] Stable Diffusion not found at {sd_url}")
        print(f"     [TIP] Start A1111 WebUI with --api flag for image generation")
        print(f"     [TIP] Download: https://github.com/AUTOMATIC1111/stable-diffusion-webui")
    
    print()  # Empty line for readability


# FastAPI startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Initialize resources when FastAPI app starts"""
    global cleanup_task, adaptive_tracker
    
    # Verify database integrity before proceeding
    if not verify_database_health():
        print("[WARNING] Database health check failed - app may not function correctly")
        print("   Try running: python migrate_to_sqlite.py")
    
    # Check for external AI services
    print("\n[CHECK] Checking for AI services...")
    await check_ai_services()
    
    # Auto-import any new JSON files dropped into folders
    print("Scanning for new JSON files to import...")
    auto_import_json_files()
    
    # Populate FTS5 search index with existing messages (one-time migration)
    print("Running FTS5 search index migration...")
    migrate_populate_fts()
    
    # Clean up old autosaved chats and empty chats on startup
    print("Running chat cleanup on startup...")
    old_chats = db_cleanup_old_autosaved_chats(days=7)
    empty_chats = db_cleanup_empty_chats()
    
    # Load semantic search model before initializing adaptive tracker
    print("Loading semantic search model...")
    if semantic_search_engine.load_model():
        print("Semantic search model loaded successfully")
    else:
        print("Warning: Failed to load semantic search model - some features may be unavailable")
    
    # Initialize adaptive relationship tracker with shared semantic model
    print("Initializing adaptive relationship tracker...")
    if initialize_adaptive_tracker(semantic_search_engine):
        print("[ADAPTIVE_TRACKER] Ready - Three-tier detection system active")
    else:
        print("[ADAPTIVE_TRACKER] Warning: Could not initialize - semantic model not loaded")

    # Initialize snapshot feature (v1.9.0)
    print("Initializing snapshot feature...")
    global snapshot_analyzer, prompt_builder, snapshot_http_client
    snapshot_http_client = httpx.AsyncClient(timeout=10.0)
    snapshot_analyzer = SnapshotAnalyzer(
        semantic_search_engine=semantic_search_engine,
        http_client=snapshot_http_client,
        config=CONFIG
    )
    prompt_builder = SnapshotPromptBuilder()
    print("[SNAPSHOT] Snapshot feature initialized")

    # Migrate danbooru tags on first run
    print("Checking danbooru tags migration...")
    db_migrate_danbooru_tags()

    # Start periodic cleanup task when app starts
    cleanup_task = asyncio.create_task(periodic_cleanup())
    print("Periodic cleanup task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the FastAPI app shuts down"""
    global cleanup_task, snapshot_http_client
    if cleanup_task:
        # Cancel the cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            print("Periodic cleanup task cancelled")
    # Close snapshot HTTP client
    if snapshot_http_client:
        await snapshot_http_client.aclose()
        print("[SNAPSHOT] HTTP client closed")

def preprocess_world_info(world_info):
    """Pre-process world info for case-insensitive matching"""
    if not world_info or "entries" not in world_info:
        return world_info

    for entry in world_info["entries"].values():
        if "key" in entry:
            entry["key"] = [k.lower() for k in entry["key"]]
        if "keysecondary" in entry:
            entry["keysecondary"] = [k.lower() for k in entry["keysecondary"]]
    return world_info


def get_cached_world_entries(world_info, recent_text, max_entries=10, semantic_threshold=0.45, is_initial_turn=False):
    """Get triggered world info entries using semantic search with keyword match priority and caching"""
    if not world_info or "entries" not in world_info:
        return [], []

    # Preprocess world info for case-insensitive matching
    world_info = preprocess_world_info(world_info)

    # Use database-backed content hash for efficient cache invalidation
    # This detects changes to world info automatically
    world_name = world_info.get("name", "")
    if world_name:
        content_hash = db_get_world_content_hash(world_name)
    else:
        # Fallback for unnamed world info (shouldn't happen in practice)
        world_content = ""
        for entry in world_info.get("entries", {}).values():
            world_content += entry.get("content", "") + "|"
        content_hash = hashlib.md5(world_content.encode('utf-8')).hexdigest()

    # Create a composite hash key
    text_hash = hashlib.md5(recent_text.lower().encode('utf-8')).hexdigest()
    config_hash = hashlib.md5(f"{max_entries}_{semantic_threshold}_{is_initial_turn}".encode('utf-8')).hexdigest()

    cache_key = f"{content_hash}_{text_hash}_{config_hash}"

    if cache_key in WORLD_INFO_CACHE:
        return WORLD_INFO_CACHE[cache_key]

    # Perform semantic search (returns tuples: (content, similarity, uid))
    semantic_results, canon_entries = semantic_search_engine.search_semantic(
        world_info,
        recent_text,
        max_entries=max_entries * 2,  # Get more to allow for keyword priority sorting
        similarity_threshold=semantic_threshold,
        is_initial_turn=is_initial_turn
    )

    # Normalizer for dedup (case/whitespace/apostrophes/punctuation)
    def _normalize_content(s: str) -> str:
        # Remove apostrophes, quotes, hyphens, and other punctuation, then normalize whitespace
        s = re.sub(r"['\"\-\.,:;!?()]+", "", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    # Build a unified list with match metadata: (content, similarity, is_keyword_match, uid)
    all_matches = []
    seen_normalized = set()
    entries = world_info.get("entries", {})
    processed_text = recent_text.lower()

    # Stricter matching with comprehensive normalization: apostrophes, possessives, plurals, punctuation
    GENERIC_KEYS = SemanticSearchEngine.GENERIC_KEYS
  
    def normalize_for_matching(s: str) -> str:
        """Normalize text for matching: remove apostrophes, quotes, hyphens, and other punctuation"""
        return re.sub(r"['\"\-\.,:;!?()]+", "", s)

    def key_matches_text(key: str, text: str, is_quoted: bool = False) -> bool:
        key_normalized = normalize_for_matching(key.strip().lower())
        if len(key_normalized) < 3 or key_normalized in GENERIC_KEYS:
            return False
        
        key_pattern = re.escape(key_normalized).replace(r"\ ", r"\s+")
        
        if is_quoted:
            # Quoted keys: strict phrase match with word boundaries
            pattern = r"\b" + key_pattern + r"\b"
        else:
            # Unquoted keys: flexible matching (semantic handles variations)
            # Use word boundaries but allow matches as part of larger phrases
            pattern = r"(?:\b|^)" + key_pattern + r"(?:\b|$)"
        
        text_normalized = normalize_for_matching(text)
        return re.search(pattern, text_normalized) is not None

    def content_matches_text(content: str, text: str) -> bool:
        content_normalized = normalize_for_matching(content.strip().lower())
        if len(content_normalized) < 3:
            return False
        content_pattern = re.escape(content_normalized).replace(r"\ ", r"\s+")
        pattern = r"\b" + content_pattern + r"\b"
        
        text_normalized = normalize_for_matching(text)
        return re.search(pattern, text_normalized) is not None

    # First, add semantic results with keyword match detection
    for content, similarity, uid in semantic_results:
        norm = _normalize_content(content)
        if not norm or norm in seen_normalized:
            continue
        
        # Check if this is also a keyword match
        entry = entries.get(uid, {})
        metadata = world_info.get('_metadata', {}).get('entries', {}).get(uid, {})
        all_keys = entry.get("key", [])
        
        # Get key type flags
        is_quoted_flags = metadata.get('is_quoted_flags', [])
        
        # Match all keys with their type flags
        is_keyword_match = False
        for i, key in enumerate(all_keys):
            if not key.strip():
                continue
            is_quoted = is_quoted_flags[i] if i < len(is_quoted_flags) else False
            if key_matches_text(key, processed_text, is_quoted):
                is_keyword_match = True
                break
        
        all_matches.append((content, similarity, is_keyword_match, uid))
        seen_normalized.add(norm)

    # Then, check for keyword matches that semantic search might have missed
    for uid, entry in entries.items():
        if entry.get("is_canon_law"):
            # Handle canon law separately
            content = entry.get("content", "")
            norm = _normalize_content(content)
            if norm and norm not in seen_normalized:
                seen_normalized.add(norm)
                canon_entries.append(content)
            continue

        content = entry.get("content", "")
        norm = _normalize_content(content)
        
        # Skip if already in results
        if norm in seen_normalized:
            continue

        # Check for keyword match
        metadata = world_info.get('_metadata', {}).get('entries', {}).get(uid, {})
        all_keys = entry.get("key", [])
        
        # Get key type flags
        is_quoted_flags = metadata.get('is_quoted_flags', [])
        
        # Check all keys with their type flags
        match_on_keys = False
        for i, key in enumerate(all_keys):
            if not key.strip():
                continue
            is_quoted = is_quoted_flags[i] if i < len(is_quoted_flags) else False
            if key_matches_text(key, processed_text, is_quoted):
                match_on_keys = True
                break
        
        match_on_content = content_matches_text(content, processed_text)

        if match_on_keys or match_on_content:
            # This is a keyword match that wasn't in semantic results
            # Give it a default similarity of 0.0 to indicate it's purely keyword-matched
            all_matches.append((content, 0.0, True, uid))
            seen_normalized.add(norm)

    # UNIFIED SORTING: Keyword matches first (True > False), then by similarity (high to low)
    # Sort by (is_keyword_match DESC, similarity DESC)
    all_matches.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Apply max_entries limit AFTER sorting
    triggered_lore = [match[0] for match in all_matches[:max_entries]] if max_entries > 0 else [match[0] for match in all_matches]

    # Deduplicate canon entries
    deduped_canon = []
    canon_seen = set()
    for content in canon_entries:
        norm = _normalize_content(content)
        if norm and norm not in canon_seen:
            canon_seen.add(norm)
            deduped_canon.append(content)
    canon_entries = deduped_canon

    result = (triggered_lore, canon_entries)
    WORLD_INFO_CACHE[cache_key] = result
    return result

def character_has_speaker(messages: List[ChatMessage], char_name: str) -> bool:
    """Check if a character has ever spoken in the message history."""
    for msg in messages:
        speaker = msg.speaker or msg.role
        if char_name in speaker:
            return True
    return False

def character_has_appeared_recently(messages: List[ChatMessage], char_name: str) -> bool:
    """Check if a character has appeared in recent N messages.
    
    This is used to determine if a character should be treated as "reappearing"
    after being absent from the conversation for a while. Characters who haven't
    appeared in recent messages should get a fresh sticky window on return.
    
    Args:
        messages: Message history (may be truncated after summarization)
        char_name: Character name to search for
        
    Returns:
        True if character found in last 20 messages, False otherwise
    """
    # Check only the most recent 20 messages
    # This threshold allows characters to be treated as "reappearing" after being absent
    recent_messages = messages[-20:] if len(messages) > 20 else messages
    
    for msg in recent_messages:
        speaker = msg.speaker or msg.role
        if char_name in speaker:
            return True
    return False

def normalize_string_for_comparison(s: str) -> str:
    """Normalize a string for robust comparison.
    
    Handles common formatting differences that can occur between
    database storage and frontend message handling.
    """
    if not s:
        return ""
    
    # Strip leading/trailing whitespace
    normalized = s.strip()
    
    # Normalize line breaks (convert \r\n to \n)
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    
    # Normalize multiple consecutive spaces to single space
    normalized = ' '.join(normalized.split())
    
    return normalized

def is_first_message_auto_added(messages: List[ChatMessage], char_obj: dict) -> bool:
    """Check if the first assistant message is character's auto-added first_mes.
    
    This handles the edge case where frontend adds character's first_mes
    to messages before any real inference has happened.
    
    Returns True if:
    - Only one message exists AND it's the auto-added first_mes (original behavior), OR
    - First assistant message exists AND it matches character's first_mes (new behavior for multi-message case)
    
    Uses normalized comparison to handle formatting differences while
    maintaining speaker name precision.
    """
    if len(messages) == 0:
        return False
    
    # Get character's first message and name
    char_name = get_character_name(char_obj)
    char_first_mes = char_obj.get("data", {}).get("first_mes", "")
    
    # Empty first_mes means it wasn't auto-added
    if not char_first_mes:
        return False
    
    # Find the first assistant message in the sequence
    first_assistant_msg = None
    for msg in messages:
        if msg.role == "assistant":
            first_assistant_msg = msg
            break
    
    if not first_assistant_msg:
        return False
    
    # Skip Visual System messages
    if first_assistant_msg.speaker == "Visual System":
        return False
    
    # Check if first assistant message has speaker matching character
    speaker_matches = first_assistant_msg.speaker == char_name
    if not speaker_matches:
        return False
    
    # Normalize and compare content
    normalized_msg_content = normalize_string_for_comparison(first_assistant_msg.content)
    normalized_char_first_mes = normalize_string_for_comparison(char_first_mes)
    
    content_matches = normalized_msg_content == normalized_char_first_mes
    
    # Debug logging
    if len(messages) == 1 and content_matches:
        print(f"[AUTO_FIRST_MES] Single-message detection: {char_name}'s first_mes detected")
    elif len(messages) > 1 and content_matches:
        print(f"[AUTO_FIRST_MES] Extended detection: First assistant message matches {char_name}'s first_mes (total messages: {len(messages)})")
    elif speaker_matches and not content_matches:
        print(f"[DEBUG] First message detection: Speaker matched '{char_name}' but content mismatch")
        print(f"[DEBUG]   Message content (first 100 chars): {normalized_msg_content[:100]}...")
        print(f"[DEBUG]   Character first_mes (first 100 chars): {normalized_char_first_mes[:100]}...")
    
    return content_matches

# Prompt Construction Engine
def construct_prompt(request: PromptRequest, character_first_turns: Dict[str, int] = None, absolute_turn: int = None, scene_update_block: str = "", character_full_card_turns: Dict[str, int] = None):
    if character_first_turns is None:
        character_first_turns = {}
    if character_full_card_turns is None:
        character_full_card_turns = {}
    settings = request.settings
    system_prompt = settings.get("system_prompt", CONFIG["system_prompt"])
    user_persona = settings.get("user_persona", "")
    user_name = settings.get("user_name", "")  # Get userName for relationships
    summary = request.summary or ""
    mode = request.mode or "narrator"
    
    # Calculate current turn number (counting user messages, 1-indexed)
    # Turn 1 = first user message, Turn 2 = second user message, etc.
    # Use absolute_turn if provided (survives summarization), otherwise calculate from messages
    if absolute_turn is not None:
        current_turn = absolute_turn
    else:
        current_turn = sum(1 for msg in request.messages if msg.role == "user")
    
    # Initial turns (1 and 2) get special treatment for world info and canon law
    is_initial_turn = current_turn <= 2
    
    # Determine chat mode
    is_group_chat = len(request.characters) >= 2
    is_single_char = len(request.characters) == 1
    is_narrator_mode = not request.characters
    
    # Collect active character names early for mode instruction
    # ⚠️ CRITICAL: Always use get_character_name() for consistency with relationship tracking
    # DO NOT: char['data']['name'] or name.split()[0] (first-name extraction breaks relationships)
    active_names = []
    for char_obj in request.characters:
        active_names.append(get_character_name(char_obj))
    
    # Extract capsules from character objects for SCENE CAST
    character_capsules = {}
    for char in request.characters:
        entity_id = char.get('_filename')
        if entity_id:
            capsule = char.get('data', {}).get('extensions', {}).get('multi_char_summary')
            if capsule:
                character_capsules[entity_id] = capsule
    
    for char in request.characters:
        if char.get('is_npc'):
            entity_id = char.get('entity_id')
            if entity_id:
                capsule = char.get('data', {}).get('extensions', {}).get('multi_char_summary')
                if capsule:
                    character_capsules[entity_id] = capsule
    
    # === RECORD FIRST TURNS FOR NEW CHARACTERS ===
    for char in request.characters:
        char_ref = char.get('_filename') or char.get('entity_id')
        
        # If not in dict, this is first appearance - record it
        if char_ref and char_ref not in character_first_turns:
            character_first_turns[char_ref] = absolute_turn
    
    # === 1. SYSTEM PROMPT ===
    narrator_instruction = " Act as a Narrator. Describe the world and speak for any NPCs the user encounters. Do not speak for the {{user}}"
    if is_narrator_mode:
        if "act as a narrator" not in system_prompt.lower():
            system_prompt += narrator_instruction
    
    full_prompt = f"### System: {system_prompt}\n"
    
    # === 2. MODE INSTRUCTIONS (moved up - sets interpretation context) ===
    if is_group_chat:
        if mode.startswith("focus:"):
            focus_char = mode.split(":", 1)[1]
            full_prompt += f"\n[MODE: FOCUS-{focus_char}]\n"
            full_prompt += FOCUSED_TEMPLATE.format(CharacterName=focus_char) + "\n"
        else:
            full_prompt += "\n[MODE: NARRATOR]\n"
            full_prompt += NARRATOR_INSTRUCTION + "\n"
    elif is_single_char:
        # Single character also gets mode instruction now
        char_name = active_names[0]
        if mode == "narrator":
            full_prompt += "\n[MODE: NARRATOR]\n"
            full_prompt += NARRATOR_INSTRUCTION + "\n"
        else:
            # Default to focused mode for single character
            full_prompt += f"\n[MODE: FOCUS-{char_name}]\n"
            full_prompt += FOCUSED_TEMPLATE.format(CharacterName=char_name) + "\n"
    # Pure narrator mode (0 characters) already has instruction appended to system prompt
    
    # === 3. LONG-TERM CONTEXT (Summary) ===
    if summary:
        full_prompt += f"### Long-Term Context (Summary of earlier turns):\n{summary}\n"
    
    # === 4. USER PERSONA ===
    if user_name or user_persona:
        full_prompt += "### User Description:\n"
        if user_name:
            full_prompt += f"Name: {user_name}\n"
        if user_persona:
            full_prompt += f"{user_persona}\n"
        full_prompt += "\n"

    # === 5. WORLD KNOWLEDGE (moved up - world context frames characters) ===
    canon_law_entries = []
    if request.world_info:
        # Use latest user message only for initial turns, otherwise use last 5 messages
        if is_initial_turn:
            # Find the latest user message
            latest_user_message = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    latest_user_message = msg.content
                    break
            recent_text = latest_user_message.lower()
        else:
            # Use the standard 5-message window for subsequent turns
            recent_text = " ".join([m.content for m in request.messages[-5:]]).lower()

        max_world_entries = settings.get("max_world_info_entries", 3)

        # Use optimized semantic search with turn detection
        triggered_lore, canon_law_entries = get_cached_world_entries(
            request.world_info,
            recent_text,
            max_entries=max_world_entries,
            semantic_threshold=0.45,
            is_initial_turn=is_initial_turn
        )
        
        # Add triggered world lore to prompt (if any matches)
        if triggered_lore:
            full_prompt += "### World Knowledge:\n" + "\n".join(triggered_lore) + "\n"

    # === 6. SCENE UPDATE (only if cast changed this turn - Phase 2) ===
    if scene_update_block:
        full_prompt += f"\n{scene_update_block}\n"

    # === 6.5. RECENT UPDATES (one-time notification, independent of intervals) ===
    # This is completely separate from SCENE CAST and canon law:
    # - Does NOT affect current_turn calculation
    # - Does NOT affect world_reinforce_freq timing
    # - Does NOT affect semantic search
    # - Character reinforcement replaced by SCENE CAST (v1.11.0)
    # Appears on ONE turn after edit, then disappears

    
    # === 6. HYBRID CHARACTER INJECTION (full cards + SCENE CAST) ===
    
    # Detect characters needing full cards (reset points)
    chars_needing_full_card = []
    for char in request.characters:
        char_ref = char.get('_filename') or char.get('entity_id')
        char_name = get_character_name(char)
        
        # Skip if no entity ID
        if not char_ref:
            continue
        
        # Get first turn from metadata (already recorded at function start)
        first_turn = character_first_turns.get(char_ref)
        
        # First appearance?
        is_first_appearance = (first_turn == absolute_turn)
        
        # Returning after long absence?
        # Only check if we have enough message history (more than just auto first_mes)
        has_appeared_recently = False
        if len(request.messages) > 0:
            has_appeared_recently = character_has_appeared_recently(
                request.messages[-20:], 
                char_name
            )
        
        is_returning = (first_turn is not None and 
                        first_turn < absolute_turn and 
                        not has_appeared_recently)
        
        # Sticky window: get full card for 2 more turns after first appearance or return
        full_card_turn = character_full_card_turns.get(char_ref)
        is_in_sticky_window = (
            full_card_turn is not None and
            absolute_turn - full_card_turn <= 2
        )
        
        if is_first_appearance or is_returning or is_in_sticky_window:
            chars_needing_full_card.append(char_ref)
            
            if is_in_sticky_window:
                print(f"[CONTEXT] {char_name}: Full card (sticky window)")
            else:
                reason = "first appearance" if is_first_appearance else "returning after absence"
                print(f"[CONTEXT] {char_name}: Full card ({reason})")
    
    # Build combined block: FULL CARDS + SCENE CAST
    if chars_needing_full_card or request.characters:
        full_prompt += "\n### Active Characters:\n\n"
    
    # Full cards for reset characters
    for char in request.characters:
        char_ref = char.get('_filename') or char.get('entity_id')
        if char_ref in chars_needing_full_card:
            full_card = build_full_character_card(char)
            full_prompt += full_card + "\n\n"
            
            # Record that this character got a full card this turn
            character_full_card_turns[char_ref] = absolute_turn
    
    # SCENE CAST capsules for remaining characters (no duplicates)
    scene_cast = build_scene_cast_block(
        request.characters,
        character_capsules,
        exclude=chars_needing_full_card
    )
    if scene_cast:
        full_prompt += scene_cast + "\n\n"

    # === 7. CHAT HISTORY (split into recent/old) ===
    # Use new window-based approach: recent 10 exchanges verbatim
    recent_messages, old_messages = split_messages_by_window(request.messages)

    full_prompt += "\n### Chat History:\n"
    for msg in recent_messages:
        # Filter out meta-messages like "Visual System" if they exist
        if msg.speaker == "Visual System":
            continue
        
        speaker = msg.speaker or ("User" if msg.role == "user" else "Narrator")
        full_prompt += f"{speaker}: {msg.content}\n"

    # === 8. CANON LAW (pinned for recency bias - right before generation) ===
    world_reinforce_freq = settings.get("world_info_reinforce_freq", 3)
    if canon_law_entries and world_reinforce_freq > 0 and should_show_canon_law(current_turn, world_reinforce_freq):
        full_prompt += "\n### Canon Law (World Rules):\n" + "\n".join(canon_law_entries) + "\n"

    # === 8.5. RELATIONSHIP CONTEXT (character-to-character and character-to-user dynamics) ===
    # Only inject if there are characters to have relationships
    if request.chat_id and (request.characters or user_name):
        relationship_context = get_relationship_context(
            chat_id=request.chat_id,
            character_objs=request.characters,  # Full character objects for entity ID extraction
            user_name=user_name,
            recent_messages=request.messages[-10:]  # Last 10 messages for relevance filtering
        )

        if relationship_context:
            full_prompt += relationship_context + "\n"

    # === 9. CONTINUE HINT + LEAD-IN ===
    if is_single_char:
        # Single character mode
        char_name = active_names[0]
        if mode == "narrator":
            # Narrator mode for single char - use narrator lead-in
            last_user_msg = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            continue_hint = get_continue_hint(mode, last_user_msg)
            if continue_hint:
                full_prompt += f"\n{continue_hint}\n"
            full_prompt += "Narrator:"
        else:
            full_prompt += f"{char_name}:"
    elif is_group_chat:
        # Multi-character mode - use mode to determine lead-in
        if mode.startswith("focus:"):
            # Focused mode: specific character speaks
            focus_char = mode.split(":", 1)[1]
            full_prompt += f"\n{focus_char}:"
        else:
            # Narrator mode: add continue hint and narrator lead-in
            last_user_msg = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            continue_hint = get_continue_hint(mode, last_user_msg)
            if continue_hint:
                full_prompt += f"\n{continue_hint}\n"
            full_prompt += "Narrator:"
    else:
        # No characters - pure narrator mode
        full_prompt += "Narrator:"
    
    return full_prompt

def should_show_canon_law(current_turn: int, freq: int) -> bool:
    """Determine if canon law should be shown on current turn."""
    return current_turn <= 2 or (current_turn > 2 and (current_turn - 2) % freq == 0)

def split_messages_by_window(
    messages: List[ChatMessage],
    max_exchanges: int = 6
) -> Tuple[List[ChatMessage], List[ChatMessage]]:
    """
    Split messages into recent (verbatim) and old (candidates for summarization).
    
    An exchange = 1 user message + 1 assistant response.
    Counts from the end, keeping most recent max_exchanges exchanges.
    
    Returns:
        (recent_messages, old_messages)
    """
    if not messages:
        return ([], [])
    
    exchange_count = 0
    split_index = 0
    
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == 'user':
            exchange_count += 1
            if exchange_count > max_exchanges:
                split_index = i + 1
                break
    
    old_messages = messages[:split_index]
    recent_messages = messages[split_index:]
    
    return (recent_messages, old_messages)

def calculate_turn_for_message(messages: List[ChatMessage], target_index: int) -> int:
    """
    Calculate which turn a message belongs to (1-indexed).
    Turn = count of user messages up to and including this message.
    """
    return sum(1 for msg in messages[:target_index+1] if msg.role == 'user')

def group_messages_into_scenes(
    messages: List[ChatMessage],
    cast_change_turns: List[int],
    max_exchanges_per_scene: int = 15
) -> List[Tuple[int, int, List[ChatMessage]]]:
    """
    Group messages into scenes for capsule generation.

    Boundaries:
    1. Cast changes force a scene boundary
    2. Otherwise, every 15 exchanges = new scene

    Returns:
        List of (start_turn, end_turn, messages) tuples
    """
    scenes = []
    current_scene_messages = []
    current_scene_start = 1
    exchange_count = 0
    turn_counter = 0

    for i, msg in enumerate(messages):
        if msg.role == 'user':
            turn_counter += 1

        is_cast_boundary = turn_counter in cast_change_turns

        if msg.role == 'user':
            exchange_count += 1
        hit_exchange_limit = (exchange_count >= max_exchanges_per_scene)

        if (is_cast_boundary or hit_exchange_limit) and current_scene_messages:
            scenes.append((
                current_scene_start,
                turn_counter - 1,
                current_scene_messages
            ))
            current_scene_messages = []
            current_scene_start = turn_counter
            exchange_count = 0

        current_scene_messages.append(msg)

    if current_scene_messages:
        scenes.append((current_scene_start, turn_counter, current_scene_messages))

    return scenes

async def generate_scene_capsule(
    messages: List[ChatMessage],
    start_turn: int,
    end_turn: int,
    departed_note: str = "",
    canon_law_entries: List[str] = None
) -> str:
    """
    Generate a scene capsule from a list of messages.
    """
    conversation_text = "\n".join([
        f"{m.speaker or m.role}: {m.content}"
        for m in messages
    ])

    canon_echo = ""
    if canon_law_entries:
        canon_echo = "\n### Note: Active Canon Laws to maintain:\n" + "\n".join(canon_law_entries)

    prompt = SCENE_CAPSULE_PROMPT.format(
        departed_note=departed_note,
        canon_echo=canon_echo,
        start_turn=start_turn,
        end_turn=end_turn,
        conversation_text=conversation_text
    )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CONFIG['kobold_url']}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_length": 200,
                    "temperature": 0.5,
                    "stop_sequence": ["###", "\nUser:", "\nAssistant:", "\nScene ("]
                },
                timeout=60.0
            )
            data = response.json()
            capsule = data["results"][0]["text"].strip()
            return capsule
        except Exception as e:
            logger.error(f"Failed to generate scene capsule (turns {start_turn}-{end_turn}): {e}")
            return ""

async def trigger_cast_change_summarization(
    chat_id: str,
    departed_characters: Set[str],
    entity_to_name: Dict[str, str],
    canon_law_entries: List[str] = None
):
    """
    Summarize old messages after a cast change.
    Simple approach: ONE capsule for all messages outside 10-exchange window.
    """
    try:
        chat = db_get_chat(chat_id)
        if not chat:
            logger.warning(f"Cast-change summarization: chat {chat_id} not found")
            return

        messages = chat.get('messages', [])
        recent, old = split_messages_by_window(messages)

        if not old:
            logger.info(f"Cast-change summarization: no old messages for chat {chat_id}")
            return

        old_start_index = len(messages) - len(recent) - len(old)
        start_turn = calculate_turn_for_message(messages, old_start_index)
        end_turn = calculate_turn_for_message(messages, len(messages) - 1)

        departed_names = [entity_to_name.get(e, e) for e in departed_characters]
        departed_note = f"Note: {', '.join(departed_names)} exited during this scene. Mention their departure."

        scene_capsule = await generate_scene_capsule(
            old,
            start_turn=start_turn,
            end_turn=end_turn,
            departed_note=departed_note,
            canon_law_entries=canon_law_entries
        )

        if not scene_capsule:
            logger.warning(f"Cast-change summarization: failed for chat {chat_id}")
            return

        existing_summary = chat.get('summary', '')
        new_summary = (existing_summary + "\n\n" + scene_capsule).strip()
        chat['summary'] = new_summary
        db_save_chat(chat_id, chat)

        logger.info(f"Cast-change summarization completed for chat {chat_id} ({len(old)} messages)")

    except Exception as e:
        logger.error(f"Cast-change summarization failed for chat {chat_id}: {e}")
        logger.exception(e)

async def trigger_threshold_summarization(
    chat_id: str,
    canon_law_entries: List[str] = None
):
    """
    Summarize old messages when context exceeds 70% threshold.
    Uses scene grouping: multiple capsules based on cast changes + 15-exchange limit.
    """
    try:
        chat = db_get_chat(chat_id)
        if not chat:
            logger.warning(f"Threshold summarization: chat {chat_id} not found")
            return

        messages = chat.get('messages', [])
        recent, old = split_messages_by_window(messages)

        if not old:
            logger.info(f"Threshold summarization: no old messages for chat {chat_id}")
            return

        metadata = chat.get('metadata', {})
        cast_change_turns = metadata.get('cast_change_turns', [])

        scenes = group_messages_into_scenes(old, cast_change_turns, max_exchanges_per_scene=15)

        if not scenes:
            logger.warning(f"Threshold summarization: no scenes for chat {chat_id}")
            return

        capsules = []
        for start, end, scene_msgs in scenes:
            capsule = await generate_scene_capsule(
                scene_msgs,
                start_turn=start,
                end_turn=end,
                departed_note="",
                canon_law_entries=canon_law_entries
            )
            if capsule:
                capsules.append(capsule)

        if not capsules:
            logger.warning(f"Threshold summarization: no capsules for chat {chat_id}")
            return

        existing_summary = chat.get('summary', '')
        combined_capsules = "\n\n".join(capsules)
        new_summary = (existing_summary + "\n\n" + combined_capsules).strip()
        chat['summary'] = new_summary
        db_save_chat(chat_id, chat)

        logger.info(f"Threshold summarization completed for chat {chat_id} ({len(scenes)} scenes, {len(old)} messages)")

    except Exception as e:
        logger.error(f"Threshold summarization failed for chat {chat_id}: {e}")
        logger.exception(e)

def count_tokens(text: str) -> int:
    """
    Estimate token count using heuristic.
    Uses ~4 chars per token for backward compatibility.
    """
    return len(text) // 4

def build_fallback_capsule(name: str, data: Dict) -> str:
    """Build minimal capsule when pre-generated capsule unavailable."""
    parts = []

    if data.get('extensions', {}).get('gender'):
        parts.append(data['extensions']['gender'].capitalize())

    if data.get('description'):
        desc = data['description'].split('.')[0][:100]
        parts.append(desc)

    if data.get('personality'):
        pers = data['personality'][:50]
        parts.append(pers)

    if data.get('scenario'):
        scenario = data['scenario'][:80]
        parts.append(f"In {scenario}")

    if data.get('mes_example'):
        example = data['mes_example'].strip()
        if example:
            lines = example.split('\n')
            for line in lines:
                if '"' in line:
                    start = line.find('"')
                    end = line.rfind('"') + 1
                    if end > start:
                        parts.append(f'Speaks like: {line[start:end]}')
                        break

    return '. '.join(parts) if parts else "No description available"

def build_full_character_card(char: Dict) -> str:
    """Build full character card for reset points.
    
    Same format for single/multi-char, global/NPC.
    """
    data = char.get('data', {})
    name = get_character_name(char)
    
    parts = [f"### Character: {name}"]
    
    if data.get('description'):
        parts.append(data['description'])
    
    if data.get('personality'):
        parts.append(data['personality'])
    
    if data.get('scenario'):
        parts.append(data['scenario'])
    
    if data.get('mes_example'):
        parts.append(f"Example dialogue:\n{data['mes_example']}")
    
    return "\n\n".join(parts)

def build_scene_cast_block(
    characters: List[Dict],
    character_capsules: Dict[str, str],
    exclude: List[str] = []
) -> str:
    """
    Build SCENE CAST block using capsule format for all active characters.
    
    This replaces the separate reinforcement cycle - shown every turn.
    
    Args:
        characters: List of character objects (global chars and NPCs, resolved)
        character_capsules: Dict of entity_id -> capsule text (includes both chars and NPCs)
        exclude: List of entity_ids to exclude from SCENE CAST (full card characters)
    """
    # Filter to only active characters
    active_chars = [c for c in characters if c.get('is_active', True)]
    
    # Filter out excluded entity_ids (full card characters)
    active_chars = [c for c in active_chars 
                    if c.get('_filename') not in exclude 
                    and c.get('entity_id') not in exclude]
    
    if not active_chars:
        return ""
    
    lines = [
        "SCENE CAST (ACTIVE ONLY)",
        "The following characters are currently present. Follow their speech styles and personalities.",
        ""
    ]
    
    for char in active_chars:
        is_npc = char.get('is_npc', False)
        
        # Get entity ID (different format for global chars vs NPCs)
        if is_npc:
            entity_id = char.get('entity_id')
        else:
            entity_id = char.get('_filename')
        
        name = get_character_name(char)
        
        # Get capsule or build fallback
        capsule = character_capsules.get(entity_id)
        if not capsule:
            data = char.get('data', {})
            capsule = build_fallback_capsule(name, data)
        
        # Format based on type
        if is_npc:
            lines.append(f"[{name} (NPC): {capsule}]")
        else:
            lines.append(f"[{name}: {capsule}]")
    
    return "\n".join(lines)

# Character Card Generation Logic (Ported from Char_card_app)
async def call_llm_helper(system_prompt: str, user_prompt: str, max_tokens: int = 500):
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "prompt": f"### System: {system_prompt}\n### User: {user_prompt}\n### Assistant:",
                "max_length": max_tokens,
                "temperature": 0.7,
                "stop_sequence": ["###", "<|im_end|>", "\n\n\n", "User:", "Assistant:"]
            }
            print(f"DEBUG LLM CALL: {KOBOLD_API_URL if 'KOBOLD_API_URL' in globals() else CONFIG['kobold_url']}/api/v1/generate")
            res = await client.post(
                f"{CONFIG['kobold_url']}/api/v1/generate",
                json=payload,
                timeout=120.0
            )
            res.raise_for_status()
            data = res.json()
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0]["text"].strip()
            else:
                print(f"LLM Error: Unexpected response structure: {data}")
                raise Exception("Unexpected response from LLM")
        except Exception as e:
            print(f"LLM Call Exception: {str(e)}")
            raise Exception(f"API Error: {str(e)}")

async def summarize_text(text: str) -> str:
    """Summarize provided text into a concise version."""
    system = "You are an expert at distilling information into concise summaries."
    
    prompt = f"""Summarize the following text into a more concise version while preserving all key information.

Requirements:
- Reduce length by 50-70%
- Keep important plot points, character actions, and world details
- Maintain neutral, factual tone
- Do NOT add new information
- Remove redundancy and filler

Text to summarize:
{text}

Output only the summarized text, nothing else."""

    result = await call_llm_helper(system, prompt, 300)
    return result.strip()

@app.post("/api/card-gen/generate-field")
async def generate_card_field(req: CardGenRequest):
    char_name = req.char_name
    field_type = req.field_type
    user_input = req.context
    
    # Handle name collision for local NPC creation
    if req.save_as == 'local_npc' and req.chat_id:
        # ============================================================================
        # ENHANCED NPC CREATION WITH EARLY VALIDATION
        # ============================================================================
        print(f"[NPC_CREATE] Starting NPC creation process")
        print(f"[NPC_CREATE] Parameters: chat_id={req.chat_id!r}, save_as={req.save_as!r}, source_mode={req.source_mode!r}")
        print(f"[NPC_CREATE] Input: field_type={field_type!r}, char_name={char_name!r}")
        
        # VALIDATE chat_id FIRST (early validation - defensive programming)
        print(f"[NPC_CREATE] Step 1: Validating chat_id")
        
        if not req.chat_id or req.chat_id.strip() == '':
            print(f"[NPC_CREATE] VALIDATION FAILED: No chat_id provided")
            return {
                "success": False,
                "error": "Chat ID is required for NPC creation. Please ensure chat is initialized first."
            }
        
        # Additional validation: Ensure chat_id is a string (not undefined or wrong type)
        if not isinstance(req.chat_id, str):
            print(f"[NPC_CREATE] VALIDATION FAILED: chat_id is not a string (type={type(req.chat_id)})")
            return {
                "success": False,
                "error": f"Invalid chat_id type. Expected string, got {type(req.chat_id).__name__}."
            }
        
        print(f"[NPC_CREATE] Step 2: Checking if chat exists in database")
        
        # Load chat data with enhanced validation
        chat = db_get_chat(req.chat_id)
        
        if not chat:
            print(f"[NPC_CREATE] VALIDATION FAILED: Chat '{req.chat_id}' not found in database")
            return {
                "success": False,
                "error": f"Chat '{req.chat_id}' not found in database. Cannot create NPC in non-existent chat. Please create a new chat or ensure the chat ID is valid."
            }
        
        print(f"[NPC_CREATE] VALIDATION PASSED: Chat '{req.chat_id}' exists and is valid")
        print(f"[NPC_CREATE] Step 3: Proceeding with name generation and collision checks")
        
        generated_name = char_name
        # If no name provided, generate one (fallback to existing default logic)
        if not generated_name or generated_name.strip() == '':
            print(f"[NPC_CREATE] No name provided, generating random name")
            # Quick random name list
            DEFAULT_NAMES = ["Alex", "Morgan", "Riley", "Jordan", "Casey", 
                           "Quinn", "Blake", "Parker", "Avery", "Sage"]
            generated_name = random.choice(DEFAULT_NAMES)
            print(f"[NPC_CREATE] Generated fallback name: {generated_name}")
        else:
            print(f"[NPC_CREATE] Using user-provided name: {generated_name}")
            
            print(f"[NPC_CREATE] Step 4: Checking for name collisions")
            # Check for name collision with global characters
            active_chars = chat.get('activeCharacters', [])
            global_char_names = []
            for char_ref in active_chars:
                # Handle both string and dict formats
                if isinstance(char_ref, dict):
                    char_ref = char_ref.get('id') or char_ref.get('name') or ''

                if isinstance(char_ref, str) and not char_ref.startswith('npc_'):
                    char_data = db_get_character(char_ref)
                    if char_data:
                        global_char_names.append(get_character_name(char_data).lower())
            
            print(f"[NPC_CREATE] Step 4a: Checking global character names: {len(global_char_names)} unique names")
            
            # Check for collision
            if generated_name.lower() in global_char_names:
                # Append suffix to NPC name
                generated_name = f"{generated_name} (NPC)"
                print(f"[NPC_CREATE] Name collision detected with global character, renamed to: {generated_name}")
            
            print(f"[NPC_CREATE] Step 4b: Checking for collisions with existing NPCs")
            # Check for collision with other NPCs
            metadata = chat.get("metadata", {}) or {}
            localnpcs = metadata.get("localnpcs", {}) or {}
            existing_npc_names = [npc.get("name", "").lower() for npc in localnpcs.values()]
            
            print(f"[NPC_CREATE] Step 4b: Found {len(existing_npc_names)} existing NPCs to check against")
            
            if generated_name.lower() in existing_npc_names:
                # Append number
                counter = 2
                base_name = generated_name
                while f"{base_name} {counter}".lower() in existing_npc_names:
                    counter += 1
                generated_name = f"{base_name} {counter}"
                print(f"[NPC_CREATE] Duplicate NPC name detected, renamed to: {generated_name}")
            
            print(f"[NPC_CREATE] Step 5: Finalizing NPC name and generating entity_id")
            # Update char_name with resolved name
            char_name = generated_name
            print(f"[NPC_CREATE] Final NPC name: {char_name}")
            
            # Generate unique entity_id for NPC
            from app.database import generate_entity_id
            entity_id = f"npc_{int(time.time())}_{generate_entity_id()}"
            print(f"[NPC_CREATE] Step 5a: Generated unique entity_id: {entity_id}")
            
            print(f"[NPC_CREATE] Step 6: Starting LLM generation for character card")
            # Generate character card fields for NPC using LLM

            system_prompt = "You are a creative character designer. Generate detailed, engaging character information in a structured format."

            user_prompt = f"""Create a character card for this character:

            Name: {char_name}
            Description: {user_input}

            Generate the following in this exact format:

            PERSONALITY_TRAITS:
            [List 4-6 personality traits, comma-separated]

            PHYSICAL_TRAITS:
            [List 4-6 physical/body traits, comma-separated]

            SCENARIO:
            [Write a scenario describing the setting/situation where you meet this character]

            FIRST_MESSAGE:
            [Write the character's first message/greeting using *asterisks* for actions and "quotes" for speech]

            DIALOGUE_EXAMPLES:
            [Write 2-3 dialogue exchanges using this exact format]

            <START>
            {{user}}: [ask a relevant question]
            {{char}}: *[action]* "[response in character]"

            <START>
            {{user}}: [ask another relevant question]
            {{char}}: *[action]* "[response in character]"

            GENRE:
            [One word genre like: Fantasy, SciFi, Modern, etc.]"""

            try:
                print(f"[NPC_CREATE] Step 6a: Starting LLM API call")
                print(f"[NPC_CREATE] Step 6a: LLM system prompt: You are a creative character designer. Generate detailed, engaging character information in a structured format.")
                print(f"[NPC_CREATE] Step 6a: LLM user prompt (first 200 chars): Create a character card for this character: Name: {char_name}, Description: {user_input[:200] if len(user_input) > 200 else user_input}...")
                
                llm_result = await call_llm_helper(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=1000
                )
                
                # Set default values
                description = user_input
                personality_traits = ["mysterious", "capable"]
                physical_traits = []
                scenario = f"You encounter {char_name}."
                first_message = f"*{char_name} looks at you.*"
                dialogue_examples = "<START>\n{{user}}: \"Hello, nice to meet you.\"\n{{char}}: *" + char_name + " nods politely.* \"Pleasure to meet you as well.\"\n\n<START>\n{{user}}: \"Can you tell me about yourself?\"\n{{char}}: *" + char_name + " considers for a moment.* \"I am " + char_name + ", and I'm here to help.\""
                genre = "General"
                tags = ["generated", "npc"]
                
                print(f"[NPC_CREATE] Step 6b: Parsing LLM response")
                print(f"[NPC_CREATE] Step 6b: LLM raw response (first 500 chars): {llm_result[:500] if llm_result and len(llm_result) > 500 else llm_result}...")
                
                # Extract sections from LLM response
                if "PERSONALITY_TRAITS:" in llm_result:
                    traits_text = llm_result.split("PERSONALITY_TRAITS:")[1].split("PHYSICAL_TRAITS:")[0].strip()
                    personality_traits = [t.strip().strip('"').strip("'") for t in traits_text.split(",") if t.strip()]
                    print(f"[NPC_CREATE] Extracted PERSONALITY_TRAITS: {len(personality_traits)} traits")

                if "PHYSICAL_TRAITS:" in llm_result:
                    physical_text = llm_result.split("PHYSICAL_TRAITS:")[1].split("SCENARIO:")[0].strip()
                    physical_traits = [t.strip().strip('"').strip("'") for t in physical_text.split(",") if t.strip()]
                    print(f"[NPC_CREATE] Extracted PHYSICAL_TRAITS: {len(physical_traits)} traits")

                if "SCENARIO:" in llm_result:
                    scenario = llm_result.split("SCENARIO:")[1].split("FIRST_MESSAGE:")[0].strip()
                    print(f"[NPC_CREATE] Extracted SCENARIO: {scenario[:80]}...")

                if "FIRST_MESSAGE:" in llm_result:
                    first_message = llm_result.split("FIRST_MESSAGE:")[1].split("DIALOGUE_EXAMPLES:")[0].strip()
                    print(f"[NPC_CREATE] Extracted FIRST_MESSAGE: {first_message[:80]}...")

                # Extract DIALOGUE_EXAMPLES for mes_example
                if "DIALOGUE_EXAMPLES:" in llm_result:
                    dialogue_examples = llm_result.split("DIALOGUE_EXAMPLES:")[1].split("GENRE:")[0].strip()
                    print(f"[NPC_CREATE] Extracted DIALOGUE_EXAMPLES: {len(dialogue_examples)} chars")
                else:
                    # Fallback: Generate default dialogue examples
                    dialogue_examples = "<START>\n"
                    dialogue_examples += "{{user}}: \"Hello, nice to meet you.\"\n"
                    dialogue_examples += "{{char}}: *" + char_name + " nods politely.* \"Pleasure to meet you as well.\"\n\n"
                    dialogue_examples += "<START>\n"
                    dialogue_examples += "{{user}}: \"Can you tell me about yourself?\"\n"
                    dialogue_examples += "{{char}}: *" + char_name + " considers for a moment.* \"I am " + char_name + ", and I'm here to help.\""
                    print(f"[NPC_CREATE] DIALOGUE_EXAMPLES not found, using fallback")

                if "GENRE:" in llm_result:
                    genre = llm_result.split("GENRE:")[1].strip().split("\n")[0]
                    print(f"[NPC_CREATE] Extracted GENRE: {genre}")

                print(f"[NPC_CREATE] Step 6c: LLM generation successful - all fields parsed")
                
            except Exception as e:
                # Handle LLM failure and use defaults
                print(f"[NPC_CREATE] Step 6d: LLM generation failed, using default values")
                print(f"[NPC_CREATE] Step 6d: Error exception: {type(e).__name__}: {str(e)[:200]}")
                print(f"[NPC_CREATE] Step 6d: Falling back to default NPC values")
                
                # Set defaults
                description = user_input
                personality_traits = ["mysterious", "capable"]
                physical_traits = ["average build"]
                scenario = f"You encounter {char_name} for the first time."
                first_message = f"*{char_name} looks up from what they were doing.*"
                dialogue_examples = "<START>\n{{user}}: \"Hello, nice to meet you.\"\n{{char}}: *" + char_name + " nods politely.* \"Pleasure to meet you as well.\"\n\n<START>\n{{user}}: \"Can you tell me about yourself?\"\n{{char}}: *" + char_name + " considers for a moment.* \"I am " + char_name + ", and I'm here to help.\""
                genre = "General"
                tags = ["generated", "npc"]

            print(f"[NPC_CREATE] Step 7: Building character data structure")
            print(f"[NPC_CREATE] Step 7a: Character name: {char_name}")
            print(f"[NPC_CREATE] Step 7b: Description: {description[:80]}...")
            print(f"[NPC_CREATE] Step 7c: Personality traits: {personality_traits}")
            print(f"[NPC_CREATE] Step 7d: Physical traits: {physical_traits}")
            print(f"[NPC_CREATE] Step 7e: Scenario: {scenario[:80]}...")
            print(f"[NPC_CREATE] Step 7f: First message: {first_message[:80]}...")
            print(f"[NPC_CREATE] Step 7g: Genre: {genre}")
            
            # Build SillyTavern-compatible character data
            
            # Add source information to creator_notes
            source_info = f"Created from {req.source_mode}: {user_input[:100]}\n\n" if req.source_mode else ""
            
            # Build PList-formatted personality and body
            # PList format: [Character's Personality= "trait1", "trait2", ...]
            plist_personality = ""
            if personality_traits:
                trait_list = ', '.join([f'"{trait}"' for trait in personality_traits])
                plist_personality = f"[{char_name}'s Personality= {trait_list}]"
            
            # PList format: [Character's body= "feature1", "feature2", ...]
            plist_body = ""
            if physical_traits:
                trait_list = ', '.join([f'"{trait}"' for trait in physical_traits])
                plist_body = f"[{char_name}'s body= {trait_list}]"
            
            # Build creator_notes (only genre, no PList)
            creator_notes = source_info
            if genre:
                creator_notes += f"[Genre: {genre}]"

            # Create full character card with generated data
            character_data = {
                "name": char_name,
                "description": plist_body,
                "personality": plist_personality,
                "scenario": scenario,
                "first_mes": first_message,
                "mes_example": dialogue_examples,
                "creator_notes": creator_notes,
                "system_prompt": "",
                "post_history_instructions": "",
                "alternate_greetings": [],
                "tags": [],  # Empty - manual tags only
                "creator": "NeuralRP NPC Generator",
                "extensions": {
                    "depth_prompt": {
                        "prompt": "",
                        "depth": 4
                    },
                    "talkativeness": 100,
                    "multi_char_summary": "",
                    "gender": req.gender or "",
                    "danbooru_tag": ""
                }
            }
            print(f"[NPC_CREATE] Step 7h: Character data structure built, keys: {list(character_data.keys())}")


            print(f"[NPC_CREATE] Step 8: Checking for duplicate NPCs before database insertion")
            
            # Get existing NPCs from database
            existing_npcs = db_get_chat_npcs(req.chat_id)
            print(f"[NPC_CREATE] Step 8a: Found {len(existing_npcs)} existing NPCs in chat")
            
            # Check for duplicates
            duplicate_check_result = None
            for npc in existing_npcs:
                if npc.get("name", "").lower() == char_name.lower():
                    duplicate_check_result = npc
                    print(f"[NPC_CREATE] Step 8a: DUPLICATE FOUND: {npc.get('name')!r} (entityid: {npc.get('entityid')})")
                    break
            
            if duplicate_check_result:
                print(f"[NPC_CREATE] Step 8b: Returning duplicate error")
                return {
                    "success": False,
                    "error": f"NPC '{char_name}' already exists in this chat",
                    "existing_entityid": duplicate_check_result.get("entityid") or duplicate_check_result.get("entity_id"),
                    "existing_name": duplicate_check_result.get("name"),
                }
            
            print(f"[NPC_CREATE] Step 8c: No duplicate found, proceeding with database insertion")
            
            # Call the NEW db_create_npc signature
            print(f"[NPC_CREATE] Step 8d: Calling db_create_npc_and_update_chat with chat_id={req.chat_id!r}, name={char_name!r}")
            print(f"[NPC_CREATE] Step 8d: Character data keys: {list(character_data.keys())}")
            
            # Call NEW atomic function - does both NPC creation AND chat update in one transaction
            success, entity_id, error = db_create_npc_and_update_chat(
                chat_id=req.chat_id,
                npc_data=character_data,
            )

            print(f"[NPC_CREATE] Step 8e: db_create_npc_and_update_chat returned: success={success}, entity_id={entity_id!r}, error={error!r}")

            if not success:
                print(
                    f"[API][NPC] db_create_npc_and_update_chat failed: chat_id={req.chat_id!r}, "
                    f"name={character_data.get('name')!r}, error={error!r}"
                )
                return {
                    "success": False,
                    "error": error or "Failed to create NPC in database",
                    "existing_entityid": entity_id,
                    "existing_name": character_data.get("name"),
                }
            
            print(f"[NPC_CREATE] Step 9: Database operation succeeded - NPC created and chat updated atomically")
            
            print(f"[NPC_CREATE] Step 10: NPC creation completed successfully")
            print(f"[NPC_CREATE] Step 10: Final summary:")
            print(f"[NPC_CREATE] Step 10:   - NPC name: {char_name}")
            print(f"[NPC_CREATE] Step 10:   - Entity ID: {entity_id}")
            print(f"[NPC_CREATE] Step 10:   - Chat ID: {req.chat_id}")
            print(f"[NPC_CREATE] Step 10:   - Total time: {int(time.time()) - int(time.time())} seconds (this is a placeholder)")
            
            # IMPORTANT: Return with consistent field names (no underscore for frontend)
            return {
                "success": True,
                "entityid": entity_id,      # Frontend expects 'entityid' (no underscore)
                "entity_id": entity_id,     # Keep for backwards compatibility
                "name": char_name,
                "message": f"NPC '{char_name}' created successfully"
            }
    
    # Customize instructions based on source mode
    source_desc = "provided chat context" if req.source_mode != "manual" else "provided text"
    
    try:
        if field_type == 'personality':
            system = f"You are an expert at analyzing characters and writing roleplay personality traits."
            prompt = f"""Based on {source_desc}, identify {char_name}'s personality traits.
Convert them into a PList personality array format.
Use this exact format: [{char_name}'s Personality= "trait1", "trait2", "trait3", ...]

Source Text:
{user_input}

Only output the personality array line, nothing else."""
            result = await call_llm_helper(system, prompt, 300)
            
        elif field_type == 'body':
            system = f"You are an expert at analyzing characters and writing physical descriptions."
            prompt = f"""Based on {source_desc}, identify {char_name}'s physical features.
Convert them into a PList body array format.
Use this exact format: [{char_name}'s body= "feature1", "feature2", "feature3", ...]

Source Text:
{user_input}

Only output the body array line, nothing else."""
            result = await call_llm_helper(system, prompt, 300)
            
        elif field_type == 'full':
            # Handle 'full' field type for both chat and manual source modes
            # This generates a complete character card from the provided context
            system = "You are a creative character designer. Generate detailed, engaging character information in a structured format."

            prompt = f"""Create a character card for this character:

Name: {char_name}
Description: {user_input}

Generate the following in this exact format:

PERSONALITY_TRAITS:
[List 4-6 personality traits, comma-separated]

PHYSICAL_TRAITS:
[List 4-6 physical/body traits, comma-separated]

DESCRIPTION:
[Write a detailed description of the character's appearance, background, and mannerisms]

SCENARIO:
[Write a scenario describing the setting/situation where you meet this character]

FIRST_MESSAGE:
[Write the character's first message/greeting using *asterisks* for actions and "quotes" for speech]

GENRE:
[One word genre like: Fantasy, SciFi, Modern, etc.]"""

            result = await call_llm_helper(system, prompt, max_tokens=1000)
            
        elif field_type == 'dialogue_likes':
            system = "You are an expert at writing roleplay dialogue in markdown format with {{char}} and {{user}} placeholders."
            prompt = f"""Based on {char_name}'s behavior in the {source_desc}, write a dialogue exchange where {{user}} asks "{char_name}, what are your likes and dislikes?", and {{char}} responds in character.

Source Text:
{user_input}

Use this exact format:
{{user}}: "{char_name}, what are your likes and dislikes?"
{{char}}: *Action.* "Speech." *More action and description.*

Make the response 3-5 sentences, vivid and in-character. Only output the dialogue exchange."""
            result = await call_llm_helper(system, prompt, 500)
            
        elif field_type == 'dialogue_story':
            system = "You are an expert at writing roleplay dialogue in markdown format with {{char}} and {{user}} placeholders."
            prompt = f"""Based on {char_name}'s behavior in the {source_desc}, write a dialogue exchange where {{user}} asks "{char_name}, tell me about your life story", and {{char}} responds with a brief life story in character.

Source Text:
{user_input}

Use this exact format:
{{user}}: "{char_name}, tell me about your life story."
{{char}}: *Action.* "Speech." *More action and description.*

Make the response 3-5 sentences, vivid and in-character. Only output the dialogue exchange."""
            result = await call_llm_helper(system, prompt, 500)
            
        elif field_type == 'genre':
            system = "You are an expert at categorizing roleplay genres."
            prompt = f"""Based on the chat context involving {char_name}, suggest an appropriate genre (fantasy, sci-fi, modern, historical, etc.):

{user_input}

Only output the genre name, nothing else."""
            result = await call_llm_helper(system, prompt, 50)
            
        elif field_type == 'tags':
            system = "You are an expert at tagging roleplay characters."
            prompt = f"""Based on the chat context involving {char_name}, suggest 2-4 relevant tags (e.g., adventure, romance, comedy, mystery, magic, etc.):

{user_input}

Only output the tags separated by commas, nothing else."""
            result = await call_llm_helper(system, prompt, 100)
            
        elif field_type == 'scenario':
            system = "You are an expert at writing roleplay scenarios."
            prompt = f"""Based on the chat context, write a one-sentence scenario describing {char_name}'s initial situation:

{user_input}

Format: You [situation]. Make it engaging and specific. Only output the scenario sentence."""
            result = await call_llm_helper(system, prompt, 200)
            
        elif field_type == 'first_message':
            system = "You are an expert at writing engaging roleplay opening messages. Match the character's voice from the chat."
            prompt = f"""Write a first message for {char_name} that would start a new roleplay, based on the provided chat history.

Chat Context:
{user_input}

Use markdown format with *actions* and "speech". Make it immersive. Only output the first message."""
            result = await call_llm_helper(system, prompt, 600)
            
        else:
            return {"success": False, "error": f"Unknown field type: {field_type}"}
        
        # Apply PList formatting to ensure proper output format
        formatted_result = extract_plist_from_llm_output(result, field_type, char_name)
        
        return {"success": True, "text": formatted_result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Character Card Field Formatting Helper Functions
def extract_plist_from_llm_output(result: str, field_type: str, char_name: str) -> str:
    """Extract PList-formatted content from LLM output and ensure proper format."""
    import re
    
    print(f"[PLIST_EXTRACT] Processing field_type='{field_type}', char_name='{char_name}'")
    print(f"[PLIST_EXTRACT] Raw result: {result[:200]}...")
    
    # If result already starts with bracket, it's likely PList formatted
    if result.strip().startswith('['):
        # Extract all bracketed entries
        bracketed_entries = re.findall(r'\[.+?\]', result)
        
        # Clean up entries
        lines = [entry.strip() for entry in bracketed_entries if entry.strip()]
        
        if lines:
            print(f"[PLIST_EXTRACT] Found {len(lines)} bracketed entries")
            return '\n'.join(lines)
    
    # Try to parse structured format if LLM returned sections
    # This handles cases where LLM uses section headers like PERSONALITY_TRAITS:, PHYSICAL_TRAITS:, etc.
    result_lower = result.lower()
    
    # Handle different field types
    if field_type in ['personality', 'body']:
        # Try multiple patterns to extract traits
        
        # Pattern 1: Section headers (PERSONALITY_TRAITS:, PHYSICAL_TRAITS:, etc.)
        trait_pattern = r'(?:personality_traits|physical_traits|personality|body):\s*(.*?)(?=\n\n|\n[A-Z]|$)'
        match = re.search(trait_pattern, result, re.IGNORECASE | re.DOTALL)
        
        if match:
            traits_text = match.group(1).strip()
            print(f"[PLIST_EXTRACT] Extracted traits text: {traits_text[:150]}...")
            
            # Extract traits from comma-separated or quoted list
            # Handle both quoted and unquoted traits
            if '"' in traits_text or "'" in traits_text:
                # Quoted traits
                traits = re.findall(r'["\']([^"\']+)["\']', traits_text)
                print(f"[PLIST_EXTRACT] Extracted {len(traits)} quoted traits")
            else:
                # Unquoted traits (comma-separated) - handle multi-line
                # Split by comma, but also handle newlines and extra text
                raw_lines = traits_text.replace('\n', ',').split(',')
                traits = [t.strip() for t in raw_lines if t.strip() and len(t.strip()) > 2]
                # Filter out non-trait words
                traits = [t for t in traits if not t.lower() in ['include', 'features', 'appearance', 'traits', 'physical']]
                print(f"[PLIST_EXTRACT] Extracted {len(traits)} unquoted traits")
            
            if traits:
                # Format as PList
                trait_list = ', '.join([f'"{trait.strip()}"' for trait in traits])
                plist_key = 'Personality' if field_type == 'personality' else 'body'
                formatted = f"[{char_name}'s {plist_key}= {trait_list}]"
                print(f"[PLIST_EXTRACT] Formatted PList: {formatted[:100]}...")
                return formatted
        
        # FALLBACK: Try to parse entire result as comma-separated list
        # This handles cases where LLM returned plain traits without section headers
        print(f"[PLIST_EXTRACT] No section patterns found, trying fallback parsing")
        
        # Remove common LLM artifacts
        cleaned = re.sub(r'\(.*?\)', '', result)  # Remove parenthetical explanations
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # Remove any bracketed content
        cleaned = cleaned.replace('\n', ',')  # Convert newlines to commas
        
        # Split by comma and clean up
        potential_traits = [t.strip() for t in cleaned.split(',') if t.strip()]
        
        # Filter out non-trait words
        traits = []
        for t in potential_traits:
            t_lower = t.lower()
            # Skip common non-trait words and sentence fragments
            skip_words = ['include', 'features', 'appearance', 'traits', 'physical', 'personality', 
                        'here', 'are', 'some', 'the', 'is', 'and', 'or', 'such as', 'like', 
                        'including', 'with', 'for', 'a', 'an', 'of']
            if (len(t) > 2 and t_lower not in skip_words and 
                not t_lower.startswith('list') and 
                not t_lower.endswith('etc') and
                not re.match(r'^\d+', t)):  # Skip numbered lists
                traits.append(t)
        
        print(f"[PLIST_EXTRACT] Fallback extracted {len(traits)} traits from: {cleaned[:100]}...")
        
        if traits and len(traits) >= 2:  # Only use if we found at least 2 traits
            # Format as PList
            trait_list = ', '.join([f'"{trait.strip()}"' for trait in traits])
            plist_key = 'Personality' if field_type == 'personality' else 'body'
            formatted = f"[{char_name}'s {plist_key}= {trait_list}]"
            print(f"[PLIST_EXTRACT] Fallback formatted PList: {formatted[:100]}...")
            return formatted
        else:
            print(f"[PLIST_EXTRACT] Fallback failed: insufficient traits found")
    
    elif field_type == 'full':
        # Full character card generation - extract multiple sections
        # Extract personality traits
        personality_traits = []
        if 'personality_traits:' in result_lower:
            traits_match = re.search(r'personality_traits:\s*([^\n]*)', result_lower, re.IGNORECASE)
            if traits_match:
                traits_text = traits_match.group(1)
                personality_traits = [t.strip().strip('"').strip("'") for t in traits_text.split(',') if t.strip()]
        
        # Extract physical traits
        physical_traits = []
        if 'physical_traits:' in result_lower:
            phys_match = re.search(r'physical_traits:\s*([^\n]*)', result_lower, re.IGNORECASE)
            if phys_match:
                phys_text = phys_match.group(1)
                physical_traits = [t.strip().strip('"').strip("'") for t in phys_text.split(',') if t.strip()]
        
        # Format personality and body as PList
        personality_result = ""
        if personality_traits:
            trait_list = ', '.join([f'"{trait}"' for trait in personality_traits])
            personality_result = f"[{char_name}'s Personality= {trait_list}]"
        
        body_result = ""
        if physical_traits:
            trait_list = ', '.join([f'"{trait}"' for trait in physical_traits])
            body_result = f"[{char_name}'s body= {trait_list}]"
        
        # Return PList-formatted personality and body
        return f"{personality_result}\n{body_result}".strip()
    
    # FALLBACK: Return result as-is for other field types
    return result.strip()

# Capsule Generation for Multi-Character Optimization
async def generate_capsule_for_character(
    char_name: str,
    description: str,
    personality: str,
    scenario: str,
    mes_example: str,
    gender: str = ""
) -> str:
    """Generate a capsule summary for use in multi-character scenarios."""
    system = "You are an expert at distilling roleplay character cards into minimal capsule summaries for efficient multi-character prompts."

    full_card_text = f"Name: {char_name}"
    if gender:
        full_card_text += f"\nGender: {gender}"
    if description:
        full_card_text += f"\n\nDescription:\n{description}"
    if personality:
        full_card_text += f"\n\nPersonality:\n{personality}"
    if scenario:
        full_card_text += f"\n\nScenario:\n{scenario}"
    if mes_example:
        full_card_text += f"\n\nExample Dialogue:\n{mes_example}"
    
    prompt = f"""Convert this character card into a capsule summary for efficient multi-character prompts.
Use this exact format (one paragraph, no line breaks):
Name: [Name]. Gender: [gender if specified]. Role: [1 sentence character archetype or typical role]. Key traits: [3-5 comma-separated personality traits]. Speech style: [short/long, formal/casual, any verbal tics]. Example line: "[One characteristic quote from descriptions]"

IMPORTANT: The scenario field describes the character's lived experience, worldview, or yearning - NOT their current fixed location or event. Distill the scenario into character motivations, yearnings, or typical situations they encounter. Do NOT include specific places (like "at her mother's house") in the Role field.

Full Card:
{full_card_text}

Output only capsule summary line, nothing else."""

    result = await call_llm_helper(system, prompt, 200)
    return result.strip()

# Character Editing Endpoints
@app.post("/api/characters/edit-field")
async def edit_character_field(req: CharacterEditRequest):
    """Edit a specific field in a character card."""
    try:
        file_path = os.path.join(DATA_DIR, "characters", req.filename)
        if not os.path.exists(file_path):
            return {"success": False, "error": "Character file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            char_data = json.load(f)
        
        # Validate field exists
        if req.field not in ["personality", "body", "dialogue", "genre", "tags", "scenario", "first_message", "mes_example"]:
            return {"success": False, "error": "Invalid field"}

        # Update the field
        if req.field == "body":
            char_data["data"]["description"] = req.new_value
        else:
            char_data["data"][req.field] = req.new_value
        
        # Save the updated character
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, indent=2, ensure_ascii=False)

        # Sync to database so changes take effect immediately in active chats
        sync_result = sync_character_from_json(req.filename)
        if not sync_result["success"]:
            print(f"Warning: Character synced to JSON but database sync failed: {sync_result['message']}")
        

        
        return {"success": True, "message": f"Field '{req.field}' updated successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/characters/edit-field-ai")
async def edit_character_field_ai(req: CharacterEditFieldRequest):
    """Use AI to generate or improve a character field.

    Takes current field value + all other character data,
    generates improved/expanded content, and OVERWRITES the field.

    Supported fields:
    - mes_example: Example dialogue exchanges
    - personality: Personality traits (PList format)
    - body: Physical description (PList format, stored in description)
    - scenario: Single sentence scenario
    - first_message: Opening greeting (Markdown format)

    Args:
        req: CharacterEditFieldRequest with filename, field, context, source_mode

    Returns:
        JSON with success status and generated text
    """
    try:
        file_path = os.path.join(DATA_DIR, "characters", req.filename)
        if not os.path.exists(file_path):
            return {"success": False, "error": "Character file not found"}

        with open(file_path, "r", encoding="utf-8") as f:
            char_data = json.load(f)

        char_name = char_data["data"]["name"]
        personality = char_data["data"].get("personality", "")
        description = char_data["data"].get("description", "")
        scenario = char_data["data"].get("scenario", "")
        mes_example = char_data["data"].get("mes_example", "")

        result = ""

        if req.field == "mes_example":
            current_example = char_data["data"].get("mes_example", "")
            result = await generate_dialogue_for_edit(
                char_name, personality, description, scenario, current_example
            )
            char_data["data"]["mes_example"] = result

        elif req.field == "personality":
            current_personality = char_data["data"].get("personality", "")
            result = await generate_personality_for_edit(
                char_name, current_personality, description, scenario
            )
            char_data["data"]["personality"] = result

        elif req.field == "body":
            current_body = char_data["data"].get("description", "")
            result = await generate_body_for_edit(
                char_name, current_body, personality, description, scenario
            )
            char_data["data"]["description"] = result

        elif req.field == "scenario":
            current_scenario = char_data["data"].get("scenario", "")
            result = await generate_scenario_for_edit(
                char_name, current_scenario, personality, description, scenario
            )
            char_data["data"]["scenario"] = result

        elif req.field == "first_message":
            current_first_msg = char_data["data"].get("first_mes", "")
            result = await generate_first_message_for_edit(
                char_name, current_first_msg, personality, description, scenario
            )
            char_data["data"]["first_mes"] = result

        else:
            return {"success": False, "error": f"Field '{req.field}' not supported for AI generation"}

        # Save updated character to JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, indent=2, ensure_ascii=False)

        # Sync to database so changes take effect immediately in active chats
        sync_result = sync_character_from_json(req.filename)
        if not sync_result["success"]:
            print(f"Warning: Character synced to JSON but database sync failed: {sync_result['message']}")



        return {"success": True, "text": result}

    except Exception as e:
        print(f"Error in edit_character_field_ai: {e}")
        return {"success": False, "error": str(e)}

# World Info Generation Logic (Ported from world-gen-app)
WORLD_PROMPTS = {
    "history": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory mentioned. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects)]\n\nRules:\n- One entry per line\n- Keep entries concise\n- Use parentheses for nested attributes\n- Output ONLY PList entries in bracketed format [Name: type(...)], no explanations\n- DO NOT use XML plist format or <plist> tags\n- Match the tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory mentioned in the chat. Use this exact format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects)]\n\nRules:\n- One entry per line\n- Keep entries concise\n- Use parentheses for nested attributes\n- Output ONLY PList entries in bracketed format [Name: type(...)], no explanations\n- DO NOT use XML plist format or <plist> tags",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature/suggestive themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory. Include mature themes where appropriate. Use this format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects), intimate_aspects(relationships, desires, power dynamics)]\n\nRules:\n- One entry per line\n- Include suggestive details naturally\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event or backstory. Include explicit adult themes naturally. Use this format:\n[EventName: type(event/era/myth), time(when), actors(who), result(what happened), legacy(effects), sexual_culture(attitudes, practices, taboos)]\n\nRules:\n- One entry per line\n- Include explicit adult content where relevant\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags"
    },
    "locations": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each location, room, town, or area mentioned. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[LocationName(nickname if any): type(room/town/area), features(physical details), atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags\n- Match tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location, room, town, or area mentioned. Use this format:\n[LocationName(nickname if any): type(room/town/area), features(physical details), atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include suggestive atmosphere and uses. Use this format:\n[LocationName(nickname): type, features, atmosphere(mood), purpose(what happens), inhabitants(who), intimate_uses(private activities)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include explicit adult themes naturally. Format:\n[LocationName(nickname): type, features, atmosphere, purpose, inhabitants, sexual_activities(what happens), kinks_associated(themes)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags"
    },
    "creatures": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each creature, monster, or character archetype. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[CreatureName: type(creature/archetype), appearance(visual traits), behavior(typical actions), culture(social norms, beliefs), habitat(where found), attitude_toward_user(how they treat {{user}})]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags\n- Match tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each creature, monster, or character archetype. Use this format:\n[CreatureName: type(creature/archetype), appearance(visual traits), behavior(typical actions), culture(social norms, beliefs), habitat(where found), attitude_toward_user(how they treat {{user}})]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for creatures or archetypes. Include suggestive behavior and interactions. Format:\n[CreatureName: type, appearance, behavior, culture, habitat, attitude_toward_user, flirtation_style(how they seduce/interact)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[CreatureName: type, appearance, behavior, culture, attitude_toward_user, sexual_behavior(explicit details), kinks(preferences), consent_culture(boundaries)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags"
    },
    "factions": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each faction, group, or organization. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[FactionName: type(faction/guild/house/clique), members(who belongs), reputation(public image), goals(what they want), methods(how they operate), attitude_toward_user(how they treat {{user}}), rivals(opposing factions)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags\n- Match tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each faction, group, or organization. Use this format:\n[FactionName: type(faction/guild/house/clique), members(who belongs), reputation(public image), goals(what they want), methods(how they operate), attitude_toward_user(how they treat {{user}}), rivals(opposing factions)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for factions. Include suggestive motivations and interactions. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, social_dynamics(power, romance, rivalries), intimate_culture(dating norms, boundaries)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, sexual_culture(practices, rituals), kinks_favored(group preferences), initiation(how to join)]\n\nRules:\n- One entry per line\n- Output ONLY PList entries in bracketed format [Name: type(...)]\n- DO NOT use XML plist format or <plist> tags"
    }
}

def parse_plist_line(line):
    line = line.strip()
    if not (line.startswith('[') and line.endswith(']')): return None
    inner = line[1:-1]
    if ':' not in inner: return None
    name_part, content_part = inner.split(':', 1)
    name_part = name_part.strip()
    alias = None
    if '(' in name_part and ')' in name_part:
        match = re.match(r'^([^(]+)\(([^)]+)\)$', name_part)
        if match:
            name = match.group(1).strip()
            alias = match.group(2).strip()
        else: name = name_part
    else: name = name_part
    keys = [name.lower()]
    if alias: keys.append(alias.lower())
    words = re.findall(r'\b([a-zA-Z]{4,})\b', content_part.lower())
    for word in words[:3]:
        if word not in ['type', 'has', 'used', 'for', 'the', 'and', 'with'] and word not in keys:
            keys.append(word)
    return {"name": name, "alias": alias, "content": line, "keys": keys}

@app.post("/api/world-gen/generate")
async def generate_world_entries(req: WorldGenRequest):
    try:
        template = WORLD_PROMPTS.get(req.section, {}).get(req.tone, "")
        if not template: return {"success": False, "error": "Invalid section or tone"}
        
        # If in manual mode, the instruction still says "Chat Context", 
        # but the content is swapped. Let's make it more generic in the template swap.
        source_text = req.context
        
        prompt = template.format(worldName=req.world_name, input=source_text)
        result = await call_llm_helper("You are a world-building expert.", prompt, 600)
        # Apply PList formatting to extract bracketed content from LLM response
        formatted_result = extract_plist_from_llm_output(result, req.section, req.world_name)
        lines = formatted_result.split('\n')
        return {"success": True, "text": '\n'.join(lines)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/world-gen/save")
async def save_generated_world_info(req: WorldSaveRequest):
    try:
        lines = req.plist_text.split('\n')
        entries = {}
        uid_counter = 0
        for line in lines:
            parsed = parse_plist_line(line)
            if parsed:
                entries[str(uid_counter)] = {
                    "uid": uid_counter,
                    "key": parsed['keys'],
                    "keysecondary": [],
                    "comment": parsed['alias'] or "",
                    "content": parsed['content'],
                    "constant": False,
                    "selective": True,
                    "selectiveLogic": 0,
                    "addMemo": True,
                    "order": 100,
                    "position": 4,
                    "disable": False,
                    "excludeRecursion": False,
                    "probability": 100,
                    "useProbability": True,
                    "displayIndex": uid_counter,
                    "depth": 5,
                    "is_canon_law": False
                }
                uid_counter += 1
        
        if not entries: return {"success": False, "error": "No valid entries found"}
        
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)
        
        # Sync to database
        try:
            db_save_world(req.world_name, entries)
        except Exception as db_error:
            print(f"Warning: Failed to sync world gen to database: {db_error}")
        
        # Clear caches to reflect the change
        WORLD_INFO_CACHE.clear()
            
        return {"success": True, "name": req.world_name}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "app/index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Index.html not found"

# Health Check Endpoints
@app.get("/api/health/kobold", response_model=ServiceStatus)
async def check_kobold_health():
    """Check KoboldCpp connection status"""
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{CONFIG['kobold_url']}/api/v1/info")
            latency = int((time.time() - start_time) * 1000)
            service_status["kobold"] = ServiceStatus(
                status="connected",
                details="KoboldCpp API responding",
                latency_ms=latency
            )
            return service_status["kobold"]
    except httpx.TimeoutException:
        service_status["kobold"] = ServiceStatus(
            status="disconnected",
            details="Connection timeout (2s)",
            latency_ms=0
        )
        return service_status["kobold"]
    except Exception as e:
        service_status["kobold"] = ServiceStatus(
            status="disconnected",
            details=str(e),
            latency_ms=0
        )
        return service_status["kobold"]

@app.get("/api/health/sd", response_model=ServiceStatus)
async def check_sd_health():
    """Check Stable Diffusion WebUI connection status"""
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{CONFIG['sd_url']}/sdapi/v1/sd-models")
            latency = int((time.time() - start_time) * 1000)
            service_status["sd"] = ServiceStatus(
                status="connected",
                details="SD WebUI API responding",
                latency_ms=latency
            )
            return service_status["sd"]
    except httpx.TimeoutException:
        service_status["sd"] = ServiceStatus(
            status="disconnected",
            details="Connection timeout (2s)",
            latency_ms=0
        )
        return service_status["sd"]
    except Exception as e:
        service_status["sd"] = ServiceStatus(
            status="disconnected",
            details=str(e),
            latency_ms=0
        )
        return service_status["sd"]

@app.get("/api/health/status")
async def get_service_status():
    """Get current status of all services"""
    return {
        "kobold": service_status["kobold"],
        "sd": service_status["sd"]
    }

@app.post("/api/health/test-all")
async def test_all_services():
    """Test all service connections"""
    # Run health checks in parallel
    await asyncio.gather(
        check_kobold_health(),
        check_sd_health()
    )
    return {
        "success": True,
        "kobold": service_status["kobold"],
        "sd": service_status["sd"]
    }

# Performance mode management endpoints
@app.get("/api/performance/status")
async def get_performance_status():
    """Get current resource status and performance mode state"""
    return {
        "performance_mode_enabled": resource_manager.performance_mode_enabled,
        "status": resource_manager.get_status(),
        "median_llm_time": performance_tracker.get_median_llm(),
        "median_sd_time": performance_tracker.get_median_sd()
    }

@app.post("/api/performance/toggle")
async def toggle_performance_mode(request: dict):
    """Enable or disable performance mode"""
    enabled = request.get("enabled", True)
    resource_manager.performance_mode_enabled = enabled
    CONFIG["performance_mode_enabled"] = enabled
    return {"success": True, "enabled": enabled}

@app.post("/api/performance/dismiss-hint")
async def dismiss_hint(request: dict):
    """Dismiss a performance hint"""
    hint_id = request.get("hint_id")
    if hint_id:
        hint_engine.dismiss_hint(hint_id)
        return {"success": True}
    return {"success": False, "error": "No hint_id provided"}

def load_character_profiles(active_chars: List[str], localnpcs: Dict) -> List[dict]:
    """Load both global characters and local NPCs into unified character profile list."""
    character_profiles = []

    for char_ref in active_chars:
        # Handle both string and dict formats
        if isinstance(char_ref, dict):
            char_ref = char_ref.get("id") or char_ref.get("name") or ""

        if isinstance(char_ref, str) and char_ref.startswith("npc_"):
            # Load NPC from local metadata
            if char_ref in localnpcs:
                npc = localnpcs[char_ref]
                
                # Validate NPC has required data
                if not npc.get('name'):
                    print(f"[ERROR] NPC {char_ref} missing name, skipping")
                    continue
                
                if not npc.get('data'):
                    print(f"[ERROR] NPC {char_ref} missing character data, skipping")
                    continue
                
                # Check if NPC is active
                if not npc.get('is_active', True):
                    print(f"[CONTEXT] Skipping inactive NPC: {npc['name']}")
                    continue
                
                # Normalize NPC data to ensure all V2 fields (including extensions.gender)
                npc_data_normalized = normalize_character_v2({'data': npc['data']})['data']
                
                # Add to profiles
                character_profiles.append({
                    'name': npc['name'],
                    'data': npc_data_normalized,
                    'entity_id': char_ref,
                    'is_npc': True,
                    'updated_at': npc.get('updated_at', 0)
                })
            else:
                print(f"[WARNING] NPC {char_ref} not found in chat metadata, skipping")
        else:
            # Load global character from database
            char_data = db_get_character(char_ref)
            if char_data:
                # Validate global character too
                if not char_data.get('data'):
                    print(f"[ERROR] Global character {char_ref} missing data, skipping")
                    continue
                
                character_profiles.append({
                    'name': get_character_name(char_data),
                    'data': char_data.get('data', {}),
                    'entity_id': char_ref,
                    'is_npc': False,
                    'updated_at': char_data.get('updated_at', 0)
                })
            else:
                print(f"[WARNING] Global character {char_ref} not found, skipping")
    
    npc_count = sum(1 for c in character_profiles if c.get('is_npc', False))
    print(f"[CONTEXT] Loaded {len(character_profiles)} characters ({npc_count} NPCs)")
    
    return character_profiles

@app.post("/api/chat")
async def chat(request: PromptRequest):
    # Always define chat_data first
    chat_data = None
    
    # Periodic cleanup of old edits (1% chance per chat request)
    # NOTE: Recent edits cleanup removed - now using timestamp-based system

    # Load chat data if chat_id provided (needed for NPCs and metadata)
    if request.chat_id:
        chat_data = db_get_chat(request.chat_id)

    # Capture absolute turn count BEFORE any summarization truncation
    # This ensures turn-based logic (sticky window, reinforcement) survives summarization
    absolute_turn = sum(1 for msg in request.messages if msg.role == "user")

    # ========== CHARACTER RESOLUTION ==========
    # Check if characters are already resolved (dicts with 'data' field)
    # This happens on subsequent messages after first resolution
    if request.characters and isinstance(request.characters[0], dict) and "data" in request.characters[0]:
        # Characters already resolved - BUT reload NPCs from metadata
        # NPCs are frequently edited mid-chat and need fresh data
        if chat_data:
            metadata = chat_data.get("metadata", {}) or {}
            localnpcs = metadata.get("localnpcs", {}) or {}
            
            # Reload NPCs from metadata to get latest edits
            npcs_updated = 0
            for i, char in enumerate(request.characters):
                if char.get("is_npc") or char.get("npcId"):
                    entity_id = char.get("entity_id") or char.get("npcId")
                    if entity_id in localnpcs:
                        # Reload NPC from metadata with latest data
                        # Normalize to ensure all V2 fields (including extensions.gender)
                        npc_data = localnpcs[entity_id].get('data', char.get('data'))
                        npc_data_normalized = normalize_character_v2({'data': npc_data})['data']
                        request.characters[i] = {
                            'name': localnpcs[entity_id].get('name', char.get('name')),
                            'data': npc_data_normalized,
                            'entity_id': entity_id,
                            'is_npc': True,
                            'updated_at': localnpcs[entity_id].get('updated_at', 0)
                        }
                        npcs_updated += 1
            
            if npcs_updated > 0:
                print(f"[CONTEXT] Reloaded {npcs_updated} NPCs from metadata (mid-chat edits detected)")

            # Also reload global characters that may have been edited (new chat or outdated cache)
            global_chars_updated = 0
            for i, char in enumerate(request.characters):
                if not char.get("is_npc") and not char.get("npcId"):
                    # Global character - check if DB has newer version
                    filename = char.get("_filename") or char.get("entity_id")
                    if filename:
                        db_updated = db_get_character_updated_at(filename) or 0
                        char_updated = char.get("updated_at", 0)
                        # Reload if: 1) new chat (no chat_data), or 2) DB has newer version
                        if not chat_data or db_updated > char_updated:
                            fresh_char = db_get_character(filename)
                            if fresh_char:
                                request.characters[i] = {
                                    'name': get_character_name(fresh_char),
                                    'data': fresh_char.get('data', {}),
                                    'entity_id': filename,
                                    '_filename': filename,
                                    'is_npc': False,
                                    'updated_at': fresh_char.get('updated_at', 0)
                                }
                                global_chars_updated += 1
                                if not chat_data:
                                    print(f"[CONTEXT] Reloaded global character {filename} from DB (new chat)")
                                else:
                                    print(f"[CONTEXT] Reloaded global character {filename} from DB (edited since cache)")

            if global_chars_updated > 0:
                print(f"[CONTEXT] Reloaded {global_chars_updated} global characters from DB")
    else:
        # Resolve character references (global + NPCs)
        if chat_data:
            metadata = chat_data.get("metadata", {}) or {}
            localnpcs = metadata.get("localnpcs", {}) or {}
            active_chars = chat_data.get("activeCharacters", [])

            # Resolve character references to full objects
            resolved_characters = load_character_profiles(active_chars, localnpcs)
            request.characters = resolved_characters


    # === CHARACTER FIRST TURNS ===
    # Load character first turn numbers from chat metadata
    character_first_turns = {}
    character_full_card_turns = {}
    if chat_data and "metadata" in chat_data:
        metadata_raw = chat_data.get("metadata", {})
        metadata_obj = metadata_raw if isinstance(metadata_raw, dict) else {}
        character_first_turns = metadata_obj.get("characterFirstTurns", {})
        character_full_card_turns = metadata_obj.get("characterFullCardTurns", {})

    # === PHASE 2: CAST CHANGE DETECTION ===
    # Extract local NPCs and previous metadata from chat_data
    local_npcs = {}
    previous_metadata = {}
    if chat_data and "metadata" in chat_data:
        metadata_raw = chat_data.get("metadata", {}) or {}
        previous_metadata = metadata_raw
        local_npcs = metadata_raw.get("localnpcs", {}) or {}
    
    # Detect cast change
    cast_changed, departed, arrived, cast_metadata_updates = detect_cast_change(
        request.characters,  # Already resolved (global chars + NPCs)
        local_npcs,          # NPCs from metadata
        request.mode,
        previous_metadata
    )
    
    # Build entity_to_name mapping for SCENE UPDATE block
    entity_to_name = build_entity_name_mapping(request.characters, local_npcs)
    
    # Build SCENE UPDATE block if needed
    scene_update_block = ""
    if cast_changed:
        scene_update_block = build_scene_update_block(departed, arrived, entity_to_name)
        print(f"[CAST_CHANGE] Departed: {departed}, Arrived: {arrived}")

    # Extract canon law for summarization context (Phase 4)
    canon_law_entries = []
    if request.world_info:
        canon_law_entries = [
            entry.get('content', '')
            for entry in request.world_info.get('entries', {}).values()
            if entry.get('is_canon_law')
        ]

    # Check for summarization need
    max_ctx = request.settings.get("max_context", CONFIG['max_context'])
    threshold = request.settings.get("summarize_threshold", CONFIG['summarize_threshold'])
    
    current_request = request
    new_summary = request.summary or ""
    
    # Initial token check
    prompt = construct_prompt(current_request, character_first_turns, absolute_turn=absolute_turn, scene_update_block=scene_update_block, character_full_card_turns=character_full_card_turns)
    tokens = await get_token_count(prompt)
    
    # Phase 4: Summarization now handled asynchronously after response
    # - Cast-change: trigger_cast_change_summarization() (simple, one capsule)
    # - Threshold: trigger_threshold_summarization() (scene grouping, multiple capsules)
    # Both run as asyncio.create_task() - non-blocking, best-effort

    # === RELATIONSHIP ANALYSIS (Step 5 of relationship tracker) ===
    # Trigger relationship analysis at summarization boundary
    # Pass full character objects (not just names) for entity ID extraction
    user_name = request.settings.get("user_name", "")

    await analyze_and_update_relationships(
        chat_id=current_request.chat_id,
        messages=current_request.messages,
        character_objs=request.characters,
        user_name=user_name
    )
    
    # === SAVE METADATA UPDATES TO CHAT ===
    # Save updated character first turn numbers and cast tracking to chat metadata
    if request.chat_id and (character_first_turns or character_full_card_turns or cast_metadata_updates):
        try:
            chat = db_get_chat(request.chat_id)
            if not chat:
                # Chat doesn't exist yet - create minimal record
                # This happens on first message of a new chat
                chat = {
                    "messages": [],
                    "metadata": {},
                    "summary": ""
                }
                db_save_chat(request.chat_id, chat)
            
            # Get or initialize metadata
            metadata = chat.get("metadata", {}) or {}
            
            # Phase 2: Save cast change tracking
            if cast_metadata_updates:
                metadata.update(cast_metadata_updates)

            # Phase 4: Track cast change turns for scene boundary detection
            if cast_changed and departed:
                current_turn_num = absolute_turn

                if "cast_change_turns" not in metadata:
                    metadata["cast_change_turns"] = []

                if current_turn_num not in metadata["cast_change_turns"]:
                    metadata["cast_change_turns"].append(current_turn_num)
                    metadata["cast_change_turns"].sort()

            # Save character first turns and full card turns
            if character_first_turns:
                metadata["characterFirstTurns"] = character_first_turns
            if character_full_card_turns:
                metadata["characterFullCardTurns"] = character_full_card_turns
            
            # Batch save all metadata updates
            chat["metadata"] = metadata
            db_save_chat(request.chat_id, chat)
            
            if cast_metadata_updates:
                print(f"[CAST_CHANGE] Saved metadata: previous_active_cast={metadata.get('previous_active_cast')}, previous_focus={metadata.get('previous_focus_character')}")
        except Exception as e:
            print(f"[METADATA] ERROR: Failed to save metadata updates: {e}")

    print(f"Generated Prompt ({tokens} tokens):\n{prompt}")
    
    # Extract settings for generation
    temp = request.settings.get("temperature", 0.7)
    max_len = request.settings.get("max_length", 250)
    mode = request.mode or "narrator"

    # Calculate stop sequences (mode-aware)
    stops = ["User:", "\nUser", "###", "\n["]
    
    is_group_chat = len(request.characters) >= 2
    
    if is_group_chat:
        if mode.startswith("focus:"):
            # Focused mode: stop on other character names (not the focused one)
            focus_char = mode.split(":", 1)[1]
            for char in request.characters:
                name = char.get("data", {}).get("name")
                if name and name != focus_char:
                    stops.append(f"{name}:")
            # Also stop on Narrator since we're in first-person mode
            stops.append("Narrator:")
        else:
            # Narrator mode: don't stop on character names - let all speak
            # Only stop on User to prevent the narrator from writing user's lines
            pass
    else:
        # Single character or no characters
        for char in request.characters:
            name = char.get("data", {}).get("name")
            if name: stops.append(f"{name}:")
        if not request.characters: 
            stops.append("Narrator:")

    # Define the LLM operation to be managed
    async def llm_operation():
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONFIG['kobold_url']}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_length": max_len,
                    "temperature": temp,
                    "stop_sequence": stops
                },
                timeout=60.0
            )
            data = response.json()
            
            # Clean any reinforcement markers that may have leaked into the response
            if "results" in data and len(data["results"]) > 0:
                data["results"][0]["text"] = clean_llm_response(data["results"][0]["text"])
            
            # Wrap response to include potential state updates (summary/truncated messages)
            data["_updated_state"] = {
                "messages": [m.dict() for m in current_request.messages],
                "summary": current_request.summary
            }
            # Include token count for frontend to use in SD optimization
            data["_token_count"] = tokens
            return data
    
    # Route through resource manager with performance tracking
    if resource_manager.performance_mode_enabled:
        start_time = time.time()
        try:
            data = await resource_manager.execute_llm(llm_operation, op_type="heavy")
            duration = time.time() - start_time
            performance_tracker.record_llm(duration, context_tokens=tokens)
            
            # Generate hints based on performance
            hints = hint_engine.generate_hint(performance_tracker, tokens)
            if hints:
                data["_performance_hints"] = hints

            # Handle chat_id properly - only generate NEW if truly missing, not if invalid
            if not current_request.chat_id:
                # No chat_id at all - generate new one
                current_request.chat_id = f"new_chat_{int(time.time())}"
            else:
                # Check if chat_id exists in database
                try:
                    existing_chat = db_get_chat(current_request.chat_id)
                    if existing_chat is None:
                        # chat_id was provided but doesn't exist in DB
                        # This can happen after browser refresh with stale ID
                        # Generate a new ID and log it
                        new_id = f"new_chat_{int(time.time())}"
                        print(f"Chat ID {current_request.chat_id} not found in DB, generating new: {new_id}")
                        current_request.chat_id = new_id
                except Exception as e:
                    print(f"Error checking chat existence: {e}")
                    # If check fails, generate new ID to be safe
                    current_request.chat_id = f"new_chat_{int(time.time())}"
            
            # Include chat_id in response (frontend will handle autosave after adding AI message)
            data["_chat_id"] = current_request.chat_id
            data["_updated_state"] = {
                "messages": [m.dict() for m in current_request.messages],
                "summary": current_request.summary
            }

            # Phase 4: ASYNC SUMMARIZATION (after response sent, before return)
            # Both triggers independent - can run same turn

            # 1. Cast-change forced summarization
            if cast_changed and departed:
                asyncio.create_task(trigger_cast_change_summarization(
                    chat_id=current_request.chat_id,
                    departed_characters=departed,
                    entity_to_name=entity_to_name,
                    canon_law_entries=canon_law_entries
                ))

            # 2. Threshold-based summarization
            max_context = request.settings.get('max_context', CONFIG['max_context'])
            total_tokens = await get_token_count(prompt) + await get_token_count(data.get("_updated_state", {}).get("summary", ""))
            if total_tokens >= (max_context * 0.80):
                data["_summarization_triggered"] = True
                asyncio.create_task(trigger_threshold_summarization(
                    chat_id=current_request.chat_id,
                    canon_law_entries=canon_law_entries
                ))

            return data
        except Exception as e:
            duration = time.time() - start_time
            performance_tracker.record_llm(duration, context_tokens=tokens)
            return {"error": str(e)}
    else:
        # Direct call when performance mode is disabled
        try:
            response = await llm_operation()

            # Phase 4: ASYNC SUMMARIZATION (after response sent, before return)
            # Both triggers independent - can run same turn

            # 1. Cast-change forced summarization
            if cast_changed and departed:
                asyncio.create_task(trigger_cast_change_summarization(
                    chat_id=current_request.chat_id,
                    departed_characters=departed,
                    entity_to_name=entity_to_name,
                    canon_law_entries=canon_law_entries
                ))

            # 2. Threshold-based summarization
            max_context = request.settings.get('max_context', CONFIG['max_context'])
            total_tokens = await get_token_count(prompt) + await get_token_count(response.get("results", [{}])[0].get("text", ""))
            if total_tokens >= (max_context * 0.80):
                response["_summarization_triggered"] = True
                asyncio.create_task(trigger_threshold_summarization(
                    chat_id=current_request.chat_id,
                    canon_law_entries=canon_law_entries
                ))

            return response
        except Exception as e:
            return {"error": str(e)}

@app.post("/api/extra/tokencount")
async def proxy_tokencount(request: dict):
    count = await get_token_count(request.get("prompt", ""))
    return {"count": count}

def select_sd_preset(context_tokens: int) -> dict:
    """Select appropriate SD preset based on context size"""
    if context_tokens >= SD_PRESETS["emergency"]["threshold"]:
        return SD_PRESETS["emergency"]
    elif context_tokens >= SD_PRESETS["light"]["threshold"]:
        return SD_PRESETS["light"]
    else:
        return SD_PRESETS["normal"]

@app.post("/api/generate-image")
async def generate_image(params: SDParams):
    processed_prompt = params.prompt
    context_tokens = params.context_tokens or 0
    
    # Identify bracketed character names [Name]
    bracketed_names = re.findall(r"\[(.*?)\]", params.prompt)
    if bracketed_names:
        # Load all characters from database (primary source)
        all_chars = db_get_all_characters()
        
        # Load NPCs from chat metadata if chat_id provided
        npcs = {}
        if params.chat_id:
            chat_data = db_get_chat(params.chat_id)
            if chat_data:
                npcs = chat_data.get('metadata', {}).get('localnpcs', {})
        
        for name in bracketed_names:
            danbooru_tag = None
            
            # First, try to match global characters
            matched_char = next((c for c in all_chars if c.get("data", {}).get("name", "").lower() == name.lower()), None)
            if matched_char:
                danbooru_tag = matched_char.get("data", {}).get("extensions", {}).get("danbooru_tag", "")
            
            # If no global character matched, try NPCs
            if not danbooru_tag and npcs:
                for npc_id, npc_data in npcs.items():
                    npc_name = npc_data.get('name', '')
                    if npc_name.lower() == name.lower():
                        danbooru_tag = npc_data.get('data', {}).get('extensions', {}).get('danbooru_tag', '')
                        break
            
            # Replace if we found a danbooru_tag
            if danbooru_tag:
                processed_prompt = processed_prompt.replace(f"[{name}]", danbooru_tag)
                print(f"[MANUAL SD] Replaced [{name}] with Danbooru tag: {danbooru_tag}")

    # Define the SD operation to be managed
    async def sd_operation():
        # Apply context-aware preset if performance mode is enabled
        if resource_manager.performance_mode_enabled:
            preset = select_sd_preset(context_tokens)
            # Use preset values only if user hasn't changed from defaults
            final_steps = preset["steps"] if params.steps == 20 else params.steps
            final_width = preset["width"] if params.width == 512 else params.width
            final_height = preset["height"] if params.height == 512 else params.height
        else:
            final_steps = params.steps
            final_width = params.width
            final_height = params.height
        
        async with httpx.AsyncClient() as client:
            payload = {
                "prompt": processed_prompt,
                "negative_prompt": params.negative_prompt,
                "steps": final_steps,
                "cfg_scale": params.cfg_scale,
                "width": final_width,
                "height": final_height,
                "sampler_name": params.sampler_name,
                "scheduler": params.scheduler
            }
            print(f"SD Prompt Construction (preset applied: {preset if resource_manager.performance_mode_enabled else 'user'}): {processed_prompt}")
            
            response = await client.post(
                f"{CONFIG['sd_url']}/sdapi/v1/txt2img",
                json=payload,
                timeout=120.0
            )
            data = response.json()
            image_base64 = data["images"][0]
            
            filename = f"sd_{int(time.time())}.png"
            file_path = os.path.join(IMAGE_DIR, filename)
            
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(image_base64))

            # Store image metadata
            store_image_metadata(filename, params)

            return {
                "url": f"/images/{filename}",
                "filename": filename
            }
    
    # Route through resource manager with performance tracking
    if resource_manager.performance_mode_enabled:
        start_time = time.time()
        try:
            result = await resource_manager.execute_sd(sd_operation)
            duration = time.time() - start_time
            performance_tracker.record_sd(duration, context_tokens=context_tokens)
            
            # Generate hints based on SD performance
            hints = hint_engine.generate_hint(performance_tracker, context_tokens, duration)
            if hints:
                result["_performance_hints"] = hints
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            performance_tracker.record_sd(duration, context_tokens=context_tokens)
            return {"error": str(e)}
    else:
        # Direct call when performance mode is disabled
        try:
            return await sd_operation()
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# SNAPSHOT ENDPOINTS (v1.9.0)
# ============================================================================

class SnapshotRequest(BaseModel):
    chat_id: str
    width: Optional[int] = None  # Image width
    height: Optional[int] = None  # Image height
    mode: Optional[str] = None  # Mode for primary character selection

def get_primary_character_name(selected_mode: Optional[str], active_characters: List[str], character_names: List[str]) -> Optional[str]:
    """
    Determine primary character name based on selected mode.

    Args:
        selected_mode: Mode from frontend ("auto", "narrator", "focus:Alice", etc.)
        active_characters: List of character references (filenames or npc_ids)
        character_names: List of character names extracted from data

    Returns:
        Character name for primary character, or None for narrator mode
    """
    if not selected_mode or not active_characters:
        return None

    # Narrator mode - no primary character
    if selected_mode == "narrator":
        return None

    # Specific character mode (e.g., "focus:Alice")
    if selected_mode.startswith("focus:"):
        focus_char_name = selected_mode.split(":", 1)[1].strip()
        # Validate that this character is in active list
        if focus_char_name in character_names:
            return focus_char_name
        return None

    # Auto mode - use first active character
    if selected_mode == "auto" and character_names:
        return character_names[0]

    return None

def filter_characters_by_mode(chars_data: List[Dict], mode: Optional[str], 
                             primary_char_name: Optional[str], 
                             for_counting: bool = False) -> List[Dict]:
    """
    Filter characters for danbooru tags or counting based on mode.
    
    Args:
        chars_data: Full list of character dicts from get_active_characters_data()
        mode: Snapshot mode ('auto', 'narrator', 'focus:Name', None)
        primary_char_name: Name of primary character (from get_primary_character_name())
        for_counting: If True, use counting logic (Narrator includes all chars)
    
    Returns:
        Filtered list of character dicts based on mode
    """
    if not chars_data:
        return []
    
    if not mode or mode == "auto":
        return chars_data
    
    if mode == "narrator":
        return chars_data if for_counting else []
    
    if mode.startswith("focus:") and primary_char_name:
        return [c for c in chars_data if c.get('name') == primary_char_name]
    
    return chars_data

async def infer_character_counts_from_conversation(
    messages: List[Dict],
    http_client,
    config,
    chat_settings: Dict
) -> str:
    """
    Infer character counts from conversation when no active characters.
    
    Args:
        messages: Chat messages
        http_client: httpx.AsyncClient for LLM requests
        config: Application config dict with kobold_url
        chat_settings: Chat settings dict with user gender/enablement
    
    Returns:
        Count tags string (e.g., "2girls, 1boy") or "" if inference fails
    """
    if not http_client or not config.get('kobold_url'):
        print("[SNAPSHOT] LLM unavailable, cannot infer character counts")
        return ""
    
    recent_messages = messages[-5:] if len(messages) > 5 else messages
    conversation_text = '\n'.join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in recent_messages
    ])
    
    prompt = f"""Analyze this conversation and determine how many characters are present in the scene.

Count characters by gender:
- Girls/Women/Females (use "Xgirls" tag, e.g., "1girl", "2girls", "3girls")
- Boys/Men/Males (use "Xboys" tag, e.g., "1boy", "2boys", "3boys")
- Others/Non-binary (use "Xothers" or appropriate tag)

IMPORTANT: Do NOT include the user in your count. The user will be counted separately.

Reply with ONLY count tags in format: "Xgirls, Xboys" or similar.
If no characters mentioned, reply with "solo" or "none".

Conversation:
{conversation_text}

Answer:"""

    try:
        response = await http_client.post(
            f"{config['kobold_url']}/api/v1/generate",
            json={
                "prompt": prompt,
                "max_length": 30,
                "temperature": 0.2,
                "top_p": 0.9
            },
            timeout=15.0
        )
        
        result = response.json().get('results', [{}])[0].get('text', '').strip()
        print(f"[SNAPSHOT] LLM inferred counts: {result}")
        
        include_user = chat_settings.get('include_user_in_snapshots', False)
        if include_user:
            user_gender = chat_settings.get('user_gender', '').lower()
            if user_gender in ['female', 'male', 'other']:
                counts = {'female': 0, 'male': 0, 'other': 0}
                for tag in result.split(','):
                    tag = tag.strip().lower()
                    if 'girl' in tag:
                        counts['female'] += int(''.join(filter(str.isdigit, tag)) or 1)
                    elif 'boy' in tag:
                        counts['male'] += int(''.join(filter(str.isdigit, tag)) or 1)
                    elif 'other' in tag or 'multiple' in tag:
                        counts['other'] += 1
                
                counts[user_gender] += 1
                
                count_tags = []
                if counts['female'] == 1:
                    count_tags.append('1girl')
                elif counts['female'] > 1:
                    count_tags.append(f'{counts["female"]}girls')
                if counts['male'] == 1:
                    count_tags.append('1boy')
                elif counts['male'] > 1:
                    count_tags.append(f'{counts["male"]}boys')
                if counts['other'] == 1 and not count_tags:
                    count_tags.append('solo')
                elif counts['other'] > 0:
                    count_tags.append('multiple')
                
                result = ', '.join(count_tags)
                print(f"[SNAPSHOT] Counts with user added: {result}")
        
        return result
        
    except Exception as e:
        print(f"[SNAPSHOT] Failed to infer character counts: {e}")
        return ""

def get_active_characters_data(chat: Dict, max_chars: int = 3, include_visual_canon: bool = False) -> List[Dict]:
    """Extract gender and danbooru tags for all active characters in chat.
    
    Args:
        chat: Chat data dict with activeCharacters and metadata
        max_chars: Maximum number of characters to process (default: 3)
        include_visual_canon: If True, load visual_canon bindings (NEW)
        
    Returns:
        List of dicts with 'name', 'gender', 'danbooru_tag', 'description', 'personality' + optional visual_canon fields for each character
    """
    chars_data = []
    active_chars = chat.get("activeCharacters", [])
    localnpcs = chat.get("metadata", {}).get("localnpcs", {})
    
    for char_ref in active_chars[:max_chars]:
        if char_ref.startswith('npc_'):
            # Load NPC from metadata
            npc_data = localnpcs.get(char_ref, {})
            if not npc_data:
                continue
            data = npc_data.get('data', {})
            extensions = data.get('extensions', {})
            
            char_dict = {
                'name': npc_data.get('name', 'Unknown'),
                'description': data.get('description', ''),
                'personality': data.get('personality', ''),
                'gender': extensions.get('gender', ''),
                'danbooru_tag': extensions.get('danbooru_tag', '')
            }
            
            # Add visual_canon data if requested (NEW)
            if include_visual_canon:
                from app.database import db_get_npc_visual_canon
                visual_canon = db_get_npc_visual_canon(chat['id'], char_ref)
                if visual_canon:
                    char_dict['visual_canon_id'] = visual_canon['visual_canon_id']
                    char_dict['visual_canon_name'] = visual_canon['visual_canon_name']
                    char_dict['visual_canon_tags'] = visual_canon['visual_canon_tags']
            
            chars_data.append(char_dict)
        else:
            # Load global character from database
            from app.database import db_get_character
            char_data = db_get_character(char_ref)
            if not char_data:
                continue
            data = char_data.get('data', {})
            extensions = data.get('extensions', {})
            
            char_dict = {
                'name': get_character_name(char_data),
                'description': data.get('description', ''),
                'personality': data.get('personality', ''),
                'gender': extensions.get('gender', ''),
                'danbooru_tag': extensions.get('danbooru_tag', '')
            }
            
            # Add visual_canon data if requested (NEW)
            if include_visual_canon:
                from app.database import db_get_character_visual_canon
                visual_canon = db_get_character_visual_canon(char_ref)
                if visual_canon:
                    char_dict['visual_canon_id'] = visual_canon['visual_canon_id']
                    char_dict['visual_canon_name'] = visual_canon['visual_canon_name']
                    char_dict['visual_canon_tags'] = visual_canon['visual_canon_tags']
            
            chars_data.append(char_dict)
    
    return chars_data

def auto_count_characters_by_gender(chars_data: List[Dict], user_gender: Optional[str] = None, include_user: bool = False) -> str:
    """Generate danbooru count tags based on character genders.
    
    Args:
        chars_data: List of character dicts with 'gender' field
        user_gender: Optional user gender ('female', 'male', 'other')
        include_user: Whether to include user in the count
        
    Returns:
        Comma-separated count tags (e.g., "2girls, 1boy") or None if no genders set
    """
    print(f"[SNAPSHOT DEBUG] auto_count called with: {len(chars_data)} chars, "
          f"user_gender={user_gender}, include_user={include_user}")
    
    gender_counts = {'female': 0, 'male': 0, 'other': 0}
    
    for char in chars_data:
        gender = char.get('gender', '').lower()
        if gender in gender_counts:
            gender_counts[gender] += 1
    
    # Include user in count if enabled
    if include_user and user_gender:
        user_gender = user_gender.lower()
        if user_gender in gender_counts:
            gender_counts[user_gender] += 1
            print(f"[SNAPSHOT DEBUG] User count added: 1{user_gender}")
            print(f"[SNAPSHOT DEBUG] All counts after adding user: {dict(gender_counts)}")
    
    count_tags = []
    if gender_counts['female'] == 1:
        count_tags.append('1girl')
    elif gender_counts['female'] > 1:
        count_tags.append(f'{gender_counts["female"]}girls')
    
    if gender_counts['male'] == 1:
        count_tags.append('1boy')
    elif gender_counts['male'] > 1:
        count_tags.append(f'{gender_counts["male"]}boys')
    
    if gender_counts['other'] > 0:
        # For 'other' gender, use 'solo' if single, 'multiple' otherwise
        if len(count_tags) == 0 and gender_counts['other'] == 1:
            count_tags.append('solo')
        elif gender_counts['other'] > 0:
            count_tags.append('multiple')
    
    return ', '.join(count_tags) if count_tags else ''

def parse_personality_plist(personality: str) -> str:
    """Extract personality traits from PList format.
    
    Args:
        personality: PList-formatted personality string like [Name's Personality= "trait1", "trait2"]
        
    Returns:
        Clean comma-separated string of traits
    """
    if not personality:
        return ""
    
    # Pattern: [Name's Personality= "trait1", "trait2", ...]
    match = re.search(r'\[.+?\'s Personality\s*=\s*(.*?)\]', personality, re.DOTALL | re.IGNORECASE)
    if match:
        plist_content = match.group(1).strip()
        traits = re.findall(r'["\']([^"\']+)["\']', plist_content)
        if traits:
            return ', '.join(traits)
    
    # Fallback: return as-is if no PList format detected
    return personality.strip()

def aggregate_danbooru_tags(chars_data: List[Dict], override_count: Optional[str] = None) -> str:
    """Aggregate danbooru tags from all characters with gender count override.

    Args:
        chars_data: List of character dicts with 'danbooru_tag' field
        override_count: Count tag(s) from auto-counting (overrides danbooru_tag count)

    Returns:
        Comma-separated danbooru tags for Block 1
    """
    all_tags = []
    total_filtered = 0

    for char in chars_data:
        danbooru_tag = char.get('danbooru_tag', '').strip()
        if not danbooru_tag:
            continue

        # Parse PList format to extract clean traits
        traits = parse_physical_body_plist(danbooru_tag)
        if traits:
            # Convert extracted traits back to comma-separated string
            danbooru_tag = ', '.join(traits)

        # Parse comma-separated tags
        tags = [t.strip() for t in danbooru_tag.split(',') if t.strip()]

        # Filter out count tags if override provided (gender wins)
        # Use EXACT match only to avoid filtering tags containing count patterns
        if override_count:
            # Common count patterns to filter (exact matches only)
            count_patterns = {'1girl', '1boy', '2girls', '2boys', '3girls', '3boys',
                           '4girls', '4boys', '5girls', '5boys', '6girls', '6boys',
                           '7girls', '7boys', '8girls', '8boys',
                           'solo', 'duo', 'trio', 'quartet', 'group', 'crowd',
                           'multiple girls', 'multiple boys', 'multiple'}
            filtered_tags = []
            for tag in tags:
                if tag.lower() in count_patterns:
                    total_filtered += 1
                    print(f"[AGGREGATE] Filtered count tag from {char.get('name', 'Unknown')}: {tag}")
                else:
                    filtered_tags.append(tag)
            tags = filtered_tags

        all_tags.extend(tags)

    if total_filtered > 0:
        print(f"[AGGREGATE] Total count tags filtered: {total_filtered}, remaining: {len(all_tags)}")

    # Limit to first 30 tags (supports 3 characters × 10 tags)
    return ', '.join(all_tags[:30])

@app.post("/api/chat/snapshot")
async def generate_chat_snapshot(request: SnapshotRequest):
    """Generate SD image from current chat scene."""
    try:
        # Load chat data
        chat = db_get_chat(request.chat_id)
        if not chat:
            return {"error": "Chat not found"}

        messages = chat.get('messages', [])
        
        # RELOAD chat from database to get latest snapshot settings
        # Settings may have been updated via /api/chats/{id}/snapshot-settings endpoint
        chat = db_get_chat(request.chat_id)
        if chat:
            # Get snapshot settings from dedicated metadata key (now using fresh chat data)
            chat_settings = chat.get('metadata', {}).get('snapshot_settings', {})
            print(f"[SNAPSHOT DEBUG] Chat reloaded from database, metadata keys: {list(chat.get('metadata', {}).keys())}")
            print(f"[SNAPSHOT DEBUG] Snapshot settings loaded: {chat_settings}")
        else:
            chat_settings = {}
            print(f"[SNAPSHOT DEBUG] Chat reload failed, using empty settings")

        # Calculate context tokens for chat messages (enables context-aware presets)
        chat_text = "\n".join([msg.get('content', '') for msg in messages])
        context_tokens = await get_token_count(chat_text)

        # Extract active characters' gender and danbooru tags WITH visual canon data (NEW)
        chars_data = get_active_characters_data(chat, max_chars=3, include_visual_canon=True)

        # Determine primary character from mode
        character_names = [c.get('name', '') for c in chars_data if c.get('name')]
        primary_char_name = get_primary_character_name(
            request.mode,
            chat.get("activeCharacters", []),
            character_names
        )

        # Load primary character card if specific character selected
        primary_char_card = None
        if primary_char_name:
            for char_data in chars_data:
                if char_data.get('name') == primary_char_name:
                    # Parse PList-formatted description and personality
                    description = char_data.get('description', '')
                    personality = char_data.get('personality', '')
                    
                    # Extract traits from PList format
                    desc_traits = parse_physical_body_plist(description)
                    person_traits = parse_personality_plist(personality)
                    
                    # Build clean card
                    desc_text = ', '.join(desc_traits) if desc_traits else description
                    person_text = person_traits if person_traits else personality
                    
                    primary_char_card = desc_text + ' ' + person_text
                    
                    # Strip character names from description and personality to prevent LLM from copying them to JSON output
                    primary_char_card = snapshot_analyzer._strip_character_names(primary_char_card, [primary_char_name])
                    break

        print(f"[SNAPSHOT] Primary character: {primary_char_name}")

        # Build user data for snapshot if enabled
        user_data = None
        
        # Debug user settings
        print(f"[SNAPSHOT DEBUG] User settings loaded: include_user={chat_settings.get('include_user_in_snapshots', False)}, "
              f"user_gender='{chat_settings.get('user_gender', 'NOT_SET')}', "
              f"user_danbooru_tag='{chat_settings.get('user_danbooru_tag', 'NOT_SET')}'")
        
        include_user = chat_settings.get('include_user_in_snapshots', False)
        if include_user:
            user_data = {
                'gender': chat_settings.get('user_gender', ''),
                'danbooru_tag': chat_settings.get('user_danbooru_tag', ''),
                'include_user': True
            }
            print(f"[SNAPSHOT DEBUG] user_data created: {user_data}")
            user_data = {
                'gender': chat_settings.get('user_gender', ''),
                'danbooru_tag': chat_settings.get('user_danbooru_tag', ''),
                'include_user': True
            }
            print(f"[SNAPSHOT] Including user in snapshot: gender={user_data['gender']}")
        
        # Filter characters for danbooru tags and counting based on mode
        chars_for_tags = filter_characters_by_mode(chars_data, request.mode, primary_char_name, for_counting=False)
        chars_for_count = filter_characters_by_mode(chars_data, request.mode, primary_char_name, for_counting=True)
        print(f"[SNAPSHOT] Mode={request.mode}, chars_for_tags={len(chars_for_tags)}, chars_for_count={len(chars_for_count)}")
        
        # Auto-count characters by gender
        user_gender = user_data.get('gender') if user_data else None
        
        # Use LLM fallback if no active characters
        if len(chars_for_count) == 0 and len(chars_data) == 0:
            print("[SNAPSHOT] No active characters, inferring counts from conversation")
            count_tags = await infer_character_counts_from_conversation(
                messages, snapshot_http_client, CONFIG, chat_settings
            )
        else:
            count_tags = auto_count_characters_by_gender(chars_for_count, user_gender=user_gender, include_user=bool(user_data))
        
        print(f"[SNAPSHOT] Auto-counted: {count_tags}")
        
        # Aggregate danbooru tags from filtered characters only
        character_tags = aggregate_danbooru_tags(chars_for_tags, override_count=count_tags if count_tags else None)
        print(f"[SNAPSHOT] Aggregated character tags: {character_tags}")

        # Analyze scene using character-scoped JSON extraction
        scene_analysis = await snapshot_analyzer.analyze_scene(
            messages,
            request.chat_id,
            character_names=character_names,
            primary_character=primary_char_name,
            primary_character_card=primary_char_card
        )
        
        # Extract scene JSON for simplified prompt building
        scene_json = scene_analysis.get('scene_json', {})
        
        # Build user tags list from user_data if present
        user_tags = []
        if user_data and user_data.get('danbooru_tag'):
            user_tags = [t.strip() for t in user_data['danbooru_tag'].split(',') if t.strip()]
        
        # Build character tags list from aggregated tags
        char_tags = []
        if character_tags:
            char_tags = [t.strip() for t in character_tags.split(',') if t.strip()]
        
        # Build prompt using new simplified builder
        positive_prompt, negative_prompt = prompt_builder.build_simple_prompt(
            scene_json=scene_json,
            character_tags=char_tags,
            user_tags=user_tags,
            character_count_tags=count_tags
        )

        # Use size parameters from request, default to 512x512
        snapshot_width = request.width if request.width else 512
        snapshot_height = request.height if request.height else 512

        # Generate image using existing generate_image function
        sd_params = SDParams(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            steps=20,
            width=snapshot_width,
            height=snapshot_height,
            cfg_scale=7.0,
            sampler_name='Euler a',
            scheduler='Automatic',
            context_tokens=context_tokens
        )

        image_result = await generate_image(sd_params)

        if "error" in image_result:
            return {"error": image_result["error"]}

        # Save snapshot to chat metadata
        snapshot_data = {
            "timestamp": int(time.time()),
            "image_url": image_result["url"],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "scene_analysis": scene_analysis,
            "characters": [{"name": c["name"], "gender": c["gender"]} for c in chars_data]
        }

        # Update chat metadata with snapshot history
        chat_metadata = chat.get('metadata', {})
        if 'snapshot_history' not in chat_metadata:
            chat_metadata['snapshot_history'] = []
        chat_metadata['snapshot_history'].append(snapshot_data)
        chat['metadata'] = chat_metadata

        db_save_chat(request.chat_id, chat)

        return {
            "success": True,
            "image_url": image_result["url"],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "scene_analysis": scene_analysis
        }

    except Exception as e:
        logger.error(f"Snapshot generation failed: {e}")
        return {"error": str(e)}

@app.get("/api/chat/{chat_id}/snapshots")
async def get_snapshot_history(chat_id: str):
    """Retrieve snapshot history for a chat."""
    chat = db_get_chat(chat_id)
    if not chat:
        return {"error": "Chat not found"}

    snapshots = chat.get('metadata', {}).get('snapshot_history', [])
    return {"snapshots": snapshots}


class SnapshotSettingsRequest(BaseModel):
    """Request model for saving snapshot user settings."""
    include_user_in_snapshots: bool = False
    user_gender: str = ""
    user_danbooru_tag: str = ""


class SnapshotRegenerateRequest(BaseModel):
    """Request model for regenerating snapshot."""
    width: Optional[int] = None  # Image width
    height: Optional[int] = None  # Image height


@app.post("/api/chats/{chat_id}/snapshot-settings")
async def save_snapshot_settings(chat_id: str, request: SnapshotSettingsRequest):
    """
    Save user snapshot settings to chat metadata.
    
    These settings control whether the user is included in snapshots
    and their gender/danbooru tag configuration.
    """
    try:
        print(f"[SNAPSHOT] save_snapshot_settings called for chat {chat_id}")
        print(f"[SNAPSHOT] Request data: include_user={request.include_user_in_snapshots}, gender={request.user_gender}, tag={request.user_danbooru_tag}")
        
        chat = db_get_chat(chat_id)
        if not chat:
            print(f"[SNAPSHOT] Chat {chat_id} not found!")
            return {"error": "Chat not found"}
        
        print(f"[SNAPSHOT] Loaded chat {chat_id}, current metadata keys: {list(chat.get('metadata', {}).keys())}")
        
        # Get or create metadata
        metadata = chat.get('metadata', {})
        if metadata is None:
            metadata = {}
        
        # Save snapshot settings in dedicated key
        metadata['snapshot_settings'] = {
            'include_user_in_snapshots': request.include_user_in_snapshots,
            'user_gender': request.user_gender,
            'user_danbooru_tag': request.user_danbooru_tag
        }
        
        chat['metadata'] = metadata
        db_save_chat(chat_id, chat)
        
        print(f"[SNAPSHOT] Saved settings for chat {chat_id}: include_user={request.include_user_in_snapshots}, gender={request.user_gender}")
        
        return {
            "success": True,
            "settings": metadata['snapshot_settings']
        }
        
    except Exception as e:
        print(f"[SNAPSHOT] Error saving settings for chat {chat_id}: {e}")
        return {"error": str(e)}


class SnapshotFavoriteRequest(BaseModel):
    """Request model for adding snapshot to favorites."""
    chat_id: str
    image_filename: str
    prompt: str
    negative_prompt: str
    scene_analysis: Dict[str, Any]
    character_ref: Optional[str] = None
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    note: Optional[str] = None


@app.post("/api/chat/{chat_id}/snapshot/regenerate")
async def regenerate_snapshot(chat_id: str, request: SnapshotRegenerateRequest):
    """
    Regenerate snapshot with variation mode enabled.

    Uses variation mode scene analysis (no examples in LLM prompt) to generate
    alternative interpretations of current scene, while maintaining the same
    character tags, visual canon data, and user settings. Variation emerges
    from lack of example priming, not from temperature changes.
    """
    try:
        # Load chat
        chat = db_get_chat(chat_id)
        if not chat:
            return {"error": "Chat not found"}

        messages = chat.get('messages', [])

        # RELOAD chat from database to get latest snapshot settings
        # Settings may have been updated via /api/chats/{id}/snapshot-settings endpoint
        chat = db_get_chat(chat_id)
        if chat:
            chat_settings = chat.get('metadata', {}).get('snapshot_settings', {})
            print(f"[SNAPSHOT VARIATION] Chat reloaded from database, snapshot settings: {chat_settings}")
        else:
            chat_settings = {}
            print(f"[SNAPSHOT VARIATION] Chat reload failed, using empty settings")

        # Build user data for snapshot if enabled
        user_data = None
        include_user = chat_settings.get('include_user_in_snapshots', False)
        if include_user:
            user_data = {
                'gender': chat_settings.get('user_gender', ''),
                'danbooru_tag': chat_settings.get('user_danbooru_tag', ''),
                'include_user': True
            }
            print(f"[SNAPSHOT VARIATION] Including user in variation: {user_data}")

        # Extract active characters' gender and danbooru tags WITH visual canon data (NEW)
        chars_data = get_active_characters_data(chat, max_chars=3, include_visual_canon=True)
        
        # Auto-count characters by gender
        user_gender = user_data.get('gender') if user_data else None
        count_tags = auto_count_characters_by_gender(chars_data, user_gender=user_gender, include_user=bool(user_data))
        print(f"[SNAPSHOT VARIATION] Auto-counted: {count_tags}")

        # Aggregate danbooru tags with gender override
        character_tags = aggregate_danbooru_tags(chars_data, override_count=count_tags if count_tags else None)
        print(f"[SNAPSHOT VARIATION] Aggregated tags: {character_tags}")
        
        # Extract character names for name-stripping (prevents name leakage to LLM)
        character_names = [c.get('name', '') for c in chars_data if c.get('name')]

        # Determine primary character for variation mode (default to first active character or narrator if no focus)
        primary_char_name = None
        primary_char_card = None
        if character_names:
            primary_char_name = character_names[0]  # Default to first character

            for char_data in chars_data:
                if char_data.get('name') == primary_char_name:
                    # Parse PList-formatted description and personality
                    description = char_data.get('description', '')
                    personality = char_data.get('personality', '')

                    # Extract traits from PList format
                    desc_traits = parse_physical_body_plist(description)
                    person_traits = parse_personality_plist(personality)

                    # Build clean card
                    desc_text = ', '.join(desc_traits) if desc_traits else description
                    person_text = person_traits if person_traits else personality

                    primary_char_card = desc_text + ' ' + person_text

                    # Strip character names from description and personality
                    primary_char_card = snapshot_analyzer._strip_character_names(primary_char_card, [primary_char_name])
                    break

        print(f"[SNAPSHOT VARIATION] Primary character: {primary_char_name}")

        # Analyze scene using VARIATION mode (creative interpretation, temperature 0.8)
        scene_analysis = await snapshot_analyzer.analyze_scene_variation(
            messages, chat_id, character_names=character_names,
            primary_character=primary_char_name,
            primary_character_card=primary_char_card
        )
        
        # Extract scene JSON for simplified prompt building
        scene_json = scene_analysis.get('scene_json', {})
        
        # Build character tags list from aggregated tags
        char_tags = []
        if character_tags:
            char_tags = [t.strip() for t in character_tags.split(',') if t.strip()]

        # Build user tags list from user_data if present
        user_tags = []
        if user_data and user_data.get('danbooru_tag'):
            user_tags = [t.strip() for t in user_data['danbooru_tag'].split(',') if t.strip()]

        # Build prompt using simplified builder
        positive_prompt, negative_prompt = prompt_builder.build_simple_prompt(
            scene_json=scene_json,
            character_tags=char_tags,
            user_tags=user_tags,
            character_count_tags=count_tags
        )

        # Generate image
        sd_params = SDParams(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            steps=20,
            width=request.width if request.width else 512,
            height=request.height if request.height else 512,
            cfg_scale=7.0,
            sampler_name='Euler a',
            scheduler='Automatic',
            context_tokens=0
        )

        image_result = await generate_image(sd_params)

        if "error" in image_result:
            return {"error": image_result["error"]}

        # Save to snapshot history
        snapshot_data = {
            "timestamp": int(time.time()),
            "image_url": image_result["url"],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "scene_analysis": scene_analysis,
            "characters": [{"name": c["name"], "gender": c["gender"]} for c in chars_data],
            "mode": "variation"
        }

        chat_metadata = chat.get('metadata', {})
        if 'snapshot_history' not in chat_metadata:
            chat_metadata['snapshot_history'] = []
        chat_metadata['snapshot_history'].append(snapshot_data)
        chat['metadata'] = chat_metadata

        db_save_chat(chat_id, chat)

        return {
            "success": True,
            "image_url": image_result["url"],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "scene_analysis": scene_analysis,
            "mode": "variation"
        }

    except Exception as e:
        logger.error(f"Snapshot regeneration failed: {e}")
        return {"error": str(e)}


@app.post("/api/chat/{chat_id}/snapshot/favorite")
async def add_snapshot_favorite(chat_id: str, request: SnapshotFavoriteRequest):
    """
    Mark snapshot as favorite.
    """
    try:
        # Extract tags from prompt
        tags = [tag.strip() for tag in request.prompt.split(',') if tag.strip()]

        # Add to favorites table (source_type='snapshot')
        fav_id = db_add_snapshot_favorite({
            'chat_id': chat_id,
            'image_filename': request.image_filename,
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'scene_type': request.scene_analysis.get('scene_type', 'other'),
            'setting': request.scene_analysis.get('setting', ''),
            'mood': request.scene_analysis.get('mood', ''),
            'character_ref': request.character_ref,
            'tags': tags,
            'steps': request.steps,
            'cfg_scale': request.cfg_scale,
            'width': request.width,
            'height': request.height,
            'source_type': 'snapshot',
            'note': request.note
        })

        logger.info(f"[SNAPSHOT] Added snapshot favorite: {request.image_filename}, saved {len(tags)} tags")

        return {
            "success": True,
            "favorite_id": fav_id,
            "tags_saved": len(tags),
            "source_type": "snapshot"
        }

    except Exception as e:
        logger.error(f"Failed to add snapshot favorite: {e}")
        return {"error": str(e)}


class ManualFavoriteRequest(BaseModel):
    """Request model for adding manual mode generation as favorite."""
    image_filename: str
    prompt: str
    negative_prompt: str
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    sampler: str = 'Euler a'
    scheduler: str = 'Automatic'
    note: Optional[str] = None
    chat_id: Optional[str] = None  # Optional: associate with current chat for jump-to-source


@app.post("/api/manual-generation/favorite")
async def add_manual_favorite(request: ManualFavoriteRequest):
    """
    Add manual mode generation to favorites.
    """
    try:
        # Extract tags from prompt (comma-separated)
        tags = [tag.strip() for tag in request.prompt.split(',') if tag.strip()]

        # Save to favorites (source_type='manual')
        fav_id = db_add_snapshot_favorite({
            'chat_id': request.chat_id,  # Optional: associate with current chat for jump-to-source
            'image_filename': request.image_filename,
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'scene_type': None,  # Manual mode has no LLM analysis
            'setting': None,  # Manual mode has no LLM analysis
            'mood': None,  # Manual mode has no LLM analysis
            'character_ref': None,  # Manual mode has no character context
            'tags': tags,
            'steps': request.steps,
            'cfg_scale': request.cfg_scale,
            'width': request.width,
            'height': request.height,
            'source_type': 'manual',
            'note': request.note
        })

        logger.info(f"[SNAPSHOT] Added manual favorite: {request.image_filename}, saved {len(tags)} tags")

        return {
            "success": True,
            "favorite_id": fav_id,
            "tags_saved": len(tags),
            "source_type": "manual"
        }

    except Exception as e:
        logger.error(f"Failed to add manual favorite: {e}")
        return {"error": str(e)}


@app.get("/api/snapshots/favorites")
async def get_all_favorites(scene_type: Optional[str] = None,
                       source_type: Optional[str] = None,
                       tags: Optional[str] = None,
                       limit: int = 50,
                       offset: int = 0):
    """
    Get all favorites, optionally filtered by scene type, source type, and tags.

    Returns list of favorite snapshot records.
    """
    try:
        # Parse tags parameter (comma-separated) to list of lowercase tags
        tag_list = None
        if tags:
            tag_list = [t.strip().lower() for t in tags.split(',') if t.strip()]

        favorites = db_get_favorites(limit=limit, scene_type=scene_type, source_type=source_type, tags=tag_list, offset=offset)

        # Parse JSON tags for each favorite
        for fav in favorites:
            if fav.get('tags'):
                try:
                    fav['tags'] = json.loads(fav['tags'])
                except:
                    fav['tags'] = []

        return {"favorites": favorites, "count": len(favorites)}

    except Exception as e:
        logger.error(f"Failed to get favorites: {e}")
        return {"error": str(e)}


@app.get("/api/snapshots/tags")
async def get_all_favorite_tags():
    """
    Get all unique tags from all favorites.

    Returns list of unique tags (alphabetically sorted).
    """
    try:
        tags = db_get_all_favorite_tags()
        return {"tags": tags}
    except Exception as e:
        logger.error(f"Failed to get favorite tags: {e}")
        return {"tags": []}


@app.get("/api/snapshots/tags/popular")
async def get_popular_favorite_tags(limit: int = 5):
    """
    Get the most popular tags from favorites.

    Returns list of {tag, count} objects sorted by count DESC.
    """
    try:
        tags = db_get_popular_favorite_tags(limit=limit)
        return [{"tag": tag_text, "count": count} for tag_text, count in tags]
    except Exception as e:
        logger.error(f"Failed to get popular favorite tags: {e}")
        return []


@app.delete("/api/snapshots/favorites/{favorite_id}")
async def delete_favorite(favorite_id: int):
    """
    Delete a favorite snapshot.

    Decrements tag frequencies for tags in deleted favorite.
    (Note: Simple implementation - doesn't decrement for now to avoid complexity)
    """
    try:
        # Get favorite before deletion (to decrement tags)
        # Note: This would require db_get_favorite_by_id function
        # For simplicity, we'll just delete and not decrement
        # (tag counts are approximate anyway)

        success = db_delete_favorite(favorite_id)

        if success:
            logger.info(f"[SNAPSHOT] Deleted favorite: {favorite_id}")
            return {"success": True}
        else:
            return {"error": "Favorite not found"}

    except Exception as e:
        logger.error(f"Failed to delete favorite: {e}")
        return {"error": str(e)}


# ============================================================================
# DANBOORU CHARACTER CASTING API ENDPOINTS
# ============================================================================

class VisualCanonRequest(BaseModel):
    """Request model for visual canon assignment."""
    gender: Optional[str] = None
    physical_tags: Optional[List[str]] = None


@app.post("/api/characters/{filename}/assign-visual-canon")
async def assign_visual_canon_character(filename: str, request: VisualCanonRequest):
    """
    Assign visual canon to a character.

    Uses entity gender and physical traits from description to find
    matching Danbooru character via semantic search.
    """
    try:
        from app.visual_canon_assigner import VisualCanonAssigner
        from app.database import db_get_character

        # Load character data
        char_data = db_get_character(filename)
        if not char_data:
            return {"success": False, "error": "Character not found"}

        # Extract constraints
        gender = request.gender or char_data.get('data', {}).get('extensions', {}).get('gender', 'unknown')
        description = char_data.get('data', {}).get('description', '')

        # Extract physical tags from description
        physical_tags = request.physical_tags or VisualCanonAssigner.extract_physical_traits(description)

        # Assign visual canon
        result = VisualCanonAssigner.assign_visual_canon(
            entity_type='character',
            entity_id=filename,
            constraints={
                'gender': gender,
                'physical_tags': physical_tags
            }
        )

        if result['success']:
            # Load character again to get visual_canon mirrored in extensions
            updated_char = db_get_character(filename)

            # Sync to JSON file
            import os
            import json
            from app.database import sync_character_from_json

            file_path = os.path.join(DATA_DIR, "characters", filename)
            with open(file_path, "r", encoding="utf-8") as f:
                char_json = json.load(f)

            # Mirror visual_canon to extensions
            if 'extensions' not in char_json['data']:
                char_json['data']['extensions'] = {}
            char_json['data']['extensions']['visual_canon_id'] = result['visual_canon_id']
            char_json['data']['extensions']['visual_canon_name'] = result['visual_canon_name']
            char_json['data']['extensions']['visual_canon_tags'] = result['visual_canon_tags']

            # Save to JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(char_json, f, indent=2, ensure_ascii=False)

            print(f"[VISUAL_CANON] Assigned to character {filename}: {result['visual_canon_name']}")

        return result

    except Exception as e:
        logger.error(f"Failed to assign visual canon to character: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/characters/{filename}/reroll-visual-canon")
async def reroll_visual_canon_character(filename: str):
    """
    Reroll visual canon for a character.

    Keeps same constraints (gender) but selects different character.
    """
    try:
        from app.visual_canon_assigner import VisualCanonAssigner
        from app.database import db_get_character

        # Load character data
        char_data = db_get_character(filename)
        if not char_data:
            return {"success": False, "error": "Character not found"}

        # Reroll visual canon
        result = VisualCanonAssigner.reroll_visual_canon(
            entity_type='character',
            entity_id=filename
        )

        if result['success']:
            # Load character again to get visual_canon mirrored in extensions
            updated_char = db_get_character(filename)

            # Sync to JSON file
            import os
            import json
            from app.database import sync_character_from_json

            file_path = os.path.join(DATA_DIR, "characters", filename)
            with open(file_path, "r", encoding="utf-8") as f:
                char_json = json.load(f)

            # Mirror visual_canon to extensions
            if 'extensions' not in char_json['data']:
                char_json['data']['extensions'] = {}
            char_json['data']['extensions']['visual_canon_id'] = result['visual_canon_id']
            char_json['data']['extensions']['visual_canon_name'] = result['visual_canon_name']
            char_json['data']['extensions']['visual_canon_tags'] = result['visual_canon_tags']

            # Save to JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(char_json, f, indent=2, ensure_ascii=False)

            print(f"[VISUAL_CANON] Rerolled for character {filename}: {result['visual_canon_name']}")

        return result

    except Exception as e:
        logger.error(f"Failed to reroll visual canon for character: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chats/{chat_id}/npcs/{npc_id}/assign-visual-canon")
async def assign_visual_canon_npc(chat_id: str, npc_id: str, request: VisualCanonRequest):
    """
    Assign visual canon to an NPC.

    Uses entity gender and physical traits from description to find
    matching Danbooru character via semantic search.
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)
        
        from app.visual_canon_assigner import VisualCanonAssigner
        from app.database import db_get_chat

        # Load chat data
        chat = db_get_chat(chat_id)
        if not chat:
            return {"success": False, "error": "Chat not found"}

        # Load NPC data
        localnpcs = chat.get('metadata', {}).get('localnpcs', {})
        npc_data = localnpcs.get(npc_id)
        if not npc_data:
            return {"success": False, "error": "NPC not found"}

        # Extract constraints
        gender = request.gender or npc_data.get('data', {}).get('extensions', {}).get('gender', 'unknown')
        description = npc_data.get('data', {}).get('description', '')

        # Extract physical tags from description
        physical_tags = request.physical_tags or VisualCanonAssigner.extract_physical_traits(description)

        # Assign visual canon
        result = VisualCanonAssigner.assign_visual_canon(
            entity_type='npc',
            entity_id=npc_id,
            chat_id=chat_id,
            constraints={
                'gender': gender,
                'physical_tags': physical_tags
            }
        )

        if result['success']:
            # Mirror visual_canon to NPC extensions
            if 'extensions' not in npc_data['data']:
                npc_data['data']['extensions'] = {}
            npc_data['data']['extensions']['visual_canon_id'] = result['visual_canon_id']
            npc_data['data']['extensions']['visual_canon_name'] = result['visual_canon_name']
            npc_data['data']['extensions']['visual_canon_tags'] = result['visual_canon_tags']

            # Save NPC to chat metadata
            chat['metadata']['localnpcs'][npc_id] = npc_data
            from app.database import db_save_chat
            db_save_chat(chat_id, chat)

            print(f"[VISUAL_CANON] Assigned to NPC {npc_id}: {result['visual_canon_name']}")

        return result

    except Exception as e:
        logger.error(f"Failed to assign visual canon to NPC: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chats/{chat_id}/npcs/{npc_id}/reroll-visual-canon")
async def reroll_visual_canon_npc(chat_id: str, npc_id: str):
    """
    Reroll visual canon for an NPC.

    Keeps same constraints (gender) but selects different character.
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)
        
        from app.visual_canon_assigner import VisualCanonAssigner
        from app.database import db_get_chat

        # Load chat data
        chat = db_get_chat(chat_id)
        if not chat:
            return {"success": False, "error": "Chat not found"}

        # Load NPC data
        localnpcs = chat.get('metadata', {}).get('localnpcs', {})
        npc_data = localnpcs.get(npc_id)
        if not npc_data:
            return {"success": False, "error": "NPC not found"}

        # Reroll visual canon
        result = VisualCanonAssigner.reroll_visual_canon(
            entity_type='npc',
            entity_id=npc_id,
            chat_id=chat_id
        )

        if result['success']:
            # Mirror visual_canon to NPC extensions
            if 'extensions' not in npc_data['data']:
                npc_data['data']['extensions'] = {}
            npc_data['data']['extensions']['visual_canon_id'] = result['visual_canon_id']
            npc_data['data']['extensions']['visual_canon_name'] = result['visual_canon_name']
            npc_data['data']['extensions']['visual_canon_tags'] = result['visual_canon_tags']

            # Save NPC to chat metadata
            chat['metadata']['localnpcs'][npc_id] = npc_data
            from app.database import db_save_chat
            db_save_chat(chat_id, chat)

            print(f"[VISUAL_CANON] Rerolled for NPC {npc_id}: {result['visual_canon_name']}")

        return result

    except Exception as e:
        logger.error(f"Failed to reroll visual canon for NPC: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


class GenerateDanbooruTagsRequest(BaseModel):
    """Request model for generating Danbooru tags from description."""
    description: str
    gender: str  # 'female', 'male', 'other'


class GenerateCapsuleRequest(BaseModel):
    """Request model for generating character capsule summary."""
    name: str
    description: str
    personality: str
    scenario: str
    mes_example: str
    gender: str


@app.post("/api/characters/{filename}/generate-danbooru-tags")
async def generate_danbooru_tags_endpoint(filename: str, request: GenerateDanbooruTagsRequest):
    """
    Generate Danbooru tags from character description using semantic search.
    
    Extracts physical traits (hair color, eye color, body type, creature features)
    and finds matching Danbooru character via progressive tag reduction.
    
    Returns suggested tags to populate the Danbooru Tag field.
    Click again to reroll (overwrites with different match).
    """
    try:
        from app.danbooru_tag_generator import generate_and_assign_to_character
        
        result = generate_and_assign_to_character(
            filename=filename,
            description=request.description,
            gender=request.gender
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate Danbooru tags for character: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/characters/{filename}/generate-capsule")
async def generate_capsule_endpoint(filename: str, request: GenerateCapsuleRequest):
    """
    Generate a capsule summary for a global character.
    
    Uses LLM to distill character card into 50-100 token summary
    for efficient multi-character prompts.
    
    Returns capsule text to populate Capsule field in editor.
    User can edit and save via character save button.
    """
    try:
        capsule = await generate_capsule_for_character(
            char_name=request.name,
            description=request.description,
            personality=request.personality,
            scenario=request.scenario,
            mes_example=request.mes_example,
            gender=request.gender
        )
        
        return {
            "success": True,
            "capsule": capsule
        }
    except Exception as e:
        logger.error(f"Failed to generate capsule for character {filename}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chats/{chat_id}/npcs/{npc_id}/generate-danbooru-tags")
async def generate_danbooru_tags_npc_endpoint(
    chat_id: str, 
    npc_id: str, 
    request: GenerateDanbooruTagsRequest
):
    """
    Generate Danbooru tags for an NPC from description.
    
    Same logic as character endpoint, but stores binding in chat_npcs table.
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)
        
        from app.danbooru_tag_generator import generate_and_assign_to_npc
        
        result = generate_and_assign_to_npc(
            chat_id=chat_id,
            npc_id=npc_id,
            description=request.description,
            gender=request.gender
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate Danbooru tags for NPC: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/chats/{chat_id}/npcs/{npc_id}/generate-capsule")
async def generate_capsule_npc_endpoint(
    chat_id: str,
    npc_id: str,
    request: GenerateCapsuleRequest
):
    """
    Generate a capsule summary for an NPC.
    
    Same logic as character endpoint, but for NPCs stored in chat metadata.
    Returns capsule text to populate Capsule field in NPC editor.
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)
        
        capsule = await generate_capsule_for_character(
            char_name=request.name,
            description=request.description,
            personality=request.personality,
            scenario=request.scenario,
            mes_example=request.mes_example,
            gender=request.gender
        )
        
        return {
            "success": True,
            "capsule": capsule
        }
    except Exception as e:
        logger.error(f"Failed to generate capsule for NPC {npc_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/danbooru-characters")
async def list_danbooru_characters(
    gender: Optional[str] = None,
    limit: int = 100
):
    """
    List available Danbooru characters.

    For testing and debugging purposes.
    """
    try:
        from app.database import db_get_danbooru_characters
        characters = db_get_danbooru_characters(gender=gender, limit=limit)
        return {"success": True, "characters": characters}
    except Exception as e:
        logger.error(f"Failed to list Danbooru characters: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/reimport")
async def reimport_json_files():
    """Manually trigger re-import of JSON files from folders."""
    try:
        import_count = auto_import_json_files()
        return {
            "success": True,
            "imported": import_count,
            "message": f"Imported {import_count['characters']} characters, {import_count['worlds']} worlds, {import_count['chats']} chats"
        }
    except Exception as e:
        print(f"Re-import failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/reimport/characters")
async def reimport_characters():
    """Manually trigger re-import of character JSON files only with capsule generation."""
    try:
        import_count = await import_characters_json_files_async()
        return {
            "success": True,
            "imported": {"characters": import_count},
            "message": f"Imported {import_count} characters"
        }
    except Exception as e:
        print(f"Character re-import failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/reimport/worldinfo")
async def reimport_world_info(request: dict = None):
    """Manually trigger re-import or smart sync of world info JSON files.
    
    Request body:
        smart_sync: If True (default), performs intelligent merge (preserves user changes)
                   If False, performs force reimport (deletes database entries first)
    """
    try:
        # Default to smart sync (True) if not specified
        smart_sync = True
        force = False
        
        if request:
            if "smart_sync" in request:
                smart_sync = bool(request["smart_sync"])
            if "force" in request:
                force = bool(request["force"])
        
        # Smart sync mode: intelligent merge
        if smart_sync and not force:
            wi_dir = os.path.join(DATA_DIR, "worldinfo")
            if os.path.exists(wi_dir):
                total_added = 0
                total_updated = 0
                total_unchanged = 0
                total_kept = 0
                
                for f in os.listdir(wi_dir):
                    if f.endswith(".json") and f != ".gitkeep":
                        name = f.replace(".json", "")
                        # Remove ALL SillyTavern suffixes: _plist, _worldinfo, _json
                        for suffix in ["_plist", "_worldinfo", "_json"]:
                            if name.endswith(suffix):
                                name = name[:-len(suffix)]
                                break
                        
                        result = sync_world_from_json(name)
                        total_added += result.get("added", 0)
                        total_kept += result.get("kept", 0)
                        total_updated += result.get("updated", 0)
                        total_unchanged += result.get("unchanged", 0)
                
                total_changes = total_added + total_updated
                message = f"Synced: {total_added} added, {total_updated} updated, {total_kept} kept, {total_unchanged} unchanged entries from JSON files"
                
                # Clear world info cache after sync
                WORLD_INFO_CACHE.clear()
                
                return {
                    "success": True,
                    "synced": {
                        "added": total_added,
                        "updated": total_updated,
                        "kept": total_kept,
                        "unchanged": total_unchanged,
                        "total": total_changes
                    },
                    "message": message
                }
        # Force reimport mode (legacy behavior)
        import_count = import_world_info_json_files(force=force)
        
        message = f"Imported {import_count} world info files"
        if force and import_count > 0:
            message += " (force reimport - deleted existing entries first)"
        
        return {
            "success": True,
            "imported": {"worlds": import_count},
            "message": message
        }
    except Exception as e:
        print(f"World info re-import failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/reimport/chats")
async def reimport_chats():
    """Manually trigger re-import of chat JSON files only."""
    try:
        import_count = import_chats_json_files()
        return {
            "success": True,
            "imported": {"chats": import_count},
            "message": f"Imported {import_count} chat files"
        }
    except Exception as e:
        print(f"Chat re-import failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/characters")
async def list_characters(tags: Optional[str] = None):
    """Get all characters from database, optionally filtered by tags (AND semantics)."""
    try:
        if tags:
            tag_list = parse_tag_string(tags)
            chars = db_get_characters_by_tags(tag_list)
        else:
            chars = db_get_all_characters()
        return chars
    except Exception as e:
        print(f"Error loading characters from database: {e}")
        return []

@app.post("/api/characters")
async def save_character(char: dict):
    """Save character to database and auto-export to JSON for SillyTavern compatibility."""
    try:
        # Use existing filename if provided, else name.json
        filename = char.get("_filename")
        if not filename:
            char_data = char.get("data", char)
            name = char_data.get("name", "NewCharacter")
            filename = f"{name}.json"
        else:
            char_data = char.get("data", {})
            name = char_data.get("name", "Unknown")
        
        # Save to database
        if not db_save_character(char, filename):
            return {"success": False, "error": "Failed to save to database"}
        
        # Auto-export to JSON for SillyTavern compatibility
        file_path = os.path.join(DATA_DIR, "characters", filename)
        save_data = char.copy()
        if "_filename" in save_data:
            del save_data["_filename"]
        
        # Normalize to V2 spec before export
        save_data = normalize_character_v2(save_data)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Character saved to DB and exported to JSON: {filename}")
        return {"success": True, "filename": filename}
    except Exception as e:
        print(f"Error saving character: {e}")
        return {"success": False, "error": str(e)}

@app.delete("/api/characters/{filename}")
async def delete_character(filename: str):
    """Delete character from database and optionally remove JSON file."""
    try:
        # Delete from database
        if not db_delete_character(filename):
            return {"error": "Character not found in database"}
        
        # Also remove JSON file if it exists
        file_path = os.path.join(DATA_DIR, "characters", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        print(f"Character deleted from DB and JSON: {filename}")
        return {"success": True}
    except Exception as e:
        print(f"Error deleting character: {e}")
        return {"error": str(e)}

@app.get("/api/characters/tags")
async def get_character_tags(limit: Optional[int] = None):
    """Get all unique character tags, optionally limited."""
    try:
        tags = db_get_all_character_tags(limit)
        return {"tags": tags}
    except Exception as e:
        print(f"Error getting character tags: {e}")
        return {"tags": []}

@app.get("/api/characters/popular-tags")
async def get_popular_character_tags(limit: int = 5):
    """Get most-used character tags (for quick chips)."""
    try:
        tags = db_get_popular_character_tags(limit)
        return [{"tag": t, "count": c} for t, c in tags]
    except Exception as e:
        print(f"Error getting popular character tags: {e}")
        return []

@app.get("/api/characters/{filename}/tags")
async def get_character_tags_for_character(filename: str):
    """Get all tags for a specific character."""
    try:
        tags = db_get_character_tags(filename)
        return {"tags": tags}
    except Exception as e:
        print(f"Error getting character tags for {filename}: {e}")
        return {"tags": []}

@app.post("/api/characters/{filename}/tags")
async def update_character_tags(filename: str, request: dict):
    """Replace all tags for a character (normalized on save)."""
    try:
        raw_tags = request.get("tags", [])
        
        # Normalize: lowercase, trim, remove empty, remove duplicates
        normalized_tags = []
        for tag in raw_tags:
            parsed = parse_tag_string(tag)
            normalized_tags.extend(parsed)
        normalized_tags = list(set(normalized_tags))  # Remove duplicates
        normalized_tags = [t for t in normalized_tags if t]  # Remove empty
        
        # Clear existing tags
        db_remove_character_tags(filename, [])
        
        # Add new tags
        if normalized_tags:
            db_add_character_tags(filename, normalized_tags)
        
        return {"success": True, "tags": normalized_tags}
    except Exception as e:
        print(f"Error updating character tags: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/world-info")
async def list_world_info(tags: Optional[str] = None):
    """Get all world info from database, optionally filtered by tags (AND semantics)."""
    try:
        if tags:
            tag_list = parse_tag_string(tags)
            worlds = db_get_worlds_by_tags(tag_list)
        else:
            worlds = db_get_all_worlds()
        return worlds
    except Exception as e:
        print(f"Error loading world info from database: {e}")
        # Fall back to JSON files
        wi_list = []
        wi_dir = os.path.join(DATA_DIR, "worldinfo")
        for f in os.listdir(wi_dir):
            if f.endswith(".json"):
                file_path = os.path.join(wi_dir, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as wi_file:
                        content = json.load(wi_file)
                        wi_list.append({"name": f.replace(".json", ""), **content})
                except Exception as e:
                    print(f"FAILED to load world info {f}: {e}")
        return wi_list

@app.post("/api/world-info")
async def save_world_info(request: dict):
    """Save world info to database and export to JSON for compatibility."""
    name = request.get("name")
    if not name: return {"error": "No name found"}
    
    data = request if "entries" in request else {"entries": request.get("data", {})}
    
    # Save to database (primary source)
    # This also clears stale embeddings in sqlite-vec via db_save_world
    try:
        db_save_world(name, data)
    except Exception as e:
        print(f"Error saving world info to database: {e}")
    
    # Clear in-memory embeddings cache to force recomputation
    # The semantic search engine will recompute embeddings on next search
    # since world info content hash will have changed
    semantic_search_engine.embeddings_cache.clear()
    print(f"Cleared semantic search embeddings cache for world info update: {name}")
    
    # Also clear world info search results cache
    WORLD_INFO_CACHE.clear()
    
    # Also export to JSON for backward compatibility
    file_path = os.path.join(DATA_DIR, "worldinfo", f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"World info saved to DB and exported to JSON: {name}")
    return {"success": True}

@app.delete("/api/world-info/{world_name}")
async def delete_world_info_endpoint(world_name: str):
    """Delete a world info from database and JSON file with change logging."""
    if not world_name:
        return {"error": "No world name provided"}
    
    try:
        # Get world data before deletion for change logging
        old_world = db_get_world(world_name)
        
        # Delete from database
        if not db_delete_world(world_name):
            return {"error": "Failed to delete world"}
        
        # Also delete JSON file
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{world_name}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Refresh world info list in memory
        # (Frontend will fetch updated list from API)
        
        print(f"World info deleted: {world_name}")
        return {"success": True, "name": world_name}
    except Exception as e:
        print(f"Error deleting world info: {e}")
        return {"error": str(e)}

@app.get("/api/world-info/tags")
async def get_world_tags(limit: Optional[int] = None):
    """Get all unique world tags, optionally limited."""
    try:
        tags = db_get_all_world_tags(limit)
        return {"tags": tags}
    except Exception as e:
        print(f"Error getting world tags: {e}")
        return {"tags": []}

@app.get("/api/world-info/popular-tags")
async def get_popular_world_tags(limit: int = 5):
    """Get most-used world tags (for quick chips)."""
    try:
        tags = db_get_popular_world_tags(limit)
        return [{"tag": t, "count": c} for t, c in tags]
    except Exception as e:
        print(f"Error getting popular world tags: {e}")
        return []

@app.get("/api/world-info/{world_name}/tags")
async def get_world_tags_for_world(world_name: str):
    """Get all tags for a specific world."""
    try:
        tags = db_get_world_tags(world_name)
        return {"tags": tags}
    except Exception as e:
        print(f"Error getting world tags for {world_name}: {e}")
        return {"tags": []}

@app.post("/api/world-info/{world_name}/tags")
async def update_world_tags(world_name: str, request: dict):
    """Replace all tags for a world (normalized on save)."""
    try:
        raw_tags = request.get("tags", [])
        
        # Normalize: lowercase, trim, remove empty, remove duplicates
        normalized_tags = []
        for tag in raw_tags:
            parsed = parse_tag_string(tag)
            normalized_tags.extend(parsed)
        normalized_tags = list(set(normalized_tags))  # Remove duplicates
        normalized_tags = [t for t in normalized_tags if t]  # Remove empty
        
        # Clear existing tags
        db_remove_world_tags(world_name, [])
        
        # Add new tags
        if normalized_tags:
            db_add_world_tags(world_name, normalized_tags)
        
        # Clear world info cache to reflect tag changes
        WORLD_INFO_CACHE.clear()
        
        return {"success": True, "tags": normalized_tags}
    except Exception as e:
        print(f"Error updating world tags: {e}")
        return {"success": False, "error": str(e)}

# Chat session management
@app.get("/api/chats")
async def list_chats():
    """Get all chats from database."""
    try:
        chats = db_get_all_chats()
        return chats
    except Exception as e:
        print(f"Error loading chats from database: {e}")
        # Fall back to JSON files
        chats = []
        chat_dir = os.path.join(DATA_DIR, "chats")
        for f in os.listdir(chat_dir):
            if f.endswith(".json"):
                file_path = os.path.join(chat_dir, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as chat_file:
                        chat_data = json.load(chat_file)
                        metadata = chat_data.get("metadata", {})
                        chats.append({
                            "id": f.replace(".json", ""),
                            "branch_name": metadata.get("branch_name")
                        })
                except:
                    chats.append({
                        "id": f.replace(".json", ""),
                        "branch_name": None
                    })
        return chats

@app.get("/api/chats/{name}")
async def load_chat(name: str, include_summarized: bool = False):
    """Load chat from database with optional archived messages.
    
    Query parameter:
        include_summarized: If True, include archived/summarized messages.
                          If False (default), only return active messages.
    """
    try:
        chat_data = db_get_chat(name, include_summarized=include_summarized)
        if chat_data:
            # Ensure id is always set (chat name is -> id)
            if 'id' not in chat_data:
                chat_data['id'] = name
            return chat_data
    except Exception as e:
        print(f"Error loading chat from database: {e}")
    
    # Fall back to JSON file (doesn't support include_summarized)
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            # Ensure id field exists for JSON files (use filename as id)
            if 'id' not in json_data:
                json_data['id'] = name
            return json_data
    return {"error": "Chat not found"}

@app.post("/api/chats")
async def save_chat(request: dict):
    """Save chat to database and export to JSON for compatibility."""
    name = request.get("name")
    if not name:
        # Reject autosave without valid chat_id - forces frontend to initialize first
        return {"success": False, "error": "Chat ID is required for autosave"}
    
    chat_data = request.get("data", {})
    
    # Save to database (primary source)
    try:
        id_mapping = db_save_chat(name, chat_data)
    except Exception as e:
        print(f"Error saving chat to database: {e}")
        return {"success": False, "error": str(e)}
    
    # Also export to JSON for backward compatibility
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    print(f"Chat saved to DB and exported to JSON: {name}")
    return {"success": True, "name": name, "id_mapping": id_mapping}

@app.delete("/api/chats/{name}")
async def delete_chat_endpoint(name: str):
    """Delete chat from database and JSON file."""
    try:
        # Delete from database
        db_delete_chat(name)
    except Exception as e:
        print(f"Error deleting chat from database: {e}")
    
    # Also remove JSON file if it exists
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Chat deleted from DB and JSON: {name}")
        return {"success": True}
    return {"error": "File not found"}

@app.get("/api/chats/{chat_id}/npcs")
async def get_chat_npcs_endpoint(chat_id: str):
    """
    Get all NPCs for a specific chat session.
    
    ARCHITECTURAL FIX: Database is single source of truth.
    Prioritize database data over metadata to ensure consistency.
    Metadata is used only as fallback for NPCs not yet in database.
    """
    try:
        # First, try to get from database (authoritative source)
        db_npcs = db_get_chat_npcs(chat_id)
        
        # Get metadata for fallback (only if DB has no NPCs)
        chat = db_get_chat(chat_id)
        if not chat:
            return {"success": False, "error": "Chat not found"}
        
        metadata_npcs = chat.get('metadata', {}).get('localnpcs', {})
        
        # Merge: DB data takes precedence, metadata fills in gaps
        merged_npcs = []
        seen_entity_ids = set()
        
        # First, add all database NPCs (authoritative source)
        for npc in db_npcs:
            entity_id = npc.get('entityid') or npc.get('entity_id')
            if entity_id and entity_id not in seen_entity_ids:
                merged_npcs.append({
                    'entityid': entity_id,
                    'entity_id': entity_id,
                    'name': npc.get('name', 'Unknown'),
                    'data': npc.get('data', {}),
                    'isnpc': True,
                    'isactive': bool(npc.get('isactive', npc.get('is_active', True))),
                    'is_active': bool(npc.get('isactive', npc.get('is_active', True))),
                    'promoted': bool(npc.get('promoted', False)),
                    'createdat': npc.get('createdat', npc.get('created_at', int(time.time()))),
                    'created_at': npc.get('createdat', npc.get('created_at', int(time.time())))
                })
                seen_entity_ids.add(entity_id)
        
        # Second, fill gaps from metadata only (for NPCs not in DB)
        for npc_id, npc_data in metadata_npcs.items():
            if npc_id not in seen_entity_ids:
                merged_npcs.append({
                    'entityid': npc_id,
                    'entity_id': npc_id,
                    'name': npc_data.get('name', 'Unknown'),
                    'data': npc_data.get('data', {}),
                    'isnpc': True,
                    'isactive': npc_data.get('is_active', True),
                    'promoted': npc_data.get('promoted', False),
                    'createdat': npc_data.get('created_at', int(time.time())),
                    'created_at': npc_data.get('created_at', int(time.time()))
                })
                seen_entity_ids.add(npc_id)
        
        print(f"[NPC_GET] Returning {len(merged_npcs)} NPCs ({len(db_npcs)} from DB, {len(merged_npcs) - len(db_npcs)} from metadata)")
        
        return {
            "success": True,
            "npcs": merged_npcs
        }
    except Exception as e:
        print(f"Error loading NPCs: {e}")
        return {"success": False, "error": str(e)}

# NPC Promotion Request Model
class NPCPromotionRequest(BaseModel):
    chat_id: str
    npc_id: str

# Forking functionality
class ForkRequest(BaseModel):
    origin_chat_name: str
    fork_from_message_id: int
    branch_name: Optional[str] = None

@app.post("/api/chats/fork")
async def fork_chat(request: ForkRequest):
    """
    Fork a chat at a specific message.
    v1.10.4: Complete transaction-based refactor with metadata remapping and validation.
    """
    origin_chat_name = request.origin_chat_name
    fork_from_message_id = request.fork_from_message_id
    branch_name = request.branch_name
    branch_chat_id = None
    
    try:
        # 1. Load origin chat
        origin_chat = db_get_chat(origin_chat_name)
        if not origin_chat:
            raise HTTPException(status_code=404, detail="Origin chat not found")
        
        # 2. Find fork point and extract messages
        messages = origin_chat.get('messages', [])
        fork_index = None
        for i, msg in enumerate(messages):
            if msg.get('id') == fork_from_message_id:
                fork_index = i
                break
        
        if fork_index is None:
            raise HTTPException(status_code=404, detail="Fork message not found")
        
        # 3. Generate branch chat ID
        branch_chat_id = f"{origin_chat_name}_fork_{int(time.time())}"
        
        # 4. Prepare metadata and NPCs
        branch_metadata = origin_chat.get("metadata", {}).copy()
        localnpcs = branch_metadata.get("localnpcs", {}).copy()
        
        # 5. Remap NPC entity IDs for branch safety
        entity_mapping = db_remap_entities_for_branch(
            origin_chat_name,
            branch_chat_id,
            localnpcs  # modified in place
        )
        
        print(f"[FORK] Entity mapping created with {len(entity_mapping)} NPCs remapped")
        for old_id, new_id in entity_mapping.items():
            print(f"[FORK]   {old_id} -> {new_id}")
        
        branch_metadata["localnpcs"] = localnpcs
        
        # 6. CRITICAL: Remap metadata entity IDs for characterCapsules
        old_capsules = branch_metadata.get("characterCapsules", {})
        if old_capsules:
            new_capsules = {
                entity_mapping.get(old_id, old_id): capsule 
                for old_id, capsule in old_capsules.items()
            }
            branch_metadata["characterCapsules"] = new_capsules
            print(f"[FORK] Remapped {len(old_capsules)} characterCapsules entries")
        
        # 7. CRITICAL: Remap metadata entity IDs for characterFirstTurns
        old_turns = branch_metadata.get("characterFirstTurns", {})
        if old_turns:
            new_turns = {
                entity_mapping.get(old_id, old_id): turn 
                for old_id, turn in old_turns.items()
            }
            branch_metadata["characterFirstTurns"] = new_turns
            print(f"[FORK] Remapped {len(old_turns)} characterFirstTurns entries")
        
        # 8. CRITICAL: Remap metadata entity IDs for cast tracking (Phase 2)
        # previous_active_cast is a list of entity IDs that may contain NPCs
        old_active_cast = branch_metadata.get("previous_active_cast", [])
        if old_active_cast:
            new_active_cast = [
                entity_mapping.get(entity_id, entity_id) 
                for entity_id in old_active_cast
            ]
            branch_metadata["previous_active_cast"] = new_active_cast
            print(f"[FORK] Remapped {len(old_active_cast)} previous_active_cast entries")
        
        # previous_focus_character is a single entity ID (or None)
        old_focus = branch_metadata.get("previous_focus_character")
        if old_focus:
            new_focus = entity_mapping.get(old_focus, old_focus)
        branch_metadata["previous_focus_character"] = new_focus
        print(f"[FORK] Remapped previous_focus_character: {old_focus} -> {new_focus}")
        
        # 9. Execute fork in single atomic transaction
        branch_data = db_fork_chat_transaction(
            origin_chat_id=origin_chat_name,
            branch_chat_id=branch_chat_id,
            fork_message_id=fork_from_message_id,
            fork_index=fork_index,
            origin_chat=origin_chat,
            entity_mapping=entity_mapping,
            localnpcs=localnpcs
        )
        
        # 10. Update branch metadata with fork-specific info
        branch_data['metadata'].update({
            'origin_chat_id': origin_chat_name,
            'origin_message_id': fork_from_message_id,
            'branch_name': branch_name or f"Fork from message {fork_from_message_id}",
            'created_at': time.time(),
            'characterCapsules': branch_metadata.get("characterCapsules", {}),
            'characterFirstTurns': branch_metadata.get("characterFirstTurns", {}),
            'previous_active_cast': branch_metadata.get("previous_active_cast", []),      # Phase 2
            'previous_focus_character': branch_metadata.get("previous_focus_character")     # Phase 2
        })
        
        # 10. Save the final branch data
        db_save_chat(branch_chat_id, branch_data, autosaved=True)
        
        # 11. Validation: Check for any "Unknown" entity references
        validation_issues = []
        for char_ref in branch_data.get('activeCharacters', []):
            if char_ref.startswith('npc_') and char_ref not in entity_mapping.values():
                validation_issues.append(f"NPC {char_ref} not found in entity mapping")
        
        if validation_issues:
            print(f"[FORK WARNING] Validation issues detected:")
            for issue in validation_issues:
                print(f"[FORK WARNING]   - {issue}")
        
        print(f"[FORK SUCCESS] Created branch {branch_chat_id} from {origin_chat_name} at message {fork_from_message_id}")
        print(f"[FORK SUCCESS] Remapped {len(entity_mapping)} NPC entity IDs, preserved {len(branch_metadata.get('characterCapsules', {}))} capsules")
        
        return JSONResponse({
            "success": True,
            "name": branch_chat_id,
            "branch_name": branch_name,
            "origin_chat_name": origin_chat_name,
            "fork_from_message_id": fork_from_message_id,
            "remapped_entities": len(entity_mapping),
            "message": f"Forked chat with {len(entity_mapping)} entities remapped"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup if branch was partially created
        if branch_chat_id:
            try:
                # Check if chat exists and delete it
                existing = db_get_chat(branch_chat_id)
                if existing:
                    db_delete_chat(branch_chat_id)
                    print(f"[FORK CLEANUP] Deleted partial branch {branch_chat_id}")
            except Exception as cleanup_error:
                print(f"[FORK CLEANUP FAILED] {cleanup_error}")
        
        print(f"[FORK ERROR] Fork failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fork failed: {str(e)}")

@app.get("/api/chats/{name}/branches")
async def get_chat_branches(name: str):
    """Get all branches that originated from this chat."""
    branches = []
    chat_dir = os.path.join(DATA_DIR, "chats")
    
    for f in os.listdir(chat_dir):
        if f.endswith(".json"):
            file_path = os.path.join(chat_dir, f)
            try:
                with open(file_path, "r", encoding="utf-8") as chat_file:
                    chat_data = json.load(chat_file)
                    metadata = chat_data.get("metadata", {})
                    
                    # Check if this chat is a branch of the requested chat
                    if metadata.get("origin_chat_id") == name:
                        branches.append({
                            "name": f.replace(".json", ""),
                            "branch_name": metadata.get("branch_name", f.replace(".json", "")),
                            "created_at": metadata.get("created_at", 0),
                            "origin_message_id": metadata.get("origin_message_id")
                        })
            except Exception as e:
                print(f"Failed to load chat {f}: {e}")
                continue
    
    # Sort branches by creation time
    branches.sort(key=lambda x: x["created_at"], reverse=True)
    return branches

@app.get("/api/chats/{name}/origin")
async def get_chat_origin(name: str):
    """Get origin information for a branch chat."""
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if not os.path.exists(file_path):
        return {"success": False, "error": "Chat not found"}
    
    with open(file_path, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    
    metadata = chat_data.get("metadata", {})
    origin_chat_id = metadata.get("origin_chat_id")
    origin_message_id = metadata.get("origin_message_id")
    branch_name = metadata.get("branch_name")
    
    if not origin_chat_id:
        return {"success": True, "is_branch": False}
    
    # Try to load origin chat to get more info
    origin_file_path = os.path.join(DATA_DIR, "chats", f"{origin_chat_id}.json")
    origin_info = None
    
    if os.path.exists(origin_file_path):
        try:
            with open(origin_file_path, "r", encoding="utf-8") as origin_file:
                origin_data = json.load(origin_file)
                origin_info = {
                    "name": origin_chat_id,
                    "branch_name": branch_name,
                    "created_at": metadata.get("created_at"),
                    "origin_message_id": origin_message_id
                }
        except:
            pass
    
    return {
        "success": True, 
        "is_branch": True,
        "origin": origin_info
    }

@app.put("/api/chats/{name}/rename-branch")
async def rename_branch(name: str, request: dict):
    """Rename a branch chat."""
    branch_name = request.get("branch_name")
    if not branch_name:
        return {"success": False, "error": "Branch name is required"}
    
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if not os.path.exists(file_path):
        return {"success": False, "error": "Chat not found"}
    
    with open(file_path, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    
    # Ensure metadata exists
    if "metadata" not in chat_data:
        chat_data["metadata"] = {}
    
    chat_data["metadata"]["branch_name"] = branch_name
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    return {"success": True, "branch_name": branch_name}

@app.put("/api/chats/{chat_id}/npcs/{npc_id}")
async def update_npc(chat_id: str, npc_id: str, request: Request):
    """
    Update an existing NPC's data.

    Architectural fix:
    - Check BOTH database (chat_npcs) AND metadata (localnpcs) for NPCs
    - Update in both locations to maintain consistency
    - If NPC only exists in metadata, sync it to database
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)
        
        body = await request.json()
        npc_data = body.get("data", {})

        if not npc_data:
            return JSONResponse({"success": False, "error": "No data provided"}, status_code=400)

        # 1) Get chat
        chat = db_get_chat(chat_id)
        if not chat:
            return JSONResponse({"success": False, "error": "Chat not found"}, status_code=404)

        # 2) Check if NPC exists in database OR metadata
        existing_npcs = db_get_chat_npcs(chat_id)
        npc_in_db = any(
            npc.get("entityid") == npc_id or npc.get("entity_id") == npc_id
            for npc in existing_npcs
        )
        
        metadata = chat.get("metadata", {}) or {}
        localnpcs = metadata.get("localnpcs", {}) or {}
        npc_in_metadata = npc_id in localnpcs

        if not npc_in_db and not npc_in_metadata:
            return JSONResponse(
                {"success": False, "error": "NPC not found in database or metadata"},
                status_code=404,
            )

        # 3) Update NPC in database (create with same entity_id if only in metadata)
        if npc_in_db:
            success = db_update_npc(chat_id, npc_id, npc_data)
            if not success:
                return JSONResponse(
                    {"success": False, "error": "Failed to update NPC in database"},
                    status_code=500,
                )
        else:
            # NPC only in metadata - sync to database with SAME entity_id
            print(f"[NPC_UPDATE] NPC {npc_id} only in metadata, syncing to database with same entity_id")
            success, error = db_create_npc_with_entity_id(chat_id, npc_id, npc_data)
            if not success:
                print(f"[NPC_UPDATE] Failed to sync NPC to database: {error}")

        # 4) Update metadata
        chat = db_get_chat(chat_id)  # reload to get latest structure
        if chat:
            metadata = chat.get("metadata", {}) or {}
            localnpcs = metadata.get("localnpcs", {}) or {}

            if npc_id in localnpcs:
                localnpcs[npc_id]["data"] = npc_data
                localnpcs[npc_id]["name"] = npc_data.get(
                    "name",
                    localnpcs[npc_id].get("name", "Unknown"),
                )
            else:
                # Add to metadata if not there
                localnpcs[npc_id] = {
                    "name": npc_data.get("name", "Unknown"),
                    "data": npc_data,
                    "is_active": True,
                    "promoted": False,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time())
                }

            metadata["localnpcs"] = localnpcs
            chat["metadata"] = metadata
            db_save_chat(chat_id, chat)
            
            # Update NPC metadata timestamp when edited
            if npc_id in localnpcs:
                localnpcs[npc_id]["updated_at"] = int(time.time())
                print(f"[NPC_UPDATE] NPC {npc_id} updated_at timestamp set")
            
            print(f"[NPC_UPDATE] Updated NPC {npc_id} in database and metadata")
            return {"success": True, "message": "NPC updated successfully"}

        return JSONResponse(
            {"success": False, "error": "Failed to sync metadata"}, status_code=500
        )

    except Exception as e:
        print(f"Error updating NPC: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/chats/{chat_id}/npcs/{npc_id}/promote")
async def promote_npc_to_global(chat_id: str, npc_id: str):
    """
    Promote a chat-local NPC to a permanent global character.
    
    ARCHITECTURAL FIX: After promoting, remove NPC from metadata to prevent stale data.
    Database is single source of truth - metadata should reflect database state.
    
    Process:
    1. Get NPC data from chat metadata
    2. Create global character file in SillyTavern v2 format with proper nested structure
    3. Add promotion history metadata to track origin
    4. Save to database with correct character card format
    5. Create new global entity ID
    6. Update chat's activeCharacters to use global character
    7. Mark NPC as promoted AND REMOVE from localnpcs metadata (prevents stale data)
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)

        # 1. Load chat and verify NPC exists
        chat = db_get_chat(chat_id)
        if not chat:
            return {"success": False, "error": "Chat not found"}
        
        # 2. Get NPC from database OR metadata
        db_npcs = db_get_chat_npcs(chat_id)
        npc_data = None
        for npc in db_npcs:
            if npc.get("entityid") == npc_id or npc.get("entity_id") == npc_id:
                npc_data = npc
                break
        
        # If not in database, check metadata
        if not npc_data:
            metadata = chat.get('metadata', {}) or {}
            localnpcs = metadata.get('localnpcs', {}) or {}

            if npc_id in localnpcs:
                metadata_npc = localnpcs[npc_id]
                npc_data = {...}  # build from localnpcs
                print(f"[NPC_PROMOTE] NPC {npc_id} found in metadata, proceeding with promotion")
        
        if not npc_data:
            return {"success": False, "error": "NPC not found in database or metadata"}
        
        npc_name = npc_data.get("name", "Unknown")
        char_data = npc_data.get("data", {})
        
        # 3. Generate global character filename
        filename = f"{npc_name}.json"
        # Handle name collisions
        existing_char = db_get_character(filename)
        if existing_char:
            counter = 1
            while f"{npc_name}_{counter}.json" in [c.get("_filename", "") for c in db_get_all_characters()]:
                counter += 1
            filename = f"{npc_name}_{counter}.json"
        
        # 4. Create proper character card structure (SillyTavern v2 format)
        character_card = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": char_data
        }
        
        # Normalize to ensure all V2 fields are present
        character_card = normalize_character_v2(character_card)
        
        # Add promotion history metadata
        char_data.setdefault('extensions', {})['promotion_history'] = {
            'promoted_from': npc_id,
            'promoted_at': int(time.time()),
            'original_chat_id': chat_id
        }
        
        # 5. Create global character file
        file_path = os.path.join(DATA_DIR, "characters", filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(character_card, f, indent=2, ensure_ascii=False)
        
        # 6. Save to database
        if not db_save_character(character_card, filename):
            return {"success": False, "error": "Failed to save character to database"}
        
        # 7. Update chat - remove NPC from metadata and add global character to activeCharacters
        metadata = chat.get('metadata', {}) or {}
        localnpcs = metadata.get('localnpcs', {}) or {}

        if npc_id in localnpcs:
            del localnpcs[npc_id]
            metadata['localnpcs'] = localnpcs

        # 8. Update activeCharacters to use global character filename
        active_chars = chat.get('activeCharacters', [])
        if npc_id in active_chars:
            active_chars.remove(npc_id)
        if filename not in active_chars:
            active_chars.append(filename)

        # 9. Mark NPC as promoted in database (if it exists there)
        db_set_npc_active(chat_id, npc_id, False)  # Deactivate in DB if exists
        chat['metadata'] = metadata
        chat['activeCharacters'] = active_chars
        db_save_chat(chat_id, chat)
        
        print(f"[NPC_PROMOTE] Promoted NPC {npc_name} ({npc_id}) to global character {filename}")
        
        return {
            "success": True,
            "filename": filename,
            "global_entity_id": filename,
            "promoted_npc_id": npc_id,
            "promoted_npc_name": npc_name
        }
        
    except Exception as e:
        print(f"Error promoting NPC: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.delete("/api/chats/{chat_id}/npcs/{npc_id}")
async def delete_npc(chat_id: str, npc_id: str):
    """
    Delete an NPC from both database and metadata.
    
    This endpoint removes the NPC from:
    - chat_npcs database table
    - localnpcs metadata
    - activeCharacters array
    
    Args:
        chat_id: Chat ID
        npc_id: Unique NPC entity ID
    
    Returns:
        {
            "success": bool,
            "message": str
        }
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)
        
        # 1. Load chat
        chat = db_get_chat(chat_id)
        if not chat:
            return JSONResponse({"success": False, "error": "Chat not found"}, status_code=404)
        
        # 2. Check if NPC exists in database OR metadata
        db_npcs = db_get_chat_npcs(chat_id)
        npc_in_db = any(
            npc.get("entityid") == npc_id or npc.get("entity_id") == npc_id
            for npc in db_npcs
        )

        metadata = chat.get("metadata", {}) or {}
        localnpcs = metadata.get("localnpcs", {}) or {}
        npc_in_metadata = npc_id in localnpcs

        if not npc_in_db and not npc_in_metadata:
            return JSONResponse(
                {"success": False, "error": "NPC not found in database or metadata"},
                status_code=404
            )

        if npc_in_metadata:
            npc_name = localnpcs[npc_id].get("name", "Unknown")
        elif npc_in_db:
            for npc in db_npcs:
                if npc.get("entityid") == npc_id or npc.get("entity_id") == npc_id:
                    npc_name = npc.get("name", "Unknown")
                    break
        
        # 3. Delete from database
        if npc_in_db:
            success = db_delete_npc(chat_id, npc_id)
            if success:
                print(f"[NPC_DELETE] Deleted NPC {npc_id} from database")
            else:
                print(f"[NPC_DELETE] Failed to delete NPC {npc_id} from database")
        
        # 4. Delete from metadata (if present)
        if npc_in_metadata:
            del localnpcs[npc_id]
            metadata["localnpcs"] = localnpcs
        
        # 5. ALWAYS remove from activeCharacters (regardless of storage location)
        active_chars = chat.get('activeCharacters', [])
        if npc_id in active_chars:
            active_chars.remove(npc_id)
            print(f"[NPC_DELETE] Removed NPC {npc_id} from activeCharacters")
        chat['activeCharacters'] = active_chars
        
        # 6. Save updated chat
        chat["metadata"] = metadata
        db_save_chat(chat_id, chat)
        print(f"[NPC_DELETE] Deleted NPC {npc_id} from metadata and activeCharacters")
        
        print(f"[NPC_DELETE] Successfully deleted NPC '{npc_name}' ({npc_id})")
        
        return {
            "success": True,
            "message": f"NPC '{npc_name}' has been deleted"
        }
        
    except Exception as e:
        print(f"Error deleting NPC: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/chats/{chat_id}/npcs/{npc_id}/toggle-active")
async def toggle_npc_active(chat_id: str, npc_id: str):
    """
    Toggle NPC active/inactive status.
    
    ARCHITECTURAL FIX: Check BOTH database AND metadata for NPC.
    Database is authoritative source, metadata is fallback.
    
    Updates:
    - is_active field in chat_npcs database (authoritative)
    - is_active field in chat_npcs metadata (for compatibility)
    - activeCharacters array (adds/removes npc_id)
    
    Args:
        chat_id: Chat ID
        npc_id: Unique NPC entity ID
    """
    try:
        from urllib.parse import unquote
        npc_id = unquote(npc_id)

        # Load chat
        chat = db_get_chat(chat_id)
        if not chat:
            return {"success": False, "error": "Chat not found"}
        
        metadata = chat.get('metadata', {}) or {}
        # Check metadata for NPC
        localnpcs = metadata.get('localnpcs', {}) or {}

        # Check database first (authoritative source)
        db_npcs = db_get_chat_npcs(chat_id)
        db_npc = None
        for npc in db_npcs:
            if npc.get('entityid') == npc_id or npc.get('entity_id') == npc_id:
                db_npc = npc
                break

        # Check metadata as fallback
        metadata_npc = localnpcs.get(npc_id) if npc_id in localnpcs else None

        # NPC must exist in at least one location
        if not db_npc and not metadata_npc:
            return {"success": False, "error": f"NPC '{npc_id}' not found in database or metadata"}

        # Get current active state (prefer database)
        if db_npc:
            current_active = db_npc.get('isactive', db_npc.get('is_active', True))
            npc_name = db_npc.get('name', 'Unknown')
        else:
            current_active = metadata_npc.get('is_active', True)
            npc_name = metadata_npc.get('name', 'Unknown')

        # Toggle active state
        new_active = not current_active

        # Update NPC in database (authoritative)
        db_set_npc_active(chat_id, npc_id, new_active)

        # Update NPC in metadata (for compatibility)
        if npc_id in localnpcs:
            localnpcs[npc_id]['is_active'] = new_active
            metadata['localnpcs'] = localnpcs
        
        # Update activeCharacters array
        active_chars = chat.get('activeCharacters', [])
        
        if new_active:
            # Activate NPC: Add if not already in array
            if npc_id not in active_chars:
                active_chars.append(npc_id)
        else:
            # Deactivate NPC: Remove from array
            if npc_id in active_chars:
                active_chars.remove(npc_id)
        
        # Save updated chat with modified metadata
        metadata['localnpcs'] = localnpcs
        chat['metadata'] = metadata
        chat['activeCharacters'] = active_chars
        db_save_chat(chat_id, chat)
        
        action = "activated" if new_active else "deactivated"
        print(f"[NPC_TOGGLE] {npc_name} ({npc_id}) {action}")
        
        return {
            "success": True,
            "npc_id": npc_id,
            "is_active": new_active,
            "message": f"NPC '{npc_name}' has been {action}"
        }
        
    except Exception as e:
        print(f"Error toggling NPC active status: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# World Info Cache Management Endpoints
@app.get("/api/world-info/cache/stats")
async def get_cache_stats():
    """Get current cache statistics."""
    return {
        "success": True,
        "stats": WORLD_INFO_CACHE.get_stats()
    }

@app.post("/api/world-info/cache/clear")
async def clear_cache():
    """Clear the world info cache."""
    WORLD_INFO_CACHE.clear()
    return {"success": True, "message": "World info cache cleared"}

@app.post("/api/world-info/cache/configure")
async def configure_cache(request: dict):
    """Configure cache settings."""
    max_size = request.get("max_size")
    if max_size is not None and max_size > 0:
        WORLD_INFO_CACHE.max_size = max_size
        return {"success": True, "max_size": max_size}
    return {"success": False, "error": "Invalid max_size value"}

@app.get("/api/world-info/cache/status")
async def get_cache_status():
    """Get detailed cache status for debugging."""
    stats = WORLD_INFO_CACHE.get_stats()
    return {
        "success": True,
        "cache_status": {
            "enabled": True,
            "implementation": "LRU",
            "current_size": stats["size"],
            "max_size": stats["max_size"],
            "usage_percent": stats["usage_percent"],
            "description": "World info entries are cached to avoid reprocessing during long sessions. Cache is automatically managed with LRU eviction."
        }
    }

# World Info Editing Endpoints
@app.post("/api/world-info/edit-entry")
async def edit_world_entry(req: WorldEditRequest):
    """Edit a specific field in a world info entry."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        if not os.path.exists(file_path):
            return {"success": False, "error": "World info file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        
        # Validate entry exists
        if req.entry_uid not in world_data.get("entries", {}):
            return {"success": False, "error": "Entry not found"}
        
        entry = world_data["entries"][req.entry_uid]
        
        # Validate field exists
        valid_fields = ["content", "key", "comment", "is_canon_law", "probability", "useProbability"]
        if req.field not in valid_fields:
            return {"success": False, "error": "Invalid field"}
        
        # Update the field
        if req.field == "key":
            # Ensure key is a list of strings
            if isinstance(req.new_value, str):
                entry["key"] = [k.strip() for k in req.new_value.split(",")]
            elif isinstance(req.new_value, list):
                entry["key"] = req.new_value
            else:
                return {"success": False, "error": "Key must be a string or list of strings"}
        elif req.field == "is_canon_law":
            entry["is_canon_law"] = bool(req.new_value)
        elif req.field == "probability":
            entry["probability"] = int(req.new_value)
        elif req.field == "useProbability":
            entry["useProbability"] = bool(req.new_value)
        else:
            entry[req.field] = req.new_value
        
        # Save the updated world info
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        
        # Sync to database
        try:
            db_save_world(req.world_name, world_data.get("entries", {}))
        except Exception as db_error:
            print(f"Warning: Failed to sync world edit to database: {db_error}")
        
        # Clear caches to reflect the change
        WORLD_INFO_CACHE.clear()
        

        
        return {"success": True, "message": f"Entry '{req.entry_uid}' field '{req.field}' updated successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/world-info/edit-entry-ai")
async def edit_world_entry_ai(req: WorldEditEntryRequest):
    """Use AI to generate or improve a world info entry. For new entries (entry_uid is null), only generates content without updating an existing entry."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        if not os.path.exists(file_path):
            return {"success": False, "error": "World info file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        
        # For new entries (entry_uid is null), just generate content without validation
        # For existing entries, validate that entry_uid exists
        if req.entry_uid is not None and req.entry_uid not in world_data.get("entries", {}):
            return {"success": False, "error": "Entry not found"}
        
        # Use existing generation logic with context
        world_req = WorldGenRequest(
            world_name=req.world_name,
            section=req.section,
            tone=req.tone,
            context=req.context,
            source_mode=req.source_mode
        )
        
        # Call the existing generation endpoint logic
        result = await generate_world_entries(world_req)
        
        if result["success"]:
            # For new entries (entry_uid is null), just return the generated text
            # The frontend will use this to populate the content field
            if req.entry_uid is None:
                return {"success": True, "text": result["text"]}
            
            # For existing entries, update with generated content
            # Parse the generated entries and update the specific entry
            lines = result["text"].split('\n')
            for line in lines:
                parsed = parse_plist_line(line)
                if parsed:
                    # Update the specific entry with new content
                    if req.entry_uid in world_data["entries"]:
                        world_data["entries"][req.entry_uid]["content"] = parsed["content"]
                        world_data["entries"][req.entry_uid]["key"] = parsed["keys"]
                    
                    # If this is a new entry, add it
                    if parsed["name"] not in [e.get("key", [""])[0] for e in world_data["entries"].values()]:
                        uid_counter = max([int(k) for k in world_data["entries"].keys()] + [0]) + 1
                        world_data["entries"][str(uid_counter)] = {
                            "uid": uid_counter,
                            "key": parsed["keys"],
                            "keysecondary": [],
                            "comment": parsed["alias"] or "",
                            "content": parsed["content"],
                            "constant": False,
                            "selective": True,
                            "selectiveLogic": 0,
                            "addMemo": True,
                            "order": 100,
                            "position": 4,
                            "disable": False,
                            "excludeRecursion": False,
                            "probability": 100,
                            "useProbability": True,
                            "displayIndex": uid_counter,
                            "depth": 5
                        }
            
            # Save the updated world info
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(world_data, f, indent=2, ensure_ascii=False)
            
            # Sync to database when updating existing entries
            try:
                db_save_world(req.world_name, world_data.get("entries", {}))
            except Exception as db_error:
                print(f"Warning: Failed to sync world AI edit to database: {db_error}")
            
            # Clear caches to reflect the change
            WORLD_INFO_CACHE.clear()
            
            return {"success": True, "text": result["text"]}
        else:
            return {"success": False, "error": result["error"]}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/world-info/delete-entry")
async def delete_world_entry(world_name: str, entry_uid: str):
    """Delete a specific world info entry and its embedding."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{world_name}.json")
        if not os.path.exists(file_path):
            return {"success": False, "error": "World info file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        
        # Validate entry exists
        if entry_uid not in world_data.get("entries", {}):
            return {"success": False, "error": "Entry not found"}
        
        # Remove the entry from world data
        del world_data["entries"][entry_uid]
        
        # Clean up the embedding from sqlite-vec to prevent orphans
        db_delete_entry_embedding(world_name, entry_uid)
        
        # Also update the database entry
        try:
            db_save_world(world_name, world_data.get("entries", {}))
        except Exception as db_error:
            print(f"Warning: Failed to sync world deletion to database: {db_error}")
        
        # Save the updated world info to JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        
        # Clear caches to reflect the change
        WORLD_INFO_CACHE.clear()
        
        return {"success": True, "message": f"Entry '{entry_uid}' and its embedding deleted successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/world-info/add-entry")
async def add_world_entry(req: WorldAddEntryRequest):
    """Add a new world info entry."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        
        # Load existing world info or create new
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                world_data = json.load(f)
        else:
            world_data = {"entries": {}}
        
        # Get existing tags from JSON (or DB if no tags in JSON)
        existing_tags = world_data.get("tags", [])
        if not existing_tags:
            # Fallback to DB for existing tags
            world_db = db_get_world(req.world_name)
            if world_db:
                existing_tags = db_get_world_tags(req.world_name)
        
        # Merge new tags with existing tags
        new_tags = req.tags if req.tags else []
        merged_tags = list(set(existing_tags + new_tags))
        
        # Update tags in world_data
        world_data["tags"] = merged_tags
        
        # Generate new UID
        uid_counter = max([int(k) for k in world_data["entries"].keys()] + [0]) + 1
        
        # Create new entry
        new_entry = {
            "uid": uid_counter,
            "key": req.entry_data.get("key", []),
            "keysecondary": [],
            "comment": req.entry_data.get("comment", ""),
            "content": req.entry_data.get("content", ""),
            "constant": req.entry_data.get("constant", False),
            "selective": req.entry_data.get("selective", True),
            "selectiveLogic": req.entry_data.get("selectiveLogic", 0),
            "addMemo": req.entry_data.get("addMemo", True),
            "order": req.entry_data.get("order", 100),
            "position": req.entry_data.get("position", 4),
            "disable": req.entry_data.get("disable", False),
            "excludeRecursion": req.entry_data.get("excludeRecursion", False),
            "probability": req.entry_data.get("probability", 100),
            "useProbability": req.entry_data.get("useProbability", True),
            "displayIndex": uid_counter,
            "depth": req.entry_data.get("depth", 5),
            "is_canon_law": req.entry_data.get("is_canon_law", False)
        }
        
        # Add the entry
        world_data["entries"][str(uid_counter)] = new_entry
        
        # Save the updated world info
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        
        # Sync to database (with tags)
        try:
            db_save_world(req.world_name, world_data.get("entries", {}), merged_tags)
        except Exception as db_error:
            print(f"Warning: Failed to sync world add to database: {db_error}")
        
        # Clear caches to reflect the change
        WORLD_INFO_CACHE.clear()
        
        return {"success": True, "entry_uid": str(uid_counter), "message": "New entry added successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Image Metadata Endpoint
class ImageMetadataRequest(BaseModel):
    filename: str

@app.get("/api/image-metadata/{filename}")
async def get_image_metadata_endpoint(filename: str):
    """Get generation parameters for a specific image from database."""
    # Try database first
    metadata = db_get_image_metadata(filename)
    if metadata:
        return {"success": True, "metadata": metadata}
    
    # Fall back to JSON file for backward compatibility
    json_metadata = load_image_metadata()
    if filename in json_metadata.get("images", {}):
        return {"success": True, "metadata": json_metadata["images"][filename]}
    
    return {"success": False, "error": "Image metadata not found"}

@app.post("/api/inpaint")
async def inpaint_image(request: InpaintRequest):
    """Perform image inpainting using Stable Diffusion img2img API."""
    try:
        # Decode base64 images
        image_bytes = base64.b64decode(request.image)
        mask_bytes = base64.b64decode(request.mask)
        
        # Re-encode for SD API (A1111 expects base64 strings)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        
        # Prepare the payload for A1111 img2img API
        payload = {
            "init_images": [image_base64],
            "mask": mask_base64,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "width": request.width,
            "height": request.height,
            "denoising_strength": request.denoising_strength,
            "sampler_name": request.sampler_name,
            "mask_blur": request.mask_blur,
            "inpainting_fill": 1,  # 'original' - use original image content
            "inpaint_full_res": False,  # 'whole picture'
            "inpaint_full_res_padding": 0,
            "inpainting_mask_invert": 0  # 0 = inpaint masked, 1 = inpaint not masked
        }
        
        # Call SD API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONFIG['sd_url']}/sdapi/v1/img2img",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Get the inpainted image
            inpainted_image_base64 = data["images"][0]
            
            # Save the result
            filename = f"inpaint_{int(time.time())}.png"
            file_path = os.path.join(IMAGE_DIR, filename)
            
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(inpainted_image_base64))
            
            # Store metadata for the inpainted image
            metadata_params = SDParams(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                width=request.width,
                height=request.height,
                sampler_name=request.sampler_name,
                scheduler="Automatic"
            )
            store_image_metadata(filename, metadata_params)
            
            return {
                "success": True,
                "url": f"/images/{filename}",
                "filename": filename
            }
            
    except httpx.HTTPStatusError as e:
        error_detail = f"SD API error: {e.response.status_code}"
        try:
            error_data = e.response.json()
            error_detail += f" - {error_data.get('detail', 'Unknown error')}"
        except:
            pass
        return {"success": False, "error": error_detail}
    except Exception as e:
        return {"success": False, "error": f"Inpainting failed: {str(e)}"}

# World Info Reinforcement Configuration Endpoints
@app.get("/api/world-info/reinforcement/config")
async def get_world_info_reinforcement_config():
    """Get current world info reinforcement configuration."""
    return {
        "success": True,
        "config": {
            "world_info_reinforce_freq": 3,  # Default value
            "description": "Frequency (in turns) for reinforcing canon law entries. 1 = every turn, 3 = every 3 turns, etc.",
            "default": 3,
            "min": 1,
            "max": 100
        }
    }

@app.post("/api/world-info/reinforcement/config")
async def set_world_info_reinforcement_config(request: dict):
    """Set world info reinforcement frequency."""
    try:
        frequency = request.get("world_info_reinforce_freq")
        
        if frequency is None:
            return {"success": False, "error": "world_info_reinforce_freq is required"}
        
        if not isinstance(frequency, int) or frequency < 1 or frequency > 100:
            return {"success": False, "error": "world_info_reinforce_freq must be an integer between 1 and 100"}
        
        # For now, we'll return the configured value
        # In a full implementation, this would be stored in user preferences or chat settings
        return {
            "success": True,
            "world_info_reinforce_freq": frequency,
            "message": f"World info reinforcement frequency set to every {frequency} turns"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# SEARCH API ENDPOINTS (FTS5 Full-Text Search)
# ============================================================================

# Search Request Models
class SearchRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    speaker: Optional[str] = None
    start_date: Optional[str] = None  # ISO format: "2026-01-01"
    end_date: Optional[str] = None
    limit: Optional[int] = 50


@app.post("/api/search/messages")
async def search_messages(request: SearchRequest):
    """
    Search across chat messages with filters.
    
    FTS5 query syntax:
    - Simple: "flame sword"
    - Phrase: '"flame sword"' (exact phrase)
    - Boolean: "flame AND sword", "flame OR sword"
    - Exclude: "flame NOT ice"
    """
    try:
        # Convert ISO dates to Unix timestamps if provided
        start_ts = None
        end_ts = None
        
        if request.start_date:
            from datetime import datetime
            start_ts = int(datetime.fromisoformat(request.start_date).timestamp())
        
        if request.end_date:
            from datetime import datetime
            end_ts = int(datetime.fromisoformat(request.end_date).timestamp())
        
        results = db_search_messages(
            query=request.query,
            chat_id=request.chat_id,
            speaker=request.speaker,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            limit=request.limit
        )
        
        return {
            "success": True,
            "query": request.query,
            "filters": {
                "chat_id": request.chat_id,
                "speaker": request.speaker,
                "date_range": f"{request.start_date or 'all'} to {request.end_date or 'all'}"
            },
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        print(f"Search error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/search/messages/{message_id}/context")
async def get_message_context_endpoint(message_id: int, context_size: int = 1):
    """
    Get a message with surrounding context for "jump to message" feature.
    """
    try:
        context = db_get_message_context(message_id, context_size)
        
        if not context:
            return {"success": False, "error": "Message not found"}
        
        return {
            "success": True,
            "context": context
        }
    
    except Exception as e:
        print(f"Error getting message context: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/search/filters")
async def get_search_filters():
    """
    Get available filter options for search UI.
    """
    try:
        speakers = db_get_available_speakers()
        
        # Get available chats
        chats = db_get_all_chats()
        
        return {
            "success": True,
            "filters": {
                "speakers": speakers,
                "chats": [{"id": c['id'], "name": c.get('branch_name') or c['id']} for c in chats]
            }
        }
    
    except Exception as e:
        print(f"Error getting search filters: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/world-info/reinforcement/status")
async def get_world_info_reinforcement_status():
    """Get current world info reinforcement status and statistics."""
    return {
        "success": True,
        "status": {
            "default_frequency": 3,
            "description": "Canon law entries are reinforced every 3 turns by default, reducing prompt bloat while maintaining consistency.",
            "benefits": [
                "Reduced prompt length",
                "Better performance",
                "Configurable control",
                "Backward compatibility"
            ],
            "implementation": "World info reinforcement is handled automatically in the prompt construction engine."
        }
    }


# ============================================================================
# CHANGE LOG API ENDPOINTS (v1.5.1 - Undo/Redo Foundation)
# ============================================================================

@app.get("/api/changes")
async def get_changes(entity_type: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 20):
    """Get recent changes for debugging/undo preview.
    
    Query parameters:
    - entity_type: Filter by 'character', 'world_info', or 'chat' (optional)
    - entity_id: Filter by specific entity ID (optional, requires entity_type)
    - limit: Maximum number of changes to return (default: 20)
    """
    try:
        changes = get_recent_changes(entity_type, entity_id, limit)
        return {
            "success": True,
            "changes": changes,
            "count": len(changes)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/changes/{entity_type}/{entity_id}")
async def get_entity_changes(entity_type: str, entity_id: str, limit: int = 10):
    """Get change history for a specific entity.
    
    Useful for showing undo options for a particular character, world, or chat.
    """
    try:
        changes = get_recent_changes(entity_type, entity_id, limit)
        return {
            "success": True,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "changes": changes,
            "count": len(changes)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/changes/{entity_type}/{entity_id}/last")
async def get_entity_last_change(entity_type: str, entity_id: str):
    """Get the most recent change for a specific entity.
    
    Useful for implementing a quick undo button.
    """
    try:
        change = get_last_change(entity_type, entity_id)
        if change:
            return {
                "success": True,
                "change": change,
                "can_undo": change["operation"] in ["UPDATE", "DELETE"]
            }
        return {
            "success": True,
            "change": None,
            "can_undo": False
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/undo/last")
async def undo_last_delete_endpoint(request: dict):
    """Undo the last DELETE operation for an entity.
    
    Request body:
    {
        "entity_type": "character" | "world_info" | "chat",
        "entity_id": "alice.json" | "FantasyWorld" | "chat_id"
    }
    
    Returns:
    {
        "success": true,
        "restored_entity": {...}
    }
    """
    try:
        entity_type = request.get("entity_type")
        entity_id = request.get("entity_id")
        
        if not entity_type or not entity_id:
            return {"success": False, "error": "entity_type and entity_id are required"}
        
        if entity_type not in ["character", "world_info", "chat"]:
            return {"success": False, "error": "Invalid entity_type. Must be 'character', 'world_info', or 'chat'"}
        
        # Call database undo function
        result = undo_last_delete(entity_type, entity_id)
        return result
    except Exception as e:
        print(f"Error in undo_last_delete: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/changes/restore")
async def restore_version(request: dict):
    """Restore an entity to a specific change version.
    
    Request body:
    {
        "change_id": 123
    }
    
    Returns:
    {
        "success": true,
        "restored_entity": {...}
    }
    """
    try:
        change_id = request.get("change_id")
        
        if not change_id:
            return {"success": False, "error": "change_id is required"}
        
        # Validate change_id is an integer
        try:
            change_id = int(change_id)
        except (ValueError, TypeError):
            return {"success": False, "error": "change_id must be a valid integer"}
        
        # Call database restore function
        result = restore_version(change_id)
        return result
    except Exception as e:
        print(f"Error in restore_version: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/changes/cleanup")
async def cleanup_old_changes(request: dict = None):
    """Manually trigger cleanup of old change log entries.
    
    Optional: Pass {"days": N} to specify cleanup window (default: 30 days).
    """
    try:
        days = 30
        if request and "days" in request:
            days = int(request["days"])
            if days < 1 or days > 365:
                return {"success": False, "error": "Days must be between 1 and 365"}
        
        success = db_cleanup_old_changes(days)
        return {
            "success": success,
            "message": f"Cleaned up change log entries older than {days} days"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/changes/stats")
async def get_change_log_stats():
    """Get statistics about the change log.
    
    Useful for monitoring log size and usage.
    """
    try:
        # Get counts by entity type
        all_changes = get_recent_changes(limit=1000)  # Get up to 1000 for stats
        
        stats = {
            "total_entries": len(all_changes),
            "by_entity_type": {},
            "by_operation": {},
            "oldest_entry": None,
            "newest_entry": None
        }
        
        for change in all_changes:
            # Count by entity type
            entity_type = change["entity_type"]
            if entity_type not in stats["by_entity_type"]:
                stats["by_entity_type"][entity_type] = 0
            stats["by_entity_type"][entity_type] += 1
            
            # Count by operation
            operation = change["operation"]
            if operation not in stats["by_operation"]:
                stats["by_operation"][operation] = 0
            stats["by_operation"][operation] += 1
        
        # Get oldest and newest
        if all_changes:
            stats["newest_entry"] = all_changes[0]["timestamp"]
            stats["oldest_entry"] = all_changes[-1]["timestamp"]
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# CHAT EXPORT FOR LLM TRAINING (Phase 7.4a)
# ============================================================================

@app.post("/api/chats/{chat_id}/export-training")
async def export_chat_for_training(chat_id: str, request: Request):
    """
    Export chat in format optimized for LLM fine-tuning (Unsloth, Axolotl, etc.).
    
    Formats supported:
    - 'alpaca': Instruction/input/output format
    - 'sharegpt': Multi-turn conversation format  
    - 'chat-ml': ChatML format
    """
    try:
        body = await request.json()
    except:
        body = {}
    
    chat = db_get_chat(chat_id)
    if not chat:
        return JSONResponse({"success": False, "error": "Chat not found"}, status_code=404)

    export_format = body.get('format', 'alpaca')
    include_system = body.get('include_system', True)
    include_world_info = body.get('include_world_info', False)
    include_npcs = body.get('include_npcs', True)

    # Get messages
    messages = chat.get('messages', [])
    
    if not messages:
        return JSONResponse({"success": False, "error": "Chat has no messages to export"}, status_code=400)

    # Get character data for context
    localnpcs = chat.get('metadata', {}).get('localnpcs', {})
    active_chars = chat.get('activeCharacters', [])

    # Build character context
    character_context = []
    for char_ref in active_chars:
        # Handle both string and dict formats
        if isinstance(char_ref, dict):
            char_ref = char_ref.get('id') or char_ref.get('name') or char_ref.get('_filename') or ''
        
        if isinstance(char_ref, str) and char_ref.startswith('npc_'):
            if char_ref in localnpcs and include_npcs:
                npc = localnpcs[char_ref]
                character_context.append(f"Character: {npc.get('name', 'Unknown')}")
                if 'description' in npc.get('data', {}):
                    character_context.append(npc['data']['description'])
        elif isinstance(char_ref, str):
            char_data = db_get_character(char_ref)
            if char_data:
                name = get_character_name(char_data)
                character_context.append(f"Character: {name}")
                if 'description' in char_data.get('data', {}):
                    character_context.append(char_data['data']['description'])

    # Build world info context (if requested)
    world_context = []
    if include_world_info:
        active_wi = chat.get('activeWI')
        if active_wi:
            # Handle both dict and string formats
            world_name = active_wi.get('name') if isinstance(active_wi, dict) else active_wi
            if world_name:
                world_data = db_get_world(world_name)
                if world_data:
                    entries = world_data.get('entries', {})
                    # Handle both dict and list formats for entries
                    if isinstance(entries, dict):
                        for uid, entry in entries.items():
                            if entry.get('is_canon_law') or entry.get('isCanonLaw'):
                                world_context.append(entry.get('content', ''))
                    elif isinstance(entries, list):
                        for entry in entries:
                            if entry.get('is_canon_law') or entry.get('isCanonLaw'):
                                world_context.append(entry.get('content', ''))

    # Format conversion
    try:
        if export_format == 'alpaca':
            training_data = format_alpaca(messages, character_context, world_context, include_system)
        elif export_format == 'sharegpt':
            training_data = format_sharegpt(messages, character_context, world_context, include_system)
        elif export_format == 'chat-ml':
            training_data = format_chatml(messages, character_context, world_context, include_system)
        else:
            return JSONResponse({"success": False, "error": f"Unsupported format: {export_format}"}, status_code=400)
    except Exception as e:
        logger.error(f"Error formatting export: {e}")
        return JSONResponse({"success": False, "error": f"Format error: {str(e)}"}, status_code=500)

    return {
        "success": True,
        "format": export_format,
        "data": training_data,
        "message_count": len(messages),
        "character_count": len(character_context),
        "metadata": {
            "chat_id": chat_id,
            "exported_at": time.time(),
            "include_system": include_system,
            "include_world_info": include_world_info,
            "include_npcs": include_npcs
        }
    }


@app.post("/api/chats/export-all-training")
async def export_all_chats_for_training(request: Request):
    """
    Export all chats as training data.
    """
    try:
        body = await request.json()
    except:
        body = {}
    
    export_format = body.get('format', 'alpaca')
    include_system = body.get('include_system', True)
    include_world_info = body.get('include_world_info', False)
    include_npcs = body.get('include_npcs', True)

    all_chats = db_get_all_chats()
    combined_data = []
    total_messages = 0
    errors = []

    for chat in all_chats:
        chat_id = chat.get('id')
        if not chat_id:
            continue

        try:
            # Get chat data directly instead of calling the endpoint
            chat_data = db_get_chat(chat_id)
            if not chat_data:
                continue
                
            messages = chat_data.get('messages', [])
            if not messages:
                continue
            
            # Build contexts (simplified for bulk export)
            character_context = []
            world_context = []
            
            # Format based on export type
            if export_format == 'alpaca':
                data = format_alpaca(messages, character_context, world_context, include_system)
            elif export_format == 'sharegpt':
                data = format_sharegpt(messages, character_context, world_context, include_system)
            elif export_format == 'chat-ml':
                data = format_chatml(messages, character_context, world_context, include_system)
            else:
                continue
            
            combined_data.extend(data if isinstance(data, list) else [data])
            total_messages += len(messages)
            
        except Exception as e:
            errors.append(f"Chat {chat_id}: {str(e)}")
            continue

    return {
        "success": True,
        "format": export_format,
        "data": combined_data,
        "total_chats": len(all_chats),
        "total_messages": total_messages,
        "exported_at": time.time(),
        "errors": errors if errors else None
    }


# ============================================================================
# TRAINING DATA FORMATTERS (Phase 7.4a - Export Chat for LLM Training)
# ============================================================================

def format_alpaca(messages: list, character_context: list, world_context: list, 
                 include_system: bool = True) -> list:
    """
    Format messages in Alpaca format for LLM fine-tuning.
    
    Alpaca format:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    training_data = []
    
    # Build system prompt from context
    system_prompt = ""
    if character_context:
        system_prompt += "\n".join(character_context) + "\n\n"
    if world_context:
        system_prompt += "World Context:\n" + "\n".join(world_context) + "\n\n"
    if include_system:
        system_prompt += CONFIG["system_prompt"]
    
    # Convert messages to instruction/input/output pairs
    for i, msg in enumerate(messages):
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        speaker = msg.get('speaker', role)
        
        if role == 'user':
            # User message - create instruction
            instruction = content
            input_text = system_prompt
            output = ""
            
            # Find the next assistant response
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.get('role') != 'user':
                    output = next_msg.get('content', '')
            
            if output:  # Only add if we have a response
                training_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                })
    
    return training_data

# ============================================================================
# FORMAT CONVERSION FUNCTIONS (Phase 7.4b)
# ============================================================================


def format_sharegpt(messages: list, character_context: list, world_context: list,
                   include_system: bool = True) -> list:
    """
    Format messages in ShareGPT format for LLM fine-tuning.
    
    ShareGPT format:
    [
        {
            "id": "...",
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }
    ]
    """
    conversations = []
    
    # Build system message
    if character_context or world_context or include_system:
        system_msg = ""
        if character_context:
            system_msg += "\n".join(character_context) + "\n\n"
        if world_context:
            system_msg += "World Context:\n" + "\n".join(world_context) + "\n\n"
        if include_system:
            system_msg += CONFIG["system_prompt"]
        
        if system_msg.strip():
            conversations.append({
                "from": "system",
                "value": system_msg.strip()
            })
    
    # Add messages as conversation turns
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        speaker = msg.get('speaker', role)
        
        # Map roles to ShareGPT format
        if role == 'user':
            from_role = "human"
        else:
            # Use speaker name for non-user roles (NPCs, narrator, etc.)
            from_role = speaker if speaker and speaker != role else "gpt"
        
        if content.strip():  # Skip empty messages
            conversations.append({
                "from": from_role,
                "value": content
            })
    
    return [{
        "id": str(int(time.time())),
        "conversations": conversations
    }]


def format_chatml(messages: list, character_context: list, world_context: list,
                 include_system: bool = True) -> list:
    """
    Format messages in ChatML format for LLM fine-tuning.
    
    ChatML format:
    <|im_start|>system\n...<|im_end|>
    <|im_start|>user\n...<|im_end|>
    <|im_start|>assistant\n...<|im_end|>
    """
    conversation = ""
    
    # Add system message if requested
    if include_system or character_context or world_context:
        system_msg = ""
        if character_context:
            system_msg += "\n".join(character_context) + "\n\n"
        if world_context:
            system_msg += "World Context:\n" + "\n".join(world_context) + "\n\n"
        if include_system:
            system_msg += CONFIG["system_prompt"]
        
        if system_msg.strip():
            conversation += f"<|im_start|>system\n{system_msg.strip()}<|im_end|>\n"
    
    # Add messages as ChatML turns
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        speaker = msg.get('speaker', role)
        
        # Map to ChatML roles
        if role == 'user':
            chatml_role = "user"
        else:
            # Use specific role name (narrator, NPC name, etc.)
            chatml_role = speaker if speaker and speaker != role else "assistant"
        
        if content.strip():  # Skip empty messages
            conversation += f"<|im_start|>{chatml_role}\n{content}<|im_end|>\n"
    
    return [conversation]


# ============================================================================
# SOFT DELETE MANAGEMENT ENDPOINTS (v1.5.3)
# ============================================================================

@app.get("/api/chats/summarized/stats")
async def get_summarized_stats():
    """Get statistics about summarized (archived) messages."""
    try:
        from app.database import db_get_summarized_message_count
        count = db_get_summarized_message_count()
        return {
            "success": True,
            "stats": {
                "summarized_message_count": count,
                "description": "Messages marked as summarized (archived but not deleted)"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/chats/cleanup-old-summarized")
async def cleanup_old_summarized(request: dict = None):
    """Permanently delete summarized messages older than specified days.
    
    Optional: Pass {"days": N} to specify cleanup window (default: 90 days).
    This is optional cleanup for users who want to reclaim disk space.
    """
    try:
        days = 90
        if request and "days" in request:
            days = int(request["days"])
            if days < 1 or days > 365:
                return {"success": False, "error": "Days must be between 1 and 365"}
        
        from app.database import db_cleanup_old_summarized_messages
        deleted_count = db_cleanup_old_summarized_messages(days)
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Cleaned up {deleted_count} old summarized messages (older than {days} days)"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/chats/{chat_id}/summarize-selection")
async def autosummarize_selection(chat_id: str, request: dict):
    """Summarize user-selected text from summary window.
    
    Allows users to highlight specific sections of their summary and
    get a condensed version, addressing bloated summary windows.
    """
    try:
        text = request.get("text", "").strip()
        if not text:
            return {"success": False, "error": "No text provided"}
        
        if len(text) < 50:
            return {"success": False, "error": "Text too short to summarize (minimum 50 characters)"}
        
        summarized = await summarize_text(text)
        
        if not summarized:
            return {"success": False, "error": "Failed to generate summary"}
        
        return {
            "success": True,
            "summarized": summarized,
            "original_length": len(text),
            "new_length": len(summarized)
        }
    except Exception as e:
        logger.error(f"Autosummarize failed for chat {chat_id}: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)