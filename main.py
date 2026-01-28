# Simple FastAPI app with NPC support
from fastapi import FastAPI
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
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

# Import the updated NPC functions with defensive lookup
from app.database import (
    db_get_all_characters, db_get_character, db_save_character, db_delete_character,
    db_get_all_worlds, db_get_world, db_save_world, db_delete_world, db_update_world_entry,
    db_get_all_chats, db_get_chat, db_save_chat, db_delete_chat,
    db_save_image_metadata, db_get_image_metadata, db_get_all_image_metadata,
    db_save_performance_metric, db_get_median_performance, db_cleanup_old_metrics,
    db_get_recent_performance_metrics,
    db_get_world_content_hash, db_get_world_entry_hash, db_update_world_entry_hash,
    db_save_embedding, db_get_embedding, db_search_similar_embeddings,
    db_delete_entry_embedding, db_create_entity,
    # NPC functions (Phase 2.2)
    db_create_npc, db_get_chat_npcs, db_get_npc_by_id, db_update_npc,
    db_increment_npc_appearance, db_set_npc_active, db_delete_npc, 
    db_promote_npc_to_character, db_copy_npcs_for_branch,
    db_create_npc_and_update_chat,  # NEW: Atomic function for NPC creation
    get_connection,
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
    db_copy_relationship_states_with_mapping,
    # Change log cleanup
    db_cleanup_old_changes,
)

# NPC helper functions are now integrated in load_character_profiles()

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
    "performance_mode_enabled": True
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

class CapsuleGenRequest(BaseModel):
    char_name: str
    description: str
    depth_prompt: str = ""

class WorldGenRequest(BaseModel):
    world_name: str
    section: str # history, locations, creatures, factions
    tone: str # neutral (simplified - only neutral option now)
    context: str
    source_mode: Optional[str] = "chat" # "chat" or "manual"

class WorldSaveRequest(BaseModel):
    world_name: str
    plist_text: str

# Editing Models
class CharacterEditRequest(BaseModel):
    filename: str
    field: str  # personality, body, dialogue, genre, tags, scenario, first_message
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

# Semantic Search Engine
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
            
            # Load model with explicit device assignment
            self.model = SentenceTransformer(self.model_name, device=device)
            
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
            # Embed only the PRIMARY key (first line/phrase); remaining keys are for understanding only
            primary_key = entry.get("key", [])
            primary_key = primary_key[0] if primary_key else ""
            primary_list = [primary_key] if primary_key and primary_key.lower() not in self.GENERIC_KEYS else []
            secondary_list = [k for k in entry.get("keysecondary", []) if k.lower() not in self.GENERIC_KEYS]
            keys = ", ".join(primary_list)
            secondary = ", ".join(secondary_list)
            if keys or secondary:
                prefixed_content = f"Keywords: {keys}"
                if secondary:
                    prefixed_content += f"; Secondary: {secondary}"
                contents.append(prefixed_content)
                uids_to_compute.append(uid)
        
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


def get_relationship_context(chat_id: str, characters: list, user_name: str, 
                            recent_messages: list) -> str:
    """
    Generate compact relationship context for prompt injection.
    Uses Tier 3 semantic filtering to only include dimensions relevant to current conversation.
    Returns 0-75 tokens depending on active interactions.
    
    CRITICAL: characters list should already be extracted with get_character_name().
    """
    import random
    
    # Identify active speakers in last 5 messages
    # Fix: Access Pydantic model attributes directly (m.speaker, m.role)
    active_speakers = set(m.speaker or m.role for m in recent_messages[-5:])
    active_characters = [c for c in characters if c in active_speakers]
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
    for char_from in active_characters:
        for char_to in active_characters:
            if char_from == char_to:
                continue
            
            state = db_get_relationship_state(chat_id, char_from, char_to)
            if state:
                entity_key = f"{char_from}→{char_to}"
                relationship_states[entity_key] = state
    
    # Character-to-user relationships
    if user_is_active and active_characters:
        user_name_or_default = user_name or "User"
        for char_from in active_characters:
            state = db_get_relationship_state(chat_id, char_from, user_name_or_default)
            if state:
                entity_key = f"{char_from}→{user_name_or_default}"
                relationship_states[entity_key] = state
    
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
                for dim in relevant_dims:
                    score = state[dim]
                    for (low, high), templates in RELATIONSHIP_TEMPLATES[dim].items():
                        if low <= score <= high:
                            if templates:
                                template = random.choice(templates)
                                lines.append(template.format(from_=char_from, to=char_to))
                
                if lines:
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
    for char_from in active_characters:
        # Check toward other characters
        for char_to in active_characters:
            if char_from == char_to:
                continue
            
            # Get relationship state
            state = db_get_relationship_state(chat_id, char_from, char_to)
            if not state:
                continue
            
            # Find dimensions far from neutral (50 ± 15)
            notable = []
            dimensions = ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']
            
            for dim in dimensions:
                score = state[dim]
                if abs(score - 50) > 15:
                    notable.append((dim, score))
            
            # Sort by extremity, take top 2
            notable.sort(key=lambda x: abs(x[1] - 50), reverse=True)
            top_two = notable[:2]
            
            # Generate templates for top two dimensions
            for dim, score in top_two:
                for (low, high), templates in RELATIONSHIP_TEMPLATES[dim].items():
                    if low <= score <= high:
                        if templates:
                            template = random.choice(templates)
                            lines.append(template.format(from_=char_from, to=char_to))
            
            if lines:
                lines.append(".")
    
    # Also check toward user if user is active (fallback only)
    if user_is_active and active_characters:
        user_name_or_default = user_name or "User"
        
        for char_from in active_characters:
            state = db_get_relationship_state(chat_id, char_from, user_name_or_default)
            if not state:
                continue
            
            # Find dimensions far from neutral
            notable = []
            dimensions = ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']
            
            for dim in dimensions:
                score = state[dim]
                if abs(score - 50) > 15:
                    notable.append((dim, score))
            
            # Sort by extremity, take top 2
            notable.sort(key=lambda x: abs(x[1] - 50), reverse=True)
            top_two = notable[:2]
            
            # Generate templates for top two dimensions
            for dim, score in top_two:
                for (low, high), templates in RELATIONSHIP_TEMPLATES[dim].items():
                    if low <= score <= high:
                        if templates:
                            template = random.choice(templates)
                            lines.append(template.format(from_=char_from, to=user_name_or_default))
            
            if lines:
                lines.append(".")
    
    if lines:
        return "### Relationship Context:\n" + "\n".join(lines)
    
    return ""


# Global variable to store cleanup task
cleanup_task = None


async def analyze_and_update_relationships(chat_id: str, messages: list, 
                                      characters: list, user_name: str = None):
    """
    Analyze relationships using direct semantic comparison.
    Triggered at summarization boundary (every 10 messages).
    
    CRITICAL: characters list should already be extracted with get_character_name().
    """
    global relationship_analyzer
    
    if relationship_analyzer is None:
        relationship_analyzer = RelationshipAnalyzer(semantic_search_engine)
    
    if len(characters) < 1:
        return
    
    # Build entity list - characters are already extracted with get_character_name()
    all_entities = characters.copy()
    if user_name:
        all_entities.append(user_name)
    
    # Get recent speakers for relevance filtering
    recent_speakers = set(m.get('speaker') for m in messages[-10:])
    
    # Analyze each directional relationship
    for char_from in characters:
        if char_from not in recent_speakers:
            continue  # Character wasn't active
        
        for entity_to in all_entities:
            if char_from == entity_to:
                continue
            
            if entity_to not in recent_speakers:
                continue  # Target wasn't active
            
            # Get current state
            current_state = db_get_relationship_state(chat_id, char_from, entity_to)
            
            if not current_state:
                current_state = {
                    'trust': 50, 'emotional_bond': 50, 'conflict': 50,
                    'power_dynamic': 50, 'fear_anxiety': 50
                }
            
            # Analyze conversation directly (no LLM generation)
            new_scores = relationship_analyzer.analyze_conversation(
                messages=messages,
                char_from=char_from,
                entity_to=entity_to,
                current_state=current_state
            )
            
            if new_scores is None:
                continue  # No interaction detected
            
            # Check for meaningful change (threshold: 5 points)
            if any(abs(new_scores[dim] - current_state[dim]) > 5 
                   for dim in ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']):
                
                db_update_relationship_state(
                    chat_id=chat_id,
                    character_from=char_from,
                    character_to=entity_to,
                    scores=new_scores,
                    last_message_id=messages[-1].get('id', 0)
                )
                
                print(f"[RELATIONSHIP] {char_from}→{entity_to}: "
                      f"trust={new_scores['trust']}, bond={new_scores['emotional_bond']}, "
                      f"conflict={new_scores['conflict']}")


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

# Auto-import JSON files on startup
def auto_import_json_files():
    """Scan JSON folders and import any new files not already in database."""
    import_count = {"characters": 0, "worlds": 0, "chats": 0}
    
    # Import characters
    char_dir = os.path.join(DATA_DIR, "characters")
    if os.path.exists(char_dir):
        for f in os.listdir(char_dir):
            if f.endswith(".json") and f != ".gitkeep":
                file_path = os.path.join(char_dir, f)
                try:
                    # Check if already in database
                    existing = db_get_character(f)
                    if existing is None:
                        # Load and import
                        with open(file_path, "r", encoding="utf-8") as cf:
                            char_data = json.load(cf)
                        if db_save_character(char_data, f):
                            import_count["characters"] += 1
                            print(f"Auto-imported character: {f}")
                except Exception as e:
                    print(f"Failed to auto-import character {f}: {e}")
    
    # Import world info
    wi_dir = os.path.join(DATA_DIR, "worldinfo")
    if os.path.exists(wi_dir):
        for f in os.listdir(wi_dir):
            if f.endswith(".json") and f != ".gitkeep":
                name = f.replace(".json", "")
                # Remove ALL SillyTavern suffixes: _plist, _worldinfo, _json
                # This handles files like "exampleworld_plist_worldinfo.json" -> "exampleworld"
                for suffix in ["_plist", "_worldinfo", "_json"]:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        # Only remove one suffix to handle files with multiple suffixes
                        break
                file_path = os.path.join(wi_dir, f)
                try:
                    # Check if already in database
                    existing = db_get_world(name)
                    if existing is None:
                        # Load and import
                        with open(file_path, "r", encoding="utf-8") as wf:
                            world_data = json.load(wf)
                        # db_save_world expects entries dict, not -> full object
                        entries = world_data.get("entries", world_data) if isinstance(world_data, dict) else {}
                        if entries and db_save_world(name, entries):
                            import_count["worlds"] += 1
                            print(f"Auto-imported world info: {name}")
                except Exception as e:
                    print(f"Failed to auto-import world info {f}: {e}")
    
    # Import chats
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
                            import_count["chats"] += 1
                            print(f"Auto-imported chat: {name}")
                except Exception as e:
                    print(f"Failed to auto-import chat {f}: {e}")
    
    total = import_count["characters"] + import_count["worlds"] + import_count["chats"]
    if total > 0:
        print(f"Auto-import complete: {import_count['characters']} characters, {import_count['worlds']} worlds, {import_count['chats']} chats")
    else:
        print("Auto-import: No new JSON files to import")
    
    return import_count

# Separate import functions for targeted reimport
def import_characters_json_files():
    """Import only character JSON files not already in database."""
    import_count = 0
    
    char_dir = os.path.join(DATA_DIR, "characters")
    if os.path.exists(char_dir):
        for f in os.listdir(char_dir):
            if f.endswith(".json") and f != ".gitkeep":
                file_path = os.path.join(char_dir, f)
                try:
                    # Check if already in database
                    existing = db_get_character(f)
                    if existing is None:
                        # Load and import
                        with open(file_path, "r", encoding="utf-8") as cf:
                            char_data = json.load(cf)
                        if db_save_character(char_data, f):
                            import_count += 1
                            print(f"Imported character: {f}")
                except Exception as e:
                    print(f"Failed to import character {f}: {e}")
    
    if import_count > 0:
        print(f"Character import complete: {import_count} characters")
    else:
        print("Character import: No new JSON files to import")
    
    return import_count

def sync_world_from_json(world_name):
    """Intelligently merge a single JSON world with its database entries.
    
    Compares JSON entries vs database entries and merges them:
    - New JSON entries: Add to database (they don't exist yet)
    - Deleted database entries: Remove from result (gone from JSON)
    - Existing entries: Keep whichever was modified more recently
    
    Returns:
        dict with 'added' (int), 'removed' (int), 'merged' (int) counts
    """
    # Load JSON file
    file_path = os.path.join(DATA_DIR, "worldinfo", f"{world_name}.json")
    
    if not os.path.exists(file_path):
        print(f"World info JSON not found: {file_path}")
        return {"added": 0, "removed": 0, "merged": 0}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            json_entries = json_data.get("entries", json_data) if isinstance(json_data, dict) else {}
    except Exception as e:
        print(f"Failed to load world info JSON {world_name}: {e}")
        return {"added": 0, "removed": 0, "merged": 0}
    
    # Load database entries
    db_world = db_get_world(world_name)
    if db_world is None:
        # No world in database - just import JSON
        entries = json_entries
        if db_save_world(world_name, entries):
            print(f"Sync: Imported {len(entries)} entries from JSON (no existing DB world)")
            return {"added": len(entries), "removed": 0, "merged": 0}
        return {"added": 0, "removed": 0, "merged": 0}
    
    db_entries = db_world.get("entries", {})
    
    # Track modification times (JSON file modification time as fallback)
    json_mtime = os.path.getmtime(file_path)
    
    # Perform intelligent merge
    merged_entries = {}
    added_count = 0
    removed_count = 0
    merged_count = 0
    
    # 1. Start with all database entries (they have modification times if edited)
    for uid, db_entry in db_entries.items():
        # Check if this entry still exists in JSON
        if uid in json_entries:
            json_entry = json_entries[uid]
            # Entry exists in both - keep the more recently modified
            # Use db_get_world_entry_hash to get hash for comparison
            db_hash = db_get_world_entry_hash(world_name, uid)
            
            # If we have hash info and it matches, consider it same
            if db_hash and json_entry.get("content") == db_entry.get("content"):
                # Same content - keep DB version (user might have edited it)
                merged_entries[uid] = db_entry
                merged_count += 1
            else:
                # Different content - prefer JSON (more recent)
                merged_entries[uid] = json_entry
                merged_count += 1
        else:
            # Entry in DB but not in JSON - remove it
            removed_count += 1
            print(f"Sync: Removed entry '{uid}' (deleted from JSON)")
    
    # 2. Add new JSON entries (not in database)
    for uid, json_entry in json_entries.items():
        if uid not in db_entries:
            merged_entries[uid] = json_entry
            added_count += 1
            print(f"Sync: Added new entry '{uid}' from JSON")
    
    # 3. Add entries that exist in both but JSON is newer
    for uid, json_entry in json_entries.items():
        if uid in db_entries:
            db_entry = db_entries[uid]
            # If content differs, use JSON version
            db_hash = db_get_world_entry_hash(world_name, uid)
            if not db_hash or json_entry.get("content") != db_entry.get("content"):
                merged_entries[uid] = json_entry
                merged_count += 1
                print(f"Sync: Updated entry '{uid}' from JSON (newer version)")
    
    # Save merged result to database and JSON
    if merged_entries:
        if db_save_world(world_name, merged_entries):
            # Update JSON file with merged result
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({"entries": merged_entries}, f, indent=2, ensure_ascii=False)
            print(f"Sync: Saved {len(merged_entries)} entries to DB and JSON")
        else:
            print(f"Sync: Failed to save merged world info to database")
    else:
        print(f"Sync: No changes to merge for world '{world_name}'")
    
    print(f"Sync Summary for '{world_name}': Added {added_count}, Removed {removed_count}, Merged {merged_count} entries")
    return {"added": added_count, "removed": removed_count, "merged": merged_count}

def cleanup_world_info_filenames():
    """Rename world info JSON files with SillyTavern suffixes to clean names.
    
    This is a one-time cleanup to ensure all world info files use standard naming:
    - exampleworld_plist_worldinfo.json -> exampleworld.json
    - exampleworld_plist.json -> exampleworld.json
    - etc.
    """
    wi_dir = os.path.join(DATA_DIR, "worldinfo")
    if not os.path.exists(wi_dir):
        return
    
    renamed_count = 0
    for f in os.listdir(wi_dir):
        if f.endswith(".json") and f != ".gitkeep":
            name = f.replace(".json", "")
            original_name = name
            
            # Remove ALL SillyTavern suffixes (keep removing until none left)
            while True:
                removed = False
                for suffix in ["_plist", "_worldinfo", "_json"]:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        removed = True
                        break
                if not removed:
                    break
            
            if name != original_name:
                # File has a suffix, rename it
                old_path = os.path.join(wi_dir, f)
                new_name = f"{name}.json"
                new_path = os.path.join(wi_dir, new_name)
                
                # Check if target file already exists
                if os.path.exists(new_path):
                    print(f"Skipping rename: {f} -> {new_name} (target already exists)")
                else:
                    try:
                        os.rename(old_path, new_path)
                        renamed_count += 1
                        print(f"Renamed world info: {f} -> {new_name}")
                    except Exception as e:
                        print(f"Failed to rename {f}: {e}")
    
    if renamed_count > 0:
        print(f"World info filename cleanup complete: {renamed_count} files renamed")

def import_world_info_json_files(force=False, smart_sync=True):
    """Import world info JSON files with intelligent merging.
    
    Args:
        force: If True, delete existing worlds before importing (useful for fixing corrupted entries)
               DEPRECATED: Smart sync is now the default and recommended approach.
        smart_sync: If True (default), performs intelligent merge of JSON and database entries.
                    If False, uses legacy behavior (force parameter).
    """
    import_count = 0
    
    wi_dir = os.path.join(DATA_DIR, "worldinfo")
    if os.path.exists(wi_dir):
        for f in os.listdir(wi_dir):
            if f.endswith(".json") and f != ".gitkeep":
                name = f.replace(".json", "")
                original_name = name
                
                # Remove ALL SillyTavern suffixes (keep removing until none left)
                while True:
                    removed = False
                    for suffix in ["_plist", "_worldinfo", "_json"]:
                        if name.endswith(suffix):
                            name = name[:-len(suffix)]
                            removed = True
                            break
                    if not removed:
                        break
                
                file_path = os.path.join(wi_dir, f)
                
                # Rename file to clean name if it had any suffixes
                if name != original_name:
                    clean_filename = f"{name}.json"
                    clean_path = os.path.join(wi_dir, clean_filename)
                    try:
                        if os.path.exists(clean_path):
                            # Target already exists, use the existing clean file
                            file_path = clean_path
                            print(f"Using existing clean file: {clean_filename}")
                        else:
                            # Rename to clean filename
                            os.rename(file_path, clean_path)
                            file_path = clean_path
                            print(f"Renamed world info: {f} -> {clean_filename}")
                    except Exception as e:
                        print(f"Failed to rename {f}: {e}")
                        # Continue with original path if rename fails
                        file_path = os.path.join(wi_dir, f)
                
                try:
                    if force:
                        # Force reimport mode (legacy behavior)
                        db_delete_world(name)
                        print(f"Force reimport: Deleted existing world '{name}'")
                        
                        # Load and import
                        with open(file_path, "r", encoding="utf-8") as wf:
                            world_data = json.load(wf)
                        # db_save_world expects entries dict, not -> full object
                        entries = world_data.get("entries", world_data) if isinstance(world_data, dict) else {}
                        if entries and db_save_world(name, entries):
                            import_count += 1
                            print(f"Imported world info: {name}")
                    elif smart_sync:
                        # Smart sync mode (new default)
                        result = sync_world_from_json(name)
                        import_count += result.get("added", 0) + result.get("merged", 0)
                    else:
                        # Legacy mode: check if already in database
                        existing = db_get_world(name)
                        if existing is None:
                            # Load and import
                            with open(file_path, "r", encoding="utf-8") as wf:
                                world_data = json.load(wf)
                            # db_save_world expects entries dict, not -> full object
                            entries = world_data.get("entries", world_data) if isinstance(world_data, dict) else {}
                            if entries and db_save_world(name, entries):
                                import_count += 1
                                print(f"Imported world info: {name}")
                except Exception as e:
                    print(f"Failed to import world info {f}: {e}")
    
    if import_count > 0:
        print(f"World info import complete: {import_count} worlds")
    else:
        print("World info import: No new JSON files to import")
    
    return import_count

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

# FastAPI startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Initialize resources when FastAPI app starts"""
    global cleanup_task, adaptive_tracker
    
    # Verify database integrity before proceeding
    if not verify_database_health():
        print("⚠️  Database health check failed - app may not function correctly")
        print("   Try running: python migrate_to_sqlite.py")
    
    # Clean up world info filenames with SillyTavern suffixes
    print("Cleaning up world info filenames...")
    cleanup_world_info_filenames()
    
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
    
    # Start periodic cleanup task when app starts
    cleanup_task = asyncio.create_task(periodic_cleanup())
    print("Periodic cleanup task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the FastAPI app shuts down"""
    global cleanup_task
    if cleanup_task:
        # Cancel the cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            print("Periodic cleanup task cancelled")

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

    def key_matches_text(key: str, text: str) -> bool:
        key_normalized = normalize_for_matching(key.strip().lower())
        if len(key_normalized) < 3 or key_normalized in GENERIC_KEYS:
            return False
        
        # Create pattern that matches both singular and plural forms
        if key_normalized.endswith('s') and len(key_normalized) > 3:
            key_base = key_normalized[:-1]
            key_pattern = re.escape(key_base).replace(r"\ ", r"\s+")
            pattern = r"\b" + key_pattern + r"s?\b"
        else:
            key_pattern = re.escape(key_normalized).replace(r"\ ", r"\s+")
            pattern = r"\b" + key_pattern + r"s?\b"
        
        text_normalized = normalize_for_matching(text)
        return re.search(pattern, text_normalized) is not None

    def content_matches_text(content: str, text: str) -> bool:
        content_normalized = normalize_for_matching(content.strip().lower())
        if len(content_normalized) < 3:
            return False
        content_pattern = re.escape(content_normalized).replace(r"\ ", r"\s+")
        pattern = r"\b" + content_pattern + r"s?\b"
        
        text_normalized = normalize_for_matching(text)
        return re.search(pattern, text_normalized) is not None

    # First, add semantic results with keyword match detection
    for content, similarity, uid in semantic_results:
        norm = _normalize_content(content)
        if not norm or norm in seen_normalized:
            continue
        
        # Check if this is also a keyword match
        entry = entries.get(uid, {})
        primary_key = entry.get("key", [])
        primary_key = [primary_key[0].lower()] if primary_key else []
        keys = primary_key + [k.lower() for k in entry.get("keysecondary", [])]
        is_keyword_match = any(key_matches_text(k, processed_text) for k in keys)
        
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
        primary_key = entry.get("key", [])
        primary_key = [primary_key[0].lower()] if primary_key else []
        keys = primary_key + [k.lower() for k in entry.get("keysecondary", [])]
        match_on_keys = any(key_matches_text(k, processed_text) for k in keys)
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

# Prompt Construction Engine
def construct_prompt(request: PromptRequest):
    settings = request.settings
    system_prompt = settings.get("system_prompt", CONFIG["system_prompt"])
    user_persona = settings.get("user_persona", "")
    user_name = settings.get("user_name", "")  # Get userName for relationships
    summary = request.summary or ""
    mode = request.mode or "narrator"
    
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
    if user_persona:
        full_prompt += f"### Your Character Description:\n{user_persona}\n"

    # === 5. WORLD KNOWLEDGE (moved up - world context frames characters) ===
    canon_law_entries = []
    if request.world_info:
        # Detect if this is an initial turn (first turn or very early in conversation)
        is_initial_turn = len(request.messages) <= 2  # First 2 turns are considered initial

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

        # === 5.5. RELATIONSHIP CONTEXT (character-to-character and character-to-user dynamics) ===
    # Only inject if there are characters to have relationships
    if request.chat_id and (request.characters or user_name):
        relationship_context = get_relationship_context(
            chat_id=request.chat_id,
            characters=active_names,  # Already extracted with get_character_name()
            user_name=user_name,
            recent_messages=request.messages[-10:]  # Last 10 messages for relevance filtering
        )
        
        if relationship_context:
            full_prompt += relationship_context + "\n"

    # === 6. CHARACTER PROFILES (characters exist in the world context) ===
    reinforcement_chunks = []
    
    for char_obj in request.characters:
        data = char_obj.get("data", {})
        name = data.get("name", "Unknown")
        description = data.get("description", "")
        depth_prompt = data.get("extensions", {}).get("depth_prompt", {}).get("prompt", "")
        multi_char_summary = data.get("extensions", {}).get("multi_char_summary", "")
        
        # Check if this character is an NPC
        is_npc = char_obj.get("is_npc", False)
        
        # Check if character has already appeared in the chat
        char_has_appeared = character_has_speaker(request.messages, name)
        
        if is_group_chat and multi_char_summary:
            # Group chat mode: use capsule on first appearance or turn 1 only
            if not char_has_appeared or len(request.messages) == 0:
                label = "[NPC]" if is_npc else "[Character]"
                print(f"[CONTEXT] {label} {name} introduced (capsule)")
                full_prompt += f"### [{name}]: {multi_char_summary}\n"
            
            # Always add to reinforcement_chunks for periodic reinforcement
            reinforcement_chunks.append(f"[{name}]: {multi_char_summary}")
        else:
            # Single character: full card on first turn only
            if len(request.messages) == 0:
                label = "[NPC]" if is_npc else "[Character]"
                print(f"[CONTEXT] {label} {name} full card on first turn")
                full_prompt += f"### Character Profile: {name}\n{description}\n"
                if depth_prompt:
                    full_prompt += f"### Context for {name}: {depth_prompt}\n"
            
            # Add depth_prompt to reinforcement_chunks for periodic reinforcement
            if depth_prompt:
                reinforcement_chunks.append(depth_prompt)

    # === 7. CHAT HISTORY (with reinforcement) ===
    full_prompt += "\n### Chat History:\n"
    reinforce_freq = settings.get("reinforce_freq", 5)
    world_reinforce_freq = settings.get("world_info_reinforce_freq", 4)  # Default to every 4 turns (optimized)
    
    # Calculate current turn number (turn = (message_count + 1) // 2, since each turn has user + assistant message)
    current_turn = (len(request.messages) + 1) // 2
    
    # Track whether canon law was added in chat history to prevent duplication
    canon_added_in_history = False
    
    # Character reinforcement logic every X turns
    if reinforce_freq > 0 and current_turn > 0 and current_turn % reinforce_freq == 0:
        print(f"[REINFORCEMENT] Turn {current_turn}: Reinforcing character profiles")
        
        if reinforcement_chunks:
            # Character reinforcement - reinforce both global characters AND active NPCs
            for chunk in reinforcement_chunks:
                # Determine if this is an NPC based on chunk format
                # NPCs are in format "[{name}]: {summary}" from capsules
                # or depth_prompt from full profiles
                is_npc_reinforcement = any(
                    char.get('is_npc', False) and 
                    char.get('data', {}).get('name', 'Unknown') in chunk
                    for char in request.characters
                )
                
                label = "[NPC]" if is_npc_reinforcement else "[Character]"
                print(f"[REINFORCEMENT] {label} profile reinforced: {chunk[:80]}...")
            
            full_prompt += "[REINFORCEMENT: " + " | ".join(reinforcement_chunks) + "]\n"
        elif is_narrator_mode:
            # Narrator reinforcement
            print(f"[REINFORCEMENT] Narrator mode reinforced")
            full_prompt += f"[REINFORCEMENT: {narrator_instruction.strip()}]\n"
    
    # World info canon law reinforcement logic every X turns
    if world_reinforce_freq > 0 and current_turn > 0 and current_turn % world_reinforce_freq == 0:
        if canon_law_entries:
            full_prompt += "[WORLD REINFORCEMENT: " + " | ".join(canon_law_entries) + "]\n"
            canon_added_in_history = True
    
    # Add chat history messages
    for msg in request.messages:
        # Filter out meta-messages like "Visual System" if they exist
        if msg.speaker == "Visual System":
            continue
        
        speaker = msg.speaker or ("User" if msg.role == "user" else "Narrator")
        full_prompt += f"{speaker}: {msg.content}\n"

    # === 8. CANON LAW (pinned for recency bias - right before generation) ===
    # Add at the end ONLY if it wasn't already added in the chat history
    # This ensures canon law appears exactly once per prompt for maximum efficiency
    if canon_law_entries and not canon_added_in_history:
        full_prompt += "\n### Canon Law (World Rules):\n" + "\n".join(canon_law_entries) + "\n"

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
            [List 4-6 personality traits, comma-separated, like: "brave", "loyal", "gruff"]

            PHYSICAL_TRAITS:
            [List 4-6 physical/body traits, comma-separated, like: "tall", "muscular", "scarred"]

            SCENARIO:
            [Write a scenario describing the setting/situation where you meet this character]

            FIRST_MESSAGE:
            [Write the character's first message/greeting using *asterisks* for actions and "quotes" for speech]

            FULL_DESCRIPTION:
            [Write a detailed description of the character's appearance, background, and mannerisms]

            GENRE:
            [One word genre like: Fantasy, SciFi, Modern, etc.]

            TAGS:
            [3-5 relevant tags, comma-separated]"""

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
                genre = "General"
                tags = ["generated", "npc"]
                
                print(f"[NPC_CREATE] Step 6b: Parsing LLM response")
                print(f"[NPC_CREATE] Step 6b: LLM raw response (first 500 chars): {llm_result[:500] if llm_result and len(llm_result) > 500 else llm_result}...")
                
                # Extract sections from LLM response
                if "FULL_DESCRIPTION:" in llm_result:
                    description = llm_result.split("FULL_DESCRIPTION:")[1].strip().split("\n\n")[0]
                    print(f"[NPC_CREATE] Extracted FULL_DESCRIPTION: {description[:80]}...")
                
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
                    first_message = llm_result.split("FIRST_MESSAGE:")[1].split("FULL_DESCRIPTION:")[0].strip()
                    print(f"[NPC_CREATE] Extracted FIRST_MESSAGE: {first_message[:80]}...")
                
                if "GENRE:" in llm_result:
                    genre = llm_result.split("GENRE:")[1].split("TAGS:")[0].strip()
                    print(f"[NPC_CREATE] Extracted GENRE: {genre}")
                
                if "TAGS:" in llm_result:
                    tags_text = llm_result.split("TAGS:")[1].strip().split("\n")[0]
                    tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                    print(f"[NPC_CREATE] Extracted TAGS: {tags}")
                
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
                genre = "General"
                tags = ["generated", "npc"]

            print(f"[NPC_CREATE] Step 7: Building character data structure")
            print(f"[NPC_CREATE] Step 7a: Character name: {char_name}")
            print(f"[NPC_CREATE] Step 7b: Description: {description[:80]}...")
            print(f"[NPC_CREATE] Step 7c: Personality traits: {personality_traits}")
            print(f"[NPC_CREATE] Step 7d: Physical traits: {physical_traits}")
            print(f"[NPC_CREATE] Step 7e: Scenario: {scenario[:80]}...")
            print(f"[NPC_CREATE] Step 7f: First message: {first_message[:80]}...")
            print(f"[NPC_CREATE] Step 7g: Genre: {genre}, Tags: {tags}")
            
            # Build PList format for creator_notes and depth_prompt
            personality_plist = ', '.join([f'"{trait}"' for trait in personality_traits])
            body_plist = ', '.join([f'"{trait}"' for trait in physical_traits]) if physical_traits else '"mysterious figure"'
            tags_list = ', '.join(tags)

            plist_format = f"""[{char_name}'s Personality= {personality_plist}]
            [{char_name}'s body= {body_plist}]
            [Genre: {genre}; Tags: {tags_list}]"""

            # Add source information to creator_notes
            source_info = f"Created from {req.source_mode}: {user_input[:100]}\n\n" if req.source_mode else ""

            # Create full character card with generated data
            character_data = {
                "name": char_name,
                "description": description,
                "personality": "",  # Empty per your example
                "scenario": scenario,
                "first_mes": first_message,
                "mes_example": "",
                "creator_notes": source_info + plist_format,
                "system_prompt": "",
                "post_history_instructions": "",
                "alternate_greetings": [],
                "tags": tags,
                "creator": "NeuralRP NPC Generator",
                "extensions": {
                    "depth_prompt": {
                        "prompt": plist_format,
                        "depth": 4
                    },
                    "talkativeness": 100
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
            prompt = f"""Based on the {source_desc}, identify {char_name}'s personality traits.
Convert them into a PList personality array format.
Use this exact format: [{char_name}'s Personality= "trait1", "trait2", "trait3", ...]

Source Text:
{user_input}

Only output the personality array line, nothing else."""
            result = await call_llm_helper(system, prompt, 300)
            
        elif field_type == 'body':
            system = f"You are an expert at analyzing characters and writing physical descriptions."
            prompt = f"""Based on the {source_desc}, identify {char_name}'s physical features.
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
[List 4-6 personality traits, comma-separated, like: "brave", "loyal", "gruff"]

PHYSICAL_TRAITS:
[List 4-6 physical/body traits, comma-separated, like: "tall", "muscular", "scarred"]

SCENARIO:
[Write a scenario describing the setting/situation where you meet this character]

FIRST_MESSAGE:
[Write the character's first message/greeting using *asterisks* for actions and "quotes" for speech]

FULL_DESCRIPTION:
[Write a detailed description of the character's appearance, background, and mannerisms]

GENRE:
[One word genre like: Fantasy, SciFi, Modern, etc.]

TAGS:
[3-5 relevant tags, comma-separated]"""

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

# PList Formatting Helper Functions
def extract_plist_from_llm_output(result: str, field_type: str, char_name: str) -> str:
    """Extract PList-formatted content from LLM output and ensure proper format."""
    import re
    
    # If result already starts with bracket, it's likely PList formatted
    if result.strip().startswith('['):
        # Extract all bracketed entries
        bracketed_entries = re.findall(r'\[.+?\]', result)
        
        # Clean up entries
        lines = [entry.strip() for entry in bracketed_entries if entry.strip()]
        
        if lines:
            return '\n'.join(lines)
    
    # Try to parse structured format if LLM returned sections
    # This handles cases where LLM uses section headers like PERSONALITY_TRAITS:, PHYSICAL_TRAITS:, etc.
    result_lower = result.lower()
    
    # Handle different field types
    if field_type in ['personality', 'body']:
        # For personality/body, extract trait lists and format as PList
        trait_pattern = r'(?:personality_traits|physical_traits|personality|body)[\s:\s*]*([^\n]*)'
        match = re.search(trait_pattern, result_lower, re.IGNORECASE)
        
        if match:
            traits_text = match.group(1)
            # Extract traits from comma-separated or quoted list
            if '"' in traits_text or "'" in traits_text:
                # Quoted traits
                traits = re.findall(r'["\']([^"\']+)["\']', traits_text)
            else:
                # Unquoted traits (comma-separated)
                traits = [t.strip() for t in traits_text.split(',') if t.strip()]
            
            if traits:
                # Format as PList
                trait_list = ', '.join([f'"{trait.strip()}"' for trait in traits])
                plist_key = 'Personality' if field_type == 'personality' else 'body'
                return f"[{char_name}'s {plist_key}= {trait_list}]"
    
    elif field_type == 'full':
        # Full character card generation - extract multiple sections
        # Extract personality traits
        personality_traits = []
        if 'personality_traits:' in result_lower:
            traits_match = re.search(r'personality_traits[:\s*]*([^\n]*)', result_lower, re.IGNORECASE)
            if traits_match:
                traits_text = traits_match.group(1)
                personality_traits = [t.strip().strip('"').strip("'") for t in traits_text.split(',') if t.strip()]
        
        # Extract physical traits
        physical_traits = []
        if 'physical_traits:' in result_lower:
            phys_match = re.search(r'physical_traits[:\s*]*([^\n]*)', result_lower, re.IGNORECASE)
            if phys_match:
                phys_text = phys_match.group(1)
                physical_traits = [t.strip().strip('"').strip("'") for t in phys_text.split(',') if t.strip()]
        
        # Extract genre and tags
        genre = "General"
        tags = []
        if 'genre:' in result_lower:
            genre_match = re.search(r'genre[:\s*]*([^\n]*)', result_lower, re.IGNORECASE)
            if genre_match:
                genre = genre_match.group(1).strip()
        
        if 'tags:' in result_lower:
            tags_match = re.search(r'tags[:\s*]*([^\n]*)', result_lower, re.IGNORECASE)
            if tags_match:
                tags_text = tags_match.group(1)
                tags = [t.strip() for t in tags_text.split(',') if t.strip()]
        
        # Extract scenario
        scenario = f"You encounter {char_name}."
        if 'scenario:' in result_lower:
            scenario_match = re.search(r'scenario[:\s*]*([^\n]*)', result_lower, re.IGNORECASE)
            if scenario_match:
                scenario = scenario_match.group(1).strip()
        
        # Extract first message
        first_message = f"*{char_name} looks at you.*"
        if 'first_message:' in result_lower:
            first_match = re.search(r'first_message[:\s*]*([^\n]+)', result_lower, re.IGNORECASE)
            if first_match:
                first_message = first_match.group(1).strip()
        
        # Extract description
        description = result
        if 'full_description:' in result_lower:
            desc_match = re.search(r'full_description[:\s*]*([^\n]+)', result_lower, re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1).strip()
        
        # Format as multiple PList entries (like NPC creation)
        personality_plist = ', '.join([f'"{trait}"' for trait in personality_traits]) if personality_traits else '"mysterious"'
        body_plist = ', '.join([f'"{trait}"' for trait in physical_traits]) if physical_traits else '"mysterious figure"'
        tags_list = ', '.join([f'"{tag}"' for tag in tags]) if tags else '"generated"'
        
        plist_format = f"""[{char_name}'s Personality= {personality_plist}]
[{char_name}'s body= {body_plist}]
[Genre: {genre}; Tags: {tags_list}]"""
        
        return plist_format
    
    # Fallback: try to extract any bracketed content
    bracketed_entries = re.findall(r'\[.+?\]', result)
    if bracketed_entries:
        lines = [entry.strip() for entry in bracketed_entries if entry.strip()]
        return '\n'.join(lines)
    
    # Last fallback: return original result
    return result.strip()

# Capsule Generation for Multi-Character Optimization
async def generate_capsule_for_character(char_name: str, description: str, depth_prompt: str = "") -> str:
    """Generate a capsule summary for use in multi-character scenarios."""
    system = "You are an expert at distilling roleplay character cards into minimal capsule summaries for efficient multi-character prompts."
    
    full_card_text = f"Name: {char_name}\n\nDescription/Dialogue:\n{description}"
    if depth_prompt:
        full_card_text += f"\n\nPersonality/Context:\n{depth_prompt}"
    
    prompt = f"""Convert this character card into a capsule summary for efficient multi-character prompts.
Use this exact format (one paragraph, no line breaks):
Name: [Name]. Role: [1 sentence role/situation]. Key traits: [3-5 comma-separated personality traits]. Speech style: [short/long, formal/casual, any verbal tics]. Example line: "[One characteristic quote from descriptions]"

Full Card:
{full_card_text}

Output only the capsule summary line, nothing else."""

    result = await call_llm_helper(system, prompt, 200)
    return result.strip()

# Mode Classification Request Model
class ModeClassifyRequest(BaseModel):
    user_message: str
    character_names: List[str]

@app.post("/api/classify-mode")
async def classify_mode(req: ModeClassifyRequest):
    """Auto-classify which mode should be used based on the user's message."""
    if len(req.character_names) < 2:
        # Single or no characters - use simple rules
        if len(req.character_names) == 1:
            return {"success": True, "mode": f"focus:{req.character_names[0]}"}
        return {"success": True, "mode": "narrator"}
    
    # Multiple characters - use LLM to classify
    char_list = ", ".join(req.character_names)
    system = "You are a classifier that determines who should respond in a roleplay scene."
    
    prompt = f"""Given this user message in a roleplay with multiple characters, determine who should respond.

Active characters: {char_list}

User message: "{req.user_message}"

Rules:
- If the user directly addresses a character by name (e.g., "Alice, what do you think?"), output: focus:CharacterName
- If the user asks a question that could be answered by any character, or describes an action affecting multiple characters, output: narrator
- If the context suggests an intimate or personal moment with a specific character, output: focus:CharacterName
- When in doubt, default to: narrator

Output ONLY one of these options, nothing else:
- narrator
- focus:{char_list.split(', ')[0]}
- focus:{char_list.split(', ')[1] if len(req.character_names) > 1 else char_list.split(', ')[0]}
{('- focus:' + char_list.split(', ')[2] if len(req.character_names) > 2 else '')}

Your answer:"""

    try:
        result = await call_llm_helper(system, prompt, 30)
        result = result.strip().lower()
        
        # Validate the result
        if result == "narrator":
            return {"success": True, "mode": "narrator"}
        elif result.startswith("focus:"):
            char_name = result.split(":", 1)[1].strip()
            # Find matching character (case insensitive)
            for name in req.character_names:
                if name.lower() == char_name.lower():
                    return {"success": True, "mode": f"focus:{name}"}
            # If no exact match, fall back to narrator
            return {"success": True, "mode": "narrator"}
        else:
            # Unrecognized output, default to narrator
            return {"success": True, "mode": "narrator"}
    except Exception as e:
        print(f"Mode classification failed: {e}")
        return {"success": True, "mode": "narrator"}

@app.post("/api/card-gen/generate-capsule")
async def generate_capsule(req: CapsuleGenRequest):
    """API endpoint to generate a capsule summary for a character."""
    try:
        capsule = await generate_capsule_for_character(req.char_name, req.description, req.depth_prompt)
        return {"success": True, "text": capsule}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
        if req.field not in ["personality", "body", "dialogue", "genre", "tags", "scenario", "first_message"]:
            return {"success": False, "error": "Invalid field"}
        
        # Update the field
        char_data["data"][req.field] = req.new_value
        
        # Save the updated character
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, indent=2, ensure_ascii=False)
        
        return {"success": True, "message": f"Field '{req.field}' updated successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/characters/edit-field-ai")
async def edit_character_field_ai(req: CharacterEditFieldRequest):
    """Use AI to generate or improve a specific field in a character card."""
    try:
        file_path = os.path.join(DATA_DIR, "characters", req.filename)
        if not os.path.exists(file_path):
            return {"success": False, "error": "Character file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            char_data = json.load(f)
        
        char_name = char_data["data"]["name"]
        
        # Use existing generation logic with context
        card_req = CardGenRequest(
            char_name=char_name,
            field_type=req.field,
            context=req.context,
            source_mode=req.source_mode
        )
        
        # Call the existing generation endpoint logic
        result = await generate_card_field(card_req)
        
        if result["success"]:
            # Update the character with the generated content
            char_data["data"][req.field] = result["text"]
            
            # Save the updated character
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(char_data, f, indent=2, ensure_ascii=False)
            
            return {"success": True, "text": result["text"]}
        else:
            return {"success": False, "error": result["error"]}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/characters/edit-capsule")
async def edit_character_capsule(req: CharacterEditRequest):
    """Edit the multi-character capsule for a character."""
    try:
        file_path = os.path.join(DATA_DIR, "characters", req.filename)
        if not os.path.exists(file_path):
            return {"success": False, "error": "Character file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            char_data = json.load(f)
        
        # Ensure extensions object exists
        if "extensions" not in char_data["data"]:
            char_data["data"]["extensions"] = {}
        
        # Update the capsule
        char_data["data"]["extensions"]["multi_char_summary"] = req.new_value
        
        # Save the updated character
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, indent=2, ensure_ascii=False)
        
        return {"success": True, "message": "Capsule updated successfully"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/characters/edit-capsule-ai")
async def edit_character_capsule_ai(req: CharacterEditFieldRequest):
    """Use AI to generate or improve the multi-character capsule."""
    try:
        file_path = os.path.join(DATA_DIR, "characters", req.filename)
        if not os.path.exists(file_path):
            return {"success": False, "error": "Character file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            char_data = json.load(f)
        
        char_name = char_data["data"]["name"]
        description = char_data["data"]["description"]
        depth_prompt = char_data["data"]["extensions"].get("depth_prompt", {}).get("prompt", "")
        
        # Use existing capsule generation logic
        capsule_req = CapsuleGenRequest(
            char_name=char_name,
            description=description,
            depth_prompt=depth_prompt
        )
        
        result = await generate_capsule(capsule_req)
        
        if result["success"]:
            # Update the character with the generated capsule
            if "extensions" not in char_data["data"]:
                char_data["data"]["extensions"] = {}
            
            char_data["data"]["extensions"]["multi_char_summary"] = result["text"]
            
            # Save the updated character
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(char_data, f, indent=2, ensure_ascii=False)
            
            return {"success": True, "text": result["text"]}
        else:
            return {"success": False, "error": result["error"]}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# World Info Generation Logic (Ported from world-gen-app)
WORLD_PROMPTS = {
    "history": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory mentioned. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects)]\n\nRules:\n- One entry per line\n- Keep entries concise\n- Use parentheses for nested attributes\n- Output ONLY the PList entries, no explanations\n- Match the tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory mentioned in the chat. Use this exact format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects)]\n\nRules:\n- One entry per line\n- Keep entries concise\n- Use parentheses for nested attributes\n- Output ONLY the PList entries, no explanations",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature/suggestive themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory. Include mature themes where appropriate. Use this format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects), intimate_aspects(relationships, desires, power dynamics)]\n\nRules:\n- One entry per line\n- Include suggestive details naturally\n- Output ONLY the PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event or backstory. Include explicit adult themes naturally. Use this format:\n[EventName: type(event/era/myth), time(when), actors(who), result(what happened), legacy(effects), sexual_culture(attitudes, practices, taboos)]\n\nRules:\n- One entry per line\n- Include explicit adult content where relevant\n- Output ONLY the PList entries"
    },
    "locations": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each location, room, town, or area mentioned. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[LocationName(nickname if any): type(room/town/area), features(physical details), atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]\n\nRules:\n- One entry per line\n- Output ONLY the PList entries\n- Match the tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location, room, town, or area mentioned. Use this format:\n[LocationName(nickname if any): type(room/town/area), features(physical details), atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include suggestive atmosphere and uses. Use this format:\n[LocationName(nickname): type, features, atmosphere(mood), purpose(what happens), inhabitants(who), intimate_uses(private activities)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include explicit adult themes naturally. Format:\n[LocationName(nickname): type, features, atmosphere, purpose, inhabitants, sexual_activities(what happens), kinks_associated(themes)]\n\nRules:\n- One entry per line- Output ONLY the PList entries"
    },
    "creatures": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each creature, monster, or character archetype. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[CreatureName: type(creature/archetype), appearance(visual traits), behavior(typical actions), culture(social norms, beliefs), habitat(where found), attitude_toward_user(how they treat {{user}})]\n\nRules:\n- One entry per line\n- Output ONLY the PList entries\n- Match the tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each creature, monster, or character archetype. Use this format:\n[CreatureName: type(creature/archetype), appearance(visual traits), behavior(typical actions), culture(social norms, beliefs), habitat(where found), attitude_toward_user(how they treat {{user}})]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for creatures or archetypes. Include suggestive behavior and interactions. Format:\n[CreatureName: type, appearance, behavior, culture, habitat, attitude_toward_user, flirtation_style(how they seduce/interact)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[CreatureName: type, appearance, behavior, culture, attitude_toward_user, sexual_behavior(explicit details), kinks(preferences), consent_culture(boundaries)]\n\nRules:\n- One entry per line- Output ONLY the PList entries"
    },
    "factions": {
        "neutral": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nContext:\n{input}\n\nCreate PList-format World Info entries for each faction, group, or organization. Adapt naturally to the content tone (SFW or NSFW as appropriate). Use this format:\n[FactionName: type(faction/guild/house/clique), members(who belongs), reputation(public image), goals(what they want), methods(how they operate), attitude_toward_user(how they treat {{user}}), rivals(opposing factions)]\n\nRules:\n- One entry per line\n- Output ONLY the PList entries\n- Match the tone and maturity level of the provided context naturally",
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each faction, group, or organization. Use this format:\n[FactionName: type(faction/guild/house/clique), members(who belongs), reputation(public image), goals(what they want), methods(how they operate), attitude_toward_user(how they treat {{user}}), rivals(opposing factions)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for factions. Include suggestive motivations and interactions. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, social_dynamics(power, romance, rivalries), intimate_culture(dating norms, boundaries)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, sexual_culture(practices, rituals), kinks_favored(group preferences), initiation(how to join)]\n\nRules:\n- One entry per line- Output ONLY the PList entries"
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
                    "depth": 5
                }
                uid_counter += 1
        
        if not entries: return {"success": False, "error": "No valid entries found"}
        
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)
            
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
                
                # Add to profiles
                character_profiles.append({
                    'name': npc['name'],
                    'data': npc['data'],
                    'entity_id': char_ref,
                    'is_npc': True
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
                    'is_npc': False
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

    # ========== CHARACTER RESOLUTION ==========
    # Check if characters are already resolved (dicts with 'data' field)
    # This happens on subsequent messages after first resolution
    if request.characters and isinstance(request.characters[0], dict) and "data" in request.characters[0]:
        # Characters already resolved - skip resolution
        print(f"[CONTEXT] Characters already resolved ({len(request.characters)} chars)")
    else:
        # Load chat data if needed
        if request.chat_id:
            chat_data = db_get_chat(request.chat_id)

        # Resolve character references (global + NPCs)
        # (you may have more logic here)

    # Now chat_data is always defined (None or a dict)
    if chat_data:
        metadata = chat_data.get("metadata", {}) or {}
        localnpcs = metadata.get("localnpcs", {}) or {}
        active_chars = chat_data.get("activeCharacters", [])

        # Resolve character references to full objects
        resolved_characters = load_character_profiles(active_chars, localnpcs)
        request.characters = resolved_characters


    
    # Auto-generate capsules for group chats (2+ characters)
    updated_characters = []
    chars_needing_capsules = []
    
    if len(request.characters) >= 2:
        for char_obj in request.characters:
            data = char_obj.get("data", {})
            extensions = data.get("extensions", {})
            multi_char_summary = extensions.get("multi_char_summary", "")
            
            if not multi_char_summary:
                # Character needs a capsule generated
                chars_needing_capsules.append(char_obj)
            updated_characters.append(char_obj)
        
        # Generate capsules for characters that need them
        for char_obj in chars_needing_capsules:
            data = char_obj.get("data", {})
            name = data.get("name", "Unknown")
            description = data.get("description", "")
            depth_prompt = data.get("extensions", {}).get("depth_prompt", {}).get("prompt", "")
            
            try:
                print(f"AUTO-GENERATING capsule for {name} (group chat detected)")
                capsule = await generate_capsule_for_character(name, description, depth_prompt)
                
                # Update the character object in memory
                if "extensions" not in data:
                    data["extensions"] = {}
                data["extensions"]["multi_char_summary"] = capsule
                
                # Save the updated capsule to the character file or NPC data
                filename = char_obj.get("_filename")
                if filename:
                    file_path = os.path.join(DATA_DIR, "characters", filename)
                    save_data = char_obj.copy()
                    if "_filename" in save_data:
                        del save_data["_filename"]
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                    print(f"SAVED capsule for {name}: {capsule[:80]}...")
                elif char_obj.get("is_npc", False):
                    npc_id = char_obj.get("entity_id")
                    if npc_id and request.chat_id:
                        try:
                            npc_data = data.copy()
                            db_update_npc(request.chat_id, npc_id, npc_data)
                            chat = db_get_chat(request.chat_id)
                            if chat:
                                metadata = chat.get("metadata", {}) or {}
                                localnpcs = metadata.get("localnpcs", {}) or {}
                                if npc_id in localnpcs:
                                    localnpcs[npc_id]["data"] = npc_data
                                    metadata["localnpcs"] = localnpcs
                                    chat["metadata"] = metadata
                                    db_save_chat(request.chat_id, chat)
                                    print(f"SAVED capsule for NPC {name}: {capsule[:80]}...")
                                else:
                                    print(f"[WARNING] NPC {npc_id} not found in metadata, capsule not saved")
                            else:
                                print(f"[WARNING] Chat {request.chat_id} not found, NPC capsule not saved")
                        except Exception as e:
                            print(f"Failed to save NPC capsule for {name}: {e}")
                else:
                    print(f"[WARNING] Character {name} has no _filename and is not NPC, capsule not saved")
            except Exception as e:
                print(f"Failed to auto-generate capsule for {name}: {e}")
        
        # Update the request with the modified characters
        request.characters = updated_characters
    
    # Check for summarization need
    max_ctx = request.settings.get("max_context", 4096)
    threshold = request.settings.get("summarize_threshold", 0.85)
    
    current_request = request
    new_summary = request.summary or ""
    
    # Initial token check
    prompt = construct_prompt(current_request)
    tokens = await get_token_count(prompt)
    
    # Summarization loop
    if tokens > max_ctx * threshold and len(current_request.messages) > 10:
        print(f"TRUNCATION TRIGGERED: {tokens} tokens exceeds {max_ctx * threshold}")

        # Extract Canon Law entries to echo into summary if needed
        canon_echo = ""
        if request.world_info:
            canon_entries = [e.get("content", "") for e in request.world_info.get("entries", {}).values() if e.get("is_canon_law")]
            if canon_entries:
                canon_echo = "\n### Note: Active Canon Laws to maintain:\n" + "\n".join(canon_entries)

        # Take oldest 10 messages to summarize
        to_summarize = current_request.messages[:10]
        remaining_messages = current_request.messages[10:]
        
        summary_text = "\n".join([f"{m.speaker or m.role}: {m.content}" for m in to_summarize])
        summarization_prompt = f"### System: Summarize the following conversation snippet into a very concise narrative for long-term memory. Focus on key plot points and character states.{canon_echo}\n\n### Conversation to summarize:\n{summary_text}\n\n### Concise Summary:"
        
        async with httpx.AsyncClient() as client:
            try:
                sum_res = await client.post(
                    f"{CONFIG['kobold_url']}/api/v1/generate",
                    json={
                        "prompt": summarization_prompt,
                        "max_length": 150,
                        "temperature": 0.5,
                        "stop_sequence": ["###", "\nUser:", "\nAssistant:"]
                    },
                    timeout=60.0
                )
                sum_data = sum_res.json()
                added_summary = sum_data["results"][0]["text"].strip()
                new_summary = (new_summary + "\n" + added_summary).strip()
                
                # Rebuild request with new summary and fewer messages
                current_request.messages = remaining_messages
                current_request.summary = new_summary
                prompt = construct_prompt(current_request)
                tokens = await get_token_count(prompt)
            except Exception as e:
                print(f"Summarization failed: {e}")
        
        # === RELATIONSHIP ANALYSIS (Step 5 of relationship tracker) ===
        # Trigger relationship analysis at summarization boundary
        # Extract character names using get_character_name() for consistency
        active_character_names = []
        for char_obj in request.characters:
            active_character_names.append(get_character_name(char_obj))
        
        await analyze_and_update_relationships(
            chat_id=current_request.chat_id,
            messages=current_request.messages,
            characters=active_character_names,
            user_name=user_name
        )

    print(f"Generated Prompt ({tokens} tokens):\n{prompt}")
    
    # Extract settings for generation
    temp = request.settings.get("temperature", 0.7)
    max_len = request.settings.get("max_length", 250)
    mode = request.mode or "narrator"

    # Calculate stop sequences (mode-aware)
    # Include reinforcement markers to prevent LLM from generating them in output
    stops = ["User:", "\nUser", "###", "[REINFORCEMENT:", "[WORLD REINFORCEMENT:", "\n["]
    
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
            
            # Update world info reinforcement counter in chat metadata
            world_reinforce_freq = current_request.settings.get("world_info_reinforce_freq", 3)
            if world_reinforce_freq > 0:
                # Calculate current counter based on message count
                current_counter = len(current_request.messages)
                
                # Initialize metadata dict if not present
                if current_request.metadata is None:
                    current_request.metadata = {}
                current_request.metadata['world_reinforce_counter'] = current_counter
            
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
            
            return data
        except Exception as e:
            duration = time.time() - start_time
            performance_tracker.record_llm(duration, context_tokens=tokens)
            return {"error": str(e)}
    else:
        # Direct call when performance mode is disabled
        try:
            return await llm_operation()
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
        
        for name in bracketed_names:
            # Case-insensitive match for name
            matched_char = next((c for c in all_chars if c.get("data", {}).get("name", "").lower() == name.lower()), None)
            if matched_char:
                # Get danbooru_tag from extensions (already injected by db_get_all_characters)
                danbooru_tag = matched_char.get("data", {}).get("extensions", {}).get("danbooru_tag", "")
                if danbooru_tag:
                    # Replace [Name] with the tag
                    processed_prompt = processed_prompt.replace(f"[{name}]", danbooru_tag)

    # Define the SD operation to be managed
    async def sd_operation():
        # Apply context-aware preset if performance mode is enabled
        if resource_manager.performance_mode_enabled:
            preset = select_sd_preset(context_tokens)
            # Use preset values if they differ from user input
            final_steps = preset["steps"] if preset["steps"] != 20 else params.steps
            final_width = preset["width"] if preset["width"] != 512 else params.width
            final_height = preset["height"] if preset["height"] != 512 else params.height
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
    """Manually trigger re-import of character JSON files only."""
    try:
        import_count = import_characters_json_files()
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
                total_removed = 0
                total_merged = 0
                
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
                        total_removed += result.get("removed", 0)
                        total_merged += result.get("merged", 0)
                
                message = f"Synced: {total_added} added, {total_removed} removed, {total_merged} merged entries from JSON files"
                return {
                    "success": True,
                    "synced": {
                        "added": total_added,
                        "removed": total_removed,
                        "merged": total_merged,
                        "total": total_added + total_removed + total_merged
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
async def list_characters():
    """Get all characters from database."""
    try:
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
        
        # Save to database
        if not db_save_character(char, filename):
            return {"success": False, "error": "Failed to save to database"}
        
        # Auto-export to JSON for SillyTavern compatibility
        file_path = os.path.join(DATA_DIR, "characters", filename)
        save_data = char.copy()
        if "_filename" in save_data:
            del save_data["_filename"]
        
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

@app.get("/api/world-info")
async def list_world_info():
    """Get all world info from database."""
    try:
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
    UPDATED: Now handles NPC entity ID remapping and relationship copying.
    """
    origin_chat_name = request.origin_chat_name
    fork_from_message_id = request.fork_from_message_id
    branch_name = request.branch_name
    
    # 1. Load origin chat
    origin_chat = db_get_chat(origin_chat_name)
    if not origin_chat:
        return {"success": False, "error": "Origin chat not found"}
    
    # 2. Find fork point and extract messages
    messages = origin_chat.get('messages', [])
    fork_index = None
    for i, msg in enumerate(messages):
        if msg.get('id') == fork_from_message_id:
            fork_index = i
            break
    
    if fork_index is None:
        return {"success": False, "error": "Fork message not found"}
    
    # 3. Create branch data (messages up to fork point)
    branch_messages = messages[:fork_index + 1]
    
    # 4. Generate branch chat ID
    branch_chat_id = f"{origin_chat_name}_fork_{int(time.time())}"

    # 5. Copy metadata with NPCs
    branch_metadata = origin_chat.get("metadata", {}).copy()
    localnpcs = branch_metadata.get("localnpcs", {}).copy()

    # 6. CRITICAL: Remap NPC entity IDs for branch safety
    entity_mapping = db_copy_npcs_for_branch(
        origin_chat_name,
        branch_chat_id,
        localnpcs  # modified in place
    )

    branch_metadata["localnpcs"] = localnpcs

    
    # 7. Update activeCharacters with new NPC entity IDs
    active_chars = origin_chat.get('activeCharacters', []).copy()
    updated_active_chars = []
    for char_ref in active_chars:
        if char_ref in entity_mapping:
            # Replace old NPC ID with new branch NPC ID
            updated_active_chars.append(entity_mapping[char_ref])
        else:
            # Keep global character filenames as-is
            updated_active_chars.append(char_ref)
    
    # 8. Assemble branch data
    branch_data = {
        'messages': branch_messages,
        'summary': origin_chat.get('summary', ''),
        'activeCharacters': updated_active_chars,
        'activeWI': origin_chat.get('activeWI'),
        'settings': origin_chat.get('settings', {}).copy(),
        'metadata': {
            **branch_metadata,
            'origin_chat_id': origin_chat_name,
            'origin_message_id': fork_from_message_id,
            'branch_name': branch_name or f"Fork from message {fork_from_message_id}",
            'created_at': time.time()
        }
    }
    
    # 9. Copy relationship states with entity ID mapping BEFORE saving branch
    # This ensures entity_mapping is complete before branch data is persisted
    db_copy_relationship_states_with_mapping(
        origin_chat_name,
        branch_chat_id,
        fork_from_message_id,
        entity_mapping
    )
    
    # 10. Save branch chat AFTER relationship copying
    db_save_chat(branch_chat_id, branch_data, autosaved=True)
    
    print(f"[FORK] Created branch {branch_chat_id} from {origin_chat_name} at message {fork_from_message_id}")
    print(f"[FORK] Remapped {len(entity_mapping)} NPC entity IDs")
    
    return {
        "success": True,
        "name": branch_chat_id,
        "branch_name": branch_name,
        "origin_chat_name": origin_chat_name,
        "fork_from_message_id": fork_from_message_id,
        "remapped_entities": len(entity_mapping)
    }

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
            from app.database import db_create_npc_with_entity_id
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
                    "created_at": int(time.time())
                }
            
            metadata["localnpcs"] = localnpcs
            chat["metadata"] = metadata
            db_save_chat(chat_id, chat)

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
    
    Returns:
        {
            "success": bool,
            "filename": str,  # Global character filename
            "global_entity_id": str,  # New entity ID for global character
            "promoted_npc_id": str  # Original NPC ID (now promoted)
        }
    """
    try:
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
            localnpcs = metadata.get('localnpcs', {})
            if npc_id in localnpcs:
                metadata_npc = localnpcs[npc_id]
                npc_data = {
                    "entityid": npc_id,
                    "entity_id": npc_id,
                    "name": metadata_npc.get("name", "Unknown"),
                    "data": metadata_npc.get("data", {}),
                    "is_active": metadata_npc.get("is_active", True),
                    "promoted": metadata_npc.get("promoted", False)
                }
                print(f"[NPC_PROMOTE] NPC {npc_id} found in metadata only, proceeding with promotion")
        
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
        
        # 8. Update activeCharacters to use global character filename
        active_chars = chat.get('activeCharacters', [])
        if npc_id in active_chars:
            active_chars.remove(npc_id)
        if filename not in active_chars:
            active_chars.append(filename)
        
        # 9. Mark NPC as promoted in database (if it exists there)
        db_set_npc_active(chat_id, npc_id, False)  # Deactivate in DB if exists
        
        # 10. Save updated chat with NPC removed from metadata
        metadata['localnpcs'] = localnpcs
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
        
        npc_name = "Unknown"
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
    
    Returns:
        {
            "success": bool,
            "npc_id": str,
            "is_active": bool,
            "message": str
        }
    """
    try:
        # Load chat
        chat = db_get_chat(chat_id)
        if not chat:
            return {"success": False, "error": "Chat not found"}
        
        metadata = chat.get('metadata', {}) or {}
        localnpcs = metadata.get('localnpcs', {}) or {}
        
        # Check database first (authoritative source)
        db_npcs = db_get_chat_npcs(chat_id)
        db_npc = None
        for npc in db_npcs:
            if npc.get('entityid') == npc_id or npc.get('entity_id') == npc_id:
                db_npc = npc
                break
        
        # Check metadata as fallback
        metadata_npc = localnpcs.get(npc_id)
        
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
        valid_fields = ["content", "key", "is_canon_law", "probability", "useProbability"]
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
async def add_world_entry(world_name: str, entry_data: dict):
    """Add a new world info entry."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{world_name}.json")
        
        # Load existing world info or create new
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                world_data = json.load(f)
        else:
            world_data = {"entries": {}}
        
        # Generate new UID
        uid_counter = max([int(k) for k in world_data["entries"].keys()] + [0]) + 1
        
        # Create new entry
        new_entry = {
            "uid": uid_counter,
            "key": entry_data.get("key", []),
            "keysecondary": [],
            "comment": entry_data.get("comment", ""),
            "content": entry_data.get("content", ""),
            "constant": entry_data.get("constant", False),
            "selective": entry_data.get("selective", True),
            "selectiveLogic": entry_data.get("selectiveLogic", 0),
            "addMemo": entry_data.get("addMemo", True),
            "order": entry_data.get("order", 100),
            "position": entry_data.get("position", 4),
            "disable": entry_data.get("disable", False),
            "excludeRecursion": entry_data.get("excludeRecursion", False),
            "probability": entry_data.get("probability", 100),
            "useProbability": entry_data.get("useProbability", True),
            "displayIndex": uid_counter,
            "depth": entry_data.get("depth", 5)
        }
        
        # Add the entry
        world_data["entries"][str(uid_counter)] = new_entry
        
        # Save the updated world info
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        
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
            "world_info_reinforce_freq": 4,  # Default value
            "description": "Frequency (in turns) for reinforcing canon law entries. 1 = every turn, 4 = every 4 turns, etc.",
            "default": 4,
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
            input_text = system_prompt if i == 0 else ""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
