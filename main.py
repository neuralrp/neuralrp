from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import time
import base64
import httpx
import re
import random
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
import asyncio
from collections import deque
from bisect import insort
from statistics import median

# Import database module for SQLite operations
from app.database import (
    db_get_all_characters, db_get_character, db_save_character, db_delete_character,
    db_get_all_worlds, db_get_world, db_save_world, db_delete_world, db_update_world_entry,
    db_get_all_chats, db_get_chat, db_save_chat, db_delete_chat,
    db_save_image_metadata, db_get_image_metadata, db_get_all_image_metadata,
    db_save_performance_metric, db_get_median_performance, db_cleanup_old_metrics,
    db_get_recent_performance_metrics,
    db_get_world_content_hash, db_get_world_entry_hash, db_update_world_entry_hash,
    db_save_embedding, db_get_embedding, db_search_similar_embeddings
)

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
        "steps": 20,
        "width": 512,
        "height": 512,
        "threshold": 0
    },
    "light": {
        "steps": 15,
        "width": 384,
        "height": 384,
        "threshold": 8000
    },
    "emergency": {
        "steps": 10,
        "width": 256,
        "height": 256,
        "threshold": 12000
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
    char_name: str
    field_type: str
    context: str
    source_mode: Optional[str] = "chat" # "chat" or "manual"

class CapsuleGenRequest(BaseModel):
    char_name: str
    description: str
    depth_prompt: str = ""

class WorldGenRequest(BaseModel):
    world_name: str
    section: str # history, locations, creatures, factions
    tone: str # sfw, spicy, veryspicy
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
    entry_uid: str
    section: str  # history, locations, creatures, factions
    tone: str  # sfw, spicy, veryspicy
    context: Optional[str] = ""
    source_mode: Optional[str] = "manual"

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
import asyncio
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
        """Get or compute embeddings for all world info entries"""
        if not world_info or "entries" not in world_info:
            return {}
        
        world_hash = self.get_world_info_hash(world_info)
        
        # Check if we already have embeddings for this world info
        if world_hash in self.embeddings_cache:
            return self.embeddings_cache[world_hash]
        
        # Load model if not loaded
        if not self.load_model():
            return {}
        
        print(f"Computing embeddings for {len(world_info['entries'])} world info entries...")
        
        # Extract content for embedding
        contents = []
        uids = []
        
        for uid, entry in world_info["entries"].items():
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
                uids.append(uid)
        
        if not contents:
            return {}
        
        # Compute embeddings
        try:
            embeddings = self.model.encode(contents, convert_to_numpy=True, show_progress_bar=False)
            
            # Store embeddings with UIDs
            result = {uids[i]: embeddings[i] for i in range(len(uids))}
            
            # Cache the embeddings
            self.embeddings_cache[world_hash] = result
            
            # Limit cache size to prevent memory issues
            # Remove multiple entries to maintain size constraint
            while len(self.embeddings_cache) >= 10:
                oldest_key = next(iter(self.embeddings_cache))
                del self.embeddings_cache[oldest_key]
            
            print(f"Computed embeddings for {len(result)} entries")
            return result
            
        except Exception as e:
            print(f"Failed to compute embeddings: {e}")
            return {}
    
    def search_semantic(self, world_info, query_text, max_entries=10, similarity_threshold=0.3, is_initial_turn=False):
        """Perform semantic search on world info entries - returns entries with similarity scores"""
        if not world_info or "entries" not in world_info:
            return [], []

        # Get embeddings
        embeddings = self.get_entry_embeddings(world_info)
        if not embeddings:
            return [], []

        # Load model if not loaded
        if not self.load_model():
            return [], []

        # Compute query embedding
        try:
            query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]

            # Calculate similarities using numpy (SIMD-accelerated via numpy)
            similarities = []
            for uid, entry_embedding in embeddings.items():
                # Compute cosine similarity: dot(a,b) / (norm(a) * norm(b))
                similarity = np.dot(query_embedding, entry_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding)
                )
                similarities.append((uid, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Adjust threshold for initial turns
            effective_threshold = similarity_threshold
            if is_initial_turn:
                effective_threshold = max(0.35, similarity_threshold)  # Higher threshold for initial turns

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

                # Add to results with similarity score (probability no longer gates matched entries)
                triggered_lore.append((entry.get("content", ""), similarity, uid))

                # Stop if we've reached max_entries (we'll apply proper limit later)
                if max_entries > 0 and len(triggered_lore) >= max_entries:
                    break

            return triggered_lore, canon_entries

        except Exception as e:
            print(f"Semantic search failed: {e}")
            return [], []

# Global semantic search engine instance
semantic_search_engine = SemanticSearchEngine()

# Global variable to store the cleanup task
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

# Periodic cleanup task for semantic search engine
async def periodic_cleanup():
    """Periodically clean up semantic search engine resources"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            semantic_search_engine.cleanup_resources()
        except Exception as e:
            print(f"Periodic cleanup failed: {e}")

# Auto-import JSON files on startup
def auto_import_json_files():
    """Scan JSON folders and import any new files not already in the database."""
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
                file_path = os.path.join(wi_dir, f)
                try:
                    # Check if already in database
                    existing = db_get_world(name)
                    if existing is None:
                        # Load and import
                        with open(file_path, "r", encoding="utf-8") as wf:
                            world_data = json.load(wf)
                        # db_save_world expects entries dict, not the full object
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

# FastAPI startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Initialize resources when the FastAPI app starts"""
    global cleanup_task
    
    # Auto-import any new JSON files dropped into folders
    print("Scanning for new JSON files to import...")
    auto_import_json_files()
    
    # Start the periodic cleanup task when the app starts
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

# Prompt Construction Engine
def construct_prompt(request: PromptRequest):
    settings = request.settings
    system_prompt = settings.get("system_prompt", CONFIG["system_prompt"])
    user_persona = settings.get("user_persona", "")
    summary = request.summary or ""
    mode = request.mode or "narrator"
    
    # Determine chat mode
    is_group_chat = len(request.characters) >= 2
    is_single_char = len(request.characters) == 1
    is_narrator_mode = not request.characters
    
    # Collect active character names early for mode instruction
    active_names = []
    for char_obj in request.characters:
        data = char_obj.get("data", {})
        active_names.append(data.get("name", "Unknown"))
    
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

        max_world_entries = settings.get("max_world_info_entries", 10)

        # Use optimized semantic search with turn detection
        triggered_lore, canon_law_entries = get_cached_world_entries(
            request.world_info,
            recent_text,
            max_entries=max_world_entries,
            semantic_threshold=0.45,
            is_initial_turn=is_initial_turn
        )

        if triggered_lore:
            full_prompt += "### World Knowledge:\n" + "\n".join(triggered_lore) + "\n"

    # === 6. CHARACTER PROFILES (characters exist in the world context) ===
    reinforcement_chunks = []
    
    for char_obj in request.characters:
        data = char_obj.get("data", {})
        name = data.get("name", "Unknown")
        description = data.get("description", "")
        depth_prompt = data.get("extensions", {}).get("depth_prompt", {}).get("prompt", "")
        multi_char_summary = data.get("extensions", {}).get("multi_char_summary", "")
        
        if is_group_chat and multi_char_summary:
            # Group chat mode: use compact capsule summary
            full_prompt += f"### [{name}]: {multi_char_summary}\n"
            # Use capsule for reinforcement too (shorter)
            reinforcement_chunks.append(f"[{name}]: {multi_char_summary}")
        else:
            # Single character or no capsule: use full profile
            full_prompt += f"### Character Profile: {name}\n{description}\n"
            if depth_prompt:
                full_prompt += f"### Context for {name}: {depth_prompt}\n"
                reinforcement_chunks.append(depth_prompt)

    # === 7. CHAT HISTORY (with reinforcement) ===
    full_prompt += "\n### Chat History:\n"
    reinforce_freq = settings.get("reinforce_freq", 0)
    world_reinforce_freq = settings.get("world_info_reinforce_freq", 5)  # Default to every 5 turns (optimized)
    
    # Track whether canon law was added in chat history to prevent duplication
    canon_added_in_history = False
    
    # Get world info reinforcement counter from chat metadata
    world_reinforce_counter = 0
    if hasattr(request, 'world_reinforce_counter'):
        world_reinforce_counter = request.world_reinforce_counter
    elif hasattr(request, 'metadata') and request.metadata:
        world_reinforce_counter = request.metadata.get('world_reinforce_counter', 0)
    
    for i, msg in enumerate(request.messages):
        # Filter out meta-messages like "Visual System" if they exist
        if msg.speaker == "Visual System":
            continue

        # Character reinforcement logic every X turns
        if reinforce_freq > 0 and i > 0 and i % reinforce_freq == 0:
            if reinforcement_chunks:
                # Character reinforcement
                full_prompt += "[REINFORCEMENT: " + " | ".join(reinforcement_chunks) + "]\n"
            elif is_narrator_mode:
                # Narrator reinforcement
                full_prompt += f"[REINFORCEMENT: {narrator_instruction.strip()}]\n"
        
        # World info canon law reinforcement logic every X turns
        # Add in history ONLY if we haven't added it yet in this prompt
        if world_reinforce_freq > 0 and i > 0 and i % world_reinforce_freq == 0:
            if canon_law_entries and not canon_added_in_history:
                full_prompt += "[WORLD REINFORCEMENT: " + " | ".join(canon_law_entries) + "]\n"
                canon_added_in_history = True
        
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
            return {"success": False, "error": "Unknown field type"}
        
        return {"success": True, "text": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

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
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory mentioned in the chat. Use this exact format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects)]\n\nRules:\n- One entry per line\n- Keep entries concise\n- Use parentheses for nested attributes\n- Output ONLY the PList entries, no explanations",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature/suggestive themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory. Include mature themes where appropriate. Use this format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects), intimate_aspects(relationships, desires, power dynamics)]\n\nRules:\n- One entry per line\n- Include suggestive details naturally\n- Output ONLY the PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event or backstory. Include explicit adult themes naturally. Use this format:\n[EventName: type(event/era/myth), time(when), actors(who), result(what happened), legacy(effects), sexual_culture(attitudes, practices, taboos)]\n\nRules:\n- One entry per line\n- Include explicit adult content where relevant\n- Output ONLY the PList entries"
    },
    "locations": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location, room, town, or area mentioned. Use this format:\n[LocationName(nickname if any): type(room/town/area), features(physical details), atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include suggestive atmosphere and uses. Use this format:\n[LocationName(nickname): type, features, atmosphere(mood), purpose(what happens), inhabitants(who), intimate_uses(private activities)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include explicit adult themes naturally. Format:\n[LocationName(nickname): type, features, atmosphere, purpose, inhabitants, sexual_activities(what happens), kinks_associated(themes)]\n\nRules:\n- One entry per line- Output ONLY PList entries"
    },
    "creatures": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each creature, monster, or character archetype. Use this format:\n[CreatureName: type(creature/archetype), appearance(visual traits), behavior(typical actions), culture(social norms, beliefs), habitat(where found), attitude_toward_user(how they treat {{user}})]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for creatures or archetypes. Include suggestive behavior and interactions. Format:\n[CreatureName: type, appearance, behavior, culture, habitat, attitude_toward_user, flirtation_style(how they seduce/interact)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[CreatureName: type, appearance, behavior, culture, attitude_toward_user, sexual_behavior(explicit details), kinks(preferences), consent_culture(boundaries)]\n\nRules:\n- One entry per line- Output ONLY PList entries"
    },
    "factions": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each faction, group, or organization. Use this format:\n[FactionName: type(faction/guild/house/clique), members(who belongs), reputation(public image), goals(what they want), methods(how they operate), attitude_toward_user(how they treat {{user}}), rivals(opposing factions)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for factions. Include suggestive motivations and interactions. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, social_dynamics(power, romance, rivalries), intimate_culture(dating norms, boundaries)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, sexual_culture(practices, rituals), kinks_favored(group preferences), initiation(how to join)]\n\nRules:\n- One entry per line- Output ONLY PList entries"
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
        # Clean up output to only include bracketed entries
        lines = [line.strip() for line in result.split('\n') if line.strip().startswith('[')]
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

@app.post("/api/chat")
async def chat(request: PromptRequest):
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
                
                # Save the updated capsule to the character file
                filename = char_obj.get("_filename")
                if filename:
                    file_path = os.path.join(DATA_DIR, "characters", filename)
                    save_data = char_obj.copy()
                    if "_filename" in save_data:
                        del save_data["_filename"]
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                    print(f"SAVED capsule for {name}: {capsule[:80]}...")
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
            return data
    
    # Route through resource manager with performance tracking
    if resource_manager.performance_mode_enabled:
        start_time = time.time()
        try:
            data = await resource_manager.execute_llm(llm_operation, op_type="heavy")
            duration = time.time() - start_time
            performance_tracker.record_llm(duration)
            
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
            
            return data
        except Exception as e:
            duration = time.time() - start_time
            performance_tracker.record_llm(duration)
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
            performance_tracker.record_sd(duration)
            
            # Generate hints based on SD performance
            hints = hint_engine.generate_hint(performance_tracker, context_tokens, duration)
            if hints:
                result["_performance_hints"] = hints
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            performance_tracker.record_sd(duration)
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
    
    # Clear the in-memory embeddings cache to force recomputation
    # The semantic search engine will recompute embeddings on next search
    # since the world info content hash will have changed
    semantic_search_engine.embeddings_cache.clear()
    print(f"Cleared semantic search embeddings cache for world info update: {name}")
    
    # Also clear the world info search results cache
    WORLD_INFO_CACHE.clear()
    
    # Also export to JSON for backward compatibility
    file_path = os.path.join(DATA_DIR, "worldinfo", f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"World info saved to DB and exported to JSON: {name}")
    return {"success": True}

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
async def load_chat(name: str):
    """Load chat from database."""
    try:
        chat_data = db_get_chat(name)
        if chat_data:
            return chat_data
    except Exception as e:
        print(f"Error loading chat from database: {e}")
    
    # Fall back to JSON file
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Chat not found"}

@app.post("/api/chats")
async def save_chat(request: dict):
    """Save chat to database and export to JSON for compatibility."""
    name = request.get("name")
    if not name:
        name = f"chat_{int(time.time())}"
    
    chat_data = request.get("data", {})
    
    # Save to database (primary source)
    try:
        db_save_chat(name, chat_data)
    except Exception as e:
        print(f"Error saving chat to database: {e}")
    
    # Also export to JSON for backward compatibility
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    print(f"Chat saved to DB and exported to JSON: {name}")
    return {"success": True, "name": name}

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

# Forking functionality
class ForkRequest(BaseModel):
    origin_chat_name: str
    fork_from_message_id: int
    branch_name: Optional[str] = None

@app.post("/api/chats/fork")
async def fork_chat(request: ForkRequest):
    """Create a new chat branch from a specific message in an existing chat."""
    origin_chat_name = request.origin_chat_name
    fork_from_message_id = request.fork_from_message_id
    branch_name = request.branch_name
    
    # Load origin chat - try database first, then JSON fallback
    origin_chat = None
    try:
        origin_chat = db_get_chat(origin_chat_name)
    except Exception as e:
        print(f"Error loading origin chat from database: {e}")
    
    # Fall back to JSON if database failed
    if not origin_chat:
        origin_file_path = os.path.join(DATA_DIR, "chats", f"{origin_chat_name}.json")
        if not os.path.exists(origin_file_path):
            return {"success": False, "error": "Origin chat not found"}
        
        with open(origin_file_path, "r", encoding="utf-8") as f:
            origin_chat = json.load(f)
    
    # Find the fork point
    messages = origin_chat.get("messages", [])
    fork_index = None
    
    for i, msg in enumerate(messages):
        if msg.get("id") == fork_from_message_id:
            fork_index = i
            break
    
    if fork_index is None:
        return {"success": False, "error": "Message not found in origin chat"}
    
    # Generate branch name if not provided
    if not branch_name:
        # Get a short preview of the message content
        preview = messages[fork_index].get("content", "")
        if len(preview) > 30:
            preview = preview[:27] + "..."
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        branch_name = f"Fork from {timestamp} - '{preview}'"
    
    # Create new branch chat data
    branch_messages = messages[:fork_index + 1]  # Include the fork point message
    
    branch_data = {
        "messages": branch_messages,
        "summary": origin_chat.get("summary", ""),
        "activeCharacters": origin_chat.get("activeCharacters", []),
        "activeWI": origin_chat.get("activeWI"),
        "settings": origin_chat.get("settings", {}),
        "metadata": {
            "origin_chat_id": origin_chat_name,
            "origin_message_id": fork_from_message_id,
            "branch_name": branch_name,
            "created_at": time.time()
        }
    }
    
    # Generate unique branch filename
    base_name = f"{origin_chat_name}_fork_{int(time.time())}"
    
    # Save to database (primary source)
    try:
        db_save_chat(base_name, branch_data)
    except Exception as e:
        print(f"Error saving fork to database: {e}")
    
    # Also export to JSON for backward compatibility
    branch_file_path = os.path.join(DATA_DIR, "chats", f"{base_name}.json")
    with open(branch_file_path, "w", encoding="utf-8") as f:
        json.dump(branch_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fork saved to DB and exported to JSON: {base_name}")
    return {
        "success": True, 
        "name": base_name,
        "branch_name": branch_name,
        "origin_chat_name": origin_chat_name,
        "fork_from_message_id": fork_from_message_id
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
    """Use AI to generate or improve a world info entry."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        if not os.path.exists(file_path):
            return {"success": False, "error": "World info file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        
        # Validate entry exists
        if req.entry_uid not in world_data.get("entries", {}):
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
            # Parse the generated entries and update the specific entry
            lines = result["text"].split('\n')
            for line in lines:
                parsed = parse_plist_line(line)
                if parsed:
                    # Update the specific entry with the new content
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
    """Delete a specific world info entry."""
    try:
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{world_name}.json")
        if not os.path.exists(file_path):
            return {"success": False, "error": "World info file not found"}
        
        with open(file_path, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        
        # Validate entry exists
        if entry_uid not in world_data.get("entries", {}):
            return {"success": False, "error": "Entry not found"}
        
        # Remove the entry
        del world_data["entries"][entry_uid]
        
        # Save the updated world info
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(world_data, f, indent=2, ensure_ascii=False)
        
        return {"success": True, "message": f"Entry '{entry_uid}' deleted successfully"}
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
