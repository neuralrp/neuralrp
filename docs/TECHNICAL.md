# NeuralRP Technical Documentation

This document covers implementation details, architecture decisions, and internal mechanics for developers and power users.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [SQLite Database](#sqlite-database)
3. [Context Assembly](#context-assembly)
4. [World Info Engine](#world-info-engine)
5. [Semantic Search](#semantic-search)
6. [Performance Mode](#performance-mode)
7. [Branching System](#branching-system)
8. [Memory & Summarization](#memory--summarization)
9. [Image Generation](#image-generation)
10. [Connection Monitoring](#connection-monitoring)
11. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)

---

## Architecture Overview

NeuralRP is a single-process FastAPI application that coordinates between:

- **KoboldCpp** (or OpenAI-compatible endpoint) for text generation
- **AUTOMATIC1111 WebUI** for image generation
- **SQLite** for persistent storage
- **sqlite-vec** for vector similarity search

All state is managed through a centralized SQLite database (`app/data/neuralrp.db`), with optional JSON export for SillyTavern compatibility.

### Project Structure

```
neuralrp/
├── main.py                 # FastAPI application, routes, and core logic
├── migrate_to_sqlite.py    # One-time JSON → SQLite migration script
├── app/
│   ├── database.py         # SQLite connection and query helpers
│   ├── index.html          # Frontend interface
│   └── data/
│       ├── neuralrp.db     # SQLite database
│       ├── characters/     # Exported character cards (JSON)
│       ├── chats/          # Exported chat sessions (JSON)
│       └── worldinfo/      # Exported world info (JSON)
└── app/images/             # Generated images (PNG)
```

---

## SQLite Database

### Schema Overview

NeuralRP uses a relational database with the following core tables:

**Characters Table**
```sql
CREATE TABLE characters (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    personality TEXT,
    first_mes TEXT,
    mes_example TEXT,
    scenario TEXT,
    system_prompt TEXT,
    danbooru_tag TEXT,
    capsule_summary TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**Chats Table**
```sql
CREATE TABLE chats (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    world_info_id INTEGER,
    summary TEXT,
    metadata TEXT,  -- JSON blob for settings, branch info, etc.
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (world_info_id) REFERENCES world_info(id)
);
```

**Messages Table**
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    character_name TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chats(id)
);
```

**World Info Tables**
```sql
CREATE TABLE world_info (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE world_entries (
    id INTEGER PRIMARY KEY,
    world_info_id INTEGER NOT NULL,
    entry_key TEXT NOT NULL,
    keys TEXT,  -- JSON array of trigger keywords
    content TEXT,
    is_canon_law BOOLEAN DEFAULT 0,
    use_probability BOOLEAN DEFAULT 0,
    probability INTEGER DEFAULT 100,
    created_at TIMESTAMP,
    FOREIGN KEY (world_info_id) REFERENCES world_info(id)
);
```

**Vector Search Table (sqlite-vec)**
```sql
CREATE VIRTUAL TABLE vec_world_entries USING vec0(
    entry_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
```

### ACID Guarantees

All data operations use SQLite transactions:
- **Atomicity**: Complete operations succeed or fail entirely
- **Consistency**: Foreign key constraints prevent orphaned records
- **Isolation**: Concurrent reads don't interfere with writes
- **Durability**: WAL mode ensures crash recovery

### SillyTavern Compatibility

NeuralRP maintains backward compatibility through auto-export:
- Character saves write both to database AND to JSON file
- World info saves write both to database AND to JSON file
- Exported files use SillyTavern V2 format
- Import reads JSON files and inserts into database

---

## Context Assembly

On each generation request, NeuralRP builds the prompt in a specific layered order to maintain stable structure:

### Layer Order

1. **System and Mode Instructions**
   - Global system prompt (tone, formatting rules)
   - Mode-specific instructions:
     - **Narrator**: Third-person, omniscient narration
     - **Focus**: First-person, locked to specific character
     - **Auto**: Model decides who speaks

2. **User Persona** (optional)
   - Short player/user description
   - Placed early to influence perspective

3. **World Info** (if enabled)
   - Canon Law entries first (always included)
   - Semantic search results second
   - Keyword fallback if semantic returns nothing

4. **Character Definitions**
   - Single character: Full card content
   - Multi-character: Capsule personas (compressed)
   - Focus mode: Selected character emphasized

5. **Conversation History**
   - Recent messages verbatim
   - Older content as summary (if summarization triggered)

6. **Generation Lead-In**
   - Final formatting instruction
   - User's latest message

### Token Budget Management

Context assembly monitors token count and adjusts:
- Target: Stay under configurable threshold (default 85%)
- When exceeded: Trigger summarization of oldest messages
- Canon Law: Never counted against caps, always included

---

## World Info Engine

### Retrieval Strategy

NeuralRP uses a hybrid approach:

1. **Semantic Search** (primary)
   - Query: Last 5 messages concatenated
   - Method: Cosine similarity against entry embeddings
   - Threshold: 0.25-0.45 depending on turn count
   - Returns: Top N entries above threshold

2. **Keyword Matching** (fallback)
   - Triggered when semantic returns no results
   - Case-insensitive substring matching
   - Configurable entry cap (default: 10)

3. **Canon Law** (always included)
   - Entries marked `is_canon_law = true`
   - Never subject to caps or probability
   - Injected at end of context to override drift

### Probability Weighting

For entries with `use_probability = true`:
```python
if random.random() * 100 > entry.probability:
    skip_entry()
```
Allows stochastic lore injection for variety in large worlds.

### Reinforcement Configuration

Canon Law reinforcement can be throttled:
- Default: Every 3 turns
- Configurable: 1-100 turns
- Reduces prompt repetition while maintaining consistency

---

## Semantic Search

### Implementation

The `SemanticSearchEngine` uses sentence transformers for embedding-based retrieval:

**Model**: `all-mpnet-base-v2`
- 768-dimensional embeddings
- Good balance of speed and accuracy
- Automatic GPU/CPU detection

**Storage**: sqlite-vec virtual table
- Disk-based, not in RAM
- SIMD-accelerated similarity (AVX2/SSE2)
- Zero idle memory overhead

### Search Algorithm

```python
def search_semantic(world_info_id, query_text, max_entries=10, threshold=0.25):
    # 1. Embed query text
    query_embedding = model.encode(query_text)
    
    # 2. Vector similarity search
    results = db.execute("""
        SELECT entry_id, distance
        FROM vec_world_entries
        WHERE world_info_id = ?
        ORDER BY embedding <-> ?
        LIMIT ?
    """, [world_info_id, query_embedding, max_entries])
    
    # 3. Filter by threshold
    return [r for r in results if (1 - r.distance) >= threshold]
```

### Cache Invalidation

Embeddings update automatically when world info changes:
- Entry content hash tracked in database
- Modified entries re-embedded on next search
- In-memory caches cleared on world info save

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Cold start | ~2-3s (embedding computation) |
| Warm search | <50ms |
| Memory per entry | ~3KB (768 floats × 4 bytes) |
| GPU acceleration | 5-10× faster than CPU |

---

## Performance Mode

### Overview

When running LLM + Stable Diffusion on the same GPU, Performance Mode prevents resource contention through intelligent queuing.

### Resource Manager

Coordinates GPU access using `asyncio.Lock()`:

```python
class ResourceManager:
    def __init__(self):
        self.heavy_lock = asyncio.Lock()
        self.active_llm = False
        self.active_sd = False
    
    async def acquire_heavy(self, operation_type):
        async with self.heavy_lock:
            # Execute heavy operation
            yield
    
    def can_proceed_light(self):
        # Light operations always proceed
        return True
```

**Operation Classification**:
- **Heavy**: Image generation, large context text, card generation
- **Light**: Small text generation, status checks, UI updates

### Performance Tracker

Uses rolling median for outlier-resistant timing:

```python
class PerformanceTracker:
    def __init__(self):
        self.llm_times = deque(maxlen=10)
        self.sd_times = deque(maxlen=10)
    
    def detect_contention(self, current_sd_time, context_tokens):
        median_sd = statistics.median(self.sd_times)
        return (current_sd_time > 3 * median_sd) and (context_tokens > 8000)
```

### SD Context-Aware Presets

Three-tier quality adjustment based on story length:

| Preset | Steps | Size | Threshold |
|--------|-------|------|-----------|
| Normal | 20 | 512×512 | 0-7999 tokens |
| Light | 15 | 384×384 | 8000-11999 tokens |
| Emergency | 10 | 256×256 | 12000+ tokens |

### Smart Hints

Context-aware suggestions triggered by metrics:
- Contention detected: "Consider generating images outside chat"
- Emergency preset active: "Quality reduced due to high context"
- Dismissible, non-repetitive notifications

---

## Branching System

### Branch Creation

When forking from a message:

1. **New chat record** created with metadata:
   ```json
   {
     "origin_chat_id": "original_chat_name",
     "origin_message_id": 12345678,
     "branch_name": "Custom Branch Name",
     "created_at": 1736279867.123
   }
   ```

2. **Messages copied** up to fork point
3. **Settings inherited** (characters, world, temperature, etc.)

### Branch Independence

- Each branch is a separate database record
- No shared mutable state
- Editing one branch doesn't affect others
- O(1) creation (row insert, not file copy)

### No Merge Semantics

Branches are independent timelines by design:
- Fork = create new "what-if" timeline
- Switch = load that timeline
- No automatic merging or conflict resolution

---

## Memory & Summarization

### Automatic Summarization

Triggered when context usage exceeds threshold:

**Trigger Conditions**:
- Total tokens > 85% of model's max context
- Message count > 10 (minimum for summarization)

**Process**:
1. Take oldest 10 messages
2. Send to LLM for summary generation
3. Prepend new summary to existing summary field
4. Remove summarized messages from active history

### Summary Storage

```sql
-- Summary stored in chat metadata
UPDATE chats SET summary = ? WHERE id = ?;
```

Summary used during context assembly but not displayed in UI.

### What Gets Kept vs Summarized

- **Verbatim**: Most recent messages up to ~85% of budget
- **Summarized**: Oldest messages when threshold exceeded
- **Result**: Sessions can run indefinitely

---

## Image Generation

### A1111 Integration

- **Endpoint**: `POST {sd_url}/sdapi/v1/txt2img`
- **Inpainting**: `POST {sd_url}/sdapi/v1/img2img` with mask
- **Response**: Base64-encoded PNG

### Character Tag Expansion

Danbooru tags assigned per-character:

```
User prompt: "[Alice] walking in park"
Expanded: "1girl, long_hair, blue_eyes, white_dress, walking in park"
```

Tags stored in character record, expanded at generation time.

### Metadata Storage

All generation parameters stored in database:

```json
{
  "prompt": "...",
  "negative_prompt": "...",
  "steps": 20,
  "cfg_scale": 8.0,
  "width": 512,
  "height": 512,
  "seed": 1234567890,
  "timestamp": "2026-01-13T14:30:00Z"
}
```

Enables reproducibility and debugging.

---

## Connection Monitoring

### Adaptive Intervals

Monitoring frequency adjusts based on connection stability:

| State | Interval | Rationale |
|-------|----------|-----------|
| Stable (5+ min) | 60s | Minimal overhead |
| Stable (initial) | 30s | Balance detection/overhead |
| Initial failure | 10s | Catch restarts quickly |
| Persistent failure (3+) | 5s | Aggressive recovery detection |

### Background Tab Optimization

Uses Page Visibility API:
- Tab hidden: All monitoring paused
- Tab visible: Immediate health check, then resume

### Health Check Endpoints

- **KoboldCpp**: `GET /api/v1/model` or `/v1/models`
- **Stable Diffusion**: `GET /sdapi/v1/options`
- **Timeout**: 5 seconds per request

---

## Design Decisions & Tradeoffs

### Why SQLite Instead of JSON Files?

**Pros**:
- ACID transactions prevent data corruption
- Indexed queries scale to 10,000+ entries
- Foreign keys ensure referential integrity
- Single file for all data (easier backup)

**Cons**:
- Requires migration from JSON
- Not human-readable without tools
- Slightly more complex schema changes

**Mitigation**: Auto-export to JSON maintains SillyTavern compatibility.

### Why sqlite-vec Instead of In-Memory Embeddings?

**Pros**:
- Zero RAM overhead for embeddings
- SIMD-accelerated similarity search
- Persists across restarts
- No pickle security vulnerabilities

**Cons**:
- Requires sqlite-vec extension
- Slightly slower than pure NumPy for small datasets

**Mitigation**: Falls back to keyword matching if sqlite-vec unavailable.

### Why Keyword Fallback Instead of Pure Semantic?

**Pros**:
- Handles edge cases where semantic fails
- Predictable behavior for users
- Works without embedding model

**Cons**:
- Can cause false positives with short keywords

**Mitigation**: Canon Law ensures critical lore always included regardless of retrieval method.

### Why Async Locking Instead of Multi-Process?

**Pros**:
- Works within single FastAPI process
- Lightweight (<1ms overhead)
- Clean async/await integration

**Cons**:
- Doesn't prevent external GPU overload
- Only works within NeuralRP process

**Mitigation**: Performance tracker detects external contention via timing analysis.

### Why 85% Summarization Threshold?

- Below 85%: Room for model to generate
- At 85%: Headroom for summary insertion
- Above 85%: Risk of truncation during generation

Configurable in settings for different model context sizes.

### Why Branches Instead of Graph?

**Pros**:
- Each branch is independent and portable
- No corruption risk affecting multiple branches
- Simple mental model ("one timeline")

**Cons**:
- No single-view graph visualization
- More records for power users with many branches

**Mitigation**: Branch management UI aggregates metadata for navigation.

---

## API Reference

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/chats` | List all chats |
| GET | `/api/chats/{name}` | Get chat by name |
| POST | `/api/chats` | Create new chat |
| PUT | `/api/chats/{name}` | Update chat |
| DELETE | `/api/chats/{name}` | Delete chat |
| POST | `/api/chats/fork` | Create branch from message |
| GET | `/api/chats/{name}/branches` | List branches of chat |

### Character Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/characters` | List all characters |
| GET | `/api/characters/{name}` | Get character by name |
| POST | `/api/characters` | Create character |
| PUT | `/api/characters/{name}` | Update character |
| DELETE | `/api/characters/{name}` | Delete character |

### World Info Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/worldinfo` | List all world info |
| GET | `/api/worldinfo/{name}` | Get world info by name |
| POST | `/api/worldinfo` | Create world info |
| PUT | `/api/worldinfo/{name}` | Update world info |
| GET | `/api/world-info/cache/stats` | Cache statistics |
| POST | `/api/world-info/cache/clear` | Clear cache |
| GET | `/api/world-info/reinforcement/config` | Get reinforcement settings |
| POST | `/api/world-info/reinforcement/config` | Update reinforcement settings |

### Generation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate` | Generate LLM response |
| POST | `/api/generate-image` | Generate image |
| POST | `/api/inpaint` | Inpaint existing image |
| POST | `/api/generate-card` | Generate character card |
| POST | `/api/generate-world` | Generate world info |

### Performance Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/performance/status` | Current operation states |
| POST | `/api/performance/toggle` | Enable/disable performance mode |
| GET | `/api/performance/hints` | Get optimization hints |

---

## Future Considerations

### Potential Enhancements

- **Timeline Visualization**: React Flow-based branch graph
- **Multi-Tier Memory**: Short/medium/long-term with different compression
- **Incremental Embeddings**: Avoid full recomputation on single entry changes
- **User-Configurable Embedding Models**: Trade speed vs accuracy

### Known Limitations

- No merge semantics for branches
- Single-process architecture (no distributed deployment)
- External GPU contention not fully managed
- No UI indicator for active summarization
