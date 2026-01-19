# NeuralRP Technical Documentation

This document explains how NeuralRP's advanced features work under the hood, designed for users who want to understand the systems behind the interface.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [SQLite Database](#sqlite-database)
3. [Context Assembly](#context-assembly)
4. [World Info Engine](#world-info-engine)
5. [Semantic Search](#semantic-search)
6. [Semantic Relationship Tracker](#semantic-relationship-tracker)
7. [Entity ID System](#entity-id-system)
8. [Message Search System](#message-search-system)
9. [Performance Mode](#performance-mode)
10. [Branching System](#branching-system)
11. [Memory & Summarization](#memory--summarization)
12. [Soft Delete System](#soft-delete-system)
13. [Autosave System](#autosave-system)
14. [Image Generation](#image-generation)
15. [Connection Monitoring](#connection-monitoring)
16. [Change History Data Recovery](#change-history-data-recovery)
17. [Undo/Redo System](#undoredo-system)
18. [Character Name Consistency](#character-name-consistency)
19. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)

---

## Architecture Overview

NeuralRP is a single FastAPI application that coordinates between:

- **KoboldCpp** (or OpenAI-compatible endpoint) for text generation
- **AUTOMATIC1111 WebUI** for image generation
- **SQLite** for persistent storage with ACID guarantees
- **sqlite-vec** for vector similarity search

All data lives in a centralized SQLite database (`app/data/neuralrp.db`), with optional JSON export for SillyTavern compatibility.

### Project Structure

```
neuralrp/
├── main.py                 # FastAPI application (2,500+ lines)
├── app/
│   ├── database.py         # SQLite operations (1,000+ lines)
│   ├── index.html          # Single-page frontend
│   ├── data/
│   │   ├── neuralrp.db     # SQLite database
│   │   ├── characters/     # Exported character JSONs
│   │   ├── worldinfo/      # Exported world info JSONs
│   │   └── chats/         # Exported chat JSONs
│   └── images/            # Generated images (PNG)
```

### Data Flow

```
User Input (Web UI)
    ↓
FastAPI Endpoint
    ↓
Resource Manager (queues operations)
    ↓
├─→ LLM Call → KoboldCpp → Response → Database save
├─→ Image Gen → Stable Diffusion → Image save → Metadata store
├─→ Character/World Edit → Database → JSON export
├─→ Relationship Analysis → Semantic embeddings → Relationship update
└─→ Semantic Search → sqlite-vec → Ranked results
```

---

## SQLite Database

### Why SQLite?

NeuralRP migrated from JSON files to SQLite in v1.5.0 for several key benefits:

**ACID Guarantees:**
- **Atomicity**: Complete operations succeed or fail entirely (no partial saves)
- **Consistency**: Foreign key constraints prevent orphaned records
- **Isolation**: Concurrent reads don't interfere with writes
- **Durability**: WAL mode ensures crash recovery

**Scalability:**
- Indexed queries scale to 10,000+ entries without performance loss
- Single file for all data (easier backup)
- Automatic cleanup of old data

### Core Tables

**Characters** - Character cards with JSON data, extensions (danbooru tags, capsules)

**Worlds + World Entries** - World info containers with embeddings linked to `vec_world_entries`

**Chats** - Chat sessions with summaries, metadata, branch info, and autosave flags

**Messages** - Individual chat messages linked to chats, with soft delete support (`summarized` field)

**vec_world_entries** - Vector embeddings (768-dimensional) for semantic search via sqlite-vec

**change_log** - Audit trail for undo/redo support (30-day retention)

**relationships** - Current relationship states between entities with five-dimensional emotional tracking

**relationship_history** - Historical snapshots of relationships (20 retained per relationship pair)

**entities** - Entity registry with unique IDs for characters, NPCs, and users

### Startup Health Check

Every time NeuralRP launches, it validates database integrity:

- **SQLite integrity check** - Detects corruption (<10ms on typical databases)
- **Core table verification** - Ensures all essential tables exist
- **Graceful degradation** - App continues even if check fails (allows manual repair)

If corruption is detected, you'll see a warning in the console advising you to run `migrate_to_sqlite.py` to rebuild from JSON backups.

### SillyTavern Compatibility

NeuralRP maintains backward compatibility through auto-export:
- Character saves write to **both** database and JSON file
- World info saves write to **both** database and JSON file
- Exported files use SillyTavern V2 format
- Import reads JSON files and inserts into database

This lets you move characters and world info between NeuralRP and SillyTavern seamlessly.

---

## Context Assembly

On every generation request, NeuralRP builds a prompt in a specific layered order to maintain stable structure.

### Layer Structure

1. **System and Mode Instructions**
   - Global system prompt (tone, formatting rules)
   - Mode-specific instructions (Narrator vs Focus)

2. **User Persona** (optional)
   - Short player/user description
   - Placed early to influence perspective

3. **World Info**
   - Canon Law entries first (always included)
   - Semantic search results second (context-aware)
   - Keyword matches if semantic returns nothing

4. **Character Definitions**
   - Single character: Full card content
   - Multi-character: Capsule personas (compressed summaries)
   - Focus mode: Selected character emphasized

5. **Conversation History**
   - Recent messages verbatim
   - Older content as summary (if summarization triggered)

6. **Relationship Context** (new in v1.6.0)
   - Current relationship states between characters, NPCs, and user
   - Five emotional dimensions: Trust, Emotional Bond, Conflict, Power Dynamic, Fear/Anxiety
   - Directional tracking (Alice→Bob separate from Bob→Alice)
   - Only included if sufficient relationship data exists

7. **Generation Lead-In**
   - Final formatting instruction
   - User's latest message

### Chat Modes

**Narrator Mode** (third-person omniscient):
- AI describes actions, thoughts, and scenes cinematically
- Any character may speak when it makes sense
- Uses format: `Name: "dialogue line"`
- Default for single-character and multi-character chats

**Focus Mode** (first-person character voice):
- Locked to specific character's perspective
- Only that character's thoughts, feelings, and words
- Stops before other characters can speak
- Useful for intimate scenes or character-focused interactions

**Auto Mode** (available via API):
- AI decides who should respond based on context
- Uses `/api/classify-mode` endpoint to analyze user message

### Token Budget Management

Context assembly monitors token count and adjusts automatically:
- **Target**: Stay under configurable threshold (default 85% of max context)
- **When exceeded**: Trigger summarization of oldest messages
- **Relationship Data**: Minimal overhead (~50-100 bytes per relationship)
- **Canon Law**: Never counted against caps, always included

---

## World Info Engine

### Retrieval Strategy

NeuralRP uses a hybrid approach to find relevant world info:

1. **Semantic Search** (primary)
   - Query: Last 5 messages concatenated
   - Method: Cosine similarity against entry embeddings
   - Threshold: 0.35-0.45 depending on turn count
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

For entries with `use_probability = true`, NeuralRP randomly includes them based on the `probability` value (1-100). This allows stochastic lore injection for variety in large worlds without overwhelming every turn.

### Reinforcement System

World info canon law can be reinforced at regular intervals to prevent drift:
- **Default**: Every 3 turns
- **Configurable**: 1-100 turns
- **Purpose**: Reduces prompt repetition while maintaining consistency

Canon law reinforcement is separate from character reinforcement (every X turns, includes character profiles).

---

## Semantic Search

### How It Works

NeuralRP uses sentence-transformers to understand meaning, not just keywords:

1. **Model**: `all-mpnet-base-v2` (768-dimensional embeddings)
2. **Storage**: Disk-based via sqlite-vec (no RAM overhead)
3. **Search**: SIMD-accelerated KNN with cosine similarity

### Key Benefits

**Persistent Embeddings**:
- Computed once and stored permanently in `neuralrp.db`
- No recomputation needed after app restart
- Load time: <50ms vs 2-3s cold start with in-memory models

**Lazy Loading**:
- Embeddings only loaded when world info is used
- Zero RAM overhead when idle
- Startup time: <1 second

**Dual-Path Architecture**:
- Primary: sqlite-vec SIMD search (fastest, disk-based)
- Fallback: NumPy calculations (always available)
- Console indicates which method was used

### Search Algorithm

On every turn, NeuralRP scans the last 5 messages:

1. **Canon Law**: Always included, never scanned
2. **Regular Entries**: Only included if semantically relevant to recent context
3. **Keyword Priority**: Keyword matches rank higher than semantic-only matches

**Trade-off**: Entries don't "stick" after being triggered. If a dragon is mentioned in Turn 1 but not in Turns 2-5, it drops out of context.

**Benefit**: Keeps context lean (~2k tokens) for 12GB VRAM optimization.

### Generic Key Filtering

To improve relevance, structural words are excluded from embeddings:

*Examples: "the", "and", "or", "city", "room", "location", "character", "person", "you", "your", "I", "my", "me", "he", "she", "they", "it", "this", "that", "is", "was", "were", "been", "have", "has", "had", "can", "will", "would", "could", "should", "must", "not", "no", "yes", "maybe", "go", "come", "get", "take", "make", "do", "be", "say", "tell", "ask", "know", "think", "feel", "see", "thing", "stuff", "something", "nothing", "everything", "one", "two", "first", "second", "next", "last", "time", "way", "day", "night", "morning", "evening", "good", "bad", "right", "wrong", "true", "false", "new", "old", "young", "small", "big", "large", "many", "much", "little", "few", "some", "all"*

### Performance

| Metric | Value |
|--------|-------|
| Search time (100 entries) | 20-50ms |
| Search time (10,000 entries) | <200ms |
| Startup time | <1 second |
| Memory overhead | 0 bytes (disk-based) |
| Storage per entry | ~3KB (768 floats × 4 bytes) |

---

## Semantic Relationship Tracker

### Overview (v1.6.0)

NeuralRP automatically tracks emotional relationships between characters, NPCs, and user using semantic embeddings. This provides emotionally consistent responses without LLM calls or context bloat.

### Five-Dimensional Emotional Model

Relationships are tracked across five emotional dimensions:

1. **Trust** (-1.0 to +1.0): How much Entity A trusts Entity B
   - +1.0: Complete trust, reveals secrets
   - 0.0: Neutral, guarded
   - -1.0: Complete distrust, hostility

2. **Emotional Bond** (-1.0 to +1.0): Strength of emotional connection
   - +1.0: Deep emotional attachment
   - 0.0: Indifferent
   - -1.0: Strong negative emotions (hate, resentment)

3. **Conflict** (-1.0 to +1.0): Level of disagreement or tension
   - +1.0: Constant conflict, opposition
   - 0.0: Harmonious
   - -1.0: Aligned, cooperative

4. **Power Dynamic** (-1.0 to +1.0): Power balance between entities
   - +1.0: Entity A has complete control over B
   - 0.0: Equal power
   - -1.0: Entity B has complete control over A

5. **Fear/Anxiety** (-1.0 to +1.0): Level of fear or anxiety
   - +1.0: Terrified, panicked
   - 0.0: Calm
   - -1.0: Fearless, confident

### Analysis Engine

**Trigger**: Every 10 messages

**Method**: Semantic embedding analysis (no LLM calls)

**Process**:
1. Extract last 10 messages
2. Identify entities present (characters, NPCs, user)
3. Compute semantic embeddings for message segments
4. Compare embeddings to previous relationship state
5. Calculate emotional deltas for each dimension
6. Update relationship scores incrementally (gradual evolution)

**Time Complexity**: O(n × m) where n=messages, m=entities

**Performance**: <20ms overhead per update

### Directional Tracking

Relationships are directional:
- Alice→Bob: How Alice feels about Bob
- Bob→Alice: How Bob feels about Alice

This allows for asymmetric relationships (e.g., Alice trusts Bob, but Bob doesn't trust Alice).

### Relationship History

Every relationship update saves a snapshot:

**Retention**: 20 most recent snapshots per relationship pair

**Automatic Pruning**: Oldest snapshots deleted when limit exceeded

**Purpose**: Track relationship evolution over time, enabling retrospective analysis

**Snapshot Structure**:
```json
{
  "from_entity_id": "char_abc123",
  "to_entity_id": "npc_xyz789",
  "timestamp": 1737254400,
  "trust": 0.5,
  "emotional_bond": 0.3,
  "conflict": -0.2,
  "power_dynamic": 0.0,
  "fear_anxiety": -0.4,
  "message_count_at_update": 50
}
```

### Context Injection

Relevant relationship states are automatically injected into prompts:

**Inclusion Criteria**:
- At least 20 messages exchanged between entities
- Relationship score magnitude > 0.3 (avoid weak signals)

**Format**:
```
### Relationship Context:
Alice → Bob (Trust: 0.5, Bond: 0.3, Conflict: -0.2)
John → User (Trust: -0.2, Fear: 0.6)
```

**Benefit**: AI understands emotional dynamics without explicit instructions, providing emotionally consistent responses.

### API Endpoints

**POST /api/relationships/update**
- Analyzes messages and updates relationship states
- Called automatically after every 10th message

**GET /api/relationships/{chat_id}**
- Returns current relationship states for all entities in chat

**GET /api/relationships/{chat_id}/history**
- Returns relationship evolution history (20 snapshots per pair)

**GET /api/entities/{chat_id}**
- Returns all entities registered for chat (characters, NPCs, user)

### Database Schema

**relationships table**:
```sql
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    chat_id TEXT NOT NULL,
    from_entity_id TEXT NOT NULL,
    to_entity_id TEXT NOT NULL,
    trust REAL DEFAULT 0.0,
    emotional_bond REAL DEFAULT 0.0,
    conflict REAL DEFAULT 0.0,
    power_dynamic REAL DEFAULT 0.0,
    fear_anxiety REAL DEFAULT 0.0,
    last_updated INTEGER,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    UNIQUE(chat_id, from_entity_id, to_entity_id)
)
```

**relationship_history table**:
```sql
CREATE TABLE relationship_history (
    id INTEGER PRIMARY KEY,
    relationship_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    trust REAL NOT NULL,
    emotional_bond REAL NOT NULL,
    conflict REAL NOT NULL,
    power_dynamic REAL NOT NULL,
    fear_anxiety REAL NOT NULL,
    message_count INTEGER NOT NULL,
    FOREIGN KEY (relationship_id) REFERENCES relationships(id) ON DELETE CASCADE
)
```

### Performance

| Metric | Value |
|--------|-------|
| Analysis overhead (every 10 messages) | <20ms |
| Storage per relationship snapshot | ~50-100 bytes |
| Context injection overhead | <5ms |
| History query time | <50ms (20 snapshots) |
| Total overhead per chat session | <0.5% of generation time |

### Design Philosophy

**Why Semantic Embeddings Instead of LLM?**
- **Speed**: <20ms vs 500ms+ for LLM analysis
- **No Context Bloat**: Embeddings computed separately, doesn't add to prompt
- **Consistency**: Same model used for world info semantic search
- **Cost**: Zero token cost

**Why Every 10 Messages?**
- **Granularity**: Fine-grained tracking without excessive overhead
- **Smoothing**: Reduces noise from individual message fluctuations
- **Performance**: Balances tracking accuracy with system load

**Why Five Dimensions?**
- **Expressive**: Captures complex emotional states
- **Interpretable**: Each dimension has clear meaning
- **Balanced**: Comprehensive without overwhelming complexity

---

## Entity ID System

### Overview (v1.6.0)

The entity ID system provides unique identification for all entities (characters, NPCs, users) to prevent name collisions and ensure relationship tracker reliability.

### Why Entity IDs?

**Problem**: Duplicate or similar names can cause confusion:
- Two different characters named "John" or "Mark"
- NPCs with names like "Guard", "Merchant"
- User character names colliding with character names

**Solution**: Unique IDs persist regardless of name variations:
- Character "John Smith" → entity ID: `char_abc123`
- NPC "Guard" → entity ID: `npc_xyz789`
- User → entity ID: `user_default`

### Entity Registration

**Automatic Registration**:
- Characters: Registered when loaded into chat
- NPCs: Registered when first mentioned in conversation
- User: Registered as `user_default` entity

**Entity Structure**:
```json
{
  "entity_id": "char_abc123",
  "entity_type": "character|npc|user",
  "name": "John Smith",
  "first_seen_timestamp": 1737254400,
  "last_seen_timestamp": 1737260000
}
```

### Name Resolution

**Character Name Consistency Helper**:
The `get_character_name()` function ensures consistent name extraction across all systems:

```python
# Always use this helper
char_name = get_character_name(character_obj)
# Returns: Full character name ("John Smith", never just "John")
```

**Entity ID Lookup**:
- Entity IDs stored in relationship tables
- Name lookup via `entities` table
- Automatic NPC entity creation on first mention

### Database Schema

**entities table**:
```sql
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,  -- 'character', 'npc', 'user'
    name TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
)
```

### Fork Safety

**Problem**: Branching creates duplicate entity records if not handled properly

**Solution**:
- Entity IDs are chat-scoped (unique within chat)
- Forks copy entities to new chat with same names
- Relationship history preserved via relationship_id foreign keys

**Example**:
```
Original Chat:
  - Entity: char_abc123 (John Smith)
  - Relationship: char_abc123 → user_default

Forked Chat:
  - Entity: char_def456 (John Smith)  -- Different ID
  - Relationship: char_def456 → user_ghi789  -- Independent tracking
```

### API Endpoints

**GET /api/entities/{chat_id}**
- Returns all entities registered for chat
- Includes entity type, name, and timestamps

### Performance

| Metric | Value |
|--------|-------|
| Entity registration overhead | <5ms |
| Name resolution time | <1ms (cached lookup) |
| Storage per entity | ~50 bytes |
| Total overhead per chat | Negligible |

### Design Philosophy

**Why Chat-Scoped Entity IDs?**
- Isolation prevents branch conflicts
- Each timeline has independent entity registry
- Simplifies database queries (no global entity table)

**Why Auto-Generate NPC Entity IDs?**
- Seamless: NPCs tracked without manual setup
- Flexible: Any named character can have relationships
- Scalable: Hundreds of NPCs supported without configuration

---

## Message Search System

### Overview

NeuralRP provides full-text search across all chat messages with advanced filtering, enabling you to quickly locate specific conversations, quotes, or topics.

### Features

- **Full-Text Search**: Search across all chat messages (both active and archived)
- **Message Context**: View surrounding messages for context (before/after snapshots)
- **Speaker Filtering**: Filter by character name
- **Term Highlighting**: Highlight matched terms in results
- **Jump-to-Message**: Navigate directly to specific message in chat
- **Phrase Support**: Use quotes for exact phrase matching

### Search Syntax

- **Simple terms**: `dragon magic` (matches either word)
- **Quoted phrases**: `"dragon magic"` (exact phrase match)
- **Wildcards**: `drag*` (matches dragon, dragons, draggable)
- **Boolean operators**: `dragon AND magic`, `dragon OR magic`
- **Negation**: `dragon NOT magic`

### API Endpoints

**GET /api/search/messages**
- Query params: `query` (required), `speaker` (optional), `limit` (default: 50)
- Returns: Results with highlighted terms and metadata
- Searches across both active and archived messages

**GET /api/search/messages/{id}/context**
- Returns: `{before: [], message: {}, after: []}` (3 messages before/after)

**GET /api/search/filters**
- Returns: Available speakers and date ranges

### Frontend Features

- Real-time search as you type
- Filter panel for speaker selection
- Click any result to jump directly to that message in the chat
- Context viewer shows surrounding conversation
- Highlighted search terms with <mark> tags

### Performance

| Operation | Typical Time |
|-----------|-------------|
| Full-text search (100 messages) | <100ms |
| Full-text search (10,000 messages) | <200ms |
| Context retrieval | <50ms |
| Term highlighting (client-side) | <10ms |

---

## Performance Mode

### Purpose

When running LLM + Stable Diffusion on the same GPU, Performance Mode prevents resource contention through intelligent queuing and automatic quality adjustment.

### Resource Management

Operations are classified by GPU impact:

- **Heavy**: Image generation, large context text, card generation
- **Light**: Small text generation, status checks, UI updates

**Queue Behavior**:
- Heavy LLM operations wait for SD to finish
- Light LLM operations interleave with SD
- SD operations always queue (always heavy)

### Context-Aware SD Presets

Image generation quality automatically adjusts based on story length:

| Preset | Steps | Size | Token Threshold |
|--------|-------|------|----------------|
| Normal | 20 | 512×512 | 0-7,999 |
| Light | 15 | 384×384 | 8,000-11,999 |
| Emergency | 10 | 256×256 | 12,000+ |

**Why**: Larger contexts require more GPU memory, leaving less for image generation. Reducing steps/size maintains performance without halting generation.

### Smart Hints

Context-aware suggestions triggered by performance metrics:

- **Contention detected**: "Images are slow because story is very long—consider a smaller model or shorter context for smoother images."
- **Emergency preset active**: "Image quality reduced due to high story context."
- **Very large context**: "Story context is very long. Consider summarizing or creating a branch to maintain performance."

Hints are dismissible and won't repeat after being dismissed.

### Performance Tracking

NeuralRP uses rolling median tracking (outlier-resistant) to detect when operations are slower than normal:

- Tracks last 10 LLM generation times
- Tracks last 10 SD generation times
- Detects contention when SD time exceeds 3× median

This ensures hints are based on your system's baseline, not arbitrary thresholds.

---

## Branching System

### Purpose

Create alternate "what-if" timelines from any point in a conversation without affecting the original.

### How It Works

1. **Fork from message**: Right-click any message and select "Fork Branch"
2. **New timeline created**: Contains all messages up to that point
3. **Independent**: Changes in branch don't affect original
4. **Settings inherited**: Characters, world info, and settings copied over

### Branch Independence

- Each branch is a separate database record
- No shared mutable state
- Editing one branch doesn't affect others
- Instant creation (O(1) row insert, not file copy)

### Branch Metadata

Each branch tracks its origin:

```json
{
  "origin_chat_id": "original_chat_name",
  "origin_message_id": 12345678,
  "branch_name": "Custom Branch Name",
  "created_at": 1736279867.123
}
```

### Management

- **Rename branches**: Change display name without affecting data
- **Delete branches**: Remove entire timeline and its messages
- **View origin**: Navigate back to original chat
- **List branches**: See all forks from a specific chat

### No Merge Semantics

Branches are independent timelines by design:
- Fork = create new timeline
- Switch = load that timeline
- No automatic merging or conflict resolution

This keeps the mental model simple: each branch is its own story.

---

## Memory & Summarization

### Automatic Summarization

When conversation context grows too large, NeuralRP automatically summarizes older messages to stay within the model's context window.

**Trigger Conditions**:
- Total tokens > 85% of model's max context
- Message count > 10 (minimum for summarization)

**Process**:
1. Take oldest 10 messages
2. Send to LLM for summary generation (150 tokens max)
3. Prepend new summary to existing summary field
4. Mark messages as `summarized=1` (soft delete, v1.6.0)

**Canon Law Echo**:
When summarizing, active canon law entries are included in the summarization prompt, ensuring world rules aren't lost in the summary.

### Summary Storage

Summaries are stored in the chat's `summary` field and included in context assembly, but not displayed in the UI. This keeps the conversation focused on recent messages while preserving long-term context.

### What Gets Kept vs Summarized

- **Verbatim**: Most recent messages up to ~85% of budget
- **Soft Deleted**: Oldest messages marked as `summarized=1` (persisted in database)
- **Result**: Sessions can run indefinitely without hitting context limits

### Manual Control

You can trigger summarization manually via API or by setting a `summarize_threshold` lower than default 85%.

---

## Soft Delete System

### Overview (v1.6.0)

The soft delete system preserves message history after summarization by marking messages as `summarized=1` instead of deleting them. This ensures persistent message IDs for relationship tracker continuity and enables full history search across active and archived messages.

### Why Soft Delete?

**Problem with Hard Delete**:
- Message IDs change after summarization
- Relationship tracker references become invalid
- Historical conversation lost permanently
- No way to search old messages

**Solution with Soft Delete**:
- Message IDs remain constant (persistent)
- Relationship tracker references stay valid
- Full history preserved and searchable
- Optional cleanup of old archives

### How It Works

**Summarization Process**:
1. Select oldest 10 messages for summarization
2. Mark messages as `summarized=1` in database
3. Generate summary and prepend to chat summary field
4. Messages remain in database with archived flag

**Active Messages**:
- `summarized=0` (default)
- Included in conversation context
- Displayed in UI
- Used for relationship analysis

**Archived Messages**:
- `summarized=1`
- Excluded from conversation context
- Hidden from UI (unless explicitly requested)
- Still searchable via search API
- Still tracked for relationship analysis

### Database Schema

**messages table** (updated with soft delete):
```sql
ALTER TABLE messages ADD COLUMN summarized BOOLEAN DEFAULT 0;

CREATE INDEX idx_messages_summarized ON messages(summarized);
```

**Performance Impact**:
- Index on `summarized` column enables fast filtering
- Minimal overhead (<10ms per chat save)
- Negligible storage impact (1 byte per message)

### API Endpoints

**GET /api/chats/{name}?include_summarized=true**
- Load chat with both active and archived messages
- Archived messages marked with `summarized: true` in response
- Optional parameter (default: false for backward compatibility)

**GET /api/chats/summarized/stats**
- Returns archive statistics:
  ```json
  {
    "total_messages": 1000,
    "active_messages": 800,
    "summarized_messages": 200,
    "oldest_summarized_timestamp": 1737000000
  }
  ```

**POST /api/chats/cleanup-old-summarized**
- Delete summarized messages older than specified days (default: 90)
- Request body: `{ "days": 90 }`
- Returns: `{ "deleted_count": 150 }`

**GET /api/search/messages** (enhanced)
- Searches across both active and archived messages
- No configuration needed (searches full history automatically)

### Relationship Tracker Continuity

**Persistent Message IDs**:
```python
# Before: Message IDs changed after summarization
message_id = 123  # Gets deleted, relationships break

# After: Message IDs persist forever
message_id = 123  # Marked summarized=1, relationships valid
```

**Analysis Accuracy**:
- Relationship tracker uses message IDs for tracking
- Soft delete ensures historical relationships preserved
- No broken references or orphaned relationship states

### Optional Cleanup

**Default Behavior**: No automatic cleanup (archives retained indefinitely)

**Manual Cleanup**:
- Trigger via API: `POST /api/chats/cleanup-old-summarized`
- Default retention: 90 days (configurable)
- Only deletes summarized messages (never active ones)

**Cleanup Strategy**:
- Gradual: Delete in batches to avoid lock contention
- Safe: Verify no active references before deletion
- Auditable: Log all cleanup operations

### Performance

| Metric | Value |
|--------|-------|
| Soft delete overhead | <10ms per chat save |
| Archive query time | <50ms (with index) |
| Cleanup time | <100ms per 100 messages |
| Storage impact | Negligible (1 byte per message) |

### Design Philosophy

**Why 90-Day Retention Default?**
- **Short-term**: Immediate access to recent history
- **Long-term**: Prevents database bloat
- **Configurable**: Users can extend or reduce based on needs
- **Balanced**: Retains most relevant history while managing storage

**Why Optional Cleanup?**
- **Flexibility**: Users control when to delete archives
- **Safety**: Prevents accidental data loss
- **Transparent**: Cleanup is explicit, not automatic
- **Auditable**: All cleanup operations logged

**Why Index on `summarized` Column?**
- **Performance**: Fast filtering of active vs archived messages
- **Scalability**: Efficient queries even with millions of messages
- **Minimal Overhead**: Small index size (1 byte per row)

---

## Autosave System

### Overview (v1.5.2)

NeuralRP automatically saves your chat on every turn, ensuring you never lose progress during active sessions.

### How It Works

- **Automatic**: Every LLM response saves to SQLite database
- **Unique IDs**: Generated as `new_chat_{timestamp}` format
- **Persistence**: Chat ID returned in API response for tracking across turns
- **No Configuration**: Always enabled, no user setup required

### Cleanup Behavior

Autosaved chats have two automatic cleanup mechanisms:

1. **7-Day Cleanup**: Autosaved chats older than 7 days are automatically removed
2. **Empty Chat Cleanup**: Chats with zero messages are removed on application startup

**Manually saved chats** (autosaved=False) are exempt from 7-day cleanup and persist indefinitely.

### Database Schema

Chats table includes `autosaved` BOOLEAN field:
- `True` = Autosaved chat (subject to 7-day cleanup)
- `False` = Manually saved chat (permanent)

### Branch Autosave

When you create a branch, it's automatically marked as autosaved and saved immediately. This creates a persistent timeline for exploration without manual saving.

### Performance

- Autosave overhead: <10ms per chat turn (SQLite transaction)
- Startup cleanup: <50ms on typical databases

---

## Image Generation

### A1111 Integration

NeuralRP connects to AUTOMATIC1111 WebUI for image generation:

- **txt2img API**: Generate images from text prompts
- **img2img API**: Inpainting with masks
- **Response**: Base64-encoded PNG

### Character Tag Expansion

You can reference characters in image prompts using bracketed names:

```
Input:  "[Alice] walking in forest"
Output: "Alice walking in forest space marine combat armor helmet visor futuristic"
```

How it works:
1. Identify bracketed names `[Name]` in prompt
2. Look up character's `danbooru_tag` extension
3. Replace `[Name]` with that tag

This ensures consistent visual appearance of characters across images.

### Metadata Storage

Every generated image stores its parameters in `app/images/image_metadata.json`:

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

This enables:
- **Reproducibility**: Regenerate images with exact same parameters
- **Debugging**: Understand why certain images look the way they do
- **Learning**: See what settings work best for your prompts

### Inpainting

Modify existing images with masks:
- Upload source image
- Draw mask over areas to change
- Provide new prompt for those areas
- Adjustable denoising strength (how much of new prompt affects the image)

---

## Connection Monitoring

### Adaptive Intervals

NeuralRP intelligently adjusts how often it checks for KoboldCpp and Stable Diffusion availability:

| State | Interval | Rationale |
|--------|-----------|-----------|
| Stable (5+ min) | 60s | Minimal overhead |
| Stable (initial) | 30s | Balance detection/overhead |
| Initial failure | 10s | Catch restarts quickly |
| Persistent failure (3+) | 5s | Aggressive recovery detection |

This reduces network overhead by 60-67% during stable connections while maintaining quick recovery when services restart.

### Background Tab Optimization

Uses Page Visibility API:
- Tab hidden: All monitoring paused (zero resource usage)
- Tab visible: Immediate health check, then resume normal interval

This is especially useful for users who keep NeuralRP open in the background.

### Health Check Endpoints

- **KoboldCpp**: `GET /api/v1/model` or `/v1/models`
- **Stable Diffusion**: `GET /sdapi/v1/options`
- **Timeout**: 5 seconds per request

### Connection Quality Indicators

The UI shows real-time status badges:
- **Idle**: No active operations
- **Running**: Operation in progress
- **Queued**: Operation waiting for resource (performance mode)

---

## Change History Data Recovery

### Overview (v1.6.0)

NeuralRP provides a complete interface for browsing, filtering, and restoring change history, enabling data recovery beyond the 30-second undo window.

### Features

**Full-Screen Modal**:
- Backdrop blur effect for focus
- Clean, modern UI with responsive design
- Accessible via Settings → "View Change History"

**Filtering**:
- **Entity Type**: All, Character, World Info, Chat
- **Entity ID**: Search by specific entity identifier
- **Result Limit**: 1-50 entries (default: 30)
- **Real-time filtering**: Updates as you type/type Enter to search

**Color-Coded Badges**:
- **Entity Type**:
  - Blue: Character
  - Green: World Info
  - Purple: Chat
- **Operation**:
  - Green: CREATE
  - Yellow: UPDATE
  - Red: DELETE

**One-Click Restore**:
- Click "Restore" button for UPDATE or DELETE operations
- Confirmation dialog prevents accidental restoration
- Automatic JSON export on restore
- Immediate data refresh after restoration

**Operation Details**:
- Shows entity ID and operation type
- Displays timestamp (human-readable)
- Shows before/after data (collapsible JSON)
- CREATE operations: "Restore" button disabled (cannot undo creation)
- UPDATE/DELETE operations: "Restore" button enabled

**Loading States**:
- Spinner shown while fetching change history
- Disabled buttons during restore operation
- Success notification after successful restoration

### API Endpoints

**GET /api/changes**
- Query params: `entity_type` (optional), `entity_id` (optional), `limit` (default: 20)
- Returns: `{changes: [], count: int}`

**GET /api/changes/{entity_type}/{entity_id}**
- Returns change history for specific entity

**GET /api/changes/{entity_type}/{entity_id}/last**
- Returns most recent change for entity

**POST /api/changes/restore**
- Restores entity to previous state from change log
- Request body: `{ "change_id": int }`
- Returns: `{ success: bool, entity_type: str, entity_id: str, restored_name: str, change_id: int }`

**GET /api/changes/stats**
- Returns statistics about change log

**POST /api/changes/cleanup**
- Manual cleanup of old change log entries
- Request body: `{ "days": int }` (default: 30)

### Frontend Implementation

**JavaScript Functions**:
- `fetchChangeHistory()` - Fetches change history with filters
- `restoreChange(change_id)` - Restores entity from change
- `canRestoreChange(change)` - Checks if change can be restored
- `refreshAllData()` - Refreshes characters, world info, chats after restore
- `openChangeHistoryModal()` - Opens full-screen modal
- `showNotification(message, type)` - Displays success/error notifications

**Modal Features**:
- Filter controls with real-time search
- Table display with color-coded badges
- Collapsible JSON viewers for before/after data
- Confirmation dialog for restore operations

### What Can Be Restored

**Phase 2 (v1.6.0)**:
- **UPDATE operations**: Restore to previous state
- **DELETE operations**: Restore deleted entity
- **All entity types**: Characters, world info, chats
- **No time limit**: Restore from entire 30-day history

**Limitations**:
- **CREATE operations**: Cannot undo creation (no previous state to restore)
- **30-day retention**: Changes older than 30 days automatically pruned

### Performance

| Metric | Value |
|--------|-------|
| Change history query time | <100ms (30 entries) |
| Restore operation time | <200ms (database write + JSON export) |
| Frontend rendering | <50ms (table + modals) |
| Total restore workflow | <500ms (network + database + refresh) |

### Design Philosophy

**Why Full-Screen Modal?**
- **Focus**: Backdrop blur draws attention to data
- **Space**: Large table fits comfortably on screen
- **Accessibility**: Clean, high-contrast interface

**Why 30-Day Retention?**
- **Storage**: Prevents unbounded database growth
- **Recovery**: Sufficient time to notice accidental changes
- **Performance**: Fast queries on 30-day subset
- **Configurable**: Users can extend if needed

**Why Color-Coded Badges?**
- **Scannability**: Quickly identify entity types and operations
- **Visual Hierarchy**: Important operations (DELETE) stand out
- **Accessibility**: Color-blind friendly with text labels

---

## Undo/Redo System

### Overview (v1.5.3 / Enhanced v1.6.0)

NeuralRP provides a safety net for accidental deletions with multiple recovery mechanisms:

1. **30-Second Undo Toast** (v1.5.3): Quick recovery for immediate mistakes
2. **Change History Data Recovery** (v1.6.0): Browse and restore from full 30-day history

### 30-Second Undo Toast

**How It Works**:
1. **Delete Operation**: You delete a character, chat, or world info entry
2. **Undo Toast Appears**: Shows "Deleted [Entity Name] [Undo] 30s" with countdown timer
3. **Click Undo**: Restores deleted entity instantly
4. **Success Notification**: Confirms restoration and refreshes list
5. **Timer Expires**: If you don't click undo within 30 seconds, toast disappears

### What Can Be Undone

- **Characters**: Full character card restoration
- **Chats**: Complete chat with all messages restored
- **World Info Entries**: Entry restoration with embedding

### Limitations

**Phase 1 (Toast - v1.5.3)**:
- Only DELETE operations can be undone
- 30-second time limit for undo
- No redo functionality
- No undo for CREATE or UPDATE operations

**Phase 2 (Change History - v1.6.0)**:
- UPDATE and DELETE operations can be restored
- No time limit (within 30-day retention)
- Full browsing, filtering, and selection interface
- No redo capability

### Change Logging Foundation

The undo system is built on the change logging infrastructure (v1.5.1), which tracks all significant operations:

**Logged Operations**:
- Characters: CREATE, UPDATE, DELETE
- World Info: CREATE, UPDATE, DELETE
- Chats: DELETE, CREATE_BRANCH, RENAME, CLEAR, ADD_CHARACTER, REMOVE_CHARACTER

**Retention**: 30 days rolling window with automatic cleanup

### Performance

- Change logging overhead: <5ms per operation
- Undo toast operation: <50ms (restores from JSON snapshot)
- Change history restore: <200ms (database write + JSON export)
- Toast display: <1ms (DOM manipulation)

---

## Character Name Consistency

### Overview (v1.6.0)

NeuralRP uses the `get_character_name()` helper function throughout the codebase to ensure consistent character name handling. This prevents the relationship tracker and other systems from getting confused by name variations.

### Critical Rule

**ALWAYS use `get_character_name()` when extracting character names from any data source.**

### Why This Matters

**Problem**: Inconsistent name extraction breaks relationship tracking:

```python
# BAD: First-name extraction
character = {"data": {"name": "Sally Smith"}}
first_name = character["data"]["name"].split()[0]  # "Sally"

# Relationship tracker looks for "Sally Smith" but finds "Sally"
# Result: Broken relationships!
```

**Solution**: Use helper function consistently:

```python
# GOOD: Use helper function
from main import get_character_name

character = {"data": {"name": "Sally Smith"}}
full_name = get_character_name(character)  # "Sally Smith"

# Relationship tracker finds "Sally Smith" consistently
# Result: Working relationships!
```

### Implementation

The helper function handles all common character reference formats:

```python
def get_character_name(character_obj: Any) -> str:
    """
    Extract character name consistently from any character reference.
    
    Args:
        character_obj: Can be:
            - Dict with 'data.name' (from db_get_character)
            - Dict with 'name' (from character card)
            - String (already a name)
    
    Returns:
        Full character name, never empty
    """
    if isinstance(character_obj, str):
        return character_obj.strip() or "Unknown"
    
    if isinstance(character_obj, dict):
        if 'data' in character_obj:
            name = character_obj['data'].get('name', 'Unknown')
            if name and isinstance(name, str):
                return name.strip()
        
        name = character_obj.get('name', 'Unknown')
        if name and isinstance(name, str):
            return name.strip()
    
    return "Unknown"
```

### Supported Input Formats

The helper function handles all these formats:

1. **Database character** (from `db_get_character`):
   ```python
   {"data": {"name": "John Smith", ...}}
   ```

2. **Character card** (from API request):
   ```python
   {"name": "Alice", ...}
   ```

3. **String** (already extracted):
   ```python
   "Bob Johnson"
   ```

4. **Nested with extensions**:
   ```python
   {"data": {"name": "Carol", "extensions": {...}}}
   ```

### Name Handling

**Returns**:
- Full character name ("John Smith", never just "John")
- Whitespace trimmed
- "Unknown" fallback if name missing

**Does NOT**:
- Extract first name only
- Create aliases or nicknames
- Modify or transform the name
- Lowercase or uppercase the name

### Usage Examples

**✅ CORRECT - Do This:**

```python
# In prompt construction
for char_obj in request.characters:
    char_name = get_character_name(char_obj)
    # char_name = "Sally Smith" (full name)

# In relationship tracking
from_char = get_character_name(character_a)
to_char = get_character_name(character_b)
state = db_get_relationship_state(chat_id, from_char, to_char)

# In speaker filtering
speaker_name = get_character_name(speaker_obj)
results = filter_by_speaker(results, speaker_name)
```

**❌ WRONG - Don't Do This:**

```python
# DON'T: Extract first name
char_name = char_obj["data"]["name"].split()[0]
# Result: "Sally" - Breaks relationships!

# DON'T: Create shortened versions
nickname = char_obj["data"]["name"].split()[0]
# Result: "Sally" - Breaks relationships!

# DON'T: Access data.name directly
name = char_obj["data"]["name"]
# Should use: get_character_name(char_obj)
```

### Testing

To verify the helper works correctly:

```python
# Test with various input formats
assert get_character_name("Sally Smith") == "Sally Smith"
assert get_character_name({"data": {"name": "John Doe"}}) == "John Doe"
assert get_character_name({"name": "Alice"}) == "Alice"
assert get_character_name("") == "Unknown"
assert get_character_name({"data": {}}) == "Unknown"

# Test with unicode names
assert get_character_name("Émilie") == "Émilie"
assert get_character_name("夏目") == "夏目"
```

### When to Use

**Use `get_character_name()` for:**
- Prompt construction (character profiles)
- Relationship tracking (entity identification)
- Speaker filtering (search and display)
- Message speaker field assignment
- Entity registration (entity ID system)
- API request/response handling

**Case-Insensitive Comparison:**
```python
# When comparing names, use .lower()
if name1.lower() == name2.lower():
    # Names match
```

### Performance

| Metric | Value |
|--------|-------|
| Function call overhead | <1ms (simple string operations) |
| Name resolution consistency | 100% (no broken references) |
| Relationship accuracy | Improved (no name collisions) |

### Related Systems

The character name consistency helper integrates with:

1. **Relationship Tracker**: Ensures entity IDs map consistently to character names
2. **Entity ID System**: Prevents name collision confusion
3. **Prompt Construction**: Consistent character names in system prompts
4. **Search System**: Reliable speaker filtering and results
5. **Change History**: Accurate entity identification in audit trail

---

## Design Decisions & Tradeoffs

### Why SQLite Instead of JSON Files?

**Pros**:
- ACID transactions prevent data corruption
- Indexed queries scale to 10,000+ entries
- Foreign keys ensure referential integrity
- Single file for all data (easier backup)

**Cons**:
- Requires migration from JSON (one-time setup)
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

- Below 85%: Room for model to generate response
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

### Why 30-Second Undo Window?

- Short enough: Forces decision before continuing work
- Long enough: Allows time to realize mistake
- Standard: Matches common UX patterns (Gmail, etc.)

Future versions may allow configurable undo duration.

### Why Semantic Relationships Instead of LLM?

**Pros**:
- Speed: <20ms vs 500ms+ for LLM analysis
- No context bloat: Embeddings computed separately
- Consistency: Same model used for world info
- Cost: Zero token cost

**Cons**:
- Limited to emotional dimensions defined in system
- Requires initial training period (20+ messages)

**Mitigation**: Five-dimensional model captures complex emotional states; gradual evolution allows relationships to develop naturally.

### Why Entity ID System?

**Pros**:
- Prevents name collisions (e.g., multiple "John" characters)
- Fork-safe: Relationships branch correctly
- Scalable: Supports hundreds of NPCs
- Type-safe: Clear distinction between characters, NPCs, users

**Cons**:
- Additional table to manage
- Slight complexity in entity lookup

**Mitigation**: Automatic entity registration and chat-scoped IDs simplify implementation.

### Why Soft Delete Instead of Hard Delete?

**Pros**:
- Persistent message IDs for relationship tracking
- Full history search across all messages
- Optional cleanup (user-controlled)
- No data loss from summarization

**Cons**:
- Slightly larger database size
- Additional index overhead

**Mitigation**: 90-day default cleanup balances storage vs retention; index ensures fast queries.

### Why Change History UI?

**Pros**:
- Complete audit trail beyond 30-second undo
- Filtering and search for easy navigation
- One-click restore for any operation
- Visual badges for quick identification

**Cons**:
- Additional frontend complexity
- 30-day retention limit

**Mitigation**: Clean, modal-based UI minimizes clutter; 30-day window sufficient for most recovery needs.

### Why Five-Dimensional Relationship Model?

**Pros**:
- Expressive: Captures complex emotional states
- Interpretable: Each dimension has clear meaning
- Balanced: Comprehensive without overwhelming
- Directional: Asymmetric relationships supported

**Cons**:
- Fixed dimensions (cannot add custom emotions)
- Requires semantic model for analysis

**Mitigation**: Five dimensions cover most common relationship dynamics; same model used for world info reduces overhead.

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Idle RAM usage | ~50MB |
| Startup time | <1 second |
| Semantic search (100 entries) | 20-50ms |
| Semantic search (10,000 entries) | <200ms |
| Relationship analysis (every 10 messages) | <20ms |
| Soft delete overhead | <10ms per turn |
| Autosave overhead | <10ms per turn |
| Cleanup overhead | <50ms on startup |
| Health check overhead | <10ms per check |
| Context assembly | <50ms |
| Total LLM generation | Depends on model (typically 500ms-5s) |
| Total image generation | Depends on settings (typically 2-10s) |

### Scalability

- **World Info Entries**: 10,000+ without degradation
- **Chats**: Thousands of chats with millions of messages
- **Search**: Linear scaling with message count
- **Embeddings**: Persistent across restarts, O(1) retrieval
- **Relationships**: Hundreds of entities with millions of state updates
- **Change History**: 30-day rolling window, fast indexed queries

---

## API Reference Summary

For complete API documentation, see the source code in `main.py`. Key endpoints include:

**Character Management**: `/api/characters` (GET, POST, DELETE, edit-field, edit-capsule)

**World Info**: `/api/world-info` (GET, POST, edit-entry, add-entry, delete-entry)

**Chats**: `/api/chats` (GET, POST, DELETE, fork, branches, cleanup-old-summarized, summarized/stats)

**LLM**: `/api/chat`, `/api/extra/tokencount`, `/api/classify-mode`

**Images**: `/api/generate-image`, `/api/inpaint`, `/api/image-metadata/{filename}`

**Search**: `/api/search/messages`, `/api/search/messages/{id}/context`, `/api/search/filters`

**Undo**: `/api/undo/last`

**Relationships**: `/api/relationships/update`, `/api/relationships/{chat_id}`, `/api/relationships/{chat_id}/history`

**Entities**: `/api/entities/{chat_id}`

**Health**: `/api/health/kobold`, `/api/health/sd`, `/api/health/status`

**Performance**: `/api/performance/status`, `/api/performance/toggle`, `/api/performance/dismiss-hint`

**Changes**: `/api/changes` (GET stats, history, restore, cleanup)

---

**Last Updated**: 2026-01-19
**Version**: 1.6.0
