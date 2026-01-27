# Changelog

All notable changes to NeuralRP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

***

## [1.7.0] - 2026-01-27

### Added
- **Chat-Scoped NPC System**: Create and manage non-player characters within individual chats
  - NPCs exist only in their chat context, preventing cross-contamination
  - Fork-safe: Branching creates independent NPC copies with unique entity IDs
  - Toggle NPCs active/inactive without deletion
  - Promote chat-scoped NPCs to global characters with audit trail
  - Entity ID format: `npc_<chat_id>_<timestamp>_<hash>`
- **Training Data Export**: Export chat data including NPC responses in Alpaca, ShareGPT, and ChatML formats
- **NPC Integration**: NPCs participate fully in relationship tracking, context assembly, and multi-character optimization

### Changed
- Context assembly now resolves both global characters and local NPCs via `load_character_profiles()`
- Chat forking enhanced to remap NPC entity IDs for branch isolation
- Relationship tracking stores NPCs using entity IDs with automatic remapping on fork

### Technical
- Database: Entities table updated with `chat_id` and `entity_type` fields for NPC support
- Chat metadata: NPCs stored in `metadata.local_npcs` object
- API endpoints: NPC management (`/api/chats/{chat_id}/npcs/*`), training export (`/api/chats/{chat_id}/export-training`)

***

## [1.6.1] - 2026-01-21

### Added
- **Adaptive Relationship Tracker**: Real-time three-tier detection of dramatic relationship shifts
  - Tier 1: Keyword detection (~0.5ms) with 60+ relationship keywords
  - Tier 2: Semantic similarity (~2-3ms) detects conversational shifts below 0.7 threshold
  - Tier 3: Dimension filtering (~1-2ms) injects only relevant dimensions
  - Spam blocker: 3-turn cooldown prevents repeated triggers
  - Total overhead: 3-5ms per turn vs 5ms static injection

### Changed
- Relationship context injection now filtered by semantic relevance (>0.35 threshold)
- Only dimensions with >15-point deviation from neutral (50) are injected
- Randomized natural language templates prevent repetitive context

***

## [1.6.0] - 2026-01-19

### Added
- **Semantic Relationship Tracker**: Automatic tracking between characters, NPCs, and users
  - Five emotional dimensions: Trust, Emotional Bond, Conflict, Power Dynamic, Fear/Anxiety
  - Analyzes every 10 messages using semantic embeddings (<20ms overhead)
  - Directional tracking (Alice→Bob separate from Bob→Alice)
  - 20-snapshot history per relationship
- **Entity ID System**: Unique IDs prevent name collisions (supports short names like "Mark" or "Jo")
- **Change History UI**: Full-screen interface for browsing, filtering, and restoring changes
  - Filter by entity type, operation, and date
  - Color-coded badges for entity types and operations
  - One-click restore for UPDATE/DELETE operations with confirmation
- **Soft Delete**: Messages marked `summarized=1` instead of deleted, preserving persistent IDs

### Changed
- Prompt construction now includes relationship context automatically
- Database schema: Added `summarized` BOOLEAN field to messages table
- Chat API: Enhanced with `/api/chats/{name}?include_summarized=true` for archived message retrieval

***

## [1.5.3] - 2026-01-18

### Added
- **Search System**: Full-text search across all chat messages
  - Message search with query highlighting and phrase support
  - Advanced filtering by speaker, date range, and limit
  - Jump-to-message functionality for quick navigation
- **Undo/Redo Phase 1**: 30-second undo toast after deleting characters, chats, or world info entries

***

## [1.5.2] - 2026-01-18

### Added
- **Autosave System**: Automatic chat persistence on every turn
  - Unique chat ID generation (`new_chat_{timestamp}`)
  - 7-day automatic cleanup of autosaved chats
  - Empty chat cleanup on application startup

***

## [1.5.1] - 2026-01-13

### Added
- **Change Logging System**: Complete audit trail for undo/redo support
  - Tracks create/update/delete operations with JSON snapshots
  - 30-day rolling cleanup
  - API endpoints: `/api/changes` for querying history
- **Database Health Check**: Automatic startup validation with `PRAGMA integrity_check`

### Fixed
- Critical SQL bug: Fixed `db_search_similar_embeddings()` WHERE clause using column alias
- Memory leak: Embeddings now properly deleted when World Info entries removed
- Data integrity: Added 768-dimension validation to prevent model mismatch crashes

***

## [1.5.0] - 2026-01-13

### Added
- **SQLite Database**: Centralized storage in `app/data/neuralrp.db` with ACID guarantees
- **sqlite-vec Integration**: Disk-based embeddings with SIMD acceleration (AVX2/SSE2)
- **Automatic Migration**: `migrate_to_sqlite.py` script for JSON → SQLite conversion

### Changed
- Vector search backend replaced pickle + sklearn with sqlite-vec
- Semantic search now uses SIMD-accelerated similarity with numpy fallback
- Performance: Idle RAM ~50MB (down from ~300MB with pickle)
- Startup time: <1 second (down from 2-5 seconds)

### Removed
- Pickle dependencies (security improvement)
- Unbounded cache growth: LRU eviction prevents memory leaks

***

## [1.4.0] - 2025-12-15

### Added
- **Image Inpainting**: Full inpainting support via Stable Diffusion img2img API
- **Image Metadata System**: Automatic storage in `app/images/image_metadata.json`
- **World Info Reinforcement Configuration**: Configurable canon law reinforcement (default: every 3 turns)

### Changed
- Semantic search: Initial turn uses 0.35 threshold, subsequent turns use 0.45 threshold
- Deduplication: Comprehensive handling of plurals, apostrophes, possessives

***

## [1.3.0] - 2025-11-20

### Added
- **In-Card Editing**: Full character and world info editing interface
- **Semantic World Info Retrieval**: Context-aware lore retrieval using `all-mpnet-base-v2`
  - Cosine similarity matching (default: 0.25 threshold)
  - Automatic GPU detection with CPU fallback
- **LRU Cache System**: Memory-safe world info caching (default: 1000 entries)

***

## [1.2.0] - 2025-10-15

### Added
- **Automatic Performance Mode**: Smart GPU resource management for LLM + SD
  - Queues heavy operations while allowing light tasks
  - Rolling median tracking for contention detection
- **SD Context-Aware Presets**: Automatic quality adjustment based on story length
  - Normal (512×512, 20 steps) for 0-7999 tokens
  - Light (384×384, 15 steps) for 8000-11999 tokens
  - Emergency (256×256, 10 steps) for 12000+ tokens
- **Smart Hint Engine**: Context-aware optimization suggestions

### Changed
- **Adaptive Connection Monitoring**: 60-67% reduction in network overhead during stable connections
- **Background Tab Optimization**: Automatic pause when browser tab not visible

***

## [1.1.0] - 2025-09-10

### Added
- **Branching System**: Fork from any message to create alternate timelines
- **Dual-Source Card Generation**: Create cards from chat OR manual text input
- **Live Editing**: AI-generated content in editable textboxes before saving
- **Efficient World Info**: Cached structures, configurable entry cap (default: 10), Canon Law entries always included

***

## [1.0.0] - 2025-08-01

### Added
- **KoboldCpp Integration**: Native and OpenAI-compatible API support
- **Stable Diffusion Integration**: AUTOMATIC1111 WebUI for image generation
- **SillyTavern Compatibility**: V2 JSON format for character cards and World Info
- **Chat Modes**: Narrator (third-person), Focus (first-person), Auto (AI chooses)
- **Multi-Character Support**: Capsule personas for distinct voices
- **Automatic Summarization**: Triggered at 85% context
- **Canon Law System**: Mark immutable World Info entries
- **Danbooru Character Tags**: Per-character visual canon with `[CharacterName]` expansion
- **AI Card Generation**: Create character cards and world info from conversations
- **Token Counter**: Real-time context usage monitoring
- **Chat Persistence**: Save/load sessions with characters, world, and images

***

## Key Changes from Original

### Removed
- **Performance tables**: Moved technical metrics to documentation, kept only headline numbers
- **"Technical", "Performance", "Notes" subsections**: Consolidated into main sections or removed redundant info
- **Verbose explanations**: Trimmed to essential information
- **Design philosophy sections**: Moved to technical docs, kept only user-facing features

### Improved
- **Scannability**: Each version now fits on ~1 screen
- **Hierarchy**: Clear Added/Changed/Fixed/Removed structure
- **Focus**: Emphasizes what users can DO, not how it works internally
- **Consistency**: Uniform formatting across all versions

### Retained
- **Technical depth**: Key implementation details for developers
- **Breaking changes**: Migration notes for v1.5.0
- **Version dates**: Full chronological history

***
