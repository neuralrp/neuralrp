# Changelog

All notable changes to NeuralRP are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

***

## [1.10.0] - 2026-02-01

### Added
- **Danbooru Tag Generator**: One-click semantic matching to generate Danbooru tags from character descriptions
  - Click "Generate Danbooru Character (Visual Canon)" hyperlink in character editor to auto-populate field
  - Click again to reroll for different results (progressive search finds best match)
  - Smart trait mapping from natural language to Danbooru vocabulary (e.g., "elf" → `pointy_ears`)
  - Works for both global characters and local NPCs
  - Best results with anime-trained SD models (Pony, Illustrious, NoobAI)
  - Optional: Generate Danbooru character library via `app/fetch_danbooru_characters.py` (requires Danbooru API key, 15-30 min)

- **Gender Field**: Visual gender selection now integrated across character editor, prompt assembly, and image generation
  - Three-button toggle in character editor (Female/Male/Other) with color-coded styling
  - NPCs can set gender just like global characters
  - Stored in SillyTavern-compatible format (`data.extensions.gender`) for full backward compatibility
  - Leave blank to use existing behavior (no breaking changes)

- **Gender-Aware LLM Prompts**: Gender explicitly stated throughout conversation context
  - Full character cards display gender field
  - Multi-character capsules include gender for accurate dialogue generation
  - Reinforcement reminders state character gender every 5 turns
  - Reduces pronoun confusion and improves consistency

- **Gender-Aware Image Generation**: Automatic character counting by gender in snapshots
  - Auto-counts active characters: `1girl`, `2girls`, `1boy`, `2boys`, `1girl, 1boy`, etc.
  - Gender field takes precedence over manual danbooru_tag count modifiers for accuracy
  - Supports multi-character scenes (up to 3 characters) with aggregated danbooru tags
  - Mixed scenarios handled gracefully with `solo`/`multiple` fallbacks

### Changed
- **Snapshot System**: Simplified API—only `chat_id` required, backend auto-resolves active characters
- **Migration**: Schema version 2 adds optional `danbooru_characters` table for visual canon library

***

## [1.9.0] - 2026-01-31

### Added
- **Snapshot Feature**: Generate Stable Diffusion images directly from chat scenes with automatic prompt construction
  - 4-block prompt structure (quality, subject, environment, style) with ~14 tags per generation
  - Hybrid scene detection: keyword minimum (2+ matches) + LLM summary (optional) + semantic matching
  - Character dropdown integration with danbooru_tag support for visual consistency
  - SQLite-vec embeddings for 1560 danbooru tags (768-dimensional vectors)
  - Snapshot history stored in chat metadata (chat-scoped, no database persistence)
  - Red light indicator for Stable Diffusion API unavailability
  - Toggleable prompt details (positive/negative prompt, scene analysis)
  - Comprehensive test suite: 48 tests across 3 phases (unit, integration, edge cases)

- **Snapshot Variation Mode**: Regenerate snapshots with novelty scoring
  - Single "🔄 Regenerate" button below snapshot images
  - Applies novelty scoring to explore less-used tag combinations
  - Same scene analysis with different tag selection for variety
  - Encourages exploration while maintaining scene relevance

- **Unified Favorites System**: User-level learning that biases ALL image generations
  - Single favorites table supports BOTH snapshot and manual mode images
  - Favorite images tracked via `source_type` field ('snapshot' or 'manual')
  - Toggle favorite status with heart icon (❤️) on snapshots
  - "Save as Favorite" button for manual mode images (Vision Panel)
  - Favorites persist across chat sessions

- **Tag Preference Tracking**: Automatic learning from user's favorited images
  - Analyzes tags in favorited images to understand user preferences
  - Automatic tag detection for danbooru tags in prompts (2+ tag threshold)
  - Tag frequency tracking for bias calculation
  - Future generations biased toward your preferred tags
  - Works for BOTH snapshot and manual mode favorites

- **Tag Detection System**: Automatic detection of danbooru tags in custom prompts with 2+ tag threshold for learning activation
- **Favorites Jump-to-Source**: Double-click any favorite image to jump directly to its original chat context with visual highlight
- **Manual Favorites Chat Association**: Manual mode favorites now store optional chat_id for jump-to-source functionality

### Changed
- **Tag Configuration System**: Standalone `app/danbooru_tags_config.py` with 1560 tags organized by 4 blocks (Quality, Subject, Environment, Style)

### Technical
- **Database Schema**: Added `danbooru_tags` and `vec_danbooru_tags` tables for 1560 danbooru tag embeddings
- **API Endpoints**: `POST /api/chat/{chat_id}/snapshot`, `GET /api/snapshot/status`, `POST /api/manual-generation/favorite` (accepts optional `chat_id`)
- **Database Setup System**: Consolidated migration management via `app/database_setup.py`
  - Single file creates all 17 core tables, indexes, triggers, and virtual tables
  - Schema version tracking for seamless upgrades (1.8.0 → 1.9.0 and beyond)
  - Automatic migration handling in `launcher.bat` - runs every startup, idempotent
- **Testing**: Playwright-based test suite with 48 comprehensive tests (unit, integration, edge cases)

***

## [1.8.2] - 2026-01-30

### Added
- **Chat-Scoped Capsule System**: Capsules stored in `chat.metadata.characterCapsules` and regenerated on-demand for multi-character chats
  - Captures character edits automatically, eliminates JSON/database persistence
  - Capsule fallback to description + personality for missing capsules
- **Demo folder and Quickstart guide** - Get started right away with demo cards and with pertinent info on the app

### Changed
- **Database Migration**: Removed `capsule` column from `characters` table
- **Capsule Generation**: Removed creation-time generation; capsules generated on-demand for multi-char chats only
- **Multi-Character Logic**: Unified capsule system for global characters and NPCs

### Fixed
- **Alpaca Export Context Missing**: Character context now included in every training example (not just first)
- **New Chat First Turn Persistence**: `characterFirstTurns` now saved correctly for new unsaved chats
- **Auto-First Message Suppressed**: Characters no longer auto-add `first_mes`; users always go first

### Removed
- **Multi-Char Capsule Editing UI**: Manual capsule editing section from character menu
- **Capsule API Endpoints**: Manual capsule generation/editing endpoints (`/api/card-gen/generate-capsule`, `/api/characters/edit-capsule*`)
- **JSON File Writing for Capsules**: Capsules no longer written to character JSON files
- **Database Capsule Column**: Removed from character queries (`db_save_character`, `db_get_character`)
- **World Info Probability UI**: Removed probability checkbox and input (SillyTavern compatibility only)

 ***

## [1.8.1] - 2026-01-29

### Added
- **Character Edit Override System**: Full card injection when characters are edited mid-chat
- **SillyTavern Compatibility**: Character card generation now uses correct field mappings
  - Personality traits saved to `personality` field (comma-separated)
  - Body traits appended to `description` field
  - Tags changed from auto-generated to manual-only
  - Removed NeuralRP-specific fields for better SillyTavern compatibility

### Changed
- **NPC Creation**: Updated to use same SillyTavern-compatible field mappings
- **Field Generation Prompts**: LLM prompts updated for SillyTavern output format
- **Frontend Card Generator**: Removed tags from auto-generation, updated field mappings

***

## [1.8.0] - 2026-01-28

### Added
- **Tag Management System**: Lightweight library-scale organization for characters and worlds
  - Add tags to characters and worlds (e.g., "campaign", "fandom", "NSFW", "WIP")
  - Filter character and world lists by tags with AND semantics (character must have ALL selected tags)
  - Quick filter chips for most-used tags (top 5) for fast access
  - Tag editor with autocomplete suggestions (shows existing tags as you type)
  - Tags normalized on save (lowercase, trimmed, no duplicates)
  - Integrated with existing search feature (filter by tags AND search by name/description)

- **Smart Sync System**: Intelligent JSON import with timestamp-based conflict resolution
  - Database now tracks `updated_at` timestamps for characters and world entries
  - Character edits sync immediately to database with automatic conflict detection (newer wins)
  - World info supports entry-level smart sync: new entries from JSON added, user additions preserved, conflicts resolved by timestamp
  - Backfills existing records with timestamps on first startup
  - API endpoints: `/api/reimport/worldinfo?smart_sync=true` for intelligent merge, `force=true` for complete reimport

- **Automatic Tag Extraction**: Tags from SillyTavern cards automatically preserved on import/save
  - Character tags extracted from `char.data.tags` array (SillyTavern V2 format)
  - World tags extracted from `world.tags` array (JSON format)
  - One-time migration script extracts tags from existing characters/worlds on startup
  - No manual tag management required - just drop cards in folder and refresh

### Changed
- **Character Sidebar**: Added tag filter bar above character list
- **World Info Sidebar**: Added tag filter bar above world list
- **Character Edit Form**: Replaced simple text input with chip-based tag editor
- **World Edit Form**: Added tag editor to world entry editing interface
- **Auto-Import**: Now uses smart sync for world info (preserves user additions from UI)

***

## [1.7.3] - 2026-01-28

### Added
- **Immediate Edit Notifications**: Character, NPC, and world info card edits now appear in "Recent Updates" on next chat turn

### Fixed
- **Character Card Editing**: Changes now sync immediately to active chats without refresh
- **NPC Card Editing**: Mid-chat NPC edits now take effect on next message
- **World Info Semantic Matching**: Keys now properly trigger entries; quoted keys require exact match, unquoted use semantic search
- **Chat Message Editing**: Fixed edit window collapsing to narrow box

***

## [1.7.2] - 2026-01-28

### Fixed
- **World Info Saving**: Fixed bug where editing world cards did not persist changes
  - Previously changes were saved to JSON but never written to database, causing edits to appear lost after refresh
  - Frontend `saveWorldEntry()` now saves all editable fields (key, comment, content, is_canon_law, useProbability, probability) instead of only content

***

## [1.7.1] - 2026-01-27

### Fixed
- **Character Card Enforcement**: Fixed critical bug where character cards and capsules appeared in every prompt instead of only on first appearance
  - Added `character_has_speaker()` helper function to track character appearance in message history
  - Group chats: Capsules now only shown on character's first appearance OR turn 1
  - Single character: Full card now only shown on turn 1 (not every turn)
  - Prevents redundant context injection and reduces token overhead significantly
- **World Info Triggered Lore**: Fixed missing semantic world info entries in prompts
  - `triggered_lore` was fetched via semantic search but never added to prompt
  - Now displays semantically-matched world knowledge entries as "### World Knowledge:" section
- **Reinforcement Logic**: Fixed reinforcement intervals to use actual turn numbers instead of message indices

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
  - Tier1: Keyword detection (~0.5ms) with 60+ relationship keywords
  - Tier2: Semantic similarity (~2-3ms) detects conversational shifts below 0.7 threshold
  - Tier3: Dimension filtering (~1-2ms) injects only relevant dimensions
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

## [1.4.0] - 2026-01-12

### Added
- **Image Inpainting**: Full inpainting support via Stable Diffusion img2img API
- **Image Metadata System**: Automatic storage in `app/images/image_metadata.json`
- **World Info Reinforcement Configuration**: Configurable canon law reinforcement (default: every 3 turns)

### Changed
- Semantic search: Initial turn uses 0.35 threshold, subsequent turns use 0.45 threshold
- Deduplication: Comprehensive handling of plurals, apostrophes, possessives

***

## [1.3.0] - 2026-01-08

### Added
- **In-Card Editing**: Full character and world info editing interface
- **Semantic World Info Retrieval**: Context-aware lore retrieval using `all-mpnet-base-v2`
  - Cosine similarity matching (default: 0.25 threshold)
  - Automatic GPU detection with CPU fallback
- **LRU Cache System**: Memory-safe world info caching (default: 1000 entries)

***

## [1.2.0] - 2026-01-08

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

## [1.1.0] - 2026-01-07

### Added
- **Branching System**: Fork from any message to create alternate timelines
- **Dual-Source Card Generation**: Create cards from chat OR manual text input
- **Live Editing**: AI-generated content in editable textboxes before saving
- **Efficient World Info**: Cached structures, configurable entry cap (default: 10), Canon Law entries always included

***

## [1.0.0] - 2026-01-06

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
