# Changelog

All notable changes to NeuralRP are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

***

## [1.10.4] - 2026-02-05

### Changed
- **Relationship Context Positioning**: Moved to end of prompt for maximum LLM attention before generation

### Fixed
- **Snapshot Variation Mode**: Implemented alternative phrase generation through example removal
- **Relationship System**: Fixed crashes and tracking failures (error logging, per-chat turn counters, entity ID standardization, ChatMessage property access, semantic filtering, keyword polarity regex, template range lookup)
- **Branching System**: Transaction-based refactor for reliable chat forking (atomic transactions, metadata remapping, NPC field preservation, cleanup on failed forks)
- **PList Generation**: Robust format detection and output for character fields and danbooru tag generation
- **Tag Filtering**: Fixed character/world tag filtering by removing stale capsule column reference
- **Turn Counting**: Fixed turn-based logic during summarization by capturing turn count at request start
- **Character/NPC Re-appearance**: Fixed injection logic for characters returning after long absence
- **NPC Bleeding**: Fixed metadata isolation between forked and original chats

### Technical
- Added `get_entity_id()` helper for consistent entity ID extraction
- Updated relationship functions to accept full character objects
- Added `NEURALRP_DISABLE_SEMANTIC_SCORING=1` environment variable

***

## [1.10.3] - 2026-02-04

### Changed
- **Snapshot System Overhaul**: Primary character focus with 5-field JSON extraction (location, action, activity, dress, expression), 20-message context window, character card injection, simplified fallback chain, enhanced danbooru tag compatibility
- **Relationship Tracker Enhancement**: Hybrid semantic + keyword scoring (70% semantic, 30% keyword) for more accurate emotional tracking

### Fixed
- **Chat Forking**: Fixed foreign key constraint causing branch creation failure when chat contains NPCs

### Technical
- Snapshot Analyzer rewrite (~400 lines modified)
- Relationship Tracker added hybrid scoring methods (~200 lines added)

***

## [1.10.2] - 2026-02-03

 ### Added
 - **Summaries Panel**: New UI sidebar panel for managing chat summaries
   - Displays current summary text (fully editable)
   - User can enter their own summary to start in the middle of a story
   - Autosave-enabled - changes persist automatically
   - Shows all generated summaries in one convenient location

 ### Addition
 - **Auto-Summaries Append to Manual Summaries**: Auto-summaries now correctly append to existing manual summary text
   - **Behavior**: Auto-summaries add new paragraphs underneath existing content (non-destructive)
   - **Format**: Existing summary + newline + new auto-generated summary
   - **Persistence**: Manual edits preserved, auto-summaries accumulate over time
   - **UI**: Textarea updates automatically after each auto-summarization via x-model binding
   - **Example**: Manual summary "Alice is brave" → After auto: "Alice is brave\n\nAlice discovered artifact."

***

### Fixed
- **Turn Calculation**: Fixed incorrect turn numbering causing canon law and character reinforcement to fire on wrong turns
  - **Old Logic**: `current_turn = len(messages) // 2` produced 0-indexed turns (0, 1, 1, 2, 2, 3, 3...)
  - **New Logic**: `current_turn = sum(1 for msg in messages if msg.role == "user")` produces 1-indexed turns (1, 2, 3, 4, 5...)
  - **Impact**: Canon law now correctly reinforces on turns 1, 2, 5, 8, 11... instead of incorrect pattern
  - **Impact**: Character reinforcement now correctly reinforces on turns 5, 10, 15, 20... instead of wrong turns
  - **is_initial_turn**: Updated to `current_turn <= 2` for consistency (turns 1 and 2)
  - **Canon Law Formula**: Updated to `is_initial_turn OR (current_turn > 2 AND (current_turn - 2) % world_reinforce_freq == 0)`

 ***

## [1.10.1] - 2026-02-03

### Changed
- **Danbooru Tag Generator**: Two-stage matching with progressive exact matching (~100ms, 8-14x faster)
- **Snapshot System**: Simplified with direct LLM JSON extraction (~40% faster, ~800 lines removed)

### Removed
- Snapshot learning system (`app/snapshot_learning_config.py`)
- Complex tag selection and semantic search for snapshot scene analysis

### Fixed
- **NPC Key Consolidation**: Fixed split-key issue causing NPC creation/edit failures
- **Database**: Schema v3 migration merges split NPC keys, added semantic search for danbooru tags

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
- **Snapshot Feature**: Generate Stable Diffusion images directly from chat scenes (5-block prompt structure, hybrid scene detection, SQLite-vec embeddings for 1560 tags)
- **Snapshot Variation Mode**: Regenerate snapshots with novelty scoring for exploration
- **Unified Favorites System**: User-level learning biasing all image generations (supports snapshot and manual modes)
- **Tag Preference Tracking**: Automatic learning from favorited images with tag frequency tracking
- **Favorites Jump-to-Source**: Double-click favorites to jump to original chat context

### Changed
- **Tag Configuration**: Standalone `app/danbooru_tags_config.py` with 1560 tags organized by 4 blocks

### Technical
- Database: Added `danbooru_tags` and `vec_danbooru_tags` tables
- Consolidated migration management via `app/database_setup.py`
- Playwright-based test suite with 48 comprehensive tests

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
- **Tag Management System**: Lightweight library-scale organization for characters and worlds (AND semantics filtering, quick filter chips, tag editor with autocomplete, automatic normalization)
- **Smart Sync System**: Intelligent JSON import with timestamp-based conflict resolution (newer wins, entry-level smart sync for world info)
- **Automatic Tag Extraction**: Tags from SillyTavern cards automatically preserved on import/save

### Changed
- Character and world sidebars: Added tag filter bars
- Character and world edit forms: Added chip-based tag editors

***

## [1.7.3] - 2026-01-28

### Added
- **Immediate Edit Notifications**: Character, NPC, and world info edits appear in "Recent Updates" on next chat turn

### Fixed
- Character and NPC card editing now sync immediately to active chats
- World info semantic matching now properly triggers entries
- Fixed chat message edit window collapsing

***

## [1.7.2] - 2026-01-28

### Fixed
- World info editing now persists changes to database

  ***

## [1.7.1] - 2026-01-27

### Fixed
- **Character Card Enforcement**: Fixed cards/capsules appearing in every prompt instead of first appearance
- **World Info Triggered Lore**: Fixed missing semantic entries in prompts
- **Reinforcement Logic**: Fixed intervals to use actual turn numbers

***

## [1.7.0] - 2026-01-27

### Added
- **Chat-Scoped NPC System**: Create and manage NPCs within individual chats (fork-safe, promote to global characters, entity ID tracking)
- **Training Data Export**: Export chat data including NPC responses in Alpaca, ShareGPT, and ChatML formats

### Changed
- Context assembly resolves both global characters and local NPCs
- Chat forking remaps NPC entity IDs for branch isolation
- Relationship tracking stores NPCs using entity IDs

***

## [1.6.1] - 2026-01-21

### Added
- **Adaptive Relationship Tracker**: Real-time three-tier detection (keyword, semantic similarity, dimension filtering) with 3-5ms overhead

### Changed
- Relationship context injection filtered by semantic relevance (>0.35 threshold)
- Only dimensions with >15-point deviation from neutral are injected

  ***

## [1.6.0] - 2026-01-19

### Added
- **Semantic Relationship Tracker**: Automatic tracking between characters, NPCs, and users (5 emotional dimensions, directional tracking, 20-snapshot history)
- **Entity ID System**: Unique IDs prevent name collisions
- **Change History UI**: Full-screen interface for browsing, filtering, and restoring changes
- **Soft Delete**: Messages marked `summarized=1` instead of deleted

### Changed
- Prompt construction includes relationship context automatically
- Database: Added `summarized` field to messages table

***

## [1.5.3] - 2026-01-18

### Added
- **Search System**: Full-text search across chat messages with highlighting and jump-to-message
- **Undo/Redo Phase 1**: 30-second undo toast after deleting characters, chats, or world info

  ***

## [1.5.2] - 2026-01-18

### Added
- **Autosave System**: Automatic chat persistence with 7-day cleanup

  ***

## [1.5.1] - 2026-01-13

### Added
- **Change Logging System**: Complete audit trail for undo/redo support
- **Database Health Check**: Automatic startup validation

### Fixed
- Fixed SQL bug in `db_search_similar_embeddings()`
- Fixed memory leak in embeddings deletion
- Added 768-dimension validation

***

## [1.5.0] - 2026-01-13

### Added
- **SQLite Database**: Centralized storage with ACID guarantees
- **sqlite-vec Integration**: Disk-based embeddings with SIMD acceleration
- **Automatic Migration**: JSON → SQLite conversion script

### Changed
- Vector search backend: Replaced pickle + sklearn with sqlite-vec
- Performance: Idle RAM ~50MB (down from ~300MB), startup <1 second

### Removed
- Pickle dependencies
- Unbounded cache growth (LRU eviction enabled)

  ***

## [1.4.0] - 2026-01-12

### Added
- **Image Inpainting**: Full inpainting support via Stable Diffusion img2img API
- **Image Metadata System**: Automatic storage
- **World Info Reinforcement Configuration**: Configurable canon law reinforcement

### Changed
- Semantic search: 0.35 threshold initial, 0.45 subsequent turns
- Deduplication: Comprehensive handling of plurals, apostrophes, possessives

***

## [1.3.0] - 2026-01-08

### Added
- **In-Card Editing**: Full character and world info editing interface
- **Semantic World Info Retrieval**: Context-aware lore retrieval using `all-mpnet-base-v2` with GPU/CPU fallback
- **LRU Cache System**: Memory-safe world info caching (default: 1000 entries)

  ***

## [1.2.0] - 2026-01-08

### Added
- **Automatic Performance Mode**: Smart GPU resource management for LLM + SD
- **SD Context-Aware Presets**: Automatic quality adjustment based on story length
- **Smart Hint Engine**: Context-aware optimization suggestions

### Changed
- **Adaptive Connection Monitoring**: 60-67% reduction in network overhead
- **Background Tab Optimization**: Automatic pause when tab not visible

***

## [1.1.0] - 2026-01-07

### Added
- **Branching System**: Fork from any message to create alternate timelines
- **Dual-Source Card Generation**: Create cards from chat OR manual text input
- **Live Editing**: AI-generated content in editable textboxes before saving
- **Efficient World Info**: Cached structures with configurable entry cap

  ***

## [1.0.0] - 2026-01-06

### Added
- **KoboldCpp Integration**: Native and OpenAI-compatible API support
- **Stable Diffusion Integration**: AUTOMATIC1111 WebUI for image generation
- **SillyTavern Compatibility**: V2 JSON format for character cards and World Info
- **Chat Modes**: Narrator, Focus, Auto
- **Multi-Character Support**: Capsule personas for distinct voices
- **Automatic Summarization**: Triggered at 85% context
- **Canon Law System**: Mark immutable World Info entries
- **Danbooru Character Tags**: Per-character visual canon with `[CharacterName]` expansion
- **AI Card Generation**: Create character cards and world info from conversations
- **Token Counter**: Real-time context usage monitoring
- **Chat Persistence**: Save/load sessions with characters, world, and images

***
