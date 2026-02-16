# Changelog

All notable changes to NeuralRP are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

***

## [2.0.3] - 2026-02-16

### Changed
- **NPC Active Status Storage**: Moved from metadata-based (`localnpcs.is_active`) to database-based (`characters.is_active`)
  - Eliminates redundant `localnpcs` dependency that caused countless bugs
  - Single source of truth for NPC data and active status
  - `is_active` now stored directly in `characters` table (default: 1)
  - Frontend now calls `/api/chats/{chat_id}/migrate-npcs` when saving chats under new names
  - NPCs are copied with new filenames to ensure independence between branches

### Fixed
- **Input Button Overlap Bug**: Moved snapshot and send buttons from absolutely positioned inside the textarea to a horizontal flex layout alongside the textarea
  - Previously: Buttons overlaid text when typing, making them hard to reach
  - Now: Buttons float on the right side alongside the textarea with dedicated spacing
  - Text never covered by buttons at any input length

### Technical
- **New Database Function**: `db_update_npc_active()` - updates NPC active status in database
- **New Database Function**: `db_migrate_npcs()` - atomically copies NPCs between chats
- **New API Endpoint**: `POST /api/chats/{chat_id}/migrate-npcs` - migrates NPCs from another chat
- **Database Migration**: Added `is_active` column to `characters` table with one-time populating from metadata
- **Reduced Metadata Redundancy**: Removed `is_active` from `localnpcs` during NPC creation
- **Updated Helper Functions**: `db_get_characters()` now includes `is_active` from database

## [2.0.2] - 2026-02-13

### Added
- **Summary Word Limit**: Automatic LLM-based condensation when chat summary exceeds configurable word count (default: 1200 words)
  - Configurable via `config.yaml` → `context.summary_word_limit`
  - Triggers on every chat turn, uses existing `summarize_text()` function to reduce summary by 50-70%
  - Logs word count before/after: `[SUMMARIZE] Summary condensed: 1200 -> ~400 words`
  - Works in both performance mode and standard mode

***

## [2.0.1] - 2026-02-12

### Added
- **Snapshot Scene Caching**: Location and dress cached during summarization for faster snapshots with reduced LLM overhead
- **Sampling Parameters**: `top_p`, `top_k`, and repetition penalty now configurable via `config.yaml` for better control over LLM output quality
- **Configuration Options**: Extended `config.yaml` support for SD presets, inpainting parameters, and generation defaults
- **Legacy Character Capsule Generation**: Automatic capsule generation on startup for characters without multi-char summaries

### Changed
- **Turn-Based Summarization**: Summarization now triggers on turn intervals (default: turn 10) instead of percentage threshold, with 0.90 threshold as backstop only
- **History Window**: Reduced from 6 to 5 exchanges for tighter context

### Fixed
- **Sticky Window Bug**: Characters now correctly switch to SCENE CAST capsules after their 3-turn sticky window, preventing infinite full card injection
- **Capsule Field Visibility**: Fixed missing capsule field in character edit screen for legacy and newly created characters
- **Error Handling**: Replaced bare exception clauses with specific exception handling and logging for better debugging

***

## [2.0.0] - 2026-02-11

### Removed
- **Forking System**: Complete removal of built-in branching/forking functionality
  - Removed `/api/chats/fork` and all related fork API endpoints from `main.py` (~520 lines)
  - Removed fork transaction helpers and entity ID remapping logic
  - Removed fork isolation checks from save operations
  - Removed branch metadata from chat save path
  - Removed fork origin linkage system
  - Forked chats in existing databases remain as plain independent chats with no origin linkage
  - Users now use standard save workflows ("Save As" with different names) instead of hidden branching model

- **Relationship System**: Complete removal of semantic relationship tracking
  - Removed `app/relationship_tracker.py` module (502 lines)
  - Removed `RELATIONSHIP_TEMPLATES` dictionary from `main.py` (~42 lines)
  - Removed `get_relationship_context()` function from `main.py` (~218 lines)
  - Removed `analyze_and_update_relationships()` function from `main.py` (~113 lines)
  - Removed relationship functions from `app/database.py` (~92 lines)
  - Removed `db_get_relationship_state()` function
  - Removed `db_get_all_relationship_states()` function
  - Removed `db_update_relationship_state()` function
  - Removed `relationship_states` table from database schema
  - Removed `entities` table from database schema
  - Removed relationship context injection from prompt assembly
  - Removed relationship analysis trigger from `/api/chat` endpoint
  - Removed `adaptive_tracker` initialization and cleanup
  - Eliminated O(N²) per-request overhead from embedding computation
  - `NEURALRP_FEATURES_SEMANTIC_SCORING_ENABLED` config option no longer needed

- **Snapshot Variation Mode**: Complete removal of variation mode functionality
  - Removed `extract_character_scene_json_variation()` function from `app/snapshot_analyzer.py` (~143 lines)
  - Removed `analyze_scene_variation()` function from `app/snapshot_analyzer.py` (~91 lines)
  - Removed `/api/chat/{chat_id}/snapshot/regenerate` endpoint from `main.py` (~162 lines)
  - Removed `SnapshotRegenerateRequest` class from `main.py` (~5 lines)
  - Updated frontend `regenerateSnapshot()` to call standard `/api/chat/snapshot` endpoint
  - Added `chat_id` parameter to regenerate request (was missing)
  - Added `mode` parameter to regenerate request (for consistency with standard snapshot)
  - Changed button title from "Regenerate with variation mode" to "Regenerate"
  - Fixed all `kobold_url` config access bugs (6 locations total across codebase)

### Changed
- **Snapshot Regenerate Behavior**: Regenerate button now generates standard snapshots (same LLM prompt with examples)
  - Results: Natural SD seed variations instead of prompt variations
  - User intent: "Regenerate" -> Generate same scene with different random seed
- **Simplified Context Assembly**: Removed relationship injection and character reinforcement intervals based on relationship state
- **Database Schema**: Removed `relationship_states` and `entities` tables
- **Metadata Structure**: Removed relationship-related metadata fields (no longer relevant)
- **Documentation**: Removed all relationship tracker sections from `docs/TECHNICAL.md` (~780 lines)

### Fixed
- **Snapshot Feature**: Fixed configuration access for `kobold_url` causing LLM unavailability
  - Changed from `config['kobold_url']` (incorrect) to `config.get('kobold', {}).get('url')` (correct)
  - Applied to: `app/snapshot_analyzer.py` (2 locations in LLM HTTP requests), `main.py` (1 location in character counting, 1 location in variation endpoint, 2 total in regenerate endpoint)

### Technical
- **Code Reduction**: Removed ~2,173 lines of complex, crosscutting code
  - Forking system: ~520 lines (endpoints, transactions, entity remapping)
  - `app/relationship_tracker.py`: 502 lines (deleted)
  - `main.py`: ~1,000 lines (functions, templates, imports, calls)
  - `app/database.py`: ~92 lines (functions, table creation)
  - `app/database_setup.py`: ~60 lines (functions, indexes)
  - `docs/TECHNICAL.md`: ~780 lines (sections removed)
- **Performance**: Eliminated blocking synchronous embedding operations during chat requests (1+ second overhead removed)
- **Maintainability**: Removed maintenance burden of forking and relationship tracking code
- **Simplicity**: Future features no longer need to consider branch or relationship state

### Migration Notes
- **Forking Data**: Existing forked chats remain as plain independent chats with no origin linkage; no special migration required
- **No Automatic Migration**: Relationship and forking data are not preserved after upgrade to v2.0.0
- **Database Migration**: Schema version 6 → 7 drops `relationship_states` and `entities` tables
- **Existing Chats**: Continue to work normally, but relationship context and branch isolation are no longer tracked
- **No Data Migration Required (Snapshot)**: All existing snapshots work correctly with standard endpoint
- **Existing Regenerates**: Snapshots with `mode: "variation"` remain in history but future regenerates use `mode: "standard"` (or omitted)

***

## [1.12.0] - 2026-02-09

### Changed
- **NPC Unification to Characters Table**: NPCs are now stored in the unified `characters` table alongside global characters, eliminating the separate `chat_npcs` table. This simplifies code, prevents sync issues, and provides a single source of truth for all character data. Both global characters and NPCs now use the same database layer and helper functions.
- **Filename-Based Entity IDs**: All entities (characters and NPCs) now use filenames as entity IDs (e.g., `alice.json`, `npc_alice_123.json`) instead of the old separate entity_id system. This unifies relationship tracking, visual canon assignment, and other entity-based operations.
- **Automatic NPC Migration**: On first startup after v1.12.0 upgrade, existing NPCs are automatically migrated from `chat_npcs` to the `characters` table with new filename-based entity IDs. The old table is preserved as `chat_npcs_backup` for rollback capability.

### Added
- **Database Schema Version 6**: Added `chat_id` column to `characters` table to distinguish between global characters (`chat_id IS NULL`) and local NPCs (`chat_id = 'chat_id'`).
- **Unified Character Loading**: `db_get_characters(chat_id)` function now retrieves both global and NPC characters from the same table, with `is_npc` flag indicating type.
- **Unified NPC CRUD Functions**: All NPC operations (create, update, delete, promote) now use the `characters` table, with same API patterns as global characters.
- **Visual Canon for NPCs**: Visual canon assignment functions (`db_assign_visual_canon_to_npc`, `db_clear_visual_canon_from_npc`, `db_get_npc_visual_canon`) now use the unified `characters` table, enabling visual canon for NPCs.

### Technical
- **Database Migration Handler**: Schema version 5 → 6 migration handles automatic NPC migration with filename generation and entity ID updates.
- **Helper Function Updates**: `get_entity_id()` now consistently returns filename for all entity types. `load_character_profiles()` updated to load NPCs from unified table.
- **API Endpoint Unification**: All NPC endpoints now use filename-based entity IDs (`/api/chats/{chat_id}/npcs/{filename}`) instead of old entity_id format.
- **Metadata Simplification**: NPC data stored in `characters` table, with only `localnpcs` in chat metadata for active status tracking.

### Fixed
- **Danbooru Tag Generation**: Fixed visual canon assignment for NPCs by updating all related functions to use the unified `characters` table instead of the deprecated `chat_npcs` table.

### Migration Notes
- **Automatic Migration**: Existing NPCs are automatically migrated on startup with no user intervention required.
- **Filename Generation**: NPCs receive filenames in format `npc_{sanitized_name}_{timestamp}.json` to ensure uniqueness.
- **Entity ID Updates**: The `entities` table is updated to use filenames as entity IDs, preserving relationship data.
- **Chat Metadata Updates**: NPC references in chat metadata are updated from old entity_id format to new filename format.
- **Backup Table**: Original `chat_npcs` table preserved as `chat_npcs_backup` for rollback capability.

***

## [1.11.1] - 2026-02-07

### Fixed
- **NPC URL Encoding Issues**: Fixed two related problems with NPCs containing spaces or special characters in names:
  - NPC save operations now work correctly (404 errors when editing NPCs)
  - SCENE UPDATE blocks now display NPC names correctly instead of "None has entered the scene"
  - URL decoding added to NPC endpoints and cast change detection to handle encoded entity IDs consistently
- **Auto Mode Classification**: Replaced missing `/api/classify-mode` endpoint with keyword-based heuristic classification
  - Character names in messages now trigger focus mode (case-insensitive matching)
  - Pattern matching for "talk to [name]" or "[name], please..." style addresses
  - Eliminates 404 errors and LLM overhead during auto mode

***

## [1.11.0] - 2026-02-06

### Changed
- **Hybrid SCENE CAST + Sticky Full Cards System**: Characters now receive full cards on first appearance, when returning after long absence, or during their first 3 turns in chat. This balances strong early reinforcement with long-term character consistency while reducing token overhead. This represents a philosophical shift from "dialog-centric" to "character-centric + summaries" that respects how LLMs actually pay attention to information. Regular character capsules every turn, combined with strategic full-card injections at critical moments, yield better character consistency than simply adding more dialog to the context.
- **History Window**: Condensed dialog with it hits the context threshold to the 6 latest turns through auto-summary, rather than the oldest 10 turns in chat window.  This RP-optimized window prevents attention decay and character-merging while keeping conversations focused on recent context.
- **Summaries Panel with Autosummarize**: New "Autosummarize" button lets you condense selected text from your summary, helping manage bloated summaries without losing key details.

### Added
- **Scene Capsule Summarization**: When conversations get long, the system now summarizes by scene—grouping messages based on which characters were active—rather than flat chronological blocks. This preserves plot flow and character dynamics better.
- **SCENE UPDATE Blocks**: When characters enter or leave your scene, you'll see a clear notification showing who arrived or departed, helping the AI adjust portrayal to the current cast. Upon entering, an aggressive auto-summarization occurs, allowing the new character the opportunity to define themselves, rather than merge with existing characters in the context window.
- **Returning Character Reinforcement**: If a character hasn't appeared for 20+ messages and then returns, they get their full card injected again to re-establish their voice.
 - **Auto-generated Character Capsules**: Imported characters automatically generate condensed personality capsules for multi-character chats, saving time and ensuring consistent voices.
 
 ### Technical
 - Database: Schema version 5 migration ensures `chat_npcs` table includes `visual_canon_id` and `visual_canon_tags` columns in base schema (migration-safe, idempotent)
 
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
