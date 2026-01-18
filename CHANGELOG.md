# Changelog

All notable changes to NeuralRP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---
## [1.5.2] - 2026-01-18

### Added
- **Fixed Autosave**: Every turn the world saves to SQLite


## [1.5.1] - 2026-01-13

### Added
- **Change Logging System**: Complete audit trail for undo/redo support
  - Tracks character, world info, and chat create/update/delete operations
  - Stores JSON snapshots (before/after) for each change
  - REST API for querying change history (`/api/changes`)
  - Automatic 30-day rolling cleanup to maintain performance
  - Statistics endpoint for monitoring log size
- **Database Health Check**: Automatic startup validation prevents silent corruption
  - Runs SQLite's `PRAGMA integrity_check` on every launch (<10ms overhead)
  - Verifies all 5 core tables exist (characters, worlds, world_entries, chats, messages)
  - Prints warning if corruption detected, advises running `migrate_to_sqlite.py`
  - Graceful degradation: App continues even if check fails (allows data export/manual repair)
- **Automatic Maintenance**: Daily cleanup of old change logs and performance metrics

### Fixed
- **Critical SQL bug**: Fixed `db_search_similar_embeddings()` WHERE clause using column alias (SQLite incompatible)
- **Memory leak**: Embeddings now properly deleted when World Info entries are removed
- **Data integrity**: Added 768-dimension validation to prevent model mismatch crashes
- **Delete path**: Individual entry deletion now syncs embeddings to database

### Changed
- **SemanticSearchEngine**: Now uses sqlite-vec for disk-based embeddings instead of in-memory cache
- **Startup behavior**: Lazy loading of embeddings (sub-second startup)
- **Memory profile**: ~50MB idle (down from 300MB with pickle)

### Technical
- Added `change_log` table with entity_type/entity_id/operation tracking
- Added `db_delete_entry_embedding()` for granular vector cleanup
- Added `EXPECTED_EMBEDDING_DIMENSIONS` constant (768 for all-mpnet-base-v2)
- Periodic background tasks for log maintenance (24h cycle)
- Database integrity check runs on every startup

### Performance
- Startup time: <1 second (down from 2-5 seconds)
- Semantic search: 20-50ms for 100 entries
- Memory footprint: ~50MB idle
- Scales to 10,000+ World Info entries

### Notes
Completes the sqlite-vec migration and adds production-ready change tracking infrastructure. Undo/Redo UI will be exposed in v1.6 with Living World Engine.

---

## [1.5.0] - 2026-01-13

### Major: Complete Architecture Migration

This release migrates NeuralRP from JSON file storage to SQLite, providing ACID guarantees, better scalability, and significantly reduced memory usage.

### Added

- **SQLite Database**: Centralized storage for characters, chats, world info, and messages in `app/data/neuralrp.db`
- **sqlite-vec Integration**: Vector similarity search using disk-based embeddings with SIMD acceleration (AVX2/SSE2)
- **Automatic Migration**: `migrate_to_sqlite.py` script for one-time JSON → SQLite conversion
- **Database Integrity Checks**: Startup validation with automatic repair capabilities
- **Foreign Key Constraints**: Referential integrity between chats, messages, and world info

### Changed

- **Vector Search Backend**: Replaced pickle + sklearn with sqlite-vec for semantic World Info search
- **Semantic Search Architecture**: sqlite-vec now serves as primary search method with automatic numpy fallback
  - First attempts SIMD-accelerated similarity search via `db_search_similar_embeddings()`
  - Falls back to in-memory numpy calculations if sqlite-vec unavailable
  - Embeddings persist across app restarts (no recomputation needed)
  - Console output indicates which method was used for transparency
- **Embedding Storage**: Now stored in `vec_world_entries` virtual table (768-dimensional) instead of RAM
- **Performance Statistics**: Rolling median tracking now persists across restarts
- **Branching Operations**: Now O(1) row inserts instead of O(n) file copies
- **Context Assembly**: No longer loads full chat history into memory

### Fixed

- **Image Generation**: `generate_image()` now reads character Danbooru tags from database instead of JSON files
- **Embedding Cache Invalidation**: sqlite-vec embeddings properly update when World Info entries are modified
- **Search Cache Consistency**: In-memory search caches clear when world info is saved

### Removed

- **Pickle Dependencies**: Eliminated pickle-based embedding storage (security improvement)
- **Unbounded Cache Growth**: LRU eviction prevents memory leaks during long sessions

### Performance

- **Idle RAM Usage**: ~50MB (down from ~300MB with pickle cache)
- **Startup Time**: Reduced (no pickle loading delay)
- **World Info Scalability**: 10,000+ entries without performance degradation
- **Zero RAM Overhead**: Vector search operates on-disk with lazy loading

### Migration Notes

- Migration is one-way: After running `migrate_to_sqlite.py`, the database is the source of truth
- JSON files continue to be generated for SillyTavern export/compatibility
- All existing chats, characters, and world info are preserved
- Embeddings are automatically regenerated and indexed in sqlite-vec

---

## [1.4.0] - 2025-12-15

### Added

- **Image Inpainting**: Full inpainting support via Stable Diffusion img2img API
  - Configurable masks, denoising strength, mask blur
  - Saved as `inpaint_{timestamp}.png` with full metadata
- **Image Metadata System**: Automatic storage of all generation parameters in `app/images/image_metadata.json`
  - Prompt, negative prompt, steps, CFG scale, dimensions, seed, timestamp
  - Enables reproducibility of previous generations
- **World Info Reinforcement Configuration**: Configurable canon law reinforcement frequency
  - Default: Every 3 turns (configurable 1-100)
  - API endpoints: GET/POST `/api/world-info/reinforcement/config`

### Changed

- **Semantic Search Improvements**:
  - Initial turn detection: Uses latest user message only with 0.35 threshold
  - Subsequent turns: Last 5 messages with 0.45 threshold for broader context
  - Keyword priority sorting: Keyword matches rank higher than semantic-only
- **Deduplication**: Comprehensive handling of plurals, apostrophes, possessives, punctuation variations
- **Generic Key Filtering**: Excludes structural keys ("the", "and", "room", "city", "type", etc.)

### Fixed

- **Semantic Search Resource Management**: 
  - Periodic cleanup task (every 5 minutes)
  - Automatic GPU memory cleanup
  - Embeddings cache limited to 5 most recent world info versions

---

## [1.3.0] - 2025-11-20

### Added

- **In-Card Editing**: Full character and world info editing interface
  - AI-assisted content generation for specific fields
  - Manual editing of all fields (personality, body, scenario, tags, keys, content)
  - Save/cancel buttons positioned for mobile-friendly UX
- **Semantic World Info Retrieval**: Intelligent, context-aware lore retrieval
  - Uses `all-mpnet-base-v2` sentence transformers
  - Cosine similarity matching with configurable thresholds (default: 0.25)
  - Automatic GPU detection and fallback to CPU
- **LRU Cache System**: Memory-safe world info caching
  - Default 1000-entry limit (configurable)
  - Automatic eviction prevents memory leaks
  - Real-time monitoring UI showing usage percentage and memory estimate
- **Cache Management API**:
  - GET `/api/world-info/cache/stats` - Statistics
  - POST `/api/world-info/cache/clear` - Manual clearing
  - POST `/api/world-info/cache/configure` - Size configuration

### Changed

- **Character Toggle**: Reliable add/remove from chat functionality
- **Responsive Design**: Improved layout across devices

---

## [1.2.0] - 2025-10-15

### Added

- **Automatic Performance Mode**: Smart GPU resource management for LLM + SD
  - Queues heavy operations while allowing light tasks to proceed
  - Rolling median tracking for contention detection
  - Master toggle to enable/disable entire system
- **SD Context-Aware Presets**: Automatic quality adjustment based on story length
  - Normal (512×512, 20 steps) for 0-7999 tokens
  - Light (384×384, 15 steps) for 8000-11999 tokens
  - Emergency (256×256, 10 steps) for 12000+ tokens
- **Smart Hint Engine**: Context-aware optimization suggestions
  - Contention hints when SD timing exceeds 3× median
  - Quality hints when emergency preset is active
  - Dismissible, non-repetitive notifications
- **Real-time Status Indicators**: Idle/running/queued badges for text and image operations

### Changed

- **Adaptive Connection Monitoring**: Intelligent health checking system
  - Reduces monitoring from 12 to 2-4 checks/minute during stable connections
  - Increases frequency during connection failures
  - 60-67% reduction in network overhead
- **Background Tab Optimization**: Automatic pause when browser tab not visible
  - Uses Page Visibility API
  - Zero resource usage when tab hidden
- **Connection Quality Tracking**: Smart intervals based on stability
  - Stable (5+ min): 60s intervals
  - Initial failure: 10s intervals
  - Persistent failure (3+): 5s intervals

---

## [1.1.0] - 2025-09-10

### Added

- **Branching System**: Fork from any message to create alternate timelines
  - Independent chat files per branch
  - Origin metadata (chat, message, timestamp)
  - Branch management UI for navigation
  - Rename/delete branch functionality
- **Dual-Source Card Generation**: Create cards from chat OR manual text input
  - Source mode toggle in Gen Card/Gen World tabs
  - Manual input textarea for plain-text descriptions
- **Live Editing Before Save**: AI-generated content in editable textboxes
  - Tweak, partially rewrite, or completely replace before saving
- **Efficient World Info**:
  - Cached world info structures (avoid reprocessing per request)
  - One-time lowercase preprocessing for faster keyword matching
  - Configurable entry cap (default: 10 regular entries)
  - Canon Law entries always included, never capped
  - Probability weighting (`useProbability` + `probability` fields)
  - Global "Enable World Info" toggle in Settings

### Changed

- **Performance**: Case-insensitive preprocessing gives ~10-20% improvement

---

## [1.0.0] - 2025-08-01

### Initial Release

- **KoboldCpp Integration**: Native and OpenAI-compatible API support
- **Stable Diffusion Integration**: AUTOMATIC1111 WebUI for image generation
- **SillyTavern Compatibility**: V2 JSON format for character cards and World Info
- **Chat Modes**:
  - Narrator: Third-person omniscient narration
  - Focus: First-person character voice
  - Auto: AI chooses speaker based on context
- **Multi-Character Support**: Capsule personas for distinct voices without prompt bloat
- **Automatic Summarization**: Triggered at 85% context (configurable threshold)
- **Canon Law System**: Mark immutable World Info entries
- **Danbooru Character Tags**: Per-character visual canon with `[CharacterName]` expansion
- **AI Card Generation**: Create character cards and world info from conversations
- **Token Counter**: Real-time context usage monitoring
- **Chat Persistence**: Save/load sessions with characters, world, and images

---

## Version Naming

- **Major** (1.x.0): Breaking changes or significant architecture shifts
- **Minor** (x.1.0): New features, backwards compatible
- **Patch** (x.x.1): Bug fixes and minor improvements

---

## Links

- [README](README.md) - Overview and quick start
- [Technical Documentation](docs/TECHNICAL.md) - Implementation details
- [GitHub Repository](https://github.com/neuralrp/neuralrp)
