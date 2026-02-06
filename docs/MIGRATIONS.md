# Database Migrations

This document tracks all database schema changes for NeuralRP.

## Schema Versioning

 - **Current Schema Version**: 5
- **Schema Version Table**: `schema_version` (id=1, version, updated_at)
- **Auto-Migration**: Runs automatically on app startup via `app/database_setup.py`

---

## Migration History
 
### Schema Version 5 (v1.11.0) - 2026-02-06
 
**Migration ID**: `4 → 5`

**Description**: Ensure Visual Canon Column Consistency

**Problem Fixed**: Base schema for `chat_npcs` table was missing `visual_canon_id` and `visual_canon_tags` columns. These were only added via migration (Schema Version 2), creating a discrepancy between fresh database creation and upgraded databases.

**Changes**:
- Added `visual_canon_id` and `visual_canon_tags` columns to `_create_chat_npcs_table()` base schema
- Migration 4 → 5 ensures these columns exist in existing databases (idempotent via `_add_column_if_not_exists()`)

**SQL Changes**:
```sql
-- Ensured in base schema (now in CREATE TABLE IF NOT EXISTS)
visual_canon_id INTEGER,
visual_canon_tags TEXT

-- Migration ensures existing databases have columns
ALTER TABLE chat_npcs ADD COLUMN visual_canon_id IF NOT EXISTS;
ALTER TABLE chat_npcs ADD COLUMN visual_canon_tags IF NOT EXISTS;
```

**Data Preservation**: No data loss (columns are nullable and already existed in upgraded databases)

**Why This Change**:
- Fresh databases created with migration-disabled environments would be missing visual canon support
- Ensures consistency between base schema and migration-added columns
- Fixes documentation mismatch in MIGRATIONS.md Core Tables section

**Impact on Code**:
- No code changes required (columns already expected and handled)
- `db_get_character()` and `db_save_character()` already handle missing columns gracefully
- Danbooru character assignment endpoints work correctly in all scenarios

**Rollback**: Not applicable (schema correction, idempotent)

---

### Schema Version 4 (v1.10.1) - 2026-02-03

**Migration ID**: `3 → 4`

**Description**: Remove Learning System

**Changes**:
- Dropped `danbooru_tag_favorites` table (learning system no longer used)
- Simplified favorites system to use `sd_favorites.tags` JSON for tag counting

**SQL Changes**:
```sql
DROP TABLE IF EXISTS danbooru_tag_favorites;
```

**Data Preservation**:
- No data loss (learning counters were auxiliary metadata)
- `sd_favorites` table fully preserved with all favorites
- Tag counting now calculated from `sd_favorites.tags` JSON on-demand

**Why This Change**:
- Learning system complexity didn't provide proportional value
- Favorites system simpler and more maintainable without learning counters
- Tag frequency can be computed from existing favorites data

**Impact on Code**:
- Removed: `db_increment_favorite_tag()`, `db_get_favorite_tag_frequency()`, `db_detect_danbooru_tags()`
- Rewritten: `db_get_popular_favorite_tags()` now queries `sd_favorites.tags` JSON
- Simplified: `add_snapshot_favorite()` and `add_manual_favorite()` no longer increment tag counts
- Removed: `app/snapshot_learning_config.py` (learning parameters)

**Rollback**: Not applicable (learning counters no longer used)

---

### Schema Version 3 (v1.10.1) - 2026-02-03

**Migration ID**: `2 → 3`

**Description**: NPC Key Consolidation

**Problem Fixed**: Split-key issue caused by old `migrate_npcs_to_db.py` script that renamed `localnpcs` → `local_npcs`, creating inconsistent state across the codebase.

**Changes**:
- Consolidates split NPC metadata keys (`local_npcs` + `localnpcs`) into single `localnpcs` key
- Runs automatically on first v1.10.1 startup for users with schema version < 3

**Migration Function**: `_consolidate_npc_metadata_keys()` in `app/database_setup.py`

**Data Preservation**:
- All NPC data preserved via merge operation
- `localnpcs` data takes precedence (newer data)
- `local_npcs` data merged in as fallback

**SQL Changes**:
- No table/column changes (data-only migration)
- Updates `chats.metadata` JSON field for all chats

**Example Migration Logic**:
```python
# Before (split keys)
chat.metadata = {
    "localnpcs": { "npc_123": {...} },  # newer
    "local_npcs": { "npc_456": {...} }   # older
}

# After (consolidated)
chat.metadata = {
    "localnpcs": { "npc_123": {...}, "npc_456": {...} }
}
```

**Impact on Code**:
- All NPC endpoints now use single `localnpcs` key
- Fixed: `update_npc`, `delete_npc`, `promote_npc`, `toggle_npc_active`
- NPC creation/edit/deletion/promotion now work correctly

**Rollback**: Not applicable (one-time consolidation)

---

### Schema Version 2 (v1.10.0) - 2026-02-01

**Migration ID**: `1 → 2`

**Description**: Danbooru Character Casting Support

**Changes**:
- Added `visual_canon_id` column to `characters` table
- Added `visual_canon_tags` column to `characters` table
- Added `visual_canon_id` column to `chat_npcs` table
- Added `visual_canon_tags` column to `chat_npcs` table
- Created `danbooru_characters` table (IF NOT EXISTS)
- Created `vec_danbooru_characters` virtual table (IF NOT EXISTS, requires sqlite-vec)

**SQL Changes**:
```sql
-- Columns added to characters table
ALTER TABLE characters ADD COLUMN visual_canon_id INTEGER;
ALTER TABLE characters ADD COLUMN visual_canon_tags TEXT;

-- Columns added to chat_npcs table
ALTER TABLE chat_npcs ADD COLUMN visual_canon_id INTEGER;
ALTER TABLE chat_npcs ADD COLUMN visual_canon_tags TEXT;

-- New tables created via IF NOT EXISTS
CREATE TABLE danbooru_characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female', 'other', 'unknown')),
    core_tags TEXT NOT NULL,
    all_tags TEXT NOT NULL,
    image_link TEXT,
    source_id TEXT UNIQUE,
    created_at INTEGER
);

CREATE VIRTUAL TABLE vec_danbooru_characters USING vec0(
    embedding float[768]
);
```

**Data Preservation**: No data loss (new columns, nullable)

**New Features Enabled**:
- Danbooru Tag Generator: Semantic search across 1394 Danbooru characters
- Visual canon assignment to characters and NPCs
- Character editor: "Generate Danbooru Character (Visual Canon)" hyperlink

**Impact on Code**:
- `db_get_character()` now loads `visual_canon_id` and `visual_canon_tags`
- `db_save_character()` saves visual canon data
- `db_search_danbooru_characters_semantically()` for semantic matching
- Character/NPC endpoints support visual canon assignment

**Rollback**: Possible but not recommended (visual canon assignments would be lost)

---

### Schema Version 1 (v1.9.0) - 2026-01-31

**Migration ID**: `0 → 1`

**Description**: Snapshot and Favorites System

**Changes**:
- Added `snapshot_data` column to `messages` table
- Created `danbooru_tags` table (IF NOT EXISTS)
- Created `vec_danbooru_tags` virtual table (IF NOT EXISTS, requires sqlite-vec)
- Created `sd_favorites` table (IF NOT EXISTS)
- Created `danbooru_tag_favorites` table (IF NOT EXISTS) - **DROPPED in Schema Version 4**
- Created `change_log` table (IF NOT EXISTS)
- Created `performance_metrics` table (IF NOT EXISTS)
- Created `image_metadata` table (IF NOT EXISTS)
- Created FTS5 virtual table `messages_fts` (IF NOT EXISTS)
- Created virtual table `vec_world_entries` (IF NOT EXISTS, requires sqlite-vec)

**SQL Changes**:
```sql
-- Column added to messages table
ALTER TABLE messages ADD COLUMN snapshot_data TEXT;

-- New tables
CREATE TABLE danbooru_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_text TEXT UNIQUE NOT NULL,
    block_num INTEGER NOT NULL,
    frequency INTEGER DEFAULT 0,
    created_at INTEGER
);

CREATE VIRTUAL TABLE vec_danbooru_tags USING vec0(
    embedding float[768]
);

CREATE TABLE sd_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    image_filename TEXT UNIQUE NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    scene_type TEXT,
    setting TEXT,
    mood TEXT,
    character_ref TEXT,
    tags TEXT,
    steps INTEGER,
    cfg_scale REAL,
    width INTEGER,
    height INTEGER,
    source_type TEXT DEFAULT 'snapshot',
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    note TEXT
);

CREATE TABLE danbooru_tag_favorites (
    tag_text TEXT PRIMARY KEY,
    favorite_count INTEGER DEFAULT 0,
    last_used INTEGER DEFAULT (strftime('%s', 'now'))
);
-- NOTE: Dropped in Schema Version 4 (v1.10.1)

CREATE TABLE change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_data TEXT,
    new_data TEXT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now')),
    undo_state TEXT DEFAULT 'active'
);

CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_type TEXT,
    duration REAL,
    context_tokens INTEGER,
    timestamp INTEGER
);

CREATE TABLE image_metadata (
    filename TEXT PRIMARY KEY,
    prompt TEXT,
    negative_prompt TEXT,
    steps INTEGER,
    cfg_scale REAL,
    width INTEGER,
    height INTEGER,
    sampler TEXT,
    scheduler TEXT,
    timestamp INTEGER
);

-- Virtual tables
CREATE VIRTUAL TABLE messages_fts USING fts5(
    message_id UNINDEXED,
    chat_id UNINDEXED,
    content,
    speaker,
    role,
    timestamp UNINDEXED,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE vec_world_entries USING vec0(
    entry_id INTEGER PRIMARY KEY,
    world_name TEXT,
    entry_uid TEXT,
    embedding FLOAT[768]
);
```

**Data Preservation**: No data loss (new tables and columns)

**New Features Enabled**:
- Snapshot feature: Generate SD images from chat context
- Favorites system: Track favorite images across snapshot and manual mode
- Tag preference tracking: Learn from favorite images (simplified in Schema Version 4)
- Change logging: Audit trail for undo/redo
- Performance metrics: Track operation performance
- Image metadata: Store SD generation parameters

**Impact on Code**:
- `db_add_snapshot_favorite()` for favorites
- `db_search_danbooru_tags_semantically()` for semantic tag search
- `db_record_change()` for change logging
- `db_add_performance_metric()` for performance tracking
- `db_save_image_metadata()` for image parameter storage

**Rollback**: Possible but not recommended (favorites, change history, performance data would be lost)

---

## Core Tables (Baseline Schema)

The following tables exist from initial schema setup (before version 1):

### Characters
```sql
CREATE TABLE characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    filename TEXT UNIQUE,
    data TEXT,
    danbooru_tag TEXT,
    created_at INTEGER,
    updated_at INTEGER
);
```

### Chats
```sql
CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    branch_name TEXT,
    summary TEXT,
    metadata TEXT,
    created_at INTEGER,
    autosaved BOOLEAN DEFAULT 1
);
```

### Messages
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    role TEXT,
    content TEXT,
    speaker TEXT,
    image_url TEXT,
    timestamp INTEGER,
    summarized BOOLEAN DEFAULT 0,
    snapshot_data TEXT,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);
```

### Worlds
```sql
CREATE TABLE worlds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
```

### World Entries
```sql
CREATE TABLE world_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER,
    uid TEXT,
    content TEXT,
    keys TEXT,
    secondary_keys TEXT,
    is_canon_law BOOLEAN,
    probability INTEGER,
    use_probability BOOLEAN,
    depth INTEGER,
    sort_order INTEGER,
    embedding_hash TEXT,
    last_embedded_at INTEGER,
    metadata TEXT,
    updated_at INTEGER,
    FOREIGN KEY (world_id) REFERENCES worlds(id) ON DELETE CASCADE
);
```

### Character Tags (Junction Table)
```sql
CREATE TABLE character_tags (
    char_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (char_id, tag),
    FOREIGN KEY (char_id) REFERENCES characters(filename) ON DELETE CASCADE
);
```

### World Tags (Junction Table)
```sql
CREATE TABLE world_tags (
    world_name TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (world_name, tag),
    FOREIGN KEY (world_name) REFERENCES worlds(name) ON DELETE CASCADE
);
```

### Chat NPCs
```sql
CREATE TABLE chat_npcs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_from_text TEXT,
    created_at INTEGER,
    last_used_turn INTEGER,
    appearance_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1,
    promoted BOOLEAN DEFAULT 0,
    promoted_at INTEGER,
    global_filename TEXT,
    visual_canon_id INTEGER,
    visual_canon_tags TEXT,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    UNIQUE(chat_id, entity_id),
    UNIQUE(chat_id, name)
);
```

### Relationship States
```sql
CREATE TABLE relationship_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    character_from TEXT NOT NULL,
    character_to TEXT NOT NULL,
    trust INTEGER DEFAULT 50,
    emotional_bond INTEGER DEFAULT 50,
    conflict INTEGER DEFAULT 50,
    power_dynamic INTEGER DEFAULT 50,
    fear_anxiety INTEGER DEFAULT 50,
    last_updated INTEGER,
    last_analyzed_message_id INTEGER,
    interaction_count INTEGER DEFAULT 0,
    history TEXT,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    UNIQUE(chat_id, character_from, character_to)
);
```

### Entities
```sql
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);
```

---

## Metadata Fields in `chats.metadata`

The `chats.metadata` JSON field stores chat-specific data:

### v1.8.0+
- `activeCharacters`: Array of active character filenames
- `activeWI`: Array of active world info UIDs
- `settings`: Chat-specific settings (reinforce_freq, world_info_reinforce_freq, etc.)

### v1.8.2+
- `localnpcs`: Object mapping NPC entity IDs to NPC data (consolidated key)
- `characterCapsules`: Object mapping character filenames to capsule strings
- `characterFirstTurns`: Object mapping character filenames to turn numbers

### v1.9.0+
- `snapshot_settings`: Object with snapshot-specific settings
  - `includeUser`: Boolean (whether user appears in snapshots)
  - `userGender`: String (user's gender for counting: "female", "male", "other")

### Deprecated Keys (v1.10.1 migration removes these)
- `local_npcs`: Split-key version, consolidated into `localnpcs`

---

## Migration Best Practices

### For Developers

1. **Always increment `SCHEMA_VERSION`** when making schema changes
2. **Use `_add_column_if_not_exists()`** for adding columns (safe for re-runs)
3. **Add migration logic to `_apply_migrations()`** for version transitions
4. **Document data preservation** in this MIGRATIONS.md file
5. **Test migrations on existing databases** before releasing

### Migration Checklist

- [ ] Does this migration preserve all existing data?
- [ ] Is the migration idempotent (safe to run multiple times)?
- [ ] Are rollback steps documented?
- [ ] Is MIGRATIONS.md updated?
- [ ] Are affected code paths updated?
- [ ] Is the migration tested on a fresh database?
- [ ] Is the migration tested on an upgraded database?

### Adding a New Migration

1. Increment `SCHEMA_VERSION` in `app/database_setup.py`
2. Add migration logic in `_apply_migrations()` function
3. Create new tables/columns using appropriate functions
4. Test migration manually
5. Update MIGRATIONS.md with details
6. Update CHANGELOG.md with migration notes

---

## Troubleshooting

### Migration Fails to Run

**Symptom**: Database not upgraded after code update

**Solutions**:
1. Check `schema_version` table: `SELECT * FROM schema_version;`
2. Verify `SCHEMA_VERSION` constant in `app/database_setup.py`
3. Check logs for migration errors
4. Manually run migration: `python app/database_setup.py`

### Data Loss After Migration

**Symptom**: Missing data after upgrade

**Solutions**:
1. Check backup before migration (database file: `app/data/neuralrp.db`)
2. Review migration logic for data preservation
3. Verify SQL operations (ALTER TABLE, UPDATE, etc.)
4. Check for cascade deletes that might have removed data

### Stuck at Old Schema Version

**Symptom**: App reports old version number

**Solutions**:
1. Verify `SCHEMA_VERSION` constant is incremented
2. Check migration condition: `if from_version < X:`
3. Ensure migration function actually runs
4. Update schema_version manually (last resort): `UPDATE schema_version SET version = X WHERE id = 1;`

---

## Database File Locations

- **Default Path**: `app/data/neuralrp.db`
- **Backup Path**: `app/data/neuralrp.db.backup` (manual backups only)
- **Test Path**: Specify via command line: `python app/database_setup.py /path/to/test.db`

---

## Related Documentation

- **Database Setup**: `app/database_setup.py`
- **Database Functions**: `app/database.py`
- **Changelog**: `CHANGELOG.md`
- **Technical Documentation**: `docs/TECHNICAL.md`
- **Development Context**: `AGENTS.md`

---

## Version History

| Schema Version | Release Version | Date | Description |
|---------------|----------------|------|-------------|
| 5 | v1.11.0 | 2026-02-06 | Ensure Visual Canon Column Consistency |
| 4 | v1.10.1 | 2026-02-03 | Remove Learning System |
| 3 | v1.10.1 | 2026-02-03 | NPC Key Consolidation |
| 2 | v1.10.0 | 2026-02-01 | Danbooru Character Casting |
| 1 | v1.9.0 | 2026-01-31 | Snapshot and Favorites System |
| 0 | v1.8.0 and earlier | Pre-2026-01-31 | Baseline schema |
