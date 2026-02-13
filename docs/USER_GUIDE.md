# NeuralRP User Guide

Comprehensive guide to all NeuralRP features, settings, and capabilities. This document explains how buttons work and features fit together, without diving into technical implementation details. For implementation details, see [Technical Documentation](TECHNICAL.md).

---

## Table of Contents

1. [Understanding Interface](#understanding-the-interface)
2. [Chat Management](#chat-management)
3. [Character System](#character-system)
4. [World Info](#world-info)
5. [Image Generation](#image-generation)
6. [Advanced Features](#advanced-features)
7. [Tips and Best Practices](#tips-and-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Understanding the Interface

### Header Menu

The top navigation bar provides access to all major features:

**Chats 💬** - Manage your chat sessions
- Save current session to database
- Refresh chat list
- View all saved chats
- Hover messages for actions: Delete, Rename, Load

**Characters 👥** - Access your character library
- Create new characters (New Character button)
- Refresh to import new JSON files from `app/data/characters/`
- Tag filter bar for quick filtering
- Character list shows badges:
  - **Green (Global)**: Saved to library, usable across chats
  - **Gray (NPC)**: Chat-scoped, created mid-chat
  - **Gray (Inactive)**: Not currently active in chat
  - **Orange (Promoted)**: NPC promoted to global character
- Hover actions: Edit, Delete, Activate/Deactivate (NPCs only), Promote to Global (NPCs only)

**World Info 🌍** - Manage your world lore
- Add new world info entries
- Refresh to import new JSON files from `app/data/worldinfo/`
- Tag filter bar for quick filtering
- Expandable entries with badges:
  - **Red (Canon Law)**: Core rules, always injected + reinforced
  - **Green (Constant)**: Always included (not semantically triggered)
  - No badge: Regular entries, injected via semantic search when relevant
- Hover actions: Edit, Delete, Expand/Collapse

**Gen Card 🪪** - AI-powered character card generator
- Source Mode: Current Chat / Manual Input
- Auto-Generate All: Creates complete character card from conversation
- Individual Fields: Generate Personality, Physical Body, Genre & Tags, Scenario, First Message, Example Dialog separately
- Save Character Card: Exports to SillyTavern V2 JSON format

**Gen World 🌍** - AI-powered world lore extractor
- Source Mode: Current Chat / Manual Input
- Analyze All Lore: Extracts history, locations, creatures, factions
- Individual Sections: Generate each lore category separately
- Finalize & Update: Saves extracted lore to world card

**Search 🔍** - Search messages across all chats
- Search Bar: Enter query text
- Filters (expandable): Filter by Chat, Speaker, Date Range
- Results: Show matching messages with Jump to Message and Show Context

**Export 📤** - Export chats for training (active when chat is saved)
- Format: Alpaca / ShareGPT / ChatML
- Options: Include system prompts, Include world info (Canon Law only), Include NPC responses
- Export Training Data: Download JSON file optimized for Unsloth/fine-tuning

**Settings ⚙️** - Configure NeuralRP
- Neural Connection Section: Connection status with Test buttons for KoboldCpp and Stable Diffusion
- General Settings: System prompt, Your Name, Your Persona, Reinforce World Canon Every X Turns, Enable World Info, Max World Info Entries, Temperature, Max Resp Tokens, Max Context, Summarize at Turn
- Long-Term Memory: Read-only view of current chat summary, Clear Memory button
- Performance Optimization: Smart Performance Mode toggle
- World Info Cache: Cache status display with Refresh/Clear buttons and cache size limit setting
- API Configuration: Kobold URL and SD API URL with Update & Test buttons
- Data Recovery: View Change History (30-day retention of all changes)
- Status Monitor: View system health, database stats, storage info

### Main Chat Area

**Messages Panel**: Displays conversation history
- User messages appear on right
- AI responses (narrator/characters) appear on left
- Editable content: Click any AI-generated message to edit
- Hover actions show available options per message type

**Input Area**: Bottom text input field
- Type your message and press Enter to send
- Shift + Enter for new line
- Chat mode dropdown selector on left
- Action buttons on right (varies by context)

**Active Character Badges**: Center of header
- Shows currently active characters
- Click × to remove a character from scene
- Blue pill = active character
- Characters shown in every turn via SCENE CAST (see Character System section)

---

## Chat Management

### Starting Conversations

**Narrator Mode (Default)**
- When no characters selected, you're in "Narrator Mode" (blue indicator in header)
- The AI acts as a third-person narrator describing events
- Type your action/description in input field
- Press Enter or click blue paper plane icon to send

**Adding Characters to Chat**
1. Click **Characters (👥)** in the header
2. Click on a character card to activate it (blue border appears)
3. Active characters appear as pill badges in the center of the header
4. To remove a character, click the **×** on its pill badge

### Chat Modes

Use the dropdown menu in the bottom input area to control who responds:

**🤖 Auto** (Default)
- AI selects best responder (character or narrator)
- Ideal for multi-character conversations
- System decides based on context and scene

**🎭 Narrator**
- Force AI to respond as narrator
- Third-person story narration mode
- Useful for scene transitions or descriptions

**👤 [Character Name]**
- Force specific character to respond
- Useful when you want to direct conversation to a particular character
- Character must be active in the scene (shown in header badges)

### Saving and Autosaving

**Manual Save**
1. Click **Chats (💬)** in the header
2. Click **"Save Session"** (green button)
3. Chat is saved to database and exported to JSON (SillyTavern compatible)

**Autosave**
- NeuralRP automatically saves chats periodically
- Autosave happens after each message (ensures no data loss)
- Saved chats persist across application restarts
- Autosave indicator shows when last save occurred

### Understanding Summarization

NeuralRP automatically summarizes long conversations to maintain context efficiency while preserving message history.

**How Summarization Works:**
- **Automatic Trigger**: When chat reaches 80% context threshold (configurable in Settings)
- **Scene Capsules**: Condenses conversation into 200-token scene summaries
- **Cast-Change Trigger**: One capsule when a character leaves the scene
- **Threshold Trigger**: Multiple scene-based capsules when context exceeds 80%
- **Scene Boundaries**: At cast changes OR every 15 exchanges

**Soft Delete System**
- Old messages are archived (not deleted), preserving message history
- Archived messages remain searchable via the Search panel
- Message IDs preserved for continuity

**Viewing and Editing Summaries**
1. Click **📋 Summaries** in the header (teal bookmark icon)
2. View current chat summary in an editable textarea
3. Edit summary text directly (auto-saved instantly)

**Autosummarize Feature (v1.11.1+)**

Highlight specific sections of your summary and condense them using the LLM:

- **How to Use:**
  1. Highlight text in the summary textarea (minimum 50 characters)
  2. Click the **Autosummarize** button (wand icon)
  3. LLM generates a condensed version replacing your selection
  4. Shows reduction percentage (e.g., "Reduced by 60% (500 → 200 chars)")
  5. Auto-saves the updated summary after summarization

- **Use Cases:**
  - Condense overly verbose sections while preserving key points
  - Reduce token usage in the summary window
  - Clean up bloated summaries from long conversations
  - Focus on essential plot points

- **Requirements:**
  - Active chat session required
  - Minimum 50 characters of selected text
  - LLM backend must be available (check KoboldCpp status in Settings)

**Configuring Summarization:**
- **Summarize at Turn**: Default 10 - triggers summarization when conversation reaches this turn
- Lower values = earlier summarization = prevents LLM collapse but less context
- Higher values = later summarization = more context but risk of repetition loops
- A 90% token backstop catches verbose conversations that overflow before the turn trigger

**Benefits:**
- Enable indefinite conversations without hitting context limits
- Keep historical conversation searchable after summarization
- Autosummarize keeps summaries concise without losing key information

---

## Character System

### Character Panel

**Character Library**
- Shows all characters available to add to chats
- Tag filter bar: Filter by multiple tags (AND semantics - character must have ALL selected tags)
- Quick filter chips: Top 5 most-used tags surface automatically
- Autocomplete: Start typing tag → suggestions appear (prevents bloat)
- Automatic extraction: SillyTavern V2 card tags preserved on import

**Character Badges:**
- **Green (Global)**: Saved to library, usable across all chats
- **Gray (NPC)**: Chat-scoped, created mid-chat
- **Gray (Inactive)**: Character exists but not currently active in this chat
- **Orange (Promoted)**: NPC promoted to global character

### Character Cards

Character cards contain the following fields:

**Name**: Character's display name

**Description**: Physical appearance and personality traits
- Physical details: Hair color, eye color, body type, clothing
- Personality: General temperament and behavioral patterns
- Used for context injection and Danbooru tag generation

**Personality**: Behavioral guidelines and voice
- How the character speaks and reacts
- Behavioral rules and preferences
- Example phrases or speech patterns

**Scenario**: Context for character's role
- Where the character is encountered
- Character's current situation
- World-building context

**Example Dialogue**: Sample conversation lines
- Provides a voice fingerprint for the LLM
- Helps maintain consistent speech patterns
- Shows the character in action

**Gender** (v1.10.0+)
- Required for the Danbooru tag generator
- Options: Female, Male, Other
- Influences semantic tag matching and character counting in snapshots

**Danbooru Tags** (Optional)
- Visual canon for image generation
- Up to 20 tags per character
- Generated via the Danbooru Tag Generator or manually entered
- Ensures consistent character appearance across all image generations

### Creating and Editing Characters

**Manual Creation**
1. Click **Characters (👥)** in the header
2. Click the **"New Character"** button
3. Fill in fields: Name, Description, Personality, Scenario, Example Dialogue, Gender
4. Add tags for library organization
5. Click "Save Character"

**Gen Card (AI-Powered Creation)**
1. Click **Gen Card (🪪)** in the header
2. Select Source Mode: Current Chat / Manual Input
3. Either:
   - From Chat: NeuralRP analyzes the conversation and generates the character
   - Manual Input: Describe the character in natural language
4. Click **"Auto-Generate All"** or generate individual fields
5. Review and edit the generated content
6. Click **"Save Character Card"** to export to SillyTavern V2 JSON

**Editing Characters**
1. In the Characters panel, hover over the character
2. Click the **Edit** button
3. Modify any field
4. Changes auto-sync to the database (v1.7.3+)
5. Characters reload automatically on the next message (no refresh needed)

### NPC System (Chat-Scoped Characters)

**Creating NPCs Mid-Chat**
1. Click the **"Create NPC"** button (green user-plus icon, bottom of chat)
2. Enter an NPC description (e.g., "Guard Marcus: middle-aged, tired, wants to be anywhere but here")
3. AI generates a complete character card
4. NPC is saved to the current chat only (chat-scoped)

**NPC vs Global Characters**

| Feature | Global Characters | NPCs |
|---------|-------------------|-------|
| **Scope** | Available to all chats | Available only to the current chat |
| **Storage** | Saved to character library | Saved in chat metadata |
| **Visibility** | Appears in Characters panel with green badge | Appears in Characters panel with gray NPC badge |
| **Promotion** | N/A | Can be promoted to a global character |
| **Entity ID** | `char_{filename}` | `npc_{sanitized_name}_{timestamp}.json` |
| **Edits** | Sync to database immediately | Reloaded every message for immediate effect |

**Promoting NPCs to Global Characters**
1. Click **Characters (👥)** in the header
2. Find the NPC (gray badge)
3. Hover over the NPC
4. Click the **Promote to Global** button (orange badge appears)
5. NPC becomes a global character available to all chats

**Why Use NPCs:**
- Create background characters on the fly
- Isolate one-off characters to specific stories
- Avoid cluttering the main character library

### Character Parity (v1.7.3+)

Global characters and NPCs behave identically in context assembly:

**Context Injection:**
- Both appear in SCENE CAST every turn (unless receiving a full card)
- Both get sticky full cards on first appearance + 2 more turns
- Both use the same capsule system for long-term context

**Edit Synchronization:**
- **Global Characters**: Edits sync to the database, take effect on the next message
- **NPCs**: Reloaded from metadata on every message, edits take effect immediately

### Character Reinforcement (SCENE CAST)

**SCENE CAST System (v1.11.0+)**

NeuralRP uses a scene-first architecture where all active characters are shown every turn via a SCENE CAST block:

**How SCENE CAST Works:**
- Every turn includes a block showing all active characters with their key traits and speech style
- Replaces periodic reinforcement (every 5 turns) with constant lightweight grounding
- **Single character chats**: Shows description + personality summary (50-150 tokens)
- **Multi-character chats**: Shows pre-generated capsules for each active character (50-100 tokens each)

**Sticky Full Cards (Reset Points)**

Full character cards (all fields: description, personality, scenario, example dialogue) appear at strategic reset points:

**When Full Cards Appear:**
1. **First appearance**: Character appears in chat for the first time
2. **Returning after absence**: Character absent for 20+ messages then reappears
3. **Sticky window**: Within 2 turns after a full card injection

**Sticky Window Duration:**
- Turn 1: Full card injected (first appearance)
- Turns 2-3: Full card injected (sticky window)
- Turns 4+: Only SCENE CAST capsule shown

**Why Sticky Full Cards:**
- Full cards (1000-1500 tokens) establish strong voice and presence
- Scenario and example dialogue provide voice fingerprints for the first few turns
- Prevents early-turn drift when the character is being established
- Capsules after turn 3 maintain voice without the full card overhead

**No Duplicates:**
Full-card characters are excluded from SCENE CAST to prevent redundant information. System ensures: Full card OR capsule, never both.

**Character/NPC Parity:**
- Global characters and NPCs receive identical treatment
- Both use full cards on reset points
- Both appear in SCENE CAST on regular turns
- Both maintain voice through capsules after the sticky window expires

---

## World Info

### World Info Panel

**World Library**
- Shows all world info collections available to add to chats
- Tag filter bar for quick filtering
- Expandable entries with badges indicating entry type

**Entry Badges:**
- **Red (Canon Law)**: Core rules, always injected into every turn + reinforced every 3 turns
- **Green (Constant)**: Always included (not semantically triggered) - useful for setting details
- **No badge**: Regular entries, injected via semantic search when relevant to conversation

### Creating and Editing World Entries

**Manual Entry Creation**
1. Click **World Info (🌍)** in the header
2. Click the **"Add Entry"** button
3. Fill in fields:
   - **Name**: Entry title
   - **Content**: Lore description
   - **Keys**: Keywords that trigger this entry
     - **Quoted keys** (`"Great Crash Landing"`): Exact phrase match only
     - **Unquoted keys** (`dragon`): Semantic search + flexible keyword matching
   - **Is Canon Law**: Mark as canon law (always injected + reinforced)
4. Add tags for library organization
5. Click "Save Entry"

**Gen World (AI-Powered Lore Extraction)**
1. Click **Gen World (🌍)** in the header
2. Select Source Mode: Current Chat / Manual Input
3. Either:
   - From Chat: NeuralRP analyzes the conversation and extracts world lore
   - Manual Input: Describe the world in natural language
4. Click **"Analyze All Lore"** or generate individual sections:
   - History
   - Locations
   - Creatures
   - Factions
5. Review and edit the generated content
6. Click **"Finalize & Update"** to save to the world card

**Editing World Entries**
1. In the World Info panel, expand the world
2. Hover over the entry
3. Click the **Edit** button
4. Modify any field
5. Changes auto-sync to the database with entry-level timestamps (v1.8.0+)

### Quoted vs Unquoted Keys

**Quoted Keys** (Exact Match)
- **Format**: `"Exact Phrase"` with quotes
- **Behavior**: Exact phrase match only, NO semantic search
- **Use Case**: Specific events, names, unique terms
- **Example**: `"Great Crash Landing"` matches only that exact phrase, not "crash landing" or "landing"

**Unquoted Keys** (Semantic Match)
- **Format**: `word` or `phrase` without quotes
- **Behavior**: Semantic search + flexible keyword matching
- **Use Case**: Concepts, creatures, objects, locations
- **Example**: `dragon` matches:
  - `dragon`, `dragons`, `draconic` (plurals, variants)
  - `wyrm`, `drake` (semantic synonyms)
  - Any content semantically related to dragons

**Thinking About Keys:**
- Use quoted keys for **bookmarks** (exact terms you want to match)
- Use unquoted keys for **concepts** (topics you want to appear when relevant)

### Canon Law

**What Is Canon Law:**
- Core rules of your world that must never be broken
- Always injected into every turn (no semantic matching needed)
- Reinforced every 3 turns at the end of the prompt to override character drift

**Use Cases:**
- Fundamental world constraints (e.g., "Magic is outlawed for commoners")
- Unbreakable rules (e.g., "The sun never rises in this realm")
- Character limitations (e.g., "Vampires cannot enter holy ground")

**Canon Law vs Regular Entries:**
- **Canon Law**: Always shown, reinforced periodically, overrides character drift
- **Regular Entries**: Shown only when semantically relevant to conversation
- **Constant Entries**: Always shown but not reinforced (useful for setting details)

### Smart Sync with JSON Files (v1.8.0)

NeuralRP provides intelligent synchronization between world info JSON files and the database:

**Timestamp-Based Conflict Resolution:**
- Compares JSON file mtime vs database entry `updated_at` timestamp
- Newer version wins (preserves most recent changes)

**Entry-Level Merging:**
- **New entries in JSON**: Added to database
- **Entries only in database**: KEPT (user additions via UI preserved)
- **Same UID, different content**: Newer version wins (timestamp comparison)

**Use Cases:**
- Edit world info in NeuralRP OR externally (text editor/SillyTavern)
- Changes sync intelligently without data loss
- User additions via UI never deleted during sync

**Triggering Sync:**
1. Place JSON file in `app/data/worldinfo/`
2. Click the **Refresh** button in the World Info panel
3. Or use `/api/reimport/worldinfo` endpoint with smart sync enabled

---

## Image Generation

### Vision Panel (Manual Generation)

**Access Vision Panel**
- Click the **purple wand icon** (toggle) in the chat toolbar
- Panel expands at the bottom of the screen
- Prompt stays in browser memory (no need to retype between generations)

**Image Generation Fields**
- **Vision Input**: Your main prompt (e.g., "cozy tavern interior, candlelight")
- **Negative Prompt** (optional): What you don't want to see (e.g., "ugly, low quality, bad anatomy")
- **Settings**:
  - **Steps**: 20-30 (higher = better quality, slower)
  - **Scale/CFG**: 7-12 (higher = more prompt adherence)
  - **Resolution**: Width × Height (e.g., 512×512 or 768×768)
  - **Model**: Select which SD model to use

**Generating Images**
1. Enter your prompt in Vision Input
2. Optional: Enter negative prompt
3. Adjust settings as needed
4. Click **"Generate Vision Sequence"**
5. Image appears in chat with generation parameters saved

**Performance Mode Integration**
- In Performance Mode, automatically reduces resolution when chat context is very long
- Prevents VRAM crashes on 8GB GPUs
- Emergency preset: Drops to 384×384 when context exceeds 15,000 tokens

### Character Tag Substitution

If you've assigned Danbooru tags to a character (in Character Editor), use `[CharacterName]` in prompts:

**How It Works:**
- Enter prompt with character reference: `[Jim] standing behind bar, warm lighting`
- NeuralRP automatically expands to: `grizzled bartender, short, stout, intimidating, gentle, beard, apron, warm lighting`
- Uses the Danbooru Tags field from the character card

**Benefits:**
- Consistent character appearance across all generations
- No need to memorize tag lists
- Easy to generate multiple images of the same character
- Works for both manual generation and snapshots

### Snapshots (Auto-Generated Scenes)

NeuralRP can automatically generate scene images based on your chat context - no manual prompts needed!

**How Snapshots Work (v1.10.3 Primary Character Focus):**

1. **📸 Snapshot Button**: Click the camera icon in the chat toolbar to generate an image
2. **Chat Mode Selection**: Choose who the snapshot focuses on via the dropdown in the snapshot dialog:
   - **🤖 Auto** (default): Focus on the first active character automatically
   - **👤 [Character Name]**: Force focus on a specific character (e.g., "Focus: Alice")
   - **🎭 Narrator**: Scene-only focus, no character centering
3. **LLM Scene Analysis**: NeuralRP uses the LLM to extract 5 key fields from 20 recent messages:
   - **Location**: Where the scene is taking place (3-5 words, e.g., "tavern interior", "dark forest", "cozy bedroom")
   - **Action**: What the primary character is doing NOW in the most recent turn (2-3 words, e.g., "hugging another", "standing", "fighting")
   - **Activity**: General event or engagement during the conversation (2-3 words, e.g., "at tavern", "reading book", "in forest")
   - **Dress**: What the primary character is wearing (2-3 words, e.g., "leather armor", "casual clothes", "swimsuit")
   - **Expression**: Facial expression of the primary character (1-2 words, e.g., "smiling", "worried", "angry", "neutral expression")
4. **Character Context Injection**: Primary character's description + personality injected into the LLM for better extraction accuracy
5. **Smart Prompt Construction**: Builds a Stable Diffusion prompt using:
   - **Quality tags**: masterpiece, best quality, high quality (first 3 only)
   - **Character tags**: Up to 20 Danbooru tags from the character's visual canon
   - **Scene tags**: Action + Activity + Expression (from the LLM)
   - **Dress**: Clothing description (from the LLM)
   - **Location**: Scene setting (from the LLM, with "at " prefix)
   - **User tags**: Up to 5 custom tags from settings

**Snapshot Focus Selection Table:**

| Mode | Primary Character | Behavior | Use Case |
|-------|-----------------|-----------|-----------|
| **🤖 Auto** (default) | First active character | Most common scenario - automatically selects first character |
| **👤 [Character Name]** | Named character | User wants a specific character (e.g., multi-char scene, favorite character) |
| **🎭 Narrator** | None | Scene-only description, no character centering |

**Dual Character Filtering:**

The mode affects snapshot generation in two ways:

1. **For Scene Extraction**: Determines whose description + personality is injected into the LLM prompt
   - Auto/focus:name → Specific character's card injected for better extraction
   - Narrator → No character card, pure scene analysis

2. **For Character Tag Selection**: Determines which character's Danbooru tags appear in the prompt
   - Auto/focus:name → Only that character's visual tags (up to 20)
   - Narrator → No character tags (scene tags only)
   - **Counting**: Mode affects scene extraction but NOT character counting (all active characters always counted)

**Example**: Scene with Alice (female), Bob (male), Charlie (male)
- Mode `auto`: Primary=Alice, tags=Alice only (blonde hair, blue eyes...)
- Mode `focus:Bob`: Primary=Bob, tags=Bob only (beard, tall...)
- Mode `narrator`: Primary=None, tags=none (scene only: tavern interior, standing)
- All modes: Character counting includes all 3 (2boys, 1girl)

**Character Tag Integration in Snapshots:**

If your character has **Danbooru tags** assigned (Character Editor → Danbooru Tags field), snapshots will use those visual tags for consistent appearance:
- **Example**: `1girl, blonde_hair, blue_eyes, elf, armor, standing`
- **Benefit**: Character looks the same across all snapshots, no manual tagging needed
- **Capacity**: Up to 20 tags per character (v1.10.3: 4x increase from v1.10.1)

**Viewing Snapshot Details:**

Each snapshot message shows:
- **Image**: The generated scene image
- **📋 Show Prompt Details** button (click to reveal):
  - Positive Prompt (all tags used)
  - Negative Prompt (quality filters: low quality, worst quality, bad anatomy)
  - Scene Analysis (5 fields from v1.10.3):
    - **Location**: Scene setting (e.g., "tavern interior", "dark forest")
    - **Action**: Primary character's current action (e.g., "hugging another", "standing")
    - **Activity**: General event (e.g., "at tavern", "reading book")
    - **Dress**: Character clothing (e.g., "leather armor", "casual clothes")
    - **Expression**: Facial expression (e.g., "smiling", "worried", "neutral expression")

**Snapshot History:**
- Snapshots are automatically saved to your chat history
- View past snapshots: Click **Chats (💬)** → Saved chats show snapshot count in list
- Scroll through the chat to find earlier snapshots

**Red Light Indicator:**
- If the snapshot button shows a red light (🔴), Stable Diffusion is unavailable
- Check that A1111 is running
- Verify the SD API URL in Settings is correct
- Ensure the `--api` flag is enabled in A1111

### Danbooru Tag Generator (v1.10.1)

One-click generation of Danbooru tags from character descriptions using progressive exact matching with semantic fallback.

**Prerequisites:**
- Character must have a **Description** with physical details
- Character must have **Gender** selected (Female/Male/Other)
- Danbooru database installed (see setup below)

**Generating Tags for a Character:**

1. **Edit a Character** with a physical Description (hair color, eye color, body type)
2. **Select Gender** (Female/Male/Other) - required for the generator
3. **Click "Generate Danbooru Character"** hyperlink (appears below Gender toggle)
4. **Two-Stage Semantic Matching:**
   - Stage 1: Natural language → Danbooru tags via semantic search (1304 tags in database)
   - Stage 2: Tags → Danbooru characters via semantic search (1394 characters in database)
5. **Auto-Population**: NeuralRP populates the Danbooru Tag field with best-matching tags
6. **Reroll for Variety**: Click the hyperlink again to get a different matching character

**Key Features:**
- **Natural Language Support**: "grey" → "gray", "skinny" → "slender", understands context
- **Exact Tag Matching**: Progressive tag reduction (N tags → N-1 → ... → 1 tag) with semantic fallback
- **Performance**: ~1.4 seconds per generation (50% faster than LLM-based approach)
- **Deterministic**: Same input always produces same results (caching works)
- **Gender Filtering**: Hard 1girl/1boy filter on all searches for consistent results
- **Creature Types**: Auto-detects elves, fairies, demons, etc. via semantic matching
- **NPC Parity**: Works identically for both global characters and chat-scoped NPCs
- **Added Tags**: "perky breasts", "tiny breasts" for better breast size matching
- **Progressive Matching**: Try all N tags, then N-1, N-2 down to 1 tag for best results

**Best Suited For:**
This feature works best with **anime-trained Stable Diffusion models**:
- **Illustrious** - Optimal for detailed character features. This is the first recommendation.
- **Pony Diffusion** - Excellent character reproduction, not quite as good as Illustrious.

The generated tags follow Danbooru tagging conventions which these models understand natively.

**Setting Up Danbooru Database:**

**Option 1: Use Included Excel File**
- Place `book1.xlsx` in the `app/data/danbooru/` folder (one-time setup)
- This is the easiest method if you have the file

**Option 2: Fetch from Danbooru API**

**Step 1: Get Danbooru API Key**
1. Go to https://danbooru.donmai.us/user/edit
2. Create an account if needed (free)
3. Generate an API key
4. Copy your API key

**Step 2: Fetch Character Data**
1. Navigate to the `app/data/danbooru/` folder
2. Open `fetch_danbooru_characters.py` in a text editor
3. Find line 34: `DANBOORU_API_KEY = "your_api_key_here"`
4. Paste your API key between the quotes
5. Right-click in the danbooru folder and "Open in terminal"
6. Type `python fetch_danbooru_characters.py`

**What the Script Does:**
- Reads `app/data/danbooru/tag_list.txt` (1394 character tags)
- Fetches 20 posts per character from the Danbooru API
- Extracts the top 20 associated tags by frequency
- Filters out 'solo' and meta tags (rating:, score:, etc.)
- Sorts tags by frequency (most common tags first)
- Saves clean data to `app/data/danbooru/book1.xlsx`

**Timing and Performance:**
- **Duration**: 15-30 minutes (fetches data for 1394 characters)
- **Rate Limiting**: 1 second delay between requests (Danbooru API limit)
- **Resume Capability**: If the script is interrupted, run again
  - Script automatically skips already-fetched characters
  - Progress saves every 10 characters
  - Resume from where you left off

**Importing Danbooru Characters to Database:**

The system includes 1394 pre-tagged Danbooru characters from a reference database:

```bash
# Run the import script (one-time setup)
python app/import_danbooru_characters.py
```

This takes ~60 seconds and generates semantic embeddings for all characters.

### Favorites System

Save your best snapshots and manual generations to a persistent library:

**Favoriting Snapshot Images:**
1. Generate a snapshot (📸 button)
2. Hover over the snapshot image
3. Click the **❤️ icon** to add to favorites
4. Heart turns red when favorited
5. Click again to unfavorite

**Favoriting Manual Images (Vision Panel):**
1. Open the Image Panel (purple wand icon)
2. Generate an image with a custom prompt
3. Click the **💾 Save as Favorite** button (appears after generation)
4. Confirm favorite (optional notes)

**What Happens When You Favorite:**
- **Image Saved**: Stored in your favorites library
- **Persistent**: Favorites saved across chat sessions
- **Jump to Source**: Double-click any favorite image to return to the original chat message

**Viewing Your Favorites:**
Access your favorited images anytime:

1. Click the **🖼️ Gallery** in the header to open the Favorites sidebar
2. Browse your saved images in a grid layout
3. Filter by source type (Snapshot or Manual) using toggle buttons
4. Filter by tags - click any tag chip to show only images with that tag
5. Search for specific tags using the search bar

**Jump to Source (Double-Click):**
Want to see the original chat context for a favorited image?

- **Double-click any favorite image** to jump directly to the chat where it was generated
- Automatically loads the chat and scrolls to the exact message
- Message is highlighted with a pink glow effect for easy identification
- Works for both snapshot and manual images

**Note**: If the chat was deleted or the image was removed, you'll see an error message.

### Inpainting

NeuralRP includes full inpainting support for editing parts of generated images:

**Accessing Inpaint Mode**
1. Generate an image (manual or snapshot)
2. Hover over the image
3. Click the **🖌️ Inpaint** button
4. Inpaint editor opens

**Drawing Mask**
1. Select brush size using the slider
2. Paint over the area you want to regenerate
3. Use the eraser tool to correct the mask
4. Ctrl + Z to undo strokes

**Generating Inpainted Image**
1. Adjust generation settings (steps, scale, etc.)
2. Modify the prompt if needed (add changes for inpainted area)
3. Click **"Generate Inpaint"**
4. New image appears with only the masked area regenerated

**Persistent Mask:**
- Mask is saved with the image
- Click **🔄 Regenerate** to regenerate with the same mask and settings
- Useful for iterating on the same area

---

## Configuration File (config.yaml)

NeuralRP uses a `config.yaml` file in the project root for global settings. This allows you to customize defaults without editing code.

### Where to Find It

```
neuralrp/
├── config.yaml          # Main configuration file
├── main.py
└── app/
```

### Key Settings

**Server:**
```yaml
server:
  port: 8000           # Web interface port
  log_level: "INFO"    # DEBUG, INFO, WARNING, ERROR
```

**LLM Connection:**
```yaml
kobold:
  url: "http://127.0.0.1:5001"   # KoboldCPP endpoint
```

**Context Management:**
```yaml
context:
  max_context: 10000              # Token limit for context window
  summarize_trigger_turn: 10      # Turn-based summarization trigger
  summarize_threshold: 0.90       # Token backstop (90% = edge case safety)
  history_window: 5               # Keep last 5 exchanges verbatim
  world_info_reinforce_freq: 3    # Reinforce canon law every 3 turns
```

**Sampling Parameters (v2.0.1+):**

These control LLM output quality and help prevent repetition loops:

```yaml
sampling:
  temperature: 0.7           # Creativity (0.1-2.0, lower = more focused)
  top_p: 0.85                # Nucleus sampling threshold
  top_k: 60                  # Vocabulary limit
  repetition_penalty: 1.12   # Anti-loop penalty (1.0-2.0)
```

### Tuning for Your Model

**For 7B models** (more repetition-prone):
```yaml
sampling:
  repetition_penalty: 1.2
  top_p: 0.8
  top_k: 40
```

**For 11-12B models** (balanced defaults):
```yaml
sampling:
  repetition_penalty: 1.12
  top_p: 0.85
  top_k: 60
```

**For 20B+ models** (less repetition-prone):
```yaml
sampling:
  repetition_penalty: 1.05
  top_p: 0.9
  top_k: 100
```

### Environment Variable Overrides

You can override any setting without editing the file:

```bash
# Windows (PowerShell)
$env:NEURALRP_SAMPLING_REPETITION_PENALTY="1.2"
$env:NEURALRP_SERVER_PORT="8080"

# Linux/macOS
export NEURALRP_SAMPLING_REPETITION_PENALTY=1.2
export NEURALRP_SERVER_PORT=8080
```

Format: `NEURALRP_{SECTION}_{KEY}` (uppercase, underscores for nested keys)

---

## Advanced Features

### Gen Card (AI Character Generator)

**Purpose**: Create optimized character cards using AI without writing from scratch

**Two Source Modes:**

**From Current Chat**
- NeuralRP analyzes the conversation history
- Extracts character traits from dialogue and interactions
- Generates a complete character card based on observed behavior
- Useful for formalizing NPCs or capturing memorable characters

**From Manual Input**
- Write a character description in natural language
- NeuralRP converts to SillyTavern V2 format automatically
- Example: "A grumpy old blacksmith named Thorne who loves crafting legendary weapons"
- Output: Full character card with all fields filled

**Generation Options:**
- **Auto-Generate All**: Creates a complete character card at once
- **Individual Fields**: Generate each section separately for fine control
  - Personality
  - Physical Body
  - Genre & Tags
  - Scenario
  - First Message
  - Example Dialog

**Output Format:**
- SillyTavern V2 JSON (PList-optimized)
- Compatible with SillyTavern and NeuralRP
- Ready to use immediately

### Gen World (AI World Lore Extractor)

**Purpose**: Extract world-building information from conversation or descriptions

**Two Source Modes:**

**From Current Chat**
- NeuralRP analyzes the entire conversation
- Identifies recurring themes, locations, characters, factions
- Generates world entries with automatic semantic keys
- Useful for building worlds organically through roleplay

**From Manual Input**
- Describe your world in natural language
- NeuralRP creates structured world entries
- Example: "A fantasy realm where magic is outlawed but secretly practiced by a rebellion"
- Output: Multiple world entries with keys and content

**Lore Categories:**
- **History**: Timeline of major events
- **Locations**: Cities, landmarks, geographic features
- **Creatures**: Races, animals, monsters
- **Factions**: Political groups, organizations, guilds

**Generation Options:**
- **Analyze All Lore**: Extract all categories at once
- **Individual Sections**: Generate each category separately
- **Finalize & Update**: Save extracted lore to the world card

### Search System

**Purpose**: Search messages across all chats to find specific content

**Search Bar:**
- Enter query text to search for
- Supports natural language queries
- Searches message content across all saved chats

**Filters (Expandable):**
- **Filter by Chat**: Limit search to a specific chat
- **Filter by Speaker**: Limit search to a specific character or narrator
- **Date Range**: Limit search to a specific time period (From/To)

**Search Results:**
- Shows matching messages with context
- **Jump to Message**: Navigate to that point in the conversation
- **Show Context**: View surrounding messages for context

### Export System

**Purpose**: Export chats for training LLMs (Unsloth, fine-tuning)

**Supported Formats:**
- **Alpaca**: Standard instruction format
- **ShareGPT**: OpenAI conversation format
- **ChatML**: Microsoft conversation format

**Export Options:**
- **Include system prompts**: Add system instructions to export
- **Include world info**: Add Canon Law entries (only)
- **Include NPC responses**: Include NPC-generated content

**Usage:**
1. Save chat to the database (Chats → Save Session)
2. Click **Export (📤)** in the header
3. Select format and options
4. Click **"Export Training Data"**
5. Download JSON file optimized for fine-tuning

### Settings Panel Overview

**General Settings:**
- **System Prompt**: Override default system instruction (tone, formatting rules)
- **Your Name**: User identifier for conversations
- **Your Persona**: User character description (affects snapshots and character interactions)
- **Reinforce World Canon Every X Turns**: Default 3 (Canon law reinforcement frequency)
- **Enable World Info**: Toggle world lore inclusion
- **Max World Info Entries**: Limit semantic matches (0 = unlimited)
- **Temperature**: LLM randomness (0.0-2.0, lower = more focused, higher = more creative)
- **Max Resp Tokens**: Response length limit
- **Max Context**: Context window size (default: 8192)
- **Summarize at Turn**: Turn number that triggers summarization (default: 10)

**Performance Mode:**
- **Smart Performance Mode**: Queues heavy operations to prevent VRAM crashes
- Automatic resolution reduction when context is very long
- Emergency preset: Drops to 384×384 when context exceeds 15,000 tokens

**Long-Term Memory:**
- **Current Summary**: Read-only view of chat summary
- **Clear Memory**: Reset conversation summary (use with caution)

**World Info Cache:**
- **Cache Status**: Display number of entries, max limit, usage %, memory
- **Refresh**: Reload world info from database
- **Clear**: Clear cached embeddings
- **Cache Size Limit**: Maximum number of entries to cache

**API Configuration:**
- **Kobold URL**: Backend API endpoint (default: http://127.0.0.1:5001)
- **SD API URL**: Stable Diffusion endpoint (default: http://127.0.0.1:7860)
- **Update & Test Buttons**: Verify connections and show status (green/yellow/red)

**Data Recovery:**
- **View Change History**: 30-day retention of all changes
- Restore characters, world info, and chats to previous states
- Browse and inspect specific changes

**Status Monitor:**
- **Service Health**: Status of KoboldCpp and Stable Diffusion connections
- **Database Stats**: Database size, table counts, connection info
- **Storage Health**: Disk space usage, data directory size
- **Maintenance Status**: Backup count, last backup time, cleanup status

### Smart Sync (v1.8.0)

**Purpose**: Intelligently synchronize JSON files with the database, preventing data loss while preserving user modifications

**Character Sync:**
- **Timestamp-based conflict resolution**: Newer version wins
- Compares JSON file mtime vs database `updated_at`
- If JSON is newer: Updates database from JSON
- If database is newer: Keeps database version (no action)

**World Info Sync:**
- **Entry-level merging**: Preserves user additions from both sources
- **New entries in JSON**: Added to database
- **Entries only in database**: KEPT (user additions via UI)
- **Same UID, different content**: Newer version wins (timestamp comparison)

**Use Cases:**
- Edit cards in NeuralRP OR externally (SillyTavern/text editor)
- Changes sync intelligently without data loss
- User additions via UI never deleted during sync

**Triggering Sync:**
- Characters/World Info panels: Click **Refresh** button
- Automatic: Imports new JSON files on startup

---

## Tips and Best Practices

### Context Management

**Why Context Summarization Helps:**
- LLMs have finite context windows (4096-8192 tokens)
- Long conversations exceed this limit quickly
- Summarization keeps recent dialogue while compressing older content
- Enables indefinite conversations without hitting context overflow

**Optimal Summarization Settings:**
- **80% threshold** (default): Good balance between recency and context
- **70-75%**: More frequent summarization, less context but better performance
- **85-90%**: Less frequent summarization, more context but higher overflow risk

**Reviewing and Editing Summaries:**
- Check summaries periodically for accuracy
- Use Autosummarize feature to condense verbose sections
- Edit summaries to ensure key plot points are preserved
- Clear memory button available if summary needs a full reset

### Character Consistency

**How to Maintain Character Voice:**
- Provide clear, detailed descriptions and personalities
- Include example dialogue showing speech patterns
- Use SCENE CAST system (automatic in v1.11.0+)
- Characters receive regular reinforcement every turn
- Sticky full cards on turns 1-3 establish strong voice

**Multi-Character Chats:**
- Each character gets their own capsule in SCENE CAST
- Capsules show key traits and speech style
- Characters maintain distinct voices via capsules
- Focus mode directs conversation to a specific character

**When Characters Start Drifting:**
- Check that description and personality fields are clear
- Add more example dialogue showing desired voice
- Lower temperature for more focused responses
- Consider editing the character card mid-conversation

### Performance Tips

**For 8GB VRAM Users:**
- Enable **Performance Mode** in Settings
- Reduce **GPU Layers** in KoboldCpp (try 20-30 layers)
- Use smaller models (L3-8B instead of 13B/27B)
- Reduce **Resolution** in image generation (512×512)
- Enable **Summarization** at 70-75% instead of 80%

**For 12GB+ VRAM Users:**
- Offload more GPU layers in KoboldCpp (35-50+ layers)
- Use larger models (Tiefighter 13B or Gemma-3-27B)
- Increase **Context Size** in KoboldCpp (4096-8192)
- Generate higher resolution images (768×768 or 1024×1024)

**General Optimization:**
- Close other GPU-intensive applications
- Reduce the number of active characters if context is large
- Limit world info entries via Max World Info Entries setting
- Use Performance Mode when running multiple LLMs

### World Info Best Practices

**Organizing World Entries:**
- Group related concepts into single entries
- Use quoted keys for specific terms ("The Great War")
- Use unquoted keys for concepts (magic, dragons)
- Mark core rules as Canon Law
- Use Constant entries for setting details (always shown)

**Effective Keywords:**
- Choose specific terms for quoted keys (exact match)
- Choose broader concepts for unquoted keys (semantic match)
- Include synonyms and variants in unquoted keys
- Test triggers by mentioning keywords in conversation

**When World Info Isn't Injecting:**
- Verify **Enable World Info** is checked in Settings
- Check that keywords are spelled correctly
- Try mentioning keywords directly in conversation
- Check World Info Cache status in Settings (Refresh if needed)
- For unquoted keys, try related terms (semantic matching)

### Image Generation Quality Tips

**Manual Generation:**
- Use descriptive prompts with specific details
- Include art style references (anime, realistic, oil painting)
- Use negative prompts to filter unwanted elements
- Start with lower steps (20) for speed, increase for quality
- Adjust scale/CFG (7-12 range works well)

**Snapshots:**
- Provide detailed descriptions for characters (helps LLM extraction)
- Set Danbooru tags for consistent character appearance
- Use Focus mode to target specific characters in multi-char scenes
- Review scene analysis fields to ensure accuracy

**Danbooru Tag Generator:**
- Provide detailed physical descriptions (hair color, eye color, body type)
- Select Gender correctly (required for generator)
- Reroll for variety if first result isn't ideal
- Works best with anime-trained models (Illustrious, Pony)

**Character Tag Substitution:**
- Set Danbooru tags for main characters once
- Use `[CharacterName]` in all prompts for consistency
- Update tags if character appearance changes
- Useful for generating image sequences of the same character

---

## Troubleshooting

### Connection Issues

**"KoboldCpp connection failed" (red status in Settings):**

1. Check that KoboldCpp is running (GUI window open)
2. Verify URL is correct (default: `http://127.0.0.1:5001`)
3. Check firewall settings (allow Python/KoboldCpp through)
4. Try restarting KoboldCpp and NeuralRP
5. If using a different port, update Kobold URL in Settings

**"Stable Diffusion connection failed" (red status in Settings):**

1. Check that A1111 is running (web interface accessible at http://localhost:7860)
2. Verify URL is correct (default: `http://127.0.0.1:7860`)
3. Ensure model is loaded in A1111 (check top-left dropdown)
4. Check that `--api` flag is enabled (default in webui-user.bat/webui.sh)
5. Try restarting A1111 and NeuralRP

**Yellow status (slow response):**
- Backend is responding but slowly
- Check GPU utilization in Task Manager / htop
- Reduce context size or summarization threshold
- Close other GPU-intensive applications

### Out of Memory / VRAM Crashes

**Symptoms:**
- NeuralRP crashes or freezes during generation
- "Out of memory" errors in console
- System becomes unresponsive

**Solutions:**

1. **Enable Performance Mode** in Settings
   - Queues heavy operations to prevent crashes
   - Automatically reduces resolution when context is large

2. **Reduce GPU Layers in KoboldCpp**
   - Try 20-30 layers instead of all
   - Reduces VRAM usage significantly

3. **Reduce Image Resolution**
   - Use 512×512 instead of 768×768
   - Snapshots automatically drop to 384×384 in emergency mode

4. **Summarize Chat**
   - Free context space by summarizing old messages
   - Use lower Summarize at Turn value (6-8) for more frequent summarization

5. **Use Smaller LLM Models**
   - Switch from 13B to 8B model
   - Consider L3-8B-Stheno for efficiency

6. **Close Other Applications**
   - Close other GPU-intensive programs
   - Free up VRAM for NeuralRP

### Character Consistency Problems

**Characters responding inconsistently or losing voice:**

1. **Check Character Card**
   - Verify description and personality are well-written and specific
   - Ensure example dialogue shows desired speech patterns
   - Add more example dialogue if voice is unclear

2. **Verify Character is Active**
   - Check that character appears as pill badge in header
   - Character must be active to be in SCENE CAST

3. **Adjust Temperature**
   - Lower temperature (0.3-0.7) for more focused responses
   - Higher temperature (1.0-1.5) for more creative but less consistent responses

4. **Check Reinforcement Frequency**
   - SCENE CAST shows character every turn (v1.11.0+)
   - Full cards appear on turns 1-3 and reset points
   - No manual adjustment needed for modern architecture

5. **Edit Character Mid-Conversation**
   - If character drifts significantly, edit character card
   - Changes take effect on next message
   - Consider updating personality or example dialogue

### World Info Not Injecting

**World lore entries not appearing in context:**

1. **Verify Enable World Info**
   - Check that **Enable World Info** is checked in Settings

2. **Check Keywords**
   - Verify world info entries have relevant keywords
   - Quoted keys (`"Event"`) require exact phrase match
   - Unquoted keys (`dragon`) use semantic matching

3. **Test Triggering**
   - Mention keyword directly in conversation
   - Try related terms for unquoted keys
   - Check Search panel to confirm entry exists

4. **Check World Info Cache**
   - Go to Settings → World Info Cache section
   - Check cache status (entries, max, usage %)
   - Click **Refresh** if cache is stale

5. **Verify Entry Type**
   - Canon Law entries always shown (red badge)
   - Constant entries always shown (green badge)
   - Regular entries only shown when semantically relevant (no badge)

### Image Generation Problems

**Images not generating or appearing:**

1. **Check SD Connection**
   - Verify SD API URL in Settings is correct
   - Check that A1111 status is green (connected)
   - Restart A1111 if connection is lost

2. **Verify Model is Loaded**
   - Open A1111 web interface (http://localhost:7860)
   - Check model dropdown in top-left
   - Select a model if none is loaded

3. **Check Red Light Indicator**
   - Snapshot button shows 🔴 when SD is unavailable
   - Resolve connection issues before generating images

**Image quality issues:**

1. **Adjust Generation Settings**
   - Increase steps (20-30) for better quality
   - Adjust scale/CFG (7-12 range)
   - Try different resolution (512×512, 768×768)

2. **Improve Prompts**
   - Add more descriptive details
   - Include art style references
   - Use negative prompts to filter unwanted elements

3. **Character Tag Issues**
   - Verify Danbooru tags are set for character
   - Ensure tags are appropriate for SD model being used
   - Works best with anime-trained models (Illustrious, Pony)

**Inpainting not working:**

1. **Verify A1111 API is Running**
   - Inpaint requires --api flag (default in A1111)
   - Check connection status in Settings

2. **Check Mask is Drawn**
   - Ensure you've painted over area you want to regenerate
   - Use brush size slider for control

3. **Regeneration Fails**
   - Try reducing resolution
   - Check VRAM usage (may need to enable Performance Mode)

### Database Issues

**Corruption Errors on Startup:**

If NeuralRP detects database corruption, you'll see a warning in console advising you to run migration script:

1. **Backup Current Database**
   - Copy `app/data/neuralrp.db` to a safe location
   - Keeps your data as backup

2. **Check JSON Exports**
   - Look in `app/data/characters/` and `app/data/worldinfo/`
   - These are auto-generated backups

3. **Run Migration Script** (if needed)
   - Follow console instructions for rebuilding from JSON backups

**Slow Performance:**

1. **Check Database Size**
   - Large databases (10,000+ messages) may be slower
   - Consider archiving old chats

2. **World Info Cache**
   - Clear cache if it's very large
   - Reduce Max World Info Entries setting

3. **Semantic Search Overhead**
   - Reduce world info entries or disable temporarily
   - Use quoted keys (faster) instead of unquoted keys

---

## Where to Get Help

**Documentation:**
- [Quickstart Guide](QUICKSTART.md) - Get started quickly
- [Technical Documentation](TECHNICAL.md) - Deep dive into implementation
- [Changelog](CHANGELOG.md) - Version history and known issues

**Community Resources:**
- **KoboldCpp**: [GitHub Issues](https://github.com/LostRuins/koboldcpp/issues) or [Discord](https://discord.gg/koboldai)
- **AUTOMATIC1111**: [GitHub Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) or [Discord](https://discord.gg/stablediffusion)
- **NeuralRP**: [GitHub Discussions](https://github.com/neuralrp/neuralrp/discussions) - Report bugs and ask questions

**Model Resources:**
- **Text Models**: Search "GGUF" on [Hugging Face](https://huggingface.co)
- **Image Models**: [CivitAI](https://civitai.com) or [Hugging Face](https://huggingface.co)
- **Text Model Recommendations**: [Bartowski's HuggingFace](https://huggingface.co/bartowski) (reliable GGUF quantizations)

---

## Document Version

This User Guide covers NeuralRP v2.0.0 and later features. For latest changes, see [CHANGELOG.md](../CHANGELOG.md).

---

## The v2.0.0+ Direction

**Simplicity and Stability**

Starting with version 2.0.0, NeuralRP embraced a philosophy of simplification: focusing on what truly matters for roleplay quality while removing complex systems that provided limited value.

**What Changed:**

- **Removed Forking**: The ability to create alternate timelines has been removed. Instead, use the simple "Save Chat Under Different Name" workflow if you want to explore alternate story paths.
- **Removed Relationship Tracking**: The automatic emotional state tracking system has been removed. Relationships now emerge naturally through dialogue quality rather than numeric injections.

**Why These Changes?**

Complex systems break in complex ways. Simple systems break in simple ways.

The forking and relationship systems, while powerful, were used by less than 5% of users but required more than 20% of development time to maintain. They added 1+ seconds of overhead to every generation request and consumed valuable token budget (100-200 tokens per turn) for features that often conflicted with natural dialogue patterns.

**What This Means for You:**

- **Faster Responses**: Removed 1+ second of blocking overhead from every generation
- **Better Character Voices**: Relationships emerge naturally from dialogue instead of being forced by numeric scores
- **More Context**: Reclaimed 100-200 tokens per turn for actual story content
- **More Reliable Features**: Fewer complex edge cases means fewer bugs and smoother experience
- **Easier Development**: New features can be added faster and with fewer integration concerns

For a deep dive into the reasoning behind these changes, see the [Philosophy documentation](PHILOSOPHY.md).

**Focus on What Works:**

NeuralRP now focuses exclusively on systems that demonstrably improve roleplay quality:

- **Scene-First Architecture**: SCENE CAST ensures consistent character voices across long conversations
- **Context Hygiene**: Smart summarization and token management enable 200+ turn conversations
- **Character Consistency**: Sticky full cards and capsule-based reinforcement maintain voice without bloat
- **Operational Excellence**: Automatic backups, structured logging, and health monitoring ensure reliable long-term operation

The goal is simple: A better roleplay engine that's easier to use, faster, and more reliable.
