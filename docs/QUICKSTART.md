# NeuralRP Quickstart Guide

Get up and running with NeuralRP in minutes. This guide covers everything you need to start roleplaying with local LLMs and Stable Diffusion.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Downloading and Setting Up Backends](#downloading-and-setting-up-backends)
3. [Installing NeuralRP](#installing-neuralrp)
4. [Configuring Backends in NeuralRP](#configuring-backends-in-neuralrp)
5. [Basics of Using the App](#basics-of-using-the-app)
6. [Header Menu Options](#header-menu-options)
7. [Demo Files Setup](#demo-files-setup)
8. [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## Prerequisites

**Hardware Requirements:**

- **Recommended:** 12-16GB VRAM GPU (NVIDIA/AMD)
- **Minimum:** 8GB VRAM (with Performance Mode enabled)

**Software Required:**

- Python 3.8+ (for NeuralRP)
- Python 3.10.6 specifically for A1111 (newer versions don't support torch)

**What You'll Need to Download:**

1. KoboldCpp (LLM inference backend)
2. AUTOMATIC1111 Stable Diffusion WebUI (image generation backend)
3. NeuralRP (this repository)

---

## Downloading and Setting Up Backends

### Step 1: Install KoboldCpp (LLM Backend)

**Download KoboldCpp:**

1. Go to the [KoboldCpp GitHub Releases](https://github.com/LostRuins/koboldcpp/releases)
2. Download the latest `koboldcpp.exe` (Windows) or appropriate binary for your system
3. Place it in a folder of your choice (e.g., `C:\KoboldCpp\`)

**Run KoboldCpp:**

1. Double-click `koboldcpp.exe`
2. The GUI will open automatically
3. Click **"Load Model"** and navigate to your GGUF model file (huggingface)
4. **Default API URL:** `http://localhost:5001`
5. Keep KoboldCpp running while using NeuralRP

**Recommended GGUF Models:**

If you're a beginner:
-https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
- **L3-8B-Stheno-v3.2** - Smaller, faster, unfiltered
- **Tiefighter 13B** - Versatile, balanced performance, need 12 gig vram 


**Where to get models:** Search for "GGUF" on [Hugging Face](https://huggingface.co)

**Performance Tips:**

- **NVIDIA GPU:** Run with `--usecuda` flag for CUDA acceleration
- **AMD/Other GPU:** Use `--usevulkan` for Vulkan support
- **GPU Layer Offloading:** Add `--gpulayers N` to offload N layers to VRAM (more layers = faster speed)
- **Context Size:** Use `--contextsize N` to increase context window (default is 2048)

---

### Step 2: Install AUTOMATIC1111 Stable Diffusion WebUI (Image Generation)

**Install A1111:**

1. **Install Python 3.10.6** (Critical: Newer Python versions don't support torch!)
   - Download from [Python.org](https://www.python.org/downloads/release/python-3106/)
   - Check "Add Python to PATH" during installation

2. **Install Git** (if not already installed)
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Clone the repository:**

   **Windows:**
   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

   **Linux:**
   ```bash
   sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

4. **Run A1111:**

   **Windows:** Double-click `webui-user.bat`
   **Linux:** Run `./webui.sh`

5. **First Run:** Wait for dependencies to install (this may take several minutes)

**Default API URL:** `http://localhost:7860`

**Download SD Models:**

- **Anything v3** - Anime style, versatile
- **PerfectDeliberate V2** - Realistic, detailed
- **Dreamshaper SDXL** - High quality, modern
- **Danbooru tag support** - https://civitai.com/models/1277670/janku-trained-noobai-rouwei-illustrious-xl

**Where to get models:** [CivitAI](https://civitai.com) or [Hugging Face](https://huggingface.co)

Place `.safetensors` model files in `stable-diffusion-webui/models/Stable-diffusion/`

**Keep A1111 running** while using NeuralRP for image generation

---

## Installing NeuralRP

### Option 1: Using Launcher (Windows - Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/neuralrp/neuralrp.git
   cd neuralrp
   ```

2. **Run the launcher:**
   ```bash
   launcher.bat
   ```

The launcher handles dependency installation and startup automatically.

### Option 2: Manual Installation (All Systems)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/neuralrp/neuralrp.git
   cd neuralrp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run NeuralRP:**
   ```bash
   python main.py
   ```

### Access NeuralRP

Once running, open your browser and navigate to:

**http://localhost:8000**

---

## Configuring Backends in NeuralRP

### First-Time Setup

1. **Open NeuralRP** in your browser (`http://localhost:8000`)

2. **Click Settings (⚙️)** in the top header menu

3. **Scroll to "API Configuration"** section (near the bottom)

4. **Configure KoboldCpp:**

   - **Kobold URL:** `http://127.0.0.1:5001`
   - Click **"Update & Test"** button
   - Wait for **green status indicator** (means connected)
   - If red/yellow, check that KoboldCpp is running and the URL is correct

5. **Configure Stable Diffusion:**

   - **SD API URL:** `http://127.0.0.1:7860`
   - Click **"Update & Test"** button
   - Wait for **green status indicator** (means connected)
   - If red/yellow, check that A1111 is running and the URL is correct

**Both services must show green status** before you can use NeuralRP effectively.

---

## Basics of Using the App

> ⚠️ **First Run Notice:** The first time you start NeuralRP, it will download a ~400MB AI model for semantic search. This takes 5-10 minutes depending on your internet speed. The app may appear to hang during this time—**this is normal**. Subsequent launches will be much faster.

### Starting a New Chat

**Narrator Mode (Default):**

- When no characters are selected, you're in "Narrator Mode" (blue indicator in header)
- Type your prompt in the input field at the bottom
- Press **Enter** or click the blue paper plane icon to send
- The AI generates a response as the narrator

**Adding Characters:**

1. Click **Characters (👥)** in the header
2. Click on a character card to activate it (blue border appears)
3. Active characters appear as **pill badges** in the center of the header
4. To remove a character, click the **×** on its pill badge

**Chat Modes:**

Use the dropdown menu in the bottom input area:

- **🤖 Auto** - AI selects best responder (character or narrator)
- **🎭 Narrator** - Story narration mode
- **👤 [Character Name]** - Force specific character to respond

**Example Workflow:**

```
1. Add "Jim the Bartender" character
2. Select "Auto" mode
3. Type: "I walk into the bar and sit on a stool."
4. Jim responds: "Well, well, well... looks like we've got a new face..."
```

### Generating Images

1. Click the **purple wand icon** (toggle) to expand the Image Panel (bottom)
2. **Vision Input:** Enter your prompt (e.g., "Jim the Bartender, cozy tavern interior, candlelight")
3. **Negative Prompt** (optional): What you don't want to see
4. **Adjust Settings:**
   - Steps: 20-30 (higher = better quality, slower)
   - Scale/CFG: 7-12 (higher = more prompt adherence)
   - Resolution: Width × Height (e.g., 512×512 or 768×768)
5. Click **"Generate Vision Sequence"**
6. Image appears in chat with generation parameters saved

**Character Tag Substitution:**

If you've assigned Danbooru tags to a character (in Character Editor), use `[CharacterName]` in prompts:

- Example: `[Jim] standing behind bar, warm lighting`
- NeuralRP automatically expands to: `grizzled bartender, short, stout, intimidating, gentle, beard, apron, warm lighting`

This ensures consistent character appearance across all generations.

**Danbooru Tag Generator (v1.10.1):**

One-click generation of Danbooru tags from character descriptions using two-stage semantic matching.

**Directions to make it work**

Go to: /neuralrp/app/data/danbooru
Open fetch_danbooru_characters.py
---NSFW Warning--- (the danbooru website can have racy images)
Put in your danbooru api key (create an account at https://danbooru.donmai.us/, the API key is free)
Right-click in danbooru folder and "Open in terminal"
Type "python fetch_danbooru_characters.py"

That command pulls 1300 character's worth of tags in, which get uploaded to the database the next time you open NeuralRP!

**How It Works (v1.10.1 Semantic System):**
1. **Edit a Character** with a physical Description (hair color, eye color, body type)
2. **Select Gender** (Female/Male/Other) - required for the generator
3. **Click "Generate Danbooru Character"** hyperlink (appears below Gender toggle)
4. **Two-Stage Semantic Matching**:
   - Stage 1: Natural language → Danbooru tags via semantic search (1304 tags in database)
   - Stage 2: Tags → Danbooru characters via semantic search (1394 characters in database)
5. **Auto-Population**: NeuralRP populates the Danbooru Tag field with best-matching tags
6. **Reroll for Variety**: Click the hyperlink again to get different matching character

**Key Features:**
- **Natural Language Support**: "grey" → "gray", "skinny" → "slender", understands context
- **No Keyword Maps**: Pure semantic matching handles synonyms, variants, context automatically
- **Performance**: ~1.4 seconds per generation (50% faster than LLM-based approach)
- **Deterministic**: Same input always produces same results (caching works)
- **Gender Filtering**: Hard 1girl/1boy filter on all searches for consistent results
- **Creature Types**: Auto-detects elves, fairies, demons, etc. via semantic matching
- **NPC Parity**: Works identically for both global characters and chat-scoped NPCs
- **Added Tags**: "perky breasts", "tiny breasts" for better breast size matching

**Best Suited For:**
This feature works best with **anime-trained Stable Diffusion models**:
- **Illustrious** - Optimal for detailed character features. This is my first recommendation.
- **Pony Diffusion** - Excellent character reproduction, not quite as good as Illustious.

The generated tags follow Danbooru tagging conventions which these models understand natively.

**Requirements:**
- Character must have a **Description** with physical details
- Character must have **Gender** selected
- Excel import: Place `Book1.xlsx` in `app/data/danbooru/` folder (one-time setup)

**Importing Danbooru Characters:**

The system includes 1394 pre-tagged Danbooru characters from a reference database:

```bash
# Run the import script (one-time setup)
python app/import_danbooru_characters.py
```

This takes ~60 seconds and generates semantic embeddings for all characters.

**Note on Semantic Search (v1.10.1):**
- ✅ Normal operation: Uses semantic search with natural language → tags → characters pipeline
- ⚠️ Warning if embeddings fail: Falls back to no-op (still generates tags, just no character matching)
- Check startup logs for: `[SETUP WARNING] Could not create vec_danbooru_characters table`

**Fetching Danbooru Character Data (Optional):**

Don't have `book1.xlsx`? Generate it automatically from Danbooru API:

**Step 1: Get Danbooru API Key**

1. Go to https://danbooru.donmai.us/user/edit
2. Create an account if needed (free)
3. Generate an API key
4. Copy your API key

**Step 2: Fetch Character Data**

1. Edit `app/fetch_danbooru_characters.py`
2. Find line 34: `DANBOORU_API_KEY = "your_api_key_here"`
3. Paste your API key between the quotes
4. Run the script:
   ```bash
   python app/fetch_danbooru_characters.py
   ```
5. Wait 15-30 minutes (fetches data for 1394 characters)

**What the Script Does:**

- Reads `app/data/danbooru/tag_list.txt` (1394 character tags)
- Fetches 20 posts per character from Danbooru API
- Extracts top 20 associated tags by frequency
- Filters out 'solo' and meta tags (rating:, score:, etc.)
- Sorts tags by frequency (most common tags first)
- Saves clean data to `app/data/danbooru/book1.xlsx`

**Resume Capability:**

- If script is interrupted, run again
- Script automatically skips already-fetched characters
- Progress saves every 10 characters
- Resume from where you left off

**Rate Limiting:**

- 1 second delay between requests (Danbooru API limit)
- Retry logic with exponential backoff (3 attempts)
- API key provides better rate limits than public access

**Example Workflow:**
1. Create character "Alice" with description: "blonde hair, blue eyes, petite elf girl"
2. Select Gender: Female
3. Click **"Generate Danbooru Character"** → System populates field with: `1girl, blonde_hair, blue_eyes, small_breasts, pointy_ears, elf`
4. Generate snapshot → Prompt uses generated tags: `1girl, blonde_hair, blue_eyes, small_breasts, pointy_ears, elf, masterpiece, best quality`

**Favorites System:**

Save your best snapshots and manual generations to a persistent library:
- **❤️ Favorite Images**: Heart icon on snapshots, "Save as Favorite" button on manual generations
- **Persistent Library**: Favorites saved across all chat sessions
- **Jump to Source**: Double-click any favorite image to return to the original chat message
- **Works for Both**: Snapshots (📸) and manual images (Vision Panel) support favorites

See "Favoriting Images (Snapshot and Manual)" section below for detailed instructions.

### Generating Snapshots (Automatic Scene Images)

NeuralRP can automatically generate scene images based on your chat context - no manual prompts needed!

**How Snapshots Work (v1.10.1 Simplified System):**

1. **📸 Snapshot Button**: Click the camera icon in the chat toolbar to generate an image
2. **LLM Scene Analysis**: NeuralRP uses the LLM to extract 3 key fields from your conversation:
    - **Location**: Where the scene takes place (e.g., "tavern", "forest", "castle")
    - **Action**: What characters are doing (e.g., "standing", "sitting", "fighting")
    - **Dress**: Clothing details (e.g., "wearing armor", "in casual clothes")
3. **Smart Prompt Construction**: Builds a Stable Diffusion prompt using:
    - **Quality tags**: masterpiece, best quality, high quality
    - **Character tags**: From assigned Danbooru tags (or semantic matching)
    - **Location**: Scene setting extracted from conversation
    - **Action**: Character activity extracted from conversation
    - **Dress**: Clothing details extracted from conversation
4. **3-Tier Fallback System**: If LLM extraction fails:
    - Tier 1: JSON extraction (primary method)
    - Tier 2: Pattern matching (keyword detection)
    - Tier 3: Empty scene (basic character tags only)
5. **Image Generation**: Sends prompt to Stable Diffusion and displays result in chat

**Character Tag Integration:**

If your character has **Danbooru tags** assigned (Character Editor → Danbooru Tags field), snapshots will use those visual tags:
- **Example**: `1girl, blonde hair, blue eyes, school uniform`
- **Benefit**: Ensures consistent character appearance across all snapshots

**User Inclusion (v1.10.1):**

Include yourself in snapshots by checking the "Include in Snapshots" checkbox:
- Your user card settings (Name, Gender) are used for tag generation
- Auto-counting adjusts character counts (e.g., "1girl, 1boy" for you + character)
- Persists per chat - check once, applies to all snapshots in that conversation

**Viewing Snapshot Details:**

Each snapshot message shows:
- **Image**: The generated scene image
- **📋 Show Prompt Details** button (click to reveal):
  - Positive Prompt (all tags used)
  - Negative Prompt (quality filters: low quality, worst quality, bad anatomy)
  - Scene Analysis (extracted location, action, dress)

**Snapshot History:**

Snapshots are automatically saved to your chat history. View past snapshots:
1. Click **Chats (💬)** in header
2. Saved chats show snapshot count in list
3. Scroll through chat to find earlier snapshots

**Red Light Indicator:**

If the snapshot button shows a red light (🔴), Stable Diffusion is unavailable:
- Check that A1111 is running
- Verify SD API URL in Settings is correct
- Ensure `--api` flag is enabled in A1111

### Favoriting Images (Snapshot and Manual)

Save your favorite images for future reference:

**Snapshot Images:**

1. Generate a snapshot (📸 button)
2. Hover over the snapshot image
3. Click the **❤️ icon** to add to favorites
4. Heart turns red when favorited
5. Click again to unfavorite

**Manual Images (Vision Panel):**

1. Open Image Panel (purple wand icon)
2. Generate image with custom prompt
3. Click **💾 Save as Favorite** button (appears after generation)
4. Confirm favorite (optional notes)

**What Happens When You Favorite:**

- **Image Saved**: Stored in your favorites library
- **Persistent**: Favorites saved across chat sessions
- **Jump to Source**: Double-click any favorite image to return to the original chat message

**Viewing Your Favorites:**

Access your favorited images anytime:

1. Click **🖼️ Gallery** in the header to open the Favorites sidebar
2. Browse your saved images in a grid layout
3. Filter by source type (Snapshot or Manual) using the toggle buttons
4. Filter by tags - click any tag chip to show only images with that tag
5. Search for specific tags using the search bar

**Jump to Source (Double-Click):**

Want to see the original chat context for a favorited image?

- **Double-click any favorite image** to jump directly to the chat where it was generated
- Automatically loads the chat and scrolls to the exact message
- Message is highlighted with a pink glow effect for easy identification
- Works for both snapshot and manual images

*Note: If the chat was deleted or the image was removed, you'll see an error message.*

### Managing Chat Summaries

NeuralRP automatically summarizes long conversations to maintain context efficiency while preserving message history.

**How Summaries Work:**

- **Automatic Trigger**: When chat reaches 85% context threshold (configurable in Settings)
- **Smart Summarization**: Condenses conversation into 150-token summaries
- **Soft Delete**: Old messages are archived (not deleted), preserving relationship tracking IDs
- **Preserved History**: Archived messages remain searchable via Search panel

**Viewing Summaries:**

1. Click **Settings (⚙️)** in the header
2. Scroll to **Long-Term Memory** section
3. View current chat summary (read-only display)
4. Click **Clear Memory** to reset summary (rarely needed)

**Configuring Summarization:**

- **Summ Threshold**: Default 85% - adjust to trigger earlier/later
- Lower threshold = more frequent summarization = less context per turn
- Higher threshold = less frequent summarization = more context per turn

**Archive Management:**

- Archived messages automatically deleted after 90 days (default)
- Manual cleanup available via API endpoint
- Full history search works across active and archived messages

**Benefits:**

- Enable indefinite conversations without hitting context limits
- Maintain relationship continuity (persistent message IDs)
- Keep historical conversation searchable after summarization

### Saving Chats

1. Click **Chats (💬)** in the header
2. Click **"Save Session"** (green button)
3. Enter a branch name if creating a fork, or leave default
4. Chat is saved to database and exported to JSON (SillyTavern compatible)

**Branching (Alternate Timelines):**

- Hover any message → Click **blue branch icon (Fork)**
- Enter branch name → Creates independent storyline from that point
- Characters, NPCs, world info, and relationships are copied to new branch
- NPC entity IDs are remapped, so they develop independently

---

## Header Menu Options

### Chats 💬
Manage your chat sessions and branching:

- **Save Session** - Save current chat to database
- **Refresh** - Reload chat list
- **Main Timelines** - View all saved chats
- **Active Branches** - View/manage branches of current chat
- Hover actions: Delete, Rename, Switch between branches

### Characters 👥
Access your character library:

- **New Character** - Open AI-powered character card generator
- **Refresh** - Import new JSON files from `app/data/characters/`
- **Tag Filter Bar** - Quick filter chips + autocomplete for tags
- **Character List** - Unified list showing:
  - **Global** (green badge) - Saved to library, usable across chats
  - **NPC** (gray badge) - Chat-scoped, created mid-chat
  - **Inactive** (gray badge) - Not currently active in chat
  - **Promoted** (orange badge) - NPC promoted to global character
- Hover actions: Edit, Delete, Activate/Deactivate (NPCs only), Promote to Global (NPCs only)

### World Info 🌍
Manage your world lore:

- **Add Entry** - Create new world info entry
- **Refresh** - Import new JSON files from `app/data/worldinfo/`
- **Tag Filter Bar** - Quick filter chips for worlds
- **World List** - Expandable entries showing:
  - **Canon Law** (red badge) - Core rules, always injected + reinforced
  - **Constant** (green badge) - Always included (not semantically triggered)
  - Regular entries (no badge) - Injected via semantic search when relevant
- Hover actions: Edit, Delete, Expand/Collapse

### Gen Card 🪪
AI-powered character card generator:

- **Source Mode:** Current Chat / Manual Input
- **Auto-Generate All** - Creates complete character card from conversation
- **Individual Fields:** Generate Personality, Physical Body, Genre & Tags, Scenario, First Message, Example Dialog separately
- **Save Character Card** - Exports to SillyTavern V2 JSON format

### Gen World 🌍
AI-powered world lore extractor:

- **Source Mode:** Current Chat / Manual Input
- **Analyze All Lore** - Extracts history, locations, creatures, factions
- **Individual Sections:** Generate each lore category separately
- **Finalize & Update** - Saves extracted lore to world card

### Search 🔍
Search messages across all chats:

- **Search Bar** - Enter query text
- **Filters** (expandable):
  - Filter by Chat
  - Filter by Speaker
  - Date Range (From/To)
- **Results** - Show matching messages with:
  - Jump to Message - Navigate to that point in conversation
  - Show Context - View surrounding messages

### Export 📤
Export chats for training (active when chat is saved):

- **Format:** Alpaca / ShareGPT / ChatML
- **Options:**
  - Include system prompts
  - Include world info (Canon Law only)
  - Include NPC responses
- **Export Training Data** - Download JSON file optimized for Unsloth/fine-tuning

### Settings ⚙️
Configure NeuralRP:

**Neural Connection Section:**
- Connection status with Test buttons:
  - **KoboldCpp** (LLM backend)
  - **Stable Diffusion** (Image generation backend)
- Shows current URL and status (green/yellow/red)

**General Settings:**
- **System Prompt** - Override default system instruction
- **Your Name** - User identifier for conversations
- **Your Persona** - User character description
- **Reinforce Character Info Every X Turns** - Default: 5
- **Reinforce World Canon Every X Turns** - Default: 3
- **Enable World Info** - Toggle world lore inclusion
- **Max World Info Entries** - Limit (0 = unlimited)
- **Temperature** - LLM randomness (0.0-2.0, lower = more focused)
- **Max Resp Tokens** - Response length limit
- **Max Context** - Context window size
- **Summ Threshold** - Summarization trigger point (default: 85%)

**Long-Term Memory:**
- Read-only view of current chat summary
- Clear Memory button

**Performance Optimization:**
- **Smart Performance Mode** - Queues heavy operations to prevent VRAM crashes

**World Info Cache:**
- Cache status display (entries, max, usage %, memory)
- Refresh / Clear buttons
- Cache size limit setting

**API Configuration:**
- **Kobold URL** - Backend API endpoint with Update & Test button
- **SD API URL** - Stable Diffusion endpoint with Update & Test button
  - **Required for:** Manual image generation (Vision Panel) AND Snapshots (📸 button)
  - **Snapshot Feature:** Click camera icon to auto-generate scene images from chat context
    - Uses the same resolution settings as manual generation (configure in Vision Panel)
    - In performance mode, automatically drops to 384×384 when chat context is very long (≥15000 tokens)
  - **Red Light Indicator:** 🔴 appears if SD is unavailable (check A1111 is running)

**Data Recovery:**
- **View Change History** - 30-day retention of all changes
- Restores characters, world info, and chats to previous states

---

## Demo Files Setup

NeuralRP includes demo files to help you get started immediately.

### Demo Files Included

**Character:** `demo/Jim the Bartender.json`

- **Name:** Jim the Bartender
- **Role:** Seasoned bartender in a rough-and-tumble town
- **Personality:** Intimidating yet comforting, gentle, kind-hearted
- **Scenario:** Welcoming new faces to his humble establishment

**World:** `demo/exampleworld.json`

- **Entry 0:** Magic Suppression - Magic outlawed for commoners
- **Entry 1:** Rising of New Magic - New magic discovered 50 years ago
- **Entry 2-6:** Cities - Eastoria, Wimville, Charlette, Pastoria, Mildom
- **Entry 7-10:** Creatures - Elves, Dwarves, Humans, Giants
- **Entry 11-12:** Factions - Empire, Rebellion
- **Entry 13-15:** Locations & Elites - Elven Grove, Orcs, Elites/Royals

### Where to Put Demo Files

1. **Copy Character:**
   ```
   demo/Jim the Bartender.json → app/data/characters/Jim the Bartender.json
   ```

2. **Copy World:**
   ```
   demo/exampleworld.json → app/data/worldinfo/exampleworld.json
   ```

### Loading Demo Files

After copying files to the correct folders:

1. **Refresh NeuralRP:**
   - Click Characters (👥) → Click **Refresh** button
   - Click World Info (🌍) → Click **Refresh** button

2. **Demo Character:**
   - Go to Characters panel
   - "Jim the Bartender" appears in character list
   - Click to activate (blue border appears)
   - Start chatting!

3. **Demo World:**
   - Go to World Info panel
   - "exampleworld" appears in world list
   - Expand to view entries (magic, cities, creatures, factions)
   - Entries automatically inject when semantically relevant

**SillyTavern Compatibility:**

Demo files are in SillyTavern V2 format. NeuralRP automatically syncs with JSON files in `app/data/characters/` and `app/data/worldinfo/`. You can edit cards externally, and NeuralRP will import changes on refresh (smart sync v1.8.0).

---

## Tips and Troubleshooting

### Performance Tips

**For 8GB VRAM Users:**

- Enable **Performance Mode** in Settings (queues heavy operations)
- Reduce **GPU Layers** in KoboldCpp (try 20-30 layers instead of all)
- Use smaller models (L3-8B instead of 13B/27B)
- Reduce **Resolution** in image generation (512×512 instead of 768×768)
  - **Note**: Snapshots use the same resolution settings as manual generation
  - **Emergency Preset**: In performance mode, snapshots automatically drop to 384×384 when chat context exceeds 15000 tokens
- Enable **Summarization** at 70-75% instead of 85%

**For 12GB+ VRAM Users:**

- Offload more GPU layers in KoboldCpp (35-50+ layers)
- Use larger models (Tiefighter 13B or Gemma-3-27B)
- Increase **Context Size** in KoboldCpp (4096-8192)
- Generate higher resolution images (768×768 or 1024×1024)
  - **Snapshots** will use your chosen resolution for both manual and auto-generated images

### Common Issues

**"KoboldCpp connection failed" (red status):**

1. Check that KoboldCpp is running
2. Verify URL is correct (default: `http://127.0.0.1:5001`)
3. Check firewall settings (allow Python/KoboldCpp through)
4. Try restarting KoboldCpp and NeuralRP

**"Stable Diffusion connection failed" (red status):**

1. Check that A1111 is running
2. Verify URL is correct (default: `http://127.0.0.1:7860`)
3. Ensure model is loaded in A1111 (check top-left dropdown)
4. Check that `--api` flag is enabled (default in webui-user.bat/webui.sh)

**"Out of Memory" / VRAM Crash:**

1. Enable **Performance Mode** in Settings
2. Reduce GPU layers in KoboldCpp
3. Reduce image resolution or steps
4. Summarize chat to free context space
5. Close other GPU-intensive applications

**Characters responding inconsistently:**

1. Check **Reinforcement Frequency** (default: 5 turns) in Settings
2. Verify character personality and description are well-written
3. Ensure character is active (appears as pill badge in header)
4. Try adjusting Temperature (lower = more focused, higher = more creative)

**World info not injecting:**

1. Verify **Enable World Info** is checked in Settings
2. Check that world info entries have relevant keywords (unquoted for semantic, quoted for exact)
3. Try mentioning the keyword directly in conversation
4. Check **World Info Cache** status in Settings (Refresh if needed)

### Advanced Features

**NPC Creation (Mid-Chat):**

- Click **"Create NPC"** button (green user-plus icon, bottom of chat)
- Enter NPC description (e.g., "Guard Marcus: middle-aged, tired, wants to be anywhere but here")
- AI generates complete character card
- NPC is saved to current chat only (chat-scoped)
- Can be promoted to global character later (Characters panel → Promote icon)

**Tag Management:**

- Characters and worlds support tags for library organization
- **AND Semantics:** Filter by multiple tags (character must have ALL selected tags)
- **Quick Filter Chips:** Top 5 most-used tags appear automatically
- **Autocomplete:** Start typing tag → suggestions appear
- **Automatic Extraction:** SillyTavern card tags preserved on import

**Relationship Tracking:**

- Five emotional dimensions tracked between all entities:
  - Trust / Emotional Bond / Conflict / Power Dynamic / Fear-Anxiety
- Updates automatically via semantic analysis (no manual management)
- Only injected when relationships deviate from neutral AND semantically relevant
- Preserved through summarization and branching

**Smart Sync (v1.8.0):**

- Characters and worlds sync intelligently between database and JSON files
- **Timestamp-based conflict resolution:** Newer version wins
- **World Entry-Level Merging:** Preserves user additions from both sources
- Edit cards in NeuralRP OR externally (SillyTavern/text editor)
- Click **Refresh** in Characters/World Info panel to sync changes

### Keyboard Shortcuts

- **Enter** - Send message
- **Shift + Enter** - New line in input field
- **Ctrl + Z** - Undo (inpaint mask painting)
- **Esc** - Close modals/dialogs

### Getting Help

**Documentation:**

- [Technical Documentation](TECHNICAL.md) - Deep dive into implementation details
- [Reinforcement System](REINFORCEMENT_SYSTEM.md) - Context assembly logic
- [Changelog](CHANGELOG.md) - Full version history

**Community Resources:**

- **KoboldCpp:** [GitHub Issues](https://github.com/LostRuins/koboldcpp/issues) or [Discord](https://discord.gg/koboldai)
- **AUTOMATIC1111:** [GitHub Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) or [Discord](https://discord.gg/stablediffusion)

**Model Resources:**

- **Text Models:** Search "GGUF" on [Hugging Face](https://huggingface.co)
- **Image Models:** [CivitAI](https://civitai.com) or [Hugging Face](https://huggingface.co)
- **Text Models Recommendations:** [Bartowski's HuggingFace](https://huggingface.co/bartowski) (reliable GGUF quantizations)

---

## Next Steps

Now that you're up and running:

1. **Explore Demo Content** - Chat with Jim the Bartender and explore the demo world
2. **Import Your Own Cards** - Drop SillyTavern cards into `app/data/characters/` and `app/data/worldinfo/`
3. **Create Custom Characters** - Use Gen Card to build characters from scratch or extract from conversation
4. **Experiment with Settings** - Adjust reinforcement intervals, temperature, and context size
5. **Generate Images** - Try character tag substitution and inpainting
6. **Branch Stories** - Create alternate timelines and see how relationships diverge

Happy roleplaying! 🎭
