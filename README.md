# NeuralRP

NeuralRP is an opinionated roleplay interface for local LLMs. It's built for users who already understand the basics — temperature, context length, model choice — and want a fast, coherent RP experience without managing dozens of knobs or extensions. Optimized for modest GPUs, NeuralRP integrates KoboldCpp and Stable Diffusion WebUI (AUTOMATIC1111) directly, applies strong narrative defaults, and turns live roleplay sessions into portable, SillyTavern-compatible character and world cards.

---

## Overview

This section covers what NeuralRP does, who it's for, and how to use it.

### Design Philosophy

NeuralRP is designed for experienced RP users who want:
- **Speed over tweakability** – Strong defaults, fewer configuration knobs, faster setup.
- **Coherence over flexibility** – Opinionated prompt structure that maintains character voices and world consistency.
- **Efficiency on constrained hardware** – Built to run text generation and image generation together on modest GPUs (like 12GB cards) without requiring massive context windows or cloud infrastructure.
- **Portability** – Generated JSON cards are optimized with PList and plain text that follow best practices for SillyTavern and other tools.
- **Strong visual support** -  Customizable per‑character image prompts and inline generations that avoid leaking text into the chat context while storing images in a dedicated folder.

If you're looking for a generic LLM dashboard with every possible parameter exposed, this isn't it. NeuralRP focuses on fast, hardware-efficient roleplay workflows with minimal friction.

### Key Differentiators

What makes NeuralRP different from other RP frontends:

1. **Dual-Source Card Generation**  
   Create character cards and world info from either chat history OR plain-text descriptions. Most tools only do one or the other.

2. **Per-Character Visual Canon (Danbooru Tags)**  
   Assign tags to each character once; use `[CharacterName]` in image prompts forever. Consistent character appearance without rewriting prompts each time.

3. **Canon Law + Scalable World Info**  
   Mark critical lore as "Canon Law" (always included, never capped). Regular entries use keyword matching with configurable caps and probability weighting, so huge worlds stay fast.

4. **Auto Mode with Smart Speaker Selection**
   Let the AI decide who speaks (narrator or which character) turn-by-turn based on context, using compressed capsule personas to keep every voice distinct.

5. **Small-GPU friendly design**
   Prompt structure, automatic summarization, and capped World Info are tuned so you can run a text model and an image model together locally on 12GB cards without massive context windows.

6. **Dedicated Narrator Mode**
   Built-in third-person narrator mode where the AI acts as game master and storyteller, separate from character voices, without needing custom cards or prompt hacks.

---

## Features

### Core Functionality

- **KoboldCpp Integration**  
  Seamlessly connect to a local KoboldCpp server for text generation (native or OpenAI-compatible API).

- **Stable Diffusion (A1111) Integration**  
  Generate images inline with chat conversations using Stable Diffusion WebUI (AUTOMATIC1111).

- **SillyTavern Compatible**  
  Accepts character cards and World Info in SillyTavern V2 JSON format, and saves new cards back out in the same format.

- **Narrator Mode**  
  Chat with characters, or select no characters for a pure third-person narrator experience in any world.

- **Multi-Character Support**  
  Chat with multiple characters in the same scene. NeuralRP uses capsule personas to keep voices distinct without bloating the prompt.

- **Branching Chat Sessions (v1.1)**  
  Fork from any message to create independent branches with their own chat files. Each branch records its origin chat and message, supports rename/delete, and appears in a dedicated branch management UI for easy navigation between alternate timelines.

### Card & World Factory

- **AI-Powered Character Creation (From Chat or Manual Text)**  
  Generate new character cards either from the current chat history or from plain-text descriptions. Start from a loose idea, roleplay it out, or paste a concept paragraph, then convert it into a structured SillyTavern-compatible character card using PList-style fields.

- **AI-Powered World Building (From Chat or Manual Text)**  
  Create World Info entries (history, locations, creatures, factions) from conversations or manually supplied lore text. NeuralRP turns freeform worldbuilding into reusable World Info JSON.

- **Automatic Formatting and Quotes**  
  The app handles field structure, escaping, and pulling out example dialogues and first messages, so you don't have to hand-edit JSON before importing into SillyTavern.

- **Live Editing Before Save (v1.1)**  
  All generated PList text appears in editable textboxes. You can tweak, partially rewrite, or completely replace the LLM output before saving the character card or world file.

- **Use as a Dedicated Card Factory**  
  Even if you prefer to play in SillyTavern, you can use NeuralRP as a focused environment to generate and refine character/world cards, then import them into other frontends.

### Advanced Features

- **Automatic Summarization**  
  When context usage reaches ~85% of the model's max, older messages are summarized so their content remains accessible to the LLM after the raw window is exceeded.This number is configurable in settings.

- **Token Counter**  
  Monitor token usage for the assembled prompt in real time.

- **Image Generation with Character Tags**  
  Assign a Danbooru-style tag to each character. Use `[CharacterName]` in the positive prompt and NeuralRP will expand it to the configured tag for consistent A1111 generations, even when you're not actively chatting with that character.

- **Chat Persistence**  
  Save and load previous chat sessions, including associated characters, world, and images.

- **Canon Law System**  
  Mark important World Info entries as **Canon Law** (click to toggle, entries highlight red). Canon Law entries are always injected and prioritized in the context to reduce world drift.

- **Efficient World Info for Large Worlds (v1.1)**  
  World Info is now optimized for big settings:
  - Cached world-info structures to avoid reprocessing on every request.  
  - One-time lowercase preprocessing of keys for faster, case-insensitive matching.  
  - Configurable cap on the number of regular entries included per turn to prevent prompt bloat, while Canon Law entries remain uncapped and always included.  
  - Optional probability weighting (`useProbability` + `probability`) to allow some entries to appear stochastically instead of every time they trigger.  
  - A global "Enable World Info" toggle in Settings for sessions where you want maximum speed or rules-light play.

- **Automatic Performance Mode (v1.2)**  
  Smart GPU resource management when running LLM + Stable Diffusion together. Queues heavy operations while allowing quick tasks to proceed, automatically adjusts SD quality under load, and provides context-aware hints. Includes rolling median performance tracking to detect contention and a master toggle to enable/disable the entire system.

### Customization Options

NeuralRP keeps the UI minimal but exposes the essentials:

- **System Prompt**  
  Customize the global behavior and style (narrator vs character focus, tone, formatting).

- **Temperature Control**  
  Adjust creativity (0.0–1.0+).

- **Reply Length**  
  Set maximum response length in tokens.

- **User Persona**  
  Define your user/player persona, used as part of the context.

- **Context Reinforcement**  
  Control how often character cards or narrator prompts are re-inserted into the context window.

### Mode Selection

- **Narrator Mode**  
  Third-person cinematic narration; the narrator may move all characters and NPCs.

- **Focus Mode**  
  Speak as a specific character in first person; the hidden prompt constrains the LLM to that character's voice.

- **Auto Mode**  
  Let the LLM automatically select whether to respond as narrator or a specific character based on context, for a more natural roleplay flow.

---

## Installation

### Prerequisites

- Python 3.8 or higher  
- [KoboldCpp](https://github.com/LostRuins/koboldcpp) running locally (e.g. `http://127.0.0.1:5001`) with the API enabled  
- [Stable Diffusion WebUI (AUTOMATIC1111)](https://github.com/AUTOMATIC1111/stable-diffusion-webui) running locally (e.g. `http://127.0.0.1:7861`) for image generation

### Quick Start (Windows)

1. Double-click `launcher.bat` to run the application.  
   - The launcher will automatically check for and install dependencies.  
   - On first run, this may take a few minutes.

2. Open your browser and navigate to:  
   `http://localhost:8000`

### Manual Installation

1. Clone or download this repository.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

---

## Configuration

### API Endpoints

By default, the app expects:
- KoboldCpp API: `http://127.0.0.1:5001`
- Stable Diffusion WebUI API: `http://127.0.0.1:7861`

To change these, modify the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "kobold_url": "http://127.0.0.1:5001",
    "sd_url": "http://127.0.0.1:7861",
    # ...
}
```

---

## Usage

### Basic Chat

1. Load or create a character card (optional) and select a world.
2. Adjust settings as needed (temperature, reply length, etc.).
3. Start chatting in the main interface in:
   - **Narrator mode**, or
   - **Focus mode** for a specific character, or
   - **Auto mode**.

### Multi-Character Chat

1. Load two or more characters into the session.
2. Use the mode selector to choose:
   - **Narrator** – Narrator controls the scene and any characters.
   - **Focus: [Name]** – Speak as a specific character in first person.
   - **Auto** – Let the LLM decide who should respond / narrate based on context.

NeuralRP automatically uses capsule/summary personas in multi-character mode to keep definitions short and distinct.

### Image Generation

1. In the image prompt, use `[CharacterName]` to reference a character.
2. The app will automatically expand `[CharacterName]` to the Danbooru tag assigned to that character.
3. Images are inserted inline into the chat and saved under `app/images/`.

### World Info

1. Navigate to the World Info tab.
2. Load or create World Info files in SillyTavern-compatible JSON format.
3. Click entries to mark them as **Canon Law** (they turn red).
4. Use the "Enable World Info" toggle in Settings to turn lore injection on or off for the current session.

Regular entries are matched via case-insensitive keyword search against the last few messages, with a configurable maximum number of entries per turn and optional probability weighting; Canon Law entries are always included.

### Creating Characters (Gen Card)

1. Open the **Gen Card** tab for your current session.
2. Choose a source mode:
   - **Current Chat** – Analyze recent conversation to infer the character's description, personality, and example lines.
   - **Manual Input** – Paste a plain-text description of the character instead of using chat history.
3. Click **Generate** to create PList-style character data.
4. Review and edit the generated text in the editable textbox.
5. Click **Save** to write a SillyTavern-compatible character card file.

### Creating World Info (Gen World)

1. Open the **Gen World** tab.
2. Choose a source mode:
   - **Current Chat** – Analyze recent world interactions and exposition.
   - **Manual Input** – Paste your own lore text or notes.
3. Click **Generate** to extract structured World Info entries (history, locations, creatures, factions) into PList-style text.
4. Edit the generated content as needed.
5. Save entries into your world's World Info file.

### Branching and Forks (v1.1)

1. In any chat, click the fork icon for a specific message to open the **Fork Dialog**.
2. Accept the auto-generated branch name (based on message content and timestamp) or provide a custom name.
3. Confirm to create a new branch; a separate chat file is created starting from that message, inheriting the current characters, world, and settings.
4. Use the **Branch Management** dialog or sidebar sections to:
   - View all branches and their origins.
   - Rename or delete branches.
   - Switch between the main timeline and any branch.

---

## Data & File Structure

All saved content lives under `app/data/` and can be backed up, synced, or version-controlled:

- **Characters** → `app/data/characters/` (SillyTavern V2 JSON format)
- **World Info** → `app/data/worldinfo/` (SillyTavern-compatible JSON)
- **Chats** (including branches) → `app/data/chats/` (JSON with messages + metadata)
- **Images** → `app/images/` (PNG files from Stable Diffusion)

---

## Project Structure

```
neuralrp/
├── main.py                 # Main FastAPI application
├── launcher.bat            # Windows launcher script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── app/
│   ├── index.html          # Frontend interface
│   ├── data/
│   │   ├── characters/     # Character card storage
│   │   ├── chats/          # Saved chat sessions (including branches)
│   │   └── worldinfo/      # World Info entries
│   └── images/             # Generated images
└── frontend/               # Frontend build files
```

---

## Requirements

```
fastapi
uvicorn[standard]
httpx
pydantic
```

---

## Troubleshooting

### "Python is not installed"
- Install Python 3.8+ from [https://www.python.org/](https://www.python.org/)
- Make sure to check "Add Python to PATH" during installation

### "Failed to install dependencies"
- Ensure you have a stable internet connection
- Try running `pip install --upgrade pip` first
- Check if you're behind a corporate firewall that blocks pip

### Can't connect to KoboldCpp/SD
- Ensure KoboldCpp is running on port 5001
- Ensure Stable Diffusion WebUI is running on port 7861
- Check that `--api` flag is enabled in KoboldCpp
- Check that `--api` flag is enabled in Stable Diffusion WebUI

### Images not generating
- Verify Stable Diffusion is running and accessible
- Check that the image directory (`app/images/`) exists and is writable
- Review the server logs for specific error messages

---

## Contributing

This project is designed to be extended and customized. Feel free to fork, modify, and improve it for your needs.

---

## License

This project is provided as-is for educational and personal use.

---

## Credits

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Frontend integration with vanilla JavaScript
- Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) card formats
- Integrates with [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## Technical Notes

This section documents implementation details, design decisions, and internal mechanics. It serves as a running log of what's been built and how it works.

### Context Assembly

On each generation, NeuralRP builds the prompt in layers to maintain stable structure and predictable behavior:

1. **System and Mode Instructions**
   - Global system prompt sets narrator vs character focus, tone, and formatting rules.
   - Mode-specific instructions (Narrator / Focus / Auto) tell the model how to respond:
     - **Narrator**: Third-person, omniscient narration controlling all characters and NPCs.
     - **Focus**: First-person voice locked to a specific character's persona.
     - **Auto**: Let the model decide which character should speak or narrate based on context.

2. **User Persona (Optional)**
   - A short description of the player/user is injected so the model can keep your POV and preferences in mind.
   - Placed early in the prompt to influence tone and perspective throughout.

3. **World Info (Keyword-Triggered + Canon Law)**
   - If World Info is enabled:
     - **Canon Law entries** are collected first (always included, never capped, never probabilistic).
     - **Regular entries** are matched via case-insensitive keyword search on the last 5 messages:
       - Configurable cap on the number of regular entries per turn (default: 10).
       - Optional probability weighting (`useProbability` + `probability` fields) allows entries to appear stochastically.
   - **Placement**:
     - Regular entries are inserted after system + mode + persona, before character definitions.
     - Canon Law entries are injected near the end of the prompt to override drift and restate core facts.

4. **Character Definitions**
   - **Single-character**: Full character card description and personality.
   - **Multi-character**: Short "capsule personas" (1-2 sentences + key traits + speech style) to avoid prompt bloat.
   - **Focus mode**: Selected character's persona is emphasized so the model speaks in that voice.
   - **Narrator mode**: Character cards are present but narration is instructed to be third-person.

5. **Conversation History and Memory**
   - Recent messages are included verbatim.
   - **Automatic summarization** triggers when total context approaches ~85% of the model's limit:
     - Oldest 10 messages are removed and distilled into a summary.
     - Summary is prepended to existing summary field and kept alongside latest turns.
     - This allows the model to reference past events without exceeding context window.

6. **Generation Lead-In**
   - A short final instruction sets the expected format (e.g., "Continue the roleplay…").
   - Followed by the user's latest message.

**Result**: This layered approach keeps Canon Law and core instructions stable, limits lore spam via caps/probabilities, and allows indefinite session length through automatic summarization.

### World Info Engine

**Keyword Matching (Not Semantic Search)**
NeuralRP uses pure keyword matching, not vector embeddings or semantic search. This keeps the system simple, fast, and predictable.

**How it works:**
- Collect the last 5 messages and lowercase them: `recent_text`.
- For each World Info entry, check if any of its keywords (also lowercased) appear in `recent_text` using substring matching: `any(k.lower() in recent_text for k in keys)`.
- If a match is found and the entry passes probability checks (if enabled), add its content to the prompt.

**Caching System (v1.1)**
To avoid reprocessing the same world info repeatedly:
- **Cache key structure**:
  ```python
  f"{str(world_info.get('entries', {}))}_{recent_text.lower()}_{max_entries}"
  ```
  - `str(world_info.get('entries', {}))`: String representation of entire world info entries dict.
  - `recent_text.lower()`: Lowercase version of last 5 messages.
  - `max_entries`: Current `max_world_info_entries` setting (default 10).
- **Cache storage**: Global in-memory dictionary `WORLD_INFO_CACHE` in `main.py`.
- **Cache lifetime**: Session-based; exists only while server is running. Never explicitly cleared, so it grows during uptime and is lost on restart.
- **Cache invalidation**: None explicitly implemented. If world info JSON changes, the cache key will differ due to the entries dict being part of the key.

**Case-Insensitive Preprocessing (v1.1)**
All keywords are lowercased once during world info loading via `preprocess_world_info()`.
- Eliminates repeated `.lower()` calls during keyword matching (~10-20% performance improvement).

**Entry Cap Logic (v1.1)**
When regular entries exceed `max_entries` (default: 10):
- The engine keeps the **first N matching entries** found while iterating through `world_info["entries"]`.
- Since Python 3.7+, dict iteration follows insertion order, so this effectively keeps the first 10 entries defined in the JSON file that match keywords.
- **Exception**: Canon Law entries (`is_canon_law: true`) are always included and do not count toward the cap.

**Probability Weighting (v1.1)**
For entries with `useProbability: true`:
- **Random roll**: Done once per entry per generation inside `get_cached_world_entries()`.
- **Roll logic**: `if random.random() * 100 > probability: skip_entry`
- **Failure result**: If an entry fails the roll, it is skipped for that turn even if its keywords match. It will not be added to `triggered_lore`.
- **Purpose**: Allows some lore to appear stochastically instead of every time it triggers, creating variation in large worlds.

**Canon Law System**
- **Definition**: World Info entries marked with `is_canon_law: true` in the JSON.
- **Behavior**:
  - Always collected, regardless of keyword matching.
  - Never subject to entry caps or probability rolls.
  - Injected near the end of the prompt (after regular world info, after character definitions) to override drift and restate immutable facts.
- **UI**: Entries marked as Canon Law are highlighted red in the World Info editor; click to toggle.
- **Enable/Disable Toggle (v1.1)**: A global "Enable World Info" checkbox in Settings.
  - When disabled, no world info (regular or Canon Law) is injected into prompts.
  - Useful for "rules-light" sessions or maximum speed with low-VRAM models.

### Branching System

**Branch Creation**
When a user forks from a specific message:
- **New chat file** is created with filename format:
  ```
  {origin_chat_name}_fork_{timestamp}.json
  ```
  Where `timestamp` is `int(time.time())` (Unix epoch).
- **Origin metadata** is stored in the branch file's top-level `metadata` object:
  ```json
  "metadata": {
      "origin_chat_id": "original_filename",
      "origin_message_id": 12345678,
      "branch_name": "Custom Branch Name",
      "created_at": 1736279867.123
  }
  ```
- **Chat state inheritance**: The new branch starts from the selected message and inherits:
  - All messages up to and including the fork point.
  - Current characters loaded.
  - World info file.
  - Settings (temperature, reply length, etc.).

**Branch Independence**
Each branch is a completely separate chat file.
- No shared mutable state between branches.
- Editing one branch does not affect any other branch or the main timeline.
- Branches can be deleted without affecting the origin chat.

**Branch Management**
- **API Endpoints**:
  - `POST /api/chats/fork` – Creates a new branch from a specific message.
  - `GET /api/chats/{name}/branches` – Lists all branches that originated from a chat.
  - `GET /api/chats/{name}/origin` – Gets origin information for a branch chat.
  - `PUT /api/chats/{name}/rename-branch` – Renames a branch.
- **UI Features**:
  - **Fork Dialog**: Modal for creating branches with auto-generated names based on message content and timestamp.
  - **Branch Management Dialog**: View and manage all branches for the current chat.
  - **Rename Branch Dialog**: Custom naming for branches.
  - **Sidebar Sections**:
    - Main Timelines: Shows original chats and branches with visual indicators.
    - Active Branches: Lists branches for the currently loaded chat.

**Design Decision: No Merge Semantics**
Branches are independent timelines, not branches that merge back together. This keeps the mental model simple:
- "Fork = create a new what-if timeline."
- "Switch = load that timeline."
- No automatic merging, rebasing, or graph-level conflict resolution.

### Memory & Summarization

**Automatic Summarization at 85%**
When context usage approaches the model's limit:
- **Trigger condition**:
  - Total tokens > 85% of model's max context window.
  - AND messages > 10 (minimum history required for summarization).
- **Process**:
  - Take oldest 10 messages verbatim.
  - Send them to the LLM to create a concise summary.
  - Prepend this new summary to the existing `summary` field in the chat file.
  - Remove those 10 messages from the active chat history.
  - Keep the rest of the messages verbatim.
- **Summary storage**:
  - Stored as a top-level `summary` field in the chat JSON file.
  - Used during prompt assembly (Section 3: LONG-TERM CONTEXT).
  - Not shown in the UI; only affects what the model sees.

**What Gets Kept vs. Summarized**
- **Verbatim retention**: Most recent messages up to ~85% of context budget.
- **Summarization threshold**: When limit is reached, oldest 10 messages are compressed.
- **Result**: Sessions can run indefinitely without losing narrative coherence or exceeding context windows.

**No UI Indicator (Currently)**
Summarization happens transparently in the backend.
- Users see the full chat history in the UI, but the model only sees recent messages + summary.
- **Future enhancement**: Add a subtle indicator when summarization has occurred (e.g., "(Summary active)" badge).

### Card & World Generation

**Dual-Source Mode (v1.1)**
Both character and world generation now support two source modes:
- **Current Chat** (`source_mode: "chat"`):
  - Frontend sends last 20-50 messages from active chat.
  - Backend analyzes conversation for character traits, world lore, etc.
- **Manual Input** (`source_mode: "manual"`):
  - Frontend sends text from a manual input textarea.
  - User pastes plain-text descriptions, notes, or concepts.
- **Backend parameter**: `source_mode` is a string in the request. Backend treats both modes as `req.context` and formats the extraction prompt accordingly.

**PList Extraction Prompts**
NeuralRP uses specific PList field structures for structured generation:
- **For Character Personality**:
  ```
  "Convert them into a PList personality array format. Use this exact format:
  [{char_name}'s Personality= "trait1", "trait2", "trait3", ...]"
  ```
- **For World Info (Locations)**:
  ```
  "[LocationName(nickname if any): type(room/town/area), features(physical details),
  atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]"
  ```
- Similar formats exist for:
  - Character descriptions
  - Example dialogues
  - First messages
  - World history, factions, creatures, etc.

**Live Editing Before Save (v1.1)**
All generated PList text appears in editable textboxes in the UI.
- Users can tweak, partially rewrite, or completely replace the LLM output.
- "Save" button captures current textbox state, not original generation.
- This makes NeuralRP viable as a primary card authoring tool, not just a helper.

**Export Format**
- **Character cards**: SillyTavern V2 JSON format (compatible with CharacterHub, Pygmalion booru, etc.).
- **World Info**: SillyTavern-compatible JSON with `entries` object containing keyword-triggered lore.

### Multi-Character Support

**Capsule Persona Compression**
To avoid prompt bloat with multiple characters, NeuralRP uses "capsule summaries":
- **Format**:
  ```
  Name: [Name].
  Role: [1 sentence role/situation].
  Key traits: [3-5 comma-separated personality traits].
  Speech style: [short/long, formal/casual, any verbal tics].
  Example line: "[One characteristic quote from descriptions]"
  ```
- **Usage**:
  - When 2+ characters are active, capsules are used in the prompt instead of full `description` fields.
  - Keeps each character distinct while saving ~80% of tokens per character.
- **Generation**:
  - Can be manually triggered via "Generate Capsule" button in character editor.
  - Auto-generated the first time a character enters a group chat (if not already defined).

**Auto Mode Speaker Selection**
In Auto mode:
- The model receives all active character capsules.
- The prompt instructs: "Choose the most appropriate character to respond, or narrate in third person if needed."
- No hard constraints; the model decides turn-by-turn who speaks.
- **Why it works**:
  - Capsule format includes speech style and example lines.
  - Recent conversation context shows who last spoke.
  - Model naturally follows conversational flow based on cues.

### Image Generation with Character Tags

**Danbooru Tag System**
Per-character visual canon:
- Each character can be assigned a Danbooru-style tag string (e.g., `"1girl, long_hair, blue_eyes, red_dress"`).
- Stored in character card JSON under a `danbooru_tag` or similar field.
- **Prompt expansion**:
  - User types: `[CharacterName] sitting in a cafe`
  - NeuralRP expands to: `1girl, long_hair, blue_eyes, red_dress, sitting in a cafe`
  - Sent to Stable Diffusion WebUI API.
- **Decoupled from chat**:
  - Tags work even when character isn't in the active chat.
  - Useful for generating reference images during world-building or card creation.

**A1111 Integration**
- **API endpoint**: `http://127.0.0.1:7861/sdapi/v1/txt2img`
- **Request format**: Standard A1111 API payload with `prompt`, `negative_prompt`, `steps`, `cfg_scale`, etc.
- **Response**: Base64-encoded PNG, decoded and saved to `app/images/`.

**Image Persistence**
- Generated images are saved with timestamp-based filenames.
- Inline display in chat UI via `<img>` tags.
- Chat JSON stores image filenames so they reload with the session.

### Dynamic Resource Management (v1.2)

**Overview**
When running both LLM (KoboldCpp) and Stable Diffusion on the same GPU, resource contention can cause slowdowns, timeouts, or out-of-memory errors. NeuralRP's performance mode intelligently manages these operations using async queuing and statistical analysis to maintain responsiveness without requiring manual intervention.

**Resource Manager Class**
Coordinates GPU access using `asyncio.Lock()` for thread-safe operation:
- **Operation classification**:
  - Light: Text generation with small contexts, status checks, UI updates
  - Heavy: Image generation, large context text, card/world generation
- **Queuing behavior**:
  - Heavy operations queue when another heavy op is in progress
  - Light operations bypass the queue and proceed immediately
  - Prevents deadlock through careful lock acquisition/release patterns
- **State tracking**: Maintains `active_llm`, `active_sd`, and queue status for real-time monitoring
- **Status reporting**: Returns `idle`, `running`, or `queued` for both text and image operations

**Performance Tracker**
Maintains rolling median timing using `collections.deque(maxlen=10)`:
- **Statistical method**: `statistics.median()` ignores outliers (cold starts, system spikes)
- **Per-operation tracking**: Separate medians for LLM and SD operations
- **Contention detection formula**: Flags resource conflict when:
(current_sd_time > 3× median_sd_time) AND (context_tokens > 8000)

- **Memory bounded**: Fixed-size rolling window prevents unbounded growth

**SD Context-Aware Presets**
Three-tier optimization automatically selected based on story length:

```python
SD_PRESETS = {
  "normal": {"steps": 20, "width": 512, "height": 512, "threshold": 0},
  "light": {"steps": 15, "width": 384, "height": 384, "threshold": 8000},
  "emergency": {"steps": 10, "width": 256, "height": 256, "threshold": 12000}
}
Automatic selection: select_sd_preset() checks current context token count

Threshold behavior:

0-7999 tokens: Normal quality (512×512, 20 steps)

8000-11999 tokens: Light quality (384×384, 15 steps) to free VRAM for text

12000+ tokens: Emergency quality (256×256, 10 steps) to avoid conflicts
```

**Smart Hint Engine**
Context-aware suggestions triggered by performance metrics:
- **Contention hints**: When SD timing exceeds 3× median with large context
- **Quality hints**: When emergency preset is active
- **Optimization tips**: Actionable suggestions (reduce context reinforcement, generate images outside chat)
- **Non-intrusive**: Dismissible notifications, no repetition

**API Endpoints**
RESTful interface for status monitoring and control:
- **GET /api/performance/status**: Returns current operation states and queue depth
- **POST /api/performance/toggle**: Enable/disable performance mode
- **GET /api/performance/hints**: Fetch current contextual hints
- **Error handling**: Graceful degradation when performance mode is disabled

**Frontend Components**
Real-time status display with automatic updates:
- **Settings toggle**: "Automatic performance mode (recommended when running LLM + SD on same GPU)"
- **Persistence**: Settings persisted via localStorage
- **Backend sync**: Immediate synchronization via togglePerformanceMode()
- **Status polling**: updatePerformanceStatus() checks backend state every 2 seconds
- **Status indicators**: Simple idle/running/queued badges for Text and Images
- **Hint display**: Contextual tips with dismiss functionality
- **Conditional UI**: Status and hints only visible when performance mode is enabled

**Thread Safety and Async Design**
- **Async locking**: asyncio.Lock() prevents race conditions between concurrent operations
- **Lock patterns**: Acquire → execute → release with exception handling
- **Deadlock prevention**: Light operations never acquire heavy locks; heavy operations use timeout patterns

**Production Readiness**
Automatically optimizes for users with:
- **Large story contexts**: 12K+ tokens
- **Single-GPU setups**: Running both LLM and SD
- **Heavy workloads**: Image generation workloads
- **Multiple operations**: Concurrent operations

**Performance Characteristics**
- **Memory overhead**: Fixed at ~160 bytes per operation type (10-element deque)
- **CPU overhead**: Median calculation is O(n log n) but only runs on 10 elements
- **Latency impact**: Lock contention adds <1ms for light operations
- **Cache behavior**: No persistent cache; all tracking is session-based

---

## Design Decisions & Tradeoffs

### Why Keyword Matching Instead of Semantic Search?
**Pros:**
- Simple, fast, predictable.
- No embedding model dependencies.
- Users can debug ("Why didn't this entry trigger? Check the keywords.").

**Cons:**
- No fuzzy matching or concept-based retrieval.
- Short keywords can cause false positives (e.g., "ai" matches "Maine").

**Mitigation:**
- Case-insensitive matching reduces some noise.
- Canon Law system ensures critical lore always appears.
- Entry caps prevent prompt bloat even with many false positives.

**Future enhancement**: Optional whole-word/regex matching for single-word keys.

### Why In-Memory Caching Instead of Persistent Cache?
**Pros:**
- Zero disk I/O overhead.
- Automatically cleared on server restart (no stale cache issues).
- Simple implementation (one global dict).

**Cons:**
- Cache grows indefinitely during server uptime.
- Lost on restart (but rebuilds quickly on first use).

**Mitigation:**
- For local desktop use, server restarts are frequent enough that memory leaks are rare.
- If needed, could add a manual "Clear Cache" API endpoint.

### Why Branches as Separate Files Instead of Single File with Graph?
**Pros:**
- Each branch is a fully independent, portable JSON file.
- No risk of corruption affecting multiple branches.
- Easy to back up, version control, or share individual branches.
- Simpler mental model: "one file = one timeline."

**Cons:**
- More files to manage for power users with many branches.
- No single-file "view all branches" graph (would require aggregation).

**Mitigation:**
- Branch management UI aggregates metadata across files for easy navigation.
- Origin metadata enables future timeline visualization without restructuring storage.

### Why 85% Summarization Threshold?
**Reasoning:**
- Below 85%: Still room for model to breathe, no urgency.
- At 85%: Enough headroom to generate a summary and insert it without exceeding limit.
- Above 85%: Risk of abrupt truncation if context fills during generation.

**Tuning:**
- Configurable in settings under "Summ Threshold"
- 85% aligns with Anthropic's Claude Code approach and community best practices.

### Why Async Locking Instead of Process-Level Resource Management?
**Pros:**
- Works within single-process FastAPI server (no multi-process coordination needed)
- `asyncio.Lock()` is lightweight and fast (<1ms overhead)
- Proper deadlock prevention through careful lock patterns
- Clean integration with existing async/await codebase

**Cons:**
- Only protects within NeuralRP process; doesn't prevent external tools from overloading GPU
- No cross-process coordination if running multiple NeuralRP instances

**Mitigation:**
- NeuralRP is designed for single-user local desktop use where one instance is typical
- Performance tracker detects contention from external sources via timing analysis
- Hints guide users to close other GPU-intensive applications when detected

---

## Future Considerations

### Potential Enhancements

**Timeline Visualization**
- Read-only branch map showing origin relationships.
- Use React Flow or similar to render conversation tree.

**Whole-Word Keyword Matching**
- Add regex-based word boundary checks for single-word keys.
- Reduce false positives without moving to semantic search.

**Persistent Cache with Invalidation**
- Move from in-memory to disk-based cache (SQLite or similar).
- Invalidate on world info file mtime change.

**Multi-Tier Memory System**
- Short-term: Last 10 turns verbatim.
- Medium-term: Turns 11-60 as bullet-point summaries.
- Long-term: Semantic search over all older content.

**UI Indicator for Summarization**
- Show "(Summary active)" badge when 85% threshold has triggered.
- Optional: Display summary text in a collapsible section.

### Known Limitations

- No merge semantics for branches: Branches are independent; no automatic reconciliation.
- No vector/semantic search: World info is keyword-only; no fuzzy concept matching.
- In-memory cache growth: Cache grows unbounded until server restart.
- No UI for summarization visibility: Users don't see when summarization happens.

---

## Version History

### v1.1 (Current)
- Branching system: Fork from any message, independent chat files, origin metadata, branch management UI.
- Dual-source card/world generation: From chat or manual text input, with live editing before save.
- Efficient World Info: Caching, case-insensitive preprocessing, configurable caps, probability weighting, enable/disable toggle.

### v1.0 (Initial Release)
- KoboldCpp and Stable Diffusion integration.
- SillyTavern-compatible character cards and World Info.
- Narrator/Focus/Auto modes.
- Multi-character support with capsule personas.
- Automatic summarization at 85% context.
- Canon Law system for immutable world facts.
- Danbooru character tagging for consistent image generation.
- Automatic summarization at 85% context.
- In-memory cache growth: Cache grows unbounded until server restart.
