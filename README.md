# NeuralRP

![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)

**tl;dr: NeuralRP is a local, no-cloud, opinionated roleplay engine that keeps long chats from turning into slop by structuring the conversation into clear scenes and aggressively summarizing old chatter, while delivering best-in-class, character-consistent Stable Diffusion images**

**Simple out of the box. Just drop your SillyTavern cards in the folders, there isn't a million different things to configure. Just go.**

**Designed on a 12 GB vRAM GPU running both a text and visual LLM at the same time. No $4,000 hardware required**

**No cloud subscriptions, no inference costs**

**Next step: Open [Quickstart Guide](docs/QUICKSTART.md) for setup, recommended LLM models, and example characters and worlds**

**Status**: Actively developed, v2.0.3 released.

---

## Why It’s Different
Through extensive testing, I have found that smaller LLMs (I roleplay with a 12b LLM) simply cannot keep characters straight, no matter what you prompt, when lots of dialog is present. Characters just merge together. To battle this, I've focused on building a system predicated on smart character and world card insertion, along with lots of auto-summarization:

1. **Context hygiene engine** — Character cards appear in full at strategic reset points (first appearance, turns 1-3 sticky window, after long absence), then lightweight reminders every turn keep voices consistent. World lore appears only when semantically relevant unless marked as "canon law", which reinforces periodically. Recent dialogue stays, older content compresses into scene summaries.

2. **Emergent cast with visual canon** — Quickly create NPCs on the fly that are isolated to one chat, have full parity with characters and can be promoted to full characters. Gender selection keeps both text and visuals targeted, and one-click Danbooru tag assignment based on character description solves the problem of visual drift, so you can stay immersed in the actual roleplay.

3. **Native Stable Diffusion, scene‑aware** — Deep AUTOMATIC1111 integration: inpainting, both manual and automatic generation via the "snapshot" feature, and automatic danbooru tag assignment based on physical features, carefully designed to not "leak" into the context window. A "Favorites" menu allows you to jump right to where the image appeared in its original chat, so you can see the context. All designed to fit into the flow of roleplay, as unintrusively as possible.

---

## NeuralRP: Storytime

When I discovered Huggingface and tuned LLMs for roleplay, I quickly developed a love for experimenting with it. I love writing, and it quickly became a passion to create worlds and characters through written dialog with interesting, creative LLMs.

However, when I started (with Ollama) I noticed something: once I hit about 12k tokens (happens quickly), LLMs completely changed: output became repetitive and garbled, it forgot the situation and world I had carefully detailed, and characters became all the same. Rather than accepting this as a fundamental limitation of small, local LLMs, I decided to move on to other frontends.

Oobabooga was marginally better, but when I tried SillyTavern it opened my eyes to prompt control: you could inject whatever you wanted at set intervals. I began to obsess over dialing in the perfect settings.

What I found was this: I could never get the underlying control I needed because the architecture is fundamentally flawed, and the SD generation extension just wasn't cutting me for me. That needed to be native to work the way I wanted. Not to mention, I was sick of all the knobs. I just wanted to find *my* setting, for *my* 12GB vRAM card that worked for me, rather than being taken out of RP immersion with workarounds and fixes.

Hence, NeuralRP was born.

It's engineered from the ground up with context window in mind. How do you solve the context window problem? By keeping it as small as possible, in the smartest way possible. Intelligent character card usage and aggressive auto-summaries to keep characters from turning into poorage. Uses SQLite and SQLite-vec backend which is a smart, fast home for truth rather than endless JSONs. Syncs intelligently with SillyTavern cards and semantically finds and injects what matters, when it matters, no matter how deep your world is.

I've built up the SD integration to match. Rather than being a bolt-on extension, it's been carefully developed to natively fit into the flow of NeuralRP with a deep feature set, scripted danbooru-optimized prompts, and as automated or manual as you want it to be (or ignore it).

The fact is, there are certain limitations with 7b-14b LLMs that you simply can't get past. But this app maximizes what those LLMs offer for roleplay.

---

## Image Generation with AUTOMATIC1111

Native integration, with the goal to be best in class:

### 1. Snapshots

- **Generated from Context** – Generate images from chat context automatically; AI analyzes the last 20 messages, providing location, action, activity, dress, and facial expression, which is then directly matched to an SD Generation-optimized prompt. 
- **How is it Different?** - It doesn't just ask the LLM for a complete prompt and shove it into A1111 like others do. It takes a specific framework optimized for Anime-based LLM generation models, using Danbooru tags that those models are specifically trained on, to create consistent images. The LLM is just delivering keywords - action, location, activity, dress, and facial expression. And no "leakage" into the context window, one of the things that would ruin my RP chats with other frontends you have probably used.
- **Intelligent analysis** - Snapshot looks across recent chat messages, intelligently gathering context for the snapshot while focusing on the most recent action for the picture.
- **Focus Mode** - More than one character in the shot? Select the one you want using focus mode.
- **You are IN THE SHOT** - Define yourself as male/female/other, and fill out your own tags in settings. You're in the snapshot, with context derived from your tags and the chat text. 

### 2. Inpainting

- Adjustable brush size
- Ctrl+Z undo
- Eraser tool
- Persistent mask between regenerations
- Full A1111 parameter control

### 3. Manual-Mode Generation

- **Pop-out Window** - Easily accessible pop out window, prompt stays in browser memory so you don't have to retype every time, re-size generation on the fly.
- **Character tag substitution** – Assign Danbooru tags to characters once, reference with `[CharacterName]` in prompts for consistent appearance without memorizing tag lists.

### 4. Browse and Save Your Images
- **Images Folder** - separate images folder where every generation is saved, even if you delete it in the chat. 
- **Favorites menu** – Favorite images to build a visual library; click on the image and go straight to the image's original location in the chat. Filter by tag.
- **Jump‑to‑source** – Double‑click favorites to jump back to the exact chat moment they came from.
- **Recreate images** - Image metadata saved, so you can see exactly what prompt generated what image. Easily recreate images with a click of a button.

### 5. Performance‑Aware Presets

Automatic selectable step/resolution reduction when context is large (>12K tokens) to prevent VRAM crashes, with sane defaults for 8–16 GB cards.

---

## Built for SillyTavern Ecosystem Compatibility

NeuralRP was designed to work *with* SillyTavern cards, not replace them.

- **Bi-Directional Smart sync** — Timestamp-based conflict resolution between JSON cards and database
- **Automatic tag preservation** — SillyTavern v2 card tags extracted and stored
- **Forward and backward compatible** — No conversion, no data loss

### Card Generation

Create optimized character cards in two ways:

- **From context** — Generate PList-optimized definitions from conversation history
- **From natural language** — Write plain-text descriptions, NeuralRP converts to PList format (prompt-list, token-optimized structure used for SillyTavern v2 cards)

Both output SillyTavern V2-compatible JSON. Prototype NPCs in conversation, then formalize them into reusable cards.

---

## What This Enables

### Multi-Character Chats That Scale

Optimized for multiple active characters with distinct voices. Capsules (compressed summaries with dialog examples) enable group chats without context overflow.

### Emergent NPCs

Create background characters mid-chat with full personality, automatic Danbooru tagging, and relationship tracking. Promote them to global characters when they matter.

### Semantic World Information

World lore injects automatically when semantically relevant to the conversation:

- **Semantic Search Engine** — sqlite-vec with all-mpnet-base-2 for meaning-based matching
- **Quoted keys** — Exact phrase match for specific terms (e.g., "Great Crash Landing")
- **Unquoted keys** — Semantic matching for concepts (e.g., dragon → dragons, draconic). You can think of quoted keys as exact bookmarks, unquoted as fuzzy search.
- **Canon law** — Enforce your core world rules on a customizable turn basis

### Intelligent Summarization

Continue conversations beyond context limits. When context approaches 80%:

- All but 6 most recent turns traded for summaries
- Relationship states preserved
- Story continuity maintained
- Summarization window to view and customize summaries (or auto-summarize your summaries!)
- Auto-summarization of summaries once they hit 1200 words

---

## Library-Scale Organization

Tag management for character and world cards:

- **AND semantics** — Filter by multiple tags (must have ALL selected)
- **Quick filter chips** — Top 5 most-used tags surface automatically
- **Autocomplete** — Suggests existing tags to prevent bloat
- **Automatic extraction** — SillyTavern V2 card tags preserved on import

---

## Stability and Operations

v2.0.0 adds operational features to support running NeuralRP as a long-term local service, not a script you restart every session.

### Data Safety

- Automatic daily SQLite backups with 30‑day retention and compression.
- Backup APIs to list, create, delete, and restore backups when needed.
- Recovery in minutes if database corruption or bad migrations occur

### Operational Hygiene

- Structured logging with rotation (size and time‑based), and configurable log levels.
- Retention policies for autosaves, unnamed chats, change logs, metrics, and summarized messages.
- Weekly VACUUM to keep the database lean instead of growing without bound.

Together, these keep NeuralRP from slowly turning into a massive mystery file with no visibility into what's going on.

### Monitoring and Configuration

- A `/api/system/status` endpoint with service health, database and disk stats, backup info, and app metadata.
- config.yaml for all settings with NEURALRP_{SECTION}_{KEY} environment variable overrides—no more editing Python source to change behavior

Together, these make NeuralRP something you can run for months without manual housekeeping or mystery failures.


## Additional Capabilities

- **Multi-mode chat** — Narrator (third-person), Focus (first-person), Auto mode that automatically detects who should be talking.
- **Live editing** — AI-generated content is editable, all user edits are saved into the database.
- **Change history** — 30-day retention with browse/restore functionality
- **Soft delete** — Messages archived instead of deleted, searchable across history
- **Export for training** — Export to Alpaca/ShareGPT/ChatML formats for Unsloth
- **Built on SQLite + SQLite-vec** - Unified data system makes it easy to "bolt on" features, tune and mod this to your heart's content. Everything is in a single file (neuralrp.db), easy to back up or inspect.
- **Migration system** — Seamless upgrades (v2 → v3 schemas) without data loss.

---

## Built for Local Deployment

NeuralRP doesn't run the models, it orchestrates them. Here are the backends you need:

- Local LLM via KoboldCpp, TabbyAPI, or Ollama (OpenAI-compatible)
- Local Stable Diffusion via AUTOMATIC1111 WebUI (optional)
- 12-16GB VRAM (8GB minimum with Performance Mode)

All data in SQLite (neuralrp.db) with automatic JSON export for SillyTavern compatibility.

---

## Known Limitations

- Requires 8k+ context models. NeuralRP's architecture won't fit in 4k context windows. Modern 7B-14B RP-tuned models (Mistral, Qwen, Llama 3, etc.) all ship with 8k+ by default.
- No cloud extensibility, all local
- Requires AUTOMATIC1111 for image generation, not compatible with ComfyUI and others at this time (run with --API).
- Requires OpenAI-compatible backend with text to text LLM running locally (KoboldCpp strongly recommended).
- Tested with KoboldCpp with a quantized 12B LLM model tuned for RP, and with A1111 running an Illustrious SD model.
- All testing done with a NVidia 3060 12GB vRAM GPU
- Running 2 LLMs with an 8GB vRAM GPU is untested, will likely lead to sub-optimal results
- Danbooru tagging is optimized for SD models like Pony and Illustrius, which are trained for that input. Other SD models are untested

## Hardware Requirements

**Recommended:**
- 12GB+ VRAM GPU (for running both LLMs at the same time)
- Python 3.8+
- KoboldCpp (LLM inference)
- AUTOMATIC1111 WebUI (image generation)

**Minimum:**
- 8GB VRAM (with Performance Mode), recommended just use 1 model at a time
- Supports: KoboldCpp, Ollama, Tabby (OpenAI-compatible)

---

## Migration Notes for v1.x Users

**Data Changes:**
- **Forking data:** Existing forked chats remain as plain independent chats with no origin linkage
- **Relationship state:** No migration path—relationship tracking is entirely removed

**Behavioral Changes:**
- **Forking workflow:** Use "Save As" with different names or summary-based new chats
- **Regeneration:** "Regenerate" button now uses standard snapshot mode (natural SD seed variations)

**What's Preserved:**
- All chat history, characters, NPCs, world info, settings
- Snapshot generation, inpainting, favorites, visual canon
- NPC functionality under unified storage system

## Quick Start

```bash
git clone https://github.com/neuralrp/neuralrp.git
cd neuralrp
launcher.bat    # Windows - handles everything automatically
```

Or manually:
```bash
pip install -r requirements.txt
python main.py
```

Navigate to http://localhost:8000

Configure LLM and image generation endpoints in Settings panel.

---

## Danbooru Tag Generator (Optional)

To enable one-click Danbooru character visual matching, run fetch script to download character data from Danbooru API:

```bash
# Edit app/fetch_danbooru_characters.py with your Danbooru API key, then:
python app/fetch_danbooru_characters.py
```

This generates `app/data/danbooru/book1.xlsx` from Danbooru's public API (~15-30 minutes for 1394 characters). See **[User Guide](docs/USER_GUIDE.md)** for detailed setup instructions including API key acquisition.

---

## Documentation

- **[Quickstart Guide](docs/QUICKSTART.md)** — Get up and running fast (installation + first-time setup + basic usage)
- **[User Guide](docs/USER_GUIDE.md)** — Complete feature manual (all features, settings, and capabilities)
- **[Technical Documentation](docs/TECHNICAL.md)** — Deep dive into architecture and implementation
- **[Changelog](CHANGELOG.md)** — Version history

---

## Credits

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.
