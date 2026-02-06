# NeuralRP

![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)

**tl;dr: NeuralRP is a local, no-cloud, opinionated roleplay engine, designed specifically to keep dialog from becoming slop by aggressively managing the context window, while keeping image generation best in class.**

**Simple out of the box. Just drop your SillyTavern cards in the folders, there isn't a million different things to configure. Just go.**

**Runs well on 8-16GB  GPU's, no $4,000 hardware required**

**No cloud subscriptions, no inference costs**

**Next step: Open [Quickstart Guide](docs/QUICKSTART.md) for setup, recommended LLM models, and example characters and worlds**

**Status**: Actively developed, v1.11.0 is still under testing. Expect bugs. Tell me about any you find in Discussions!
---

## Table of Contents

- [Why It's Different](#why-its-different)
- [NeuralRP: Storytime](#neuralrp-storytime)
- [Core Philosophy: Conversation First](#core-philosophy-conversation-first)
- [Built for SillyTavern Ecosystem Compatibility](#built-for-sillytavern-ecosystem-compatibility)
- [What This Enables](#what-this-enables)
- [Image Generation with AUTOMATIC1111](#image-generation-with-automatic1111)
- [Library-Scale Organization](#library-scale-organization)
- [Additional Capabilities](#additional-capabilities)
- [Built for Local Deployment](#built-for-local-deployment)
- [Known Limitations](#known-limitations)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Danbooru Tag Generator (Optional)](#danbooru-tag-generator-optional)
- [Documentation](#documentation)
- [Credits](#credits)
- [License](#license)

---

## v1.11.0: Scene-First Architecture

This new release represents a large architectural and philosophical shift for NeuralRP. Through extensive testing, I have found that smaller LLMs (I roleplay with a 12b LLM) simply cannot keep characters straight, no matter what you prompt, when lots of dialog is present. Characters just merge together. 

To keep this from happening, I have combined consistent, small chunks of character reinforcement and a "spike" of reinforcement for new characters, along with smart auto-summarization based on scene change, to ensure the LLM both recognizes the change, and doesn't "merge" the new and old characters together.

The result: a "scene-first" focus. When a new character or NPC enters, there's a reason for that. You want that different energy, someone who will contribute something new or unique. This design maximizes that, while keeping in minid the limitations of small, locally-runnable LLMs.

What's New
1. **Continuous Character Presence (SCENE CAST)**
- Every turn shows who's in the scene with their key traits and speech style
- Replaces "reinforce every 5 turns" with constant lightweight grounding
- Single character: description + personality summary
- Multi-character: pre-generated capsules for each active character

2. **Strategic Full Card Injection**
- Full character cards appear at reset points: first appearance, sticky window (turns 1-3), and after long absence (20+ messages)
- New characters establish voice immediately with rich context
- Returning characters get a "memory refresh" to prevent drift

3. **Smarter Auto-Summarization**
- Cast-change: Compresses old content when characters leave (~200 token capsule per scene)
- Threshold: Groups old messages into scenes when context hits 80% (less aggressive than before)
- Both run in the background after you get your response (non-blocking)

4. **Tighter History Window + 8k Baseline**
- Keeps last 6 exchanges (12 messages) verbatim instead of 10
- max_context default increased from 4096 to 8192 (tested and designed for this window)
- summarize_threshold raised to 0.80 (use more of the window before compressing)
- Tuned for modern 7B-14B models (Mistral, Qwen, Llama 3)

Benefits: Characters maintain consistent voices through long conversations. Better token efficiency means more room for world info and detailed responses. Auto-summarization keeps performance smooth without manual intervention.

Breaking Changes: reinforce_freq setting removed (SCENE CAST replaces it). Existing chats work immediately; old summaries preserved.

---

## Why It’s Different

1. **Context hygiene engine** — Character cards appear in full at strategic reset points (first appearance, turns 1-3 sticky window, after long absence), then lightweight SCENE CAST reminders every turn keep voices consistent. World lore appears only when semantically relevant unless marked as "canon law", which reinforces periodically. Recent dialogue (last 6 exchanges) stays detailed and verbatim; older content compresses into scene summaries. ~50–60% of your 8k context stays as recent conversation and character grounding, with plenty of headroom for world info and long responses.

2. **Emergent cast with visual canon** — Quickly create NPCs on the fly that are isolated to one chat, have full parity with characters, can be promoted, and stay branch‑safe. Gender selection keeps both text and visuals targeted, and one-click Danbooru tag assignment based on character description solves the problem of visual drift, so you can stay immersed in the actual roleplay.

3. **Native Stable Diffusion, scene‑aware** — Deep AUTOMATIC1111 integration: inpainting, both manual and automatic generation via the "snapshot" feature, and per‑image metadata, carefully designed to not "leak" into the context window. A "Favorites" menu allows you to jump right to where the image appeared in it's original chat, so you can see the context. All designed to fit into the flow of roleplay, as unintrusively as possible.

---

## NeuralRP: Storytime

When I discovered Huggingface and tuned LLMs for roleplay, I quickly developed a love for experimenting with it. I love writing, and it quickly became a passion to create worlds and characters through written dialog with interesting, creative LLMs.

However, when I started (with Ollama) I noticed something: once I hit about 12k tokens (happens quickly), LLMs completely changed: output became repetitive and garbled, it forgot the situation and world I had carefully detailed, and characters became all the same. Rather than accepting this as a fundamental limitation of small, local LLMs, I decided to move on to other frontends.

Oobabooga was marginally better, but when I tried SillyTavern it opened my eyes to prompt control: you could inject whatever you wanted at set intervals. I began to obsess over dialing in the perfect settings.

What I found was this: I could never get the underlying control I needed because the architecture is fundamentally flawed, and the SD generation extension just wasn't cutting me for me. That needed to be native to work the way I wanted. Not to mention, I was sick of all the knobs. I just wanted to find *my* setting, for *my* 12GB vRAM card that worked for me, rather than being taken out of RP immersion with workarounds and fixes.

Hence, NeuralRP was born.

It's engineered from the ground up with context window in mind. How do you solve the context window problem? By keeping it as small as possibe, in the smartest way possible. Intelligent character card usage and aggressive auto-summaries to keep characters from turning into poorage. Uses SQLite and SQLite-vec backend which is a smart, fast home for truth rather than endless JSONs. Syncs intelligently with SillyTavern cards and semantically finds and injects what matters, when it matters, no matter how deep your world is.

I've built up the SD integration to match. Rather than being a bolt-on extension, its been carefully developed to natively fit into the flow of NeuralRP with a deep feature set, scripted danbooru-optimized prompts, and as automated or manual as you want it to be (or ignore it).

The fact is, there are certain limitations with 7b-14b LLMs that you simply can't get past. But this app maximizes what those LLMs offer for roleplay.

---

## Core Philosophy: Scene-First

NeuralRP v1.11.0 uses scene-first architecture for maximum character consistency across long conversations.

- **SCENE CAST every turn** — All active characters shown with their capsules every turn (replaces periodic reinforcement)
- **6-exchange history window (RP-optimized)** — Last 6 exchanges shown verbatim, older content summarized into scene capsules
- **Hybrid full card system** — Sticky full cards on turns 1-3, capsules on turns 4+, reset points for returning characters
- **Context-aware retrieval** — World lore appears when semantically relevant. Canon law reinforced every 3 turns.
- **Directional relationships** — Alice→Bob ≠ Bob→Alice, tracked automatically, injected when meaningful
- **Scalability by design** — SCENE CAST (400-600 tokens) + 6 exchanges (900-1500 tokens) + sticky full cards (500-2000, turns 1-3). Supports 5+ character group chats.

**Why Scene-First?**
Regular character reinforcement along with smart summarization beats more dialog in context. SCENE CAST ensures consistent voice across 50+ turn conversations while the 6-exchange window prevents attention decay. Sticky full cards (first 3 turns + reset points) provide strong early reinforcement without constant repetition.

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
- **Unquoted keys** — Semantic matching for concepts (e.g., dragon → dragons, draconic). YOu can think of quoted keys as exact bookmarks, unquoted as fuzzy search.
- **Canon law** — Enforce your core world rules on a customizable turn basis

### Relationship Tracking

Five emotional dimensions tracked between all entities (trust, bond, conflict, power, fear):

- Directional tracking (Alice→Bob separate from Bob→Alice)
- Automatic updates via semantic analysis
- Only injected when relevant to current scene
- Preserved through summarization
- User is also tracked and develops relationships with Characters and NPC's

### Intelligent Summarization

Continue conversations beyond context limits. When context approaches 80%:

- All but 6 most recent turns traded for summaries
- Relationship states preserved
- Story continuity maintained
- Summarization window to view and customize summaries (or auto-summarize your summaries!)

Relationship state is one of the most difficult things to maintain when summaries happen. The relationship system brings continuity without having to update character cards manually.

### Branching Timelines

Fork any message to create alternate storylines. Characters, NPCs, world info, and relationships copied. NPCs and relationships develop independently across branches.

---

## Image Generation with AUTOMATIC1111

Native integration, with the goal to be best in class:

### 1. Performance‑Aware Presets

Automatic selectable step/resolution reduction when context is large (>12K tokens) to prevent VRAM crashes, with sane defaults for 8–16 GB cards.

### 2. Inpainting

- Adjustable brush size
- Ctrl+Z undo
- Eraser tool
- Persistent mask between regenerations
- Full A1111 parameter control

### 3. Manual-Mode Generation

- **Pop-out Window** - Easily accesssable pop out window, prompt stays in browser memory so you don't have to retype every time, re-size generation on the fly.
- **Character tag substitution** – Assign Danbooru tags to characters once, reference with `[CharacterName]` in prompts for consistent appearance without memorizing tag lists.

### 4. Snapshots

- **Generated from Context** – Generate images from chat context automatically; AI analyzes the last 20 messages, providing location, action, activity, dress, and facial expression, which is then directly matched to an SD Generation-optimized prompt. 
- **How is it Different?** - It doesn't just ask the LLM for a complete prompt and shove it into A1111 like others do. It takes a specific framework optimized for Anime-based LLM generation models, using Danbooru tags that those models are specifically trained on, to create consistent images. The LLM is just delivering keywords - action, location, activity, dress, and facial expression. And no "leakage" into the context window, one of the things that would ruin my RP chats with other frontends you have probably used.
- **You are IN THE SHOT** - Define yourself as male/female/other, and fill out your own tags in settings. You're in the snapshot, with context derived from your tags and the chat text. 

### 5. Browse and Save Your Images
- **Images Folder** - separate images folder where every generation is saved, even if you delete it in the chat. 
- **Favorites menu** – Favorite images to build a visual library; click on the image and go straight to the image's original location in the chat. Filter by tag.
- **Jump‑to‑source** – Double‑click favorites to jump back to the exact chat moment they came from.
- **Recreate images** - Image metadata saved, so you can see exactly what prompt generated what image. Easily recreate images with a click of a button.

---

## Library-Scale Organization

Tag management for character and world cards:

- **AND semantics** — Filter by multiple tags (must have ALL selected)
- **Quick filter chips** — Top 5 most-used tags surface automatically
- **Autocomplete** — Suggests existing tags to prevent bloat
- **Automatic extraction** — SillyTavern V2 card tags preserved on import

---

## Additional Capabilities

- **Multi-mode chat** — Narrator (third-person), Focus (first-person), Auto mode that automatically detects who should be talking.
- **Live editing** — AI-generated content is editable, all user edits are saved into the database.
- **Change history** — 30-day retention with browse/restore functionality
- **Soft delete** — Messages archived instead of deleted, searchable across history
- **Export for training** — Export to Alpaca/ShareGPT/ChatML formats for Unsloth
- **Built on SQLite + SQLite-vec** - Unified data system makes it easy to "bolt on" features, tune and mod this to you heart's content. Everything is in a single file (neuralrp.db), easy to back up or inspect.
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

- Requires 8k+ context models. NeuralRP's scene-first architecture won't fit in 4k context windows. Modern 7B-14B RP-tuned models (Mistral, Qwen, Llama 3, etc.) all ship with 8k+ by default.
- No cloud extensibility, all local
- Requires AUTOMATIC1111 for image generation, not compatible with ComfyUI and others at this time (run with --API).
- Requires OpenAI-compatible backend with text to text LLM running locally (KoboldCcp strongly recommended).
- Tested with KoboldCcp with a quantized 12B LLM model tuned for RP, and with A1111 running an Illustrious SD model.
- All testing done with a NVidia 3060 12GB vRAM GPU
- Running 2 LLMs with an 8GB vRAM GPU is untested, will likely lead to sub-optimal results
- Danbooru tagging is optimized for SD models like Pony and Illustrious, which are trained for that input. Other SD models are untested

## Hardware Requirements

**Recommended:**
- 12GB+ VRAM GPU (for running both LLM's at the same time)
- Python 3.8+
- KoboldCpp (LLM inference)
- AUTOMATIC1111 WebUI (image generation)

**Minimum:**
- 8GB VRAM (with Performance Mode), recommended just use 1 model at a time
- Supports: KoboldCpp, Ollama, Tabby (OpenAI-compatible)

---

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
