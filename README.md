# NeuralRP

![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)

**tl;dr: NeuralRP is a local, no-cloud roleplay engine that keeps characters, NPCs, worlds, and images coherent across 30k+ tokens on midrange GPUs. Think less about tuning and more about immersing in roleplay.**

**Simple out of the box. Just drop your cards in the folders, there isn't a million different things to configure. Just go.**

**8 GB vRAM? Can't pay $4,000 for a sick GPU? Neither can I.**

**No cloud subscriptions, no inference costs**

**[Quickstart Guide](docs/QUICKSTART.md)**

---

## Why It’s Different

1. **Context hygiene engine** — With 7B-12B LLM's, every token you put into the context window matters, and everything about this app cares about that. Character cards present in full on the first 3 turns, then token-controlled portions appear on customizable fixed intervals. World lore appears only when semantically relevant. 70–80% of your context stays as live dialogue, even with 3+ characters and deep worlds.

2. **Emergent cast with visual canon** — Quickly create NPCs on the fly that are isolated to one chat, have full parity with characters, can be promoted, and stay branch‑safe. Gender selection keeps both text and visuals targeted, and one-click Danbooru tag assignment based on character description solves the problem of visual drift, so you can stay immersed in the actual roleplay.

3. **Native Stable Diffusion, scene‑aware** — Deep AUTOMATIC1111 integration: inpainting, both manual and automatic generation via the "snapshot" feature, and per‑image metadata, carefully designed to not "leak" into the context window. A "Favorites" menu allows you to jump right to where the image appeared in it's original chat, so you can see the context. All designed to fit into the flow of roleplay, as unintrusively as possible.

---

## NeuralRP: Storytime

When I discovered Huggingface and tuned LLMs for roleplay, I quickly developed a love for experimenting with it. I love writing, and it quickly became a passion to create worlds and characters through written dialog with interesting, creative LLMs.

However, when I started (with Ollama) I noticed something: once I hit about 12k tokens (happens quickly), LLMs completely changed: output became repetitive and garbled, it forgot the situation and world I had carefully detailed, and characters became all the same. Rather than accepting this as a fundamental limitation of small, local LLMs, I decided to move on to other frontends.

Oobabooga was marginally better, but when I tried SillyTavern it opened my eyes to prompt control: you could inject whatever you wanted at set intervals. I began to obsess over dialing in the perfect settings.

What I found was this: I could never get the underlying control I needed because the architecture is fundamentally flawed, and the SD generation extension just wasn't cutting me for me. That needed to be native to work the way I wanted. Not to mention, I was sick of all the knobs. I just wanted to find *my* setting, for *my* 12GB vRAM card that worked for me, rather than being taken out of RP immersion with workarounds and fixes.

Hence, NeuralRP was born.

It's engineered from the ground up with context window in mind. No full character card dump every turn. Uses SQLite backend which syncs intelligently with SillyTavern cards, semantically finds and injects what matters, when it matters, no matter how deep your world is.

I've built up the SD integration to match - rather than being a bolt-on extension, its been carefully developed to natively fit into the flow of NeuralRP with a deep feature set, and as automated or manual as you want it to be (or ignore it).

The fact is, there are certain limitations with 7B-12B LLMs that you simply can't get past. But this app maximizes what 7B-14B LLMs offer for roleplay.

---

## Core Philosophy: Conversation First

70-80% of your context budget should be dialogue, not metadata. Characters, world info, and relationships exist to support conversation, not dominate the prompt.

- **Inject on first 3 turns** — Full character cards (single) or capsules (multi-char) for early-turn consistency
- **Just-in-time grounding** — World lore appears when semantically relevant
- **Directional relationships** — Alice→Bob ≠ Bob→Alice, tracked automatically
- **Scalability by design** — 1 character = ~4-6% of context. 3 characters = ~20-30%. After sticky window, ~80% for dialogue.
- **Narrator Mode** - "Just chat", no world or character cards needed. Let it all develop naturally, elevating NPC's and worlds with AI-powered tools within the app. Or, easily use your pre-built deep worlds and characters from v2 cards.

---

## Built for SillyTavern Ecosystem Compatibility

NeuralRP was designed to work *with* SillyTavern cards, not replace them.

- **Bi-Directional Smart sync** — Timestamp-based conflict resolution between JSON cards and database
- **Automatic tag preservation** — SillyTavern V2 card tags extracted and stored
- **Forward and backward compatible** — No conversion, no data loss

### Card Generation

Create optimized character cards in two ways:

- **From context** — Generate PList-optimized definitions from conversation history
- **From natural language** — Write plain-text descriptions, NeuralRP converts to PList format

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
- **Unquoted keys** — Semantic matching for concepts (e.g., dragon → dragons, draconic)
- **Canon law** — Core world rules always included to prevent physics/magic violations

### Relationship Tracking

Five emotional dimensions tracked between all entities (trust, bond, conflict, power, fear):

- Directional tracking (Alice→Bob separate from Bob→Alice)
- Automatic updates via semantic analysis
- Only injected when relevant to current scene
- Preserved through summarization
- User is also tracked and develops relationships with Characters and NPC's

### Intelligent Summarization

Continue conversations beyond context limits. When context approaches 85%:

- Oldest turns traded for summaries
- Relationship states preserved
- Story continuity maintained

Relationship state is one of the most difficult things to maintain when summaries happen. The relationship system brings continuity without having to update character cards manually.

### Branching Timelines

Fork any message to create alternate storylines. Characters, NPCs, world info, and relationships copied. NPCs and relationships develop independently across branches.

---

## Image Generation with AUTOMATIC1111

Native integration, not an afterthought:

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

- **Generated from Context** – Generate images from chat context automatically; AI analyzes the last turn, providing location, actions, and dress, which is then directly matched to a LLM-optimized prompt. Semantic matching is used as a fallback.
- **How is it Different?** - It doesn't just ask the LLM for a complete prompt and shove it into A1111, it doesn't leak text into the context window. It takes a specific framework optimized for Anime-based LLM generation models, using Danbooru tags that those models are specifically trained on, to create consistent images. The LLM is just delivering keywords - action, location, and dress. This is a fundamentally different process from every other solution out there. 

### 5. Browse and Save Your Images
- **Images Folder** - separate images folder where every generation is saved, even if you delete it in the chat
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

- **Multi-mode chat** — Narrator (third-person), Focus (first-person), Auto modes
- **Live editing** — AI-generated content appears in editable textboxes before saving
- **Change history** — 30-day retention with browse/restore functionality
- **Soft delete** — Messages archived instead of deleted, searchable across history
- **Export for training** — Export to Alpaca/ShareGPT/ChatML formats for Unsloth
- **Built on SQLite + SQLite-vec** - Unified data system makes it easy to "bolt on" features, tune and mod this to you heart's content. Just add another table. 
- **Migration system** — Seamless upgrades (v2 → v3 schemas) without data loss.

---

## Built for Local Deployment

NeuralRP assumes you're running:

- Local LLM via KoboldCpp, TabbyAPI, or Ollama (OpenAI-compatible)
- Local Stable Diffusion via AUTOMATIC1111 WebUI (optional)
- 12-16GB VRAM (8GB minimum with Performance Mode)

All data in SQLite (neuralrp.db) with automatic JSON export for SillyTavern compatibility.

---

## Hardware Requirements

**Recommended:**
- 12-16GB VRAM GPU (for running both LLM's at the same time)
- Python 3.8+
- KoboldCpp (LLM inference)
- AUTOMATIC1111 WebUI (image generation)

**Minimum:**
- 8GB VRAM (with Performance Mode)
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

To enable one-click Danbooru character visual matching, run the fetch script to download character data from Danbooru API:

```bash
# Edit app/fetch_danbooru_characters.py with your Danbooru API key, then:
python app/fetch_danbooru_characters.py
```

This generates `app/data/danbooru/book1.xlsx` from Danbooru's public API (~15-30 minutes for 1394 characters). See **[Quickstart Guide](docs/QUICKSTART.md)** for detailed setup instructions including API key acquisition.

---

## Documentation

- **[Quickstart Guide](docs/QUICKSTART.md)** — Get up and running fast
- **[Technical Documentation](docs/TECHNICAL.md)** — Deep dive into architecture and implementation
- **[Changelog](CHANGELOG.md)** — Version history

---

## Credits

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.
