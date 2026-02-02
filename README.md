# NeuralRP

![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)

**tl;dr: NeuralRP is a local, no-cloud roleplay engine that keeps characters, NPCs, worlds, and images coherent across 30k+ tokens on midrange GPUs—without you babysitting templates or extensions.**

**Simple out of the box. Just drop your cards in the folders, there isn't a million different things to configure. Just go.**

**8 GB vRAM? Can't pay $4,000 for a sick GPU? Neither can I.**

**No cloud subscriptions, no inference costs**

**[Quickstart Guide](docs/QUICKSTART.md)**

---

## Why It’s Different

1. **Context hygiene engine** — Characters inject in full on their first 3 turns, then reinforce with minimal constraints. World lore appears only when semantically relevant. 70–80% of your context stays as live dialogue, even with 5–6 characters and deep worlds.

2. **Emergent cast with visual canon** — Chat‑scoped NPCs have full parity with globals, can be promoted, and stay branch‑safe. v1.10 adds gender‑aware casting and Danbooru‑based visual canon, so those NPCs can also have stable, consistent imagery over long campaigns.

3. **Native Stable Diffusion, scene‑aware** — Deep AUTOMATIC1111 integration: performance‑aware presets, inpainting, character tag substitution, and per‑image metadata. v1.9+ adds scene snapshots, favorites, and tag‑based visual learning so the engine can auto‑compose prompts that match both the scene and your preferred style.

---

## What’s New in v1.9.0–v1.10.0

- **Danbooru character casting (v1.10)** – One‑click visual canon: NeuralRP analyzes your character’s description (hair, eyes, etc) and binds them to a matching Danbooru reference character. Works for both globals and NPCs, with reroll support and priority tags in snapshot prompts.

- **Gender‑aware engine (v1.10)** – First‑class gender field (female/male/other) wired into character profiles, reinforcement, and snapshots (automatic `1girl/2girls/1boy/2boys` tagging), while staying compatible with SillyTavern cards.

- **Scene snapshots + favorites (v1.9)** – One‑click, scene‑aware Stable Diffusion images using a 4‑block prompt builder; favorites gallery with jump‑to‑source, tag filtering, and persistent image history across chats.

- **Visual learning (v1.9)** – NeuralRP learns from your favorited images, detecting Danbooru tags and biasing future generations toward your preferred styles. Custom tags from manual generations become part of a searchable, embedding‑backed visual library.

---

## NeuralRP: Storytime

When I discovered Huggingface and tuned LLMs for roleplay, I quickly developed a love for experimenting with it. I love writing, and it quickly became a passion to create worlds and characters through written dialog with interesting, creative LLMs.

However, when I started (with Ollama) I noticed something: once I hit about 12k tokens (happens quickly), LLMs completely changed: output became repetitive and garbled, it forgot the situation and world I had carefully detailed, and characters became all the same. Rather than accepting this as a fundamental limitation of small, local LLMs, I decided to move on to other frontends.

Oobabooga was marginally better, but when I tried SillyTavern it opened my eyes to prompt control: you could inject whatever you wanted at set intervals. I began to obsess over dialing in the perfect settings.

What I found was this: I could never get the underlying control I needed because the architecture is fundamentally flawed, and the SD generation extension just wasn't cutting me for me. That needed to be native to work the way I wanted. Not to mention, I was sick of all the knobs. I just wanted to find *my* setting, for *my* 12GB vRAM card that worked for me, rather than being taken out of RP immersion with workarounds and fixes.

Hence, NeuralRP was born.

It's engineered from the ground up with context window in mind. No full character card dump every turn. Uses SQLite backend which syncs intelligently with SillyTavern cards, semantically finds and injects what matters, when it matters, no matter how deep your world is.

The fact is, there are certain limitations with 7B-12B LLMs that you simply can't get past. But this app maximizes what locally runnable LLMs offer for roleplay.

---

## Core Philosophy: Conversation First

70-80% of your context budget should be dialogue, not metadata. Characters, world info, and relationships exist to support conversation, not dominate the prompt.

- **Inject on first 3 turns** — Full character cards (single) or capsules (multi-char) for early-turn consistency
- **Just-in-time grounding** — World lore appears when semantically relevant
- **Directional relationships** — Alice→Bob ≠ Bob→Alice, tracked automatically
- **Scalability by design** — 1 character = ~4-6% of context. 5 characters = ~20-30%. After sticky window, ~80% for dialogue.

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

Optimized for 5-6 active characters with distinct voices. Capsules (compressed summaries with dialog examples) enable group chats without context overflow.

### Emergent NPCs

Create background characters mid-chat (bartender, guard, merchant) with full personality and relationship tracking. Promote them to global characters when they matter.

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

### Intelligent Summarization

Continue conversations beyond context limits. When context approaches 85%:

- Oldest turns traded for summaries
- Relationship states preserved
- Story continuity maintained

### Branching Timelines

Fork any message to create alternate storylines. Characters, NPCs, world info, and relationships copied. NPCs develop independently across branches.

---

## Image Generation with AUTOMATIC1111

Native integration, not an afterthought.

### Performance‑Aware Presets

Automatic step/resolution reduction when context is large (>12K tokens) to prevent VRAM crashes, with sane defaults for 8–16 GB cards.

### Inpainting

- Adjustable brush size
- Ctrl+Z undo
- Eraser tool
- Persistent mask between regenerations
- Full A1111 parameter control

### Character‑Aware Prompts

- **Character tag substitution** – Assign Danbooru tags to characters once, reference with `[CharacterName]` in prompts for consistent appearance without memorizing tag lists.
- **Visual canon casting (v1.10)** – Optional Danbooru character binding so snapshots and manual prompts can use canon tags for that character first.

### Snapshot + Visual Learning (v1.9+)

- **Scene snapshots** – Click 📸 to generate images from chat context automatically; AI analyzes characters, setting, mood, and action and builds a 4‑block prompt.
- **Favorites and style learning** – Favorite images to build a visual library; NeuralRP detects Danbooru tags in those prompts and biases future generations toward your preferred styles.
- **Jump‑to‑source** – Double‑click favorites to jump back to the exact chat moment they came from.
---

## Library-Scale Organization (v1.8.0)

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
- 12-16GB VRAM GPU (NVIDIA/AMD)
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
