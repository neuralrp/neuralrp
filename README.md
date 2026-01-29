![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)

# NeuralRP
When I discovered Huggingface and tuned LLM's for roleplay, I quickly developed a love for experimenting with it. I love writing, and it quickly became a passion to create worlds and characters through written dialog with interesting, creative LLM's. 

However, when I started (with Ollama) I noticed something: once I hit about 12k tokens (happens quickly), LLM's completely changed: output became repetitive and garbled, it forgot the situation and world I had carefully detailed in the system prompt, and characters became all the same. Rather than accepting this as a fundamental limitation of small, local LLM's, I decided to move on to other frontends.

Oobabooga was marginially better, but when I tried SillyTavern it opened up my eyes to prompt control: you could inject whatever you wanted into the prompt at a set number of turns. You could use character and world cards to enforce your vision, so the LLM wouldn't forget as easily. I began to obsess over dialing in the perfect settings, investigating what worked well and what didn't.

What I found was this: I could never get the underlying control I needed because the architecture is fundamentally flawed, and the SD generation extension just wasn't cutting it. That needed to be native to work the way I wanted to. Not to mention, I was sick of all the knobs. I just wanted to find *my* setting, for *my* 12GB vRAM card, that worked for me. And I was sick of SD gen work-arounds and editing poor LLM replies taking me out RP immersion.

Hence, NeuralRP was born. 

It's engineered from the ground up with context window in mind, every decision, every feature. No full character card dump every turn. No token leakage from SD generation. Use of a database backend, compatible with SillyTavern character and world cards, that semantically finds and injects what matters, when it matters.

The fact is, there are certain limitations with 7B-12B LLM's that you simply can't get past. But this app I built, in my opinion, maximizes what locally runnable LLM's offer for roleplay. 

## What Makes the Architecture Different

**Context assembly, not context dumping.** Every piece of story data is managed:

- Character cards inject in full on first appearance, then reinforce with 50-200 token capsules or PList constraints every few turns. Not repeated in full every turn.

- World information uses semantic search via sqlite-vec embeddings to inject lore relevant to the last 5 messages. Not dumped wholesale or manually triggered.

- Relationships between entities track across five emotional dimensions (trust, bond, conflict, power, fear). These update and trigger automatically via semantic analysis. Not manually managed, dumped in context recklessly, or forgotten by turn 20.

- NPCs exist as chat-scoped entities with their own relationship states, and can be promoted to global characters mid-conversation. And you can create their cards using AI in-app.

- Summarization trades old context for new when the budget nears 85%, preserving story continuity without losing relationship state.

The outcome: Relationship status tracked through summarizations. NPCs that emerge naturally without breaking immersion. Alternate timelines that branch with independent relationship states.

## Core Philosophy: Conversation First

70-80% of your context budget should be dialogue, not metadata. Characters, world info, and relationships exist to support conversation, not dominate the prompt. AND, this is supported by both experience and studies: a short, strongly worded prompt beats a meandering, unfocused one. Why should't RP optimize for that?

- **Inject once, reinforce minimally** — Full character cards on first appearance, then 50-200 token PList statements or character capsules every N turns

- **Just-in-time grounding** — World lore appears when semantically relevant, not before

- **Directional relationships** — Alice→Bob ≠ Bob→Alice, tracked automatically via semantic embeddings

- **Scalability by design** — 1 character = 5-8% of context. 5 characters = 15-20% of context. The rest is conversation.

Short, strong statements make an outsized impact on smaller, less intelligent LLMs. This is the stuff that keeps small LLM's rolling past 30k+ tokens worth of conversation.

## Built for SillyTavern Ecosystem Compatibility

NeuralRP was designed to work with SillyTavern cards, not replace them. Every compatibility decision prioritizes preserving your existing work. This was a core feature since v1.0.0, and it's continued through all the feature updates. The JSON and SQLite sync is intelligent and just makes sense. 

- **Bi-Directional Smart sync** — Timestamp-based conflict resolution between JSON cards and the NeuralRP database. Edit cards in either NeuralRP or externally without losing changes.

- **Automatic tag preservation** — Import a card with tags? They're extracted and stored. Export it? Tags are included.

- **Entry-level world info merging** — Edit a world in both places? NeuralRP merges entries intelligently, preserving additions from both sources.

- **Forward and backward compatible** — Characters created in NeuralRP work in SillyTavern. Characters from SillyTavern work in NeuralRP. No conversion, no data loss.

### Card Generation Machine

One of my original goals for NeuralRP was to in part be a card generation tool for SillyTavern. I did tons of research and experimentation on what made a good card, another obsession of mine. I created multiple python apps for this very purpose before NeuralRP, then merged it in. These tools I built bring in all my learnings. You can create optimized character cards in two ways:

- **From context** — Generate PList-optimized character definitions directly from conversation history. The AI analyzes how a character actually behaved in chat and extracts personality traits, speech patterns, and behavioral rules into clean PList format.

- **From natural language** — Write plain-text sentences describing a character and NeuralRP converts it into PList format optimized for LLM consumption.

Both methods output SillyTavern V2-compatible JSON files. Prototype NPC's/characters in conversation, then formalize them into reusable cards. OR, just write "Ben: a grizzled bartender with a heart of gold". The LLM and python does the rest.

## What This Enables

### Multi-Character Chats That Scale

Optimized to run multiple active characters with distinct voices and full personality tracking without context overflow. Capsules (compressed character summaries with dialog examples) enable group chats that other tools can't sustain past 2-3 characters. I've literally never been able to do this effectively with another front end before.

### Emergent NPCs

Create background characters mid-chat (bartender, guard, merchant) with full personality cards and relationship tracking. Promote them to global characters when they matter. NPCs are chat-scoped by default — "Guard Marcus" in Chat A ≠ "Guard Marcus" in Chat B — with automatic entity remapping on branching.

### Semantic World Information

A major question I faced is, how do you ensure your world card entries get injected at the right time? What if you have hundreds of entries? What if you want one of your entries injected at specific intervals, at different points in the coversation? And the trickiest one, what if you want your "Elf" entry to trigger for "elves, elven, the elf king, elf sanctuary" etc, but you don't want your "The Great Crash Landing" historical event entry to trigger on "this is a great bow!"? I think I've solved this problem.

- **Semantic Search Engine** - Remember when I said SillyTavern's architecture was fundamentally flawed? This is what I mean. NeuralRP is powered by all-mpnet-base-v2 for vector database semantic search, using sentence-transformers to understand meaning, not just keywords. And it stays just as fast for 10 entries or 1000 entries. SQLite-vec just came out in 2024. This is new stuff.

- **Quoted keys** (`"Great Crash Landing"`) — Exact phrase match, prevents false positives

- **Unquoted keys** (`dragon`) — Semantic search, catches plurals and synonyms

- **Canon law** — Core world rules always included and reinforced every N turns to prevent physics/magic violations

### Relationship Tracking

Five emotional dimensions tracked between all entities (characters, NPCs, user) with automatic updates via semantic analysis:

- Trust / Emotional Bond / Conflict / Power Dynamic / Fear-Anxiety

- Directional (Alice→Bob is tracked separately from Bob→Alice)

- Only injected when relationships deviate significantly from neutral and are semantically relevant to the current scene

- No LLM calls required — uses embeddings for sub-20ms updates. Tight, pre-worded sentences ensure maximum impact for minimal token use.

This is powerful for one reason: it keeps relationships tracked through auto-summarization, which can "mush-ify" relationships real quick.

### Intelligent Summarization

Continue conversations beyond context limits. When context approaches 85%:

- Oldest 10 turns are traded for summary versions

- Relationship states are preserved

- Character definitions remain intact

- Story continuity is maintained

### Branching Timelines

Fork any message to create alternate storylines. All characters, NPCs, world info, and relationships are copied. NPC entity IDs are remapped, so they can develop independently with each story arc.

## Library-Scale Organization (v1.8.0)

Tag management for character and world cards:

- **AND semantics** — Filter by multiple tags (must have ALL selected tags)

- **Quick filter chips** — Top 5 most-used tags surface automatically

- **Autocomplete** — Suggests existing tags to prevent tag bloat

- **Automatic extraction** — SillyTavern V2 card tags preserved on import

- **Normalization** — Lowercase, trimmed, deduplicated automatically

## Image Generation with AUTOMATIC1111 Integration

Most RP tools bolt on image generation as an afterthought. NeuralRP integrates it deeper than any other RP application I've tried. It's native, takes full advantage of A1111, and run alongside your LLM using in-built optimization tools designed for this very purpose.

### Performance-Aware Presets

If your chat context is large (>12K tokens), NeuralRP automatically reduces SD steps/resolution to prevent VRAM crashes. You get feedback when image generation is slow due to context size, with a suggestion to summarize.

### Inpainting with Actual Features

- **Adjustable brush size** — Paint masks with precision control

- **Ctrl+Z undo** — Made a mistake painting? Undo it.

- **Eraser tool** — Remove parts of your mask without repainting everything

- **Persistent paint mask** — If you don't like the inpaint result, the mask stays. Regenerate without repainting.

- **Full A1111 parameter control** — Steps, CFG, sampling method, denoising strength

### Character Tag Substitution

Assign Danbooru tags to characters once (blue eyes, warrior, short hair, armor), then reference them in prompts with `[CharacterName]`. NeuralRP automatically expands the tag.

Consistent character appearance across hundreds of generations without memorizing tag lists. Want to generate Alice in a new scene? Write `[Alice]` standing in a forest and her full Danbooru profile is injected automatically.

### Persistent Metadata

Every generated image saves all generation parameters to the database:

- Prompt (positive and negative)

- Model checkpoint

- Steps, CFG scale, sampling method

- Seed, dimensions, denoising strength

Click any image, click "Regenerate" — exact reproduction. Or tweak parameters and generate variations.

### No Prompt Fragment Leakage

SD prompts are kept separate from LLM conversation context. Image generation doesn't pollute your chat history or confuse the LLM with Danbooru syntax.

## Additional Capabilities

- **Multi-mode chat** — Narrator (third-person), Focus (first-person), Auto modes

- **Live editing** — AI-generated content appears in editable textboxes before saving

- **Change history** — 30-day retention across characters, world info, and chats with browse/restore functionality

- **Soft delete** — Messages archived instead of deleted, searchable across active + archived history

- **Export for training** — Export chats to JSON in Alpaca/ShareGPT/ChatML formats optimized for Unsloth


## Built for Local Deployment

NeuralRP assumes you're running:

- Local LLM via KoboldCpp, TabbyAPI, or Ollama (OpenAI-compatible endpoints)

- Local Stable Diffusion via AUTOMATIC1111 WebUI (optional)

- 12-16GB VRAM with performance mode to queue heavy operations and prevent crashes

All data lives in a single SQLite database (neuralrp.db) with ACID guarantees. No cloud sync, no external dependencies beyond your local inference stack. And it syncs back to your JSON cards.

## Hardware Requirements

**Recommended:**

- 12-16GB VRAM GPU (NVIDIA/AMD)

- Python 3.8+

- KoboldCpp (for LLM inference)

- AUTOMATIC1111 WebUI (for image generation)

**Minimum:**

- 8GB VRAM (with Performance Mode enabled)

- Supports: KoboldCpp, Ollama, Tabby (OpenAI-compatible endpoints)

## Quick Start

**Clone repository**

```bash
git clone https://github.com/neuralrp/neuralrp.git
cd neuralrp
```

**Run NeuralRP**

```bash
launcher.bat    # On Windows - handles everything automatically
```

Or for other systems:

```bash
pip install -r requirements.txt
python main.py
```

**Open browser**

Navigate to http://localhost:8000

Configure your LLM and image generation endpoints in the app's Settings panel.

## Data Structure

```
app/
├── data/
│   ├── neuralrp.db          # SQLite database (characters, chats, world info, embeddings, relationships)
│   ├── characters/          # SillyTavern V2 JSON cards (auto-synced)
│   ├── chats/               # Chat sessions (auto-synced)
│   └── worldinfo/           # World Card JSON (auto-synced)
└── images/                  # Generated images
```

All data stored in SQLite but automatically exported to JSON files for SillyTavern compatibility. You can edit files externally — smart sync (v1.8.0) uses timestamps to prevent data loss during conflicts.

## Documentation

- [Technical Documentation](docs/TECHNICAL.md) — Deep dive into implementation details, context assembly logic, and design decisions

- [Changelog](CHANGELOG.md) — Full version history

## Credits

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.