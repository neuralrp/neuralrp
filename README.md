![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)

# NeuralRP

NeuralRP is a local AI roleplay application built around context hygiene. Most tools in this space dump full character cards into prompts every turn until you hit token limits. NeuralRP treats context as something to assemble intelligently from a SQLite database — inject what's needed when it's needed, reinforce periodically, and keep the bulk of your tokens for actual conversation.

The result: multi-character chats that last hundreds of turns without context overflow, character drift, or world inconsistency. This isn't a SillyTavern frontend. It's a different approach to the same problem.

## What Makes the Architecture Different

**Context assembly, not context dumping.** Every piece of story data is managed:

- Characters inject in full on first appearance, then reinforce with 50-200 token capsules or PList constraints every few turns. Not repeated in full every turn.

- World information uses semantic search (sqlite-vec embeddings) to inject only lore relevant to the last 5 messages. Not dumped wholesale or manually triggered.

- Relationships between entities track across five emotional dimensions (trust, bond, conflict, power, fear) and update automatically via semantic analysis. Not manually managed or forgotten by turn 20.

- NPCs exist as chat-scoped entities with their own relationship states, and can be promoted to global characters mid-conversation.

- Summarization trades old context for new when the budget nears 85%, preserving story continuity without losing relationship state.

The outcome: 5+ character group chats that last 100+ turns. NPCs that emerge naturally. Alternate timelines that branch with independent relationship states. All on 12-16GB VRAM.

## Core Philosophy: Conversation First

70-80% of your context budget should be dialogue, not metadata. Characters, world info, and relationships exist to support conversation, not dominate the prompt.

- **Inject once, reinforce minimally** — Full character cards on first appearance, then 50-200 token capsules every N turns

- **Just-in-time grounding** — World lore appears when semantically relevant, not before

- **Directional relationships** — Alice→Bob ≠ Bob→Alice, tracked automatically via semantic embeddings

- **Scalability by design** — 1 character = 5-8% of context. 5 characters = 15-20% of context. The rest is conversation.

The difference between a 3-character chat that breaks at turn 30 and a 6-character chat that sustains through turn 200.

## Built for SillyTavern Ecosystem Compatibility

NeuralRP was designed to work with SillyTavern cards, not replace them. Every compatibility decision prioritizes preserving your existing work.

### Bidirectional Sync Without Data Loss

- **Smart sync (v1.8.0)** — Timestamp-based conflict resolution. Edit cards in either NeuralRP or externally without losing changes.

- **Automatic tag preservation** — Import a card with tags? They're extracted and stored. Export it? Tags are included.

- **Entry-level world info merging** — Edit a world in both places? NeuralRP merges entries intelligently, preserving additions from both sources.

- **Forward and backward compatible** — Characters created in NeuralRP work in SillyTavern. Characters from SillyTavern work in NeuralRP. No conversion, no data loss.

### Card Generation Machine (v1.1/v1.2 Original Thesis)

One of NeuralRP's original goals was to be a card generation tool for SillyTavern. You can create optimized character cards in two ways:

- **From context** — Generate PList-optimized character definitions directly from conversation history. The AI analyzes how a character actually behaved in chat and extracts personality traits, speech patterns, and behavioral rules into clean PList format.

- **From natural language** — Write plain-text sentences describing a character and NeuralRP converts it into PList format optimized for LLM consumption.

Both methods output SillyTavern V2-compatible JSON files. The idea was to prototype characters in conversation, then formalize them into reusable cards.

## What This Enables

### Multi-Character Chats That Scale

Run 5+ active characters with distinct voices and full personality tracking without context overflow. Capsules (compressed character summaries with dialog examples) enable group chats that other tools can't sustain past 2-3 characters.

### Emergent NPCs

Create background characters mid-chat (bartender, guard, merchant) with full personality cards and relationship tracking. Promote them to global characters when they matter. NPCs are chat-scoped by default — "Guard Marcus" in Chat A ≠ "Guard Marcus" in Chat B — with automatic entity remapping on branching.

### Semantic World Information

World lore appears when contextually relevant via embedding-based search, not regex triggers or manual injection.

- **Quoted keys** (`"Great Crash Landing"`) — Exact phrase match, prevents false positives

- **Unquoted keys** (`dragon`) — Semantic search, catches plurals and synonyms

- **Canon law** — Core world rules always included and reinforced every N turns to prevent physics/magic violations

### Relationship Tracking

Five emotional dimensions tracked between all entities (characters, NPCs, user) with automatic updates via semantic analysis:

- Trust / Emotional Bond / Conflict / Power Dynamic / Fear-Anxiety

- Directional (Alice→Bob is tracked separately from Bob→Alice)

- Only injected when relationships deviate significantly from neutral and are semantically relevant to the current scene

- No LLM calls required — uses embeddings for sub-20ms updates

### Intelligent Summarization

Continue conversations beyond context limits. When context approaches 85%:

- Old messages are traded for summary versions

- Relationship states are preserved

- Character definitions remain intact

- Story continuity is maintained

### Branching Timelines

Fork any message to create alternate storylines. All characters, NPCs, world info, and relationships are copied. NPC entity IDs are remapped — so "Guard Marcus" in Branch A evolves independently from "Guard Marcus" in Branch B.

## Library-Scale Organization (v1.8.0)

Tag management for 100+ character and world cards:

- **AND semantics** — Filter by multiple tags (must have ALL selected tags)

- **Quick filter chips** — Top 5 most-used tags surface automatically

- **Autocomplete** — Suggests existing tags to prevent tag bloat

- **Automatic extraction** — SillyTavern V2 card tags preserved on import

- **Normalization** — Lowercase, trimmed, deduplicated automatically

## Image Generation with AUTOMATIC1111 Integration

Most RP tools bolt on image generation as an afterthought. NeuralRP integrates AUTOMATIC1111 WebUI deeper than any other RP application.

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

### Performance-Aware Presets

If your chat context is large (>12K tokens), NeuralRP automatically reduces SD steps/resolution to prevent VRAM crashes. You get feedback when image generation is slow due to context size, with a suggestion to summarize.

## Additional Capabilities

- **Multi-mode chat** — Narrator (third-person), Focus (first-person), Auto modes

- **Live editing** — AI-generated content appears in editable textboxes before saving

- **Change history** — 30-day retention across characters, world info, and chats with browse/restore functionality

- **Soft delete** — Messages archived instead of deleted, searchable across active + archived history

- **Export for training** — Export chats to JSON in Alpaca/ShareGPT/ChatML formats optimized for Unsloth

## The Problem: Attention Decay, Not Token Overflow

With modern LLMs (Llama 3, Nemo) and 12GB+ VRAM, you can run 8192+ token contexts with image generation simultaneously. Token overflow isn't the crisis it was 2 years ago.

Context hygiene still matters:

- **Attention decay** — Even with 8k tokens, content from turns 1-10 receives less attention than turns 80-90. Character definitions fade over long conversations.

- **Quality vs quantity** — Just because you can fit 5 full character cards (5000 tokens) doesn't mean you should. Redundant repetition wastes context space.

- **Scalability** — Larger contexts enable richer experiences, but intelligent injection is what makes them viable.

- **Character drift** — Without reinforcement, Alice starts using slang, Bob becomes aggressive, and characters blend together after 50+ turns.

## Context Hygiene in Practice

### Smart Character Injection

**Single character chats:**

- Full card on turn 1 (500-2000 tokens)

- PList reinforcement every 5 turns (50-200 tokens)

- Result: ~8% of context for character, ~80% for conversation

**Multi-character chats:**

- Capsule summaries on first appearance (50-100 tokens each)

- Capsule reinforcement every 5 turns (voice examples with dialog)

- Result: ~15% of context for 5 characters, ~75% for conversation

Without hygiene: 60%+ of tokens consumed by redundant character definitions.

### First Appearance Detection

Characters and NPCs are only defined when they enter the scene. A character added at turn 50 gets their capsule injected immediately — not pre-defined 50 turns prior.

### Adaptive Canon Law

Core world info reinforces every N turns (default: 3) instead of every turn. Canon law is always included, while triggered lore appears only when semantically relevant.

### Relationship Context Filtering

Five emotional dimensions tracked, but only injected when:

- Relationships deviate meaningfully from neutral (>15 points)

- They're semantically relevant to the current conversation

A fight scene won't get "Alice trusts Bob" injected unless trust is relevant to the conflict.

## Real-World Impact

**Without context hygiene:**

- Characters blend into identical voices after 50+ turns

- Character drift: Alice starts using slang, Bob becomes aggressive

- Relationship context bloats prompts with irrelevant dimensions

- Group chat quality degrades with 5+ characters

**With context hygiene:**

- 6+ character group chats work

- Characters stay in character for 200+ turns

- Distinct voices maintained throughout

- Relationships inject only when semantically relevant

- World rules maintained throughout 100+ turn conversations

## Built for Local Deployment

NeuralRP assumes you're running:

- Local LLM via KoboldCpp, TabbyAPI, or Ollama (OpenAI-compatible endpoints)

- Local Stable Diffusion via AUTOMATIC1111 WebUI (optional)

- 12-16GB VRAM with performance mode to queue heavy operations and prevent crashes

All data lives in a single SQLite database (neuralrp.db) with ACID guarantees. No cloud sync, no external dependencies beyond your local inference stack.

### Why SQLite?

- **ACID guarantees** — Atomic, consistent, isolated, durable operations

- **Single file** — Easier backup, no file fragmentation

- **Indexed queries** — Scale to 10,000+ entries without performance loss

- **Smart sync** — Auto-exports to SillyTavern V2 JSON format for compatibility

- **Change history** — 30-day retention with browse/restore functionality

- **Soft delete** — Messages archived, not deleted

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

## Version 1.8.0 — Tag Management

- Tag management for characters and worlds with AND semantics filtering

- Quick filter chips for top 5 most-used tags

- Tag editor with autocomplete suggestions

- Automatic tag extraction from SillyTavern V2 cards

- Normalized tags (lowercase, trimmed, deduplicated)

- Junction table design for many-to-many relationships

- Smart sync with timestamp-based conflict resolution

- Handles 100+ cards without performance degradation

## Credits

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.