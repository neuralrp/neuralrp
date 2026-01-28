![Screenshot 2026-01-28 075248](https://github.com/user-attachments/assets/339e9fc7-ff88-4c35-860b-71f3b640e1a5)
# NeuralRP

I created this because I wanted a local AI roleplay tool that was both simple, but also incorporated character and world cards. From there, I built (using AI) a chat and visual platform with quite a bit going on under the covers to optimize RP with 12-16 GB VRAM cards. Very niche, but if that niche is you, there's some unique features built into this app worth checking out.

## What Makes It Different

**Relationship Tracking**

Characters remember how they feel about each other, and the user. Five emotional dimensions (trust, bond, conflict, power, fear) track automatically.

**NPC Management**

Create, manage, and promote non-player characters within individual chats. NPCs exist only in their chat context, preventing cross-contamination. Fork chats and NPCs branch independently with their own relationship states. Promote NPCs to global characters when you want to reuse them.

**12GB VRAM Optimization**

Performance Mode queues heavy operations while allowing light tasks to proceed. Run KoboldCCP and Stable Diffusion together on 12 GB vRAM without crashes.

**Persistent World State**

Branch safely, summarize without breaking relationships, and recover from accidental deletions with full change history.

**SillyTavern Card Optimized**

Import your cards automatically by dropping them in the character and world folders. Create PList-optimized cards from actual dialog in the chat or from writing plain-text sentences and the app converts it for you.

## Other Cool Features

- **Multi-Character Chat** - Narrator (third-person), Focus (first-person), and Auto modes, so the AI knows who you're talking to.
- **Automatic Summarization** - Continue conversations beyond context limits without losing coherence. Key for locally running chat, you can keep going indefinitely.
- **Search Function** - Vector-based semantic search finds relevant lore automatically.
- **Live Editing** - AI-generated content appears in editable textboxes before saving to database.
- **Integrated Stable Diffusion** - Generate images inline during chat without switching tools.
- **Inpainting Support** - Deep inpainting features taking advantage of A1111, without ever leaving NeuralRP's UI
- **Per-Character Tags** - Assign Danbooru tags once, reference with `[CharacterName]` in prompts.
- **Image Metadata** - All generation parameters stored for reproducibility.
- **Change History** - Restore across characters, world info, and chats.
- **Undo Safety** - 30-second undo toast after deletions with one-click restoration.
- **Soft Delete** - Messages archived instead of deleted. Preserves history after summarization with persistent IDs.
- **Export for LORA Training** - Export to JSON already optimized for Unsloth training

## Prompt Obsession

One of the driving features of this project was an obsession with prompt hygiene, which is essential if you want chat to work well with a local LLM. Every design choice and default has a clean, precise prompt to the LLM in mind—even with modern 8k+ context windows.

**The Problem: Attention Decay, Not Token Overflow**

With modern LLMs (Llama 3, Nemo) and 12GB+ VRAM setups, you can run 8192+ token contexts with image generation simultaneously. Token overflow is no longer the crisis it was 2 years ago. However, context hygiene still matters because:

- **Attention decay**: Even with 8k tokens, content from turns 1-10 receives significantly less attention than turns 80-90. LLMs naturally prioritize recent context, causing character definitions to "fade" over long conversations
- **Quality vs quantity**: Just because you CAN fit 5 full character cards (5000 tokens) doesn't mean you SHOULD. Redundant repetition wastes context space that could drive better narrative
- **Scalability**: Larger contexts enable richer experiences (6+ character group chats, 200+ turn conversations), but intelligent injection is what makes those experiences viable
- **Character drift**: Without reinforcement, Alice starts using slang, Bob becomes aggressive, and characters blend together after 50+ turns—even in large context windows

**Examples:**

- **Smart character injection strategies** - Different approaches for single vs multi-character chats:
  - Single character: Full card on turn 1 (one-time), then PList reinforcement every 5 turns (behavioral constraints)
  - Multi-character: Capsule summaries on first appearance, then capsule reinforcement every 5 turns (voice examples)
  - Why? Even with 8k tokens, capsules (50-100 tokens each) enable 5-6 character group chats vs 3-4 with full cards, while preserving distinct voices through dialog examples

- **First appearance detection** - Characters and NPCs are only defined when they actually enter scene, not before or after. A character added mid-chat (turn 50) gets their capsule injected immediately, preventing the need to pre-define them 50 turns prior when they weren't relevant

- **Relationship tracking** - Five emotional dimensions tracked via semantic embeddings (no LLM calls). Only injects context when (1) relationships are meaningful enough to matter (deviates >15 points from neutral), and (2) they're semantically relevant to current conversation. A fight scene won't get "Alice trusts Bob" injected unless trust is actually relevant to conflict. Adaptive tier-3 filtering prevents irrelevant relationship dimensions from bloating prompts

- **Adaptive canon law** - Core world info reinforces every N turns (default: 3) instead of every single turn, preventing repetition while keeping characters grounded in the setting. This also allows you to tune what the characters see on the fly. Canon law is separate from triggered lore—always included + reinforced, while matched lore appears only when semantically relevant to recent context

- **Capsule personas** - Multi-character chats compress full character cards (500-1000 tokens) into 50-100 token summaries with dialog examples, saving 80-90% overhead per character while preserving distinct voices. In 5-6 character group chats, this difference is what makes the experience viable vs character-voice-blending chaos

- **Semantic world info** - Only retrieves lore semantically relevant to the last 5 messages. A 10,000-entry database just injects what's needed because irrelevant entries stay out. Supports both quoted keys (`"The Great Crash Landing"` - exact phrase match, prevents false positives) and unquoted keys (`dragon` - semantic search, catches plurals and synonyms naturally)

- **Character introduction (one-time)** - Full card/capsule injected when character first appears in chat, then periodically via reinforcement
- **Periodic reinforcement (every N turns)** - Minimal reminders to prevent drift (PList for single char, capsules for multi-char). Separate from introduction—reinforcement re-anchors behavior, doesn't redefine character

- **Configurable character and world card insertion** - Configure how often character and world cards appear in the prompt in settings, with tested defaults already populated (default: 5 turns character, 3 turns world)

- **Token-efficient architecture** - Every feature designed to maximize narrative richness within 8192+ token context:
  - Single char: ~8% tokens (full card + PList), ~80% conversation
  - Multi-char (5): ~15% tokens (capsules), ~75% conversation
  - World info: ~10% tokens (canon law + matched lore)
  - Without hygiene: 60%+ tokens consumed by redundant character/world definitions

**Real-World Impact:**

Without prompt hygiene:
- Characters blend into identical voices after 50+ turns (even in large contexts)
- Character drift: Alice starts using slang, Bob becomes aggressive without changes to their cards
- Relationship context bloats prompts with irrelevant dimensions ("Alice trusts Bob" injected during fight scenes)
- Group chat quality degrades with 5+ characters (no voice differentiation)

With prompt hygiene:
- 6+ character group chats work (capsules scale efficiently)
- Characters stay in character for 200+ turns (reinforcement re-anchors behavior)
- Distinct voices maintained throughout (capsule dialog examples + PList)
- Relationships inject only when semantically relevant (adaptive filtering)
- World rules maintained throughout 100+ turn conversations

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

1. **Clone repository**
   ```bash
   git clone https://github.com/neuralrp/neuralrp.git
   cd neuralrp
   ```

2. **Run NeuralRP**
   ```bash
   launcher.bat    # On Windows - handles everything automatically
   ```
   Or for other systems:
   ```bash (or double-click in folder)
   pip install -r requirements.txt
   python main.py
   ```

3. **Open browser**
   Navigate to `http://localhost:8000`
   
   Configure your LLM and image generation endpoints in the app's Settings panel.

## Documentation

- [**Technical Documentation**](docs/TECHNICAL.md) - Implementation details
- [**Changelog**](CHANGELOG.md) - Version history

## Data Structure

```
app/
├── data/
│   ├── neuralrp.db          # SQLite database (characters, chats, world info, embeddings, relationships)
│   ├── characters/          # SillyTavern V2 JSON cards
│   ├── chats/               # Chat sessions
│   └── worldinfo/           # World Card JSON
└── images/                  # Generated images
```

**Note:** Exported chats are downloaded directly to your browser's Downloads folder as JSON files (for LLM training), not saved to any server directory.

## Credits

**Version 1.7.3** - Context Hygiene & Edit Synchronization
- Character, NPC, and world info card edits sync immediately to active chats
- Quoted vs unquoted world info keys for precision/flexibility balance
- Enhanced character/NPC/world injection strategies for 8k+ token contexts

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.
