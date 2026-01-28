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

One of the driving features of this project was an obsession with prompt hygiene, which is essential if you want chat to work well with a non-frontier LLM. Every design choice and default has a clean, precise prompt to the LLM in mind:

Examples:

- **Relationship tracking** - Five emotional dimensions tracked via semantic embeddings (no LLM calls). Only injects context when (1) relationships are meaningful enough to matter, and (2) they're semantically relevant to the current conversation. A fight scene won't get "Alice trusts Bob" injected unless trust is actually part of the conflict.

- **Adaptive canon law** - Core world info reinforces every N turns (default: 3) instead of every single turn, preventing repetition while keeping characters grounded in the setting. This also allows you to tune what the characters see on the fly.

- **Capsule personas** - Multi-character chats compress full character cards (500-1000 tokens) into 50-100 token summaries, saving 60-80% overhead per character while preserving distinct voices.

- **Semantic world info** - Only retrieves lore semantically relevant to the last 5 messages. A 10,000-entry database still injects ~2k tokens because irrelevant entries stay out.

- **Chat-scoped NPCs** - NPCs exist only in their chat context and don't pollute the global character roster. No "guard from Chapter 2" appearing in unrelated prompts.

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

All content lives in `app/data/`:

```
app/data/
├── neuralrp.db          # SQLite database (characters, chats, world info, embeddings, relationships)
├── characters/          # Exported SillyTavern V2 JSON cards
├── chats/               # Exported chat sessions
└── worldinfo/           # Exported world info JSON
```

## Credits

**Version 1.7.2** - World Info Saving Fix

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.
