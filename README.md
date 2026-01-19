# NeuralRP

**The only roleplay tool that runs LLM + Stable Diffusion together on a 12GB GPU.**

NeuralRP combines text generation and image synthesis in one optimized package. While other tools make you choose between visual richness and memory overhead, NeuralRP does both—with automatic VRAM management that keeps everything responsive.

**What makes it different:**

- **Semantic Relationship Tracking** - Characters automatically remember how they feel about each other across conversations. No manual notes, no forgetting after summarization. Alice's growing trust in Bob, her conflict with Marcus—it all persists and evolves naturally.

- **Integrated Image Generation** - Generate character portraits, scene illustrations, and visual reactions without switching apps. Per-character tags mean you type `[Alice]` once instead of "blonde hair, blue eyes, medieval armor" every time.

- **12GB Optimization** - Performance Mode intelligently queues heavy operations so you can run Mistral 7B and Stable Diffusion on a gaming GPU (RTX 3060, 4060 Ti) without crashes or freezes.

- **Persistent World State** - SQLite-backed architecture with semantic world info retrieval. Your lore stays consistent across branches, forks, and 100+ message sessions.

Built for users tired of SillyTavern's complexity and RisuAI's forgetfulness. If you want coherent narratives with rich visuals on modest hardware, NeuralRP is for you.

---

## Why NeuralRP?

**You're running into these problems:**

| Problem | Other Tools | NeuralRP |
|---------|-------------|----------|
| LLM + SD crashes on 12GB VRAM | Choose one or manually swap models | Automatic queuing, both run together |
| Characters forget relationships | Manual notes or prompt injection | Semantic tracker evolves 5 emotional dimensions |
| Image prompts are repetitive | Type full description every time | Per-character tags, reference with `[Name]` |
| Branching breaks world state | SillyTavern's branches lose context | Fork-safe SQLite preserves relationships + lore |
| Configuration overwhelm | 100+ settings, 50+ extensions | Opinionated defaults, ships ready to use |

If you've ever thought *"I just want this to work without fighting it,"* you're the target audience.

---

## Design Philosophy

**Strong defaults beat endless configurability.**

NeuralRP doesn't try to be everything. It does one thing exceptionally well: local roleplay with integrated visuals and persistent world state. We expose what matters (temperature, personas, relationships) and hide what doesn't (cache sizes, embedding models).

This is opinionated software. If you want to tweak everything, SillyTavern exists. If you want something that just works, you're in the right place.

---

## Core Features

### Play
- **Semantic Relationship Tracker** - Characters remember emotional states with 5-dimensional tracking (trust, bond, conflict, power, fear). Analyzes every 10 messages using embeddings, no LLM calls required.
- **Multi-Character Chat** - Narrator (third-person), Focus (first-person), and Auto modes. Capsule personas keep voices distinct.
- **Branching Narratives** - Fork from any message, explore alternate timelines independently with full state preservation.
- **Infinite Sessions** - Automatic summarization when context fills. Chat for 100+ messages without losing coherence.

[→ Technical details on relationship tracking](docs/TECHNICAL.md#relationship-tracking)

### Build
- **Dual-Source Card Generation** - Create character cards from chat history OR plain-text descriptions using LLM assistance.
- **World Info with Canon Law** - Mark critical lore as always-included. Regular entries use semantic search for intelligent retrieval.
- **Live Editing** - AI-generated content appears in editable textboxes before saving to database.
- **SillyTavern Compatible** - Import/export character cards and world info seamlessly.

### Visualize
- **Integrated Stable Diffusion** - Generate images inline during chat without switching tools.
- **Per-Character Tags** - Assign Danbooru tags once, reference with `[CharacterName]` in prompts forever.
- **Inpainting** - Modify existing images with mask-based regeneration.
- **Image Metadata** - All generation parameters stored for reproducibility.

### Data Recovery
- **Change History Browser** - Full-screen interface for browsing and restoring changes across characters, world info, and chats.
- **30-Second Undo Toast** - Safety net for accidental deletions with one-click restoration.
- **Soft Delete System** - Messages marked as archived instead of deleted. Preserves history after summarization with persistent IDs.

### Performance
- **Automatic Resource Management** - Queues heavy operations (image generation, embeddings) while allowing light tasks (chat) to proceed.
- **Semantic World Info** - Intelligent lore retrieval with vector search, falls back to keyword matching.
- **Persistent Vector Embeddings** - Computed once, stored in sqlite-vec, survive restarts.
- **SQLite Architecture** - ACID transactions prevent data corruption. Scales to 10,000+ entries.

---

## Hardware Requirements

**Recommended:**
- 12-16GB VRAM GPU (NVIDIA/AMD)
- Python 3.8+
- KoboldCpp (for LLM inference)
- AUTOMATIC1111 WebUI (for image generation)

**Minimum:**
- 8GB VRAM (with Performance Mode enabled)
- Supports: KoboldCpp, Ollama, Tabby (OpenAI-compatible endpoints)

---

## Quick Start

1. **Clone repository**
   ```bash
   git clone https://github.com/neuralrp/neuralrp.git
   cd neuralrp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure endpoints** (edit `main.py` if needed)
   - KoboldCpp API: `http://127.0.0.1:5001`
   - Stable Diffusion API: `http://127.0.0.1:7861`

4. **Run NeuralRP**
   ```bash
   python main.py
   ```
   Or use `launcher.bat` on Windows

5. **Open browser**
   Navigate to `http://localhost:8000`

---

## What's Included

- **SQLite-backed storage** for characters, chats, world info, and images
- **Semantic search** using vector embeddings (sqlite-vec) for intelligent lore retrieval
- **Semantic relationship tracker** with five-dimensional emotional state analysis
- **Entity ID system** for unique identification preventing name collisions
- **Relationship history** with 20-snapshot tracking per relationship
- **Change history data recovery UI** for browsing, filtering, and restoring changes
- **Soft delete system** for preserving message history with persistent IDs
- **Automatic performance management** for LLM + SD on same GPU
- **Adaptive connection monitoring** that reduces overhead during stable operation
- **Branch management** with origin tracking and timeline independence
- **SillyTavern compatibility** for character cards and world info

---

## Documentation

- [**Technical Documentation**](docs/TECHNICAL.md) - Deep dives into context assembly, SQLite architecture, performance mode, semantic search, relationship tracking
- [**Changelog**](CHANGELOG.md) - Version history and release notes
- [**Soft Delete Implementation**](docs/SOFT_DELETE_IMPLEMENTATION.md) - Technical details of message archive system
- [**Message ID Fix**](docs/MESSAGE_ID_FIX.md) - Database migration for persistent message IDs
- [**Contributing**](CONTRIBUTING.md) - For developers (coming soon)

---

## Troubleshooting

**Can't connect to KoboldCpp/Stable Diffusion?**
- Ensure services are running with `--api` flags enabled
- Check ports: KoboldCpp (5001), SD WebUI (7861)
- Verify firewall isn't blocking localhost connections

**Images not generating?**
- Confirm Stable Diffusion WebUI is running and accessible
- Check `app/images/` directory exists and is writable
- Review server logs for specific error messages

**Performance issues with LLM + SD together?**
- Enable Automatic Performance Mode in Settings
- Consider reducing context reinforcement frequency
- Generate images outside of heavy chat sessions

---

## Data Structure

All content lives in `app/data/` and can be backed up or version-controlled:

```
app/data/
├── neuralrp.db          # SQLite database (characters, chats, world info, embeddings, relationships)
│                        # - includes soft delete support (summarized column)
│                        # - relationship tracking tables (relationships, relationship_history, entities)
│                        # - entity ID system for unique identification
│                        # - change log for undo/redo support
├── characters/          # Exported SillyTavern V2 JSON cards
├── chats/               # Exported chat sessions
└── worldinfo/           # Exported world info JSON
```

---

## Credits

**Version 1.6.0** (2026-01-19) - Semantic Relationship Tracking, Entity ID System, Enhanced Change History UI

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.
