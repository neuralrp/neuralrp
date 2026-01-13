# NeuralRP

**Hardware-efficient roleplay for local LLMs.** NeuralRP is built for experienced users who want SillyTavern-quality RP on modest GPUs (12-16GB VRAM). No configuration sprawl, no extension management—just integrated LLM + Stable Diffusion with persistent world state and strong narrative defaults.

If you already know what temperature and context length mean, and you're tired of tweaking 100+ settings to get coherent output, NeuralRP is for you.

---

## Why NeuralRP?

NeuralRP solves specific problems for local LLM users:

- **Tired of SillyTavern's 100+ extensions?** → NeuralRP ships with opinionated defaults that just work
- **Running LLM + SD on 12GB VRAM?** → Automatic Performance Mode queues operations intelligently
- **Want branching narratives without Git?** → File-based branches with UI management
- **Character cards drift over time?** → Canon Law + semantic World Info keep lore consistent
- **Building cards for other tools is tedious?** → Dual-source card generation (from chat or text)
- **Image prompts are repetitive?** → Per-character Danbooru tags, reference with `[CharacterName]`

---

## Core Features

### Play
- **Chat Modes**: Narrator (third-person GM), Focus (first-person character), Auto (AI chooses speaker)
- **Multi-Character Support**: Capsule personas keep voices distinct without prompt bloat
- **Branching**: Fork from any message, independent timelines, rename/delete/switch via UI
- **Persistent Memory**: Automatic summarization when context fills, sessions run indefinitely

### Build
- **Dual-Source Card Generation**: Create character cards from chat history OR plain-text descriptions
- **World Info with Canon Law**: Mark critical lore as always-included, regular entries use semantic search
- **Live Editing**: AI-generated content appears in editable textboxes before saving
- **SillyTavern Compatible**: Import/export character cards and world info seamlessly

### Visualize
- **Integrated Stable Diffusion**: Generate images inline during chat
- **Per-Character Tags**: Assign Danbooru tags once, use `[CharacterName]` in prompts forever
- **Inpainting**: Modify existing images with mask-based regeneration
- **Image Metadata**: All generation parameters stored for reproducibility

### Performance
- **Automatic Resource Management**: Queues heavy operations, allows light tasks to proceed
- **Semantic World Info**: Intelligent lore retrieval with vector search, falls back to keywords
- **SQLite-Backed Architecture**: ACID transactions prevent data corruption, scales to 10,000+ entries
- **Token Monitoring**: Real-time context usage tracking with configurable summarization thresholds

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

1. **Clone the repository**
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

## Philosophy

NeuralRP is opinionated. It doesn't try to be everything to everyone. It does one thing exceptionally well: **local, hardware-efficient roleplay with integrated image generation and persistent world state.**

We believe strong defaults beat endless configurability. The interface exposes what matters (temperature, context, personas) and hides what doesn't (cache sizes, embedding models, lock patterns). If you want to tweak those, you can—check the technical docs.

This is the focused alternative to SillyTavern: fewer knobs, more immersion.

---

## What's Included

- **SQLite-backed storage** for characters, chats, world info, and images
- **Semantic search** using vector embeddings for intelligent lore retrieval
- **Automatic performance management** for LLM + SD on the same GPU
- **Adaptive connection monitoring** that reduces overhead during stable operation
- **Branch management** with origin tracking and timeline independence
- **SillyTavern compatibility** for character cards and world info

---

## Documentation

- [**Technical Documentation**](docs/TECHNICAL.md) - Deep dives into context assembly, SQLite architecture, performance mode, semantic search
- [**Changelog**](CHANGELOG.md) - Version history and release notes
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
├── neuralrp.db          # SQLite database (characters, chats, world info, embeddings)
├── characters/          # Exported SillyTavern V2 JSON cards
├── chats/               # Exported chat sessions
└── worldinfo/           # Exported world info JSON

app/images/              # Generated images from Stable Diffusion
```

---

## Credits

Built with [FastAPI](https://fastapi.tiangolo.com/) • Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) • Integrates [KoboldCpp](https://github.com/LostRuins/koboldcpp) and [AUTOMATIC1111 Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## License

Provided as-is for educational and personal use. See [LICENSE](LICENSE) for details.
