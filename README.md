# NeuralRP

NeuralRP is an opinionated roleplay interface for local LLMs. It’s built for users who already understand the basics — temperature, context length, model choice — and want a fast, coherent RP experience without managing dozens of knobs or extensions. NeuralRP integrates KoboldCpp and Stable Diffusion WebUI (AUTOMATIC1111) directly, applies strong narrative defaults, and turns live roleplay into portable, SillyTavern‑compatible character and world cards.

## Features

### Core Functionality

- **KoboldCpp Integration**  
  Seamlessly connect to a local KoboldCpp server for text generation (native or OpenAI-compatible API).

- **Stable Diffusion (A1111) Integration**  
  Generate images inline with chat conversations using Stable Diffusion WebUI (AUTOMATIC1111).

- **SillyTavern Compatible**  
  Accepts character cards and World Info in SillyTavern V2 JSON format, and saves new cards back out in the same format.

- **Narrator Mode**  
  Chat with characters, or select no characters for a pure third-person narrator experience in any world.

- **Multi-Character Support**  
  Chat with multiple characters in the same scene using optimized prompts and capsule personas to keep voices distinct.

### Card & World Factory

- **AI-Powered Character Creation**  
  Generate new character cards directly from chat context. Start from a loose idea, roleplay it out, then convert the conversation into a structured SillyTavern-compatible character card.

- **AI-Powered World Building**  
  Create World Info entries (history, locations, creatures, factions) from conversations. NeuralRP turns freeform worldbuilding into reusable World Info JSON.

- **Automatic Formatting and Quotes**  
  The app handles field structure, escaping, and pulling out example dialogues and first messages, so you don’t have to hand-edit JSON before importing into SillyTavern.

- **Use as a Dedicated Card Factory**  
  Even if you prefer to play in SillyTavern, you can use NeuralRP as a focused environment to generate and refine character/world cards, then import them into other frontends.

### Advanced Features

- **Automatic Summarization**  
  When context usage reaches ~85% of the model’s max, older messages are summarized so their content remains accessible to the LLM after the raw window is exceeded.

- **Token Counter**  
  Monitor token usage for the assembled prompt in real time.

- **Image Generation with Character Tags**  
  Assign a Danbooru-style tag to each character. Use `[CharacterName]` in the positive prompt and NeuralRP will expand it to the configured tag for consistent A1111 generations, even when you’re not actively chatting with that character.

- **Chat Persistence**  
  Save and load previous chat sessions, including associated characters, world, and images.

- **Canon Law System**  
  Mark important World Info entries as **Canon Law** (click to toggle, entries highlight red). Canon Law entries are injected late and prioritized in the context to reduce world drift.

### Customization Options

NeuralRP keeps the UI minimal but exposes the essentials:

- **System Prompt**  
  Customize the global behavior and style (narrator vs character focus, tone, formatting).

- **Temperature Control**  
  Adjust creativity (0.0–1.0+).

- **Reply Length**  
  Set maximum response length in tokens.

- **User Persona**  
  Define your user/player persona, used as part of the context.

- **Context Reinforcement**  
  Control how often character cards or narrator prompts are re-inserted into the context window.

### Mode Selection

- **Narrator Mode**  
  Third-person cinematic narration; the narrator may move all characters and NPCs.

- **Focus Mode**  
  Speak as a specific character in first person; the hidden prompt constrains the LLM to that character’s voice.

- **Auto Mode**  
  Let the LLM automatically select whether to respond as narrator or a specific character based on context, for a more natural roleplay flow.

## Installation

### Prerequisites

- Python 3.8 or higher
- [KoboldCpp](https://github.com/LostRuins/koboldcpp) running locally (e.g. `http://127.0.0.1:5001`) with the API enabled
- [Stable Diffusion WebUI (AUTOMATIC1111)](https://github.com/AUTOMATIC1111/stable-diffusion-webui) running locally (e.g. `http://127.0.0.1:7861`) for image generation

### Quick Start (Windows)

1. Double-click `launcher.bat` to run the application.  
   - The launcher will automatically check for and install dependencies.  
   - On first run, this may take a few minutes.

2. Open your browser and navigate to:  
   `http://localhost:8000`

### Manual Installation

1. Clone or download this repository.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   python main.py
   ```
 +++++++ REPLACE

## Usage

### Basic Chat

1. Load or create a character card (optional) and select a world.  
2. Adjust settings as needed (temperature, reply length, etc.).  
3. Start chatting in the main interface in:
   - Narrator mode, or  
   - Focus mode for a specific character, or  
   - Auto mode.

### Multi-Character Chat

1. Load two or more characters into the session.  
2. Use the mode selector to choose:
   - **Narrator** – Narrator controls the scene and any characters.  
   - **Focus: [Name]** – Speak as a specific character in first person.  
   - **Auto** – Let the LLM decide who should respond / narrate based on context.

NeuralRP automatically uses capsule/summary personas in multi-character mode to keep definitions short and distinct.

### Image Generation

1. In the image prompt, use `[CharacterName]` to reference a character.  
2. The app will automatically expand `[CharacterName]` to the Danbooru tag assigned to that character.  
3. Images are inserted inline into the chat and saved under `app/images/`.

### World Info

1. Navigate to the World Info tab.  
2. Load or create World Info files in SillyTavern-compatible JSON format.  
3. Click entries to mark them as **Canon Law** (they turn red).  
4. Canon Law entries are prioritized and placed late in the prompt to strongly influence generation.

### Creating Characters from Chat

1. Have a conversation that explores a character’s behavior, backstory, and voice.  
2. Use the **Generate Character Card** feature for that chat.  
3. NeuralRP analyzes the conversation and creates a SillyTavern-compatible character card (including description, personality, example messages, and first message).

### Creating World Info from Chat

1. Run a conversation or scenario in a given world.  
2. Use the World Info generator to extract:
   - History / backstory  
   - Locations  
   - Creatures / monsters  
   - Factions / organizations  
3. Select tone (e.g. SFW / Spicy variants) as needed.  
4. Save the generated entries into your world’s World Info file.

## Configuration

### API Endpoints

By default, the app expects:

- KoboldCpp API: `http://127.0.0.1:5001`
- Stable Diffusion WebUI API: `http://127.0.0.1:7861`

To change these, modify the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "kobold_url": "http://127.0.0.1:5001",
    "sd_url": "http://127.0.0.1:7861",
    # ...
}

## Project Structure

```
neuralrp/
├── main.py                 # Main FastAPI application
├── launcher.bat            # Windows launcher script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── app/
│   ├── index.html         # Frontend interface
│   ├── data/
│   │   ├── characters/    # Character card storage
│   │   ├── chats/         # Saved chat sessions
│   │   └── worldinfo/     # World Info entries
│   └── images/            # Generated images
└── frontend/              # Frontend build files
```

## Requirements

```
fastapi
uvicorn[standard]
httpx
pydantic
```

## Troubleshooting

### "Python is not installed"
- Install Python 3.8+ from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

### "Failed to install dependencies"
- Ensure you have a stable internet connection
- Try running `pip install --upgrade pip` first
- Check if you're behind a corporate firewall that blocks pip

### Can't connect to Koboldcpp/SD
- Ensure Koboldcpp is running on port 5001
- Ensure Stable Diffusion WebUI is running on port 7861
- Check that `--api` flag is enabled in Koboldcpp
- Check that `--api` flag is enabled in Stable Diffusion WebUI

### Images not generating
- Verify Stable Diffusion is running and accessible
- Check that the image directory (`app/images/`) exists and is writable
- Review the server logs for specific error messages

## Contributing

This project is designed to be extended and customized. Feel free to fork, modify, and improve it for your needs.

## License

This project is provided as-is for educational and personal use.

## Credits

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Frontend integration with vanilla JavaScript
- Compatible with [SillyTavern](https://github.com/SillyTavern/SillyTavern) card formats
- Integrates with [Koboldcpp](https://github.com/Koboldcpp/Koboldcpp-Client) and [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
