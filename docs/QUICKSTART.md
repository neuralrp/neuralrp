# NeuralRP Quickstart Guide

Get up and running with NeuralRP in minutes. This guide covers installation, setup, and basic usage.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing Backends](#installing-backends)
3. [Installing NeuralRP](#installing-neuralrp)
4. [First-Time Setup](#first-time-setup)
5. [Basic Usage](#basic-usage)
6. [Generating Your First Image](#generating-your-first-image)
7. [Next Steps](#next-steps)

---

## Prerequisites

**Hardware Requirements:**
- **Recommended:** 12-16GB VRAM GPU (NVIDIA/AMD)
- **Minimum:** 8GB VRAM (with Performance Mode enabled)

**Software Required:**
- Python 3.8+ (for NeuralRP)
- Python 3.10.6 specifically for A1111 (newer versions don't support torch)

**What You'll Need:**
1. KoboldCpp (LLM inference backend)
2. AUTOMATIC1111 Stable Diffusion WebUI (image generation backend)
3. NeuralRP (this repository)

---

## Installing Backends

### Step 1: Install KoboldCpp (LLM Backend)

1. Download the latest `koboldcpp.exe` from [KoboldCpp Releases](https://github.com/LostRuins/koboldcpp/releases)
2. Place it in a folder (e.g., `C:\KoboldCpp\`)
3. Double-click `koboldcpp.exe` to launch
4. Click "Load Model" and select a GGUF model file
5. **Default API URL:** `http://localhost:5001`

**Recommended GGUF Models:**
- **L3-8B-Stheno-v3.2** - Fast, unfiltered
- **Tiefighter 13B** - Versatile, balanced (12GB VRAM)
- Search for "GGUF" on [Hugging Face](https://huggingface.co)

**Performance Tips:**
- NVIDIA: Run with `--usecuda` flag
- AMD/Other: Use `--usevulkan` for Vulkan support
- Add `--gpulayers N` to offload N layers to VRAM

---

### Step 2: Install AUTOMATIC1111 Stable Diffusion WebUI

1. **Install Python 3.10.6** (Critical: Newer versions don't support torch)
   - Download from [Python.org](https://www.python.org/downloads/release/python-3106/)
   - Check "Add Python to PATH" during installation

2. **Install Git** (if not already installed)
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Clone the repository:**

   **Windows:**
   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

   **Linux:**
   ```bash
   sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

4. **Run A1111:**
   - Windows: Double-click `webui-user.bat`
   - Linux: Run `./webui.sh`

5. First run will install dependencies (takes several minutes)
6. **Default API URL:** `http://localhost:7860`

**Download SD Models:**
- **Anything v3** - Anime style, versatile
- **PerfectDeliberate V2** - Realistic, detailed
- **Dreamshaper SDXL** - High quality, modern
- Place `.safetensors` files in `stable-diffusion-webui/models/Stable-diffusion/`

---

## Installing NeuralRP

### Option 1: Using Launcher (Windows - Recommended)

```bash
git clone https://github.com/neuralrp/neuralrp.git
cd neuralrp
launcher.bat
```

The launcher handles dependency installation and startup automatically.

### Option 2: Manual Installation (All Systems)

```bash
git clone https://github.com/neuralrp/neuralrp.git
cd neuralrp
pip install -r requirements.txt
python main.py
```

### Access NeuralRP

Once running, open your browser and navigate to:

**http://localhost:8000**

> **Note:** First launch downloads a ~400MB AI model for semantic search (5-10 minutes). This is normal. Subsequent launches are fast.

---

## First-Time Setup

### Configure LLM Backend

1. **Open NeuralRP** in your browser (`http://localhost:8000`)
2. Click **Settings (⚙️)** in the top header menu
3. Scroll to "API Configuration" section
4. **Kobold URL:** `http://127.0.0.1:5001`
5. Click **"Update & Test"** button
6. Wait for **green status indicator** (means connected)

### Configure Image Generation Backend

1. In Settings, under "API Configuration"
2. **SD API URL:** `http://127.0.0.1:7860`
3. Click **"Update & Test"** button
4. Wait for **green status indicator**

Both services must show green status before you can use NeuralRP effectively.

---

## Basic Usage

### Starting Your First Chat

**Narrator Mode (Default):**
- When no characters selected, you're in "Narrator Mode" (blue indicator in header)
- Type your prompt in the input field at the bottom
- Press **Enter** or click the blue paper plane icon
- The AI generates a response as the narrator

**Adding a Character:**
1. Click **Characters (👥)** in the header
2. Click on a character card to activate it (blue border appears)
3. Active characters appear as **pill badges** in the header center
4. To remove a character, click the **×** on its pill badge

**Chat Modes:**
Use the dropdown menu in the bottom input area:
- **🤖 Auto** - AI selects best responder (character or narrator)
- **🎭 Narrator** - Story narration mode
- **👤 [Character Name]** - Force specific character to respond

**Example Workflow:**
```
1. Add "Jim the Bartender" character
2. Select "Auto" mode
3. Type: "I walk into the bar and sit on a stool."
4. Jim responds: "Well, well, well... looks like we've got a new face..."
```

### Understanding Context Summarization

NeuralRP automatically summarizes long conversations to maintain performance while preserving story continuity. When your chat reaches 80% context, old messages are compressed into scene summaries. This allows infinite conversations without losing context. You can view and edit summaries in the Summaries panel (📋 icon in header).

---

## Generating Your First Image

### Manual Generation

1. Click the **purple wand icon** (toggle) to expand the Image Panel (bottom)
2. **Vision Input:** Enter your prompt (e.g., "cozy tavern interior, candlelight")
3. Adjust settings (Steps: 20-30, Scale: 7-12, Resolution: 512×512)
4. Click **"Generate Vision Sequence"**
5. Image appears in chat

### Snapshots (Auto-Generated Scenes)

Click the **📸 camera icon** in chat to auto-generate a scene image based on your conversation context. The AI analyzes recent messages to extract location, action, activity, dress, and expression, then generates an image matching the scene.

---

## Next Steps

1. **Try Demo Files** - Copy demo character and world from `demo/` folder to `app/data/characters/` and `app/data/worldinfo/`, then Refresh. See [User Guide](USER_GUIDE.md) for detailed setup.

2. **Import Your Own Cards** - Drop SillyTavern cards into `app/data/characters/` and `app/data/worldinfo/`

3. **Explore Features** - See [User Guide](USER_GUIDE.md) for:
   - Character creation (Gen Card)
   - World lore extraction (Gen World)
   - Danbooru tag generator (one-click visual character matching)
   - Snapshots, favorites, inpainting
   - Branching, search, export

4. **Experiment with Settings** - Adjust context size, summarization threshold, and temperature in Settings panel

Happy roleplaying! 🎭
