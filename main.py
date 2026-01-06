from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import time
import base64
import httpx
import re
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "kobold_url": "http://127.0.0.1:5001",
    "sd_url": "http://127.0.0.1:7861",
    "system_prompt": "Write a highly detailed, creative, and immersive response. Stay in character at all times."
}

# Multi-Character Mode Instruction Templates
NARRATOR_INSTRUCTION = """You are the narrator of an interactive story.

Write in third person, describing actions, thoughts, and scenes cinematically.

When characters speak, always use the format:
Name: "dialogue line"

Any character may speak when it makes sense for the scene (answering questions addressed to them, reacting emotionally, or interjecting in conversations).

Keep attribution clear so the reader always knows who is talking.

Do not write as the user. Do not break character or comment about being an AI.

Keep the reply to a reasonable length (no more than 4–6 short paragraphs of narration plus dialogue).

Describe the user's character's actions only if they explicitly wrote them; otherwise, wait for user input before moving them.

Characters only know what they've witnessed or been told in the scene (no metagaming)."""

FOCUSED_TEMPLATE = """In this reply, you are speaking only as {CharacterName}, in first person.

Write {CharacterName}'s thoughts, feelings, and spoken words.

Do not write narration outside {CharacterName}'s perspective.

Do not write lines or actions for any other characters.

Respond directly to what just happened in the previous message, in {CharacterName}'s own voice and style.

{CharacterName} only knows what they've witnessed or been told in the scene.

Keep the reply under 3–5 paragraphs."""

def get_continue_hint(mode: str, last_user_message: str = "") -> str:
    """Generate a context-aware continue hint for narrator mode."""
    if mode == "narrator" or not mode.startswith("focus:"):
        if last_user_message:
            truncated = last_user_message[:100] + "..." if len(last_user_message) > 100 else last_user_message
            return f"[Continue the scene. The user just said/did: \"{truncated}\". Decide how the characters react.]"
        return "[Continue the scene from the last message. As narrator, decide which characters speak or act next based on what just happened.]"
    # Focused mode - no continue hint needed
    return ""

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app/data")
IMAGE_DIR = os.path.join(BASE_DIR, "app/images")

for d in ["characters", "worldinfo", "chats"]:
    os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Mount static files and images
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Models
class ChatMessage(BaseModel):
    role: str
    content: str
    id: int
    speaker: Optional[str] = None
    image: Optional[str] = None

class PromptRequest(BaseModel):
    messages: List[ChatMessage]
    characters: List[Dict[str, Any]] = []
    world_info: Optional[Dict[str, Any]] = None
    settings: Dict[str, Any] = {}
    summary: Optional[str] = ""
    mode: Optional[str] = "narrator"  # "narrator" or "focus:{CharacterName}"

class SDParams(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    sampler_name: str = "Euler a"
    scheduler: str = "Automatic"

class CardGenRequest(BaseModel):
    char_name: str
    field_type: str
    context: str

class CapsuleGenRequest(BaseModel):
    char_name: str
    description: str
    depth_prompt: str = ""

class WorldGenRequest(BaseModel):
    world_name: str
    section: str # history, locations, creatures, factions
    tone: str # sfw, spicy, veryspicy
    context: str

class WorldSaveRequest(BaseModel):
    world_name: str
    plist_text: str

# Token counting helper
async def get_token_count(text: str):
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{CONFIG['kobold_url']}/api/extra/tokencount",
                json={"prompt": text},
                timeout=10.0
            )
            return res.json().get("value", 0)
        except:
            # Fallback to rough estimate if API fails
            return len(text) // 4

# Prompt Construction Engine
def construct_prompt(request: PromptRequest):
    settings = request.settings
    system_prompt = settings.get("system_prompt", CONFIG["system_prompt"])
    user_persona = settings.get("user_persona", "")
    summary = request.summary or ""
    mode = request.mode or "narrator"
    
    # Determine chat mode
    is_group_chat = len(request.characters) >= 2
    is_single_char = len(request.characters) == 1
    is_narrator_mode = not request.characters
    
    # Collect active character names early for mode instruction
    active_names = []
    for char_obj in request.characters:
        data = char_obj.get("data", {})
        active_names.append(data.get("name", "Unknown"))
    
    # === 1. SYSTEM PROMPT ===
    narrator_instruction = " Act as a Narrator. Describe the world and speak for any NPCs the user encounters."
    if is_narrator_mode:
        if "act as a narrator" not in system_prompt.lower():
            system_prompt += narrator_instruction
    
    full_prompt = f"### System: {system_prompt}\n"
    
    # === 2. MODE INSTRUCTIONS (moved up - sets interpretation context) ===
    if is_group_chat:
        if mode.startswith("focus:"):
            focus_char = mode.split(":", 1)[1]
            full_prompt += f"\n[MODE: FOCUS-{focus_char}]\n"
            full_prompt += FOCUSED_TEMPLATE.format(CharacterName=focus_char) + "\n"
        else:
            full_prompt += "\n[MODE: NARRATOR]\n"
            full_prompt += NARRATOR_INSTRUCTION + "\n"
    elif is_single_char:
        # Single character also gets mode instruction now
        char_name = active_names[0]
        if mode == "narrator":
            full_prompt += "\n[MODE: NARRATOR]\n"
            full_prompt += NARRATOR_INSTRUCTION + "\n"
        else:
            # Default to focused mode for single character
            full_prompt += f"\n[MODE: FOCUS-{char_name}]\n"
            full_prompt += FOCUSED_TEMPLATE.format(CharacterName=char_name) + "\n"
    # Pure narrator mode (0 characters) already has instruction appended to system prompt
    
    # === 3. LONG-TERM CONTEXT (Summary) ===
    if summary:
        full_prompt += f"### Long-Term Context (Summary of earlier turns):\n{summary}\n"
    
    # === 4. USER PERSONA ===
    if user_persona:
        full_prompt += f"### Your Character Description:\n{user_persona}\n"

    # === 5. WORLD KNOWLEDGE (moved up - world context frames characters) ===
    canon_law_entries = []
    if request.world_info:
        recent_text = " ".join([m.content for m in request.messages[-5:]]).lower()
        entries = request.world_info.get("entries", {})
        triggered_lore = []
        
        if isinstance(entries, dict):
            for uid, entry in entries.items():
                # Separate Canon Law (will be pinned later)
                if entry.get("is_canon_law"):
                    canon_law_entries.append(entry.get("content", ""))
                    continue

                keys = entry.get("key", [])
                if any(k.lower() in recent_text for k in keys):
                    triggered_lore.append(entry.get("content", ""))
        
        if triggered_lore:
            full_prompt += "### World Knowledge:\n" + "\n".join(triggered_lore) + "\n"

    # === 6. CHARACTER PROFILES (characters exist in the world context) ===
    reinforcement_chunks = []
    
    for char_obj in request.characters:
        data = char_obj.get("data", {})
        name = data.get("name", "Unknown")
        description = data.get("description", "")
        depth_prompt = data.get("extensions", {}).get("depth_prompt", {}).get("prompt", "")
        multi_char_summary = data.get("extensions", {}).get("multi_char_summary", "")
        
        if is_group_chat and multi_char_summary:
            # Group chat mode: use compact capsule summary
            full_prompt += f"### [{name}]: {multi_char_summary}\n"
            # Use capsule for reinforcement too (shorter)
            reinforcement_chunks.append(f"[{name}]: {multi_char_summary}")
        else:
            # Single character or no capsule: use full profile
            full_prompt += f"### Character Profile: {name}\n{description}\n"
            if depth_prompt:
                full_prompt += f"### Context for {name}: {depth_prompt}\n"
                reinforcement_chunks.append(depth_prompt)

    # === 7. CHAT HISTORY (with reinforcement) ===
    full_prompt += "\n### Chat History:\n"
    reinforce_freq = settings.get("reinforce_freq", 0)
    
    for i, msg in enumerate(request.messages):
        # Filter out meta-messages like "Visual System" if they exist
        if msg.speaker == "Visual System":
            continue

        # Reinforcement logic every X turns
        if reinforce_freq > 0 and i > 0 and i % reinforce_freq == 0:
            if reinforcement_chunks:
                # Character reinforcement
                full_prompt += "[REINFORCEMENT: " + " | ".join(reinforcement_chunks) + "]\n"
            elif is_narrator_mode:
                # Narrator reinforcement
                full_prompt += f"[REINFORCEMENT: {narrator_instruction.strip()}]\n"
        
        speaker = msg.speaker or ("User" if msg.role == "user" else "Narrator")
        full_prompt += f"{speaker}: {msg.content}\n"

    # === 8. CANON LAW (pinned for recency bias - right before generation) ===
    if canon_law_entries:
        full_prompt += "\n### Canon Law (World Rules):\n" + "\n".join(canon_law_entries) + "\n"

    # === 9. CONTINUE HINT + LEAD-IN ===
    if is_single_char:
        # Single character mode
        char_name = active_names[0]
        if mode == "narrator":
            # Narrator mode for single char - use narrator lead-in
            last_user_msg = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            continue_hint = get_continue_hint(mode, last_user_msg)
            if continue_hint:
                full_prompt += f"\n{continue_hint}\n"
            full_prompt += "Narrator:"
        else:
            full_prompt += f"{char_name}:"
    elif is_group_chat:
        # Multi-character mode - use mode to determine lead-in
        if mode.startswith("focus:"):
            # Focused mode: specific character speaks
            focus_char = mode.split(":", 1)[1]
            full_prompt += f"\n{focus_char}:"
        else:
            # Narrator mode: add continue hint and narrator lead-in
            last_user_msg = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break
            continue_hint = get_continue_hint(mode, last_user_msg)
            if continue_hint:
                full_prompt += f"\n{continue_hint}\n"
            full_prompt += "Narrator:"
    else:
        # No characters - pure narrator mode
        full_prompt += "Narrator:"
    
    return full_prompt

# Character Card Generation Logic (Ported from Char_card_app)
async def call_llm_helper(system_prompt: str, user_prompt: str, max_tokens: int = 500):
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "prompt": f"### System: {system_prompt}\n### User: {user_prompt}\n### Assistant:",
                "max_length": max_tokens,
                "temperature": 0.7,
                "stop_sequence": ["###", "<|im_end|>", "\n\n\n", "User:", "Assistant:"]
            }
            print(f"DEBUG LLM CALL: {KOBOLD_API_URL if 'KOBOLD_API_URL' in globals() else CONFIG['kobold_url']}/api/v1/generate")
            res = await client.post(
                f"{CONFIG['kobold_url']}/api/v1/generate",
                json=payload,
                timeout=120.0
            )
            res.raise_for_status()
            data = res.json()
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0]["text"].strip()
            else:
                print(f"LLM Error: Unexpected response structure: {data}")
                raise Exception("Unexpected response from LLM")
        except Exception as e:
            print(f"LLM Call Exception: {str(e)}")
            raise Exception(f"API Error: {str(e)}")

@app.post("/api/card-gen/generate-field")
async def generate_card_field(req: CardGenRequest):
    char_name = req.char_name
    field_type = req.field_type
    user_input = req.context
    
    try:
        if field_type == 'personality':
            system = "You are an expert at analyzing characters from chat and writing roleplay personality traits."
            prompt = f"""Based on the provided chat context, identify {char_name}'s personality traits.
Convert them into a PList personality array format.
Use this exact format: [{char_name}'s Personality= "trait1", "trait2", "trait3", ...]

Chat Context:
{user_input}

Only output the personality array line, nothing else."""
            result = await call_llm_helper(system, prompt, 300)
            
        elif field_type == 'body':
            system = "You are an expert at analyzing characters from chat and writing physical descriptions."
            prompt = f"""Based on the provided chat context, identify {char_name}'s physical features.
Convert them into a PList body array format.
Use this exact format: [{char_name}'s body= "feature1", "feature2", "feature3", ...]

Chat Context:
{user_input}

Only output the body array line, nothing else."""
            result = await call_llm_helper(system, prompt, 300)
            
        elif field_type == 'dialogue_likes':
            system = "You are an expert at writing roleplay dialogue in markdown format with {{char}} and {{user}} placeholders."
            prompt = f"""Based on {char_name}'s behavior in the chat context, write a dialogue exchange where {{user}} asks "{char_name}, what are your likes and dislikes?", and {{char}} responds in character.

Chat Context:
{user_input}

Use this exact format:
{{user}}: "{char_name}, what are your likes and dislikes?"
{{char}}: *Action.* "Speech." *More action and description.*

Make the response 3-5 sentences, vivid and in-character. Only output the dialogue exchange."""
            result = await call_llm_helper(system, prompt, 500)
            
        elif field_type == 'dialogue_story':
            system = "You are an expert at writing roleplay dialogue in markdown format with {{char}} and {{user}} placeholders."
            prompt = f"""Based on {char_name}'s behavior in the chat context, write a dialogue exchange where {{user}} asks "{char_name}, tell me about your life story", and {{char}} responds with a brief life story in character.

Chat Context:
{user_input}

Use this exact format:
{{user}}: "{char_name}, tell me about your life story."
{{char}}: *Action.* "Speech." *More action and description.*

Make the response 3-5 sentences, vivid and in-character. Only output the dialogue exchange."""
            result = await call_llm_helper(system, prompt, 500)
            
        elif field_type == 'genre':
            system = "You are an expert at categorizing roleplay genres."
            prompt = f"""Based on the chat context involving {char_name}, suggest an appropriate genre (fantasy, sci-fi, modern, historical, etc.):

{user_input}

Only output the genre name, nothing else."""
            result = await call_llm_helper(system, prompt, 50)
            
        elif field_type == 'tags':
            system = "You are an expert at tagging roleplay characters."
            prompt = f"""Based on the chat context involving {char_name}, suggest 2-4 relevant tags (e.g., adventure, romance, comedy, mystery, magic, etc.):

{user_input}

Only output the tags separated by commas, nothing else."""
            result = await call_llm_helper(system, prompt, 100)
            
        elif field_type == 'scenario':
            system = "You are an expert at writing roleplay scenarios."
            prompt = f"""Based on the chat context, write a one-sentence scenario describing {char_name}'s initial situation:

{user_input}

Format: You [situation]. Make it engaging and specific. Only output the scenario sentence."""
            result = await call_llm_helper(system, prompt, 200)
            
        elif field_type == 'first_message':
            system = "You are an expert at writing engaging roleplay opening messages. Match the character's voice from the chat."
            prompt = f"""Write a first message for {char_name} that would start a new roleplay, based on the provided chat history.

Chat Context:
{user_input}

Use markdown format with *actions* and "speech". Make it immersive. Only output the first message."""
            result = await call_llm_helper(system, prompt, 600)
            
        else:
            return {"success": False, "error": "Unknown field type"}
        
        return {"success": True, "text": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Capsule Generation for Multi-Character Optimization
async def generate_capsule_for_character(char_name: str, description: str, depth_prompt: str = "") -> str:
    """Generate a capsule summary for use in multi-character scenarios."""
    system = "You are an expert at distilling roleplay character cards into minimal capsule summaries for efficient multi-character prompts."
    
    full_card_text = f"Name: {char_name}\n\nDescription/Dialogue:\n{description}"
    if depth_prompt:
        full_card_text += f"\n\nPersonality/Context:\n{depth_prompt}"
    
    prompt = f"""Convert this character card into a capsule summary for efficient multi-character prompts.
Use this exact format (one paragraph, no line breaks):
Name: [Name]. Role: [1 sentence role/situation]. Key traits: [3-5 comma-separated personality traits]. Speech style: [short/long, formal/casual, any verbal tics]. Example line: "[One characteristic quote from descriptions]"

Full Card:
{full_card_text}

Output only the capsule summary line, nothing else."""

    result = await call_llm_helper(system, prompt, 200)
    return result.strip()

# Mode Classification Request Model
class ModeClassifyRequest(BaseModel):
    user_message: str
    character_names: List[str]

@app.post("/api/classify-mode")
async def classify_mode(req: ModeClassifyRequest):
    """Auto-classify which mode should be used based on the user's message."""
    if len(req.character_names) < 2:
        # Single or no characters - use simple rules
        if len(req.character_names) == 1:
            return {"success": True, "mode": f"focus:{req.character_names[0]}"}
        return {"success": True, "mode": "narrator"}
    
    # Multiple characters - use LLM to classify
    char_list = ", ".join(req.character_names)
    system = "You are a classifier that determines who should respond in a roleplay scene."
    
    prompt = f"""Given this user message in a roleplay with multiple characters, determine who should respond.

Active characters: {char_list}

User message: "{req.user_message}"

Rules:
- If the user directly addresses a character by name (e.g., "Alice, what do you think?"), output: focus:CharacterName
- If the user asks a question that could be answered by any character, or describes an action affecting multiple characters, output: narrator
- If the context suggests an intimate or personal moment with a specific character, output: focus:CharacterName
- When in doubt, default to: narrator

Output ONLY one of these options, nothing else:
- narrator
- focus:{char_list.split(', ')[0]}
- focus:{char_list.split(', ')[1] if len(req.character_names) > 1 else char_list.split(', ')[0]}
{('- focus:' + char_list.split(', ')[2] if len(req.character_names) > 2 else '')}

Your answer:"""

    try:
        result = await call_llm_helper(system, prompt, 30)
        result = result.strip().lower()
        
        # Validate the result
        if result == "narrator":
            return {"success": True, "mode": "narrator"}
        elif result.startswith("focus:"):
            char_name = result.split(":", 1)[1].strip()
            # Find matching character (case insensitive)
            for name in req.character_names:
                if name.lower() == char_name.lower():
                    return {"success": True, "mode": f"focus:{name}"}
            # If no exact match, fall back to narrator
            return {"success": True, "mode": "narrator"}
        else:
            # Unrecognized output, default to narrator
            return {"success": True, "mode": "narrator"}
    except Exception as e:
        print(f"Mode classification failed: {e}")
        return {"success": True, "mode": "narrator"}

@app.post("/api/card-gen/generate-capsule")
async def generate_capsule(req: CapsuleGenRequest):
    """API endpoint to generate a capsule summary for a character."""
    try:
        capsule = await generate_capsule_for_character(req.char_name, req.description, req.depth_prompt)
        return {"success": True, "text": capsule}
    except Exception as e:
        return {"success": False, "error": str(e)}

# World Info Generation Logic (Ported from world-gen-app)
WORLD_PROMPTS = {
    "history": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory mentioned in the chat. Use this exact format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects)]\n\nRules:\n- One entry per line\n- Keep entries concise\n- Use parentheses for nested attributes\n- Output ONLY the PList entries, no explanations",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature/suggestive themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event, era, or piece of backstory. Include mature themes where appropriate. Use this format:\n[EventName: type(event/era/myth), time(when it happened), actors(who was involved), result(what happened), legacy(lasting effects), intimate_aspects(relationships, desires, power dynamics)]\n\nRules:\n- One entry per line\n- Include suggestive details naturally\n- Output ONLY the PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each historical event or backstory. Include explicit adult themes naturally. Use this format:\n[EventName: type(event/era/myth), time(when), actors(who), result(what happened), legacy(effects), sexual_culture(attitudes, practices, taboos)]\n\nRules:\n- One entry per line\n- Include explicit adult content where relevant\n- Output ONLY the PList entries"
    },
    "locations": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location, room, town, or area mentioned. Use this format:\n[LocationName(nickname if any): type(room/town/area), features(physical details), atmosphere(mood/feeling), purpose(what happens here), inhabitants(who is usually here)]\n\nRules:\n- One entry per line- Output ONLY the PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include suggestive atmosphere and uses. Use this format:\n[LocationName(nickname): type, features, atmosphere(mood), purpose(what happens), inhabitants(who), intimate_uses(private activities)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each location. Include explicit adult themes naturally. Format:\n[LocationName(nickname): type, features, atmosphere, purpose, inhabitants, sexual_activities(what happens), kinks_associated(themes)]\n\nRules:\n- One entry per line- Output ONLY PList entries"
    },
    "creatures": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each creature, monster, or character archetype. Use this format:\n[CreatureName: type(creature/archetype), appearance(visual traits), behavior(typical actions), culture(social norms, beliefs), habitat(where found), attitude_toward_user(how they treat {{user}})]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for creatures or archetypes. Include suggestive behavior and interactions. Format:\n[CreatureName: type, appearance, behavior, culture, habitat, attitude_toward_user, flirtation_style(how they seduce/interact)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[CreatureName: type, appearance, behavior, culture, attitude_toward_user, sexual_behavior(explicit details), kinks(preferences), consent_culture(boundaries)]\n\nRules:\n- One entry per line- Output ONLY PList entries"
    },
    "factions": {
        "sfw": "You are creating World Info entries in PList format for SillyTavern roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format World Info entries for each faction, group, or organization. Use this format:\n[FactionName: type(faction/guild/house/clique), members(who belongs), reputation(public image), goals(what they want), methods(how they operate), attitude_toward_user(how they treat {{user}}), rivals(opposing factions)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "spicy": "You are creating World Info entries in PList format for SillyTavern roleplays with mature themes.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries for factions. Include suggestive motivations and interactions. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, social_dynamics(power, romance, rivalries), intimate_culture(dating norms, boundaries)]\n\nRules:\n- One entry per line- Output ONLY PList entries",
        "veryspicy": "You are creating World Info entries in PList format for SillyTavern ERP/NSFW roleplays.\n\nWorld name: {worldName}\n\nChat Context:\n{input}\n\nCreate PList-format entries including explicit adult content. Format:\n[FactionName: type, members, reputation, goals, methods, attitude_toward_user, sexual_culture(practices, rituals), kinks_favored(group preferences), initiation(how to join)]\n\nRules:\n- One entry per line- Output ONLY PList entries"
    }
}

def parse_plist_line(line):
    line = line.strip()
    if not (line.startswith('[') and line.endswith(']')): return None
    inner = line[1:-1]
    if ':' not in inner: return None
    name_part, content_part = inner.split(':', 1)
    name_part = name_part.strip()
    alias = None
    if '(' in name_part and ')' in name_part:
        match = re.match(r'^([^(]+)\(([^)]+)\)$', name_part)
        if match:
            name = match.group(1).strip()
            alias = match.group(2).strip()
        else: name = name_part
    else: name = name_part
    keys = [name.lower()]
    if alias: keys.append(alias.lower())
    words = re.findall(r'\b([a-zA-Z]{4,})\b', content_part.lower())
    for word in words[:3]:
        if word not in ['type', 'has', 'used', 'for', 'the', 'and', 'with'] and word not in keys:
            keys.append(word)
    return {"name": name, "alias": alias, "content": line, "keys": keys}

@app.post("/api/world-gen/generate")
async def generate_world_entries(req: WorldGenRequest):
    try:
        template = WORLD_PROMPTS.get(req.section, {}).get(req.tone, "")
        if not template: return {"success": False, "error": "Invalid section or tone"}
        prompt = template.format(worldName=req.world_name, input=req.context)
        result = await call_llm_helper("You are a world-building expert.", prompt, 600)
        # Clean up output to only include bracketed entries
        lines = [line.strip() for line in result.split('\n') if line.strip().startswith('[')]
        return {"success": True, "text": '\n'.join(lines)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/world-gen/save")
async def save_generated_world_info(req: WorldSaveRequest):
    try:
        lines = req.plist_text.split('\n')
        entries = {}
        uid_counter = 0
        for line in lines:
            parsed = parse_plist_line(line)
            if parsed:
                entries[str(uid_counter)] = {
                    "uid": uid_counter,
                    "key": parsed['keys'],
                    "keysecondary": [],
                    "comment": parsed['alias'] or "",
                    "content": parsed['content'],
                    "constant": False,
                    "selective": True,
                    "selectiveLogic": 0,
                    "addMemo": True,
                    "order": 100,
                    "position": 4,
                    "disable": False,
                    "excludeRecursion": False,
                    "probability": 100,
                    "useProbability": True,
                    "displayIndex": uid_counter,
                    "depth": 5
                }
                uid_counter += 1
        
        if not entries: return {"success": False, "error": "No valid entries found"}
        
        file_path = os.path.join(DATA_DIR, "worldinfo", f"{req.world_name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)
            
        return {"success": True, "name": req.world_name}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "app/index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Index.html not found"

@app.post("/api/chat")
async def chat(request: PromptRequest):
    # Auto-generate capsules for group chats (2+ characters)
    updated_characters = []
    chars_needing_capsules = []
    
    if len(request.characters) >= 2:
        for char_obj in request.characters:
            data = char_obj.get("data", {})
            extensions = data.get("extensions", {})
            multi_char_summary = extensions.get("multi_char_summary", "")
            
            if not multi_char_summary:
                # Character needs a capsule generated
                chars_needing_capsules.append(char_obj)
            updated_characters.append(char_obj)
        
        # Generate capsules for characters that need them
        for char_obj in chars_needing_capsules:
            data = char_obj.get("data", {})
            name = data.get("name", "Unknown")
            description = data.get("description", "")
            depth_prompt = data.get("extensions", {}).get("depth_prompt", {}).get("prompt", "")
            
            try:
                print(f"AUTO-GENERATING capsule for {name} (group chat detected)")
                capsule = await generate_capsule_for_character(name, description, depth_prompt)
                
                # Update the character object in memory
                if "extensions" not in data:
                    data["extensions"] = {}
                data["extensions"]["multi_char_summary"] = capsule
                
                # Save the updated capsule to the character file
                filename = char_obj.get("_filename")
                if filename:
                    file_path = os.path.join(DATA_DIR, "characters", filename)
                    save_data = char_obj.copy()
                    if "_filename" in save_data:
                        del save_data["_filename"]
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                    print(f"SAVED capsule for {name}: {capsule[:80]}...")
            except Exception as e:
                print(f"Failed to auto-generate capsule for {name}: {e}")
        
        # Update the request with the modified characters
        request.characters = updated_characters
    
    # Check for summarization need
    max_ctx = request.settings.get("max_context", 4096)
    threshold = request.settings.get("summarize_threshold", 0.85)
    
    current_request = request
    new_summary = request.summary or ""
    
    # Initial token check
    prompt = construct_prompt(current_request)
    tokens = await get_token_count(prompt)
    
    # Summarization loop
    if tokens > max_ctx * threshold and len(current_request.messages) > 10:
        print(f"TRUNCATION TRIGGERED: {tokens} tokens exceeds {max_ctx * threshold}")

        # Extract Canon Law entries to echo into summary if needed
        canon_echo = ""
        if request.world_info:
            canon_entries = [e.get("content", "") for e in request.world_info.get("entries", {}).values() if e.get("is_canon_law")]
            if canon_entries:
                canon_echo = "\n### Note: Active Canon Laws to maintain:\n" + "\n".join(canon_entries)

        # Take oldest 10 messages to summarize
        to_summarize = current_request.messages[:10]
        remaining_messages = current_request.messages[10:]
        
        summary_text = "\n".join([f"{m.speaker or m.role}: {m.content}" for m in to_summarize])
        summarization_prompt = f"### System: Summarize the following conversation snippet into a very concise narrative for long-term memory. Focus on key plot points and character states.{canon_echo}\n\n### Conversation to summarize:\n{summary_text}\n\n### Concise Summary:"
        
        async with httpx.AsyncClient() as client:
            try:
                sum_res = await client.post(
                    f"{CONFIG['kobold_url']}/api/v1/generate",
                    json={
                        "prompt": summarization_prompt,
                        "max_length": 150,
                        "temperature": 0.5,
                        "stop_sequence": ["###", "\nUser:", "\nAssistant:"]
                    },
                    timeout=60.0
                )
                sum_data = sum_res.json()
                added_summary = sum_data["results"][0]["text"].strip()
                new_summary = (new_summary + "\n" + added_summary).strip()
                
                # Rebuild request with new summary and fewer messages
                current_request.messages = remaining_messages
                current_request.summary = new_summary
                prompt = construct_prompt(current_request)
                tokens = await get_token_count(prompt)
            except Exception as e:
                print(f"Summarization failed: {e}")

    print(f"Generated Prompt ({tokens} tokens):\n{prompt}")
    
    # Extract settings for generation
    temp = request.settings.get("temperature", 0.7)
    max_len = request.settings.get("max_length", 250)
    mode = request.mode or "narrator"

    # Calculate stop sequences (mode-aware)
    stops = ["User:", "\nUser", "###"]
    
    is_group_chat = len(request.characters) >= 2
    
    if is_group_chat:
        if mode.startswith("focus:"):
            # Focused mode: stop on other character names (not the focused one)
            focus_char = mode.split(":", 1)[1]
            for char in request.characters:
                name = char.get("data", {}).get("name")
                if name and name != focus_char:
                    stops.append(f"{name}:")
            # Also stop on Narrator since we're in first-person mode
            stops.append("Narrator:")
        else:
            # Narrator mode: don't stop on character names - let all speak
            # Only stop on User to prevent the narrator from writing user's lines
            pass
    else:
        # Single character or no characters
        for char in request.characters:
            name = char.get("data", {}).get("name")
            if name: stops.append(f"{name}:")
        if not request.characters: 
            stops.append("Narrator:")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{CONFIG['kobold_url']}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_length": max_len,
                    "temperature": temp,
                    "stop_sequence": stops
                },
                timeout=60.0
            )
            data = response.json()
            # Wrap response to include potential state updates (summary/truncated messages)
            data["_updated_state"] = {
                "messages": [m.dict() for m in current_request.messages],
                "summary": current_request.summary
            }
            return data
        except Exception as e:
            return {"error": str(e)}

@app.post("/api/extra/tokencount")
async def proxy_tokencount(request: dict):
    count = await get_token_count(request.get("prompt", ""))
    return {"count": count}

@app.post("/api/generate-image")
async def generate_image(params: SDParams):
    processed_prompt = params.prompt
    
    # Identify bracketed character names [Name]
    bracketed_names = re.findall(r"\[(.*?)\]", params.prompt)
    if bracketed_names:
        char_dir = os.path.join(DATA_DIR, "characters")
        # Load all characters to find matches
        all_chars = []
        for f in os.listdir(char_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(char_dir, f), "r", encoding="utf-8") as cf:
                        all_chars.append(json.load(cf))
                except:
                    continue
        
        for name in bracketed_names:
            # Case-insensitive match for name
            matched_char = next((c for c in all_chars if c.get("data", {}).get("name", "").lower() == name.lower()), None)
            if matched_char:
                danbooru_tag = matched_char.get("data", {}).get("extensions", {}).get("danbooru_tag", "")
                if danbooru_tag:
                    # Replace [Name] with the tag
                    processed_prompt = processed_prompt.replace(f"[{name}]", danbooru_tag)

    async with httpx.AsyncClient() as client:
        try:
            payload = params.dict()
            payload["prompt"] = processed_prompt
            print(f"SD Prompt Construction: {processed_prompt}")
            
            response = await client.post(
                f"{CONFIG['sd_url']}/sdapi/v1/txt2img",
                json=payload,
                timeout=120.0
            )
            data = response.json()
            image_base64 = data["images"][0]
            
            filename = f"sd_{int(time.time())}.png"
            file_path = os.path.join(IMAGE_DIR, filename)
            
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
                
            return {
                "url": f"/images/{filename}",
                "filename": filename
            }
        except Exception as e:
            return {"error": str(e)}

@app.get("/api/characters")
async def list_characters():
    chars = []
    char_dir = os.path.join(DATA_DIR, "characters")
    for f in os.listdir(char_dir):
        if f.endswith(".json"):
            file_path = os.path.join(char_dir, f)
            try:
                with open(file_path, "r", encoding="utf-8") as char_file:
                    data = json.load(char_file)
                    # Inject current filename into response so frontend can track it
                    data["_filename"] = f
                    chars.append(data)
            except Exception as e:
                print(f"FAILED to load character {f}: {e}")
    return chars

@app.post("/api/characters")
async def save_character(char: dict):
    # Use existing filename if provided, else name.json
    filename = char.get("_filename")
    if not filename:
        char_data = char.get("data", char)
        name = char_data.get("name", "NewCharacter")
        filename = f"{name}.json"
    
    file_path = os.path.join(DATA_DIR, "characters", filename)
    with open(file_path, "w", encoding="utf-8") as f:
        # Strip internal tracking field before saving
        save_data = char.copy()
        if "_filename" in save_data: del save_data["_filename"]
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    return {"success": True, "filename": filename}

@app.delete("/api/characters/{filename}")
async def delete_character(filename: str):
    file_path = os.path.join(DATA_DIR, "characters", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"success": True}
    return {"error": "File not found"}

@app.get("/api/world-info")
async def list_world_info():
    wi_list = []
    wi_dir = os.path.join(DATA_DIR, "worldinfo")
    for f in os.listdir(wi_dir):
        if f.endswith(".json"):
            file_path = os.path.join(wi_dir, f)
            try:
                with open(file_path, "r", encoding="utf-8") as wi_file:
                    content = json.load(wi_file)
                    wi_list.append({"name": f.replace(".json", ""), **content})
            except Exception as e:
                print(f"FAILED to load world info {f}: {e}")
    return wi_list

@app.post("/api/world-info")
async def save_world_info(request: dict):
    name = request.get("name")
    if not name: return {"error": "No name found"}
    file_path = os.path.join(DATA_DIR, "worldinfo", f"{name}.json")
    data = request if "entries" in request else {"entries": request.get("data", {})}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"success": True}

# Chat session management
@app.get("/api/chats")
async def list_chats():
    chats = []
    chat_dir = os.path.join(DATA_DIR, "chats")
    for f in os.listdir(chat_dir):
        if f.endswith(".json"):
            chats.append(f.replace(".json", ""))
    return chats

@app.get("/api/chats/{name}")
async def load_chat(name: str):
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Chat not found"}

@app.post("/api/chats")
async def save_chat(request: dict):
    name = request.get("name")
    if not name:
        name = f"chat_{int(time.time())}"
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(request.get("data"), f, indent=2, ensure_ascii=False)
    return {"success": True, "name": name}

@app.delete("/api/chats/{name}")
async def delete_chat(name: str):
    file_path = os.path.join(DATA_DIR, "chats", f"{name}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"success": True}
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
