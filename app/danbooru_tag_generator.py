"""
Danbooru Tag Generator

Generates Danbooru tags from character descriptions using semantic search
and progressive tag reduction for optimal character matching.
"""

from typing import Dict, List, Any, Optional, Tuple
import random

from app.database import (
    db_search_danbooru_characters_semantically,
    db_get_danbooru_characters,
    db_assign_visual_canon_to_character,
    db_assign_visual_canon_to_npc,
    db_get_chat,
    db_get_npc_by_id,
    db_create_npc_with_entity_id,
    db_save_chat
)


# Trait mapping dictionaries
HAIR_COLORS = [
    'blonde', 'brown', 'black', 'red', 'white', 'silver', 'pink', 'blue', 
    'purple', 'green', 'orange', 'gray', 'aqua', 'blonde', 'brunette'
]

EYE_COLORS = [
    'blue', 'green', 'brown', 'red', 'purple', 'yellow', 'black', 'gray', 
    'amber', 'turquoise', 'hazel', 'pink'
]

HAIR_STYLES = {
    'very_long': 'very_long_hair',
    'extremely_long': 'very_long_hair',
    'waist_length': 'very_long_hair',
    'long': 'long_hair',
    'medium': 'medium_hair',
    'shoulder_length': 'medium_hair',
    'short': 'short_hair',
    'bob': 'bob_cut',
    'bun': 'hair_bun',
    'double_bun': 'double_bun',
    'twintails': 'twintails',
    'twin_tails': 'twintails',
    'pigtails': 'twintails',
    'ponytail': 'ponytail',
    'braid': 'braid',
    'braided': 'braid',
    'twin_braids': 'twin_braids',
    'ahoge': 'ahoge',
    'hime_cut': 'hime_cut',
    'drill_hair': 'drill_hair'
}

BREAST_SIZE_MAP = {
    'flat': 'flat_chest',
    'no_breasts': 'flat_chest',
    'flat_chest': 'flat_chest',
    'child': 'flat_chest',
    'loli': 'flat_chest',
    'young': 'flat_chest',
    'small': 'small_breasts',
    'small_breasts': 'small_breasts',
    'petite': 'small_breasts',
    'slender': 'small_breasts',
    'medium': 'medium_breasts',
    'average': 'medium_breasts',
    'medium_breasts': 'medium_breasts',
    'large': 'large_breasts',
    'big': 'large_breasts',
    'curvy': 'large_breasts',
    'busty': 'large_breasts',
    'voluptuous': 'large_breasts',
    'large_breasts': 'large_breasts',
    'huge': 'huge_breasts',
    'massive': 'huge_breasts',
    'very_curvy': 'huge_breasts',
    'huge_breasts': 'huge_breasts'
}

SKIN_TONE_MAP = {
    'tan': 'dark_skin',
    'tanned': 'dark_skin',
    'dark': 'dark_skin',
    'dark_skin': 'dark_skin',
    'brown': 'dark_skin',
    'brown_skin': 'dark_skin',
    'chocolate': 'dark_skin',
    'ebony': 'dark_skin'
}

CREATURE_TYPES = {
    'elf': {'tags': ['pointy_ears'], 'search': 'elf'},
    'elven': {'tags': ['pointy_ears'], 'search': 'elf'},
    'fairy': {'tags': ['wings'], 'search': 'fairy'},
    'dwarf': {'tags': ['short'], 'search': 'dwarf'},
    'angel': {'tags': ['wings', 'halo'], 'search': 'angel'},
    'bunny': {'tags': ['animal_ears', 'bunny_ears'], 'search': 'bunny'},
    'rabbit': {'tags': ['animal_ears', 'bunny_ears'], 'search': 'bunny'},
    'cat': {'tags': ['animal_ears', 'cat_ears'], 'search': 'cat'},
    'neko': {'tags': ['animal_ears', 'cat_ears'], 'search': 'cat'},
    'dog': {'tags': ['animal_ears', 'dog_ears'], 'search': 'dog'},
    'inu': {'tags': ['animal_ears', 'dog_ears'], 'search': 'dog'},
    'demon': {'tags': ['horns', 'tail'], 'search': 'demon'},
    'devil': {'tags': ['horns', 'tail'], 'search': 'demon'},
    'succubus': {'tags': ['horns', 'tail', 'wings'], 'search': 'succubus'},
    'dragon': {'tags': ['horns', 'tail'], 'search': 'dragon'},
    'kitsune': {'tags': ['animal_ears', 'fox_ears', 'tail'], 'search': 'kitsune'},
    'fox': {'tags': ['animal_ears', 'fox_ears', 'tail'], 'search': 'fox'},
    'wolf': {'tags': ['animal_ears', 'wolf_ears', 'tail'], 'search': 'wolf'},
    'oni': {'tags': ['horns'], 'search': 'oni'}
}

LOLI_KEYWORDS = ['loli', 'child', 'little_girl', 'young_girl', 'young', 'little', 'small_girl']


def extract_hair_color(description: str) -> Optional[str]:
    """Extract hair color from description."""
    desc_lower = description.lower()
    for color in HAIR_COLORS:
        if color in desc_lower and ('hair' in desc_lower or 'haired' in desc_lower):
            return color
    return None


def extract_eye_color(description: str) -> Optional[str]:
    """Extract eye color from description."""
    desc_lower = description.lower()
    for color in EYE_COLORS:
        # Check for "blue eyes", "blue_eyes", "blue-eyed"
        patterns = [
            f"{color} eyes",
            f"{color}_eyes",
            f"{color}-eyed",
            f"{color} eye"
        ]
        for pattern in patterns:
            if pattern in desc_lower:
                return color
    return None


def extract_hair_style(description: str) -> Optional[str]:
    """Extract hair style from description."""
    desc_lower = description.lower()
    for keyword, tag in HAIR_STYLES.items():
        if keyword in desc_lower:
            return tag
    return None


def extract_breast_size(description: str) -> Optional[str]:
    """Extract breast size from description using danbooru vocabulary."""
    desc_lower = description.lower()
    for keyword, tag in BREAST_SIZE_MAP.items():
        if keyword in desc_lower:
            return tag
    return None


def extract_skin_tone(description: str) -> Optional[str]:
    """Extract skin tone from description."""
    desc_lower = description.lower()
    for keyword, tag in SKIN_TONE_MAP.items():
        if keyword in desc_lower:
            return tag
    return None


def extract_creature_features(description: str) -> Tuple[Optional[str], List[str]]:
    """
    Extract creature type and associated physical features.
    Returns: (creature_search_tag, [physical_feature_tags])
    """
    desc_lower = description.lower()
    
    for creature, data in CREATURE_TYPES.items():
        if creature in desc_lower:
            return data['search'], data['tags']
    
    return None, []


def is_loli(description: str) -> bool:
    """Check if description indicates loli character."""
    desc_lower = description.lower()
    for keyword in LOLI_KEYWORDS:
        if keyword in desc_lower:
            return True
    return False


def extract_all_traits(description: str) -> Dict[str, Any]:
    """
    Extract all physical traits from description.
    Returns dictionary of traits for danbooru tag generation.
    """
    traits = {
        'hair_color': extract_hair_color(description),
        'eye_color': extract_eye_color(description),
        'hair_style': extract_hair_style(description),
        'breast_size': extract_breast_size(description),
        'skin_tone': extract_skin_tone(description),
        'loli': is_loli(description)
    }
    
    creature_type, creature_features = extract_creature_features(description)
    if creature_type:
        traits['creature_type'] = creature_type
        traits['creature_features'] = creature_features
    
    return traits


def build_search_tags(traits: Dict[str, Any]) -> List[str]:
    """
    Build list of tags for semantic search from extracted traits.
    Returns tags in priority order (most important first).
    """
    tags = []
    
    # Priority 1: Hair color (most distinctive visual feature)
    if traits.get('hair_color'):
        tags.append(f"{traits['hair_color']}_hair")
    
    # Priority 2: Eye color
    if traits.get('eye_color'):
        tags.append(f"{traits['eye_color']}_eyes")
    
    # Priority 3: Hair style
    if traits.get('hair_style'):
        tags.append(traits['hair_style'])
    
    # Priority 4: Creature type (if present, very distinctive)
    if traits.get('creature_type'):
        tags.append(traits['creature_type'])
        # Also add creature-specific physical features
        if traits.get('creature_features'):
            tags.extend(traits['creature_features'])
    
    # Priority 5: Breast size (body type indicator)
    if traits.get('breast_size'):
        tags.append(traits['breast_size'])
    
    # Priority 6: Loli tag (age/body type)
    if traits.get('loli'):
        tags.append('loli')
    
    # Priority 7: Skin tone (only if non-default)
    if traits.get('skin_tone'):
        tags.append(traits['skin_tone'])
    
    return tags


def search_with_progressive_reduction(
    tags: List[str],
    gender: str,
    k: int = 10,
    threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Search for Danbooru characters with progressive tag reduction.
    
    Algorithm:
    1. Try search with all tags
    2. If no results, remove least important tag
    3. Repeat until results found or only 1 tag remains
    4. If still no results, return random gender matches
    """
    current_tags = tags.copy()
    
    while len(current_tags) > 0:
        # Build query from current tags
        query_text = " ".join(current_tags)
        
        print(f"[DANBOORU_SEARCH] Searching with {len(current_tags)} tags: {query_text}")
        
        # Search with gender as hard filter
        results = db_search_danbooru_characters_semantically(
            query_text=query_text,
            gender=gender,
            k=k,
            threshold=threshold
        )
        
        if results:
            print(f"[DANBOORU_SEARCH] Found {len(results)} matches")
            return results
        
        # No results, remove last (least important) tag
        removed_tag = current_tags.pop()
        print(f"[DANBOORU_SEARCH] No results, removing tag: {removed_tag}")
    
    # No results even with single tags, fall back to random gender match
    print(f"[DANBOORU_SEARCH] No semantic matches, using random gender match")
    fallback = db_get_danbooru_characters(gender=gender, limit=50)
    
    if fallback:
        return fallback
    
    return []


def build_danbooru_tag_string(
    traits: Dict[str, Any],
    gender: str,
    matched_character: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build the final Danbooru tag string for populating the field.
    
    Format: "character_tag, 1girl/1boy, hair_color, eye_color, other_traits..."
    """
    tags = []
    
    # Add character name/tag from matched character (column 0 from database)
    if matched_character and matched_character.get('all_tags'):
        all_tags = matched_character['all_tags'].split(',')
        if all_tags:
            character_tag = all_tags[0].strip()
            if character_tag:
                tags.append(character_tag)
    
    # Add count tag based on gender
    if gender == 'female':
        tags.append('1girl')
    elif gender == 'male':
        tags.append('1boy')
    elif gender == 'other':
        tags.append('solo')
    else:
        # Unknown gender, try to infer from matched character or default to 1girl
        if matched_character and matched_character.get('gender'):
            gender_map = {'female': '1girl', 'male': '1boy'}
            tags.append(gender_map.get(matched_character['gender'], '1girl'))
        else:
            tags.append('1girl')
    
    # Add physical traits in order of importance
    if traits.get('hair_color'):
        tags.append(f"{traits['hair_color']}_hair")
    
    if traits.get('eye_color'):
        tags.append(f"{traits['eye_color']}_eyes")
    
    if traits.get('hair_style'):
        tags.append(traits['hair_style'])
    
    # Add creature features before body type
    if traits.get('creature_features'):
        for feature in traits['creature_features']:
            if feature not in tags:  # Avoid duplicates
                tags.append(feature)
    
    if traits.get('breast_size'):
        tags.append(traits['breast_size'])
    
    if traits.get('skin_tone'):
        tags.append(traits['skin_tone'])
    
    if traits.get('loli'):
        tags.append('loli')
    
    # Add some core tags from matched character if available (but not too many)
    if matched_character and matched_character.get('core_tags'):
        core_tags = matched_character['core_tags'].split(',')[:3]  # Max 3 core tags
        for tag in core_tags:
            tag_clean = tag.strip()
            if tag_clean and tag_clean not in tags:
                tags.append(tag_clean)
    
    return ", ".join(tags)


def generate_tags_from_description(
    description: str,
    gender: str
) -> Dict[str, Any]:
    """
    Main function to generate Danbooru tags from character description.
    
    Args:
        description: Character physical description
        gender: 'female', 'male', 'other', or 'unknown'
    
    Returns:
        {
            'success': bool,
            'suggested_tags': str,  # Danbooru tag string to populate field
            'visual_canon_id': int or None,
            'visual_canon_name': str or None,
            'extracted_traits': dict,  # For debugging
            'message': str
        }
    """
    if not description or not description.strip():
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': {},
            'message': 'No description provided'
        }
    
    if not gender or gender == 'unknown':
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': {},
            'message': 'Please select a gender (Female/Male/Other) first'
        }
    
    # Step 1: Extract all physical traits
    traits = extract_all_traits(description)
    print(f"[DANBOORU_GEN] Extracted traits: {traits}")
    
    # Step 2: Build search tags
    search_tags = build_search_tags(traits)
    print(f"[DANBOORU_GEN] Search tags: {search_tags}")
    
    if not search_tags:
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': traits,
            'message': 'Could not extract physical traits from description. Try adding details about hair color, eye color, etc.'
        }
    
    # Step 3: Progressive search
    results = search_with_progressive_reduction(search_tags, gender)
    
    if not results:
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': traits,
            'message': 'No matching Danbooru character found. Try adding more physical description details.'
        }
    
    # Step 4: Random selection from top results (for variety on reroll)
    selected = random.choice(results[:min(10, len(results))])
    
    # Step 5: Build final tag string
    suggested_tags = build_danbooru_tag_string(traits, gender, selected)
    
    return {
        'success': True,
        'suggested_tags': suggested_tags,
        'visual_canon_id': selected.get('id'),
        'visual_canon_name': selected.get('name'),
        'extracted_traits': traits,
        'message': f'Generated tags from Danbooru character: {selected.get("name", "Unknown")}'
    }


def generate_and_assign_to_character(
    filename: str,
    description: str,
    gender: str
) -> Dict[str, Any]:
    """
    Generate tags and assign visual canon to a global character.
    
    Returns same format as generate_tags_from_description but also
    stores the binding in the database.
    """
    result = generate_tags_from_description(description, gender)
    
    if result['success'] and result['visual_canon_id']:
        # Store the visual canon binding
        success = db_assign_visual_canon_to_character(
            filename,
            result['visual_canon_id'],
            result['suggested_tags']
        )
        
        if success:
            print(f"[DANBOORU_GEN] Assigned visual canon to character {filename}")
        else:
            print(f"[DANBOORU_GEN] Warning: Could not store visual canon binding")
    
    return result


def generate_and_assign_to_npc(
    chat_id: str,
    npc_id: str,
    description: str,
    gender: str
) -> Dict[str, Any]:
    """
    Generate tags and assign visual canon to an NPC.

    Strategy:
    1. Generate tags from description
    2. Check if NPC exists in database
    3. If not in DB, create minimal NPC row from metadata
    4. Assign visual canon to database (Option A)
    5. Store visual canon in metadata as fallback (Option B)
    """
    result = generate_tags_from_description(description, gender)
    
    if result['success'] and result['visual_canon_id']:
        # Check if NPC exists in database
        npc_in_db = db_get_npc_by_id(npc_id, chat_id)
        
        if not npc_in_db:
            # NPC only in metadata - sync to database first
            print(f"[DANBOORU_GEN] NPC {npc_id} not in database, creating from metadata")
            
            # Load chat to get metadata
            chat = db_get_chat(chat_id)
            if not chat:
                print(f"[DANBOORU_GEN] Error: Chat {chat_id} not found")
                return result
            
            # Get NPC data from metadata
            metadata_npcs = chat.get('metadata', {}).get('localnpcs', {})
            npc_metadata = metadata_npcs.get(npc_id)
            
            if not npc_metadata:
                print(f"[DANBOORU_GEN] Error: NPC {npc_id} not found in metadata")
                return result
            
            # Create NPC in database with same entity_id
            npc_data = npc_metadata.get('data', {})
            success, error = db_create_npc_with_entity_id(chat_id, npc_id, npc_data)
            
            if success:
                print(f"[DANBOORU_GEN] Created NPC {npc_id} in database")
            else:
                print(f"[DANBOORU_GEN] Warning: Failed to create NPC in database: {error}")
                # Continue to metadata fallback anyway
        
        # Option A: Store the visual canon binding in database
        db_success = db_assign_visual_canon_to_npc(
            chat_id,
            npc_id,
            result['visual_canon_id'],
            result['suggested_tags']
        )
        
        if db_success:
            print(f"[DANBOORU_GEN] Assigned visual canon to NPC {npc_id} in database")
        else:
            print(f"[DANBOORU_GEN] Warning: Could not store visual canon binding in database")
        
        # Option B: Store visual canon in metadata as fallback
        chat = db_get_chat(chat_id)
        if chat:
            metadata = chat.get('metadata', {})
            if 'localnpcs' not in metadata:
                metadata['localnpcs'] = {}
            
            if npc_id in metadata['localnpcs']:
                # Update NPC in metadata with visual canon
                metadata['localnpcs'][npc_id]['visual_canon_id'] = result['visual_canon_id']
                metadata['localnpcs'][npc_id]['visual_canon_tags'] = result['suggested_tags']
                
                # Save back to chat
                chat['metadata'] = metadata
                db_save_chat(chat_id, chat)
                print(f"[DANBOORU_GEN] Stored visual canon in metadata for NPC {npc_id} (fallback)")
    
    return result
