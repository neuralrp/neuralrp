"""
Danbooru Tag Generator (Semantic-Based)

Generates Danbooru tags from character descriptions using semantic matching.
Two-stage approach:
1. Stage 1: Match physical traits to danbooru_tags (spaced format)
2. Stage 2: Search danbooru_characters for closest match

Replaces keyword-based extraction with semantic search for better accuracy.
"""

from typing import Dict, List, Any, Optional, Tuple
import random
import re

from app.database import (
    db_search_danbooru_tags_semantically,
    db_search_danbooru_characters_semantically,
    db_get_danbooru_characters,
    db_assign_visual_canon_to_character,
    db_assign_visual_canon_to_npc,
    db_get_chat,
    db_get_npc_by_id,
    db_create_npc_with_entity_id,
    db_save_chat
)


# ============================================================================
# CONSTANTS (Hair patterns, body type mappings, tag priorities)
# ============================================================================

# Hair color patterns for splitting compound terms
HAIR_COLOR_PATTERNS = [
    'blonde', 'brown', 'black', 'red', 'white', 'silver', 'pink', 'blue',
    'green', 'purple', 'orange', 'gray', 'multicolored', 'gradient',
    'platinum', 'strawberry', 'auburn', 'chestnut', 'copper', 'golden',
    'honey', 'lavender', 'teal', 'aqua', 'cyan'
]

# Hair style/length patterns for splitting compound terms
HAIR_STYLE_PATTERNS = [
    'long', 'short', 'medium', 'very long', 'ponytail', 'twintails', 'braid',
    'braided', 'bun', 'bob', 'pixie', 'layered', 'asymmetrical', 'wavy',
    'straight', 'curly', 'spiky', 'messy', 'neat', 'wet', 'windblown',
    'hime cut', 'afro', 'mohawk', 'bald'
]

# Body type to breast size mappings (synonyms for semantic matching)
BODY_TYPE_MAPPINGS = {
    'small_breasts': ['petite', 'slender', 'slim', 'skinny', 'thin', 'lean',
                     'waif', 'willowy', 'fragile', 'delicate', 'slight', 'lanky'],

    'medium_breasts': ['athletic', 'toned', 'fit', 'average', 'normal',
                     'moderate', 'standard', 'regular'],

    'large_breasts': ['curvy', 'voluptuous', 'busty', 'full-figured', 'hourglass',
                    'shapely', 'womanly', 'plump', 'chubby', 'thick'],

    'huge_breasts': ['overly large', 'massive', 'enormous', 'gigantic', 'excessively large',
                    'oversized', 'colossal', 'immense'],

    'flat_chest': ['teen', 'loli', 'young', 'child', 'childlike', 'youthful',
                  'prepubescent', 'underage', 'adolescent', 'flat']
}

# Traits to auto-remove from database search (aesthetic/abstract descriptors)
TRAITS_TO_REMOVE = {
    'pretty', 'gorgeous', 'beautiful', 'cute', 'handsome', 'attractive',
    'powerful', 'strong', 'mighty', 'magical', 'mysterious', 'charismatic',
    'intelligent', 'wise', 'brave', 'noble', 'fierce', 'terrifying'
}

# Tag category importance for progressive search (higher = keep longer)
TAG_CATEGORY_IMPORTANCE = {
    'hair_color': 100,      # Hair color (highest priority)
    'hair_style': 90,        # Hair style/length
    'body_type': 80,        # Body type (breast size)
    'eye_color': 70,        # Eye color
    'accessories': 60,       # Accessories (pointy_ears, choker, etc)
    'clothes': 50,          # Clothing
}

# Hair color keywords (spaced format)
HAIR_COLOR_KEYWORDS = ['blonde hair', 'brown hair', 'black hair', 'red hair', 'white hair',
                     'silver hair', 'pink hair', 'blue hair', 'green hair', 'purple hair',
                     'orange hair', 'gray hair', 'multicolored hair', 'platinum hair']

# Hair style keywords (spaced format)
HAIR_STYLE_KEYWORDS = ['long hair', 'short hair', 'medium hair', 'very long hair',
                     'ponytail', 'twintails', 'braid', 'braided', 'bun', 'bob',
                     'pixie cut', 'wavy hair', 'straight hair', 'curly hair', 'spiky hair']

# Eye color keywords (spaced format)
EYE_COLOR_KEYWORDS = ['blue eyes', 'green eyes', 'brown eyes', 'red eyes', 'purple eyes',
                      'yellow eyes', 'orange eyes', 'pink eyes', 'golden eyes', 'amber eyes',
                      'black eyes', 'white eyes', 'silver eyes', 'gray eyes']

# Accessory keywords (spaced format)
ACCESSORY_KEYWORDS = ['pointy ears', 'animal ears', 'choker', 'hair ornament', 'hairband',
                     'headband', 'wings', 'tail', 'horns', 'halo', 'mask', 'glasses',
                     'eyepatch', 'ribbon', 'bowtie']

# Clothing keywords (spaced format)
CLOTHING_KEYWORDS = ['armor', 'dress', 'gown', 'uniform', 'kimono', 'robe', 'cloak',
                     'jacket', 'coat', 'shirt', 'blouse', 'pants', 'skirt', 'swimsuit',
                     'lingerie', 'underwear', 'shoes', 'boots', 'hat', 'crown']


def parse_physical_body_plist(description: str) -> List[str]:
    """
    Extract physical traits from PList body description.

    Args:
        description: Character description (may be PList formatted or plain text)

    Returns:
        List of physical traits extracted from body field

    Examples:
        Input: "[Lila's body= \"Skinny\", \"perky breasts\", ...]"
        Output: ["Skinny", "perky breasts", ...]

        Input: "[Sarah the Sorcerer's body= \"pretty\", \"powerful\", ...]"
        Output: ["pretty", "powerful", ...]

        Input: "She has long blonde hair and pointed ears"
        Output: ["She has long blonde hair and pointed ears"]
    """
    if not description:
        return []

    # Generic PList format pattern: matches any character name
    pattern = r'\[.+?\'s body\s*=\s*(.*?)\]'
    match = re.search(pattern, description, re.DOTALL | re.IGNORECASE)

    if match:
        plist_content = match.group(1).strip()

        # Extract quoted traits from PList
        traits = re.findall(r'"([^"]+)"', plist_content)

        if traits:
            print(f"[DANBOORU_GEN] Parsed {len(traits)} traits from PList: {traits[:5]}...")
            return traits

    # Fallback: treat entire description as a single trait
    print(f"[DANBOORU_GEN] No PList format detected, using full description as trait")
    return [description.strip()]


def split_compound_hair_terms(traits: List[str]) -> List[str]:
    """
    Split compound hair terms into individual tags.

    Examples:
        "long blonde hair" → ["long hair", "blonde hair"]
        "short brown hair" → ["short hair", "brown hair"]
        "wavy black hair" → ["wavy hair", "black hair"]

    Args:
        traits: List of raw traits from PList

    Returns:
        List of traits with compound hair terms expanded
    """
    expanded_traits = []

    for trait in traits:
        trait_lower = trait.lower()

        # Check if trait contains 'hair' keyword
        if 'hair' in trait_lower:
            # Find hair color in trait
            found_color = None
            for color in HAIR_COLOR_PATTERNS:
                if color in trait_lower:
                    found_color = color
                    break

            # Find hair style in trait
            found_style = None
            for style in HAIR_STYLE_PATTERNS:
                if style in trait_lower:
                    found_style = style
                    break

            # If we found both color and style, split into separate traits
            if found_color and found_style:
                expanded_traits.append(f"{found_style} hair")
                expanded_traits.append(f"{found_color} hair")
                print(f"[DANBOORU_GEN] Split compound hair term: '{trait}' -> ['{found_style} hair', '{found_color} hair']")
            else:
                # No compound detected, keep original
                expanded_traits.append(trait)
        else:
            expanded_traits.append(trait)

    return expanded_traits


def map_body_type_to_breast_size(trait: str) -> Optional[str]:
    """
    Map body type descriptors to breast size Danbooru tags using exact keyword + semantic search.

    Args:
        trait: Body type trait from description (e.g., "petite", "athletic")

    Returns:
        Danbooru breast size tag (e.g., "small breasts") or None if no match
    """
    trait_lower = trait.lower()

    # Check against known mappings first (exact keyword match)
    for breast_tag, body_synonyms in BODY_TYPE_MAPPINGS.items():
        if trait_lower in body_synonyms:
            spaced_tag = breast_tag.replace('_', ' ')
            print(f"[DANBOORU_GEN] Body type mapping: '{trait}' -> '{spaced_tag}' (exact)")
            return spaced_tag

    # Use semantic search as fallback for close matches
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return None

    # Build corpus: each breast tag + all its synonyms
    corpus = []
    corpus_labels = []  # Maps corpus index to breast tag

    for breast_tag, synonyms in BODY_TYPE_MAPPINGS.items():
        corpus_labels.append(breast_tag)
        corpus.append(breast_tag.replace('_', ' '))  # Add target tag
        corpus.extend(synonyms)  # Add all synonyms

    # Generate embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    trait_embedding = model.encode([trait_lower])
    corpus_embeddings = model.encode(corpus)

    # Find best match using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(trait_embedding, corpus_embeddings)[0]

    best_idx = similarities.argmax()
    best_similarity = similarities[best_idx]

    # Map back to breast tag (corpus_labels aligned with first entry of each group)
    # Each group has 1 + len(synonyms) items, so we need to find which group
    group_sizes = [1 + len(synonyms) for synonyms in BODY_TYPE_MAPPINGS.values()]
    cumulative = 0
    for i, size in enumerate(group_sizes):
        if best_idx < cumulative + size:
            best_breast_tag = list(BODY_TYPE_MAPPINGS.keys())[i]
            break
        cumulative += size
    else:
        return None  # Shouldn't happen

    # Threshold for semantic match (code-only, no UI adjustment)
    SEMANTIC_THRESHOLD = 0.7

    if best_similarity > SEMANTIC_THRESHOLD:
        spaced_tag = best_breast_tag.replace('_', ' ')
        print(f"[DANBOORU_GEN] Body type mapping: '{trait}' -> '{spaced_tag}' (semantic, sim={best_similarity:.3f})")
        return spaced_tag

    return None


def filter_tags_for_search(tags: List[str]) -> List[str]:
    """
    Remove aesthetic/abstract traits that shouldn't be used for database search.

    Args:
        tags: List of normalized Danbooru tags

    Returns:
        List of tags with auto-remove traits filtered out
    """
    filtered = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower not in TRAITS_TO_REMOVE:
            filtered.append(tag)
        else:
            print(f"[DANBOORU_GEN] Auto-removed trait from search: '{tag}'")

    return filtered


def get_tag_importance(tag: str) -> int:
    """
    Get importance score for a tag (higher = keep longer in progressive search).

    Args:
        tag: Danbooru tag (spaced format)

    Returns:
        Importance score (higher values kept longer)
    """
    tag_lower = tag.lower()

    # Check tag category
    for color in HAIR_COLOR_KEYWORDS:
        if color == tag_lower or color in tag_lower:
            return TAG_CATEGORY_IMPORTANCE['hair_color']

    for style in HAIR_STYLE_KEYWORDS:
        if style == tag_lower or style in tag_lower:
            return TAG_CATEGORY_IMPORTANCE['hair_style']

    for eyes in EYE_COLOR_KEYWORDS:
        if eyes == tag_lower or eyes in tag_lower:
            return TAG_CATEGORY_IMPORTANCE['eye_color']

    for accessory in ACCESSORY_KEYWORDS:
        if accessory == tag_lower or accessory in tag_lower:
            return TAG_CATEGORY_IMPORTANCE['accessories']

    for cloth in CLOTHING_KEYWORDS:
        if cloth == tag_lower or cloth in tag_lower:
            return TAG_CATEGORY_IMPORTANCE['clothes']

    # Check if it's a body type (breast size)
    if 'breasts' in tag_lower or 'chest' in tag_lower:
        return TAG_CATEGORY_IMPORTANCE['body_type']

    # Default: low importance (remove early)
    return 0


def convert_to_underscored(tag: str) -> str:
    """
    Convert spaced Danbooru tag to underscored format for character search.

    Args:
        tag: Spaced format tag (e.g., "gray hair")

    Returns:
        Underscored format tag (e.g., "gray_hair")

    Examples:
        "gray hair" → "gray_hair"
        "small breasts" → "small_breasts"
        "pointy ears" → "pointy_ears"
    """
    return tag.replace(' ', '_').lower()


def normalize_traits_to_danbooru_tags(
    traits: List[str],
    max_traits: int = 8,
    threshold: float = 0.6  # Code-only parameter
) -> List[str]:
    """
    Convert natural language traits to Danbooru tags via semantic search.

    Pipeline:
    1. Split compound hair terms (e.g., "long blonde hair" → "long hair", "blonde hair")
    2. Map body types to breast sizes (e.g., "petite" → "small breasts")
    3. Semantic search remaining traits
    4. Deduplicate results
    5. Limit to max_traits

    Args:
        traits: List of natural language traits from PList
        max_traits: Maximum number of tags to return
        threshold: Minimum similarity threshold for semantic search (code-only parameter)

    Returns:
        List of normalized Danbooru tags (spaced format)

    Example:
        Input: ["skinny", "perky breasts", "long blonde hair", "petite"]
        Output: ["slim", "perky breasts", "long hair", "blonde hair", "small breasts"]
    """
    matched_tags = []

    # Step 1: Split compound hair terms
    expanded_traits = split_compound_hair_terms(traits)

    for trait in expanded_traits:
        if not trait or not trait.strip():
            continue

        print(f"[DANBOORU_GEN] Processing trait: '{trait}'")

        # Step 2: Check if it's a body type that maps to breast size
        breast_tag = map_body_type_to_breast_size(trait)
        if breast_tag:
            matched_tags.append(breast_tag)
            continue

        # Step 3: Semantic search danbooru_tags
        results = db_search_danbooru_tags_semantically(
            query_text=trait,
            k=1,  # Only need top match
            threshold=threshold
        )

        if results:
            top_match = results[0]
            tag_text = top_match['tag_text']
            similarity = top_match['similarity']

            print(f"[DANBOORU_GEN] Match: '{tag_text}' (similarity: {similarity:.3f})")

            # Add to matched tags if not duplicate
            if tag_text not in matched_tags:
                matched_tags.append(tag_text)
        else:
            print(f"[DANBOORU_GEN] No match found for: '{trait}'")

    # Step 4: Limit to max_traits
    if len(matched_tags) > max_traits:
        matched_tags = matched_tags[:max_traits]
        print(f"[DANBOORU_GEN] Limited to {max_traits} tags")

    return matched_tags


def count_matching_tags(character_tags: str, search_tags: List[str]) -> int:
    """
    Count how many search tags are present in the character's tags.
    
    Args:
        character_tags: Comma-separated string of character's tags
        search_tags: List of underscored tags to search for
    
    Returns:
        Number of matching tags
    """
    char_tags_lower = character_tags.lower()
    matches = 0
    for tag in search_tags:
        if tag.lower() in char_tags_lower:
            matches += 1
    return matches


def search_danbooru_character(
    tags: List[str],  # Spaced format
    gender: str,
    threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Find closest Danbooru character using progressive EXACT tag matching.

    Strategy:
    1. Query ALL characters of matching gender from database
    2. Progressive exact matching: Start with all N tags
    3. Filter characters by exact tag presence (count matching tags)
    4. Track best matches at each tag count level (N, N-1, N-2...)
    5. Return best match from highest tag count that has matches
    6. Only fall back to semantic search if no exact matches at all

    This ensures we find characters with actual matching tags rather than
    relying on semantic similarity which may not correlate with tag presence.

    Tag Priority (most to least important):
    - Hair color (100)
    - Hair style/length (90)
    - Body type/breast size (80)
    - Eye color (70)
    - Accessories (60)
    - Clothes (50)

    Args:
        tags: List of Danbooru tags (spaced format)
        gender: Character gender ('female', 'male', 'other')
        threshold: Minimum similarity threshold (unused, kept for API compatibility)

    Returns:
        Character dictionary with best tag match, or None if no results
    """
    if not tags:
        print("[DANBOORU_GEN] No tags provided for character search")
        return None

    # Auto-remove aesthetic/abstract traits
    filtered_tags = filter_tags_for_search(tags)

    if not filtered_tags:
        print("[DANBOORU_GEN] All tags were auto-removed, using random gender match")
        fallback = db_get_danbooru_characters(gender=gender, limit=10)
        if fallback:
            return random.choice(fallback)
        return None

    # Get ALL characters of matching gender (not just via semantic search)
    print(f"[DANBOORU_GEN] Loading all {gender} characters for exact tag matching...")
    all_characters = db_get_danbooru_characters(gender=gender, limit=9999)
    
    if not all_characters:
        print(f"[DANBOORU_GEN] No {gender} characters found in database")
        return None
    
    print(f"[DANBOORU_GEN] Loaded {len(all_characters)} characters for matching")

    # Sort tags by importance (most important first)
    sorted_tags = sorted(filtered_tags, key=get_tag_importance, reverse=True)
    underscored_tags = [convert_to_underscored(tag) for tag in sorted_tags]
    
    print(f"[DANBOORU_GEN] Tags to match (in priority order): {underscored_tags}")

    # Progressive matching: Try all tags, then N-1, then N-2, etc.
    best_matches_by_count = {}  # key: num_tags_matched, value: list of characters
    
    for num_tags in range(len(underscored_tags), 0, -1):
        # Get the top N most important tags for this iteration
        tags_to_match = underscored_tags[:num_tags]
        print(f"[DANBOORU_GEN] Checking for matches with {num_tags} tags: {tags_to_match}")
        
        matches_at_this_level = []
        
        for char in all_characters:
            char_all_tags = char.get('all_tags', '')
            matching_count = count_matching_tags(char_all_tags, tags_to_match)
            match_ratio = matching_count / len(tags_to_match) if tags_to_match else 0
            
            # Only consider characters that match ALL tags at this level
            if match_ratio == 1.0:
                char['_match_count'] = num_tags
                char['_matched_tags'] = tags_to_match.copy()
                matches_at_this_level.append(char)
        
        if matches_at_this_level:
            best_matches_by_count[num_tags] = matches_at_this_level
            print(f"[DANBOORU_GEN] Found {len(matches_at_this_level)} characters matching all {num_tags} tags")
        else:
            print(f"[DANBOORU_GEN] No characters match all {num_tags} tags")
    
    # Select the best match from the highest tag count that has matches
    for num_tags in range(len(underscored_tags), 0, -1):
        if num_tags in best_matches_by_count:
            matches = best_matches_by_count[num_tags]
            # Randomly select from top 5 for variety (or all if less than 5)
            top_matches = matches[:min(5, len(matches))]
            selected = random.choice(top_matches)
            
            if num_tags == len(underscored_tags):
                print(f"[DANBOORU_GEN] PERFECT MATCH: '{selected['name']}' matches all {num_tags} requested tags")
            else:
                print(f"[DANBOORU_GEN] PARTIAL MATCH: '{selected['name']}' matches {num_tags}/{len(underscored_tags)} tags: {selected['_matched_tags']}")
            
            return selected
    
    # No exact matches at any level - fall back to semantic search as last resort
    print(f"[DANBOORU_GEN] No exact tag matches found, falling back to semantic search...")
    
    query_text = f"{gender} {' '.join(underscored_tags)}"
    results = db_search_danbooru_characters_semantically(
        query_text=query_text,
        gender=gender,
        k=20,
        threshold=threshold
    )
    
    if results:
        best = results[0]
        print(f"[DANBOORU_GEN] Semantic fallback: '{best['name']}' (similarity: {best.get('similarity', 0):.3f})")
        return best
    
    # Absolute last resort: random character
    print("[DANBOORU_GEN] No matches at all, using random character")
    return random.choice(all_characters) if all_characters else None


def build_final_tag_string(
    character: Optional[Dict[str, Any]],
    raw_traits: List[str],  # Original PList physical traits
    gender: str,
    max_traits: int = 8
) -> str:
    """
    Build final Danbooru tag string for populating the field.

    Format: "character_tag, 1girl/1boy/solo, trait1, trait2, ..."

    Uses ORIGINAL PList physical traits (not normalized tags).

    Args:
        character: Matched Danbooru character (optional)
        raw_traits: List of original PList physical traits
        gender: Character gender ('female', 'male', 'other')
        max_traits: Maximum number of physical traits to include

    Returns:
        Comma-separated Danbooru tag string

    Example:
        Input: character={'name': 'akaza_akari'}, raw_traits=['pretty', 'powerful', 'long blonde hair', 'petite'], gender='female'
        Output: "akaza_akari, 1girl, pretty, powerful, long blonde hair, petite"
    """
    tags = []

    # Add character name/tag if found
    if character and character.get('name'):
        # Extract first tag from all_tags (character name is first element)
        all_tags = character.get('all_tags', '').split(',')
        if all_tags:
            character_tag = all_tags[0].strip()
            if character_tag:
                tags.append(character_tag)

    # Add gender count tag based on character card gender (not character database)
    if gender == 'female':
        tags.append('1girl')
    elif gender == 'male':
        tags.append('1boy')
    elif gender == 'other':
        tags.append('solo')
    else:
        # Unknown gender, default to 1girl
        tags.append('1girl')

    # Add original PList physical traits (limit to max_traits)
    if raw_traits:
        tags.extend(raw_traits[:max_traits])

    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_tags.append(tag)

    return ', '.join(unique_tags)


def generate_tags_from_description(
    description: str,
    gender: str
) -> Dict[str, Any]:
    """
    NEW: Semantic-based tag generation (two-stage approach).

    1. Parse PList body description
    2. Normalize traits to Danbooru tags (Stage 1)
    3. Find closest Danbooru character (Stage 2)
    4. Build final tag string

    Args:
        description: Character physical description (PList or plain text)
        gender: Character gender from character card ('female', 'male', 'other')

    Returns:
        {
            'success': bool,
            'suggested_tags': str,  # Danbooru tag string to populate field
            'visual_canon_id': int or None,
            'visual_canon_name': str or None,
            'extracted_traits': list,  # Physical traits found
            'normalized_tags': list,  # Danbooru tags after normalization
            'message': str
        }
    """
    # Validate inputs
    if not description or not description.strip():
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': [],
            'normalized_tags': [],
            'message': 'No description provided'
        }

    if not gender or gender == 'unknown':
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': [],
            'normalized_tags': [],
            'message': 'Please select a gender (Female/Male/Other) first'
        }

    print(f"[DANBOORU_GEN] Starting tag generation")

    # Step 1: Parse PList body → Extract raw traits
    raw_traits = parse_physical_body_plist(description)
    print(f"[DANBOORU_GEN] Extracted {len(raw_traits)} raw traits")

    if not raw_traits:
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': [],
            'normalized_tags': [],
            'message': 'Could not extract physical traits from description. Ensure description contains body details in PList format: "[Name\'s body= \"trait1\", \"trait2\", ...]"'
        }

    # Step 2: Normalize traits → Danbooru tags (Stage 1)
    normalized_tags = normalize_traits_to_danbooru_tags(
        traits=raw_traits,
        max_traits=8,
        threshold=0.6
    )

    if not normalized_tags:
        return {
            'success': False,
            'suggested_tags': '',
            'visual_canon_id': None,
            'visual_canon_name': None,
            'extracted_traits': raw_traits,
            'normalized_tags': [],
            'message': 'Could not match physical traits to Danbooru tags. Try using more specific descriptions (e.g., "long blonde hair" instead of "blonde").'
        }

    print(f"[DANBOORU_GEN] Normalized to {len(normalized_tags)} Danbooru tags: {normalized_tags}")

    # Step 3: Find closest Danbooru character (Stage 2)
    # Gender filter is applied in search function
    matched_character = search_danbooru_character(
        tags=normalized_tags,
        gender=gender,
        threshold=0.5
    )

    # Step 4: Build final tag string
    suggested_tags = build_final_tag_string(
        character=matched_character,
        raw_traits=raw_traits,
        gender=gender,
        max_traits=8
    )

    success = matched_character is not None

    return {
        'success': success,
        'suggested_tags': suggested_tags,
        'visual_canon_id': matched_character.get('id') if matched_character else None,
        'visual_canon_name': matched_character.get('name') if matched_character else None,
        'extracted_traits': raw_traits,
        'normalized_tags': normalized_tags,
        'message': f'Generated tags from Danbooru character: {matched_character.get("name", "Unknown")}' if matched_character else 'No character match found (using tags only)'
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
    4. Assign visual canon to database
    5. Store visual canon in metadata as fallback
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

        # Store the visual canon binding in database
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

        # Store visual canon in metadata as fallback
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
