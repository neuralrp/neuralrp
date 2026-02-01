"""
Danbooru Tag Configuration for Snapshot Feature

This file contains hardcoded Danbooru tags organized into 4 blocks:
- Block 0: Quality/Masterpiece tags (60 tags)
- Block 1: Subject/Entity tags (600 tags)
- Block 2: Environment/Setting tags (300 tags)
- Block 3: Style/Rendering tags (600 tags)

Edit this file to customize tag lists. Changes take effect on next app restart.

Block Size Targets (NOT configurable via UI, edit here to change):
- Block 0: 3 tags (quality)
- Block 1: 5 tags (subject)
- Block 2: 2 tags (environment)
- Block 3: 4 tags (style)

Total Tags: ~1560
"""

from typing import Dict, List

# ============================================================================
# CONFIGURATION (Edit these values to change behavior)
# ============================================================================

# Tag count per block in final prompt
BLOCK_TARGETS: Dict[int, int] = {
    0: 3,   # Quality
    1: 5,   # Subject
    2: 2,   # Environment
    3: 4    # Style
}

# Guardrail fallback tags (used when semantic matching fails)
FALLBACK_TAGS: Dict[int, List[str]] = {
    0: ["masterpiece", "best quality", "high quality"],
    1: ["1girl", "solo", "portrait"],
    2: ["simple background"],
    3: ["cinematic lighting", "detailed"]
}

# Minimum matches required per block before fallback triggers
MIN_MATCHES: Dict[int, int] = {
    0: 0,   # Quality is hardwired, no minimum
    1: 2,   # Subject needs at least 2 matches
    2: 1,   # Environment needs at least 1 match
    3: 2    # Style needs at least 2 matches
}

# Universal negative tags (always used, no scene-specific negatives)
UNIVERSAL_NEGATIVES: List[str] = [
    "low quality", "worst quality", "bad anatomy"
]

# ============================================================================
# BLOCK 0: QUALITY/MASTERPIECE (60 tags)
# ============================================================================

BLOCK_0: List[str] = [
    # Core quality (always use first 3)
    "masterpiece", "best quality", "high quality",
    # Extended quality
    "ultra detailed", "extremely detailed", "intricate details",
    "absurdres", "highres", "high resolution",
    "professional", "award winning", "perfect", "flawless",
    "cinematic", "dramatic", "sharp focus", "depth of field",
    "8k", "4k", "detailed", "refined", "polished",
    "exceptional", "magnificent", "stunning", "breathtaking",
    "glorious", "spectacular", "outstanding", "extraordinary",
    "supreme", "unparalleled", "elite", "premium",
    "top-tier", "first-class", "world-class", "superior",
    "excellent", "exquisite", "brilliant", "radiant",
    "luminous", "vibrant", "vivid", "saturated",
    "colorful", "harmonious", "balanced", "aesthetic",
    "appealing", "attractive", "beautiful", "elegant",
    "graceful", "sophisticated", "stylish", "fashionable",
    "modern", "contemporary", "crisp", "clean"
]

# ============================================================================
# BLOCK 1: SUBJECT/ENTITY (600 tags)
# ============================================================================

BLOCK_1: List[str] = [
    # === Gender/Count (20 tags) ===
    "1girl", "1boy", "2girls", "2boys", "3girls", "3boys",
    "4girls", "4boys", "5girls", "5boys", "6+girls", "6+boys",
    "solo", "duo", "trio", "quartet", "group", "crowd",
    "multiple girls", "multiple boys",
    
    # === Hair Color (30 tags) ===
    "blonde hair", "brown hair", "black hair", "red hair", "blue hair",
    "green hair", "purple hair", "pink hair", "white hair", "silver hair",
    "gray hair", "orange hair", "multicolored hair", "gradient hair",
    "two-tone hair", "streaked hair", "highlighted hair", "dark hair",
    "light hair", "platinum blonde", "strawberry blonde", "auburn hair",
    "chestnut hair", "copper hair", "golden hair", "honey hair",
    "lavender hair", "teal hair", "aqua hair", "cyan hair",
    
    # === Hair Style (50 tags) ===
    "long hair", "short hair", "medium hair", "very long hair",
    "ponytail", "twintails", "twin tails", "braid", "braids",
    "bun", "hair bun", "double bun", "side ponytail", "high ponytail",
    "low ponytail", "messy ponytail", "french braid", "fishtail braid",
    "crown braid", "side braid", "twin braids", "single braid",
    "hair ribbon", "hair ornament", "hairband", "headband",
    "hair between eyes", "ahoge", "antenna hair", "hime cut",
    "bob cut", "pixie cut", "layered hair", "asymmetrical hair",
    "wavy hair", "straight hair", "curly hair", "spiky hair",
    "messy hair", "neat hair", "wet hair", "windblown hair",
    "hair over one eye", "hair over eyes", "bangs", "blunt bangs",
    "parted bangs", "swept bangs", "side bangs", "forehead",
    
    # === Eyes (40 tags) ===
    "blue eyes", "green eyes", "brown eyes", "red eyes", "purple eyes",
    "yellow eyes", "orange eyes", "pink eyes", "golden eyes", "amber eyes",
    "heterochromia", "multicolored eyes", "glowing eyes", "bright eyes",
    "closed eyes", "half-closed eyes", "open eyes", "narrow eyes",
    "wide eyes", "looking at viewer", "looking away", "looking back",
    "looking up", "looking down", "looking to the side", "eye contact",
    "upward gaze", "downward gaze", "sideways glance", "staring",
    "tear in eye", "tears", "crying", "teary eyes",
    "sparkle in eye", "dilated pupils", "slit pupils", "heart-shaped pupils",
    "empty eyes", "crazy eyes",
    
    # === Expression (60 tags) ===
    "smile", "smiling", "grin", "laughing", "giggling",
    "frowning", "scowling", "angry", "furious", "rage",
    "sad", "crying", "sobbing", "depressed", "melancholy",
    "surprised", "shocked", "astonished", "amazed", "startled",
    "blushing", "embarrassed", "flustered", "shy", "bashful",
    "determined", "serious", "focused", "intense", "fierce",
    "confident", "smug", "proud", "arrogant", "haughty",
    "nervous", "anxious", "worried", "scared", "terrified",
    "confused", "puzzled", "curious", "interested", "intrigued",
    "bored", "tired", "sleepy", "exhausted", "drowsy",
    "happy", "joyful", "excited", "enthusiastic", "cheerful",
    "calm", "serene", "peaceful", "relaxed", "content",
    
    # === Body Type (30 tags) ===
    "tall", "short", "petite", "slim", "slender",
    "muscular", "athletic", "toned", "fit", "buff",
    "curvy", "voluptuous", "thicc", "plump", "chubby",
    "child", "teen", "young adult", "adult", "mature",
    "loli", "shota", "milf", "dilf", "elderly",
    "large breasts", "medium breasts", "small breasts", "flat chest", "huge breasts",
    
    # === Pose (80 tags) ===
    "standing", "sitting", "lying down", "kneeling", "crouching",
    "squatting", "leaning", "bending", "stretching", "jumping",
    "running", "walking", "dancing", "fighting", "attacking",
    "defending", "blocking", "dodging", "falling", "flying",
    "floating", "hovering", "swimming", "diving", "climbing",
    "arms up", "arms down", "arms crossed", "arms behind back", "arms behind head",
    "hand on hip", "hands on hips", "hand on chest", "hand on face", "hand on chin",
    "pointing", "waving", "reaching", "grabbing", "holding",
    "hugging", "embracing", "kissing", "fighting stance", "battle stance",
    "ready stance", "relaxed pose", "dynamic pose", "action pose", "static pose",
    "casual pose", "formal pose", "dramatic pose", "heroic pose", "villain pose",
    "sitting on chair", "sitting on floor", "sitting on bed", "sitting cross-legged", "seiza",
    "lying on back", "lying on side", "lying on stomach", "sprawled", "curled up",
    "from above", "from below", "from behind", "from side", "three-quarter view",
    "profile", "back view", "frontal view", "full body", "upper body",
    "lower body", "cowboy shot", "portrait", "close-up", "extreme close-up",
    
    # === Clothing (150 tags) ===
    "dress", "gown", "sundress", "wedding dress", "ball gown",
    "skirt", "miniskirt", "long skirt", "pleated skirt", "pencil skirt",
    "pants", "jeans", "shorts", "hotpants", "leggings",
    "shirt", "blouse", "t-shirt", "tank top", "crop top",
    "jacket", "coat", "blazer", "hoodie", "cardigan",
    "sweater", "pullover", "turtleneck", "vest", "waistcoat",
    "suit", "tuxedo", "business suit", "formal wear", "casual wear",
    "uniform", "school uniform", "military uniform", "maid outfit", "nurse outfit",
    "armor", "plate armor", "leather armor", "chainmail", "fantasy armor",
    "robe", "cloak", "cape", "mantle", "poncho",
    "kimono", "yukata", "hanfu", "qipao", "cheongsam",
    "swimsuit", "bikini", "one-piece swimsuit", "school swimsuit", "competition swimsuit",
    "underwear", "lingerie", "bra", "panties", "stockings",
    "nightgown", "pajamas", "sleepwear", "negligee", "babydoll",
    "gloves", "fingerless gloves", "long gloves", "elbow gloves", "mittens",
    "boots", "high heels", "sneakers", "sandals", "barefoot",
    "hat", "cap", "beret", "hood", "crown",
    "glasses", "sunglasses", "monocle", "goggles", "eyepatch",
    "jewelry", "necklace", "earrings", "bracelet", "ring",
    "scarf", "necktie", "bowtie", "ribbon", "choker",
    "belt", "sash", "corset", "apron", "suspenders",
    "wings", "tail", "horns", "ears", "halo",
    "mask", "helmet", "headphones", "headset", "tiara",
    "weapon", "sword", "gun", "staff", "wand",
    "bag", "backpack", "purse", "satchel", "messenger bag",
    
    # === Actions (60 tags) ===
    "holding sword", "holding gun", "holding staff", "holding book", "holding cup",
    "eating", "drinking", "smoking", "reading", "writing",
    "playing instrument", "playing guitar", "playing piano", "singing", "dancing",
    "cooking", "cleaning", "working", "studying", "sleeping",
    "bathing", "showering", "swimming", "exercising", "training",
    "fighting", "casting spell", "using magic", "summoning", "healing",
    "talking", "shouting", "whispering", "laughing", "crying",
    "thinking", "meditating", "praying", "celebrating", "mourning",
    "traveling", "exploring", "hunting", "gathering", "crafting",
    "riding horse", "driving", "piloting", "sailing", "flying",
    "sneaking", "hiding", "watching", "guarding", "protecting",
    "attacking", "defending", "retreating", "charging", "dodging",
]

# ============================================================================
# BLOCK 2: ENVIRONMENT/SETTING (300 tags)
# ============================================================================

BLOCK_2: List[str] = [
    # === Nature Locations (60 tags) ===
    "forest", "jungle", "woods", "grove", "clearing",
    "meadow", "field", "grassland", "prairie", "savanna",
    "mountain", "cliff", "canyon", "valley", "hill",
    "beach", "shore", "coast", "island", "peninsula",
    "ocean", "sea", "lake", "river", "stream",
    "waterfall", "pond", "swamp", "marsh", "wetland",
    "desert", "dunes", "oasis", "wasteland", "badlands",
    "tundra", "arctic", "glacier", "ice field", "frozen lake",
    "volcano", "lava field", "hot spring", "geyser", "crater",
    "cave", "cavern", "grotto", "underground", "mines",
    "sky", "clouds", "atmosphere", "stratosphere", "space",
    "garden", "park", "botanical garden", "greenhouse", "orchard",
    
    # === Urban Locations (60 tags) ===
    "city", "town", "village", "metropolis", "downtown",
    "street", "alley", "boulevard", "avenue", "plaza",
    "building", "skyscraper", "tower", "apartment", "house",
    "castle", "palace", "fortress", "citadel", "keep",
    "temple", "shrine", "church", "cathedral", "monastery",
    "tavern", "inn", "bar", "pub", "restaurant",
    "shop", "store", "market", "bazaar", "marketplace",
    "library", "museum", "gallery", "theater", "arena",
    "school", "academy", "university", "classroom", "dormitory",
    "hospital", "clinic", "laboratory", "workshop", "forge",
    "prison", "dungeon", "jail", "cell", "torture chamber",
    "graveyard", "cemetery", "crypt", "mausoleum", "tomb",
    
    # === Interior Locations (50 tags) ===
    "room", "bedroom", "living room", "dining room", "kitchen",
    "bathroom", "hallway", "corridor", "staircase", "attic",
    "basement", "cellar", "storage room", "closet", "wardrobe",
    "throne room", "ballroom", "banquet hall", "grand hall", "foyer",
    "office", "study", "den", "lounge", "parlor",
    "balcony", "terrace", "rooftop", "patio", "veranda",
    "bath", "hot tub", "pool", "spa", "sauna",
    "stable", "barn", "shed", "garage", "warehouse",
    "bridge", "dock", "pier", "harbor", "port",
    "train station", "airport", "bus stop", "subway", "platform",
    
    # === Time of Day (20 tags) ===
    "day", "daytime", "morning", "noon", "afternoon",
    "evening", "dusk", "sunset", "twilight", "golden hour",
    "night", "nighttime", "midnight", "late night", "dawn",
    "sunrise", "blue hour", "magic hour", "overcast day", "bright day",
    
    # === Weather/Atmosphere (40 tags) ===
    "sunny", "clear sky", "blue sky", "cloudy", "overcast",
    "rainy", "rain", "heavy rain", "light rain", "drizzle",
    "snowy", "snow", "snowfall", "blizzard", "snowstorm",
    "stormy", "storm", "thunderstorm", "lightning", "thunder",
    "foggy", "fog", "mist", "misty", "haze",
    "windy", "wind", "breeze", "gust", "hurricane",
    "humid", "dry", "cold", "hot", "warm",
    "freezing", "scorching", "mild", "pleasant", "harsh",
    
    # === Background Elements (50 tags) ===
    "tree", "trees", "flowers", "grass", "plants",
    "rocks", "stones", "boulders", "pebbles", "gravel",
    "water", "waves", "ripples", "reflection", "waterfall",
    "fire", "flames", "smoke", "steam", "embers",
    "moon", "sun", "stars", "aurora", "meteor",
    "clouds", "sky", "horizon", "mountains in background", "city in background",
    "simple background", "white background", "black background", "gradient background", "blurred background",
    "detailed background", "intricate background", "fantasy background", "abstract background", "no background",
    "indoors", "outdoors", "interior", "exterior", "landscape",
    "scenic", "panorama", "vista", "view", "scenery",
    
    # === Fantasy/Sci-Fi Settings (20 tags) ===
    "fantasy world", "magical realm", "enchanted forest", "fairy tale", "mythical",
    "sci-fi", "futuristic", "cyberpunk", "steampunk", "post-apocalyptic",
    "alien planet", "space station", "spaceship interior", "dystopian", "utopian",
    "parallel world", "dream world", "nightmare realm", "spirit world", "underworld",
]

# ============================================================================
# BLOCK 3: STYLE/RENDERING (600 tags)
# ============================================================================

BLOCK_3: List[str] = [
    # === Lighting (80 tags) ===
    "cinematic lighting", "dramatic lighting", "soft lighting", "natural lighting", "studio lighting",
    "backlight", "backlighting", "rim lighting", "rim light", "silhouette",
    "volumetric lighting", "volumetric light", "god rays", "light rays", "sunbeams",
    "point light", "spotlight", "ambient light", "diffused light", "harsh light",
    "warm light", "cool light", "cold light", "golden light", "silver light",
    "neon lighting", "neon glow", "neon lights", "blacklight", "UV light",
    "candlelight", "firelight", "torchlight", "moonlight", "starlight",
    "sunlight", "daylight", "twilight light", "dawn light", "dusk light",
    "bioluminescent", "bioluminescence", "glowing", "luminous", "radiant",
    "sparkles", "particles", "light particles", "dust particles", "floating particles",
    "lens flare", "bloom", "glow effect", "light leak", "chromatic aberration",
    "shadow", "shadows", "dramatic shadow", "soft shadow", "hard shadow",
    "ambient occlusion", "global illumination", "radiosity", "ray tracing", "path tracing",
    "subsurface scattering", "caustics", "reflection", "refraction", "specular",
    "highlight", "highlights", "lowlight", "contrast", "high contrast",
    "low contrast", "exposure", "overexposed", "underexposed", "balanced exposure",
    
    # === Art Style (100 tags) ===
    "anime", "anime style", "manga style", "japanese animation", "korean webtoon",
    "cartoon", "western cartoon", "disney style", "pixar style", "dreamworks style",
    "realistic", "photorealistic", "hyperrealistic", "semi-realistic", "stylized realistic",
    "illustration", "book illustration", "storybook", "fairy tale style", "children's book",
    "concept art", "game concept", "movie concept", "character concept", "environment concept",
    "digital art", "digital painting", "digital illustration", "cg", "cgi",
    "traditional art", "traditional media", "hand drawn", "hand painted", "handcrafted",
    "oil painting", "watercolor", "gouache", "acrylic", "tempera",
    "pencil drawing", "pen drawing", "ink drawing", "charcoal", "graphite",
    "sketch", "rough sketch", "finished sketch", "lineart", "line art",
    "cel shading", "cel shaded", "flat color", "flat shading", "simple shading",
    "detailed shading", "complex shading", "gradient shading", "smooth shading", "soft shading",
    "painterly", "impressionist", "expressionist", "surrealist", "abstract",
    "minimalist", "maximalist", "detailed", "intricate", "ornate",
    "retro", "vintage", "classic", "modern", "contemporary",
    "pixel art", "8-bit", "16-bit", "voxel", "low poly",
    "3d render", "3d art", "blender", "unreal engine", "unity",
    "vector art", "flat design", "material design", "isometric", "orthographic",
    "ukiyo-e", "art nouveau", "art deco", "gothic", "baroque",
    "pop art", "comic book", "graphic novel", "manga panel", "doujinshi",
    
    # === Camera/Composition (80 tags) ===
    "depth of field", "shallow depth of field", "deep depth of field", "bokeh", "background blur",
    "wide angle", "ultra wide", "fisheye", "telephoto", "zoom",
    "macro", "micro", "close-up", "extreme close-up", "medium close-up",
    "medium shot", "medium full shot", "full shot", "long shot", "extreme long shot",
    "high angle", "low angle", "eye level", "bird's eye view", "worm's eye view",
    "dutch angle", "tilted", "canted angle", "dynamic angle", "dramatic angle",
    "rule of thirds", "golden ratio", "centered composition", "symmetrical", "asymmetrical",
    "framing", "natural framing", "geometric framing", "negative space", "positive space",
    "leading lines", "diagonal lines", "curved lines", "s-curve", "converging lines",
    "foreground", "midground", "background", "layers", "depth",
    "panorama", "panoramic", "wide shot", "establishing shot", "landscape orientation",
    "portrait orientation", "square format", "cinematic aspect ratio", "widescreen", "letterbox",
    "pov", "first person", "third person", "over the shoulder", "reverse shot",
    "split screen", "montage", "collage", "vignette", "frame within frame",
    "motion blur", "speed lines", "action lines", "impact frame", "freeze frame",
    "tilt shift", "miniature effect", "toy camera", "lomography", "film grain",
    
    # === Color/Tone (80 tags) ===
    "vibrant colors", "vivid colors", "bright colors", "bold colors", "saturated colors",
    "muted colors", "desaturated", "pastel colors", "soft colors", "light colors",
    "dark colors", "deep colors", "rich colors", "earthy colors", "earth tones",
    "warm colors", "warm tones", "warm palette", "cool colors", "cool tones",
    "cool palette", "neutral colors", "neutral tones", "balanced colors", "harmonious colors",
    "complementary colors", "analogous colors", "triadic colors", "split complementary", "tetradic",
    "monochrome", "monochromatic", "black and white", "grayscale", "sepia",
    "duotone", "tritone", "color splash", "selective color", "color pop",
    "high saturation", "low saturation", "oversaturated", "undersaturated", "natural saturation",
    "high key", "low key", "middle key", "full range", "limited palette",
    "neon colors", "fluorescent", "electric colors", "psychedelic", "trippy",
    "sunset colors", "sunrise colors", "twilight colors", "night colors", "golden colors",
    "blue hour", "magic hour", "autumn colors", "spring colors", "summer colors",
    "winter colors", "seasonal colors", "natural colors", "artificial colors", "synthetic colors",
    "color harmony", "color contrast", "color balance", "color grading", "color correction",
    "teal and orange", "orange and blue", "red and cyan", "purple and yellow", "green and magenta",
    
    # === Mood/Atmosphere (80 tags) ===
    "dramatic", "epic", "cinematic", "theatrical", "grand",
    "peaceful", "serene", "tranquil", "calm", "relaxing",
    "mysterious", "enigmatic", "secretive", "cryptic", "arcane",
    "romantic", "intimate", "sensual", "passionate", "tender",
    "dark", "gloomy", "somber", "melancholic", "depressing",
    "bright", "cheerful", "happy", "joyful", "uplifting",
    "tense", "suspenseful", "thrilling", "exciting", "intense",
    "scary", "horror", "creepy", "eerie", "unsettling",
    "magical", "mystical", "enchanting", "whimsical", "fantastical",
    "futuristic", "technological", "digital", "cyber", "electronic",
    "nostalgic", "retro", "vintage", "classic", "timeless",
    "elegant", "sophisticated", "refined", "classy", "luxurious",
    "gritty", "raw", "rough", "industrial", "urban",
    "natural", "organic", "earthy", "rustic", "primitive",
    "ethereal", "heavenly", "divine", "angelic", "celestial",
    "demonic", "hellish", "infernal", "dark fantasy", "grimdark",
    
    # === Quality Modifiers (80 tags) ===
    "highly detailed", "extremely detailed", "incredibly detailed", "insanely detailed", "ultra detailed",
    "fine details", "micro details", "intricate details", "complex details", "subtle details",
    "sharp", "crisp", "clear", "focused", "in focus",
    "soft", "smooth", "gentle", "delicate", "refined",
    "textured", "rough texture", "smooth texture", "fabric texture", "skin texture",
    "realistic texture", "stylized texture", "detailed texture", "subtle texture", "pronounced texture",
    "clean", "polished", "finished", "professional", "studio quality",
    "raw", "unprocessed", "natural look", "authentic", "genuine",
    "artistic", "creative", "imaginative", "original", "unique",
    "beautiful", "gorgeous", "stunning", "breathtaking", "awe-inspiring",
    "epic scale", "grand scale", "massive", "immense", "colossal",
    "intimate scale", "personal", "close", "detailed focus", "character focus",
    "environment focus", "background focus", "action focus", "emotion focus", "story focus",
    "balanced composition", "dynamic composition", "static composition", "asymmetric composition", "geometric composition",
    "award winning", "masterful", "expert", "skilled", "talented",
    "trending on artstation", "featured on pixiv", "popular", "viral", "famous",
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_tags() -> Dict[int, List[str]]:
    """Return all tags organized by block number."""
    return {
        0: BLOCK_0,
        1: BLOCK_1,
        2: BLOCK_2,
        3: BLOCK_3
    }

def get_block_tags(block_num: int) -> List[str]:
    """Get tags for a specific block."""
    return get_all_tags().get(block_num, [])

def get_block_target(block_num: int) -> int:
    """Get target tag count for a block in final prompt."""
    return BLOCK_TARGETS.get(block_num, 3)

def get_all_targets() -> Dict[int, int]:
    """Get all block targets."""
    return BLOCK_TARGETS.copy()

def get_fallback_tags(block_num: int) -> List[str]:
    """Get fallback tags for a specific block."""
    return FALLBACK_TAGS.get(block_num, [])

def get_min_matches(block_num: int) -> int:
    """Get minimum required matches for a block before fallback triggers."""
    return MIN_MATCHES.get(block_num, 1)

def get_universal_negatives() -> List[str]:
    """Get universal negative tags."""
    return UNIVERSAL_NEGATIVES.copy()

def get_total_tag_count() -> int:
    """Get total number of tags across all blocks."""
    all_tags = get_all_tags()
    return sum(len(tags) for tags in all_tags.values())

def get_tags_as_tuples() -> List[tuple]:
    """Get all tags as (tag_text, block_num) tuples for database migration."""
    result = []
    all_tags = get_all_tags()
    for block_num, tags in all_tags.items():
        for tag_text in tags:
            result.append((tag_text, block_num))
    return result
