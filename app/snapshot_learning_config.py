"""
Configuration for snapshot learning and variation mode.
Applies to BOTH snapshot and manual mode image generation.
"""

# Novelty Scoring Parameters
NOVELTY_BONUS = 0.3          # Maximum novelty bonus for variation mode (0.0-1.0)
NOVELTY_WEIGHT = 0.3          # How much novelty affects final score (0.0-1.0)

# Favorite Bias Parameters
FAVORITE_BONUS = 0.15         # Per-favorite additive bonus (points per favorite)
FAVORITE_MAX_BONUS = 0.6      # Maximum favorite bias (capped at 4+ favorites)

# Semantic Matching Parameters
SEMANTIC_THRESHOLD = 0.3        # Minimum semantic score for consideration (0.0-1.0)
SELECTION_TOP_K = 50           # Number of candidates to evaluate per block

# Tag Detection Parameters
TAG_DETECTION_THRESHOLD = 2     # Minimum danbooru tags to trigger learning (default: 2)

# Selection Strategy
USE_WEIGHTED_SELECTION = True   # If True, use probability-weighted selection (avoids over-fitting)
# If False, use deterministic top-N (faster, less variety)
