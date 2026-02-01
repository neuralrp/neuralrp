"""
Snapshot Prompt Builder

Builds 4-block SD prompts from scene analysis results.
Uses danbooru_tags_config.py for block targets and fallbacks.

Block Structure:
- Block 0: Quality tags (hardwired, first N from config)
- Block 1: Subject tags (character + semantic matches)
- Block 2: Environment tags (setting semantic matches)
- Block 3: Style tags (mood/atmosphere semantic matches)
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import random

from app.danbooru_tags_config import (
    get_block_tags, get_block_target, get_fallback_tags,
    get_universal_negatives, get_min_matches, BLOCK_0
)

# Import learning config
from app.snapshot_learning_config import (
    NOVELTY_BONUS, NOVELTY_WEIGHT,
    FAVORITE_BONUS, FAVORITE_MAX_BONUS,
    SEMANTIC_THRESHOLD, SELECTION_TOP_K,
    TAG_DETECTION_THRESHOLD,
    USE_WEIGHTED_SELECTION
)


class SnapshotPromptBuilder:
    """Builds 4-block SD prompts from analysis results."""

    def __init__(self, analyzer):
        """
        Initialize prompt builder.

        Args:
            analyzer: SnapshotAnalyzer instance for semantic matching
        """
        self.analyzer = analyzer

    def _calculate_tag_score(self,
                            tag_text: str,
                            semantic_score: float,
                            variation_mode: bool) -> float:
        """
        Calculate final tag score with learning bias.

        Applies to BOTH snapshot and manual mode generation.

        Args:
            tag_text: Tag name (e.g., "1girl", "forest")
            semantic_score: Similarity score from embedding (0.0-1.0)
            variation_mode: If True, apply novelty scoring

        Returns:
            float: Final score for tag selection
        """
        from app.database import db_get_favorite_tag_frequency

        # Start with semantic relevance (primary factor)
        final_score = semantic_score

        # Get favorite frequency (how often this tag appears in user's favorites)
        favorite_freq = db_get_favorite_tag_frequency(tag_text)

        # Add favorite bias (user preference)
        # Linear growth with cap at FAVORITE_MAX_BONUS
        favorite_bias = min(favorite_freq * FAVORITE_BONUS, FAVORITE_MAX_BONUS)
        final_score += favorite_bias

        # Add novelty bonus (variation mode only)
        # Encourages exploration of less-used tags
        if variation_mode and favorite_freq > 0:
            # Novelty: 1.0 for never-used, decreases with frequency
            novelty_score = 1.0 / (1.0 + favorite_freq)
            final_score += novelty_score * NOVELTY_BONUS

        return final_score


    def _select_tags_weighted(self,
                             scored_candidates: List[Tuple[str, float]],
                             count: int) -> List[str]:
        """
        Select tags with weighted probability (avoids over-fitting).

        Instead of always picking top-N by score, selects randomly
        with probability proportional to score. This prevents always
        using same tags when user has strong preferences.

        Args:
            scored_candidates: List of (tag_text, final_score) tuples
            count: Number of tags to select

        Returns:
            List[str]: Selected tag texts
        """
        if not scored_candidates:
            return []

        # Extract scores
        scores = [score for _, score in scored_candidates]
        total_score = sum(scores)

        # If all scores are 0, fall back to top-N
        if total_score == 0:
            return [tag for tag, _ in scored_candidates[:count]]

        # Calculate probabilities
        probabilities = [s / total_score for s in scores]

        # Weighted random selection
        selected = []
        for _ in range(min(count, len(scored_candidates))):
            # Choose index based on probability
            chosen_idx = np.random.choice(len(scored_candidates), p=probabilities)
            selected.append(scored_candidates[chosen_idx][0])

            # Remove chosen from candidates
            scored_candidates.pop(chosen_idx)
            probabilities.pop(chosen_idx)

            # Recalculate probabilities for remaining candidates
            if probabilities:
                remaining_scores = [score for _, score in scored_candidates]
                remaining_total = sum(remaining_scores)
                probabilities = [s / remaining_total for s in remaining_scores]

        return selected

    def build_4_block_prompt(self,
                            scene_analysis: Dict[str, Any],
                            character_tag: Optional[str] = None,
                            variation_mode: bool = False) -> Tuple[str, str]:
        """
        Build SD prompt using 4-block structure with learning support.

        Args:
            scene_analysis: From SnapshotAnalyzer.analyze_scene()
            character_tag: Danbooru tag for selected character (from dropdown)
            variation_mode: If True, apply novelty scoring (for "Regenerate" button)

        Returns:
            (positive_prompt, negative_prompt)
        """
        # Extract scene info
        scene_type = scene_analysis.get('scene_type', 'other')
        setting = scene_analysis.get('setting', '')
        mood = scene_analysis.get('mood', '')

        mode_str = "VARIATION" if variation_mode else "NORMAL"
        print(f"[SNAPSHOT] Building prompt: scene_type={scene_type}, "
              f"setting={setting}, mood={mood}, char_tag={bool(character_tag)}, mode={mode_str}")

        # Build query text for semantic matching
        query_text = f"{scene_type} {setting} {mood}".strip()

        # ========================================
        # Block 0: Quality Tags (hardwired, NO LEARNING)
        # ========================================
        target_0 = get_block_target(0)
        block_0 = BLOCK_0[:target_0]  # First N quality tags

        # ========================================
        # Block 1: Subject Tags (WITH LEARNING)
        # ========================================
        block_1 = self._build_block_1(query_text, character_tag, variation_mode)

        # ========================================
        # Block 2: Environment Tags (WITH LEARNING)
        # ========================================
        block_2 = self._build_block_2(query_text, setting, scene_type, variation_mode)

        # ========================================
        # Block 3: Style/Rendering Tags (WITH LEARNING)
        # ========================================
        block_3 = self._build_block_3(query_text, mood, scene_type, variation_mode)

        # ========================================
        # Assemble Final Prompt
        # ========================================
        positive_parts = []

        # Add blocks in order (0-3 are config tags, 4 is user-added custom tags)
        positive_parts.extend(block_0)
        positive_parts.extend(block_1)
        positive_parts.extend(block_2)
        positive_parts.extend(block_3)
        
        # Add Block 4: User-added custom tags (from manual mode favorites)
        block_4 = self._build_block_4()
        positive_parts.extend(block_4)

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in positive_parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)

        # Join with commas (standard SD format)
        positive_prompt = ', '.join(unique_parts)

        # Generate negative prompt (universal only)
        negative_prompt = self._build_negative_prompt()

        print(f"[SNAPSHOT] {mode_str} prompt built: {len(unique_parts)} tags")

        return positive_prompt, negative_prompt

    def _build_block_1(self,
                       query_text: str,
                       character_tag: Optional[str],
                       variation_mode: bool = False) -> List[str]:
        """
        Build Block 1: Subject tags with learning.

        Args:
            query_text: Scene description for semantic matching
            character_tag: Danbooru tag for selected character
            variation_mode: If True, apply novelty scoring

        Returns:
            List[str]: Selected subject tags
        """
        block_1 = []
        target = get_block_target(1)
        min_matches = get_min_matches(1)

        # Priority 1: Character tag (if provided via dropdown)
        if character_tag:
            # Parse character danbooru_tag (comma-separated)
            char_tags = [t.strip() for t in character_tag.split(',') if t.strip()]
            block_1.extend(char_tags[:3])  # Limit to 3 character tags

            # Track frequency for used tags
            from app.database import db_increment_tag_frequency
            for tag in char_tags[:3]:
                db_increment_tag_frequency(tag)

        # Priority 2: Semantic matching for scene subjects (WITH LEARNING)
        if len(block_1) < target and query_text:
            # Get semantic matches (expanded pool for learning)
            subject_matches = self.analyzer.match_tags_semantically(
                query_text, block_num=1, k=SELECTION_TOP_K, threshold=SEMANTIC_THRESHOLD
            )

            # Score candidates with learning algorithm
            scored_candidates = []
            for tag_text, semantic_score in subject_matches:
                # Skip if already in block (e.g., character tag)
                if tag_text.lower() in [t.lower() for t in block_1]:
                    continue

                # Skip if below semantic threshold
                if semantic_score < SEMANTIC_THRESHOLD:
                    continue

                # Calculate final score with learning
                final_score = self._calculate_tag_score(tag_text, semantic_score, variation_mode)
                scored_candidates.append((tag_text, final_score))

            # Sort by final score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Select tags based on mode
            if variation_mode and USE_WEIGHTED_SELECTION:
                # Variation mode: weighted random selection (more variety)
                needed = target - len(block_1)
                selected = self._select_tags_weighted(scored_candidates, needed)
                block_1.extend(selected)
            else:
                # Normal mode: deterministic top-N
                for tag_text, _ in scored_candidates:
                    if len(block_1) >= target:
                        break
                    block_1.append(tag_text)
                    from app.database import db_increment_tag_frequency
                    db_increment_tag_frequency(tag_text)

        # Apply guardrail fallback if minimum not met
        if len(block_1) < min_matches:
            for fallback in get_fallback_tags(1):
                if fallback not in block_1:
                    block_1.append(fallback)
                    if len(block_1) >= min_matches:
                        break

        # Limit to target size
        return block_1[:target]

    def _build_block_2(self,
                       query_text: str,
                       setting: str,
                       scene_type: str,
                       variation_mode: bool = False) -> List[str]:
        """
        Build Block 2: Environment tags with learning.

        Args:
            query_text: Scene description for semantic matching
            setting: Scene setting from analysis
            scene_type: Scene type from analysis
            variation_mode: If True, apply novelty scoring

        Returns:
            List[str]: Selected environment tags
        """
        block_2 = []
        target = get_block_target(2)
        min_matches = get_min_matches(2)

        # Priority 1: Setting from scene analysis
        if setting:
            setting_matches = self.analyzer.match_tags_semantically(
                setting, block_num=2, k=SELECTION_TOP_K, threshold=SEMANTIC_THRESHOLD
            )

            # Score and select with learning
            scored_candidates = []
            for tag_text, semantic_score in setting_matches:
                if semantic_score >= SEMANTIC_THRESHOLD:
                    final_score = self._calculate_tag_score(tag_text, semantic_score, variation_mode)
                    scored_candidates.append((tag_text, final_score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if variation_mode and USE_WEIGHTED_SELECTION:
                needed = target - len(block_2)
                selected = self._select_tags_weighted(scored_candidates, needed)
                block_2.extend(selected)
            else:
                for tag_text, _ in scored_candidates:
                    if len(block_2) >= target:
                        break
                    block_2.append(tag_text)
                    from app.database import db_increment_tag_frequency
                    db_increment_tag_frequency(tag_text)

        # Priority 2: Scene type matching
        if len(block_2) < target and scene_type != 'other':
            scene_matches = self.analyzer.match_tags_semantically(
                scene_type, block_num=2, k=SELECTION_TOP_K, threshold=SEMANTIC_THRESHOLD
            )

            scored_candidates = []
            for tag_text, semantic_score in scene_matches:
                if semantic_score >= SEMANTIC_THRESHOLD and tag_text not in block_2:
                    final_score = self._calculate_tag_score(tag_text, semantic_score, variation_mode)
                    scored_candidates.append((tag_text, final_score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if variation_mode and USE_WEIGHTED_SELECTION:
                needed = target - len(block_2)
                selected = self._select_tags_weighted(scored_candidates, needed)
                block_2.extend(selected)
            else:
                for tag_text, _ in scored_candidates:
                    if len(block_2) >= target:
                        break
                    block_2.append(tag_text)
                    from app.database import db_increment_tag_frequency
                    db_increment_tag_frequency(tag_text)

        # Apply guardrail fallback
        if len(block_2) < min_matches:
            for fallback in get_fallback_tags(2):
                if fallback not in block_2:
                    block_2.append(fallback)
                    if len(block_2) >= min_matches:
                        break

        return block_2[:target]

    def _build_block_3(self,
                       query_text: str,
                       mood: str,
                       scene_type: str,
                       variation_mode: bool = False) -> List[str]:
        """
        Build Block 3: Style/Rendering tags with learning.

        Args:
            query_text: Scene description for semantic matching
            mood: Scene mood from analysis
            scene_type: Scene type from analysis
            variation_mode: If True, apply novelty scoring

        Returns:
            List[str]: Selected style tags
        """
        block_3 = []
        target = get_block_target(3)
        min_matches = get_min_matches(3)

        # Priority1: Mood from scene analysis
        if mood:
            mood_matches = self.analyzer.match_tags_semantically(
                mood, block_num=3, k=SELECTION_TOP_K, threshold=SEMANTIC_THRESHOLD
            )

            scored_candidates = []
            for tag_text, semantic_score in mood_matches:
                if semantic_score >= SEMANTIC_THRESHOLD:
                    final_score = self._calculate_tag_score(tag_text, semantic_score, variation_mode)
                    scored_candidates.append((tag_text, final_score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if variation_mode and USE_WEIGHTED_SELECTION:
                needed = target - len(block_3)
                selected = self._select_tags_weighted(scored_candidates, needed)
                block_3.extend(selected)
            else:
                for tag_text, _ in scored_candidates:
                    if len(block_3) >= target:
                        break
                    block_3.append(tag_text)
                    from app.database import db_increment_tag_frequency
                    db_increment_tag_frequency(tag_text)

        # Priority 2: Scene type matching for style
        if len(block_3) < target and scene_type != 'other':
            scene_style_matches = self.analyzer.match_tags_semantically(
                scene_type, block_num=3, k=SELECTION_TOP_K, threshold=SEMANTIC_THRESHOLD
            )

            scored_candidates = []
            for tag_text, semantic_score in scene_style_matches:
                if semantic_score >= SEMANTIC_THRESHOLD and tag_text not in block_3:
                    final_score = self._calculate_tag_score(tag_text, semantic_score, variation_mode)
                    scored_candidates.append((tag_text, final_score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if variation_mode and USE_WEIGHTED_SELECTION:
                needed = target - len(block_3)
                selected = self._select_tags_weighted(scored_candidates, needed)
                block_3.extend(selected)
            else:
                for tag_text, _ in scored_candidates:
                    if len(block_3) >= target:
                        break
                    block_3.append(tag_text)
                    from app.database import db_increment_tag_frequency
                    db_increment_tag_frequency(tag_text)

        # Apply guardrail fallback
        if len(block_3) < min_matches:
            for fallback in get_fallback_tags(3):
                if fallback not in block_3:
                    block_3.append(fallback)
                    if len(block_3) >= min_matches:
                        break

        return block_3[:target]

    def _build_block_4(self) -> List[str]:
        """
        Build Block 4: User-added custom tags (dynamic library expansion).
        
        These are tags added by users through manual mode favorites that aren't
        in the original 1560 danbooru tags config. They appear at the end of the
        prompt to provide additional customization without disrupting core structure.
        
        Returns:
            List[str]: User-added custom tags from block 4
        """
        block_4 = []
        
        try:
            from app.database import get_connection
            
            with get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all user-added tags (block_num=4)
                cursor.execute(
                    "SELECT tag_text FROM danbooru_tags WHERE block_num = 4 ORDER BY created_at DESC"
                )
                
                rows = cursor.fetchall()
                for row in rows:
                    tag_text = row[0] if isinstance(row, tuple) else row['tag_text']
                    if tag_text and tag_text not in block_4:
                        block_4.append(tag_text)
                        
                if block_4:
                    print(f"[SNAPSHOT] Block 4 (User): {len(block_4)} custom tags")
                    
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Could not load user tags from block 4: {e}")
        
        return block_4

    def _build_negative_prompt(self) -> str:
        """
        Build negative prompt using universal negatives only.

        No scene-specific negatives (simplified per design).
        """
        return ', '.join(get_universal_negatives())
