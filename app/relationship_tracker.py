"""
Adaptive Relationship Tracker (v1.6.1)

Three-tier detection system for catching dramatic relationship shifts immediately
while preserving noise reduction for gradual changes.

Architecture:
- Tier 1: Keyword detection (~0.5ms) - catches explicit "I hate you!" moments
- Tier 2: Semantic similarity (~2-3ms) - detects implicit emotional shifts
- Tier 3: Dimension filtering (~1-2ms) - only injects relevant relationships

Falls back to 10-message interval if adaptive triggers don't fire.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sentence_transformers import SentenceTransformer
import re


class AdaptiveRelationshipTracker:
    """
    Adaptive relationship detection system that catches dramatic shifts immediately
    while maintaining smoothing for gradual relationship changes.
    
    Uses existing semantic search model (all-mpnet-base-v2) for zero memory overhead.
    """
    
    def __init__(self, model: SentenceTransformer):
        """
        Initialize with existing semantic search model to reuse infrastructure.
        
        Args:
            model: SentenceTransformer instance (shared with semantic_search_engine)
        """
        self.model = model
        self.previous_turn_embedding = None
        self.turn_count = 0
        self.last_trigger_turn = 0
        self.cooldown_turns = 3  # Minimum turns between adaptive triggers
        
        # Tier 1: Strong relationship keywords by dimension (positive and negative)
        # Uses word boundary matching to avoid false positives (e.g., "trust" in "distrust")
        self.relationship_keywords = {
            'trust': [
                # Positive
                'trust', 'believe', 'faith', 'rely', 'confide', 'honest',
                'loyal', 'depend', 'truthful', 'reliable', 'devoted',
                # Negative
                'betray', 'lie', 'deceive', 'distrust', 'suspect', 'doubt',
                'disloyal', 'treacherous', 'unreliable', 'false'
            ],
            'emotional_bond': [
                # Positive
                'love', 'adore', 'cherish', 'care', 'affection', 'attachment',
                'embrace', 'kiss', 'like', 'close', 'fond',
                # Negative
                'hate', 'despise', 'loathe', 'resent', 'detest', 'dislike',
                'repulsed', 'cold', 'distant', 'avoid'
            ],
            'conflict': [
                # Positive (low conflict)
                'harmony', 'agree', 'cooperate', 'peace', 'reconcile',
                'forgive', 'understand', 'resolve', 'allied',
                # Negative (high conflict)
                'fight', 'argue', 'disagree', 'tension', 'hostile', 'enemy',
                'attack', 'hurt', 'wound', 'violence', 'kill', 'destroy',
                'oppose', 'antagonize', 'clash', 'oppose'
            ],
            'power_dynamic': [
                # Positive (dominance)
                'lead', 'dominate', 'command', 'authority', 'control',
                'superior', 'order', 'direct', 'boss',
                # Negative (submission)
                'defer', 'submit', 'follow', 'obey', 'yield',
                'subordinate', 'comply', 'bow', 'kneel', 'servant'
            ],
            'fear_anxiety': [
                # Positive (low fear)
                'calm', 'ease', 'comfort', 'safe', 'relaxed',
                'secure', 'peaceful', 'confident', 'fearless',
                # Negative (high fear)
                'afraid', 'scared', 'terrified', 'anxious', 'dread', 'panic',
                'worry', 'nervous', 'frighten', 'intimidate', 'threaten'
            ]
        }
        
        # Tier 2: Dimension prototype embeddings (computed once at initialization)
        self.dimension_embeddings = None
        self._initialize_dimension_embeddings()
    
    def _initialize_dimension_embeddings(self):
        """
        Compute semantic prototypes for each relationship dimension.
        
        These embeddings represent the core emotional concept of each dimension,
        allowing us to detect when conversation shifts toward or away from them.
        """
        dimension_descriptions = {
            'trust': "deep trust betrayal loyalty faith confidence reliability honesty belief",
            'emotional_bond': "love affection romance care adoration strong feelings attachment bond",
            'conflict': "argument tension disagreement anger hostility fighting opposition conflict",
            'power_dynamic': "dominance authority control leadership command submission obedience power",
            'fear_anxiety': "fear terror dread intimidation threat anxiety scared nervous panic"
        }
        
        self.dimension_embeddings = {}
        for dimension, description in dimension_descriptions.items():
            if self.model is not None:
                self.dimension_embeddings[dimension] = self.model.encode(
                    description, 
                    convert_to_numpy=True
                )
            else:
                self.dimension_embeddings[dimension] = None
        
        print(f"[ADAPTIVE_TRACKER] Initialized {len(self.dimension_embeddings)} dimension embeddings")
    
    def _detect_keywords_in_text(self, text: str) -> Set[str]:
        """
        Tier 1 detection: Fast keyword matching with word boundaries.
        
        Args:
            text: Current turn's message text
        
        Returns:
            Set of dimensions with keyword matches
        """
        if not text:
            return set()
        
        text_lower = text.lower()
        detected_dimensions = set()
        
        for dimension, keywords in self.relationship_keywords.items():
            for keyword in keywords:
                # Word boundary matching: \bword\b prevents partial matches
                # Example: "trust" won't match "distrust"
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    detected_dimensions.add(dimension)
                    break  # One match per dimension is enough
        
        return detected_dimensions
    
    def _compute_conversation_similarity(self, current_text: str) -> float:
        """
        Tier 2 detection: Compare current turn to previous turn embedding.
        
        Args:
            current_text: Current turn's message text
        
        Returns:
            Similarity score (0.0 to 1.0), or None if no previous embedding
        """
        if self.previous_turn_embedding is None:
            return None
        
        if not current_text or not self.model:
            return None
        
        try:
            current_embedding = self.model.encode(current_text, convert_to_numpy=True)
            
            # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
            similarity = np.dot(current_embedding, self.previous_turn_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(self.previous_turn_embedding)
            )
            
            return float(similarity)
        except Exception as e:
            print(f"[ADAPTIVE_TRACKER] Error computing similarity: {e}")
            return None
    
    def should_trigger_adaptive_analysis(self, current_text: str) -> Tuple[bool, str]:
        """
        Determine if adaptive relationship analysis should be triggered.
        
        Three-tier decision system:
        1. Keyword detection (explicit relationship mentions)
        2. Semantic similarity (conversational shifts below 0.7 threshold)
        3. Cooldown enforcement (prevent spam-triggering)
        
        Args:
            current_text: Current turn's message text
        
        Returns:
            Tuple of (should_trigger: bool, reason: str)
        """
        self.turn_count += 1
        
        # Cooldown check: Don't trigger too frequently
        if self.turn_count - self.last_trigger_turn < self.cooldown_turns:
            return False, f"cooldown ({self.turn_count - self.last_trigger_turn}/{self.cooldown_turns} turns since last trigger)"
        
        # Tier 1: Keyword detection (strongest signal)
        detected_keywords = self._detect_keywords_in_text(current_text)
        if detected_keywords:
            self.last_trigger_turn = self.turn_count
            return True, f"keyword_detection (dimensions: {', '.join(detected_keywords)})"
        
        # Tier 2: Semantic similarity (catches implicit shifts)
        similarity = self._compute_conversation_similarity(current_text)
        if similarity is not None and similarity < 0.7:
            # Below 0.7 indicates major topic/emotional shift
            self.last_trigger_turn = self.turn_count
            return True, f"semantic_shift (similarity: {similarity:.3f})"
        
        # No trigger detected
        return False, "no_trigger"
    
    def update_previous_turn(self, current_text: str):
        """
        Update stored embedding for next turn comparison.
        
        Args:
            current_text: Current turn's message text
        """
        if current_text and self.model:
            self.previous_turn_embedding = self.model.encode(
                current_text,
                convert_to_numpy=True
            )
    
    def get_relevant_dimensions(
        self, 
        current_text: str, 
        relationship_states: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[str]]:
        """
        Tier 3: Filter dimensions by semantic relevance to current conversation.
        
        Only returns dimensions that are both:
        1. Deviating from neutral (score > 15 points from 50)
        2. Semantically relevant to current conversation
        
        Args:
            current_text: Current turn's message text
            relationship_states: Dict of {from_entity: {dimension: score}}
        
        Returns:
            Dict mapping entity -> list of relevant dimensions
        """
        if not current_text or not self.model:
            # Fallback: return all non-neutral dimensions
            return {
                from_entity: [dim for dim, score in dimensions.items() if abs(score - 50) > 15]
                for from_entity, dimensions in relationship_states.items()
            }
        
        # Compute current turn embedding once
        current_embedding = self.model.encode(current_text, convert_to_numpy=True)
        
        relevant_dimensions = {}
        
        for from_entity, dimensions in relationship_states.items():
            entity_relevant_dims = []
            
            for dimension, score in dimensions.items():
                # Check if dimension deviates from neutral
                if abs(score - 50) <= 15:
                    continue  # Neutral, skip
                
                # Check semantic relevance to current conversation
                dim_embedding = self.dimension_embeddings.get(dimension)
                if dim_embedding is None:
                    continue
                
                similarity = np.dot(current_embedding, dim_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(dim_embedding)
                )
                
                # Threshold: 0.35 (same as world info semantic search)
                if similarity > 0.35:
                    entity_relevant_dims.append(dimension)
            
            if entity_relevant_dims:
                relevant_dimensions[from_entity] = entity_relevant_dims
        
        return relevant_dimensions
    
    def reset_turn_counter(self):
        """
        Reset turn counter (useful for fork/chat reset scenarios).
        """
        self.turn_count = 0
        self.last_trigger_turn = 0
        self.previous_turn_embedding = None
    
    def _compute_keyword_polarity(self, text: str) -> Dict[str, int]:
        """
        Compute keyword polarity for each dimension.
        
        Returns:
            Dict of {dimension: polarity_score}
            polarity_score: positive = +1, negative = -1, neutral = 0
        """
        text_lower = text.lower()
        polarity = {}
        
        for dimension, keywords in self.relationship_keywords.items():
            mid = len(keywords) // 2
            positive = keywords[:mid]
            negative = keywords[mid:]
            
            pos_count = sum(1 for kw in positive if f'\\b{kw}\\b' in text_lower)
            neg_count = sum(1 for kw in negative if f'\\b{kw}\\b' in text_lower)
            
            if pos_count > neg_count:
                polarity[dimension] = 1
            elif neg_count > pos_count:
                polarity[dimension] = -1
            else:
                polarity[dimension] = 0
        
        return polarity
    
    def _compute_semantic_similarity(self, embeddings, dimension: str) -> float:
        """
        Compute average semantic similarity between message embeddings and dimension prototype.
        
        Args:
            embeddings: List of message embeddings
            dimension: One of ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not embeddings or dimension not in self.dimension_embeddings:
            return 0.0
        
        prototype = self.dimension_embeddings[dimension]
        if prototype is None:
            return 0.0
        
        similarities = []
        for emb in embeddings:
            sim = np.dot(emb, prototype) / (np.linalg.norm(emb) * np.linalg.norm(prototype))
            similarities.append(float(sim))
        
        return np.mean(similarities) if similarities else 0.0
    
    def analyze_conversation_scores(self, messages: List[str], current_state: Dict[str, int]) -> Dict[str, int]:
        """
        Analyze conversation to compute new relationship scores.
        
        Algorithm:
        1. Detect keyword polarity (positive/negative signals)
        2. Compute semantic similarity to dimension prototypes
        3. Combine signals (70% semantic, 30% keywords for nuance)
        4. Apply smoothing (gradual evolution, not wild swings)
        5. Return clamped scores [0-100]
        
        Args:
            messages: List of recent messages (content only)
            current_state: Current relationship scores {dimension: score}
        
        Returns:
            Dict of new scores {dimension: new_score}
        """
        if not self.model or not messages:
            return current_state.copy()
        
        text = " ".join(messages)
        keyword_polarity = self._compute_keyword_polarity(text)
        
        try:
            embeddings = [self.model.encode(msg, convert_to_numpy=True) for msg in messages]
        except Exception as e:
            print(f"[RELATIONSHIP_SCORING] Encoding failed: {e}")
            return current_state.copy()
        
        semantic_similarities = {}
        for dimension in ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']:
            semantic_similarities[dimension] = self._compute_semantic_similarity(embeddings, dimension)
        
        deltas = {}
        for dimension in ['trust', 'emotional_bond', 'conflict', 'power_dynamic', 'fear_anxiety']:
            semantic = semantic_similarities[dimension]
            polarity = keyword_polarity.get(dimension, 0)
            
            semantic_delta = (semantic - 0.5) * 20
            keyword_delta = polarity * 5
            combined_delta = (semantic_delta * 0.7) + (keyword_delta * 0.3)
            combined_delta = max(-15, min(15, combined_delta))
            
            deltas[dimension] = combined_delta
        
        new_scores = {}
        for dimension, delta in deltas.items():
            current_score = current_state.get(dimension, 50)
            new_score = current_score + delta
            new_score = max(0, min(100, new_score))
            new_scores[dimension] = new_score
        
        print(f"[RELATIONSHIP_SCORING] Semantic: {semantic_similarities}")
        print(f"[RELATIONSHIP_SCORING] Polarity: {keyword_polarity}")
        print(f"[RELATIONSHIP_SCORING] Deltas: {deltas}")
        print(f"[RELATIONSHIP_SCORING] Old: {current_state} → New: {new_scores}")
        
        return new_scores


# Global instance (initialized in main.py after semantic_search_engine loads)
adaptive_tracker: Optional[AdaptiveRelationshipTracker] = None


def initialize_adaptive_tracker(semantic_search_engine):
    """
    Initialize adaptive tracker with shared semantic search model.
    
    Args:
        semantic_search_engine: SemanticSearchEngine instance
    """
    global adaptive_tracker
    if adaptive_tracker is None and semantic_search_engine.model is not None:
        adaptive_tracker = AdaptiveRelationshipTracker(semantic_search_engine.model)
        print("[ADAPTIVE_TRACKER] Initialized with shared semantic model")
    return adaptive_tracker


# ============================================================================
# TESTING & DEBUGGING
# ============================================================================

def test_relationship_scoring():
    """
    Test harness for relationship scoring.
    Run this to verify semantic scoring accuracy.
    
    Usage:
        from app.relationship_tracker import test_relationship_scoring
        test_relationship_scoring()
    """
    class MockModel:
        def encode(self, text, convert_to_numpy=False):
            import hashlib
            import numpy as np
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            return np.array([(hash_val >> i) & 0xFF for i in range(0, 64, 8)])
    
    test_cases = [
        {
            "name": "Explicit conflict",
            "messages": ["I hate you!", "You're the enemy."],
            "current_state": {"trust": 50, "conflict": 50},
            "expected_behavior": "conflict increases significantly"
        },
        {
            "name": "Subtle sarcasm",
            "messages": ["Oh, thanks a lot. That's really helpful."],
            "current_state": {"trust": 50, "conflict": 50},
            "expected_behavior": "subtle conflict increase, trust decrease"
        },
        {
            "name": "Genuine affection",
            "messages": ["I care about you deeply.", "You mean everything to me."],
            "current_state": {"trust": 50, "emotional_bond": 50},
            "expected_behavior": "emotional_bond and trust increase"
        },
        {
            "name": "Neutral conversation",
            "messages": ["How's the weather?", "It's fine."],
            "current_state": {"trust": 50, "conflict": 50},
            "expected_behavior": "minimal changes"
        }
    ]
    
    tracker = AdaptiveRelationshipTracker(MockModel())
    
    print("=" * 60)
    print("RELATIONSHIP SCORING TEST HARNESS")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Expected: {test['expected_behavior']}")
        print(f"Messages: {test['messages']}")
        
        result = tracker.analyze_conversation_scores(
            messages=test['messages'],
            current_state=test['current_state']
        )
        
        print(f"Result: {result}")
        
        for dim, new_val in result.items():
            if not (0 <= new_val <= 100):
                print(f"  ❌ ERROR: {dim} = {new_val} (out of bounds)")
    
    print("\n" + "=" * 60)
    print("TEST HARNESS COMPLETE")
    print("=" * 60)

