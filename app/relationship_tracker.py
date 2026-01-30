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
