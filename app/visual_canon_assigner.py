"""
Visual Canon Assigner

Automatically assigns Danbooru visual canon characters to NPCs/characters
based on physical traits, gender, and semantic matching.
"""

from typing import Dict, Any, Optional, List
import random

from app.database import (
    db_search_danbooru_characters_semantically,
    db_assign_visual_canon_to_character,
    db_assign_visual_canon_to_npc,
    db_get_danbooru_characters,
    db_get_character_visual_canon,
    db_get_npc_visual_canon
)


class VisualCanonAssigner:
    """Assigns visual canon characters based on constraints."""
    
    @staticmethod
    def extract_physical_traits(description: str) -> List[str]:
        """Extract physical traits from character description."""
        traits = []
        desc_lower = description.lower()
        
        # Common physical trait keywords
        hair_colors = ['blonde', 'brown', 'black', 'red', 'white', 'silver', 'pink', 'blue']
        hair_styles = ['long', 'short', 'ponytail', 'twintails', 'bob', 'straight', 'wavy']
        eye_colors = ['blue', 'green', 'brown', 'red', 'purple', 'yellow', 'black']
        body_types = ['slim', 'athletic', 'curvy', 'petite', 'tall', 'short']
        clothing = ['armor', 'dress', 'uniform', 'casual', 'formal', 'kimono']
        
        # Extract hair color
        for color in hair_colors:
            if color in desc_lower:
                traits.append(f"{color}_hair")
        
        # Extract hair style
        for style in hair_styles:
            if style in desc_lower:
                traits.append(f"{style}_hair")
        
        # Extract eye color
        for color in eye_colors:
            if f"{color} eyes" in desc_lower or f"{color}_eyes" in desc_lower:
                traits.append(f"{color}_eyes")
        
        # Extract body type
        for body in body_types:
            if body in desc_lower:
                traits.append(body)
        
        # Extract clothing
        for cloth in clothing:
            if cloth in desc_lower:
                traits.append(cloth)
        
        return traits
    
    @staticmethod
    def assign_visual_canon(
        entity_type: str,  # 'character' or 'npc'
        entity_id: str,    # filename for characters, entity_id for NPCs
        chat_id: Optional[str] = None,  # Required for NPCs
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Assign visual canon to an entity.
        
        Args:
            entity_type: 'character' or 'npc'
            entity_id: Character filename or NPC entity_id
            chat_id: Chat ID (required for NPCs)
            constraints: Optional dict with 'gender' and 'physical_tags'
        
        Returns:
            {
                'success': bool,
                'visual_canon_id': int,
                'visual_canon_name': str,
                'visual_canon_tags': str,
                'message': str
            }
        """
        constraints = constraints or {}
        gender = constraints.get('gender', 'unknown')
        physical_tags = constraints.get('physical_tags', [])
        
        # Build query for semantic search
        if physical_tags:
            query_text = ' '.join(physical_tags)
        else:
            query_text = f"{gender} character"
        
        print(f"[CANON_ASSIGN] Searching for visual canon: entity_type={entity_type}, "
              f"gender={gender}, tags={physical_tags}")
        
        # Search for matching characters (top 20, then random)
        matches = db_search_danbooru_characters_semantically(
            query_text=query_text,
            gender=gender,
            k=20,
            threshold=0.4
        )
        
        if not matches:
            # Fallback: random character with matching gender
            print(f"[CANON_ASSIGN] No semantic matches, using random selection")
            fallback_matches = db_get_danbooru_characters(gender=gender, limit=50)
            if not fallback_matches:
                return {
                    'success': False,
                    'message': f'No Danbooru characters found for gender: {gender}'
                }
            matches = [{'id': m['id'], 'name': m['name'], 'all_tags': m['all_tags']}
                      for m in fallback_matches]
        
        # Randomly select from top matches
        selected = random.choice(matches)
        canon_id = selected['id']
        canon_name = selected['name']
        canon_tags = selected.get('all_tags', '')
        
        print(f"[CANON_ASSIGN] Selected visual canon: {canon_name} (id={canon_id})")
        
        # Assign to entity
        if entity_type == 'character':
            success = db_assign_visual_canon_to_character(entity_id, canon_id, canon_tags)
        elif entity_type == 'npc':
            if not chat_id:
                return {'success': False, 'message': 'chat_id required for NPC assignment'}
            success = db_assign_visual_canon_to_npc(chat_id, entity_id, canon_id, canon_tags)
        else:
            return {'success': False, 'message': f'Invalid entity_type: {entity_type}'}
        
        if success:
            return {
                'success': True,
                'visual_canon_id': canon_id,
                'visual_canon_name': canon_name,
                'visual_canon_tags': canon_tags,
                'message': f'Assigned visual canon: {canon_name}'
            }
        else:
            return {'success': False, 'message': 'Failed to assign visual canon'}
    
    @staticmethod
    def reroll_visual_canon(
        entity_type: str,
        entity_id: str,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reroll visual canon (get new random match, excluding current).
        
        Args:
            entity_type: 'character' or 'npc'
            entity_id: Character filename or NPC entity_id
            chat_id: Chat ID (required for NPCs)
        
        Returns:
            Same as assign_visual_canon()
        """
        print(f"[CANON_ASSIGN] Rerolling visual canon: {entity_type}={entity_id}")
        
        # Get current binding to read constraints
        if entity_type == 'character':
            current_canon = db_get_character_visual_canon(entity_id)
        elif entity_type == 'npc':
            if not chat_id:
                return {'success': False, 'message': 'chat_id required for NPC reroll'}
            current_canon = db_get_npc_visual_canon(chat_id, entity_id)
        else:
            return {'success': False, 'message': f'Invalid entity_type: {entity_type}'}
        
        # Extract constraints from current canon (or use defaults)
        constraints = {
            'gender': current_canon.get('visual_canon_gender', 'unknown') if current_canon else 'unknown',
            'physical_tags': []  # Clear physical tags on reroll, let semantic search decide
        }
        
        # Get new assignment (will randomly select from matches)
        result = VisualCanonAssigner.assign_visual_canon(
            entity_type,
            entity_id,
            chat_id,
            constraints
        )
        
        # Ensure we don't get the same canon (re-roll once if same)
        if result['success'] and current_canon:
            current_id = current_canon.get('visual_canon_id')
            new_id = result['visual_canon_id']
            max_attempts = 3
            attempts = 0
            
            while new_id == current_id and attempts < max_attempts:
                print(f"[CANON_ASSIGN] Got same canon, re-rolling... (attempt {attempts + 1})")
                result = VisualCanonAssigner.assign_visual_canon(
                    entity_type,
                    entity_id,
                    chat_id,
                    constraints
                )
                new_id = result['visual_canon_id']
                attempts += 1
        
        return result
