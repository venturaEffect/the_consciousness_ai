# models/core/consciousness_core.py

"""
Core consciousness implementation that integrates the foundational narrative model
with emotional development and meta-memory systems.

Key components:
- LLaMA 3.3 foundation model for narrative generation
- Meta-memory system for experience reinforcement  
- Controlled adaptation mechanisms
- Emotional integration through EGNN
"""

import torch
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.language.llama_3_3 import LlamaForCausalLM
from models.predictive.attention_mechanism import ConsciousnessAttention
import logging
from simulations.api.simulation_manager import SimulationManager

@dataclass
class ConsciousnessState:
    """Enhanced state tracking for consciousness development"""
    emotional_awareness: float = 0.0
    narrative_coherence: float = 0.0
    memory_stability: float = 0.0 
    attention_focus: float = 0.0
    meta_memory_weight: float = 0.0
    imagination_activity: float = 0.0

class ConsciousnessCore:
    def __init__(self, config):
        """Initialize the consciousness system with enhanced components"""
        self.config = config
        
        # Initialize foundational narrative model (LLaMA 3.3)
        self.narrator = LlamaForCausalLM.from_pretrained(
            config.model_paths.llama,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Initialize key subsystems
        self.memory = EmotionalMemoryCore(config)
        self.emotion = EmotionalGraphNetwork()
        self.attention = ConsciousnessAttention(config)
        
        # Meta-memory tracking
        self.meta_memory = {
            'stable_patterns': [],
            'novel_experiences': [],
            'reinforcement_weights': {}
        }
        
        # Experience thresholds
        self.novelty_threshold = config.consciousness.memory.novelty_threshold
        self.stability_threshold = config.consciousness.memory.stability_threshold
        
    def process_experience(
        self,
        input_state: Dict[str, torch.Tensor],
        emotional_context: Optional[Dict] = None,
        imagination_context: Optional[Dict] = None
    ) -> Tuple[Dict, ConsciousnessState]:
        """Process new experiences through the enhanced consciousness pipeline"""
        
        # Get emotional embedding from EGNN
        emotional_embedding = self.emotion.analyze(
            input_state,
            self.meta_memory['stable_patterns']
        )
        
        # Generate narrative understanding 
        narrative = self._generate_narrative(
            input_state,
            emotional_embedding,
            imagination_context
        )
        
        # Update meta-memory with controlled adaptation
        stability_score = self._update_meta_memory(
            emotional_embedding,
            narrative
        )
        
        # Track consciousness state
        current_state = ConsciousnessState(
            emotional_awareness=emotional_embedding.mean().item(),
            narrative_coherence=narrative['coherence_score'],
            memory_stability=stability_score,
            attention_focus=self.attention.get_focus_score(),
            meta_memory_weight=len(self.meta_memory['stable_patterns']),
            imagination_activity=imagination_context['activity_score'] if imagination_context else 0.0
        )
        
        return {
            'narrative': narrative,
            'emotional_context': emotional_embedding,
            'meta_memory_state': self.meta_memory
        }, current_state
        
    def _generate_narrative(
        self,
        input_state: Dict[str, torch.Tensor],
        emotional_context: torch.Tensor,
        imagination_context: Optional[Dict] = None
    ) -> Dict:
        """Generate narrative understanding using LLaMA 3.3"""
        # Prepare prompt with context
        prompt = self._prepare_narrative_prompt(
            input_state,
            emotional_context,
            imagination_context
        )
        
        # Generate response from LLaMA
        with torch.no_grad():
            output = self.narrator.generate(
                prompt,
                max_length=self.config.generation.max_length,
                temperature=self.config.generation.temperature
            )
            
        return self._parse_narrative_response(output)
        
    def _update_meta_memory(
        self,
        emotional_embedding: torch.Tensor,
        narrative: Dict
    ) -> float:
        """Update meta-memory with controlled adaptation"""
        # Calculate stability score
        stability_score = self._calculate_stability(
            emotional_embedding,
            narrative
        )
        
        # Handle novel experiences
        if stability_score < self.novelty_threshold:
            self.meta_memory['novel_experiences'].append({
                'embedding': emotional_embedding,
                'narrative': narrative,
                'weight': 0.1  # Start with low weight
            })
            
        # Reinforce stable patterns
        elif stability_score > self.stability_threshold:
            self._reinforce_pattern(emotional_embedding, narrative)
            
        return stability_score