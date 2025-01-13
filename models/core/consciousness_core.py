# models/core/consciousness_core.py
"""
Core consciousness implementation that integrates all ACM components.
Key functionalities:
- Manages consciousness emergence through attention mechanisms
- Integrates emotional, memory and learning subsystems
- Implements the consciousness gating system
- Coordinates with DreamerV3 for world modeling

Dependencies:
- models/memory/emotional_memory_core.py for emotional memory storage
- models/emotion/emotional_processing.py for affect handling
- models/predictive/dreamer_emotional_wrapper.py for world modeling
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
import logging
from simulations.api.simulation_manager import SimulationManager

@dataclass
class ConsciousnessState:
    """Tracks the current state of consciousness development"""
    emotional_awareness: float = 0.0
    attention_stability: float = 0.0
    memory_coherence: float = 0.0
    survival_adaptation: float = 0.0
    stress_management: float = 0.0
    learning_progress: float = 0.0

class ConsciousnessCore(nn.Module):
    """
    Core consciousness development system integrating:
    1. Emotional memory formation through stress-induced attention
    2. Multimodal fusion with emotional context
    3. Adaptive learning based on consciousness state
    4. Memory-guided behavior generation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize core components
        self.fusion = EmotionalMemoryFusion(config)
        self.memory = EmotionalMemoryCore(config)
        self.attention = ConsciousnessAttention(config)
        self.monitor = ConsciousnessMonitor(config)
        
        # Consciousness state
        self.state = ConsciousnessState()
        
        # Learning parameters
        self.base_lr = config.get('base_learning_rate', 0.001)
        self.min_lr = config.get('min_learning_rate', 0.0001)
        
        # Development thresholds
        self.attention_threshold = config.get('attention_threshold', 0.7)
        self.emotional_threshold = config.get('emotional_threshold', 0.6)
        self.memory_threshold = config.get('memory_threshold', 0.7)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def process_experience(
        self,
        current_state: Dict[str, torch.Tensor],
        emotion_values: Dict[str, float],
        stress_level: float,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process new experience for consciousness development"""
        
        # Validate tensor shapes
        assert current_state['encoded_state'].shape[0] == emotion_values['valence'].shape[0], "Shape mismatch in current_state and emotion_values"
        
        # Get attention based on stress and emotion
        attention_output, attention_metrics = self.attention.forward(
            input_state=current_state.get('encoded_state'),
            emotional_context=self.fusion.emotion_network.get_embedding(emotion_values),
            environment_context=context.get('environment_embedding') if context else None
        )
        
        # Check if experience is significant
        if self._is_significant_experience(
            attention_level=attention_metrics['attention_level'],
            emotion_values=emotion_values,
            stress_level=stress_level
        ):
            # Process through fusion system
            fusion_output, fusion_info = self.fusion.forward(
                text_input=current_state.get('text'),
                vision_input=current_state.get('vision'),
                audio_input=current_state.get('audio'),
                emotional_context=emotion_values,
                memory_context=self._get_relevant_memories(emotion_values)
            )
            
            # Store experience in memory
            self.memory.store_experience(
                state=current_state,
                emotion_values=emotion_values,
                attention_metrics=attention_metrics,
                fusion_info=fusion_info,
                stress_level=stress_level,
                context=context
            )
            
            # Update consciousness state
            self._update_consciousness_state(
                attention_metrics=attention_metrics,
                fusion_info=fusion_info,
                stress_level=stress_level
            )
            
            # Monitor development
            development_report = self.monitor.evaluate_development(
                current_state=current_state,
                emotion_values=emotion_values,
                attention_metrics=attention_metrics,
                stress_level=stress_level
            )
            
            return {
                'attention_output': attention_output,
                'fusion_output': fusion_output,
                'consciousness_state': self.get_consciousness_state(),
                'development_report': development_report
            }
            
        return {'attention_output': attention_output}
        
    def _is_significant_experience(
        self,
        attention_level: float,
        emotion_values: Dict[str, float],
        stress_level: float
    ) -> bool:
        """Determine if experience is significant for consciousness development"""
        # Check attention threshold
        if attention_level < self.attention_threshold:
            return False
            
        # Check emotional intensity
        emotional_intensity = sum(abs(v) for v in emotion_values.values()) / len(emotion_values)
        if emotional_intensity < self.emotional_threshold:
            return False
            
        # Check stress significance
        if stress_level < self.config['thresholds']['stress']:
            return False
            
        return True
        
    def _update_consciousness_state(
        self,
        attention_metrics: Dict[str, float],
        fusion_info: Dict,
        stress_level: float
    ):
        """Update consciousness development state"""
        # Update emotional awareness
        self.state.emotional_awareness = self._calculate_emotional_awareness(
            fusion_info.get('emotional_coherence', 0.0)
        )
        
        # Update attention stability
        self.state.attention_stability = self._calculate_attention_stability(
            attention_metrics
        )
        
        # Update memory coherence
        self.state.memory_coherence = self._calculate_memory_coherence()
        
        # Update survival adaptation
        self.state.survival_adaptation = self._calculate_survival_adaptation(
            stress_level
        )
        
        # Update stress management
        self.state.stress_management = self._calculate_stress_management(
            stress_level
        )
        
        # Update learning progress
        self.state.learning_progress = self._calculate_learning_progress()
        
    def get_consciousness_state(self) -> Dict:
        """Get current consciousness development state"""
        return {
            'emotional_awareness': self.state.emotional_awareness,
            'attention_stability': self.state.attention_stability,
            'memory_coherence': self.state.memory_coherence,
            'survival_adaptation': self.state.survival_adaptation,
            'stress_management': self.state.stress_management,
            'learning_progress': self.state.learning_progress,
            'consciousness_level': self._calculate_consciousness_level()
        }

    def _calculate_consciousness_level(self) -> float:
        """Calculate overall consciousness level"""
        weights = {
            'emotional_awareness': 0.25,
            'attention_stability': 0.20,
            'memory_coherence': 0.20,
            'survival_adaptation': 0.15,
            'stress_management': 0.10,
            'learning_progress': 0.10
        }
        
        return sum(
            getattr(self.state, metric) * weight
            for metric, weight in weights.items()
        )