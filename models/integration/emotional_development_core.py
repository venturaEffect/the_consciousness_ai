# models/integration/emotional_development_core.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.predictive.attention_mechanism import ConsciousnessAttention

@dataclass
class DevelopmentState:
    """Tracks consciousness development state"""
    emotional_awareness: float = 0.0
    attention_stability: float = 0.0
    memory_coherence: float = 0.0
    stress_adaptation: float = 0.0
    learning_progress: float = 0.0
    
class EmotionalDevelopmentCore:
    """
    Core integration of emotional learning and consciousness development
    
    Key Features:
    1. Stress-induced attention activation
    2. Emotional memory formation
    3. Consciousness metrics tracking
    4. Adaptive learning rates
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize core components
        self.fusion = EmotionalMemoryFusion(config)
        self.memory = EmotionalMemoryCore(config)
        self.attention = ConsciousnessAttention(config)
        
        # Development state
        self.state = DevelopmentState()
        self.experience_history = []
        
        # Learning parameters
        self.base_lr = config.get('base_learning_rate', 0.001)
        self.min_lr = config.get('min_learning_rate', 0.0001)
        
    def process_experience(
        self,
        current_state: Dict[str, torch.Tensor],
        emotion_values: Dict[str, float],
        stress_level: float,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process new experience for consciousness development"""
        
        # Get attention based on stress
        attention_output, attention_metrics = self.attention.forward(
            input_state=current_state.get('encoded_state'),
            emotional_context=self.fusion.emotion_network.get_embedding(emotion_values),
            environment_context=context.get('environment_embedding') if context else None
        )
        
        # Process through fusion system
        fusion_output, fusion_info = self.fusion.forward(
            text_input=current_state.get('text'),
            vision_input=current_state.get('vision'),
            audio_input=current_state.get('audio'),
            emotional_context=emotion_values,
            memory_context=self._get_relevant_memories(emotion_values)
        )
        
        # Store experience if significant
        if self._is_significant_experience(
            attention_level=attention_metrics['attention_level'],
            emotion_values=emotion_values,
            stress_level=stress_level
        ):
            self._store_experience(
                state=current_state,
                emotion_values=emotion_values,
                attention_metrics=attention_metrics,
                fusion_info=fusion_info,
                stress_level=stress_level,
                context=context
            )
        
        # Update development state
        self._update_development_state(
            attention_metrics=attention_metrics,
            fusion_info=fusion_info,
            stress_level=stress_level
        )
        
        # Calculate effective learning rate
        effective_lr = self._calculate_learning_rate()
        
        return {
            'attention_output': attention_output,
            'fusion_output': fusion_output,
            'development_state': self.get_development_state(),
            'learning_rate': effective_lr
        }
        
    def _is_significant_experience(
        self,
        attention_level: float,
        emotion_values: Dict[str, float],
        stress_level: float
    ) -> bool:
        """Determine if experience is significant for development"""
        # Check attention threshold
        if attention_level < self.config['thresholds']['attention']:
            return False
            
        # Check emotional intensity
        emotional_intensity = sum(abs(v) for v in emotion_values.values()) / len(emotion_values)
        if emotional_intensity < self.config['thresholds']['emotion']:
            return False
            
        # Check stress significance
        if stress_level < self.config['thresholds']['stress']:
            return False
            
        return True
        
    def _store_experience(self, **kwargs):
        """Store significant experience"""
        self.experience_history.append(kwargs)
        self.memory.store_experience(**kwargs)
        
    def _update_development_state(
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
        
        # Update stress adaptation
        self.state.stress_adaptation = self._calculate_stress_adaptation(
            stress_level
        )
        
        # Update learning progress
        self.state.learning_progress = self._calculate_learning_progress()
        
    def _calculate_learning_rate(self) -> float:
        """Calculate effective learning rate based on development"""
        # Scale learning rate by consciousness level
        consciousness_factor = (
            self.state.emotional_awareness +
            self.state.attention_stability +
            self.state.memory_coherence
        ) / 3.0
        
        effective_lr = self.base_lr * consciousness_factor
        
        # Ensure minimum learning rate
        return max(self.min_lr, effective_lr)
        
    def get_development_state(self) -> Dict:
        """Get current development state"""
        return {
            'emotional_awareness': self.state.emotional_awareness,
            'attention_stability': self.state.attention_stability, 
            'memory_coherence': self.state.memory_coherence,
            'stress_adaptation': self.state.stress_adaptation,
            'learning_progress': self.state.learning_progress,
            'consciousness_score': self._calculate_consciousness_score()
        }