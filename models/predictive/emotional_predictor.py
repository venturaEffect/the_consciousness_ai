# models/predictive/emotional_predictor.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class EmotionalState:
    """Tracks emotional state development"""
    valence: float = 0.0  # Pleasure-displeasure
    arousal: float = 0.0  # Energy level
    dominance: float = 0.0  # Control level
    stress_level: float = 0.0
    attention_focus: float = 0.0
    emotional_stability: float = 0.0

class EmotionalPredictor(nn.Module):
    """
    Predicts emotional development and stress responses
    
    Key Features:
    1. Multimodal emotion prediction
    2. Stress-induced attention modulation
    3. Temporal emotional stability tracking
    4. Consciousness-weighted predictions
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Core parameters
        self.hidden_size = config.get('hidden_size', 768)
        self.num_emotions = config.get('num_emotions', 3)  # VAD dimensions
        self.num_heads = config.get('num_heads', 8)
        
        # Neural components
        self.emotional_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Attention for temporal context
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # Emotion prediction heads
        self.valence_head = nn.Linear(self.hidden_size, 1)
        self.arousal_head = nn.Linear(self.hidden_size, 1)
        self.dominance_head = nn.Linear(self.hidden_size, 1)
        
        # Stress prediction
        self.stress_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        )
        
        # State tracking
        self.state = EmotionalState()
        self.history: List[EmotionalState] = []
        
    def forward(
        self,
        input_state: torch.Tensor,
        attention_context: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Process input state for emotional predictions"""
        
        # Encode emotional features
        emotional_features = self.emotional_encoder(input_state)
        
        # Apply temporal attention if context available
        if attention_context is not None:
            emotional_features, _ = self.temporal_attention(
                query=emotional_features,
                key=attention_context,
                value=attention_context
            )
            
        # Predict emotional dimensions (VAD)
        valence = torch.sigmoid(self.valence_head(emotional_features))
        arousal = torch.sigmoid(self.arousal_head(emotional_features))
        dominance = torch.sigmoid(self.dominance_head(emotional_features))
        
        # Calculate stress level
        stress_input = torch.cat([
            emotional_features,
            memory_context if memory_context is not None else torch.zeros_like(emotional_features)
        ], dim=-1)
        stress_level = torch.sigmoid(self.stress_predictor(stress_input))
        
        # Update emotional state
        self._update_state(
            valence=valence.mean().item(),
            arousal=arousal.mean().item(),
            dominance=dominance.mean().item(),
            stress_level=stress_level.mean().item()
        )
        
        predictions = {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'stress_level': stress_level
        }
        
        metrics = self.get_metrics()
        
        return predictions, metrics
        
    def _update_state(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        stress_level: float
    ):
        """Update emotional state tracking"""
        # Update current state
        self.state.valence = valence
        self.state.arousal = arousal
        self.state.dominance = dominance
        self.state.stress_level = stress_level
        
        # Calculate stability
        self.state.emotional_stability = self._calculate_stability()
        
        # Calculate attention focus from arousal and stress
        self.state.attention_focus = self._calculate_attention_focus(
            arousal=arousal,
            stress_level=stress_level
        )
        
        # Store state
        self.history.append(EmotionalState(**vars(self.state)))
        
        # Maintain history size
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
            
    def _calculate_stability(self) -> float:
        """Calculate emotional stability from history"""
        if len(self.history) < 2:
            return 1.0
            
        # Calculate variance of emotional dimensions
        recent_states = self.history[-100:]
        valence_var = np.var([s.valence for s in recent_states])
        arousal_var = np.var([s.arousal for s in recent_states])
        dominance_var = np.var([s.dominance for s in recent_states])
        
        # Higher stability = lower variance
        return float(1.0 / (1.0 + (valence_var + arousal_var + dominance_var) / 3))
        
    def _calculate_attention_focus(
        self,
        arousal: float,
        stress_level: float
    ) -> float:
        """Calculate attention focus level"""
        # Attention increases with both arousal and stress
        base_attention = (arousal + stress_level) / 2
        
        # Modulate by stability
        return float(base_attention * (1.0 + self.state.emotional_stability))
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current emotional metrics"""
        return {
            'valence': self.state.valence,
            'arousal': self.state.arousal,
            'dominance': self.state.dominance,
            'stress_level': self.state.stress_level,
            'attention_focus': self.state.attention_focus,
            'emotional_stability': self.state.emotional_stability
        }