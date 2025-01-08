# models/core/consciousness_core.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.nn import functional as F
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.evaluation.consciousness_monitor import ConsciousnessMonitor

@dataclass
class ConsciousnessState:
    """Tracks emotional and consciousness state"""
    emotional_awareness: float = 0.0
    attention_stability: float = 0.0
    memory_coherence: float = 0.0
    stress_adaptation: float = 0.0
    consciousness_level: float = 0.0

class ConsciousnessCore(nn.Module):
    """
    Core module integrating emotional learning with consciousness development
    
    Key Features:
    1. Stress-induced attention activation
    2. Emotional memory formation
    3. Consciousness-weighted learning
    4. Temporal coherence tracking
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Core dimensions
        self.hidden_size = config.get('hidden_size', 768)
        self.num_emotions = config.get('num_emotions', 3)
        self.num_heads = config.get('num_heads', 12)
        
        # Emotional encoding
        self.emotion_encoder = nn.Sequential(
            nn.Linear(self.num_emotions, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Attention mechanism 
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.get('dropout', 0.1)
        )
        
        # Memory integration
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Stress modulation
        self.stress_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Consciousness development
        self.consciousness_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # State tracking
        self.state = ConsciousnessState()
        self.memory_buffer = []
        
    def forward(
        self,
        emotional_input: Dict[str, torch.Tensor],
        memory_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Process emotional input for consciousness development"""
        
        # Encode emotional state
        emotion_values = torch.stack([
            emotional_input['valence'],
            emotional_input['arousal'],
            emotional_input['dominance']
        ], dim=-1)
        
        emotional_features = self.emotion_encoder(emotion_values)
        
        # Process memory context
        if memory_context is not None:
            memory_features = self.memory_encoder(memory_context)
            
            # Calculate attention
            attention_output, attention_weights = self.attention(
                query=emotional_features,
                key=memory_features,
                value=memory_features,
                attn_mask=attention_mask
            )
        else:
            attention_output = emotional_features
            attention_weights = None
            
        # Calculate stress level
        stress_level = self.stress_gate(attention_output)
        
        # Update consciousness state
        consciousness_input = torch.cat([
            emotional_features,
            attention_output,
            memory_features if memory_context is not None else torch.zeros_like(emotional_features)
        ], dim=-1)
        
        consciousness_level = self.consciousness_predictor(consciousness_input)
        
        # Update state
        self._update_state(
            emotional_features=emotional_features,
            attention_output=attention_output,
            stress_level=stress_level,
            consciousness_level=consciousness_level
        )
        
        # Store significant experiences
        if self._is_significant_experience(stress_level):
            self._store_experience(
                emotional_features=emotional_features,
                attention_output=attention_output,
                stress_level=stress_level,
                consciousness_level=consciousness_level
            )
            
        return attention_output, self.get_metrics()
        
    def _update_state(
        self,
        emotional_features: torch.Tensor,
        attention_output: torch.Tensor,
        stress_level: torch.Tensor,
        consciousness_level: torch.Tensor
    ):
        """Update consciousness development state"""
        # Update emotional awareness
        self.state.emotional_awareness = float(
            torch.mean(emotional_features).item()
        )
        
        # Update attention stability
        self.state.attention_stability = float(
            1.0 - torch.std(attention_output).item()
        )
        
        # Update memory coherence
        self.state.memory_coherence = self._calculate_memory_coherence()
        
        # Update stress adaptation
        self.state.stress_adaptation = float(
            torch.mean(1.0 - stress_level).item()
        )
        
        # Update consciousness level
        self.state.consciousness_level = float(
            consciousness_level.mean().item()
        )
        
    def _is_significant_experience(self, stress_level: torch.Tensor) -> bool:
        """Determine if experience should be stored"""
        return float(stress_level.mean().item()) > self.config.get('stress_threshold', 0.6)
        
    def _store_experience(
        self,
        emotional_features: torch.Tensor,
        attention_output: torch.Tensor,
        stress_level: torch.Tensor,
        consciousness_level: torch.Tensor
    ):
        """Store significant experience"""
        experience = {
            'emotional_features': emotional_features.detach(),
            'attention_output': attention_output.detach(),
            'stress_level': float(stress_level.mean().item()),
            'consciousness_level': float(consciousness_level.mean().item()),
            'timestamp': torch.tensor(time.time())
        }
        
        self.memory_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.memory_buffer) > self.config.get('max_memories', 1000):
            self.memory_buffer = self.memory_buffer[-self.config.get('max_memories', 1000):]
            
    def _calculate_memory_coherence(self) -> float:
        """Calculate coherence of stored memories"""
        if len(self.memory_buffer) < 2:
            return 0.0
            
        recent_memories = self.memory_buffer[-100:]
        coherence_scores = []
        
        for i in range(len(recent_memories) - 1):
            score = F.cosine_similarity(
                recent_memories[i]['emotional_features'],
                recent_memories[i + 1]['emotional_features']
            ).mean()
            coherence_scores.append(score)
            
        return float(torch.mean(torch.stack(coherence_scores)).item())
        
    def get_metrics(self) -> Dict:
        """Get current consciousness metrics"""
        return {
            'emotional_awareness': self.state.emotional_awareness,
            'attention_stability': self.state.attention_stability,
            'memory_coherence': self.state.memory_coherence,
            'stress_adaptation': self.state.stress_adaptation,
            'consciousness_level': self.state.consciousness_level
        }