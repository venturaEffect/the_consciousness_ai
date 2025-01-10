"""
Emotional Context Processing

Implements emotional state processing for memory formation through:
1. Emotional state encoding
2. Context-based memory indexing
3. Temporal emotional coherence
4. Consciousness-weighted processing

Based on holonic principles where emotional context influences both 
local processing and global system behavior.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class EmotionalContextNetwork(nn.Module):
    """
    Processes emotional context for memory formation and retrieval
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Emotion embedding 
        self.emotion_embedder = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['embedding_dim'])
        )
        
        # Temporal processing
        self.temporal_processor = nn.GRU(
            input_size=config['embedding_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['n_layers']
        )
        
        # Context integration
        self.context_integration = nn.MultiheadAttention(
            embed_dim=config['hidden_dim'],
            num_heads=config['n_heads']
        )

    def forward(
        self,
        emotional_state: Dict[str, float],
        memory_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Process emotional state with optional memory context"""
        
        # Get emotion embedding
        emotion_values = torch.tensor([
            emotional_state[k] for k in sorted(emotional_state.keys())
        ])
        emotion_embedding = self.emotion_embedder(emotion_values)
        
        # Process temporal context if available
        if memory_context is not None:
            temporal_features, _ = self.temporal_processor(
                memory_context.unsqueeze(0)
            )
            
            # Integrate with current emotion
            context_integrated, attention_weights = self.context_integration(
                emotion_embedding.unsqueeze(0),
                temporal_features,
                temporal_features
            )
            
            emotion_embedding = context_integrated.squeeze(0)
            
        return emotion_embedding