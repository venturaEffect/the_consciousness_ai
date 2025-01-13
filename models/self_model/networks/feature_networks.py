"""
Feature Extraction Networks for Self Model in ACM

This module implements:
1. Core feature extraction for self representation
2. Integration with emotional context
3. Memory-based feature enhancement
4. Attention-driven feature selection

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for memory context
- models/core/consciousness_core.py for attention
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class FeatureNetwork(nn.Module):
    def __init__(self, config: Dict):
        """Initialize feature extraction networks"""
        super().__init__()
        self.config = config
        
        # Core feature extractors
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.emotional_encoder = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.feature_dim)
        )
        
    def extract_features(
        self,
        visual_input: torch.Tensor,
        emotional_context: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Extract multimodal features"""
        # Extract visual features
        visual_features = self.visual_encoder(visual_input)
        
        # Extract emotional features if context provided
        emotional_features = None
        if emotional_context is not None:
            emotional_features = self.emotional_encoder(
                emotional_context['features']
            )
            
        # Combine features
        combined = self._combine_features(
            visual_features, 
            emotional_features
        )
        
        return combined, {
            'visual_norm': torch.norm(visual_features).item(),
            'emotional_norm': torch.norm(emotional_features).item() if emotional_features is not None else 0.0
        }

class EmotionalStateNetwork(nn.Module):
    """
    Encodes emotional state information into latent representations.
    Uses a transformer-based architecture for temporal emotion processing.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_dim = config['emotional_hidden_dim']
        
        # Emotion embedding layers
        self.emotion_embedder = nn.Sequential(
            nn.Linear(config['emotion_dim'], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # Temporal processing
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=config['n_heads'],
                dim_feedforward=config['ff_dim']
            ),
            num_layers=config['n_layers']
        )
        
        # Output projection
        self.output_projector = nn.Linear(self.hidden_dim, config['embedding_dim'])

    def forward(self, emotion_values: Dict[str, float]) -> torch.Tensor:
        """Process emotional state into embedding"""
        # Convert emotion values to tensor
        emotion_tensor = self._dict_to_tensor(emotion_values)
        
        # Get embeddings
        embeddings = self.emotion_embedder(emotion_tensor)
        
        # Process through transformer
        temporal_features = self.temporal_transformer(embeddings)
        
        # Project to output space
        return self.output_projector(temporal_features)

class BehavioralNetwork(nn.Module):
    """
    Encodes behavioral patterns and action histories into latent space.
    Implements behavioral pattern recognition through temporal convolutions.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Behavioral feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=config['behavior_dim'],
                out_channels=config['behavior_hidden'],
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(config['behavior_hidden']),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=config['behavior_hidden'],
                out_channels=config['embedding_dim'],
                kernel_size=3,
                padding=1
            )
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config['embedding_dim'],
            num_heads=config['n_heads']
        )

    def forward(self, behavioral_sequence: torch.Tensor) -> torch.Tensor:
        """Process behavioral sequence into embedding"""
        # Extract behavioral features
        features = self.feature_extractor(behavioral_sequence)
        
        # Apply self-attention
        attended_features, _ = self.attention(features, features, features)
        
        return attended_features

class SocialContextNetwork(nn.Module):
    """
    Processes social interaction context and feedback.
    Implements social learning through feedback integration.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Social context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config['social_dim'], config['social_hidden']),
            nn.LayerNorm(config['social_hidden']),
            nn.GELU(),
            nn.Linear(config['social_hidden'], config['embedding_dim'])
        )
        
        # Feedback integration
        self.feedback_gate = nn.Sequential(
            nn.Linear(config['embedding_dim'] * 2, config['embedding_dim']),
            nn.Sigmoid()
        )

    def forward(
        self,
        social_context: torch.Tensor,
        prev_representation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process social context and integrate with previous representation"""
        # Encode social context
        context_embedding = self.context_encoder(social_context)
        
        # Integrate with previous representation if available
        if prev_representation is not None:
            gate = self.feedback_gate(
                torch.cat([context_embedding, prev_representation], dim=-1)
            )
            return gate * context_embedding + (1 - gate) * prev_representation
            
        return context_embedding