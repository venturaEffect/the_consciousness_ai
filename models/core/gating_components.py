"""
Gating Component Networks

Implements specialized gating mechanisms for different aspects of consciousness:
1. Attention-based gating
2. Emotional salience gating  
3. Stress response gating
4. Temporal coherence gating

Each component acts both independently and as part of the holonic system.
"""

class AttentionGate(nn.Module):
    """Gates information flow based on attention levels"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['state_dim']),
            nn.Sigmoid()
        )

    def forward(
        self, 
        x: torch.Tensor,
        attention_level: float
    ) -> torch.Tensor:
        """Apply attention-based gating"""
        gate_values = self.attention_net(x)
        return x * gate_values * attention_level

class EmotionalGate(nn.Module):
    """Gates information based on emotional salience"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.emotion_encoder = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['state_dim']),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        emotional_context: Dict[str, float]
    ) -> torch.Tensor:
        """Apply emotion-based gating"""
        emotion_tensor = torch.tensor([
            emotional_context[k] for k in sorted(emotional_context.keys())
        ])
        gate_values = self.emotion_encoder(emotion_tensor)
        return x * gate_values

class TemporalCoherenceGate(nn.Module):
    """Gates information based on temporal consistency"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config['state_dim'],
            num_heads=config['n_heads']
        )
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(config['state_dim'], config['state_dim']),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        temporal_context: torch.Tensor
    ) -> torch.Tensor:
        """Apply temporal coherence gating"""
        # Apply temporal attention
        attended_features, _ = self.temporal_attention(
            x.unsqueeze(0),
            temporal_context.unsqueeze(0),
            temporal_context.unsqueeze(0)
        )
        
        # Generate gate values
        gate_values = self.gate_net(attended_features.squeeze(0))
        
        return x * gate_values