"""
Emotional Context Network

Implements emotional state encoding and processing for memory formation.
Provides emotional context for consciousness development.
"""

class EmotionalContextNetwork(nn.Module):
    """
    Processes emotional context for memory integration
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.emotion_embedding = nn.Linear(
            config['emotion_dim'],
            config['emotion_hidden_dim']
        )
        
        self.context_processor = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config['emotion_hidden_dim'],
                nhead=config['n_heads']
            ),
            num_layers=config['n_layers']
        )
        
        self.output_projection = nn.Linear(
            config['emotion_hidden_dim'],
            config['embedding_dim']
        )

    def forward(self, emotion_values: Dict[str, float]) -> torch.Tensor:
        """
        Process emotional context into embedding
        
        Args:
            emotion_values: Dictionary of emotion dimensions and values
        """
        # Convert emotion values to tensor
        emotion_tensor = self._dict_to_tensor(emotion_values)
        
        # Get initial embedding
        emotion_embedding = self.emotion_embedding(emotion_tensor)
        
        # Process through transformer
        context_features = self.context_processor(emotion_embedding)
        
        # Project to output space
        return self.output_projection(context_features)

    def _dict_to_tensor(self, emotion_dict: Dict[str, float]) -> torch.Tensor:
        """Convert emotion dictionary to tensor"""
        return torch.tensor([
            emotion_dict[k] for k in sorted(emotion_dict.keys())
        ])