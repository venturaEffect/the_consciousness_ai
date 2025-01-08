# models/fusion/emotional_memory_fusion.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.generative.generative_emotional_core import GenerativeEmotionalCore

@dataclass
class FusionConfig:
    """Configuration for multimodal fusion"""
    text_model: str = "llama-3.3"
    vision_model: str = "palm-e"
    audio_model: str = "whisper-v3"
    fusion_hidden_size: int = 768
    num_fusion_layers: int = 3
    dropout: float = 0.1
    emotional_weight: float = 0.8

class EmotionalMemoryFusion(nn.Module):
    """
    Fuses multimodal inputs with emotional context for memory formation
    
    Key Features:
    1. Multimodal input processing (text, vision, audio)
    2. Emotional context integration
    3. Memory-guided fusion
    4. Generative emotional output
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Initialize core components
        self.emotion_network = EmotionalGraphNetwork()
        self.memory_core = EmotionalMemoryCore(config)
        self.generative_core = GenerativeEmotionalCore(config)
        
        # Multimodal encoders
        self.text_encoder = AutoModel.from_pretrained(config.text_model)
        self.vision_encoder = AutoModel.from_pretrained(config.vision_model)
        self.audio_encoder = AutoModel.from_pretrained(config.audio_model)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.fusion_hidden_size,
                nhead=8,
                dropout=config.dropout
            ) for _ in range(config.num_fusion_layers)
        ])
        
        # Output projections
        self.emotional_projection = nn.Linear(
            config.fusion_hidden_size,
            config.fusion_hidden_size
        )
        
    def forward(
        self,
        text_input: Optional[torch.Tensor] = None,
        vision_input: Optional[torch.Tensor] = None,
        audio_input: Optional[torch.Tensor] = None,
        emotional_context: Optional[Dict[str, float]] = None,
        memory_context: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process multimodal inputs with emotional and memory context
        """
        # Get modality embeddings
        embeddings = []
        
        if text_input is not None:
            text_embedding = self.text_encoder(text_input).last_hidden_state
            embeddings.append(text_embedding)
            
        if vision_input is not None:
            vision_embedding = self.vision_encoder(vision_input).last_hidden_state
            embeddings.append(vision_embedding)
            
        if audio_input is not None:
            audio_embedding = self.audio_encoder(audio_input).last_hidden_state
            embeddings.append(audio_embedding)
            
        # Get emotional embedding if context provided
        if emotional_context is not None:
            emotional_embedding = self.emotion_network.get_embedding(
                emotional_context
            )
            embeddings.append(emotional_embedding)
            
        # Combine embeddings
        if len(embeddings) == 0:
            raise ValueError("No inputs provided")
            
        combined = torch.cat(embeddings, dim=1)
        
        # Apply fusion layers
        fused = combined
        for layer in self.fusion_layers:
            fused = layer(fused)
            
        # Get memory context if provided
        if memory_context is not None:
            memory_embedding = self.memory_core.get_memory_embedding(
                memory_context
            )
            # Add memory context through attention
            fused = self._apply_memory_attention(fused, memory_embedding)
            
        # Project to emotional space
        emotional_output = self.emotional_projection(fused)
        
        # Generate response using fused representation
        response = self.generative_core.generate_response(
            emotional_output,
            emotional_context=emotional_context
        )
        
        return emotional_output, {
            'response': response,
            'emotional_context': emotional_context,
            'fusion_quality': self._calculate_fusion_quality(embeddings)
        }
        
    def _apply_memory_attention(
        self,
        fused: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention between fused representation and memory"""
        attention = torch.matmul(fused, memory.transpose(-2, -1))
        attention = torch.softmax(attention, dim=-1)
        return torch.matmul(attention, memory)
        
    def _calculate_fusion_quality(
        self,
        embeddings: List[torch.Tensor]
    ) -> float:
        """Calculate quality of multimodal fusion"""
        if len(embeddings) < 2:
            return 1.0
            
        # Calculate average cosine similarity between embeddings
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = torch.cosine_similarity(
                    embeddings[i].mean(dim=1),
                    embeddings[j].mean(dim=1)
                ).mean()
                similarities.append(sim)
                
        return float(torch.mean(torch.stack(similarities)).item())