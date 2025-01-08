# models/generative/generative_emotional_core.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import LlamaTokenizer, LlamaForCausalLM
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.emotional_memory_core import EmotionalMemoryCore

@dataclass
class GenerativeConfig:
    """Configuration for generative emotional processing"""
    model_name: str = "llama-3.3"
    max_length: int = 1024
    temperature: float = 0.7
    emotional_weight: float = 0.8
    memory_weight: float = 0.6
    top_k_memories: int = 5

class GenerativeEmotionalCore:
    """
    Integrates generative AI with emotional memory for consciousness development
    
    Key Features:
    1. Emotional memory-conditioned generation
    2. Experience-based narrative creation
    3. Emotional context preservation
    4. Memory-guided response generation
    """
    
    def __init__(self, config: GenerativeConfig):
        self.config = config
        
        # Initialize core components
        self.tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
        self.model = LlamaForCausalLM.from_pretrained(config.model_name)
        self.emotion_network = EmotionalGraphNetwork()
        self.memory_core = EmotionalMemoryCore(config)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate_response(
        self,
        prompt: str,
        emotional_context: Dict[str, float],
        situation_context: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """Generate emotionally-aware response"""
        
        # Retrieve relevant emotional memories
        relevant_memories = self.memory_core.retrieve_similar_memories(
            emotion_query=emotional_context,
            k=self.config.top_k_memories
        )
        
        # Create emotional embedding
        emotional_embedding = self.emotion_network.get_embedding(emotional_context)
        
        # Prepare context with emotional memories
        context = self._prepare_generation_context(
            prompt=prompt,
            emotional_embedding=emotional_embedding,
            memories=relevant_memories,
            situation=situation_context
        )
        
        # Generate response
        generated_ids = self.model.generate(
            input_ids=context["input_ids"].to(self.device),
            attention_mask=context["attention_mask"].to(self.device),
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Update emotional memory
        self._store_interaction_memory(
            prompt=prompt,
            response=response,
            emotional_context=emotional_context,
            situation_context=situation_context
        )
        
        return response, self._get_generation_metadata(
            context=context,
            response=response,
            emotional_context=emotional_context
        )
        
    def _prepare_generation_context(
        self,
        prompt: str,
        emotional_embedding: torch.Tensor,
        memories: List[Dict],
        situation: Optional[Dict]
    ) -> Dict:
        """Prepare context for generation with emotional conditioning"""
        
        # Create memory context string
        memory_context = self._format_memory_context(memories)
        
        # Create emotional prefix
        emotional_prefix = self._create_emotional_prefix(emotional_embedding)
        
        # Combine context elements
        full_context = f"{emotional_prefix}\n{memory_context}\nCurrent situation: {situation}\n\nPrompt: {prompt}\nResponse:"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_context,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return tokenized
        
    def _format_memory_context(self, memories: List[Dict]) -> str:
        """Format memories into context string"""
        context_parts = []
        
        for memory in memories:
            context_parts.append(
                f"Previous experience ({memory['emotion_values']['valence']:.2f} valence): {memory['narrative']}"
            )
            
        return "\n".join(context_parts)
        
    def _create_emotional_prefix(self, emotional_embedding: torch.Tensor) -> str:
        """Create emotional conditioning prefix"""
        # Project emotional embedding to text space
        emotional_projection = self.model.get_input_embeddings()(
            emotional_embedding.unsqueeze(0)
        )
        
        # Generate emotional context tokens
        emotional_tokens = self.model.generate(
            inputs_embeds=emotional_projection,
            max_length=50,
            temperature=0.5,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(emotional_tokens[0], skip_special_tokens=True)
        
    def _store_interaction_memory(
        self,
        prompt: str,
        response: str,
        emotional_context: Dict[str, float],
        situation_context: Optional[Dict]
    ):
        """Store interaction in emotional memory"""
        self.memory_core.store_experience({
            'prompt': prompt,
            'response': response,
            'emotion_values': emotional_context,
            'context': situation_context,
            'timestamp': np.datetime64('now')
        })
        
    def _get_generation_metadata(
        self,
        context: Dict,
        response: str,
        emotional_context: Dict[str, float]
    ) -> Dict:
        """Get metadata about the generation process"""
        return {
            'context_length': len(context['input_ids'][0]),
            'response_length': len(response.split()),
            'emotional_context': emotional_context,
            'generation_timestamp': np.datetime64('now')
        }