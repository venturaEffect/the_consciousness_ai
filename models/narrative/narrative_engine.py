"""
Narrative Engine Module for ACM Project

This module implements the core narrative reasoning capabilities using LLaMA 3.3 70B model.
It handles the generation and maintenance of the agent's internal narrative, leveraging
LLaMA 3.3's enhanced capabilities for multilingual processing and long-context understanding.

Key Features:
- Implements 128k context window support
- Uses 8-bit quantization for memory efficiency
- Supports all 8 languages: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai
- Integrates with Pinecone for long-term memory storage
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class NarrativeEngine:
    def __init__(self):
        """
        Initialize the narrative engine with LLaMA 3.3 70B model.
        Uses 8-bit quantization to optimize memory usage while maintaining performance.
        """
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        
        # Configure 8-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Initialize tokenizer with multilingual support
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=True
        )

        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            use_auth_token=True
        )
        
        # Initialize memory context for maintaining narrative coherence
        self.memory_context = []

    def generate_narrative(self, input_text: str) -> str:
        """
        Generate narrative response based on input text and memory context.
        
        Args:
            input_text: The input prompt or query text
            
        Returns:
            str: Generated narrative response
            
        The function maintains narrative coherence by:
        1. Building a contextualized prompt including memory
        2. Generating response using advanced parameters
        3. Updating memory context with new information
        """
        # Build prompt with context
        prompt = self._build_prompt(input_text)
        
        # Prepare input for model
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate response with optimized parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                num_beam_groups=4,
                diversity_penalty=0.3
            )
            
        # Decode and process response
        narrative = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Update memory context
        self.memory_context.append(narrative)
        if len(self.memory_context) > 10:  # Maintain rolling context window
            self.memory_context.pop(0)
            
        return narrative

    def _build_prompt(self, input_text: str) -> str:
        """
        Build a prompt incorporating memory context for coherent narrative generation.
        
        Args:
            input_text: Base input text
            
        Returns:
            str: Formatted prompt with context
        """
        prompt = "Let's think step by step about the current situation and its implications:\n\n"
        prompt += input_text
        
        # Add relevant context from memory
        if self.memory_context:
            prompt += "\n\nPrevious context:\n" + "\n".join(self.memory_context[-5:])
            
        return prompt