"""
Narrative Engine Module for ACM Project

Handles narrative reasoning and coherence using LLaMA 3.3 70B model.
Supports long-context processing, emotional integration, and multilingual narratives.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pinecone import Index


class NarrativeEngine:
    def __init__(self, memory_index_name="narrative_memory", max_context=128):
        """
        Initialize the Narrative Engine with LLaMA 3.3.
        Args:
            memory_index_name (str): Name of the Pinecone index for narrative memory.
            max_context (int): Maximum number of past narratives to keep in memory.
        """
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=True
        )
        self.memory_index = Index(memory_index_name)
        self.memory_context = []
        self.max_context = max_context

    def generate_narrative(self, input_text):
        """
        Generate a narrative response based on input and memory.
        Args:
            input_text (str): Input prompt or query.
        Returns:
            str: Generated narrative response.
        """
        # Retrieve relevant memory
        memory_context = self._retrieve_memory(input_text)
        combined_input = self._build_prompt(input_text, memory_context)

        # Generate response
        inputs = self.tokenizer(combined_input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                num_beam_groups=4,
                diversity_penalty=0.3
            )

        narrative = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Update memory context
        self._store_memory(narrative)
        return narrative

    def _build_prompt(self, input_text, memory_context):
        """
        Build a prompt with retrieved memory context.
        Args:
            input_text (str): Current input text.
            memory_context (list): Retrieved memory context.
        Returns:
            str: Combined prompt for the model.
        """
        prompt = "Let’s think step by step and maintain coherence:\n\n"
        prompt += input_text
        if memory_context:
            prompt += "\n\nRelevant past context:\n" + "\n".join(memory_context)
        return prompt

    def _store_memory(self, narrative):
        """
        Store the generated narrative in memory.
        Args:
            narrative (str): The generated narrative.
        """
        embedding = self.tokenizer(narrative, return_tensors="pt").input_ids.mean(dim=1).tolist()
        self.memory_index.upsert([(narrative[:50], embedding, {"text": narrative})])
        self.memory_context.append(narrative)
        if len(self.memory_context) > self.max_context:
            self.memory_context.pop(0)

    def _retrieve_memory(self, input_text):
        """
        Retrieve relevant memories for the input.
        Args:
            input_text (str): Current input text.
        Returns:
            list: List of relevant memory texts.
        """
        embedding = self.tokenizer(input_text, return_tensors="pt").input_ids.mean(dim=1).tolist()
        results = self.memory_index.query(embedding, top_k=5, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"]]


# Example Usage
if __name__ == "__main__":
    engine = NarrativeEngine()
    input_prompt = "Describe how an AI agent might respond to emotional stress."
    response = engine.generate_narrative(input_prompt)
    print(f"Generated Narrative: {response}")
