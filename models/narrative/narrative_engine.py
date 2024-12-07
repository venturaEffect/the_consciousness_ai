from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class NarrativeEngine:
    def __init__(self, model_name="anthropic/claude-2-100k"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.memory_context = []
        
    def generate_narrative(self, current_state, emotional_context):
        # Combine current state with historical context
        context = self._build_context(current_state, emotional_context)
        
        # Generate coherent narrative
        inputs = self.tokenizer(context, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=1000,
            temperature=0.7,
            do_sample=True
        )
        
        narrative = self.tokenizer.decode(outputs[0])
        self.memory_context.append(narrative)
        return narrative
        
    def _build_context(self, current_state, emotional_context):
        """Build context string from current state and emotional context"""
        # Convert state and context to string format
        state_str = " ".join([f"{k}: {v}" for k, v in current_state.items()])
        emotion_str = " ".join([f"{k}: {v}" for k, v in emotional_context.items()])
        
        # Combine with historical context
        context = f"Current State: {state_str}\nEmotional Context: {emotion_str}\n"
        if self.memory_context:
            context += f"\nPrevious Narratives:\n" + "\n".join(self.memory_context[-5:])
            
        return context