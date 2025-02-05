# python: models/cognitive/chain_of_thought.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChainOfThought:
    def __init__(self, memory, llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", num_recent: int = 10):
        """
        memory: Reference to the memory system.
        llm_model_name: Name of the LLM to use for generating chain-of-thought narratives.
        num_recent: Number of recent experiences to aggregate.
        """
        self.memory = memory
        self.num_recent = num_recent
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).to("cuda")
        self.system_prompt = (
            "Respond with a clear, structured chain-of-thought narrative. "
            "Introspect over the recent emotional experiences and identify patterns, strengths, and areas for self-model refinement."
        )
        self.chain_format = (
            "<reasoning>\n{reasoning}\n</reasoning>\n"
            "<answer>\n{answer}\n</answer>\n"
        )

    def aggregate_experiences(self) -> str:
        """
        Retrieves recent experiences from memory and aggregates them into a textual summary.
        """
        recent_experiences = self.memory.get_recent_experiences(limit=self.num_recent)
        if not recent_experiences:
            return "No recent experiences available."
        summaries = []
        for i, exp in enumerate(recent_experiences):
            # Assume each experience contains an 'emotion' dict.
            emotion = exp.get("emotion", {})
            summaries.append(
                f"Experience {i+1}: V:{emotion.get('valence', 0.0):.1f}, "
                f"A:{emotion.get('arousal', 0.1):.1f}, D:{emotion.get('dominance', 0.0):.1f}"
            )
        return "\n".join(summaries)

    def generate_chain(self) -> str:
        """
        Generates the chain-of-thought narrative by prompting the LLM with aggregated experiences.
        """
        aggregated = self.aggregate_experiences()
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Recent Experiences:\n{aggregated}\n\n"
            f"Generate chain-of-thought:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If the output already follows our XML format, return it. Otherwise, format it.
        if "<reasoning>" in generated_text and "<answer>" in generated_text:
            return generated_text.strip()
        else:
            # Very basic fallback: split the output into two parts.
            lines = generated_text.strip().splitlines()
            reasoning = lines[0] if lines else "No reasoning provided."
            answer = lines[-1] if len(lines) > 1 else "No answer provided."
            return self.chain_format.format(reasoning=reasoning, answer=answer).strip()