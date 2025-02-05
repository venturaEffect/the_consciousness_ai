# python: models/cognitive/chain_of_thought.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.generative.imagination_generator import generate_imagery  # new module

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
            "Introspect over the recent emotional experiences and identify patterns, strengths, and areas for self-improvement. "
            "Also, describe an imagined visual scenario (image or video frame) that illustrates these insights."
        )
        self.chain_format = (
            "<reasoning>\n{reasoning}\n</reasoning>\n"
            "<narrative>\n{answer}\n</narrative>\n"
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
            emotion = exp.get("emotion", {})
            summaries.append(
                f"Experience {i+1}: V:{emotion.get('valence', 0.0):.1f}, "
                f"A:{emotion.get('arousal', 0.0):.1f}, D:{emotion.get('dominance', 0.0):.1f}"
            )
        return "\n".join(summaries)

    def generate_chain(self) -> str:
        """
        Generates the chain-of-thought narrative by prompting the LLM with aggregated experiences.
        The output includes a structured reasoning section and a narrative that includes imagined visual details.
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
        
        # Basic parsing: if the output doesn't include our expected tags, wrap the text.
        if "<reasoning>" in generated_text and "<narrative>" in generated_text:
            chain_output = generated_text.strip()
        else:
            lines = generated_text.strip().splitlines()
            reasoning = lines[0] if lines else "No reasoning provided."
            answer = lines[-1] if len(lines) > 1 else "No narrative provided."
            chain_output = self.chain_format.format(reasoning=reasoning, answer=answer).strip()
        return chain_output

    def generate_multimodal_thought(self) -> dict:
        """
        Uses the chain-of-thought narrative to generate additional multimodal (image/video) outputs.
        Returns a dictionary with text, and paths/URLs for generated image or video content.
        """
        chain_text = self.generate_chain()
        # Call the imagination generator module to produce an image or video based on the chain text.
        visual_output = generate_imagery(chain_text)
        return {
            "chain_text": chain_text,
            "visual_output": visual_output
        }