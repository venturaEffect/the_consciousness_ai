from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class NarrativeEngine:
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=True
        )
        self.memory_context = []

    def generate_narrative(self, input_text):
        prompt = self._build_prompt(input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
        narrative = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.memory_context.append(narrative)
        return narrative

    def _build_prompt(self, input_text):
        prompt = "Let's think step by step about the current situation and its implications:\n\n"
        prompt += input_text
        if self.memory_context:
            prompt += "\n\nPrevious context:\n" + "\n".join(self.memory_context[-5:])
        return prompt