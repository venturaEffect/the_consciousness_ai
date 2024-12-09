# models/language/long_context_integration.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LongContextIntegration:
    def __init__(self, model_name="mosaicml/mpt-7b-storywriter"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()

    def process_long_input(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=65536
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result