# models/vision-language/pali-2/pali2_integration.py
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torch

class PaLI2Integration:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl"):
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.eval()

    def generate_caption(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption