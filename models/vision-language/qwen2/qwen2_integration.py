import torch
from typing import Dict, Any, Optional, List
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Qwen2VLIntegration:
    """
    Integration for Qwen2-VL-7B-Instruct model.
    Handles loading, processing, and inference for vision-language tasks.
    Supports 4-bit quantization via bitsandbytes.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen2-VL-7B-Instruct")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = None
        self.model = None
        
        self._load_model()

    def _load_model(self):
        """Load the model and processor with quantization settings."""
        logger.info(f"Loading Qwen2-VL model: {self.model_name}")
        
        quantization_config = None
        if self.config.get("quantization", {}).get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            q_cfg = self.config["quantization"]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, q_cfg.get("bnb_4bit_compute_dtype", "float16"))
            )
            logger.info("4-bit quantization enabled.")

        try:
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                device_map="auto" if quantization_config else None,
                trust_remote_code=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            if not quantization_config:
                self.model.to(self.device)
                
            self.model.eval()
            logger.info("Qwen2-VL loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL: {e}")
            raise e

    def analyze_scene(self, image_input: Any, prompt: str = "Describe this scene in detail.") -> str:
        """
        Analyze an image (or list of images) with a text prompt.
        
        Args:
            image_input: PIL Image, path, or base64.
            prompt: Text prompt for the analysis.
            
        Returns:
            Generated text description.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)

        # Generate
        gen_config = self.config.get("generation", {})
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=gen_config.get("max_new_tokens", 128),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9)
            )
            
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def get_embeddings(self, image_input: Any) -> torch.Tensor:
        """
        Extract visual embeddings from the model's vision tower.
        Useful for storing in Vector Memory.
        """
        # Note: Qwen2-VL's architecture is complex. This extracts features 
        # from the vision tower before the projection layer if accessible,
        # or we might use the last hidden states of a dummy generation.
        
        # Simplified approach: Return last hidden state of the vision encoder
        # This requires digging into the model structure or doing a forward pass
        # without generation.
        raise NotImplementedError("Visual embeddings not yet implemented for Qwen2-VL")

