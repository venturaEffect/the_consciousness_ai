from models.integration.video_llama3_integration import VideoLLaMA3Integration
from models.language.llama3_processor import Llama3Processor
from models.audio.whisper_processor import WhisperProcessor
import torch
import torch.nn as nn
from typing import Dict, Any

class MultimodalEmotionDetector:
    def __init__(self, config: Dict):
        self.video_llama = VideoLLaMA3Integration(config['video_llama3'])
        self.llama = Llama3Processor(config['llama3'])
        self.whisper = WhisperProcessor(config['whisper'])
        self.fusion_layer = nn.Linear(1024 + 768 + 512, 512)
    
    def process_inputs(
        self,
        visual_input: torch.Tensor,
        audio_input: torch.Tensor,
        text_input: str
    ) -> Dict[str, Any]:
        visual_context = self.video_llama.process_stream_frame(visual_input)
        audio_text = self.whisper.transcribe(audio_input)
        audio_context = self.whisper.process_audio(audio_input)
        text_embedding = self.llama.process_text(text_input + " " + audio_text)
        
        fused = self.fusion_layer(torch.cat([
            visual_context['embedding'],
            text_embedding,
            audio_context
        ], dim=-1))
        
        return self._classify_emotions(fused)