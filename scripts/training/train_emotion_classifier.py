import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn

class MultimodalEmotionModel(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", num_emotions=27):
        super().__init__()
        self.text_encoder = AutoModelForSequenceClassification.from_pretrained(text_model_name)
        self.vision_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50')
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        fusion_dim = 1024
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + 2048 + 64, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_emotions)
        )

    def forward(self, text_inputs, image_inputs, audio_inputs):
        text_features = self.text_encoder(**text_inputs).logits
        vision_features = self.vision_encoder(image_inputs)
        audio_features = self.audio_encoder(audio_inputs)
        
        # Fusion
        combined = torch.cat([text_features, vision_features, audio_features], dim=1)
        return self.fusion_layer(combined)