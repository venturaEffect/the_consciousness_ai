# models/speech/whisper_integration.py
import whisper

class WhisperIntegration:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]