class EmotionContextTracker:
    def __init__(self):
        self.emotion_history = []

    def update_emotion(self, emotion, intensity):
        self.emotion_history.append({"emotion": emotion, "intensity": intensity})
        if len(self.emotion_history) > 100:  # Limit history size
            self.emotion_history.pop(0)

    def get_recent_emotions(self):
        return self.emotion_history[-10:]
