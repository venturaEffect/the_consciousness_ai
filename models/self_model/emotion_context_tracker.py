class EmotionContextTracker:
    """
    Tracks recent emotional states and provides convenient methods
    for querying and extracting emotional values.
    """

    def __init__(self):
        self.emotion_history = []
        self._current_emotion = {}  # Stores the most recently updated emotion.

    def update_emotion(self, emotion: str, intensity: float) -> None:
        """
        Update the current emotional context with a single named emotion
        and its intensity. Also keeps a running history up to 100 entries.

        Args:
            emotion: Name/key of the emotion (e.g., 'valence' or 'joy').
            intensity: Numeric intensity of this emotion.
        """
        self._current_emotion = {emotion: intensity}
        self.emotion_history.append(self._current_emotion)
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)

    def get_recent_emotions(self):
        """
        Return the last 10 emotional entries from the history.
        """
        return self.emotion_history[-10:]

    @property
    def current_emotion(self) -> dict:
        """
        Return the most recently updated emotional context as a dictionary.
        """
        return self._current_emotion

    def get_emotional_value(self, emotion_values: dict) -> float:
        """
        Extract a scalar measure (e.g., valence) from a dictionary
        of emotion signals.

        Args:
            emotion_values: Dictionary of emotional signals (e.g., {'valence': 0.8, ...}).

        Returns:
            A float representing one dimension of the emotion, e.g. valence.
            Defaults to 0.0 if that dimension is not found.
        """
        return emotion_values.get('valence', 0.0)
