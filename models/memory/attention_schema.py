# filepath: /c:/Users/zaesa/OneDrive/Escritorio/Artificial_counsciousness/the_consciousness_ai/models/memory/attention_schema.py
class AttentionSchema:
    def __init__(self):
        self.focus_history = []

    def update(self, current_focus: dict):
        """
        Capture the current focus and intention data.
        """
        self.focus_history.append(current_focus)

    def get_overview(self):
        """
        Returns an aggregated view of focus history,
        supporting introspection and meta-awareness.
        For example, this can be a processed trend or a simple list.
        """
        return self.focus_history