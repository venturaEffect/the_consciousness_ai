# filepath: /c:/Users/zaesa/OneDrive/Escritorio/Artificial_counsciousness/the_consciousness_ai/models/memory/attention_schema.py
# models/memory/attention_schema.py

from typing import Dict, List
import numpy as np
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FocusEntry:
    timestamp: datetime
    visual_focus: Dict
    audio_focus: Dict 
    consciousness_state: Dict
    emotion_state: Dict
    importance: float

class AttentionSchema:
    def __init__(self, history_size: int = 1000):
        self.history: List[FocusEntry] = []
        self.history_size = history_size
        self.lock = asyncio.Lock()

    async def update(self, current_focus: Dict) -> None:
        """Update attention schema with new focus data"""
        async with self.lock:
            entry = FocusEntry(
                timestamp=datetime.now(),
                visual_focus=current_focus.get('visual', {}),
                audio_focus=current_focus.get('audio', {}),
                consciousness_state=current_focus.get('consciousness', {}),
                emotion_state=current_focus.get('emotion', {}),
                importance=self._calculate_importance(current_focus)
            )
            
            self.history.append(entry)
            if len(self.history) > self.history_size:
                self.history.pop(0)

    async def get_overview(self) -> Dict:
        """Get aggregated view of attention history"""
        async with self.lock:
            if not self.history:
                return {}
                
            recent = self.history[-10:]
            
            return {
                'current_focus': self._get_current_focus(),
                'attention_trends': self._analyze_trends(recent),
                'emotional_state': self._aggregate_emotions(recent),
                'consciousness_summary': self._summarize_consciousness(recent)
            }

    def _calculate_importance(self, focus_data: Dict) -> float:
        # Implement importance calculation based on emotional valence,
        # consciousness state activation, etc.
        pass

    def _get_current_focus(self) -> Dict:
        return self.history[-1] if self.history else {}

    def _analyze_trends(self, entries: List[FocusEntry]) -> Dict:
        # Analyze temporal patterns in attention
        pass

    def _aggregate_emotions(self, entries: List[FocusEntry]) -> Dict:
        # Aggregate emotional states over time
        pass

    def _summarize_consciousness(self, entries: List[FocusEntry]) -> Dict:
        # Summarize consciousness states
        pass