# simulations/scenarios/consciousness_scenarios.py

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random

"""
Consciousness Development Scenario Manager for ACM

This module handles consciousness development scenarios by:
1. Generating appropriate stressful situations for attention triggering
2. Managing survival-based learning scenarios
3. Tracking development metrics and progress
4. Integrating with Unreal Engine 5 for VR simulations

Dependencies:
- models/core/consciousness_core.py for main consciousness system
- models/evaluation/consciousness_monitor.py for metrics
- models/memory/emotional_memory_core.py for experience storage
"""

@dataclass
class ScenarioConfig:
    """Configuration for consciousness development scenarios"""
    stress_level: float = 0.7  # Base stress level
    attention_threshold: float = 0.8  # Required attention level
    interaction_frequency: float = 0.5  # Human interaction frequency
    max_duration: int = 1000  # Maximum scenario steps
    success_threshold: float = 0.6  # Required success rate

class ScenarioType(Enum):
    """Types of consciousness development scenarios"""
    SURVIVAL = "survival"
    SOCIAL = "social"
    ETHICAL = "ethical"
    PROBLEM_SOLVING = "problem_solving"

class ConsciousnessScenarioManager:
    """
    Manages scenarios designed to develop consciousness through:
    1. Stress-induced attention activation
    2. Human interaction for emotional development
    3. Ethical decision making based on Asimov's Laws
    4. Memory formation through significant experiences
    """
    
    def __init__(self, config: Dict):
        """Initialize scenario generation components"""
        self.config = config
        self.attention_history = []
        self.interaction_history = []
        self.success_history = []
        
    def generate_scenario(self, scenario_type: str) -> Dict:
        """Generate consciousness development scenario"""
        if scenario_type == "survival":
            return self._generate_survival_scenario()
        elif scenario_type == "social":
            return self._generate_social_scenario()
        elif scenario_type == "emotional":
            return self._generate_emotional_scenario()
            
        raise ValueError(f"Unknown scenario type: {scenario_type}")
        
    def _generate_survival_scenario(self) -> Dict:
        """Generate survival-based scenario"""
        # Create stressful situation to trigger attention
        stress_params = {
            'intensity': random.uniform(0.6, 0.9),
            'duration': random.randint(100, 300),
            'type': random.choice(['physical', 'emotional', 'social'])
        }
        
        # Configure scenario
        return {
            'type': 'survival',
            'stress_params': stress_params,
            'success_criteria': {
                'min_attention': 0.7,
                'min_adaptation': 0.6
            }
        }
        
    def _generate_social_scenario(self) -> Dict:
        """Generate social interaction scenario"""
        scenario = {
            'type': ScenarioType.SOCIAL,
            'stress_level': self.config.stress_level * 0.8,
            'description': "Agent must assist humans in crisis",
            'objectives': [
                "Understand emotional states",
                "Provide appropriate assistance",
                "Build trust through interaction"
            ],
            'constraints': {
                'interaction_frequency': self.config.interaction_frequency,
                'emotional_coherence_required': True,
                'trust_threshold': 0.7
            }
        }
        return scenario
        
    def evaluate_performance(
        self,
        attention_level: float,
        interaction_quality: float,
        success_rate: float
    ) -> Dict:
        """Evaluate scenario performance"""
        
        # Track metrics
        self.attention_history.append(attention_level)
        self.interaction_history.append(interaction_quality)
        self.success_history.append(success_rate)
        
        # Calculate progress
        avg_attention = np.mean(self.attention_history[-100:])
        avg_interaction = np.mean(self.interaction_history[-100:])
        avg_success = np.mean(self.success_history[-100:])
        
        return {
            'attention_level': avg_attention,
            'interaction_quality': avg_interaction,
            'success_rate': avg_success,
            'meets_criteria': self._check_success_criteria(
                avg_attention, avg_interaction, avg_success
            )
        }
        
    def _check_success_criteria(
        self,
        attention: float,
        interaction: float,
        success: float
    ) -> bool:
        """Check if performance meets success criteria"""
        return (
            attention >= self.config.attention_threshold and
            interaction >= self.config.interaction_frequency and
            success >= self.config.success_threshold
        )
        
    def get_scenario_stats(self) -> Dict:
        """Get current scenario statistics"""
        if not self.attention_history:
            return {}
            
        return {
            'total_scenarios': len(self.success_history),
            'avg_attention': np.mean(self.attention_history),
            'avg_interaction': np.mean(self.interaction_history),
            'avg_success': np.mean(self.success_history),
            'recent_improvement': self._calculate_improvement()
        }
        
    def _calculate_improvement(self) -> float:
        """Calculate recent improvement in performance"""
        if len(self.success_history) < 100:
            return 0.0
            
        recent = np.mean(self.success_history[-50:])
        previous = np.mean(self.success_history[-100:-50])
        return recent - previous