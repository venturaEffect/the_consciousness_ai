# models/controller/simulation_controller.py

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.evaluation.emotional_evaluation import EmotionalEvaluator
from models.narrative.narrative_engine import NarrativeEngine
from simulations.scenarios.consciousness_scenarios import ConsciousnessScenarioManager
from simulations.api.simulation_manager import SimulationManager
from simulations.enviroments.pavilion_vr_environment import PavilionVREnvironment

@dataclass
class SimulationMetrics:
    """Tracks simulation and consciousness development metrics"""
    episode_count: int = 0
    total_reward: float = 0.0
    consciousness_score: float = 0.0
    emotional_coherence: float = 0.0
    attention_stability: float = 0.0
    learning_progress: float = 0.0

class ConsciousnessSimulationController:
    """
    Main controller for consciousness development simulations.
    Integrates emotional learning, attention mechanisms, and memory systems.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize core components
        self.dreamer = DreamerEmotionalWrapper(config)
        self.fusion = EmotionalMemoryFusion(config)
        self.evaluator = EmotionalEvaluator(config)
        self.narrative = NarrativeEngine()
        self.scenario_manager = ConsciousnessScenarioManager(config)
        
        # Metrics tracking
        self.metrics = SimulationMetrics()
        self.episode_history = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('consciousness_development.log'),
                logging.StreamHandler()
            ]
        )
        
    def run_episode(self, scenario_type: str) -> Dict:
        """Run a single consciousness development episode"""
        
        # Generate scenario
        scenario = self.scenario_manager.generate_scenario(scenario_type)
        
        # Track episode metrics
        episode_data = []
        total_reward = 0.0
        
        # Initial state
        state = self._get_initial_state(scenario)
        done = False
        step = 0
        
        while not done and step < self.config['max_steps']:
            # Process multimodal inputs
            fusion_output, fusion_info = self.fusion.forward(
                text_input=state.get('text'),
                vision_input=state.get('vision'),
                audio_input=state.get('audio'),
                emotional_context=state.get('emotion')
            )
            
            # Get action from policy
            action = self.dreamer.get_action(
                fusion_output,
                emotion_context=fusion_info['emotional_context']
            )
            
            # Execute action and get next state
            next_state, reward, done, info = self._execute_action(
                action, 
                scenario
            )
            
            # Evaluate emotional state
            evaluation = self.evaluator.evaluate_interaction(
                state=state,
                action=action,
                emotion_values=info['emotion'],
                attention_level=info['attention'],
                narrative=info.get('narrative', ''),
                stress_level=info.get('stress', 0.0)
            )
            
            # Generate narrative
            narrative = self.narrative.generate_experience_narrative(
                state=state,
                action=action,
                emotion=evaluation['emotional_context'],
                include_context=True
            )
            
            # Store experience
            self._store_experience(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                emotion=evaluation['emotional_context'],
                narrative=narrative,
                done=done
            )
            
            # Update metrics
            total_reward += reward
            episode_data.append({
                'step': step,
                'state': state,
                'action': action,
                'reward': reward,
                'emotion': evaluation['emotional_context'],
                'attention': info['attention'],
                'narrative': narrative
            })
            
            # Update state
            state = next_state
            step += 1
            
        # Update episode metrics
        self.metrics.episode_count += 1
        self.metrics.total_reward += total_reward
        
        # Calculate episode results
        results = self._calculate_episode_results(
            episode_data=episode_data,
            total_reward=total_reward,
            evaluation=evaluation
        )
        
        # Log progress
        self._log_episode_progress(results)
        
        return results
        
    def _get_initial_state(self, scenario: Dict) -> Dict:
        """Get initial state for scenario"""
        return {
            'text': scenario.get('description', ''),
            'vision': scenario.get('initial_observation'),
            'audio': scenario.get('audio_context'),
            'emotion': {
                'valence': 0.5,
                'arousal': 0.5,
                'dominance': 0.5
            }
        }
        
    def _execute_action(
        self,
        action: torch.Tensor,
        scenario: Dict
    ) -> Tuple[Dict, float, bool, Dict]:
        """Execute action in simulation"""
        # Implementation depends on specific simulation environment
        raise NotImplementedError
        
    def _store_experience(self, **kwargs):
        """Store experience in memory"""
        self.fusion.memory_core.store_experience(kwargs)
        
    def _calculate_episode_results(
        self,
        episode_data: List[Dict],
        total_reward: float,
        evaluation: Dict
    ) -> Dict:
        """Calculate episode results and metrics"""
        return {
            'total_reward': total_reward,
            'steps': len(episode_data),
            'consciousness_score': evaluation['consciousness_score'],
            'emotional_coherence': evaluation['emotional_awareness'],
            'attention_stability': evaluation['attention_stability'],
            'learning_progress': self._calculate_learning_progress(),
            'episode_data': episode_data
        }
        
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress"""
        if len(self.episode_history) < 2:
            return 0.0
            
        recent_rewards = [ep['total_reward'] for ep in self.episode_history[-10:]]
        previous_rewards = [ep['total_reward'] for ep in self.episode_history[-20:-10]]
        
        return float(np.mean(recent_rewards) - np.mean(previous_rewards))
        
    def _log_episode_progress(self, results: Dict):
        """Log episode progress"""
        msg = f"\nEpisode {self.metrics.episode_count} Results:\n"
        msg += f"Total Reward: {results['total_reward']:.3f}\n"
        msg += f"Consciousness Score: {results['consciousness_score']:.3f}\n"
        msg += f"Emotional Coherence: {results['emotional_coherence']:.3f}\n"
        msg += f"Attention Stability: {results['attention_stability']:.3f}\n"
        msg += f"Learning Progress: {results['learning_progress']:.3f}\n"
        
        logging.info(msg)