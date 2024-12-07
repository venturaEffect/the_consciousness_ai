import torch
from dreamerv3_torch import DreamerV3

class WorldModel:
    def __init__(self):
        self.model = DreamerV3(
            obs_shape=(3, 64, 64),
            action_shape=(8,),
            hidden_size=200
        )
        
    def predict_next_state(self, current_state, action):
        """Predict next simulation state based on current state and action"""
        with torch.no_grad():
            predicted_state = self.model.imagine(current_state, action)
        return predicted_state