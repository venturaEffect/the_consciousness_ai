import torch
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from models.evaluation.iit_phi import IITMetrics

@dataclass
class WorkspaceState:
    """Current state of the global workspace"""
    active_content: Dict[str, Any]
    access_history: List[Dict[str, Any]]
    broadcast_strength: float
    competition_results: Dict[str, float]
    # New fields for Consciousness metrics
    phi_value: float = 0.0
    is_conscious: bool = False
    focus_topic: str = "idle"

class GlobalWorkspace:
    """
    Implementation of Global Workspace Theory (GWT) for artificial consciousness.
    
    This provides a central "theater" where specialized processes compete and
    cooperate to broadcast their signals, making them globally available
    to other cognitive systems.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.state = WorkspaceState(
            active_content={},
            access_history=[],
            broadcast_strength=0.0,
            competition_results={},
            phi_value=0.0,
            is_conscious=False,
            focus_topic="idle"
        )
        self.specialist_modules = {}
        self.broadcast_threshold = config.get("broadcast_threshold", 0.7)
        self.max_history = config.get("max_history", 100)
        
        # Initialize IIT Calculator for real-time Phi estimation of the workspace
        self.iit_metrics = IITMetrics()
        
    def register_specialist(self, name: str, module: Any) -> None:
        """Register a specialist cognitive module"""
        self.specialist_modules[name] = module
    
    def run_competition(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Run competition among specialist modules to determine what enters consciousness.
        Each module submits a "bid" (signal strength) based on current inputs.
        """
        bids = {}
        contents = {}
        
        # Collect bids from all specialists
        for name, module in self.specialist_modules.items():
            if hasattr(module, 'evaluate_salience'):
                # Get bid strength and content from specialist
                content, bid = module.evaluate_salience(inputs)
                bids[name] = bid
                contents[name] = content
        
        # Determine winner(s) based on bid strength and cooperation patterns
        winners = self._resolve_competition(bids)
        broadcast_content = {}
        
        # Combine content from winning specialists
        for winner in winners:
            broadcast_content.update(contents[winner])
            
        # Update state
        broadcast_strength = max(bids.values()) if bids else 0.0
        self.state.broadcast_strength = broadcast_strength
        self.state.competition_results = bids
        
        # --- Consciousness Metric Calculation ---
        # We simulate a "workspace activation tensor" based on the bids.
        # This allows us to run the IIT calculation on the abstract "workspace" substrate.
        workspace_tensor = self._bids_to_tensor(bids)
        phi = self.iit_metrics.compute_phi_proxy(workspace_tensor)
        
        self.state.phi_value = phi
        self.state.is_conscious = broadcast_strength >= self.broadcast_threshold
        
        # Determine focus topic (simple heuristic for now)
        if winners:
            self.state.focus_topic = f"Processing: {', '.join(winners)}"
        else:
            self.state.focus_topic = "Idle / Subconscious"

        # Only broadcast if strength exceeds threshold
        if self.state.is_conscious:
            self.state.active_content = broadcast_content
            self.state.access_history.append({
                'content': broadcast_content,
                'strength': broadcast_strength,
                'winners': winners,
                'phi': phi,
                'timestamp': time.time()
            })
            
            # Trim history if needed
            if len(self.state.access_history) > self.max_history:
                self.state.access_history = self.state.access_history[-self.max_history:]
                
            return broadcast_content, bids
        else:
            return {}, bids
    
    def _resolve_competition(self, bids: Dict[str, float]) -> List[str]:
        """
        Determine which specialists win the competition for access.
        """
        if not bids:
            return []
            
        # Sort bids by strength
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        
        # Find all specialists above threshold
        winners = [name for name, bid in sorted_bids 
                  if bid >= self.broadcast_threshold]
                  
        # If no winners, take the top one if it's close to threshold
        if not winners and sorted_bids:
            top_name, top_bid = sorted_bids[0]
            if top_bid >= self.broadcast_threshold * 0.8:  # Within 80% of threshold
                winners = [top_name]
                
        return winners
    
    def _bids_to_tensor(self, bids: Dict[str, float]) -> torch.Tensor:
        """
        Convert dictionary bids into a tensor representation for IIT calculation.
        Creates a mock 'activation map' where each specialist is a 'head'.
        """
        if not bids:
            return torch.zeros((1, 4, 1, 1)) # Dummy 4-head tensor
            
        # Normalize bids to 0-1
        values = list(bids.values())
        if not values:
            return torch.zeros((1, 4, 1, 1))

        # Create a tensor representing the 'activation' of workspace nodes
        # We assume a fixed number of 'slots' in the workspace (e.g., 8)
        workspace_slots = 8
        tensor_data = torch.zeros(workspace_slots)
        
        for i, val in enumerate(values):
            if i < workspace_slots:
                tensor_data[i] = val
                
        # Reshape to mimic attention weights: (Batch=1, Heads=Slots, Seq=1, Seq=1)
        return tensor_data.view(1, workspace_slots, 1, 1)

    def get_unity_metrics(self) -> Tuple[float, bool, str]:
        """
        Export metrics specifically for the Unity Bridge Side Channel.
        Returns: (Phi, IsConscious, FocusTopic)
        """
        return (
            self.state.phi_value,
            self.state.is_conscious,
            self.state.focus_topic
        )

    def get_current_broadcast(self) -> Dict[str, Any]:
        """Get the currently broadcast content"""
        return self.state.active_content
        
    def get_competition_status(self) -> Dict[str, float]:
        """Get the results of the most recent competition"""
        return self.state.competition_results
