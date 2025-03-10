import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time

@dataclass
class WorkspaceMessage:
    source: str
    content: Any
    priority: float

@dataclass
class WorkspaceState:
    """Current state of the global workspace"""
    active_content: Dict[str, Any] = None
    access_history: List[Dict[str, Any]] = None
    broadcast_strength: float = 0.0
    competition_results: Dict[str, float] = None

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
            competition_results={}
        )
        self.specialist_modules = {}
        self.broadcast_threshold = config.get("broadcast_threshold", 0.7)
        self.max_history = config.get("max_history", 100)
        
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
        
        # Only broadcast if strength exceeds threshold
        if broadcast_strength >= self.broadcast_threshold:
            self.state.active_content = broadcast_content
            self.state.access_history.append({
                'content': broadcast_content,
                'strength': broadcast_strength,
                'winners': winners,
                'timestamp': time.time()
            })
            
            # Trim history if needed
            if len(self.state.access_history) > self.max_history:
                self.state.access_history = self.state.access_history[-self.max_history:]
                
            # Return broadcast content and competition results
            return broadcast_content, bids
        else:
            # No broadcast occurred
            return {}, bids
    
    def _resolve_competition(self, bids: Dict[str, float]) -> List[str]:
        """
        Determine which specialists win the competition for access.
        
        In basic implementation, highest bid wins, but this can be extended
        to account for coalitions, temporal dynamics, etc.
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
        
    def get_current_broadcast(self) -> Dict[str, Any]:
        """Get the currently broadcast content"""
        return self.state.active_content
        
    def get_competition_status(self) -> Dict[str, float]:
        """Get the results of the most recent competition"""
        return self.state.competition_results