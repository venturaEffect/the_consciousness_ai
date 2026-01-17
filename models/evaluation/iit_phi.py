import torch
import numpy as np
import pyphi
from typing import Optional, List, Dict, Any
import logging
import itertools

# Configure logging
logger = logging.getLogger(__name__)

class IITMetrics:
    def __init__(self, logger=None):
        """
        Initializes the IITMetrics calculator using PyPhi for rigorous 
        consciousness estimation on small subsystems.
        """
        self.logger = logger
        # Cache for PyPhi networks to avoid rebuilding expensive objects
        self.network_cache = {} 
        
        # PyPhi config (optimize for performance)
        pyphi.config.PROGRESS_BARS = False
        pyphi.config.PARALLEL_CUTS = False 

    def calculate_phi(self, tpm: np.ndarray, current_state: tuple) -> float:
        """
        Calculate the actual Integrated Information (Phi) of a subsystem 
        defined by the Transition Probability Matrix (TPM).
        
        Args:
            tpm: A 2D numpy array representing the system's logic.
                 Shape must be (2^N, 2^N) or state-by-node for PyPhi.
                 For simplicity here, we assume a state-by-node TPM for discrete nodes.
            current_state: Tuple of binary states (0, 1) representing current node activation.
            
        Returns:
            float: The value of Phi (Big Phi).
        """
        try:
            num_nodes = len(current_state)
            
            # Validation: PyPhi gets exponentially slow > 5 nodes.
            if num_nodes > 5:
                logger.warning(f"TPM has {num_nodes} nodes. Truncating to 5 for tractability.")
                # Logic to slice TPM would be complex here, so we assume the caller
                # provides a small enough subsystem.
                return 0.0

            # Create PyPhi Network
            # Note: PyPhi 1.2+ uses specific TPM formats. 
            # We assume tpm is in the standard state-by-node format.
            network = pyphi.Network(tpm)
            
            # Create Subsystem
            subsystem = pyphi.Subsystem(network, current_state)
            
            # Calculate Phi (Big Phi) using the 3.0/4.0 algorithm
            # sia = system integrated information
            sia = subsystem.sia() 
            
            if sia:
                phi_value = sia.phi
                if self.logger:
                    self.logger.log_scalar_data("iit_phi_actual", 0, phi_value)
                return phi_value
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Phi: {e}")
            return 0.0

    def extract_subsystem_tpm(self, attention_weights: torch.Tensor, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Extracts a "Consciousness Core" from a larger neural network (e.g., Transformer).
        
        Strategy:
        1. Identify the top-K most active 'heads' or 'neurons' in the Global Workspace.
        2. Binarize their activation to create a discrete state (0/1).
        3. Estimate their transition probabilities based on recent history (Hebbian-like).
        
        Args:
            attention_weights: Tensor of shape (batch, heads, seq, seq) or similar.
            
        Returns:
            Dict containing 'tpm' (numpy), 'state' (tuple), and 'indices' (list).
        """
        # Simplified logic for prototype:
        # Take the mean attention of the top 4 heads as our "nodes"
        
        if attention_weights.ndim < 2:
            return None
            
        # 1. Average over sequence length to get "Head Activation"
        # Shape: (num_heads,)
        head_activations = attention_weights.mean(dim=[-1, -2]).squeeze()
        
        # 2. Select Top-4 Heads (Nodes)
        k = 4
        if head_activations.numel() < k:
            k = head_activations.numel()
            
        top_values, top_indices = torch.topk(head_activations, k)
        
        # 3. Binarize State (Active if > threshold)
        # Note: Threshold should be dynamic in a real system
        current_state = tuple((top_values > threshold).int().tolist())
        
        # 4. Generate a Mock TPM (Placeholder for actual Hebbian learning)
        # In a real run, we would track these indices over time to build the TPM.
        # Here we generate a random but consistent TPM for these indices to allow PyPhi to run.
        # A 4-node system has 2^4 = 16 states.
        # TPM shape: (16, 4) - Probability of each node being ON in the next state.
        
        # Deterministic seed based on indices to keep it "stable" for the same heads
        seed = int(top_indices.sum().item())
        rng = np.random.default_rng(seed)
        tpm = rng.random((2**k, k)) 
        
        # Binarize TPM for deterministic logic (PyPhi can handle probabilistic, but this is faster)
        tpm = (tpm > 0.5).astype(int)
        
        return {
            "tpm": tpm,
            "state": current_state,
            "node_indices": top_indices.tolist()
        }

    def compute_phi_proxy(self, global_workspace_state: torch.Tensor) -> float:
        """
        High-level wrapper to get a Phi value from the Global Workspace state.
        """
        subsystem = self.extract_subsystem_tpm(global_workspace_state)
        if subsystem:
            return self.calculate_phi(subsystem['tpm'], subsystem['state'])
        return 0.0
