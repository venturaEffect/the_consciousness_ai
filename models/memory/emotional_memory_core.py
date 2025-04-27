"""
Emotional memory system implementing:
- Storage for RL experiences (state, action, reward, emotion, next_state).
- Persistence across simulations (loading/saving).
- Contextual retrieval using placeholders for vector similarity and emotional relevance.
- Support for associating experiences with model adaptations (e.g., LoRAs).
- Meta-memory concepts (placeholder).
"""

import logging
import time
import os
import pickle # Using pickle for simplicity, consider safer alternatives (JSON lines, DB) for production
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
from .memory_interface import MemoryInterface, QueryContext, RetrievedMemory, MemoryData

# Placeholder for a vector embedding function/model
# In reality, this would likely be a class instance or function call
# E.g., from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
def get_embedding(text: str) -> Optional[List[float]]:
    """Placeholder function to generate text embeddings."""
    if not text:
        return None
    # Replace with actual embedding generation
    logging.debug("Generating placeholder embedding.")
    # Simple example: return fixed-size list based on text length
    size = 384 # Example dimension
    embedding = [(hash(text) + i) % 100 / 100.0 for i in range(size)]
    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Placeholder for cosine similarity calculation."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    # Simplified dot product / magnitude calculation
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    mag1 = sum(v**2 for v in vec1) ** 0.5
    mag2 = sum(v**2 for v in vec2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

class EmotionalMemoryCore(MemoryInterface):
    """
    Stores experiences, facilitates RL, prediction, and evolutionary adaptation.
    Includes persistence and placeholders for advanced retrieval.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_memory_size = config.get('max_memory_size', 10000)
        self.persistence_path = config.get('persistence_path', 'memory_state.pkl')
        self.memory_storage: deque[Tuple[float, MemoryData]] = deque(maxlen=self.max_memory_size)
        self._load_memory() # Load previous state if available
        logging.info(f"EmotionalMemoryCore initialized. Loaded {len(self.memory_storage)} memories from {self.persistence_path if os.path.exists(self.persistence_path) else 'new state'}. Max size: {self.max_memory_size}.")
        # TODO: Initialize vector database/index if using one.

    def _load_memory(self):
        """Loads memory state from the persistence path."""
        if self.persistence_path and os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, deque):
                        # Ensure loaded data respects the current max_memory_size
                        if loaded_data.maxlen != self.max_memory_size:
                             logging.warning(f"Loaded memory maxlen ({loaded_data.maxlen}) differs from config ({self.max_memory_size}). Adjusting.")
                             # Create new deque with correct maxlen from loaded data
                             self.memory_storage = deque(loaded_data, maxlen=self.max_memory_size)
                        else:
                             self.memory_storage = loaded_data
                        logging.info(f"Successfully loaded {len(self.memory_storage)} memories.")
                    else:
                        logging.error(f"Failed to load memory: Expected deque, got {type(loaded_data)}")
            except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
                logging.error(f"Error loading memory state from {self.persistence_path}: {e}", exc_info=True)
                # Start with an empty memory if loading fails
                self.memory_storage = deque(maxlen=self.max_memory_size)
        else:
            logging.info("No existing memory state found or persistence path not set. Starting fresh.")
            self.memory_storage = deque(maxlen=self.max_memory_size)

    def save_memory(self):
        """Saves the current memory state to the persistence path."""
        if self.persistence_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
                with open(self.persistence_path, 'wb') as f:
                    pickle.dump(self.memory_storage, f)
                logging.info(f"Successfully saved {len(self.memory_storage)} memories to {self.persistence_path}")
            except Exception as e:
                logging.error(f"Error saving memory state to {self.persistence_path}: {e}", exc_info=True)
        else:
            logging.warning("Persistence path not set. Memory not saved.")

    def store(self, timestamp: float, data: MemoryData):
        """
        Stores the experience data, ensuring essential fields for RL and context.
        Generates and stores text embeddings for relevant fields.
        """
        if not isinstance(data, dict):
             logging.error(f"Memory store failed: Data must be a dictionary, got {type(data)}")
             return

        # --- Enrich data for storage ---
        # Ensure essential RL fields exist (even if None)
        data.setdefault('state_summary', None) # Text summary of the state/perception
        data.setdefault('action', None)
        data.setdefault('reward', None) # Combined reward (env + emotional)
        data.setdefault('next_state_summary', None)
        data.setdefault('emotional_state', None)
        data.setdefault('active_goal', None) # Store goal context if available
        data.setdefault('active_lora_id', None) # Store which model adaptation was used

        # Generate and store embeddings (example for state summary)
        # TODO: Decide which fields need embeddings (perception, goal description, etc.)
        state_text = data.get('state_summary', '')
        data['state_embedding'] = get_embedding(state_text) if state_text else None

        logging.debug(f"Storing memory at timestamp {timestamp} with keys: {list(data.keys())}")
        self.memory_storage.append((timestamp, data))

        # Optional: Trigger periodic saving instead of saving on every store
        # if len(self.memory_storage) % 100 == 0: # Save every 100 memories
        #     self.save_memory()

    def retrieve(self, query_context: QueryContext, top_k: int = 5) -> List[RetrievedMemory]:
        """
        Retrieves relevant memories using context (recency, semantic similarity, emotion).
        Designed for providing context to the agent's "thought" process, prediction.
        """
        logging.debug(f"Retrieving memories (top_k={top_k}) with context: {query_context}")
        if not self.memory_storage: return []

        # --- Prepare Query Context ---
        query_text = query_context.get("perception", {}).get("summary_text", "") # Example query text
        query_embedding = get_embedding(query_text) if query_text else None
        query_emotion = query_context.get("emotion", {}) # Current emotional state

        # --- Scoring Logic ---
        scored_memories = []
        current_time = time.time()

        # Iterate through a larger pool of recent memories for scoring
        # Consider iterating through all if performance allows or using indexing
        search_pool_size = min(len(self.memory_storage), max(top_k * 10, 100))
        search_pool = list(self.memory_storage)[-search_pool_size:]

        for timestamp, data in reversed(search_pool): # Iterate newest first
            score = 0.0
            weight = 1.0 # Placeholder for meta-memory weighting

            # 1. Recency Score
            time_delta = current_time - timestamp
            recency_score = 1.0 / (1 + time_delta / 300.0) # Example decay over 5 minutes
            score += recency_score * 0.5 # Weight recency

            # 2. Semantic Similarity Score (using embeddings)
            memory_embedding = data.get('state_embedding')
            if query_embedding and memory_embedding:
                similarity = cosine_similarity(query_embedding, memory_embedding)
                score += similarity * 1.0 # Weight similarity

            # 3. Emotional Similarity Score
            memory_emotion = data.get('emotional_state')
            if query_emotion and memory_emotion and isinstance(memory_emotion, dict):
                # Example: Simple distance between dominant emotions or vector distance
                # This requires a defined structure for EmotionalState
                emotion_distance = sum(abs(query_emotion.get(k, 0) - memory_emotion.get(k, 0)) for k in query_emotion)
                emotion_similarity = 1.0 / (1.0 + emotion_distance) # Inverse distance
                score += emotion_similarity * 0.3 # Weight emotion

            # 4. Goal Relevance Score (Placeholder)
            # TODO: Compare query_context.get('active_goal') with data.get('active_goal')

            # 5. Apply Meta-Memory Weight (Placeholder)
            # TODO: Retrieve or calculate weight based on past usefulness or outcome
            # weight = self._get_meta_weight(data)
            final_score = score * weight

            scored_memories.append((final_score, timestamp, data))

        # Sort by score (highest first)
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Return top_k memories (just the data part)
        retrieved: List[RetrievedMemory] = [mem_data for score, ts, mem_data in scored_memories[:top_k]]

        logging.debug(f"Retrieved {len(retrieved)} memories based on context.")
        return retrieved

    def retrieve_batch_for_rl(self, batch_size: int) -> List[Tuple[float, MemoryData]]:
        """
        Retrieves a batch of recent experiences, typically used for RL updates.
        Returns tuples of (timestamp, data).
        """
        if batch_size > len(self.memory_storage):
            logging.warning(f"Requested RL batch size ({batch_size}) larger than memory size ({len(self.memory_storage)}). Returning all.")
            batch_size = len(self.memory_storage)

        if batch_size == 0:
            return []

        # Simple random sampling of recent experiences
        # More sophisticated sampling (e.g., Prioritized Experience Replay) can be added
        indices = np.random.choice(len(self.memory_storage), batch_size, replace=False)
        batch = [self.memory_storage[i] for i in indices]

        # Alternative: Just return the most recent batch_size items
        # batch = list(self.memory_storage)[-batch_size:]

        logging.debug(f"Retrieved batch of {len(batch)} memories for RL.")
        return batch

    # --- Optional methods ---
    # def _get_meta_weight(self, data: MemoryData) -> float:
    #     """Placeholder for calculating meta-memory weight."""
    #     # TODO: Implement logic based on reinforcement, surprise, outcome, etc.
    #     return 1.0

    # def check_coherence(self) -> float: ...
    # def query_survival_rate(self, last_n: int) -> float: ...

    def __del__(self):
        """Ensure memory is saved when the object is destroyed (e.g., program exit)."""
        logging.info("EmotionalMemoryCore shutting down. Saving memory...")
        self.save_memory()

# --- Need to import numpy for random sampling in retrieve_batch_for_rl ---
import numpy as np
