import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import pinecone

from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.evaluation.consciousness_metrics import ConsciousnessMetrics


@dataclass
class MemoryIndexConfig:
    """Configuration for emotional memory indexing."""
    vector_dimension: int = 768
    index_name: str = "emotional-memories"
    metric: str = "cosine"
    pod_type: str = "p1.x1"
    embedding_batch_size: int = 32


class EmotionalMemoryIndex:
    """
    Indexes and retrieves emotional memories using vector similarity.

    Key Features:
    1. Emotional context embedding
    2. Fast similarity search
    3. Temporal coherence tracking
    4. Consciousness-relevant retrieval
    """

    def __init__(self, config: MemoryIndexConfig):
        """
        Initialize the emotional memory index.

        Args:
            config: MemoryIndexConfig containing index parameters.
        """
        self.config = config
        self.emotion_network = EmotionalGraphNetwork()
        # If your ConsciousnessMetrics requires a config, pass it here. Otherwise, leave empty.
        self.consciousness_metrics = ConsciousnessMetrics({})

        # Initialize Pinecone index
        self._init_vector_store()

        # Simple counters and stats
        self.total_memories = 0
        # Minimal placeholder for memory statistics
        self.memory_stats = {
            "emotional_coherence": 0.0,
            "temporal_consistency": 0.0,
            "consciousness_relevance": 0.0
        }

    def _init_vector_store(self) -> None:
        """Initialize Pinecone vector store, creating the index if it doesn't exist."""
        # Ensure pinecone is initialized externally (e.g., pinecone.init(api_key=..., etc.)
        if self.config.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.config.index_name,
                dimension=self.config.vector_dimension,
                metric=self.config.metric,
                pod_type=self.config.pod_type
            )
        self.index = pinecone.Index(self.config.index_name)

    def store_memory(
        self,
        state: torch.Tensor,
        emotion_values: Dict[str, float],
        attention_level: float,
        narrative: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Store emotional memory with indexed metadata.

        Args:
            state: Tensor representing state or environment info.
            emotion_values: Dict of emotional signals (e.g., valence, arousal).
            attention_level: Numeric indicator of attention/consciousness.
            narrative: Text describing the experience.
            context: Optional dict for extra metadata (e.g., timestamps).

        Returns:
            A string memory ID.
        """
        # Generate emotional embedding.
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)

        # Calculate consciousness relevance (placeholder).
        # The test code calls `consciousness_metrics.evaluate_emotional_awareness([...])`,
        # so we replicate that here.
        awareness_result = self.consciousness_metrics.evaluate_emotional_awareness([
            {
                "state": state,
                "emotion": emotion_values,
                "attention": attention_level,
                "narrative": narrative
            }
        ])
        consciousness_score = awareness_result.get("mean_emotional_awareness", 0.0)

        # Prepare vector and metadata.
        vector = emotional_embedding.cpu().numpy()
        memory_id = f"memory_{self.total_memories}"
        metadata = {
            "emotion_values": emotion_values,
            "attention_level": float(attention_level),
            "narrative": narrative,
            "consciousness_score": float(consciousness_score),
            "timestamp": context["timestamp"] if context and "timestamp" in context else 0.0
        }

        # Upsert into Pinecone.
        self.index.upsert(
            vectors=[(memory_id, vector, metadata)],
            namespace="emotional_memories"
        )

        self.total_memories += 1
        self._update_memory_stats(consciousness_score)
        return memory_id

    def retrieve_similar_memories(
        self,
        emotion_query: Dict[str, float],
        k: int = 5,
        min_consciousness_score: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve similar memories based on emotional context.

        Args:
            emotion_query: Dict of emotion signals to build the query vector.
            k: Number of results to return after filtering.
            min_consciousness_score: Minimum consciousness score to be included.

        Returns:
            A list of memory dicts with keys: id, emotion_values, attention_level, narrative,
            consciousness_score, and similarity.
        """
        query_embedding = self.emotion_network.get_embedding(emotion_query)
        results = self.index.query(
            vector=query_embedding.cpu().numpy(),
            top_k=k * 2,  # Over-fetch to allow filtering
            namespace="emotional_memories",
            include_metadata=True
        )

        memories = []
        for match in results.matches:
            c_score = match.metadata["consciousness_score"]
            if c_score >= min_consciousness_score:
                memories.append({
                    "id": match.id,
                    "emotion_values": match.metadata["emotion_values"],
                    "attention_level": match.metadata["attention_level"],
                    "narrative": match.metadata["narrative"],
                    "consciousness_score": c_score,
                    "similarity": match.score
                })

        # Sort by combined similarity + consciousness_score.
        memories.sort(
            key=lambda x: (x["similarity"] + x["consciousness_score"]) / 2.0,
            reverse=True
        )
        return memories[:k]

    def get_temporal_sequence(
        self,
        start_time: float,
        end_time: float,
        min_consciousness_score: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve memories within a given time window, also filtering by consciousness_score.

        Args:
            start_time: Start of time window.
            end_time: End of time window.
            min_consciousness_score: Filter out memories below this threshold.

        Returns:
            A list of memory dicts sorted by timestamp.
        """
        dummy_vec = [0.0] * self.config.vector_dimension
        results = self.index.query(
            vector=dummy_vec,
            top_k=10000,  # large fetch
            namespace="emotional_memories",
            filter={
                "timestamp": {"$gte": start_time, "$lte": end_time},
                "consciousness_score": {"$gte": min_consciousness_score}
            },
            include_metadata=True
        )

        memories = []
        for match in results.matches:
            md = match.metadata
            memories.append({
                "id": match.id,
                "emotion_values": md["emotion_values"],
                "attention_level": md["attention_level"],
                "narrative": md["narrative"],
                "consciousness_score": md["consciousness_score"],
                "timestamp": md["timestamp"]
            })
        # Sort by timestamp ascending
        memories.sort(key=lambda x: x["timestamp"])
        return memories

    def _update_memory_stats(self, consciousness_score: float) -> None:
        """
        Update memory stats incrementally (placeholder logic).
        """
        alpha = 0.01
        old_val = self.memory_stats["consciousness_relevance"]
        new_val = (1 - alpha) * old_val + alpha * consciousness_score
        self.memory_stats["consciousness_relevance"] = new_val

        # You could similarly update emotional_coherence, temporal_consistency, etc.

    def _calculate_temporal_consistency(self, m1: Dict, m2: Dict) -> float:
        """
        Compare two memories to produce a consistency measure from 0.0 to 1.0.
        """
        # Compare emotional difference
        emo_diff = []
        for k in m1["emotion_values"]:
            emo_diff.append(abs(m1["emotion_values"][k] - m2["emotion_values"][k]))
        emotion_consistency = 1.0 - np.mean(emo_diff)

        # Compare consciousness difference
        cs_diff = abs(m1["consciousness_score"] - m2["consciousness_score"])
        consciousness_consistency = 1.0 - cs_diff

        return (emotion_consistency + consciousness_consistency) / 2.0

