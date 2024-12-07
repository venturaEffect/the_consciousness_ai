from pinecone import Pinecone
import numpy as np
from typing import List, Dict, Any
import time

class MemoryCore:
    def __init__(self, api_key: str, environment: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index("consciousness-memory")
        
    def store_experience(self, 
                        embedding: List[float], 
                        metadata: Dict[str, Any],
                        emotional_context: Dict[str, float]):
        """Store an experience with emotional context"""
        vector_id = f"exp_{np.random.uuid4()}"
        self.index.upsert(
            vectors=[(
                vector_id,
                embedding,
                {
                    **metadata,
                    "emotional_valence": emotional_context.get("valence"),
                    "emotional_arousal": emotional_context.get("arousal"),
                    "timestamp": time.time()
                }
            )]
        )
        
    def retrieve_similar_experiences(self, 
                                   query_embedding: List[float],
                                   emotional_filter: Dict[str, float] = None,
                                   top_k: int = 5):
        """Retrieve experiences with emotional context filtering"""
        filter_query = {}
        if emotional_filter:
            filter_query = {
                "emotional_valence": {"$gte": emotional_filter["min_valence"]},
                "emotional_arousal": {"$gte": emotional_filter["min_arousal"]}
            }
            
        return self.index.query(
            vector=query_embedding,
            filter=filter_query,
            top_k=top_k
        )
