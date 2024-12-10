"""
Memory Core Module for ACM Project

Handles memory embedding, storage, and retrieval using Pinecone vector databases.
Optimized for narrative coherence and emotional meta-memory indexing.
"""

import pinecone
import logging
import uuid
import numpy as np


class MemoryCore:
    def __init__(self, api_key, index_name="memory-core", dimension=768, environment="us-west1-gcp"):
        """
        Initialize Memory Core with Pinecone.
        Args:
            api_key (str): Pinecone API key.
            index_name (str): Name of the Pinecone index.
            dimension (int): Dimension of memory embeddings.
            environment (str): Pinecone environment for the index.
        """
        logging.basicConfig(level=logging.INFO)
        pinecone.init(api_key=api_key, environment=environment)
        
        if index_name not in pinecone.list_indexes():
            logging.info(f"Creating Pinecone index: {index_name}")
            pinecone.create_index(index_name, dimension=dimension)
        
        self.index = pinecone.Index(index_name)
        logging.info(f"Memory Core initialized with index: {index_name}")

    def store_memory(self, text, embedding):
        """
        Store a single memory with its embedding.
        Args:
            text (str): Text content of the memory.
            embedding (list): Embedding vector for the memory.
        """
        try:
            vector_id = str(uuid.uuid4())
            metadata = {"text": text}
            self.index.upsert([(vector_id, embedding, metadata)])
            logging.info(f"Memory stored: {text[:50]}...")
        except Exception as e:
            logging.error(f"Error storing memory: {e}")

    def batch_store_memories(self, texts, embeddings):
        """
        Store multiple memories in a batch operation.
        Args:
            texts (list): List of text contents.
            embeddings (list): List of embedding vectors.
        """
        try:
            upserts = [
                (str(uuid.uuid4()), embeddings[i], {"text": texts[i]})
                for i in range(len(texts))
            ]
            self.index.upsert(upserts)
            logging.info(f"Batch of {len(texts)} memories stored.")
        except Exception as e:
            logging.error(f"Error in batch memory storage: {e}")

    def retrieve_memory(self, query_embedding, top_k=5):
        """
        Retrieve the top K most relevant memories for a given embedding.
        Args:
            query_embedding (list): Embedding vector for the query.
            top_k (int): Number of top matches to retrieve.
        Returns:
            list: List of memory texts from the top matches.
        """
        try:
            results = self.index.query(query_embedding, top_k=top_k, include_metadata=True)
            matches = [match["metadata"]["text"] for match in results["matches"]]
            logging.info(f"Retrieved {len(matches)} memories.")
            return matches
        except Exception as e:
            logging.error(f"Error retrieving memory: {e}")
            return []

    def delete_memory(self, vector_id):
        """
        Delete a memory by its vector ID.
        Args:
            vector_id (str): The unique ID of the memory vector.
        """
        try:
            self.index.delete(ids=[vector_id])
            logging.info(f"Memory with ID {vector_id} deleted.")
        except Exception as e:
            logging.error(f"Error deleting memory: {e}")


# Example Usage
if __name__ == "__main__":
    # Replace with your Pinecone API key
    api_key = "your-pinecone-api-key"
    
    memory_core = MemoryCore(api_key)
    
    # Example memory storage
    text = "AI agents must adapt to emotional stress in simulations."
    embedding = np.random.random(768).tolist()  # Replace with actual embedding logic
    memory_core.store_memory(text, embedding)
    
    # Example memory retrieval
    query_embedding = np.random.random(768).tolist()  # Replace with actual query embedding
    memories = memory_core.retrieve_memory(query_embedding, top_k=3)
    print(f"Retrieved Memories: {memories}")
