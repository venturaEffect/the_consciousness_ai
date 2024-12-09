# models/memory/memory_core.py
import pinecone
import uuid

class MemoryCore:
    def __init__(self, api_key, environment="us-west1-gcp", index_name="memory-core", dimension=768):
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension)
        self.index = pinecone.Index(index_name)

    def store_memory(self, embedding, metadata=None):
        vector_id = str(uuid.uuid4())
        self.index.upsert([(vector_id, embedding.tolist(), metadata)])

    def retrieve_memory(self, query_embedding, top_k=5):
        results = self.index.query(query_embedding.tolist(), top_k=top_k, include_metadata=True)
        return results["matches"]