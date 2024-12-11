class MemoryCore:
    def retrieve_memory(self, query_embedding, top_k=5, filters=None):
        results = self.index.query(
            query_embedding.tolist(), top_k=top_k, include_metadata=True, filter=filters
        )
        return results