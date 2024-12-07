import unittest
from pinecone import Index

class TestMemoryCore(unittest.TestCase):
    def setUp(self):
        # Connect to Pinecone index
        self.index = Index("ac_memory")

    def test_add_vector(self):
        # Add a vector to memory
        self.index.upsert([("test-id", [0.1, 0.2, 0.3])])
        result = self.index.query([0.1, 0.2, 0.3], top_k=1)
        self.assertEqual(result["matches"][0]["id"], "test-id")

    def tearDown(self):
        # Clean up the index
        self.index.delete(["test-id"])

if __name__ == "__main__":
    unittest.main()
