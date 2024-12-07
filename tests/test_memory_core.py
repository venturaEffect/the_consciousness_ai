import unittest
from pinecone import Index

class TestMemoryCore(unittest.TestCase):
    def setUp(self):
        # Connect to a Pinecone index (replace with your index name)
        self.index = Index("emotional-memory")

    def test_add_and_retrieve(self):
        # Add a sample vector
        self.index.upsert([("test-id", [0.1, 0.2, 0.3])])
        result = self.index.query([0.1, 0.2, 0.3], top_k=1)
        self.assertEqual(result["matches"][0]["id"], "test-id")

    def tearDown(self):
        self.index.delete(["test-id"])

if __name__ == "__main__":
    unittest.main()
