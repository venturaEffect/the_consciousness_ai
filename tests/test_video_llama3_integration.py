import unittest
from models.integration.video_llama3_integration import VideoLLaMA3Integration

class TestVideoLLaMA3Integration(unittest.TestCase):
    def setUp(self):
        self.config = {
            'video_llama3': {
                'model_name': "DAMO-NLP-SG/VideoLLaMA3",
                'device': "cpu"
            }
        }
        self.integration = VideoLLaMA3Integration(self.config['video_llama3'])

    def test_load_model(self):
        model, processor = self.integration._load_video_llama3_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

    def test_process_video(self):
        # Add test for video processing
        pass

    def test_model_variant_switching(self):
        """Test enhanced model variant switching"""
        config = {
            "model_variants": {
                "default": "DAMO-NLP-SG/Llama3.3",
                "abliterated": "huihui-ai/Llama-3.3-70B-Instruct-abliterated"
            }
        }
        
        integration = VideoLLaMA3Integration(config)
        self.assertEqual(integration.current_variant, "default")
        
        integration.set_model_variant("abliterated")
        self.assertEqual(integration.current_variant, "abliterated")
        
        # Test invalid variant
        with self.assertRaises(ValueError):
            integration.set_model_variant("invalid_variant")

if __name__ == '__main__':
    unittest.main()