import unittest
import numpy as np
from models.integration.video_llama3_integration import VideoLLaMA3Integration

class TestVideoLLaMA3Integration(unittest.TestCase):
    def setUp(self):
        self.config = {
            'video_llama3': {
                'model_name': "DAMO-NLP-SG/VideoLLaMA3",
                'device': "cpu",
                'memory_config': {
                    'max_buffer_size': 32,
                    'cleanup_threshold': 0.8
                },
                'ace_config': {
                    'animation_quality': "high",
                    'latency_target_ms': 100
                }
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

    async def test_memory_optimization(self):
        frame = np.random.rand(480, 640, 3)
        result = await self.integration.process_stream_frame(frame)
        
        self.assertIn('memory_metrics', result)
        self.assertIn('ace_result', result)
        
        metrics = result['memory_metrics']
        self.assertGreaterEqual(metrics.get('compression_ratio', 0), 0)

if __name__ == '__main__':
    unittest.main()