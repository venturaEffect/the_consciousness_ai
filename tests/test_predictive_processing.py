import pytest
import numpy as np
from models.perception.predictive_processor import PredictiveProcessor

@pytest.mark.asyncio
async def test_prediction_generation():
    processor = PredictiveProcessor()
    current_state = {
        'visual': np.zeros((64, 64, 3)),
        'audio': np.zeros(1000),
    }
    
    prediction = await processor.predict_next_state(current_state)
    assert prediction is not None
    assert 'visual' in prediction
    assert 'audio' in prediction