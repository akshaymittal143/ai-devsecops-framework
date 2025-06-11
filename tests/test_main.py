"""Unit tests for the AI-DevSecOps framework"""

import pytest
import numpy as np
from src.main import LSTMThreatDetector, load_config

def test_lstm_detector_initialization():
    """Test LSTM detector initialization"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    assert detector.input_dim == 64
    assert detector.sequence_length == 10
    assert detector.model is not None

def test_model_training():
    """Test model training with dummy data"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    X = np.random.random((100, 10, 64))
    y = np.random.randint(0, 2, (100,))
    history = detector.train(X, y, epochs=1)
    assert 'loss' in history
    assert 'accuracy' in history

def test_config_loading():
    """Test configuration loading"""
    config = load_config('config/settings.yaml')
    assert config['application_name'] == 'ai-devsecops-framework'
    assert config['version'] == '1.0.0'

def test_invalid_config():
    """Test handling of invalid configuration"""
    with pytest.raises(Exception):
        load_config('nonexistent.yaml')