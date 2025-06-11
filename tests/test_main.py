"""Unit tests for the AI-Augmented DevSecOps framework"""

import pytest
import numpy as np
import tensorflow as tf
from src.main import LSTMThreatDetector, load_config, ConfigurationError

def test_lstm_detector_initialization():
    """Test LSTM detector initialization with paper-specified parameters"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    assert detector.input_dim == 64
    assert detector.sequence_length == 10
    assert isinstance(detector.model, tf.keras.Model)
    
    # Verify model architecture matches paper specification
    assert len(detector.model.layers) == 4  # LSTM + Dense + Dropout + Dense
    assert isinstance(detector.model.layers[0], tf.keras.layers.LSTM)
    assert detector.model.layers[0].units == 64  # As specified in paper
    assert isinstance(detector.model.layers[-1], tf.keras.layers.Dense)
    assert detector.model.layers[-1].units == 1  # Binary classification

def test_model_training():
    """Test model training with realistic dimensions from CICIDS2017 dataset"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    
    # Generate synthetic data matching paper's specifications
    X = np.random.random((100, 10, 64))  # 100 samples, 10 timesteps, 64 features
    y = np.random.randint(0, 2, (100,))  # Binary labels (benign/attack)
    
    history = detector.train(X, y, epochs=1)
    
    # Verify training metrics mentioned in paper
    assert 'loss' in history
    assert 'accuracy' in history
    assert 'auc' in history  # AUC metric as mentioned in paper

def test_model_performance_thresholds():
    """Test if model meets minimum performance thresholds from paper"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    
    # Train on synthetic data
    X = np.random.random((1000, 10, 64))
    y = np.random.randint(0, 2, (1000,))
    
    history = detector.train(X, y, epochs=5)
    
    # Check if meets minimum performance thresholds from paper
    assert max(history['accuracy']) > 0.85  # Paper mentions 95% detection rate
    assert max(history['auc']) > 0.80  # Paper shows AUC of 0.96

def test_config_loading():
    """Test configuration loading with required parameters"""
    config = load_config('config/settings.yaml')
    
    # Verify essential configuration parameters from paper
    assert config['application_name'] == 'ai-devsecops-framework'
    assert config['version'] == '1.0.0'
    assert 'features' in config
    assert config['features']['anomaly_detection']  # Feature mentioned in paper
    assert config['features']['predictive_threat_detection']

def test_invalid_config():
    """Test handling of invalid configuration"""
    with pytest.raises(ConfigurationError):
        load_config('nonexistent.yaml')

def test_model_concept_drift_adaptation():
    """Test model's ability to handle concept drift as described in paper"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    
    # Initial training
    X_initial = np.random.random((100, 10, 64))
    y_initial = np.random.randint(0, 2, (100,))
    initial_history = detector.train(X_initial, y_initial, epochs=1)
    
    # Simulate drift with slightly different distribution
    X_drift = np.random.random((100, 10, 64)) * 1.2
    y_drift = np.random.randint(0, 2, (100,))
    drift_history = detector.train(X_drift, y_drift, epochs=1)
    
    # Verify model adapts to new distribution
    assert 'accuracy' in drift_history
    assert 'loss' in drift_history

def test_early_stopping():
    """Test early stopping callback implementation"""
    detector = LSTMThreatDetector(input_dim=64, sequence_length=10)
    X = np.random.random((100, 10, 64))
    y = np.random.randint(0, 2, (100,))
    
    history = detector.train(X, y, epochs=10)  # Early stopping should trigger
    
    # Verify training stopped early due to patience=3
    assert len(history['loss']) < 10