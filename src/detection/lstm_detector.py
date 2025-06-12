from typing import Dict, Tuple
import tensorflow as tf
import numpy as np

class LSTMDetector:
    """Real-time threat detection using LSTM neural networks"""
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def predict(self, sequence: np.ndarray) -> Tuple[bool, float]:
        """Predicts if a sequence contains threats"""
        score = self.model.predict(sequence)
        return bool(score > self.config['threshold']), float(score)