import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DetectionEngine:
    """AI-based threat detection engine"""
    
    def __init__(self, model_path: str = None):
        self.model = self._load_model(model_path) if model_path else self._build_model()
        self.threshold = 0.95  # Detection threshold from paper
        
    def _build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture from paper"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(10, 64)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            return model
        except Exception as e:
            logger.error(f"Model building failed: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Run threat detection on input features"""
        try:
            predictions = self.model.predict(features)
            alerts = []
            
            for i, pred in enumerate(predictions):
                if pred[0] > self.threshold:
                    alerts.append({
                        'timestamp': datetime.now().isoformat(),
                        'confidence': float(pred[0]),
                        'feature_index': i,
                        'severity': 'HIGH' if pred[0] > 0.98 else 'MEDIUM'
                    })
            
            return predictions, alerts
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise