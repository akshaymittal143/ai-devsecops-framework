# filepath: src/devsecops/data/ingestion.py
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

class CICIDSDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
    
    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess CICIDS2017 dataset"""
        df = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        df = df.dropna()
        
        # Split features and labels
        X = df.drop('Label', axis=1)
        y = df['Label'].map({'BENIGN': 0, 'ATTACK': 1})
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y