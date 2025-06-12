import pandas as pd
import numpy as np
from typing import Dict

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        
    def preprocess(self, dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        df = self._handle_missing(df)
        df[self.config['numerical_features']] = self._normalize(df[self.config['numerical_features']])
        df = self._encode_categorical(df, self.config['categorical_features'])
        
        if self.config['model_type'] == 'lstm':
            df = self._create_sequences(df)
            
        return df