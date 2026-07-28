"""
Feature Engineering Module - Tạo features từ indicators và market models

Module này kết hợp:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, VWAP, etc.)
- Market model outputs (regime, volatility forecasts, etc.)
- Lagged features, rolling statistics
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings


class FeatureEngineer:
    """
    Feature Engineering class để tạo features từ indicators và market models
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        use_lags: bool = True,
        n_lags: int = 5,
        use_rolling_stats: bool = True,
        rolling_windows: List[int] = [5, 10, 20],
        scaler_type: str = 'robust'  # 'standard' or 'robust'
    ):
        """
        Args:
            sequence_length: Độ dài sequence cho Transformer
            use_lags: Có tạo lagged features không
            n_lags: Số lượng lags
            use_rolling_stats: Có tạo rolling statistics không
            rolling_windows: List các window sizes cho rolling stats
            scaler_type: Loại scaler ('standard' hoặc 'robust')
        """
        self.sequence_length = sequence_length
        self.use_lags = use_lags
        self.n_lags = n_lags
        self.use_rolling_stats = use_rolling_stats
        self.rolling_windows = rolling_windows
        self.scaler_type = scaler_type
        
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
    
    def create_features(
        self,
        df: pd.DataFrame,
        indicators: Optional[Dict[str, pd.Series]] = None,
        market_models: Optional[Dict[str, any]] = None
    ) -> pd.DataFrame:
        """
        Tạo features từ DataFrame với indicators và market models
        
        Args:
            df: DataFrame với price data (cần có 'close', 'high', 'low', 'open')
            indicators: Dict với indicator names và Series values
            market_models: Dict với market model outputs (regime, volatility, etc.)
        
        Returns:
            DataFrame với features, index khớp với df.index
        """
        features_list = []
        
        # 1. Basic price features
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            log_returns = np.log(df['close'] / df['close'].shift(1))
            features_list.append(pd.DataFrame({
                'return': returns,
                'log_return': log_returns,
                'price': df['close'],
            }))
        
        # 2. Indicator features
        if indicators is not None:
            for name, series in indicators.items():
                if isinstance(series, pd.Series):
                    features_list.append(pd.DataFrame({f'ind_{name}': series}))
        
        # 3. Market model features
        if market_models is not None:
            # Regime features
            if 'regime' in market_models:
                regime_info = market_models['regime']
                if isinstance(regime_info, dict):
                    # Regime probabilities
                    if 'regime_probabilities' in regime_info:
                        regime_probs = regime_info['regime_probabilities']
                        if isinstance(regime_probs, pd.DataFrame):
                            features_list.append(regime_probs.add_prefix('regime_prob_'))
                    
                    # Current regime (one-hot encoded)
                    if 'current_regime_id' in regime_info:
                        regime_id = regime_info['current_regime_id']
                        n_regimes = 4  # Default
                        if 'regime_probabilities' in regime_info:
                            n_regimes = len(regime_info['regime_probabilities'].columns)
                        
                        regime_onehot = np.zeros((len(df), n_regimes))
                        regime_onehot[:, regime_id] = 1.0
                        features_list.append(pd.DataFrame(
                            regime_onehot,
                            index=df.index,
                            columns=[f'regime_{i}' for i in range(n_regimes)]
                        ))
            
            # Volatility features (nếu có)
            if 'garch' in market_models:
                garch_info = market_models['garch']
                if isinstance(garch_info, dict) and 'forecast_vol' in garch_info:
                    features_list.append(pd.DataFrame({
                        'garch_vol': pd.Series(
                            garch_info['forecast_vol'],
                            index=df.index
                        )
                    }))
        
        # Combine all features
        if not features_list:
            raise ValueError("Không có features nào được tạo")
        
        features_df = pd.concat(features_list, axis=1)
        features_df = features_df.ffill().bfill()
        
        # 4. Lagged features
        if self.use_lags:
            lagged_features = []
            for col in features_df.columns:
                for lag in range(1, self.n_lags + 1):
                    lagged_features.append(features_df[col].shift(lag).rename(f'{col}_lag{lag}'))
            if lagged_features:
                features_df = pd.concat([features_df] + lagged_features, axis=1)
        
        # 5. Rolling statistics
        if self.use_rolling_stats:
            rolling_features = []
            for col in features_df.select_dtypes(include=[np.number]).columns:
                for window in self.rolling_windows:
                    rolling_features.append(
                        features_df[col].rolling(window).mean().rename(f'{col}_ma{window}')
                    )
                    rolling_features.append(
                        features_df[col].rolling(window).std().rename(f'{col}_std{window}')
                    )
            if rolling_features:
                features_df = pd.concat([features_df] + rolling_features, axis=1)
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        return features_df
    
    def fit_scaler(self, features_df: pd.DataFrame):
        """Fit scaler trên features"""
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Only fit on numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        self.scaler.fit(features_df[numeric_cols].fillna(0))
        self.is_fitted = True
    
    def transform_features(
        self,
        features_df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Transform features thành numpy array với scaling
        
        Args:
            features_df: DataFrame với features
            fit_scaler: Nếu True, fit scaler trước khi transform
        
        Returns:
            numpy array [n_samples, n_features]
        """
        if fit_scaler or not self.is_fitted:
            self.fit_scaler(features_df)
        
        # Fill NaN
        features_df = features_df.fillna(0)
        
        # Scale
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_scaled = self.scaler.transform(features_df[numeric_cols])
        
        return features_scaled
    
    def create_sequences(
        self,
        features_array: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Tạo sequences từ features array cho Transformer
        
        Args:
            features_array: [n_samples, n_features] array
            targets: [n_samples] target values (optional)
        
        Returns:
            Nếu có targets: (X_sequences, y_sequences)
            Nếu không: X_sequences
            X_sequences: [n_sequences, sequence_length, n_features]
            y_sequences: [n_sequences]
        """
        n_samples, n_features = features_array.shape
        
        if n_samples < self.sequence_length:
            raise ValueError(f"Cần ít nhất {self.sequence_length} samples")
        
        n_sequences = n_samples - self.sequence_length + 1
        
        # Create sequences
        X_sequences = np.zeros((n_sequences, self.sequence_length, n_features))
        for i in range(n_sequences):
            X_sequences[i] = features_array[i:i+self.sequence_length]
        
        if targets is not None:
            # Targets là giá trị tại cuối sequence
            y_sequences = targets[self.sequence_length-1:]
            return X_sequences, y_sequences
        else:
            return X_sequences


def create_features(
    df: pd.DataFrame,
    indicators: Optional[Dict[str, pd.Series]] = None,
    market_models: Optional[Dict[str, any]] = None,
    sequence_length: int = 20,
    use_lags: bool = True,
    scale: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function để tạo features
    
    Args:
        df: DataFrame với price data
        indicators: Dict với indicators
        market_models: Dict với market models
        sequence_length: Độ dài sequence
        use_lags: Có tạo lags không
        scale: Có scale features không
    
    Returns:
        Tuple (features_array, features_df)
        - features_array: [n_samples, sequence_length, n_features] cho Transformer
        - features_df: DataFrame với raw features
    """
    engineer = FeatureEngineer(
        sequence_length=sequence_length,
        use_lags=use_lags
    )
    
    # Create features
    features_df = engineer.create_features(df, indicators, market_models)
    
    # Scale và tạo sequences
    if scale:
        features_array = engineer.transform_features(features_df, fit_scaler=True)
    else:
        features_array = features_df.select_dtypes(include=[np.number]).fillna(0).values
    
    # Create sequences
    features_sequences = engineer.create_sequences(features_array)
    
    return features_sequences, features_df

