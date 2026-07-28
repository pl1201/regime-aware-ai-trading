"""
Regime-Aware Transformer Strategy - Phương án 1

Strategy này kết hợp:
1. HMM cho regime detection (Tầng 1)
2. Transformer cho conditional distribution learning (Tầng 2)
3. Expected Value calculation cho decision making (Tầng 3)

Đây là implementation của Phương án 1: Regime-Aware Conditional Distribution Learning
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import warnings

from ..base import BaseStrategy, StrategyResult
from algo_trading.indicators import (
    rsi, macd, bollinger_bands, atr, vwap,
    sma, ema
)

try:
    from algo_trading.market_models.regime import detect_regime_hmm
    from algo_trading.ml.features import FeatureEngineer
    from algo_trading.ml.models.transformer_distribution import TransformerDistributionWrapper
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    warnings.warn(f"Some modules not available: {e}")


class RegimeTransformerStrategy(BaseStrategy):
   
    name = "Regime-Aware Transformer"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        ev_threshold: float = 0.001,
        position_sizing: str = 'fixed',
        risk_per_trade: float = 0.02,
        allowed_regimes: Optional[list] = None,
        sequence_length: int = 20,
        indicators: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize strategy
        
        Args:
            model_path: Path đến trained model file (.pth)
            ev_threshold: Minimum EV để trade
            position_sizing: 'fixed' hoặc 'kelly'
            risk_per_trade: Risk per trade (nếu fixed sizing)
            allowed_regimes: List allowed regimes
            sequence_length: Sequence length cho Transformer
            indicators: List indicators to use
        """
        super().__init__(**kwargs)
        
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available. Install dependencies.")
        
        self.model_path = model_path
        self.ev_threshold = ev_threshold
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.allowed_regimes = allowed_regimes or ['trending', 'ranging']
        self.sequence_length = sequence_length
        self.indicators_list = indicators or ['RSI', 'MACD', 'BB', 'ATR', 'VWAP', 'SMA', 'EMA']
        
        # Initialize components
        self.model = None
        self.feature_engineer = None
        self.regime_detector = None
        
        # Load model nếu có
        if model_path:
            try:
                self.model = TransformerDistributionWrapper.load(model_path)
            except Exception as e:
                warnings.warn(f"Could not load model from {model_path}: {e}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Tính tất cả indicators"""
        indicators = {}
        close = df['close']
        
        if 'RSI' in self.indicators_list:
            indicators['rsi'] = rsi(close, 14)
        
        if 'MACD' in self.indicators_list:
            macd_line, macd_signal, macd_hist = macd(close)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
        
        if 'BB' in self.indicators_list:
            bb_upper, bb_middle, bb_lower = bollinger_bands(close)
            indicators['bb_upper'] = bb_upper
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        if 'ATR' in self.indicators_list:
            indicators['atr'] = atr(df, 14)
            indicators['atr_ratio'] = indicators['atr'] / close
        
        if 'VWAP' in self.indicators_list and 'volume' in df.columns:
            indicators['vwap'] = vwap(df)
            indicators['vwap_distance'] = (close - indicators['vwap']) / indicators['vwap']
        
        if 'SMA' in self.indicators_list:
            indicators['sma_20'] = sma(close, 20)
            indicators['sma_50'] = sma(close, 50)
        
        if 'EMA' in self.indicators_list:
            indicators['ema_20'] = ema(close, 20)
            indicators['ema_50'] = ema(close, 50)
        
        return indicators
    
    def _detect_regime(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Detect market regime sử dụng HMM"""
        try:
            regime_info = detect_regime_hmm(
                df,
                indicators=indicators,
                lookback_window=500
            )
            return regime_info
        except Exception as e:
            warnings.warn(f"Regime detection failed: {e}. Using fallback.")
            # Fallback: simple regime detection
            return self._simple_regime_detection(df, indicators)
    
    def _simple_regime_detection(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        macd_hist = indicators.get('macd_hist', pd.Series(0, index=df.index))
        rsi_val = indicators.get('rsi', pd.Series(50, index=df.index))
        bb_width = indicators.get('bb_width', pd.Series(0.02, index=df.index))
        
        # Simple rules
        trending = (macd_hist > 0.01) & (rsi_val > 30) & (rsi_val < 70)
        ranging = (abs(macd_hist) < 0.01) & (bb_width < 0.05)
        volatile = bb_width > 0.08
        calm = (bb_width < 0.03) & ~trending
        
        regime = pd.Series(0, index=df.index)  # Default: trending
        regime[calm] = 3
        regime[ranging] = 1
        regime[volatile] = 2
        
        regime_names = ['trending', 'ranging', 'volatile', 'calm']
        current_regime = regime_names[int(regime.iloc[-1])]
        
        return {
            'current_regime': current_regime,
            'current_regime_id': int(regime.iloc[-1]),
            'regime': regime,
            'regime_probabilities': pd.DataFrame(),
            'transition_matrix': pd.DataFrame(),
            'stationary_distribution': pd.Series(),
            'detector': None
        }
    
    def _calculate_expected_value(
        self,
        predicted_dist: Dict[str, np.ndarray],
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        commission: float = 0.001
    ) -> Dict[str, float]:
        expected_return = predicted_dist['mean'].item() if isinstance(predicted_dist['mean'], np.ndarray) else predicted_dist['mean']
        win_prob = predicted_dist['win_prob'].item() if isinstance(predicted_dist['win_prob'], np.ndarray) else predicted_dist['win_prob']
        ev_long = (
            win_prob * take_profit_pct -
            (1 - win_prob) * stop_loss_pct -
            commission
        )
        
        ev_short = (
            win_prob * stop_loss_pct -  # Short profit khi giá giảm
            (1 - win_prob) * take_profit_pct -  # Short loss khi giá tăng
            commission
        )
        
        # Net EV
        ev_net = max(ev_long, ev_short)
        recommended_direction = 'long' if ev_long > ev_short else 'short' if ev_short > 0 else 'none'
        
        return {
            'ev_long': ev_long,
            'ev_short': ev_short,
            'ev_net': ev_net,
            'recommended_direction': recommended_direction,
            'expected_return': expected_return,
            'win_probability': win_prob
        }
    
    def _calculate_position_size(
        self,
        ev_info: Dict[str, float],
        current_price: float,
        stop_loss_pct: float = 0.02
    ) -> float:
        """
        Tính position size
        
        Args:
            ev_info: Expected Value info
            current_price: Current price
            stop_loss_pct: Stop loss percentage
        
        Returns:
            Position size (fraction of capital)
        """
        if self.position_sizing == 'kelly':
            # Kelly Criterion
            win_prob = ev_info['win_probability']
            win_amount = 0.04  # Take profit
            loss_amount = 0.02  # Stop loss
            
            if loss_amount > 0:
                kelly_fraction = (win_prob * win_amount - (1 - win_prob) * loss_amount) / win_amount
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% (fractional Kelly)
                return kelly_fraction
            else:
                return 0.0
        else:
            # Fixed position sizing dựa trên risk per trade
            return self.risk_per_trade / stop_loss_pct
    
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """
        Generate trading signals
        
        Workflow:
        1. Tính indicators
        2. Detect regime (HMM)
        3. Tạo features và predict conditional distribution (Transformer)
        4. Tính Expected Value
        5. Generate signals dựa trên EV và regime
        """
        if self.model is None:
            # Nếu không có model, return zero signals
            signals = pd.Series(0, index=df.index)
            return StrategyResult(
                signals=signals,
                meta={'error': 'Model not loaded'}
            )
        
        # Step 1: Calculate indicators
        indicators = self._calculate_indicators(df)
        
        # Step 2: Detect regime
        regime_info = self._detect_regime(df, indicators)
        current_regime = regime_info['current_regime']
        current_regime_id = regime_info['current_regime_id']
        
        # Check if regime is allowed
        if current_regime not in self.allowed_regimes:
            signals = pd.Series(0, index=df.index)
            return StrategyResult(
                signals=signals,
                meta={
                    'regime': current_regime,
                    'reason': f'Regime {current_regime} not in allowed regimes'
                }
            )
        
        # Step 3: Create features và predict
        if self.feature_engineer is None:
            # CRITICAL: Đảm bảo FeatureEngineer config khớp với training
            # Nếu có model, thử detect config phù hợp dựa trên input_dim
            if self.model and hasattr(self.model, 'model') and hasattr(self.model.model, 'input_dim'):
                expected_dim = self.model.model.input_dim
                # Model với 672 features thường được train với config đơn giản hơn
                # Thử config đơn giản trước (không lags hoặc ít lags)
                if expected_dim < 800:
                    # Config đơn giản: có thể không có lags hoặc ít rolling stats
                    self.feature_engineer = FeatureEngineer(
                        sequence_length=self.sequence_length,
                        use_lags=False,  # Tắt lags để giảm features
                        n_lags=0,
                        use_rolling_stats=True,
                        rolling_windows=[5, 10],  # Ít windows hơn
                        scaler_type='robust'
                    )
                else:
                    # Config đầy đủ cho model lớn hơn
                    self.feature_engineer = FeatureEngineer(
                        sequence_length=self.sequence_length,
                        use_lags=True,
                        n_lags=5,
                        use_rolling_stats=True,
                        rolling_windows=[5, 10, 20],
                        scaler_type='robust'
                    )
            else:
                # Default config khi chưa có model
                self.feature_engineer = FeatureEngineer(
                    sequence_length=self.sequence_length,
                    use_lags=True,
                    n_lags=5,
                    use_rolling_stats=True,
                    rolling_windows=[5, 10, 20],
                    scaler_type='robust'
                )
        
        try:
            # Create features
            features_df = self.feature_engineer.create_features(
                df,
                indicators=indicators,
                market_models={'regime': regime_info}
            )
            
            # Transform features
            features_array = self.feature_engineer.transform_features(features_df, fit_scaler=True)
            
            # CRITICAL: Validate và auto-fix feature dimensions
            if self.model and hasattr(self.model, 'model') and hasattr(self.model.model, 'input_dim'):
                expected_input_dim = self.model.model.input_dim
                actual_input_dim = features_array.shape[1] if features_array.ndim >= 2 else features_array.shape[0]
                
                if actual_input_dim != expected_input_dim:
                    # Thử các config khác nhau để match với model
                    warnings.warn(
                        f"Feature dimension mismatch: Model expects {expected_input_dim}, got {actual_input_dim}. "
                        f"Attempting to adjust FeatureEngineer config..."
                    )
                    
                    base_features = len(features_df.columns)
                    configs_to_try = [
                        {'use_lags': False, 'n_lags': 0, 'use_rolling_stats': False, 'rolling_windows': []},
                        {'use_lags': True, 'n_lags': 1, 'use_rolling_stats': False, 'rolling_windows': []},
                        {'use_lags': True, 'n_lags': 2, 'use_rolling_stats': False, 'rolling_windows': []},
                        {'use_lags': True, 'n_lags': 3, 'use_rolling_stats': False, 'rolling_windows': []},
                        {'use_lags': False, 'n_lags': 0, 'use_rolling_stats': True, 'rolling_windows': [5]},
                        {'use_lags': False, 'n_lags': 0, 'use_rolling_stats': True, 'rolling_windows': [5, 10]},
                        {'use_lags': False, 'n_lags': 0, 'use_rolling_stats': True, 'rolling_windows': [10]},
                        {'use_lags': True, 'n_lags': 1, 'use_rolling_stats': True, 'rolling_windows': [5]},
                        {'use_lags': True, 'n_lags': 1, 'use_rolling_stats': True, 'rolling_windows': [5, 10]},
                        {'use_lags': True, 'n_lags': 2, 'use_rolling_stats': True, 'rolling_windows': [5]},
                        {'use_lags': True, 'n_lags': 2, 'use_rolling_stats': True, 'rolling_windows': [5, 10]},
                        {'use_lags': True, 'n_lags': 4, 'use_rolling_stats': False, 'rolling_windows': []},
                        {'use_lags': True, 'n_lags': 4, 'use_rolling_stats': True, 'rolling_windows': [5]},
                    ]
                    
                    matched = False
                    for config in configs_to_try:
                        try:
                            test_engineer = FeatureEngineer(
                                sequence_length=self.sequence_length,
                                use_lags=config.get('use_lags', False),
                                n_lags=config.get('n_lags', 5),
                                use_rolling_stats=config.get('use_rolling_stats', False),
                                rolling_windows=config.get('rolling_windows', [5, 10, 20]),
                                scaler_type='robust'
                            )
                            
                            test_features_df = test_engineer.create_features(df, indicators=indicators, market_models={'regime': regime_info})
                            test_features_array = test_engineer.transform_features(test_features_df, fit_scaler=True)
                            test_dim = test_features_array.shape[1] if test_features_array.ndim >= 2 else test_features_array.shape[0]
                            
                            # Debug: Log mỗi config thử
                            if abs(test_dim - expected_input_dim) <= 5:  # Gần đúng (cho phép sai số nhỏ)
                                warnings.warn(
                                    f"🔍 Testing config {config}: got {test_dim} features "
                                    f"(expected {expected_input_dim}, diff: {abs(test_dim - expected_input_dim)})"
                                )
                            
                            if test_dim == expected_input_dim:
                                self.feature_engineer = test_engineer
                                features_df = test_features_df
                                features_array = test_features_array
                                matched = True
                                warnings.warn(f"✅ Auto-matched FeatureEngineer config! Using: {config}")
                                break
                        except Exception as e:
                            # Log error để debug
                            warnings.warn(f"⚠️ Config {config} failed: {str(e)[:100]}")
                            continue
                    
                    if not matched:
                        error_msg = (
                            f"Feature dimension mismatch và không thể auto-fix!\n"
                            f"  Model expects: {expected_input_dim} features\n"
                            f"  Actual features: {actual_input_dim} features\n"
                            f"  Base features (without lags/rolling): {base_features}\n"
                            f"  Current config:\n"
                            f"    - use_lags: {self.feature_engineer.use_lags}\n"
                            f"    - n_lags: {self.feature_engineer.n_lags}\n"
                            f"    - use_rolling_stats: {self.feature_engineer.use_rolling_stats}\n"
                            f"    - rolling_windows: {self.feature_engineer.rolling_windows}\n"
                            f"  Solution: Re-train model với FeatureEngineer config hiện tại hoặc "
                            f"điều chỉnh config để match với model đã train."
                        )
                        warnings.warn(error_msg)
                        signals = pd.Series(0, index=df.index)
                        return StrategyResult(
                            signals=signals,
                            meta={'error': error_msg, 'expected_dim': expected_input_dim, 'actual_dim': actual_input_dim}
                        )
            
            # Create sequences
            features_sequences = self.feature_engineer.create_sequences(features_array)
            
            if len(features_sequences) == 0:
                signals = pd.Series(0, index=df.index)
                return StrategyResult(signals=signals, meta={'error': 'Not enough data for sequences'})
            
            n_sequences = len(features_sequences)
            regime_series = regime_info['regime']
            regime_ids_for_sequences = regime_series.values[-n_sequences:]
            if len(regime_ids_for_sequences) != n_sequences:
                if len(regime_ids_for_sequences) > n_sequences:
                    regime_ids_for_sequences = regime_ids_for_sequences[-n_sequences:]
                else:
                    last_val = regime_ids_for_sequences[-1] if len(regime_ids_for_sequences) > 0 else 0
                    regime_ids_for_sequences = np.concatenate([
                        np.full(n_sequences - len(regime_ids_for_sequences), last_val),
                        regime_ids_for_sequences
                    ])
            
            # Predict in batches
            all_predictions = []
            batch_size = 32
            for i in range(0, n_sequences, batch_size):
                batch_sequences = features_sequences[i:i+batch_size]
                batch_regime_ids = regime_ids_for_sequences[i:i+batch_size]
                
                # Validate batch shape
                if batch_sequences.ndim == 3:
                    batch_seq_len, batch_feat_dim = batch_sequences.shape[1], batch_sequences.shape[2]
                    if self.model and hasattr(self.model, 'model') and hasattr(self.model.model, 'input_dim'):
                        if batch_feat_dim != self.model.model.input_dim:
                            warnings.warn(
                                f"Skipping batch {i}: feature dim mismatch "
                                f"(got {batch_feat_dim}, expected {self.model.model.input_dim})"
                            )
                            continue
                
                try:
                    batch_pred = self.model.predict(batch_sequences, batch_regime_ids)
                except RuntimeError as e:
                    if "shapes cannot be multiplied" in str(e):
                        error_msg = (
                            f"Feature dimension mismatch in batch {i}!\n"
                            f"  Error: {e}\n"
                            f"  Batch shape: {batch_sequences.shape}\n"
                            f"  Model expects: {self.model.model.input_dim if hasattr(self.model, 'model') else 'unknown'} features\n"
                            f"  Solution: Re-train model với FeatureEngineer config hiện tại hoặc "
                            f"sử dụng cùng FeatureEngineer config như training."
                        )
                        warnings.warn(f"Error in signal generation: {error_msg}")
                        signals = pd.Series(0, index=df.index)
                        return StrategyResult(signals=signals, meta={'error': error_msg})
                    raise
                if isinstance(batch_pred, dict):
                    first_key = list(batch_pred.keys())[0]
                    first_val = batch_pred[first_key]
                    if isinstance(first_val, np.ndarray) and first_val.ndim > 0 and len(first_val) > 1:
                        batch_len = len(first_val)
                        for j in range(batch_len):
                            pred_dict = {}
                            for key, val in batch_pred.items():
                                if isinstance(val, np.ndarray) and val.ndim > 0:
                                    if len(val) == batch_len:
                                        pred_dict[key] = val[j] if val.ndim == 1 else val[j:j+1]
                                    else:
                                        pred_dict[key] = val
                                else:
                                    pred_dict[key] = val
                            all_predictions.append(pred_dict)
                    else:
                        all_predictions.append(batch_pred)
                else:
                    # Fallback: treat as single prediction
                    all_predictions.append(batch_pred)
            
            # Step 5: Generate signals cho tất cả timesteps
            signals = pd.Series(0, index=df.index)
            start_idx = self.sequence_length - 1
            
            regime_names = ['trending', 'ranging', 'volatile', 'calm']
            
            for i, pred_dist in enumerate(all_predictions):
                # Get regime cho sequence này
                seq_regime_id = regime_ids_for_sequences[i]
                seq_regime_name = regime_names[seq_regime_id] if 0 <= seq_regime_id < len(regime_names) else 'trending'
                
                # Check if regime is allowed
                if seq_regime_name not in self.allowed_regimes:
                    continue
                
                # Calculate Expected Value
                ev_info = self._calculate_expected_value(pred_dist)
                
                # Generate signal nếu EV > threshold
                if ev_info['ev_net'] > self.ev_threshold:
                    timestep_idx = start_idx + i
                    if timestep_idx < len(df):
                        current_price = df['close'].iloc[timestep_idx]
                        position_size = self._calculate_position_size(ev_info, current_price)
                        
                        if ev_info['recommended_direction'] == 'long':
                            signals.iloc[timestep_idx] = position_size
                        elif ev_info['recommended_direction'] == 'short':
                            signals.iloc[timestep_idx] = -position_size
            
            # Meta information (use last prediction)
            last_pred_dist = all_predictions[-1] if all_predictions else None
            if last_pred_dist:
                ev_info = self._calculate_expected_value(last_pred_dist)
                meta = {
                    'regime': current_regime,
                    'regime_id': current_regime_id,
                    'total_signals': int((signals != 0).sum()),
                    'long_signals': int((signals > 0).sum()),
                    'short_signals': int((signals < 0).sum()),
                    'ev_long': ev_info['ev_long'],
                    'ev_short': ev_info['ev_short'],
                    'ev_net': ev_info['ev_net'],
                    'recommended_direction': ev_info['recommended_direction'],
                    'expected_return': ev_info['expected_return'],
                    'win_probability': ev_info['win_probability'],
                    'predicted_distribution': {
                        'quantiles': last_pred_dist['quantiles'].tolist() if isinstance(last_pred_dist['quantiles'], np.ndarray) else last_pred_dist['quantiles'],
                        'mean': float(last_pred_dist['mean']) if isinstance(last_pred_dist['mean'], np.ndarray) else last_pred_dist['mean'],
                        'std': float(last_pred_dist['std']) if isinstance(last_pred_dist['std'], np.ndarray) else last_pred_dist['std'],
                    }
                }
            else:
                meta = {
                    'regime': current_regime,
                    'regime_id': current_regime_id,
                    'total_signals': 0,
                    'error': 'No predictions generated'
                }
            
            return StrategyResult(signals=signals, meta=meta)
            
        except Exception as e:
            warnings.warn(f"Error in signal generation: {e}")
            import traceback
            traceback.print_exc()
            signals = pd.Series(0, index=df.index)
            return StrategyResult(
                signals=signals,
                meta={'error': str(e)}
            )

