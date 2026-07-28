"""
Regime Detection Module - Phát hiện regime của thị trường sử dụng HMM

Module này sử dụng Hidden Markov Model (HMM) để phát hiện các regime ẩn của thị trường:
- Trending (xu hướng): giá có xu hướng rõ ràng lên hoặc xuống
- Ranging (sideways): giá dao động trong một khoảng
- Volatile (biến động cao): volatility cao, giá biến động mạnh
- Calm (ổn định): volatility thấp, giá ổn định

Input: Technical indicators (RSI, MACD, Bollinger Bands width, ATR, Volume)
Output: Current regime, regime probabilities, transition matrix
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn not available. Install with: pip install hmmlearn")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")


class RegimeDetector:
    
    REGIME_NAMES = ['trending', 'ranging', 'volatile', 'calm']
    
    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Args:
            n_regimes: Số lượng regimes (mặc định 4)
            n_iter: Số iterations cho HMM training
            random_state: Random seed
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        
        # HMM model với Gaussian emissions
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='diag', 
            n_iter=n_iter,
            random_state=random_state
        )
        
        self.is_fitted = False
        self.regime_names = self.REGIME_NAMES[:n_regimes]
        self.scaler = None  # Store scaler for feature normalization
    
    def fit(
        self,
        observations: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None
    ) -> 'RegimeDetector':
        """
        Train HMM model trên observations
        
        Args:
            observations: DataFrame với columns là indicators (RSI, MACD, BB_width, ATR, etc.)
            regime_labels: Optional ground truth labels (nếu có) để validate
        
        Returns:
            self
        """
        if observations.isna().any().any():
            # Fill NaN với forward fill và backward fill
            observations = observations.ffill().bfill()
        
        # Remove infinite values
        observations = observations.replace([np.inf, -np.inf], np.nan)
        observations = observations.ffill().bfill()
        
        # Remove constant columns (variance = 0)
        numeric_cols = observations.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if observations[col].std() < 1e-8:
                # Replace constant column with small random noise
                observations[col] = observations[col] + np.random.randn(len(observations)) * 1e-6
        
        X = observations.values.astype(np.float64)
        
        # Check for NaN or Inf
        if np.isnan(X).any() or np.isinf(X).any():
            # Fill remaining NaN/Inf with column means
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X) | np.isinf(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        # Normalize features để tránh covariance issues
        if not SKLEARN_AVAILABLE:
            # Fallback: manual normalization
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
            X_scaled = (X - X_mean) / X_std
            scaler = None  # Can't store scaler without sklearn
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Add small regularization to covariance để đảm bảo positive-definite
        # Sử dụng covariance_type='diag' thay vì 'full' để tránh issues
        if self.model.covariance_type == 'full':
            # Try full first, fallback to diag if fails
            try:
                self.model.fit(X_scaled)
            except (ValueError, np.linalg.LinAlgError):
                # Fallback to diagonal covariance
                self.model.covariance_type = 'diag'
                self.model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type='diag',
                    n_iter=self.n_iter,
                    random_state=self.random_state
                )
                self.model.fit(X_scaled)
        else:
            self.model.fit(X_scaled)
        
        self.is_fitted = True
        self.scaler = scaler 
        
        return self
    
    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """
        Predict regime cho mỗi observation
        
        Args:
            observations: DataFrame với indicators
        
        Returns:
            Series với regime labels (0, 1, 2, 3)
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        if observations.isna().any().any():
            observations = observations.ffill().bfill()
        
        # Remove infinite values
        observations = observations.replace([np.inf, -np.inf], np.nan)
        observations = observations.ffill().bfill()
        
        X = observations.values.astype(np.float64)
        
        # Check for NaN or Inf
        if np.isnan(X).any() or np.isinf(X).any():
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X) | np.isinf(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        # Scale với scaler đã fit
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        regimes = self.model.predict(X)
        
        return pd.Series(regimes, index=observations.index, name='regime')
    
    def predict_proba(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities cho mỗi observation
        
        **Cách HMM tính xác suất regime:**
        
        1. **Emission Probabilities (P(observation | regime)):**
           - Mỗi regime có một Gaussian distribution với mean và covariance riêng
           - HMM học được các parameters này từ training data:
             * μ_regime: Mean vector của indicators trong regime đó
             * Σ_regime: Covariance matrix (diagonal trong trường hợp này)
           - Với mỗi observation x_t, tính: P(x_t | regime_i) = N(x_t; μ_i, Σ_i)
        
        2. **Forward-Backward Algorithm:**
           - Forward probability α_t(i) = P(regime_i tại t, observations 1..t)
           - Backward probability β_t(i) = P(observations t+1..T | regime_i tại t)
           - Posterior probability: P(regime_i tại t | all observations) = α_t(i) * β_t(i) / P(observations)
        
        3. **Kết quả:**
           - Mỗi row trong output là xác suất của 4 regimes tại thời điểm đó
           - Tổng các xác suất = 1.0
        
        **Ví dụ:**
        - Nếu RSI cao, MACD dương, BB_width nhỏ → High prob cho "trending"
        - Nếu ATR cao, Volume cao → High prob cho "volatile"
        - Nếu tất cả indicators đều ở mức trung bình → High prob cho "calm"
        
        Args:
            observations: DataFrame với indicators (RSI, MACD, BB_width, ATR, Volume)
        
        Returns:
            DataFrame với columns là probabilities cho mỗi regime:
            - prob_trending: Xác suất regime "trending"
            - prob_ranging: Xác suất regime "ranging"  
            - prob_volatile: Xác suất regime "volatile"
            - prob_calm: Xác suất regime "calm"
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        if observations.isna().any().any():
            observations = observations.ffill().bfill()
        
        # Remove infinite values
        observations = observations.replace([np.inf, -np.inf], np.nan)
        observations = observations.ffill().bfill()
        
        X = observations.values.astype(np.float64)
        
        # Check for NaN or Inf
        if np.isnan(X).any() or np.isinf(X).any():
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X) | np.isinf(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        # Scale với scaler đã fit
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X)
        
        return pd.DataFrame(
            probs,
            index=observations.index,
            columns=[f'prob_{name}' for name in self.regime_names]
        )
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Lấy transition matrix: P(regime_t+1 | regime_t)
        
        Returns:
            DataFrame với transition probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train.")
        
        transmat = self.model.transmat_
        return pd.DataFrame(
            transmat,
            index=self.regime_names,
            columns=self.regime_names
        )
    
    def get_stationary_distribution(self) -> pd.Series:
        
        if not self.is_fitted:
            raise ValueError("Model chưa được train.")
        
        # Stationary distribution là eigenvector của transition matrix
        # với eigenvalue = 1
        transmat = self.model.transmat_
        eigenvals, eigenvecs = np.linalg.eig(transmat.T)
        stationary_idx = np.argmax(np.abs(eigenvals - 1.0))
        stationary = eigenvecs[:, stationary_idx].real
        stationary = stationary / stationary.sum()
        
        return pd.Series(
            stationary,
            index=self.regime_names,
            name='stationary_prob'
        )
    
    def get_regime_name(self, regime_id: int) -> str:
        """Lấy tên regime từ ID"""
        return self.regime_names[regime_id]


def detect_regime_hmm(
    df: pd.DataFrame,
    indicators: Optional[Dict[str, pd.Series]] = None,
    n_regimes: int = 4,
    lookback_window: int = 1000
) -> Dict[str, any]:
    if not HMMLEARN_AVAILABLE:
        # Fallback: simple rule-based regime detection
        return _simple_regime_detection(df)
    
    # Prepare observations
    if indicators is None:
        # Auto-calculate indicators
        from algo_trading.indicators import rsi, macd, bollinger_bands, atr
        
        indicators = {
            'rsi': rsi(df['close'], 14),
            'macd_hist': macd(df['close'])[2],  # MACD histogram
            'bb_width': (bollinger_bands(df['close'])[0] - bollinger_bands(df['close'])[2]) / df['close'],
            'atr': atr(df, 14) / df['close'],  # Normalized ATR
        }
        if 'volume' in df.columns:
            indicators['volume'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Create observations DataFrame
    obs_df = pd.DataFrame(indicators)
    obs_df = obs_df.dropna()

    min_hmm_obs = 120
    n_obs = len(obs_df)
    if n_obs < min_hmm_obs:
        return _simple_regime_detection(df)

    window = min(lookback_window, n_obs)
    train_obs = obs_df.iloc[-window:]

    # Train HMM on lookback window only.
    detector = RegimeDetector(n_regimes=n_regimes)
    detector.fit(train_obs)

    # Predict only on the same lookback window to avoid look-ahead on older history.
    latent_regimes = detector.predict(train_obs)
    latent_probs = detector.predict_proba(train_obs)

    # Map latent states -> canonical regimes by observed characteristics, not raw state ID.
    canonical_names = ['trending', 'ranging', 'volatile', 'calm']

    def _pick_col(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c in train_obs.columns:
                return c
        return None

    col_macd = _pick_col(['macd_hist'])
    col_rsi = _pick_col(['rsi'])
    col_bb = _pick_col(['bb_width'])
    col_atr = _pick_col(['atr', 'atr_ratio'])
    col_vol = _pick_col(['volatility_20'])

    latent_ids = list(range(detector.n_regimes))
    state_stats: Dict[int, Dict[str, float]] = {}
    for sid in latent_ids:
        mask = (latent_regimes.values == sid)
        if not mask.any():
            state_stats[sid] = {'trend_score': 0.0, 'vol_score': 0.0}
            continue

        trend_score = 0.0
        vol_score = 0.0

        if col_macd is not None:
            trend_score += float(np.nanmean(np.abs(train_obs.loc[mask, col_macd].values)))
        if col_rsi is not None:
            trend_score += float(np.nanmean(np.abs(train_obs.loc[mask, col_rsi].values - 50.0))) / 50.0
        if col_bb is not None:
            vol_score += float(np.nanmean(np.abs(train_obs.loc[mask, col_bb].values)))
        if col_atr is not None:
            vol_score += float(np.nanmean(np.abs(train_obs.loc[mask, col_atr].values)))
        if col_vol is not None:
            vol_score += float(np.nanmean(np.abs(train_obs.loc[mask, col_vol].values)))

        state_stats[sid] = {
            'trend_score': trend_score,
            'vol_score': vol_score,
        }

    # Assign canonical IDs: trending=0, ranging=1, volatile=2, calm=3.
    latent_to_canonical: Dict[int, int] = {sid: sid for sid in latent_ids}
    if latent_ids:
        sid_volatile = max(latent_ids, key=lambda s: state_stats[s]['vol_score'])
        sid_calm = min(latent_ids, key=lambda s: state_stats[s]['vol_score'])
        latent_to_canonical[sid_volatile] = 2
        latent_to_canonical[sid_calm] = 3

        remaining = [s for s in latent_ids if s not in {sid_volatile, sid_calm}]
        if remaining:
            sid_trending = max(remaining, key=lambda s: state_stats[s]['trend_score'])
            latent_to_canonical[sid_trending] = 0
            remaining = [s for s in remaining if s != sid_trending]
            for s in remaining:
                latent_to_canonical[s] = 1

    # Build canonical regime series on full df index; only lookback window has HMM-inferred values.
    regimes_window = latent_regimes.map(latent_to_canonical).astype(int)
    regimes_full = pd.Series(np.nan, index=df.index, dtype=float)
    regimes_full.loc[train_obs.index] = regimes_window.values
    regimes_full = regimes_full.ffill().fillna(0).astype(int)

    # Remap probabilities to canonical regime columns.
    probs_values = latent_probs.values
    probs_canonical_window = np.zeros((len(latent_probs), len(canonical_names)), dtype=float)
    for latent_sid in latent_ids:
        if latent_sid >= probs_values.shape[1]:
            continue
        canonical_id = latent_to_canonical.get(latent_sid, latent_sid)
        if 0 <= canonical_id < len(canonical_names):
            probs_canonical_window[:, canonical_id] += probs_values[:, latent_sid]

    regime_probs = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=[f'prob_{name}' for name in canonical_names],
        dtype=float,
    )
    regime_probs.loc[train_obs.index, :] = probs_canonical_window
    regime_probs = regime_probs.ffill().fillna(1.0 / len(canonical_names))

    # Optional smoothing: nới lỏng xác suất để tránh 1 regime chiếm gần như 100%
    # Giúp snapshot trực quan hơn, phân bố xác suất giữa các regime đồng đều hơn.
    try:
        alpha = 0.6 
        probs_values = regime_probs.values
        n_regimes_eff = probs_values.shape[1]
        uniform_prior = np.full_like(probs_values, 1.0 / n_regimes_eff)
        smoothed = alpha * probs_values + (1.0 - alpha) * uniform_prior
        # Chuẩn hóa lại cho chắc chắn mỗi hàng sum = 1
        smoothed /= smoothed.sum(axis=1, keepdims=True)
        regime_probs = pd.DataFrame(
            smoothed,
            index=regime_probs.index,
            columns=regime_probs.columns,
        )
    except Exception:
        # Nếu có lỗi (edge case), giữ nguyên regime_probs gốc
        pass
    
    # Get current regime (canonical)
    current_regime_id = int(regimes_window.iloc[-1])
    current_regime_name = canonical_names[current_regime_id]

    # Remap transition/stationary distributions to canonical ordering.
    transition_latent = detector.get_transition_matrix().values
    transition_canonical = np.zeros((len(canonical_names), len(canonical_names)), dtype=float)
    for i_sid in latent_ids:
        for j_sid in latent_ids:
            i_c = latent_to_canonical.get(i_sid, i_sid)
            j_c = latent_to_canonical.get(j_sid, j_sid)
            if 0 <= i_c < len(canonical_names) and 0 <= j_c < len(canonical_names):
                transition_canonical[i_c, j_c] += transition_latent[i_sid, j_sid]

    row_sums = transition_canonical.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_canonical = transition_canonical / row_sums

    stationary_latent = detector.get_stationary_distribution().values
    stationary_canonical = np.zeros(len(canonical_names), dtype=float)
    for sid in latent_ids:
        c = latent_to_canonical.get(sid, sid)
        if 0 <= c < len(canonical_names) and sid < len(stationary_latent):
            stationary_canonical[c] += stationary_latent[sid]

    if stationary_canonical.sum() <= 0:
        stationary_canonical[:] = 1.0 / len(canonical_names)
    else:
        stationary_canonical /= stationary_canonical.sum()
    
    return {
        'current_regime': current_regime_name,
        'current_regime_id': int(current_regime_id),
        'regime': regimes_full,
        'regime_probabilities': regime_probs,
        'transition_matrix': pd.DataFrame(
            transition_canonical,
            index=canonical_names,
            columns=canonical_names,
        ),
        'stationary_distribution': pd.Series(
            stationary_canonical,
            index=canonical_names,
            name='stationary_prob',
        ),
        'detector': detector  # Keep detector for future predictions
    }


def _simple_regime_detection(df: pd.DataFrame) -> Dict[str, any]:
    from algo_trading.indicators import rsi, macd, bollinger_bands, atr
    rsi_val = rsi(df['close'], 14)
    macd_line, macd_signal, macd_hist = macd(df['close'])
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
    atr_val = atr(df, 14)
    trending = (macd_hist > 0) & (rsi_val > 30) & (rsi_val < 70)
    
    # Ranging: Price trong Bollinger Bands và MACD histogram nhỏ
    ranging = (df['close'] > bb_lower) & (df['close'] < bb_upper) & (abs(macd_hist) < 0.01)
    
    # Volatile: ATR cao
    volatile = atr_val > atr_val.rolling(20).mean() * 1.5
    
    # Calm: ATR thấp và không trending
    calm = (atr_val < atr_val.rolling(20).mean() * 0.7) & ~trending
    
    # Assign regimes (priority: volatile > trending > ranging > calm)
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
        'regime_probabilities': pd.DataFrame(),  # Empty for fallback
        'transition_matrix': pd.DataFrame(),
        'stationary_distribution': pd.Series(),
        'detector': None
    }

