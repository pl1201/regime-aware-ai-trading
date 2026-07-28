from algo_trading.strategies import (
    SMAEMACrossStrategy,
    RSIDivergenceStrategy,
    MACDMomentumStrategy,
    BollingerBreakoutStrategy,
    VWAPMeanReversionStrategy,
    RenkoTrendStrategy,
    VolumeProfileImbalanceStrategy,
    OUProcessMeanReversionStrategy,
    KalmanFilterForecastStrategy,
    ARIMAStrategy,
    LSTMTransformerStrategy,
    StatArbCointegrationStrategy,
    GARCHVolatilityStrategy,
)

try:
    from algo_trading.strategies.ml.regime_transformer_strategy import RegimeTransformerStrategy
    HAS_REGIME_TRANSFORMER = True
except ImportError:
    HAS_REGIME_TRANSFORMER = False
    RegimeTransformerStrategy = None

STRATEGY_MAP = {
    'SMA/EMA Crossover': ('sma_ema', SMAEMACrossStrategy, {"fast":20,"slow":50,"ma_type":"ema"}),
    'RSI + Divergence': ('rsi_div', RSIDivergenceStrategy, {"period":14,"overbought":70,"oversold":30,"lookback":5}),
    'MACD Momentum': ('macd', MACDMomentumStrategy, {"fast":12,"slow":26,"signal":9}),
    'Bollinger Breakout': ('bb_breakout', BollingerBreakoutStrategy, {"window":20,"k":2.0}),
    'VWAP Mean Reversion': ('vwap_mr', VWAPMeanReversionStrategy, {"thr":1.5}),
    'Renko Trend': ('renko_trend', RenkoTrendStrategy, {"brick_atr":14,"brick_k":1.0}),
    'Volume Profile Imbalance': ('vol_profile', VolumeProfileImbalanceStrategy, {"window":200,"bins":20}),
    'OU Mean Reversion': ('ou_mr', OUProcessMeanReversionStrategy, {"lookback":100,"z":1.5}),
    'Kalman Forecast': ('kalman', KalmanFilterForecastStrategy, {"q":1e-4,"r":1e-3}),
    'ARIMA/SARIMA': ('arima', ARIMAStrategy, {"order":[1,1,1]}),
    'LSTM/Transformer': ('lstm', LSTMTransformerStrategy, {"lookback":50}),
    'StatArb Cointegration': ('stat_arb', StatArbCointegrationStrategy, {"lookback":250,"z":2.0}),
    'GARCH Volatility': ('garch_vol', GARCHVolatilityStrategy, {"window":250}),
}

if HAS_REGIME_TRANSFORMER:
    STRATEGY_MAP["Regime-Aware Transformer"] = (
        "regime_transformer",
        RegimeTransformerStrategy,
        {
            "model_path": None,
            "ev_threshold": 0.001,
            "position_sizing": "fixed",
            "risk_per_trade": 0.02,
            "allowed_regimes": ["trending", "ranging"],
            "sequence_length": 20,
        },
    )

try:
    from algo_trading.strategies.ml import RegimeEnsembleStrategy, RegimeEnsembleBanditStrategy
    HAS_REGIME_ENSEMBLE = True
except Exception:
    RegimeEnsembleStrategy = None
    RegimeEnsembleBanditStrategy = None
    HAS_REGIME_ENSEMBLE = False

if HAS_REGIME_ENSEMBLE:
    STRATEGY_MAP["Regime Ensemble (ML)"] = (
        "regime_ensemble",
        RegimeEnsembleStrategy,
        {
            "model_path": "models/regime_ensemble.pkl",  # single ensemble model
            "proba_threshold": 0.55,
            "allowed_regimes": ["trending", "ranging", "calm"],
            "use_direction_output": False,
            "use_dynamic_threshold": True,
        },
    )
    STRATEGY_MAP["Regime Ensemble (Bandit)"] = (
        "regime_ensemble_bandit",
        RegimeEnsembleBanditStrategy,
        {
            # ví dụ 3 base models, bạn đổi path cho đúng với models đã train
            "model_paths": {
                "rf": "models/regime_bandit_rf.pkl",
                "gb": "models/regime_bandit_gb.pkl",
                "logit": "models/regime_bandit_logit.pkl",
            },
            "proba_threshold": 0.55,
            "allowed_regimes": ["trending", "ranging", "calm"],
            "bandit_type": "ucb",
            "epsilon": 0.1,
            "reward_mode": "direction",
        },
    )



