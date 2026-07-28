"""
Ví dụ sử dụng Regime-Aware Transformer Strategy

Đây là ví dụ hoàn chỉnh về cách:
1. Tính indicators và detect regime
2. Train Transformer model
3. Sử dụng strategy để generate signals
4. Backtest strategy
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from algo_trading.data_loader import load_data
from algo_trading.indicators import rsi, macd, bollinger_bands, atr, vwap
from algo_trading.market_models.regime import detect_regime_hmm
from algo_trading.ml.features import FeatureEngineer
from algo_trading.ml.training import train_transformer_model
from algo_trading.strategies.ml.regime_transformer_strategy import RegimeTransformerStrategy


def example_1_prepare_data():
    """Ví dụ 1: Chuẩn bị data và tính indicators"""
    print("=" * 60)
    print("VÍ DỤ 1: Chuẩn bị data và tính indicators")
    print("=" * 60)
    
    # Load data (giả sử có function load_data)
    # df = load_data('BTCUSDT', '1h')
    
    # Hoặc tạo sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    df = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices,
        'volume': np.random.rand(1000) * 1000
    }, index=dates)
    
    # Tính indicators
    indicators = {
        'rsi': rsi(df['close'], 14),
        'macd_hist': macd(df['close'])[2],  # MACD histogram
        'bb_width': (bollinger_bands(df['close'])[0] - bollinger_bands(df['close'])[2]) / df['close'],
        'atr': atr(df, 14) / df['close'],  # Normalized ATR
    }
    
    if 'volume' in df.columns:
        indicators['volume'] = df['volume'] / df['volume'].rolling(20).mean()
    
    print(f"\nData shape: {df.shape}")
    print(f"\nIndicators calculated:")
    for name, series in indicators.items():
        print(f"  - {name}: {series.notna().sum()} non-null values")
    
    return df, indicators


def example_2_detect_regime(df, indicators):
    """Ví dụ 2: Detect regime sử dụng HMM"""
    print("\n" + "=" * 60)
    print("VÍ DỤ 2: Detect regime với HMM")
    print("=" * 60)
    
    try:
        regime_info = detect_regime_hmm(
            df,
            indicators=indicators,
            lookback_window=500
        )
        
        print(f"\nCurrent regime: {regime_info['current_regime']}")
        print(f"Current regime ID: {regime_info['current_regime_id']}")
        
        if 'regime_probabilities' in regime_info and not regime_info['regime_probabilities'].empty:
            print("\nRegime probabilities (last 5 periods):")
            print(regime_info['regime_probabilities'].tail())
        
        if 'transition_matrix' in regime_info and not regime_info['transition_matrix'].empty:
            print("\nTransition matrix:")
            print(regime_info['transition_matrix'])
        
        return regime_info
    
    except ImportError:
        print("\n⚠️  hmmlearn not installed. Using fallback regime detection.")
        # Fallback sẽ được tự động sử dụng trong detect_regime_hmm
        regime_info = detect_regime_hmm(df, indicators=indicators)
        print(f"\nCurrent regime (fallback): {regime_info['current_regime']}")
        return regime_info


def example_3_train_model(df, indicators, regime_info):
    """Ví dụ 3: Train Transformer model"""
    print("\n" + "=" * 60)
    print("VÍ DỤ 3: Train Transformer model")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("\n⚠️  PyTorch not installed. Cannot train model.")
        print("Install with: pip install torch")
        return None
    
    # Tạo features
    engineer = FeatureEngineer(sequence_length=20)
    features_df = engineer.create_features(
        df,
        indicators=indicators,
        market_models={'regime': regime_info}
    )
    
    print(f"\nFeatures created: {features_df.shape[1]} features")
    print(f"Feature columns: {list(features_df.columns[:10])}...")  # Show first 10
    
    # Scale features
    features_array = engineer.transform_features(features_df, fit_scaler=True)
    
    # Prepare targets (future returns)
    returns = df['close'].pct_change().shift(-1).fillna(0).values
    
    # Create sequences
    features_sequences = engineer.create_sequences(features_array)
    returns_sequences = returns[len(returns) - len(features_sequences):]
    
    # Regime IDs
    regime_ids_sequences = None
    if 'regime' in regime_info and isinstance(regime_info['regime'], pd.Series):
        regime_ids_sequences = regime_info['regime'].values
        regime_ids_sequences = regime_ids_sequences[len(regime_ids_sequences) - len(features_sequences):]
    
    print(f"\nSequences created: {features_sequences.shape}")
    print(f"  - Sequence length: {features_sequences.shape[1]}")
    print(f"  - Number of features: {features_sequences.shape[2]}")
    print(f"  - Number of sequences: {features_sequences.shape[0]}")
    
    # Train model (với config nhỏ để nhanh)
    print("\nTraining model...")
    model = train_transformer_model(
        features_sequences[-500:],  # Use last 500 sequences để nhanh
        returns_sequences[-500:],
        regime_ids_sequences[-500:] if regime_ids_sequences is not None else None,
        model_config={
            'd_model': 64,  # Smaller để nhanh
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1
        },
        training_config={
            'batch_size': 16,
            'epochs': 20,  # Ít epochs để nhanh
            'learning_rate': 1e-4,
            'weight_decay': 1e-5
        },
        validation_split=0.2
    )
    
    print("\n✅ Model trained successfully!")
    
    # Save model
    model_path = 'regime_transformer_model_example.pth'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return model, model_path


def example_4_use_strategy(df, model_path):
    """Ví dụ 4: Sử dụng strategy để generate signals"""
    print("\n" + "=" * 60)
    print("VÍ DỤ 4: Sử dụng strategy để generate signals")
    print("=" * 60)
    
    # Tạo strategy
    strategy = RegimeTransformerStrategy(
        model_path=model_path,
        ev_threshold=0.001,  # Minimum EV để trade
        position_sizing='fixed',
        risk_per_trade=0.02,  # 2% risk per trade
        allowed_regimes=['trending', 'ranging'],
        sequence_length=20
    )
    
    print("\nStrategy created with config:")
    print(f"  - EV threshold: {strategy.ev_threshold}")
    print(f"  - Position sizing: {strategy.position_sizing}")
    print(f"  - Allowed regimes: {strategy.allowed_regimes}")
    
    # Generate signals
    print("\nGenerating signals...")
    result = strategy.generate_signals(df)
    
    print(f"\nSignals generated:")
    print(f"  - Total periods: {len(result.signals)}")
    print(f"  - Non-zero signals: {(result.signals != 0).sum()}")
    print(f"  - Long signals: {(result.signals > 0).sum()}")
    print(f"  - Short signals: {(result.signals < 0).sum()}")
    
    # Meta information
    if 'meta' in result.__dict__ and result.meta:
        meta = result.meta
        print(f"\nMeta information:")
        if 'regime' in meta:
            print(f"  - Current regime: {meta['regime']}")
        if 'ev_net' in meta:
            print(f"  - Expected Value (net): {meta['ev_net']:.6f}")
        if 'win_probability' in meta:
            print(f"  - Win probability: {meta['win_probability']:.4f}")
        if 'recommended_direction' in meta:
            print(f"  - Recommended direction: {meta['recommended_direction']}")
    
    return result


def main():
    """Chạy tất cả ví dụ"""
    print("\n" + "=" * 60)
    print("REGIME-AWARE TRANSFORMER STRATEGY - VÍ DỤ SỬ DỤNG")
    print("=" * 60)
    
    # Example 1: Prepare data
    df, indicators = example_1_prepare_data()
    
    # Example 2: Detect regime
    regime_info = example_2_detect_regime(df, indicators)
    
    # Example 3: Train model
    model_result = example_3_train_model(df, indicators, regime_info)
    if model_result is None:
        print("\n⚠️  Cannot continue without trained model.")
        return
    
    model, model_path = model_result
    
    # Example 4: Use strategy
    result = example_4_use_strategy(df, model_path)
    
    print("\n" + "=" * 60)
    print("✅ TẤT CẢ VÍ DỤ ĐÃ HOÀN THÀNH!")
    print("=" * 60)
    print("\nĐể sử dụng trong production:")
    print("1. Train model trên historical data đầy đủ")
    print("2. Save model")
    print("3. Load model trong strategy")
    print("4. Generate signals và backtest")
    print("\nXem thêm: algo_trading/strategies/ml/README_REGIME_TRANSFORMER.md")


if __name__ == '__main__':
    main()

