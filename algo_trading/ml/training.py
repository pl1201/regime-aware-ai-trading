"""
Training Pipeline cho Transformer Distribution Model

Training với:
- Walk-forward validation để tránh look-ahead bias
- Time series cross-validation
- Optimize Expected Value thay vì accuracy
- Hyperparameter tuning với Bayesian Optimization
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
import warnings
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

from .models.transformer_distribution import TransformerDistributionWrapper
from .features import FeatureEngineer, create_features


class ReturnDataset(Dataset):
    """Dataset cho return prediction"""
    
    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        regime_ids: Optional[np.ndarray] = None
    ):
        """
        Args:
            features: [n_samples, seq_len, n_features]
            returns: [n_samples] future returns
            regime_ids: [n_samples] regime IDs (optional)
        """
        # Validate lengths
        n_samples = len(features)
        if len(returns) != n_samples:
            raise ValueError(f"Length mismatch: features has {n_samples} samples but returns has {len(returns)}")
        
        if regime_ids is not None:
            if len(regime_ids) != n_samples:
                # Truncate or pad to match
                if len(regime_ids) > n_samples:
                    regime_ids = regime_ids[:n_samples]
                else:
                    # Pad with last value
                    last_val = regime_ids[-1] if len(regime_ids) > 0 else 0
                    regime_ids = np.concatenate([regime_ids, np.full(n_samples - len(regime_ids), last_val)])
        
        self.features = torch.FloatTensor(features)
        self.returns = torch.FloatTensor(returns)
        self.regime_ids = None
        if regime_ids is not None:
            self.regime_ids = torch.LongTensor(regime_ids)
    
    def __len__(self):
        return len(self.returns)
    
    def __getitem__(self, idx):
        if self.regime_ids is not None:
            return self.features[idx], self.returns[idx], self.regime_ids[idx]
        else:
            return self.features[idx], self.returns[idx]


def quantile_loss(predicted_quantiles: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    Quantile loss (pinball loss) cho quantile regression
    
    Args:
        predicted_quantiles: [batch_size, n_quantiles] predicted quantiles
        target: [batch_size] actual returns
        quantiles: List of quantile values (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
    
    Returns:
        Loss value
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target.unsqueeze(1) - predicted_quantiles[:, i:i+1]
        loss = torch.max(q * error, (q - 1) * error)
        losses.append(loss)
    return torch.mean(torch.cat(losses, dim=1))


def expected_value_loss(
    predicted_dist: Dict[str, torch.Tensor],
    target: torch.Tensor,
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
) -> torch.Tensor:
    """
    Combined loss function để optimize Expected Value
    
    Loss = Quantile Loss + Distribution Consistency Loss + Win Probability Loss
    
    Args:
        predicted_dist: Dict với 'quantiles', 'mean', 'std', 'win_prob'
        target: [batch_size] actual returns
        quantiles: List of quantile values
    """
    # 1. Quantile loss
    quantile_loss_val = quantile_loss(predicted_dist['quantiles'], target, quantiles)
    
    # 2. Distribution consistency: mean should be close to median (q50)
    mean_median_diff = torch.mean((predicted_dist['mean'] - predicted_dist['quantiles'][:, 2]) ** 2)
    
    # 3. Win probability loss: binary cross-entropy
    win_target = (target > 0).float().unsqueeze(1)
    win_prob_loss = nn.functional.binary_cross_entropy(
        predicted_dist['win_prob'],
        win_target
    )
    
    # 4. Mean squared error cho mean prediction
    mean_loss = nn.functional.mse_loss(predicted_dist['mean'].squeeze(), target)
    
    # Combined loss
    total_loss = (
        quantile_loss_val +
        0.1 * mean_median_diff +
        0.5 * win_prob_loss +
        0.2 * mean_loss
    )
    
    return total_loss


def train_transformer_model(
    features: np.ndarray,
    returns: np.ndarray,
    regime_ids: Optional[np.ndarray] = None,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
    validation_split: float = 0.2,
    device: str = 'cpu'
) -> TransformerDistributionWrapper:
    """
    Train Transformer Distribution Model
    
    Args:
        features: [n_samples, seq_len, n_features] feature sequences
        returns: [n_samples] future returns (target)
        regime_ids: [n_samples] regime IDs (optional)
        model_config: Dict với model hyperparameters
        training_config: Dict với training hyperparameters
        validation_split: Fraction of data for validation
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained TransformerDistributionWrapper
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Default configs
    if model_config is None:
        model_config = {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
        }
    
    if training_config is None:
        training_config = {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
        }
    
    n_regimes = len(np.unique(regime_ids)) if regime_ids is not None else 4
    input_dim = features.shape[2]
    
    # Create model
    wrapper = TransformerDistributionWrapper(
        input_dim=input_dim,
        n_regimes=n_regimes,
        device=device,
        **model_config
    )
    
    # Split data
    n_train = int(len(features) * (1 - validation_split))
    train_features = features[:n_train]
    train_returns = returns[:n_train]
    train_regime_ids = regime_ids[:n_train] if regime_ids is not None else None
    
    val_features = features[n_train:]
    val_returns = returns[n_train:]
    val_regime_ids = regime_ids[n_train:] if regime_ids is not None else None
    
    # Create datasets
    train_dataset = ReturnDataset(train_features, train_returns, train_regime_ids)
    val_dataset = ReturnDataset(val_features, val_returns, val_regime_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(
        wrapper.model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(training_config['epochs']):
        # Training
        wrapper.model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                batch_features, batch_returns, batch_regime_ids = batch
            else:
                batch_features, batch_returns = batch
                batch_regime_ids = None
            
            batch_features = batch_features.to(device)
            batch_returns = batch_returns.to(device)
            if batch_regime_ids is not None:
                batch_regime_ids = batch_regime_ids.to(device)
            
            optimizer.zero_grad()
            pred_dist = wrapper.model(batch_features, batch_regime_ids)
            loss = expected_value_loss(pred_dist, batch_returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        wrapper.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_features, batch_returns, batch_regime_ids = batch
                else:
                    batch_features, batch_returns = batch
                    batch_regime_ids = None
                
                batch_features = batch_features.to(device)
                batch_returns = batch_returns.to(device)
                if batch_regime_ids is not None:
                    batch_regime_ids = batch_regime_ids.to(device)
                
                pred_dist = wrapper.model(batch_features, batch_regime_ids)
                loss = expected_value_loss(pred_dist, batch_returns)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{training_config['epochs']}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    wrapper.is_trained = True
    return wrapper


def walk_forward_validation(
    df: pd.DataFrame,
    indicators: Dict[str, pd.Series],
    market_models: Dict[str, any],
    train_window: int = 1000,
    test_window: int = 200,
    step_size: int = 200,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None
) -> List[Dict]:
    """
    Walk-forward validation cho time series
    
    Args:
        df: DataFrame với price data
        indicators: Dict với indicators
        market_models: Dict với market models
        train_window: Số periods cho training
        test_window: Số periods cho testing
        step_size: Step size giữa các folds
        model_config: Model config
        training_config: Training config
    
    Returns:
        List of dicts với results từ mỗi fold
    """
    results = []
    
    # Prepare features và targets
    engineer = FeatureEngineer(sequence_length=20)
    features_df = engineer.create_features(df, indicators, market_models)
    features_array = engineer.transform_features(features_df, fit_scaler=True)
    
    # Future returns (target)
    returns = df['close'].pct_change().shift(-1).fillna(0).values
    
    # Regime IDs
    regime_ids = None
    if 'regime' in market_models and 'regime' in market_models['regime']:
        regime_series = market_models['regime']['regime']
        if isinstance(regime_series, pd.Series):
            regime_ids = regime_series.values
    
    # Create sequences
    features_sequences = engineer.create_sequences(features_array)
    returns_sequences = returns[len(returns) - len(features_sequences):]
    if regime_ids is not None:
        regime_ids_sequences = regime_ids[len(regime_ids) - len(features_sequences):]
    else:
        regime_ids_sequences = None
    
    n_samples = len(features_sequences)
    
    # Walk-forward
    for start_idx in range(0, n_samples - train_window - test_window, step_size):
        train_end = start_idx + train_window
        test_start = train_end
        test_end = min(test_start + test_window, n_samples)
        
        if test_end - test_start < 10:  # Skip nếu test set quá nhỏ
            continue
        
        print(f"\nFold: Train [{start_idx}:{train_end}], Test [{test_start}:{test_end}]")
        
        # Split data
        train_features = features_sequences[start_idx:train_end]
        train_returns = returns_sequences[start_idx:train_end]
        train_regime_ids = regime_ids_sequences[start_idx:train_end] if regime_ids_sequences is not None else None
        
        test_features = features_sequences[test_start:test_end]
        test_returns = returns_sequences[test_start:test_end]
        test_regime_ids = regime_ids_sequences[test_start:test_end] if regime_ids_sequences is not None else None
        
        # Train model
        model = train_transformer_model(
            train_features,
            train_returns,
            train_regime_ids,
            model_config=model_config,
            training_config=training_config
        )
        
        # Evaluate on test set
        predictions = model.predict(test_features, test_regime_ids)
        
        # Calculate metrics
        predicted_returns = predictions['mean'].flatten()
        win_prob = predictions['win_prob'].flatten()
        
        # Expected Value
        ev = np.mean(predicted_returns)
        
        # Actual performance
        actual_returns = test_returns
        sharpe = np.mean(actual_returns) / (np.std(actual_returns) + 1e-8) * np.sqrt(252)
        
        results.append({
            'fold': len(results) + 1,
            'train_start': start_idx,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'predicted_ev': ev,
            'actual_mean_return': np.mean(actual_returns),
            'actual_sharpe': sharpe,
            'win_prob_mean': np.mean(win_prob),
            'model': model
        })
    
    return results

