"""
Transformer Distribution Model - Học conditional return distribution

Model này sử dụng Transformer encoder để học conditional probability distribution:
P(return | regime, indicators, market_state)

Output không phải là point prediction mà là full distribution:
- Quantiles (q10, q25, q50, q75, q90)
- Moments (mean, std, skewness, kurtosis)
- Win probability: P(return > 0 | state)

Architecture:
- Input: Sequence of features [indicator_t-n, ..., indicator_t]
- Transformer Encoder: Học temporal dependencies
- Regime Embedding: Inject regime information vào Transformer
- Distribution Head: Output full distribution parameters
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


class RegimeEmbedding(nn.Module):
    """
    Embedding layer cho regime information
    Inject regime vào Transformer thông qua embedding
    """
    def __init__(self, n_regimes: int = 4, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(n_regimes, embed_dim)
    
    def forward(self, regime_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regime_ids: [batch_size] tensor với regime IDs
        Returns:
            [batch_size, embed_dim] regime embeddings
        """
        return self.embedding(regime_ids)


class TransformerDistributionModel(nn.Module):
    """
    Transformer model để học conditional return distribution
    
    Architecture:
    1. Feature embedding: Linear projection của input features
    2. Regime embedding: Embed regime ID
    3. Positional encoding: Thêm positional information
    4. Transformer encoder: Học temporal dependencies
    5. Distribution head: Output distribution parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        use_regime_embedding: bool = True
    ):
        """
        Args:
            input_dim: Số lượng features (indicators + market model outputs)
            n_regimes: Số lượng regimes
            d_model: Dimension của Transformer model
            nhead: Số attention heads
            num_layers: Số Transformer encoder layers
            dim_feedforward: Dimension của feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_regime_embedding: Có sử dụng regime embedding không
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.n_regimes = n_regimes
        self.d_model = d_model
        self.use_regime_embedding = use_regime_embedding
        
        # Feature embedding: project input features to d_model
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # Regime embedding (optional)
        if use_regime_embedding:
            self.regime_embedding = RegimeEmbedding(n_regimes, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Distribution head: output distribution parameters
        # Output: quantiles (5) + moments (4: mean, std, skew, kurt) = 9 values
        self.distribution_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 9)  # 5 quantiles + 4 moments
        )
        
        # Win probability head: P(return > 0 | state)
        self.win_prob_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: [batch_size, seq_len, input_dim] input features
            regime_ids: [batch_size] regime IDs (optional)
        
        Returns:
            Dict với:
                - 'quantiles': [batch_size, 5] (q10, q25, q50, q75, q90)
                - 'mean': [batch_size, 1]
                - 'std': [batch_size, 1]
                - 'skew': [batch_size, 1]
                - 'kurt': [batch_size, 1]
                - 'win_prob': [batch_size, 1] P(return > 0)
        """
        batch_size, seq_len, _ = features.shape
        
        # Feature embedding
        x = self.feature_embedding(features)  # [batch_size, seq_len, d_model]
        
        # Add regime embedding (broadcast to sequence)
        if self.use_regime_embedding and regime_ids is not None:
            regime_emb = self.regime_embedding(regime_ids)  # [batch_size, d_model]
            regime_emb = regime_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
            x = x + regime_emb
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # Use last timestep for prediction
        x_last = x[:, -1, :]  # [batch_size, d_model]
        
        # Distribution head
        dist_params = self.distribution_head(x_last)  # [batch_size, 9]
        
        # Parse distribution parameters
        quantiles = dist_params[:, :5]  # [batch_size, 5]
        mean = dist_params[:, 5:6]  # [batch_size, 1]
        std = F.softplus(dist_params[:, 6:7])  # Ensure positive
        skew = dist_params[:, 7:8]  # [batch_size, 1]
        kurt = F.softplus(dist_params[:, 8:9]) + 1.0  # Kurtosis >= 1
        
        # Win probability
        win_prob = self.win_prob_head(x_last)  # [batch_size, 1]
        
        return {
            'quantiles': quantiles,
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurt': kurt,
            'win_prob': win_prob
        }


class PositionalEncoding(nn.Module):
    """
    Positional encoding cho Transformer
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDistributionWrapper:
    """
    Wrapper class để dễ sử dụng TransformerDistributionModel
    Xử lý data preprocessing, training, và prediction
    """
    
    def __init__(
        self,
        model: Optional[TransformerDistributionModel] = None,
        input_dim: Optional[int] = None,
        n_regimes: int = 4,
        device: str = 'cpu',
        **model_kwargs
    ):
        """
        Args:
            model: Pre-trained model (optional)
            input_dim: Số lượng input features
            n_regimes: Số lượng regimes
            device: 'cpu' hoặc 'cuda'
            **model_kwargs: Additional kwargs cho TransformerDistributionModel
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.device = torch.device(device)
        self.n_regimes = n_regimes
        
        if model is None:
            if input_dim is None:
                raise ValueError("Cần input_dim hoặc pre-trained model")
            self.model = TransformerDistributionModel(
                input_dim=input_dim,
                n_regimes=n_regimes,
                **model_kwargs
            ).to(self.device)
        else:
            self.model = model.to(self.device)
            self.input_dim = model.input_dim
        
        self.is_trained = False
    
    def predict(
        self,
        features: np.ndarray,
        regime_ids: Optional[np.ndarray] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Predict conditional distribution
        
        Args:
            features: [n_samples, seq_len, input_dim] hoặc [seq_len, input_dim]
            regime_ids: [n_samples] hoặc scalar (optional)
            return_dict: Nếu True, trả về dict; nếu False, chỉ trả về quantiles
        
        Returns:
            Dict với distribution parameters hoặc quantiles array
        """
        self.model.eval()
        
        # Convert to tensor
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        features = np.array(features)
        if features.ndim == 2:
            features = features[np.newaxis, :, :]  # Add batch dimension
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        regime_ids_tensor = None
        if regime_ids is not None:
            regime_ids = np.array(regime_ids)
            if regime_ids.ndim == 0:
                regime_ids = regime_ids[np.newaxis]
            regime_ids_tensor = torch.LongTensor(regime_ids).to(self.device)
        
        with torch.no_grad():
            output = self.model(features_tensor, regime_ids_tensor)
        
        # Convert to numpy
        result = {}
        for key, value in output.items():
            result[key] = value.cpu().numpy()
            if result[key].shape[0] == 1:
                result[key] = result[key][0]  # Remove batch dimension if single sample
        
        if return_dict:
            return result
        else:
            return result['quantiles']
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'n_regimes': self.n_regimes,
            'model_config': {
                'd_model': self.model.d_model,
                'use_regime_embedding': self.model.use_regime_embedding,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model"""
        checkpoint = torch.load(path, map_location=device)
        model = TransformerDistributionModel(
            input_dim=checkpoint['input_dim'],
            n_regimes=checkpoint['n_regimes'],
            **checkpoint['model_config']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        wrapper = cls(model=model, device=device)
        wrapper.is_trained = True
        return wrapper






























