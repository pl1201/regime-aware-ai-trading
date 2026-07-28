
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification

    Focal Loss giúp model tập trung vào hard samples bằng cách:
    1. Down-weight easy samples (already classified correctly)
    2. Focus on hard samples (misclassified or low confidence)
    3. Handle class imbalance tự động mà không cần SMOTE
    """

    def __init__(
        self,
        alpha: float = 0.25,  # Class balancing weight
        gamma: float = 2.0,   # Focusing parameter
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Args:
            alpha: Class balancing weight (0.25 cho minority classes)
            gamma: Focusing parameter (1-5, cao hơn = tập trung hơn vào hard samples)
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Index to ignore in loss calculation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss

        Args:
            inputs: Predictions (logits) [batch_size, num_classes]
            targets: True labels [batch_size] or [batch_size, num_classes]

        Returns:
            Focal loss value
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Focal Loss")

        # Handle one-hot encoded targets
        if targets.dim() == inputs.dim() - 1:
            # Convert to one-hot
            targets = F.one_hot(targets, inputs.size(-1)).float()
        elif targets.dim() != inputs.dim():
            raise ValueError(f"Target dimensions {targets.dim()} not compatible with input dimensions {inputs.dim()}")

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets.argmax(dim=1), reduction='none')

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def get_focal_weights(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Get focal weights cho từng sample để analysis

        Args:
            inputs: Predictions (logits)
            targets: True labels

        Returns:
            Focal weights cho từng sample
        """
        if not HAS_TORCH:
            return torch.tensor([])

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal weights
        focal_weights = self.alpha * (1 - pt) ** self.gamma

        return focal_weights


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss với class-specific weights
    """

    def __init__(
        self,
        num_classes: int,
        alpha: Optional[Union[float, List[float]]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            num_classes: Số lượng classes
            alpha: Class weights (None = equal weights, float = same weight cho tất cả, list = weights riêng)
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(MultiClassFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            # Equal weights
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, (list, tuple)):
            if len(alpha) != num_classes:
                raise ValueError(f"Alpha length {len(alpha)} must match num_classes {num_classes}")
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            # Same weight cho tất cả
            self.alpha = torch.full((num_classes,), alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate multi-class focal loss

        Args:
            inputs: Predictions (logits) [batch_size, num_classes]
            targets: True labels [batch_size]

        Returns:
            Multi-class focal loss
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Focal Loss")

        # Ensure alpha is on same device as inputs
        alpha = self.alpha.to(inputs.device)

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Get class-specific alpha
        alpha_t = alpha.gather(0, targets.data.view(-1))

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def focal_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> float:
    """
    Focal Loss implementation cho numpy arrays (không cần PyTorch)

    Args:
        y_true: True labels (one-hot encoded) [batch_size, num_classes]
        y_pred: Predicted probabilities [batch_size, num_classes]
        alpha: Class balancing weight
        gamma: Focusing parameter

    Returns:
        Focal loss value
    """
    # Clip predictions để tránh log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Cross entropy
    ce = -np.sum(y_true * np.log(y_pred), axis=1)

    # Probabilities of true class
    pt = np.sum(y_true * y_pred, axis=1)

    # Focal loss
    focal_weights = alpha * (1 - pt) ** gamma
    focal_loss = focal_weights * ce

    return np.mean(focal_loss)


class FocalLossOptimizer:
    """
    Helper class để optimize focal loss parameters
    """

    def __init__(
        self,
        class_distribution: Dict[int, int],
        target_gamma_range: Tuple[float, float] = (1.0, 5.0),
        target_alpha_range: Tuple[float, float] = (0.1, 0.9)
    ):
        """
        Args:
            class_distribution: Dict với class_id -> count
            target_gamma_range: Range cho gamma optimization
            target_alpha_range: Range cho alpha optimization
        """
        self.class_distribution = class_distribution
        self.target_gamma_range = target_gamma_range
        self.target_alpha_range = target_alpha_range

    def suggest_alpha_gamma(self) -> Tuple[float, float]:
        """
        Suggest alpha và gamma values dựa trên class distribution

        Returns:
            Tuple of (alpha, gamma)
        """
        # Tính class imbalance ratio
        counts = list(self.class_distribution.values())
        if len(counts) < 2:
            return 0.25, 2.0

        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / (min_count + 1e-8)

        # Alpha: cao hơn khi imbalance nghiêm trọng
        if imbalance_ratio > 10:
            alpha = 0.75  # Focus mạnh vào minority
        elif imbalance_ratio > 5:
            alpha = 0.50
        elif imbalance_ratio > 2:
            alpha = 0.35
        else:
            alpha = 0.25

        # Gamma: cao hơn khi imbalance nghiêm trọng
        if imbalance_ratio > 10:
            gamma = 3.0
        elif imbalance_ratio > 5:
            gamma = 2.5
        elif imbalance_ratio > 2:
            gamma = 2.0
        else:
            gamma = 1.5

        # Clamp to target ranges
        alpha = np.clip(alpha, self.target_alpha_range[0], self.target_alpha_range[1])
        gamma = np.clip(gamma, self.target_gamma_range[0], self.target_gamma_range[1])

        return alpha, gamma

    def get_class_weights(self) -> List[float]:
        """
        Get class weights cho imbalance correction

        Returns:
            List of weights cho từng class
        """
        if not self.class_distribution:
            return [1.0]

        total_samples = sum(self.class_distribution.values())
        weights = []

        for class_id in sorted(self.class_distribution.keys()):
            count = self.class_distribution[class_id]
            # Inverse frequency weighting
            weight = total_samples / (len(self.class_distribution) * count + 1e-8)
            weights.append(weight)

        return weights


def create_focal_loss(
    num_classes: int,
    alpha: Optional[Union[float, List[float]]] = None,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> Union[FocalLoss, MultiClassFocalLoss]:
    """
    Convenience function để tạo focal loss

    Args:
        num_classes: Số lượng classes
        alpha: Class weights
        gamma: Focusing parameter
        reduction: Reduction method

    Returns:
        FocalLoss hoặc MultiClassFocalLoss instance
    """
    if num_classes == 2:
        return FocalLoss(alpha=alpha or 0.25, gamma=gamma, reduction=reduction)
    else:
        return MultiClassFocalLoss(
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            reduction=reduction
        )


# Example usage với scikit-learn compatible wrapper
class FocalLossClassifier:
    """
    Scikit-learn compatible wrapper cho Focal Loss
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        learning_rate: float = 0.01,
        epochs: int = 100
    ):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            learning_rate: Learning rate cho training
            epochs: Số epochs training
        """
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model với focal loss

        Args:
            X: Features [n_samples, n_features]
            y: Labels [n_samples]
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Focal Loss training")

        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create simple neural network
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(np.unique(y))),
        )

        # Create optimizer và loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Probabilities [n_samples, n_classes]
        """
        if not self.is_fitted or not HAS_TORCH:
            raise ValueError("Model not fitted or PyTorch not available")

        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Predicted labels [n_samples]
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)