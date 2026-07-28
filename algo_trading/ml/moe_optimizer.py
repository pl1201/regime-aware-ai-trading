"""
Hyperparameter Optimization for Enhanced MOE v2

Mục tiêu:
- Tối ưu hóa hyperparameters cho từng expert
- Cross-validation để đảm bảo model stability
- Out-of-sample testing
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional, Any
import warnings
import joblib
from scipy.stats import randint, uniform

from .dynamic_moe_v2_enhanced import DynamicMOE_v2_Enhanced


class MOEHyperparameterOptimizer:
    """
    Hyperparameter optimizer cho Enhanced MOE v2
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}

    def optimize_expert_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expert_type: str = "trend",
        cv_folds: int = 5,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa hyperparameters cho từng expert

        Args:
            X: Feature matrix
            y: Target labels
            expert_type: Loại expert ("trend", "range", "volatility")
            cv_folds: Số folds cho cross-validation
            n_iter: Số iterations cho RandomizedSearch

        Returns:
            Dictionary với best parameters và score
        """
        if expert_type == "trend":
            # Trend Detector (Gradient Boosting)
            model = GradientBoostingClassifier(random_state=self.random_state)
            param_distributions = {
                'n_estimators': randint(100, 1000),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.6, 0.4),
                'min_samples_split': randint(10, 50),
                'min_samples_leaf': randint(5, 25)
            }
        elif expert_type == "range":
            # Range Finder (Random Forest)
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            param_distributions = {
                'n_estimators': randint(100, 500),
                'max_depth': randint(5, 15),
                'min_samples_split': randint(10, 30),
                'min_samples_leaf': randint(5, 20),
                'max_features': ['sqrt', 'log2', None]
            }
        else:  # volatility
            # Volatility Breakout (Random Forest)
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            param_distributions = {
                'n_estimators': randint(150, 400),
                'max_depth': randint(4, 12),
                'min_samples_split': randint(8, 25),
                'min_samples_leaf': randint(4, 15),
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
            }

        # Randomized search
        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )

        search.fit(X, y)

        self.best_params[expert_type] = search.best_params_
        self.best_scores[expert_type] = search.best_score_

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def optimize_gating_network(
        self,
        X: np.ndarray,
        regime_ids: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa gating network hyperparameters
        """
        model = LogisticRegression(random_state=self.random_state, max_iter=2000)
        param_grid = {
            'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }

        search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )

        search.fit(X, regime_ids)

        self.best_params['gating'] = search.best_params_
        self.best_scores['gating'] = search.best_score_

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def get_best_params(self) -> Dict[str, Any]:
        """Lấy best parameters đã tìm được"""
        return self.best_params

    def get_best_scores(self) -> Dict[str, float]:
        """Lấy best scores đã tìm được"""
        return self.best_scores


def cross_validate_moe(
    model: DynamicMOE_v2_Enhanced,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: str = 'f1_macro'
) -> Dict[str, Any]:
    """
    Cross-validation cho MOE model

    Args:
        model: MOE model đã được train
        X: Feature matrix
        y: Target labels
        cv_folds: Số folds
        scoring: Scoring method

    Returns:
        Dictionary với CV results
    """
    from sklearn.model_selection import cross_validate

    # Cross validate
    cv_results = cross_validate(
        model, X, y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    return {
        'train_scores': cv_results['train_score'],
        'test_scores': cv_results['test_score'],
        'train_mean': cv_results['train_score'].mean(),
        'train_std': cv_results['train_score'].std(),
        'test_mean': cv_results['test_score'].mean(),
        'test_std': cv_results['test_score'].std(),
        'cv_results': cv_results
    }


def walk_forward_validation(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    train_size: float = 0.7,
    test_size: float = 0.3
) -> Dict[str, Any]:
    """
    Walk-forward validation để test model stability

    Args:
        model_class: Class của model để train
        X: Feature matrix
        y: Target labels
        n_splits: Số split cho validation
        train_size: Tỷ lệ train set
        test_size: Tỷ lệ test set

    Returns:
        Dictionary với validation results
    """
    n_samples = len(X)
    split_size = n_samples // n_splits

    results = []
    train_scores = []
    test_scores = []

    for i in range(n_splits - 1):
        train_start = 0
        train_end = int((i + 1) * split_size * train_size)
        test_start = train_end
        test_end = min(int(train_end + split_size * test_size), n_samples)

        if test_end <= train_end:
            continue

        X_train, X_test = X[train_start:train_end], X[test_start:test_end]
        y_train, y_test = y[train_start:train_end], y[test_start:test_end]

        # Train model
        model = model_class()
        model.fit(X_train, y_train)

        # Test
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

        results.append({
            'split': i,
            'train_score': train_score,
            'test_score': test_score,
            'train_size': train_end - train_start,
            'test_size': test_end - test_start
        })

    return {
        'results': results,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'train_mean': np.mean(train_scores),
        'train_std': np.std(train_scores),
        'test_mean': np.mean(test_scores),
        'test_std': np.std(test_scores),
        'stability_ratio': np.mean(test_scores) / (np.mean(train_scores) + 1e-8)
    }


def save_optimization_results(results: Dict[str, Any], filepath: str):
    """Lưu kết quả optimization"""
    joblib.dump(results, filepath)


def load_optimization_results(filepath: str) -> Dict[str, Any]:
    """Load kết quả optimization"""
    return joblib.load(filepath)


# Quick test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([-1, 0, 1], n_samples)

    # Test optimizer
    optimizer = MOEHyperparameterOptimizer()

    print("Optimizing Trend Expert...")
    trend_results = optimizer.optimize_expert_hyperparameters(
        X, y, expert_type="trend", n_iter=20
    )
    print(f"Trend Expert Best Score: {trend_results['best_score']:.4f}")

    print("Optimizing Range Expert...")
    range_results = optimizer.optimize_expert_hyperparameters(
        X, y, expert_type="range", n_iter=20
    )
    print(f"Range Expert Best Score: {range_results['best_score']:.4f}")

    print("Optimizing Volatility Expert...")
    vol_results = optimizer.optimize_expert_hyperparameters(
        X, y, expert_type="volatility", n_iter=20
    )
    print(f"Volatility Expert Best Score: {vol_results['best_score']:.4f}")

    print(f"\nBest Parameters: {optimizer.get_best_params()}")
    print(f"Best Scores: {optimizer.get_best_scores()}")