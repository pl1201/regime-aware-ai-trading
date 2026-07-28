from __future__ import annotations


from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    cb = None  # type: ignore


class CatBoostWrapper(BaseEstimator, ClassifierMixin):

    _estimator_type = "classifier"

    def __init__(self, **catboost_params: Any) -> None:
        self.catboost_params: Dict[str, Any] = catboost_params
        self.catboost_model: Any = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # Core sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, **fit_params: Any) -> "CatBoostWrapper":
        if not HAS_CAT:
            raise ImportError("CatBoost chưa được cài đặt. Chạy: pip install catboost")

        params = dict(self.catboost_params)
        params.setdefault("verbose", False)
        
        # Đảm bảo có auto_class_weights nếu không được set
        if "auto_class_weights" not in params:
            params["auto_class_weights"] = "Balanced"

        self.catboost_model = cb.CatBoostClassifier(**params)
        self.catboost_model.fit(X, y, **fit_params)

        # Set attrs that sklearn cross_val_predict / stacking expect
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.catboost_model is None:
            raise ValueError("Model chưa được fit. Gọi fit() trước.")
        return self.catboost_model.predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.catboost_model is None:
            raise ValueError("Model chưa được fit. Gọi fit() trước.")
        return self.catboost_model.predict_proba(X)

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self.catboost_params)

    def set_params(self, **params: Any) -> "CatBoostWrapper":
        self.catboost_params.update(params)
        return self

    def __sklearn_tags__(self) -> Dict[str, Any]:
        """Fix for sklearn compatibility with newer versions"""
        return {
            'estimator_type': 'classifier',
            'requires_fit': True,
            'requires_y': True,
            'X_types': ['2darray'],
            'poor_score': True,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'multiclass_only': False,
            'binary_only': False,
            'requires_dense': [True, False],
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'preserves_dtype': [True],
            'requires_y': True,
            'poor_score': True,
            'stateless': False,
            'pairwise': False
        }


