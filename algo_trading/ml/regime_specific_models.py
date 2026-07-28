"""
Regime-Specific Models Module

Triển khai phương pháp train và predict với models riêng cho từng regime
theo mô tả trong PHUONG_PHAP_LUONG_HOA.md section 9.3.

Ý tưởng:
- Chia training data theo regime
- Train một model riêng cho mỗi regime: f_r cho regime r
- Trong inference: 
  - Xác định regime hiện tại r_t bằng HMM
  - Sử dụng model tương ứng f_{r_t} để predict
  - Hoặc dùng weighted ensemble: Σ_r P(s_t = r) × f_r(x_t)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
from pathlib import Path

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from sklearn.utils.class_weight import compute_class_weight
    HAS_CLASS_WEIGHT = True
except ImportError:
    HAS_CLASS_WEIGHT = False


# ============================================================================
# COMPATIBILITY FIXES
# ============================================================================

# Fix for CatBoost compatibility with newer scikit-learn
if HAS_CAT:
    class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
        """
        Wrapper for CatBoostClassifier to fix sklearn compatibility issues.
        """
        def __init__(self, **kwargs):
            self.catboost = cb.CatBoostClassifier(**kwargs)
            self._estimator_type = "classifier"

        def fit(self, X, y, **kwargs):
            return self.catboost.fit(X, y, **kwargs)

        def predict(self, X):
            return self.catboost.predict(X)

        def predict_proba(self, X):
            return self.catboost.predict_proba(X)

        def get_params(self, deep=True):
            return self.catboost.get_params(deep=deep)

        def set_params(self, **params):
            self.catboost.set_params(**params)
            return self

        @property
        def classes_(self):
            return self.catboost.classes_

        @property
        def feature_importances_(self):
            return self.catboost.feature_importances_

        def __sklearn_tags__(self):
            """Fix for sklearn compatibility"""
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


class RegimeSpecificModels:
    """
    Class để quản lý và train models riêng cho từng regime
    """
    
    REGIME_NAMES = ['trending', 'ranging', 'volatile', 'calm']
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        model_params: Optional[Dict[str, Any]] = None,
        use_weighted_ensemble: bool = True
    ):

        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_weighted_ensemble = use_weighted_ensemble
        
        # Value: trained model
        self.models: Dict[Union[int, str], Any] = {}
        
        # Scalers cho từng regime
        self.scalers: Dict[Union[int, str], Any] = {}
        
        # Feature names (để đảm bảo consistency)
        self.feature_names: Optional[List[str]] = None
        
        # Flag để biết đã train chưa
        self.is_trained = False

        # Mapping label cho từng regime (để encode [-1,0,1] -> [0,1,2] cho XGBoost)
        # classes_mapping[regime_id] = np.array(sorted_unique_labels)
        self.classes_mapping: Dict[Union[int, str], np.ndarray] = {}
    
    def _create_model(self, regime_id: int, class_weight_dict: Optional[Dict[int, float]] = None) -> Any:
        """
        Tạo một model instance cho regime cụ thể
        
        Args:
            regime_id: ID của regime (0-3)
            class_weight_dict: Dictionary mapping class labels to weights (optional)
        
        Returns:
            Model instance
        """
        if self.model_type == 'xgboost':
            if not HAS_XGB:
                raise ImportError("xgboost not available")
            # Detect number of classes from y_train if available
            # Default to multi-class, will be adjusted during fit if needed
            params = {
                'objective': 'multi:softprob',  # Multi-class by default
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'use_label_encoder': False,
                **self.model_params
            }
            # Add class weights if provided (XGBoost uses sample_weight parameter during fit)
            # Note: XGBoost doesn't support class_weight parameter directly, we'll use sample_weight in fit()
            return xgb.XGBClassifier(**params)
        
        elif self.model_type == 'lightgbm':
            if not HAS_LGB:
                raise ImportError("lightgbm not available")
            params = {
                'objective': 'multiclass',  # Multi-class by default
                'metric': 'multi_logloss',
                'random_state': 42,
                'verbosity': -1,
                **self.model_params
            }
            # Use custom class_weight_dict if provided, otherwise use 'balanced'
            if class_weight_dict is not None:
                params['class_weight'] = class_weight_dict
            else:
                params['class_weight'] = 'balanced'
            return lgb.LGBMClassifier(**params)
        
        elif self.model_type == 'catboost':
            if not HAS_CAT:
                raise ImportError("catboost not available")
            params = {
                'objective': 'MultiClass',  # Multi-class
                'random_state': 42,
                'verbose': False,
                **self.model_params
            }
            # CatBoost uses auto_class_weights, keep it as 'Balanced' for now
            # Note: CatBoost doesn't support custom class_weight_dict directly
            if 'auto_class_weights' not in params:
                params['auto_class_weights'] = 'Balanced'
            return CatBoostClassifierWrapper(**params)
        
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            params = {
                'random_state': 42,
                'n_jobs': -1,
                **self.model_params
            }
            # Use custom class_weight_dict if provided, otherwise use 'balanced'
            if class_weight_dict is not None:
                params['class_weight'] = class_weight_dict
            else:
                params['class_weight'] = 'balanced'
            return RandomForestClassifier(**params)
        
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            params = {
                'random_state': 42,
                **self.model_params
            }
            # GradientBoosting doesn't support class_weight, we'll use sample_weight in fit()
            return GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def split_data_by_regime(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_ids: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Chia data theo regime
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            regime_ids: Regime IDs [n_samples] (0-3)
            feature_names: Optional feature names
        
        Returns:
            Dict với key là regime_id, value là (X_regime, y_regime)
        """
        regime_data = {}
        
        unique_regimes = np.unique(regime_ids)
        for regime_id in unique_regimes:
            mask = regime_ids == regime_id
            X_regime = X[mask]
            y_regime = y[mask]
            
            if len(X_regime) > 0:  # Chỉ thêm nếu có data
                regime_data[int(regime_id)] = (X_regime, y_regime)
        
        return regime_data
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        regime_ids: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'RegimeSpecificModels':

        # Convert DataFrame to numpy nếu cần
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Chia data theo regime
        regime_data = self.split_data_by_regime(X, y, regime_ids, feature_names)
        
        print(f"📊 Training regime-specific models...")
        print(f"   Found {len(regime_data)} regimes with data")
        
        # Train một model cho mỗi regime
        for regime_id, (X_regime, y_regime) in regime_data.items():
            regime_name = self.REGIME_NAMES[regime_id] if regime_id < len(self.REGIME_NAMES) else f"regime_{regime_id}"
            
            print(f"   Training {regime_name} (ID={regime_id}): {len(X_regime)} samples")
            
            # Tạo scaler cho regime này
            scaler = RobustScaler()
            X_regime_scaled = scaler.fit_transform(X_regime)
            self.scalers[regime_id] = scaler
            
            # Tính class weights cho regime này với boost factor cho non-zero classes
            unique_classes_regime = np.unique(y_regime)
            n_classes_regime = len(unique_classes_regime)
            
            # Tính base class weights
            if HAS_CLASS_WEIGHT and n_classes_regime > 1:
                base_class_weights = compute_class_weight('balanced', classes=unique_classes_regime, y=y_regime)
                class_weight_dict_regime = {int(cls): float(weight) for cls, weight in zip(unique_classes_regime, base_class_weights)}
                
                # Boost non-zero classes (LONG/SHORT) với factor 4.0 để tăng trọng số
                boost_factor = 4.0
                for cls in list(class_weight_dict_regime.keys()):
                    if cls != 0:
                        class_weight_dict_regime[cls] *= boost_factor
                
                print(f"      Class distribution: {dict(zip(*np.unique(y_regime, return_counts=True)))}")
                print(f"      Class weights (boosted ±1 x{boost_factor}): {class_weight_dict_regime}")
            else:
                class_weight_dict_regime = None
            
            # Tạo và train model với class weights
            model = self._create_model(regime_id, class_weight_dict_regime)
            
            # Encode labels cho XGBoost nếu cần ([-1,0,1] -> [0,1,2])
            y_regime_train = y_regime
            if self.model_type == 'xgboost' and HAS_XGB:
                classes_regime = np.unique(y_regime)
                classes_sorted = np.sort(classes_regime)
                self.classes_mapping[regime_id] = classes_sorted
                class_to_index = {int(cls): idx for idx, cls in enumerate(classes_sorted)}
                y_regime_train = np.array([class_to_index[int(lbl)] for lbl in y_regime])
            
            # Adjust objective based on number of classes
            if self.model_type == 'xgboost' and HAS_XGB:
                if n_classes_regime == 2:
                    model.set_params(objective='binary:logistic', eval_metric='logloss')
                else:
                    model.set_params(objective='multi:softprob', eval_metric='mlogloss')
            elif self.model_type == 'lightgbm' and HAS_LGB:
                if n_classes_regime == 2:
                    model.set_params(objective='binary', metric='binary_logloss')
                else:
                    model.set_params(objective='multiclass', metric='multi_logloss')
            elif self.model_type == 'catboost' and HAS_CAT:
                if n_classes_regime == 2:
                    model.set_params(objective='Logloss')
                else:
                    model.set_params(objective='MultiClass')
            
            # Tính sample weights cho XGBoost và GradientBoosting (không hỗ trợ class_weight trực tiếp)
            sample_weight_regime = None
            if self.model_type in ['xgboost', 'gradient_boosting'] and class_weight_dict_regime is not None:
                # Tạo sample_weight array từ class_weight_dict
                sample_weight_regime = np.array([class_weight_dict_regime.get(int(label), 1.0) for label in y_regime])
            
            # Convert to DataFrame nếu cần (cho LightGBM/XGBoost)
            if self.model_type in ['lightgbm', 'xgboost'] and feature_names is not None:
                X_regime_scaled_df = pd.DataFrame(X_regime_scaled, columns=feature_names)
                if sample_weight_regime is not None:
                    model.fit(X_regime_scaled_df, y_regime_train, sample_weight=sample_weight_regime)
                else:
                    model.fit(X_regime_scaled_df, y_regime_train)
            else:
                if sample_weight_regime is not None:
                    model.fit(X_regime_scaled, y_regime_train, sample_weight=sample_weight_regime)
                else:
                    model.fit(X_regime_scaled, y_regime_train)
            
            self.models[regime_id] = model
        
        self.is_trained = True
        print(f"✅ Trained {len(self.models)} regime-specific models")
        
        return self
    
    def predict_single_regime(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regime_id: int
    ) -> np.ndarray:
        """
        Predict sử dụng model của một regime cụ thể
        
        Args:
            X: Feature matrix [n_samples, n_features] hoặc DataFrame
            regime_id: Regime ID (0-3)
        
        Returns:
            Predictions [n_samples]
        """
        if not self.is_trained:
            raise ValueError("Models chưa được train. Gọi fit() trước.")
        
        if regime_id not in self.models:
            # Fallback: dùng model của regime đầu tiên có sẵn
            available_regimes = list(self.models.keys())
            if len(available_regimes) == 0:
                raise ValueError("Không có model nào được train")
            regime_id = available_regimes[0]
            warnings.warn(f"Regime {regime_id} không có model, dùng model của regime {available_regimes[0]}")
        

        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                missing_data = {col: 0.0 for col in missing_cols}
                X = pd.concat([X, pd.DataFrame(missing_data, index=X.index)], axis=1)
            X = X.reindex(columns=self.feature_names)
            X = X.values
        elif isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        scaler = self.scalers[regime_id]
        X_scaled = scaler.transform(X)
        
        # Predict
        model = self.models[regime_id]
        
        # Convert to DataFrame nếu cần
        if self.model_type in ['lightgbm', 'xgboost'] and self.feature_names is not None:
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            preds = model.predict(X_scaled_df)
        else:
            preds = model.predict(X_scaled)
        
        # Decode labels lại về không gian gốc nếu dùng XGBoost với encoding
        if self.model_type == 'xgboost' and regime_id in self.classes_mapping:
            classes = self.classes_mapping[regime_id]
            preds = np.array([classes[int(idx)] for idx in preds])
        
        return preds
    
    def predict_proba_single_regime(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regime_id: int
    ) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Models chưa được train. Gọi fit() trước.")
        
        if regime_id not in self.models:
            # Fallback: dùng model của regime đầu tiên có sẵn
            available_regimes = list(self.models.keys())
            if len(available_regimes) == 0:
                raise ValueError("Không có model nào được train")
            regime_id = available_regimes[0]
            warnings.warn(f"Regime {regime_id} không có model, dùng model của regime {available_regimes[0]}")
        
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                missing_data = {col: 0.0 for col in missing_cols}
                X = pd.concat([X, pd.DataFrame(missing_data, index=X.index)], axis=1)
            X = X.reindex(columns=self.feature_names)
            X = X.values
        elif isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        scaler = self.scalers[regime_id]
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        model = self.models[regime_id]
        
        # Convert to DataFrame nếu cần
        if self.model_type in ['lightgbm', 'xgboost'] and self.feature_names is not None:
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            probs_raw = model.predict_proba(X_scaled_df)
        else:
            probs_raw = model.predict_proba(X_scaled)

        try:
            classes = getattr(model, "classes_", None)
            if classes is None:
                # Fallback: nếu không có thông tin classes, giả định binary đã đúng dạng
                if probs_raw.shape[1] == 2:
                    return probs_raw
                # Nếu nhiều hơn 2 class mà không biết mapping, lấy 2 cột cuối làm [short, long]
                return probs_raw[:, -2:]

            classes = np.array(classes)
            # Với XGBoost đã encode, dùng mapping lưu trong classes_mapping
            if self.model_type == "xgboost" and regime_id in self.classes_mapping:
                classes = self.classes_mapping[regime_id]

            idx_short = np.where(classes == -1)[0]
            idx_long = np.where(classes == 1)[0]

            if probs_raw.ndim == 1:
                probs_raw = probs_raw.reshape(-1, 1)

            if len(idx_short) > 0:
                p_short = probs_raw[:, idx_short].sum(axis=1)
            else:
                p_short = np.zeros(probs_raw.shape[0])

            if len(idx_long) > 0:
                p_long = probs_raw[:, idx_long].sum(axis=1)
            else:
                p_long = np.zeros(probs_raw.shape[0])

            probabilities = np.stack([p_short, p_long], axis=1)
            return probabilities
        except Exception:
            # Nếu có bất kỳ lỗi nào trong mapping, fallback về probs_raw (nếu đã là 2 cột)
            if probs_raw.shape[1] == 2:
                return probs_raw
            # Ngược lại, cắt về 2 cột đầu
            return probs_raw[:, :2]
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regime_ids: Optional[np.ndarray] = None,
        regime_probabilities: Optional[np.ndarray] = None
    ) -> np.ndarray:

        if not self.is_trained:
            raise ValueError("Models chưa được train. Gọi fit() trước.")
        
        n_samples = X.shape[0]
        
        # Mode 1: Weighted ensemble (nếu có regime_probabilities)
        if self.use_weighted_ensemble and regime_probabilities is not None:
            # Weighted ensemble: Σ_r P(s_t = r) × f_r(x_t)
            predictions = np.zeros(n_samples)
            
            for regime_id in self.models.keys():
                regime_probs = regime_probabilities[:, regime_id]
                regime_preds = self.predict_single_regime(X, regime_id)
                predictions += regime_probs * regime_preds
            
            # Convert to binary predictions (0 or 1)
            predictions = (predictions > 0.5).astype(int)
            return predictions
        
        # Mode 2: Single regime (dùng regime có xác suất cao nhất)
        if regime_ids is None:
            raise ValueError("Cần cung cấp regime_ids hoặc regime_probabilities")
        
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            regime_id = int(regime_ids[i])
            predictions[i] = self.predict_single_regime(X[i:i+1], regime_id)[0]
        
        return predictions.astype(int)
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regime_ids: Optional[np.ndarray] = None,
        regime_probabilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict probabilities sử dụng regime-specific models
        
        Args:
            X: Feature matrix [n_samples, n_features] hoặc DataFrame
            regime_ids: Regime IDs [n_samples] (0-3)
            regime_probabilities: Regime probabilities [n_samples, n_regimes]
        
        Returns:
            Probabilities [n_samples, n_classes]
        """
        if not self.is_trained:
            raise ValueError("Models chưa được train. Gọi fit() trước.")
        
        n_samples = X.shape[0]
        n_classes = 2  # Binary classification (có thể mở rộng sau)
        
        # Mode 1: Weighted ensemble
        if self.use_weighted_ensemble and regime_probabilities is not None:
            probabilities = np.zeros((n_samples, n_classes))
            
            for regime_id in self.models.keys():
                regime_probs = regime_probabilities[:, regime_id]
                regime_probas = self.predict_proba_single_regime(X, regime_id)
                probabilities += regime_probs.reshape(-1, 1) * regime_probas
            
            return probabilities
        
        # Mode 2: Single regime
        if regime_ids is None:
            raise ValueError("Cần cung cấp regime_ids hoặc regime_probabilities")
        
        probabilities = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            regime_id = int(regime_ids[i])
            probabilities[i] = self.predict_proba_single_regime(X[i:i+1], regime_id)[0]
        
        return probabilities

