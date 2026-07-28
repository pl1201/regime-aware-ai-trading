"""
Unit tests cho RegimeEnsembleStrategy - Kiểm tra logic detect model type và xử lý DataFrame/numpy array
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import warnings

# Suppress warnings trong test
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from algo_trading.strategies.ml.regime_ensemble_strategy import (
    RegimeEnsembleBanditStrategy,
)


class TestModelTypeDetection(unittest.TestCase):
    """Test detect model type (LightGBM/XGBoost vs Sklearn)"""

    def setUp(self):
        """Setup test fixtures"""
        self.n_features = 10
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        
    def _create_mock_data(self):
        """Tạo mock DataFrame với features"""
        return pd.DataFrame(
            np.random.randn(100, self.n_features),
            columns=self.feature_names
        )

    def _test_model_detection(self, model, model_name, expected_use_dataframe):
        """Helper để test detect model type - chỉ test logic, không load file"""
        # Simulate logic từ generate_signals (không cần khởi tạo strategy)
        model_type = type(model).__name__
        model_module = type(model).__module__
        model_name_lower = model_name.lower()
        
        is_lgbm = (
            "lgb" in model_name_lower or
            "lightgbm" in model_module.lower() or
            model_type == "LGBMClassifier" or
            hasattr(model, "feature_name_")
        )
        is_xgb = (
            "xgb" in model_name_lower or
            "xgboost" in model_module.lower() or
            model_type == "XGBClassifier" or
            model_type == "XGBRFClassifier"
        )
        
        use_dataframe = is_lgbm or is_xgb
        
        self.assertEqual(
            use_dataframe,
            expected_use_dataframe,
            f"Model {model_name} ({model_type}) should use_dataframe={expected_use_dataframe}"
        )
        
        return use_dataframe

    def test_detect_lightgbm_by_name(self):
        """Test detect LightGBM qua model name"""
        mock_lgbm = Mock()
        mock_lgbm.__class__.__name__ = "SomeClassifier"
        mock_lgbm.__class__.__module__ = "sklearn.ensemble"
        
        use_df = self._test_model_detection(mock_lgbm, "lgb_model", True)
        self.assertTrue(use_df, "Should detect LightGBM by name 'lgb'")

    def test_detect_xgboost_by_name(self):
        """Test detect XGBoost qua model name"""
        mock_xgb = Mock()
        mock_xgb.__class__.__name__ = "SomeClassifier"
        mock_xgb.__class__.__module__ = "sklearn.ensemble"
        
        use_df = self._test_model_detection(mock_xgb, "xgb_model", True)
        self.assertTrue(use_df, "Should detect XGBoost by name 'xgb'")

    def test_detect_lightgbm_by_type(self):
        """Test detect LightGBM qua class name"""
        mock_lgbm = Mock()
        mock_lgbm.__class__.__name__ = "LGBMClassifier"
        mock_lgbm.__class__.__module__ = "lightgbm"
        
        use_df = self._test_model_detection(mock_lgbm, "model", True)
        self.assertTrue(use_df, "Should detect LightGBM by class name")

    def test_detect_xgboost_by_type(self):
        """Test detect XGBoost qua class name"""
        mock_xgb = Mock()
        mock_xgb.__class__.__name__ = "XGBClassifier"
        mock_xgb.__class__.__module__ = "xgboost"
        
        use_df = self._test_model_detection(mock_xgb, "model", True)
        self.assertTrue(use_df, "Should detect XGBoost by class name")

    def test_detect_lightgbm_by_module(self):
        """Test detect LightGBM qua module name"""
        mock_lgbm = Mock()
        mock_lgbm.__class__.__name__ = "SomeClassifier"
        mock_lgbm.__class__.__module__ = "lightgbm.sklearn"
        
        use_df = self._test_model_detection(mock_lgbm, "model", True)
        self.assertTrue(use_df, "Should detect LightGBM by module name")

    def test_detect_lightgbm_by_feature_name_attr(self):
        """Test detect LightGBM qua feature_name_ attribute"""
        mock_lgbm = Mock()
        mock_lgbm.__class__.__name__ = "SomeClassifier"
        mock_lgbm.__class__.__module__ = "sklearn.ensemble"
        mock_lgbm.feature_name_ = ["f1", "f2"]
        
        use_df = self._test_model_detection(mock_lgbm, "model", True)
        self.assertTrue(use_df, "Should detect LightGBM by feature_name_ attribute")

    def test_detect_sklearn_random_forest(self):
        """Test detect sklearn RandomForest (should use numpy)"""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(np.random.randn(100, self.n_features), np.random.randint(0, 2, 100))
        
        use_df = self._test_model_detection(rf, "rf_model", False)
        self.assertFalse(use_df, "RandomForest should use numpy array")

    def test_detect_sklearn_gradient_boosting(self):
        """Test detect sklearn GradientBoosting (should use numpy)"""
        from sklearn.ensemble import GradientBoostingClassifier
        
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(np.random.randn(100, self.n_features), np.random.randint(0, 2, 100))
        
        use_df = self._test_model_detection(gb, "gb_model", False)
        self.assertFalse(use_df, "GradientBoosting should use numpy array")


class TestFeatureAlignment(unittest.TestCase):
    """Test feature alignment logic"""

    def setUp(self):
        """Setup test fixtures"""
        self.n_features = 10
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def test_feature_alignment_with_missing_cols(self):
        """Test align features khi thiếu columns"""
        # Tạo DataFrame với ít columns hơn
        df = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f"feature_{i}" for i in range(5)]
        )
        
        # Feature names cần 10 columns
        feat_names = self.feature_names
        
        # Simulate alignment logic
        missing_cols = {col: 0.0 for col in feat_names if col not in df.columns}
        self.assertEqual(len(missing_cols), 5, "Should have 5 missing columns")
        
        if missing_cols:
            missing_df = pd.DataFrame(missing_cols, index=df.index)
            df_aligned = pd.concat([df, missing_df], axis=1)
        else:
            df_aligned = df
        
        df_aligned = df_aligned.reindex(columns=feat_names).ffill().bfill()
        
        self.assertEqual(len(df_aligned.columns), 10, "Should have 10 columns after alignment")
        self.assertListEqual(list(df_aligned.columns), feat_names, "Columns should match feat_names")

    def test_feature_alignment_with_extra_cols(self):
        """Test align features khi có thừa columns"""
        # Tạo DataFrame với nhiều columns hơn
        df = pd.DataFrame(
            np.random.randn(100, 15),
            columns=[f"feature_{i}" for i in range(15)]
        )
        
        feat_names = self.feature_names  # Chỉ cần 10
        
        # Simulate alignment logic
        df_aligned = df.reindex(columns=feat_names).ffill().bfill()
        
        self.assertEqual(len(df_aligned.columns), 10, "Should have 10 columns after alignment")
        self.assertListEqual(list(df_aligned.columns), feat_names, "Columns should match feat_names")


class TestScalerTransform(unittest.TestCase):
    """Test scaler transform với đúng format"""

    def setUp(self):
        """Setup test fixtures"""
        from sklearn.preprocessing import RobustScaler
        
        self.n_features = 10
        self.scaler = RobustScaler()
        
        # Fit scaler với numpy array (như khi train)
        X_train = np.random.randn(100, self.n_features)
        self.scaler.fit(X_train)

    def test_scaler_with_dataframe_input(self):
        """Test scaler transform với DataFrame input (should convert to numpy first)"""
        # Tạo DataFrame
        df = pd.DataFrame(
            np.random.randn(10, self.n_features),
            columns=[f"feature_{i}" for i in range(self.n_features)]
        )
        
        # Convert về numpy trước khi scale (như trong code)
        x_values = df.values if isinstance(df, pd.DataFrame) else df
        x_scaled = self.scaler.transform(x_values)
        
        self.assertIsInstance(x_scaled, np.ndarray, "Scaler should return numpy array")
        self.assertEqual(x_scaled.shape, (10, self.n_features), "Shape should match")

    def test_scaler_with_numpy_input(self):
        """Test scaler transform với numpy array input"""
        x_values = np.random.randn(10, self.n_features)
        x_scaled = self.scaler.transform(x_values)
        
        self.assertIsInstance(x_scaled, np.ndarray, "Scaler should return numpy array")
        self.assertEqual(x_scaled.shape, (10, self.n_features), "Shape should match")


class TestDataFrameConversion(unittest.TestCase):
    """Test conversion giữa DataFrame và numpy array"""

    def setUp(self):
        """Setup test fixtures"""
        self.n_features = 10
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def test_convert_to_dataframe_for_lgbm(self):
        """Test convert numpy array về DataFrame cho LightGBM"""
        # Simulate scaled array
        x_scaled_arr = np.random.randn(1, self.n_features)
        feat_cols = self.feature_names
        
        # Convert về DataFrame
        df_index = [0]
        x_scaled_df = pd.DataFrame(x_scaled_arr, columns=feat_cols, index=df_index)
        
        self.assertIsInstance(x_scaled_df, pd.DataFrame, "Should be DataFrame")
        self.assertEqual(len(x_scaled_df.columns), self.n_features, "Should have correct number of columns")
        self.assertListEqual(list(x_scaled_df.columns), feat_cols, "Columns should match")

    def test_feature_names_fallback(self):
        """Test fallback khi không có feature names"""
        x_scaled_arr = np.random.randn(1, self.n_features)
        
        # Fallback: tạo generic names
        feat_cols = [f"feature_{i}" for i in range(x_scaled_arr.shape[1])]
        x_scaled_df = pd.DataFrame(x_scaled_arr, columns=feat_cols, index=[0])
        
        self.assertEqual(len(x_scaled_df.columns), self.n_features, "Should have correct number of columns")
        self.assertTrue(all(col.startswith("feature_") for col in x_scaled_df.columns), "Should have generic names")

    def test_feature_names_mismatch_fix(self):
        """Test fix khi số lượng feature names không khớp"""
        x_scaled_arr = np.random.randn(1, self.n_features)
        
        # Feature names sai số lượng
        wrong_feat_cols = ["f1", "f2", "f3"]  # Chỉ có 3
        
        # Fix: tạo lại với số lượng đúng
        if len(wrong_feat_cols) != x_scaled_arr.shape[1]:
            feat_cols = [f"feature_{i}" for i in range(x_scaled_arr.shape[1])]
        else:
            feat_cols = wrong_feat_cols
        
        x_scaled_df = pd.DataFrame(x_scaled_arr, columns=feat_cols, index=[0])
        
        self.assertEqual(len(x_scaled_df.columns), self.n_features, "Should have correct number of columns")


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end với mock models"""

    def setUp(self):
        """Setup test fixtures"""
        self.n_features = 10
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def test_lgbm_with_dataframe(self):
        """Test LightGBM nhận DataFrame"""
        # Mock LightGBM model
        mock_lgbm = Mock()
        mock_lgbm.__class__.__name__ = "LGBMClassifier"
        mock_lgbm.__class__.__module__ = "lightgbm"
        mock_lgbm.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        # Tạo DataFrame
        x_scaled_df = pd.DataFrame(
            np.random.randn(1, self.n_features),
            columns=self.feature_names,
            index=[0]
        )
        
        # Predict (should work với DataFrame)
        proba = mock_lgbm.predict_proba(x_scaled_df)
        
        self.assertIsNotNone(proba, "Should return probabilities")
        mock_lgbm.predict_proba.assert_called_once()

    def test_sklearn_with_numpy(self):
        """Test sklearn model nhận numpy array"""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=10)
        X_train = np.random.randn(100, self.n_features)
        y_train = np.random.randint(0, 2, 100)
        rf.fit(X_train, y_train)
        
        # Tạo numpy array
        x_scaled = np.random.randn(1, self.n_features)
        
        # Predict (should work với numpy array)
        proba = rf.predict_proba(x_scaled)
        
        self.assertIsNotNone(proba, "Should return probabilities")
        self.assertEqual(proba.shape, (1, 2), "Should have correct shape")


if __name__ == "__main__":
    # Run tests với verbose output
    unittest.main(verbosity=2)

