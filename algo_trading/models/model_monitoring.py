"""
Model Monitoring & Auto-Retrain Module

Tự động phát hiện model degradation và trigger retrain để:
- Bảo vệ bot khỏi performance drop
- Luôn duy trì model optimal
- Giảm thời gian monitor manual
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
import warnings
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta

try:
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Install with: pip install scipy")


class ModelMonitor:
    """
    Monitor model performance và tự động trigger retrain khi cần
    """

    def __init__(
        self,
        model_name: str = "trading_model",
        performance_threshold: float = 0.1,  # 10% drop in performance
        drift_threshold: float = 0.1,  # KL-divergence threshold
        min_samples_for_monitoring: int = 50,
        auto_retrain_callback: Optional[Callable] = None
    ):
        """
        Args:
            model_name: Tên model để tracking
            performance_threshold: Ngưỡng performance drop để trigger retrain
            drift_threshold: Ngưỡng distribution drift để trigger retrain
            min_samples_for_monitoring: Số samples tối thiểu để monitor
            auto_retrain_callback: Function được gọi khi cần retrain
        """
        self.model_name = model_name
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.min_samples_for_monitoring = min_samples_for_monitoring
        self.auto_retrain_callback = auto_retrain_callback

        # Baseline metrics (được lưu từ training)
        self.baseline_performance = None
        self.baseline_prediction_distribution = None
        self.baseline_feature_importance = None

        # Monitoring history
        self.monitoring_history = []
        self.last_monitor_time = None
        self.is_monitoring = False

        # Alert system
        self.alerts = []
        self.alert_thresholds = {
            'performance_drop': performance_threshold,
            'drift_detected': drift_threshold,
            'low_confidence': 0.3,
            'high_volatility': 2.0
        }

    def set_baseline(
        self,
        performance_metrics: Dict[str, float],
        prediction_distribution: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ):
        """
        Set baseline metrics từ training

        Args:
            performance_metrics: Dict với metrics từ backtest
            prediction_distribution: Distribution của predictions
            feature_importance: Feature importance từ training
        """
        self.baseline_performance = performance_metrics.copy()
        self.baseline_prediction_distribution = prediction_distribution
        self.baseline_feature_importance = feature_importance
        self.is_monitoring = True

    def check_performance_degradation(
        self,
        recent_performance: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Check performance degradation

        Args:
            recent_performance: Performance metrics từ live trading

        Returns:
            Dict với kết quả kiểm tra
        """
        if not self.baseline_performance or not self.is_monitoring:
            return {'status': 'no_baseline', 'needs_retrain': False}

        degradation_detected = False
        degradation_details = {}

        for metric_name, baseline_value in self.baseline_performance.items():
            if metric_name in recent_performance:
                current_value = recent_performance[metric_name]
                # Tính % change (âm = giảm performance)
                if baseline_value != 0:
                    change_pct = (current_value - baseline_value) / abs(baseline_value)
                else:
                    change_pct = current_value if current_value != 0 else 0

                degradation_details[metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change_pct,
                    'degraded': change_pct < -self.performance_threshold
                }

                if change_pct < -self.performance_threshold:
                    degradation_detected = True

        return {
            'status': 'degraded' if degradation_detected else 'ok',
            'needs_retrain': degradation_detected,
            'details': degradation_details
        }

    def check_prediction_drift(
        self,
        current_predictions: np.ndarray,
        prediction_labels: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Check prediction distribution drift

        Args:
            current_predictions: Predictions từ live trading
            prediction_labels: Labels cho predictions (ví dụ: ['LONG', 'NEUTRAL', 'SHORT'])

        Returns:
            Dict với kết quả kiểm tra
        """
        if self.baseline_prediction_distribution is None or not HAS_SCIPY:
            return {'status': 'no_baseline_or_scipy', 'needs_retrain': False}

        # Convert predictions to distribution
        if len(current_predictions.shape) > 1:
            # Nếu là probabilities, lấy argmax để có class predictions
            if current_predictions.shape[1] > 1:
                class_predictions = np.argmax(current_predictions, axis=1)
            else:
                class_predictions = current_predictions.flatten()
        else:
            class_predictions = current_predictions

        # Tính distribution hiện tại
        unique_classes = np.unique(class_predictions)
        current_dist = np.array([
            np.sum(class_predictions == cls) / len(class_predictions)
            for cls in unique_classes
        ])

        # Align với baseline distribution
        baseline_classes = np.arange(len(self.baseline_prediction_distribution))
        aligned_current_dist = np.zeros_like(self.baseline_prediction_distribution)

        for i, cls in enumerate(baseline_classes):
            if cls in unique_classes:
                idx = np.where(unique_classes == cls)[0]
                if len(idx) > 0:
                    aligned_current_dist[i] = current_dist[idx[0]]

        # Tính KL-divergence
        try:
            # Thêm epsilon để tránh log(0)
            baseline_dist_safe = np.clip(self.baseline_prediction_distribution, 1e-10, 1.0)
            current_dist_safe = np.clip(aligned_current_dist, 1e-10, 1.0)

            # KL divergence: D(P||Q) = sum(P * log(P/Q))
            kl_div = np.sum(baseline_dist_safe * np.log(baseline_dist_safe / current_dist_safe))
        except Exception as e:
            warnings.warn(f"Error calculating KL divergence: {e}")
            kl_div = 0.0

        drift_detected = kl_div > self.drift_threshold

        return {
            'status': 'drifted' if drift_detected else 'ok',
            'needs_retrain': drift_detected,
            'kl_divergence': kl_div,
            'baseline_distribution': self.baseline_prediction_distribution.tolist(),
            'current_distribution': aligned_current_dist.tolist(),
            'classes': prediction_labels or [str(i) for i in range(len(aligned_current_dist))]
        }

    def check_feature_importance_drift(
        self,
        current_feature_importance: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Check feature importance drift

        Args:
            current_feature_importance: Feature importance từ live data

        Returns:
            Dict với kết quả kiểm tra
        """
        if not self.baseline_feature_importance:
            return {'status': 'no_baseline', 'needs_retrain': False}

        # Tính correlation giữa baseline và current
        baseline_features = set(self.baseline_feature_importance.keys())
        current_features = set(current_feature_importance.keys())
        common_features = baseline_features.intersection(current_features)

        if len(common_features) < 5:  # Quá ít features chung
            return {'status': 'insufficient_overlap', 'needs_retrain': True}

        # Tính correlation
        baseline_values = []
        current_values = []

        for feature in common_features:
            baseline_values.append(self.baseline_feature_importance[feature])
            current_values.append(current_feature_importance[feature])

        baseline_values = np.array(baseline_values)
        current_values = np.array(current_values)

        # Normalize
        baseline_norm = baseline_values / (np.sum(baseline_values) + 1e-10)
        current_norm = current_values / (np.sum(current_values) + 1e-10)

        # Correlation
        correlation = np.corrcoef(baseline_norm, current_norm)[0, 1] if len(baseline_norm) > 1 else 1.0

        # Drift nếu correlation thấp
        drift_detected = correlation < 0.7

        return {
            'status': 'drifted' if drift_detected else 'ok',
            'needs_retrain': drift_detected,
            'correlation': correlation,
            'common_features': len(common_features),
            'baseline_features': len(baseline_features),
            'current_features': len(current_features)
        }

    def monitor(
        self,
        recent_performance: Dict[str, float],
        current_predictions: Optional[np.ndarray] = None,
        current_feature_importance: Optional[Dict[str, float]] = None,
        prediction_labels: Optional[List[str]] = None,
        additional_metrics: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Monitor tất cả aspects và quyết định có cần retrain không

        Args:
            recent_performance: Performance metrics từ live trading
            current_predictions: Predictions từ live trading
            current_feature_importance: Feature importance từ live data
            prediction_labels: Labels cho predictions
            additional_metrics: Metrics bổ sung

        Returns:
            Dict với kết quả monitoring và decision
        """
        if not self.is_monitoring:
            return {'status': 'not_monitoring', 'needs_retrain': False}

        # Check performance degradation
        perf_result = self.check_performance_degradation(recent_performance)

        # Check prediction drift
        drift_result = {'status': 'no_data', 'needs_retrain': False}
        if current_predictions is not None and len(current_predictions) >= self.min_samples_for_monitoring:
            drift_result = self.check_prediction_drift(current_predictions, prediction_labels)

        # Check feature importance drift
        feature_result = {'status': 'no_data', 'needs_retrain': False}
        if current_feature_importance:
            feature_result = self.check_feature_importance_drift(current_feature_importance)

        # Tổng hợp kết quả
        needs_retrain = (
            perf_result['needs_retrain'] or
            drift_result['needs_retrain'] or
            feature_result['needs_retrain']
        )

        # Log monitoring history
        monitoring_record = {
            'timestamp': datetime.now().isoformat(),
            'performance_check': perf_result,
            'drift_check': drift_result,
            'feature_check': feature_result,
            'needs_retrain': needs_retrain,
            'additional_metrics': additional_metrics or {}
        }

        self.monitoring_history.append(monitoring_record)
        self.last_monitor_time = datetime.now()

        # Trigger retrain nếu cần
        if needs_retrain and self.auto_retrain_callback:
            try:
                self.auto_retrain_callback()
                monitoring_record['retrain_triggered'] = True
            except Exception as e:
                warnings.warn(f"Auto-retrain failed: {e}")
                monitoring_record['retrain_triggered'] = False
                monitoring_record['retrain_error'] = str(e)

        return monitoring_record

    def add_alert(self, alert_type: str, message: str, severity: str = 'warning'):
        """
        Add alert để tracking

        Args:
            alert_type: Loại alert
            message: Nội dung alert
            severity: Mức độ nghiêm trọng ('info', 'warning', 'error')
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        self.alerts.append(alert)

    def get_monitoring_summary(self) -> Dict[str, any]:
        """
        Get summary của monitoring

        Returns:
            Dict với summary statistics
        """
        if not self.monitoring_history:
            return {'status': 'no_data', 'alerts': len(self.alerts)}

        # Tính các metrics
        total_checks = len(self.monitoring_history)
        retrain_triggers = sum(1 for record in self.monitoring_history if record.get('needs_retrain', False))
        performance_degradations = sum(
            1 for record in self.monitoring_history
            if record.get('performance_check', {}).get('needs_retrain', False)
        )
        drift_detections = sum(
            1 for record in self.monitoring_history
            if record.get('drift_check', {}).get('needs_retrain', False)
        )

        # Recent performance trend
        recent_history = self.monitoring_history[-10:] if len(self.monitoring_history) >= 10 else self.monitoring_history
        recent_retrains = sum(1 for record in recent_history if record.get('needs_retrain', False))

        return {
            'status': 'active',
            'total_monitoring_checks': total_checks,
            'retrain_triggers': retrain_triggers,
            'performance_degradations': performance_degradations,
            'drift_detections': drift_detections,
            'recent_retrain_trend': recent_retrains,
            'last_monitor_time': self.last_monitor_time.isoformat() if self.last_monitor_time else None,
            'alerts': len(self.alerts),
            'monitoring_since': self.monitoring_history[0]['timestamp'] if self.monitoring_history else None
        }

    def save_monitoring_state(self, filepath: str):
        """
        Save monitoring state để persist

        Args:
            filepath: Đường dẫn file để lưu
        """
        state = {
            'model_name': self.model_name,
            'baseline_performance': self.baseline_performance,
            'baseline_prediction_distribution': (
                self.baseline_prediction_distribution.tolist()
                if self.baseline_prediction_distribution is not None else None
            ),
            'baseline_feature_importance': self.baseline_feature_importance,
            'monitoring_history': self.monitoring_history,
            'alerts': self.alerts,
            'last_monitor_time': self.last_monitor_time.isoformat() if self.last_monitor_time else None
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_monitoring_state(self, filepath: str):
        """
        Load monitoring state từ file

        Args:
            filepath: Đường dẫn file để load
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.model_name = state.get('model_name', self.model_name)
            self.baseline_performance = state.get('baseline_performance')
            baseline_dist = state.get('baseline_prediction_distribution')
            self.baseline_prediction_distribution = (
                np.array(baseline_dist) if baseline_dist is not None else None
            )
            self.baseline_feature_importance = state.get('baseline_feature_importance')
            self.monitoring_history = state.get('monitoring_history', [])
            self.alerts = state.get('alerts', [])

            last_monitor_time = state.get('last_monitor_time')
            if last_monitor_time:
                self.last_monitor_time = datetime.fromisoformat(last_monitor_time)

            self.is_monitoring = True
        except Exception as e:
            warnings.warn(f"Failed to load monitoring state: {e}")


def create_model_monitor(
    model_name: str = "trading_model",
    performance_threshold: float = 0.1,
    drift_threshold: float = 0.1,
    auto_retrain_callback: Optional[Callable] = None
) -> ModelMonitor:
    """
    Convenience function để tạo model monitor

    Args:
        model_name: Tên model
        performance_threshold: Ngưỡng performance drop
        drift_threshold: Ngưỡng distribution drift
        auto_retrain_callback: Function để retrain

    Returns:
        ModelMonitor instance
    """
    return ModelMonitor(
        model_name=model_name,
        performance_threshold=performance_threshold,
        drift_threshold=drift_threshold,
        auto_retrain_callback=auto_retrain_callback
    )