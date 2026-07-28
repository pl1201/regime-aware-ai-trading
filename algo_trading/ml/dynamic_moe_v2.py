import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow components
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Using MLP instead of LSTM for RangeFinder.")

# Import multi-timeframe utilities
from algo_trading.ml.multi_timeframe import get_multi_timeframe_feature_names

class TrendDetector(BaseEstimator, ClassifierMixin):
    '''XGBoost với Focal Loss cho xu hướng mạnh'''

    def __init__(self, **kwargs):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,  # Cân bằng class
            random_state=42,
            **kwargs
        )

    def fit(self, X, y, sample_weight=None):
        # Tính trọng số Focal Loss đơn giản: tăng trọng số cho class 1 khi dự đoán sai
        if sample_weight is None:
            sample_weight = np.ones(len(y))

        # Huấn luyện model trước để có predict_proba
        self.model.fit(X, y, sample_weight=sample_weight)

        # Tăng trọng số cho các sample class 1 khó phân loại
        try:
            probas = self.model.predict_proba(X)[:, 1]
            focal_weights = sample_weight * (1 - probas)**2 * (y + 0.5)  # Tăng trọng số khi dự đoán sai class 1
            self.model.fit(X, y, sample_weight=focal_weights)
        except:
            # Nếu không thể tính focal weights, giữ nguyên model đã fit
            pass

        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

class RangeFinder(BaseEstimator, ClassifierMixin):
    '''LSTM cho thị trường ngang giá (nếu có TensorFlow) hoặc MLP'''

    def __init__(self, sequence_length=10, lstm_units=32, hidden_layers=[64, 32]):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = None
        self.use_lstm = TENSORFLOW_AVAILABLE

    def _prepare_sequences(self, X):
        '''Chuyển đổi dữ liệu thành chuỗi 10 nến'''
        if len(X) < self.sequence_length:
            return np.array([])

        X_seq = []
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i+self.sequence_length])
        return np.array(X_seq)

    def fit(self, X, y):
        if self.use_lstm:
            # Chuẩn bị dữ liệu chuỗi
            X_seq = self._prepare_sequences(X)
            if len(X_seq) == 0:
                raise ValueError("Không đủ dữ liệu để tạo chuỗi")

            y_seq = y[self.sequence_length-1:]

            # Xây dựng mô hình LSTM đơn giản
            self.model = Sequential([
                Input(shape=(self.sequence_length, X.shape[1])),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Huấn luyện với early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Chia train/val
            split = int(0.8 * len(X_seq))
            X_train, X_val = X_seq[:split], X_seq[split:]
            y_train, y_val = y_seq[:split], y_seq[split:]

            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
        else:
            # Sử dụng MLP nếu không có TensorFlow
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )

            self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X):
        if self.use_lstm:
            if len(X) < self.sequence_length:
                # Trả về xác suất trung tính nếu không đủ dữ liệu
                return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

            X_seq = self._prepare_sequences(X)
            if len(X_seq) == 0:
                # Trả về xác suất trung tính nếu không đủ dữ liệu
                return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

            probas = self.model.predict(X_seq, verbose=0).flatten()
            # Mở rộng dự đoán về kích thước ban đầu
            result = np.full(len(X), 0.5)
            result[self.sequence_length-1:] = probas

            # Trả về xác suất 2 chiều
            return np.column_stack([1 - result, result])
        else:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            probas = self.model.predict_proba(X_scaled)
            return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

class VolatilityBreakout(BaseEstimator, ClassifierMixin):
    '''CatBoost với các đặc trưng biến động'''

    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            loss_function='Logloss',
            random_state=42,
            verbose=False,
            **kwargs
        )

    def fit(self, X, y):
        # Thêm các đặc trưng biến động
        X_with_vol = X.copy()
        if isinstance(X, pd.DataFrame) and len(X) > 10:
            X_with_vol = X_with_vol.copy()  # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
            if 'atr_14' in X.columns:
                X_with_vol['atr_change'] = X['atr_14'].pct_change().fillna(0)
            if 'volume' in X.columns:
                X_with_vol['volume_ratio'] = X['volume'] / X['volume'].rolling(20).mean().fillna(1)
            if all(col in X.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                X_with_vol['bb_width'] = (X['bb_upper'] - X['bb_lower']) / X['bb_middle']
            if 'atr_change' in X_with_vol.columns:
                X_with_vol['volatility_regime'] = (X_with_vol['atr_change'].abs() > X_with_vol['atr_change'].abs().quantile(0.7)).astype(int)

        self.model.fit(X_with_vol, y)
        return self

    def predict_proba(self, X):
        X_with_vol = X.copy()
        if isinstance(X, pd.DataFrame) and len(X) > 10:
            X_with_vol = X_with_vol.copy()  # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
            if 'atr_14' in X.columns:
                X_with_vol['atr_change'] = X['atr_14'].pct_change().fillna(0)
            if 'volume' in X.columns:
                X_with_vol['volume_ratio'] = X['volume'] / X['volume'].rolling(20).mean().fillna(1)
            if all(col in X.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                X_with_vol['bb_width'] = (X['bb_upper'] - X['bb_lower']) / X['bb_middle']
            if 'atr_change' in X_with_vol.columns:
                X_with_vol['volatility_regime'] = (X_with_vol['atr_change'].abs() > X_with_vol['atr_change'].abs().quantile(0.7)).astype(int)

        return self.model.predict_proba(X_with_vol)

    def predict(self, X):
        return self.model.predict(X)

class DynamicMOE_v2(BaseEstimator, ClassifierMixin):
    '''Mixture of Experts v2 - 3 chuyên gia theo chế độ với đa khung thời gian và vùng cung cầu'''

    def __init__(self, gate_hidden_layers=[64, 32], gate_dropout=0.2):
        self.gate_hidden_layers = gate_hidden_layers
        self.gate_dropout = gate_dropout
        self.gate_network = None
        self.experts = {
            'trend': TrendDetector(),
            'range': RangeFinder(),
            'volatility': VolatilityBreakout()
        }
        self.feature_names = None

        # Các đặc trưng mới từ multi-timeframe và supply/demand zones
        self.regime_features = [
            'regime_id', 'atr_14', 'bollinger_width', 'volume_ratio_5',
            'rsi_14', 'macd_hist', 'volatility_regime', 'trend_strength',
            'price_position', 'entropy',
            # Multi-timeframe features
            'multi_tf_trend_consensus',
            'volatility_divergence',
            'near_demand_zone',
            'near_supply_zone',
            'in_demand_zone',
            'in_supply_zone',
            'fib_dist_nearest',
            'swing_high',
            'swing_low'
        ]

    def _extract_regime_features(self, X, regime_ids):
        '''Trích xuất các đặc trưng dùng cho gate network (đã cập nhật với multi-timeframe)'''
        if isinstance(X, pd.DataFrame):
            X_features = X.copy()
        else:
            X_features = pd.DataFrame(X, columns=self.feature_names)

        # Tạo các đặc trưng mới cho gate
        features = pd.DataFrame()
        features['regime_id'] = regime_ids
        features['atr_14'] = X_features.get('atr_14', np.zeros(len(X_features)))

        # Tính bollinger width
        if all(col in X_features.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            features['bollinger_width'] = (X_features['bb_upper'] - X_features['bb_lower']) / X_features['bb_middle']
        else:
            features['bollinger_width'] = np.zeros(len(X_features))

        # Tính volume ratio
        if 'volume' in X_features.columns:
            features['volume_ratio_5'] = X_features['volume'] / X_features['volume'].rolling(5).mean().fillna(1)
        else:
            features['volume_ratio_5'] = np.ones(len(X_features))

        features['rsi_14'] = X_features.get('rsi_14', np.full(len(X_features), 50.0))
        features['macd_hist'] = X_features.get('macd_hist', np.zeros(len(X_features)))

        # Tính volatility regime
        if 'atr_14' in X_features.columns:
            features['volatility_regime'] = (X_features['atr_14'] > X_features['atr_14'].quantile(0.7)).astype(int)
        else:
            features['volatility_regime'] = np.zeros(len(X_features))

        # Tính trend strength
        if 'close' in X_features.columns:
            ma50 = X_features['close'].rolling(50).mean()
            features['trend_strength'] = (X_features['close'] - ma50) / (ma50 + 1e-8)
        else:
            features['trend_strength'] = np.zeros(len(X_features))

        # Tính price position
        if all(col in X_features.columns for col in ['close', 'low_20', 'high_20']):
            features['price_position'] = (X_features['close'] - X_features['low_20']) / (X_features['high_20'] - X_features['low_20'] + 1e-8)
        else:
            features['price_position'] = np.full(len(X_features), 0.5)

        # Tính entropy
        if 'close' in X_features.columns:
            features['entropy'] = X_features['close'].pct_change().abs().rolling(5).std().fillna(0)
        else:
            features['entropy'] = np.zeros(len(X_features))

        # Thêm các đặc trưng multi-timeframe và supply/demand zones
        for feature in ['multi_tf_trend_consensus', 'volatility_divergence',
                       'near_demand_zone', 'near_supply_zone', 'in_demand_zone',
                       'in_supply_zone', 'fib_dist_nearest', 'swing_high', 'swing_low']:
            features[feature] = X_features.get(feature, np.zeros(len(X_features)))

        return features[self.regime_features]

    def fit(self, X, y, regime_ids=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Tách dữ liệu thành các phần để train từng expert
        X_expert = X

        # Train các expert
        for expert_name, expert in self.experts.items():
            print(f"Training {expert_name} expert...")
            expert.fit(X_expert, y)

        # Train gate network
        if regime_ids is None:
            # Nếu không có regime_id, tạo giả
            regime_ids = np.random.choice([0, 1, 2], size=len(y))

        gate_features = self._extract_regime_features(X, regime_ids)

        # Tạo nhãn cho gate: chọn expert tốt nhất dựa trên hiệu suất
        # (Trong thực tế, nên dùng cross-validation để chọn expert tốt nhất cho từng sample)
        # Ở đây, chúng ta sẽ dùng một cách đơn giản: chọn expert có xác suất cao nhất
        expert_probas = {}
        for expert_name, expert in self.experts.items():
            proba = expert.predict_proba(X_expert)
            # Đảm bảo proba có 2 cột
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            expert_probas[expert_name] = proba[:, 1]

        # Chọn expert có xác suất cao nhất cho mỗi sample
        expert_choices = np.argmax(np.array([expert_probas['trend'], expert_probas['range'], expert_probas['volatility']]), axis=0)

        # Xây dựng gate network
        self.gate_network = MLPClassifier(
            hidden_layer_sizes=self.gate_hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=500,
            random_state=42
        )

        self.gate_network.fit(gate_features, expert_choices)

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_expert = X
        else:
            X_expert = pd.DataFrame(X, columns=self.feature_names)

        # Dự đoán từ các expert
        expert_probas = {}
        for expert_name, expert in self.experts.items():
            proba = expert.predict_proba(X_expert)
            # Đảm bảo proba có 2 cột
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            expert_probas[expert_name] = proba[:, 1]

        # Trích xuất đặc trưng cho gate
        # Giả định regime_id là 0 (trending) nếu không có
        regime_ids = np.zeros(len(X_expert))
        gate_features = self._extract_regime_features(X_expert, regime_ids)

        # Dự đoán expert lựa chọn từ gate
        expert_choices = self.gate_network.predict(gate_features)

        # Kết hợp xác suất theo lựa chọn của gate
        final_proba = np.zeros(len(X_expert))
        for i, choice in enumerate(expert_choices):
            if choice == 0:  # trend
                final_proba[i] = expert_probas['trend'][i]
            elif choice == 1:  # range
                final_proba[i] = expert_probas['range'][i]
            else:  # volatility
                final_proba[i] = expert_probas['volatility'][i]

        return np.column_stack([1 - final_proba, final_proba])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

# Hàm tiện ích để lưu mô hình

def save_moe_v2(moe_model, filepath):
    joblib.dump(moe_model, filepath)
    print(f"Saved MOE v2 model to {filepath}")

# Hàm tiện ích để load mô hình

def load_moe_v2(filepath):
    moe_model = joblib.load(filepath)
    print(f"Loaded MOE v2 model from {filepath}")
    return moe_model