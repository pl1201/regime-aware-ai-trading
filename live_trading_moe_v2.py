"""
Live Trading Script cho MOE v2 Enhanced Model với OKX
"""
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt

# Thêm đường dẫn để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module cần thiết
from algo_trading.ml.dynamic_moe_v2_enhanced import DynamicMOE_v2_Enhanced
from algo_trading.filters.signal_quality_filter import signal_quality_filter
from algo_trading.risk.dynamic_risk_manager import DynamicRiskManager, RiskConfig
from algo_trading.data_loader.data_loader import load_multi_timeframe_data
from algo_trading.feature_engineering.feature_generator import add_multi_timeframe_features
from algo_trading.utils.data_utils import align_dataframes, derive_regime_ids

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MOE_v2_LiveTrader:
    def __init__(self, model_path: str, config_path: str = None):
        """Khởi tạo live trader"""
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.risk_manager = None
        self.exchange = None
        self.last_signal_time = None
        self.last_signal = None
        self.position = None

        # Load model
        self.load_model()

        # Khởi tạo risk manager
        risk_config = RiskConfig(
            max_risk_per_trade=0.02,
            max_daily_risk=0.06,
            max_drawdown_limit=0.25,
            tp_sl_ratio=3.0,
            min_risk_reward=1.8,
            enable_trailing_stop=True
        )
        self.risk_manager = DynamicRiskManager(risk_config)

        # Khởi tạo OKX exchange
        self.init_exchange()

    def load_model(self):
        """Load model từ file"""
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Đã load model từ {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Không thể load model: {e}")
            raise

    def init_exchange(self):
        """Khởi tạo kết nối với OKX"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap'
                }
            })
            logger.info(f"✅ Đã kết nối với OKX")
        except Exception as e:
            logger.error(f"❌ Không thể kết nối với OKX: {e}")
            raise

    def fetch_latest_data(self, symbol: str = 'BTC/USDT:USDT', timeframe: str = '1h'):
        """Lấy dữ liệu mới nhất từ OKX"""
        try:
            # Lấy 100 cây nến gần nhất
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"❌ Không thể lấy dữ liệu từ OKX: {e}")
            return None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn bị features cho model"""
        try:
            # Thêm multi-timeframe features
            df_features = add_multi_timeframe_features(df)

            # Align data
            df_aligned = align_dataframes([df_features])[0]

            return df_aligned
        except Exception as e:
            logger.error(f"❌ Không thể chuẩn bị features: {e}")
            return None

    def get_signal(self, df: pd.DataFrame) -> int:
        """Lấy tín hiệu từ model"""
        try:
            if self.model is None:
                logger.error("❌ Model chưa được load")
                return 0

            # Chuẩn bị features
            df_features = self.prepare_features(df)
            if df_features is None or len(df_features) == 0:
                return 0

            # Lấy feature names từ model
            feature_names = getattr(self.model, 'feature_names', [])
            if not feature_names:
                logger.error("❌ Model không có feature names")
                return 0

            # Lọc features theo model
            X = df_features[feature_names].values

            # Dự đoán
            prediction = self.model.predict(X)
            signal = prediction[-1]  # Lấy tín hiệu mới nhất

            # Áp dụng signal quality filter
            latest_features = df_features.iloc[-1:].copy()
            quality_filter = signal_quality_filter(latest_features)

            if not quality_filter[0]:
                logger.info("⚠️  Tín hiệu không đạt chất lượng")
                return 0

            return int(signal)
        except Exception as e:
            logger.error(f"❌ Không thể lấy tín hiệu: {e}")
            return 0

    def calculate_position_size(self, symbol: str, signal: int) -> float:
        """Tính toán kích thước vị thế"""
        try:
            # Lấy thông tin thị trường
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']

            # Lấy balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            # Tính position size theo risk management
            position_size = self.risk_manager.calculate_position_size(
                account_balance=usdt_balance,
                price=price,
                signal_confidence=abs(signal) / 2.0,  # Chuyển từ [-1,1] sang [0,1]
                volatility=0.015,  # Volatility mặc định
                current_drawdown=0.0  # Drawdown hiện tại
            )

            return position_size
        except Exception as e:
            logger.error(f"❌ Không thể tính position size: {e}")
            return 0.0

    def execute_trade(self, symbol: str, signal: int, position_size: float):
        """Thực hiện giao dịch"""
        try:
            side = 'buy' if signal > 0 else 'sell'
            amount = abs(position_size)

            if amount <= 0:
                logger.info("⚠️  Không có vị thế để thực hiện")
                return

            # Thực hiện giao dịch
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"✅ Đã thực hiện lệnh {side} {amount} {symbol}")

            # Lưu thông tin vị thế
            self.position = {
                'side': side,
                'amount': amount,
                'symbol': symbol,
                'timestamp': datetime.now()
            }

            return order
        except Exception as e:
            logger.error(f"❌ Không thể thực hiện giao dịch: {e}")
            return None

    def run(self, symbol: str = 'BTC/USDT:USDT', check_interval: int = 3600):
        """Chạy live trading"""
        logger.info("🚀 Bắt đầu live trading với MOE v2 Enhanced Model")

        while True:
            try:
                # Lấy thời gian hiện tại
                current_time = datetime.now()
                logger.info(f"⏰ Kiểm tra tín hiệu lúc {current_time}")

                # Lấy dữ liệu mới nhất
                df = self.fetch_latest_data(symbol)
                if df is None or len(df) < 50:
                    logger.warning("⚠️  Không đủ dữ liệu để phân tích")
                    time.sleep(check_interval)
                    continue

                # Lấy tín hiệu
                signal = self.get_signal(df)
                logger.info(f"📡 Tín hiệu: {signal}")

                # Nếu có tín hiệu và chưa có vị thế
                if signal != 0 and self.position is None:
                    logger.info(f"🎯 Có tín hiệu giao dịch: {signal}")

                    # Tính position size
                    position_size = self.calculate_position_size(symbol, signal)
                    logger.info(f"💰 Position size: {position_size}")

                    # Thực hiện giao dịch
                    if position_size > 0:
                        order = self.execute_trade(symbol, signal, position_size)
                        if order:
                            self.last_signal_time = current_time
                            self.last_signal = signal

                # Kiểm tra trailing stop nếu có vị thế
                elif self.position is not None:
                    self.check_trailing_stop(symbol)

                # Chờ đến lần kiểm tra tiếp theo
                logger.info(f"💤 Ngủ {check_interval} giây trước lần kiểm tra tiếp theo")
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Người dùng dừng live trading")
                break
            except Exception as e:
                logger.error(f"❌ Lỗi trong quá trình live trading: {e}")
                time.sleep(60)  # Đợi 1 phút rồi tiếp tục

    def check_trailing_stop(self, symbol: str):
        """Kiểm tra trailing stop"""
        try:
            if self.position is None:
                return

            # Lấy giá hiện tại
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Kiểm tra trailing stop (đơn giản)
            if self.position['side'] == 'buy':
                # Nếu giá giảm 1.5% thì đóng lệnh
                entry_price = self.position.get('entry_price', current_price)
                if current_price < entry_price * 0.985:  # Giảm 1.5%
                    self.close_position(symbol)
            elif self.position['side'] == 'sell':
                # Nếu giá tăng 1.5% thì đóng lệnh
                entry_price = self.position.get('entry_price', current_price)
                if current_price > entry_price * 1.015:  # Tăng 1.5%
                    self.close_position(symbol)

        except Exception as e:
            logger.error(f"❌ Lỗi khi kiểm tra trailing stop: {e}")

    def close_position(self, symbol: str):
        """Đóng vị thế hiện tại"""
        try:
            if self.position is None:
                return

            side = 'sell' if self.position['side'] == 'buy' else 'buy'
            amount = self.position['amount']

            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"✅ Đã đóng vị thế {side} {amount} {symbol}")

            self.position = None
        except Exception as e:
            logger.error(f"❌ Không thể đóng vị thế: {e}")

def main():
    """Hàm main"""
    # Đường dẫn đến model
    model_path = "models/dynamic_moe_v2_enhanced_final_v2.pkl"

    # Kiểm tra model tồn tại
    if not os.path.exists(model_path):
        logger.error(f"❌ Không tìm thấy model tại {model_path}")
        return

    # Tạo live trader
    trader = MOE_v2_LiveTrader(model_path)

    # Chạy live trading
    try:
        trader.run(symbol='BTC/USDT:USDT', check_interval=3600)  # Kiểm tra mỗi giờ
    except KeyboardInterrupt:
        logger.info("🛑 Người dùng dừng live trading")
    except Exception as e:
        logger.error(f"❌ Lỗi khi chạy live trading: {e}")

if __name__ == "__main__":
    main()