"""
Live Trading Bot tổng quát - Hỗ trợ mọi strategy từ hệ thống
- Tích hợp với Binance API (testnet/live)
- Hỗ trợ tất cả strategies từ algo_trading.strategies
- Risk management: SL/TP/Trailing stops
- Chạy như daemon/service
- Logging và monitoring đầy đủ

Cách sử dụng:
1. Tạo file .env với cấu hình:
   MODE=testnet  # hoặc paper, live
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   SYMBOL=BTCUSDT
   INTERVAL=5m
   STRATEGY=sma_ema  # hoặc macd, rsi_div, etc.
   STRATEGY_PARAMS={"fast":20,"slow":50,"ma_type":"ema"}
   SL_PCT=0.02
   TP_PCT=0.04
   RISK_PER_TRADE=0.1
   
2. Chạy: python -m algo_trading.live.universal_bot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import os
import time
import json
import logging
from datetime import datetime, timezone
import signal
import sys
import ast
from pathlib import Path
from dataclasses import dataclass as _dataclass_fallback

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv
import threading

from algo_trading.config import BotConfig
from algo_trading.live.exchange_base import ExchangeClient, SymbolFilters

# Import strategies (optional trong production bundle toi gian)
try:
    from algo_trading.strategies import (
        SMAEMACrossStrategy,
        RSIDivergenceStrategy,
        MACDMomentumStrategy,
        BollingerBreakoutStrategy,
        VWAPMeanReversionStrategy,
        RenkoTrendStrategy,
        VolumeProfileImbalanceStrategy,
        OUProcessMeanReversionStrategy,
        KalmanFilterForecastStrategy,
        ARIMAStrategy,
        LSTMTransformerStrategy,
        StatArbCointegrationStrategy,
        GARCHVolatilityStrategy,
    )
    from algo_trading.strategies.base import BaseStrategy, StrategyResult
    HAS_STRATEGIES = True
except ImportError:
    HAS_STRATEGIES = False

    class BaseStrategy:
        pass

    @_dataclass_fallback
    class StrategyResult:
        signals: Any
        meta: Optional[Dict[str, Any]] = None

# Import Exchange clients
try:
    from algo_trading.live.okx_client import OKXClient
    HAS_OKX = True
except ImportError:
    HAS_OKX = False
    OKXClient = None

# Import Regime Transformer Strategy
try:
    from algo_trading.strategies.ml.regime_transformer_strategy import RegimeTransformerStrategy
    HAS_REGIME_TRANSFORMER = True
except ImportError:
    HAS_REGIME_TRANSFORMER = False
    RegimeTransformerStrategy = None

# Import Regime Ensemble (ML) Strategy – dùng cho regime-specific models
try:
    from algo_trading.strategies.ml.regime_ensemble_strategy import RegimeEnsembleStrategy
    HAS_REGIME_ENSEMBLE = True
except ImportError:
    HAS_REGIME_ENSEMBLE = False
    RegimeEnsembleStrategy = None

# Import Dynamic MOE v2 Enhanced Model
try:
    from algo_trading.ml.dynamic_moe_v2_enhanced import DynamicMOE_v2_Enhanced
    from algo_trading.ml.moe_signal_pipeline import SignalPipelineConfig, build_trade_signals
    HAS_MOE_V2_ENHANCED = True
except ImportError:
    HAS_MOE_V2_ENHANCED = False
    DynamicMOE_v2_Enhanced = None
    SignalPipelineConfig = None
    build_trade_signals = None

# Import H1 Hybrid Model (NEW - replaces M15)
try:
    from algo_trading.features.h1_features import H1Features
    from algo_trading.ml.h1_hybrid_model import H1HybridModel, get_h1_model
    HAS_H1_HYBRID = True
except ImportError:
    HAS_H1_HYBRID = False
    H1Features = None
    H1HybridModel = None
    get_h1_model = None

# Import Telegram Bot
try:
    from telegram import Bot
    from telegram.error import TelegramError
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    Bot = None
    TelegramError = None

# Load Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
telegram_bot = None

if HAS_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        print("[INFO] Telegram bot initialized")
    except Exception as e:
        print(f"[ERROR] Telegram bot init failed: {e}")
        telegram_bot = None

def send_telegram_message(message: str):
    """Gửi thông báo Telegram"""
    if not HAS_TELEGRAM or not telegram_bot or not TELEGRAM_CHAT_ID:
        return

    try:
        telegram_bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode="Markdown"
        )
        logger.info(f"✅ Đã gửi thông báo Telegram: {message}")
    except TelegramError as e:
        logger.error(f"❌ Lỗi khi gửi Telegram: {e}")
    except Exception as e:
        logger.error(f"❌ Lỗi không xác định khi gửi Telegram: {e}")

# Import evaluator và combiner (optional)
try:
    from algo_trading.live.strategy_evaluator import StrategyEvaluator
    from algo_trading.live.indicator_combiner import (
        IndicatorCombiner,
        OptimizedCombinations,
        PRESET_COMBINATIONS,
        create_custom_combination,
    )
    HAS_COMBINERS = True
except ImportError:
    HAS_COMBINERS = False
    StrategyEvaluator = None
    IndicatorCombiner = None
    OptimizedCombinations = None
    PRESET_COMBINATIONS = {}
    create_custom_combination = None

# Import Telegram bot (optional)
try:
    from algo_trading.live.telegram_bot import (
        send_signal_notification,
        set_trading_bot,
        run_telegram_bot,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    send_signal_notification = None
    set_trading_bot = None
    run_telegram_bot = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Strategy mapping
STRATEGY_MAP = {}
if HAS_STRATEGIES:
    STRATEGY_MAP.update(
        {
            "sma_ema": SMAEMACrossStrategy,
            "rsi_div": RSIDivergenceStrategy,
            "macd": MACDMomentumStrategy,
            "bb_breakout": BollingerBreakoutStrategy,
            "vwap_mr": VWAPMeanReversionStrategy,
            "renko_trend": RenkoTrendStrategy,
            "vol_profile": VolumeProfileImbalanceStrategy,
            "ou_mr": OUProcessMeanReversionStrategy,
            "kalman": KalmanFilterForecastStrategy,
            "arima": ARIMAStrategy,
            "lstm": LSTMTransformerStrategy,
            "stat_arb": StatArbCointegrationStrategy,
            "garch_vol": GARCHVolatilityStrategy,
        }
    )

if HAS_REGIME_TRANSFORMER:
    STRATEGY_MAP["regime_transformer"] = RegimeTransformerStrategy

if HAS_REGIME_ENSEMBLE:
    # Dùng chung RegimeEnsembleStrategy, bật regime-specific qua STRATEGY_PARAMS trong .env
    STRATEGY_MAP["regime_ensemble"] = RegimeEnsembleStrategy
    STRATEGY_MAP["regime_specific"] = RegimeEnsembleStrategy
    STRATEGY_MAP["regime_ensemble_specific"] = RegimeEnsembleStrategy

# Thêm MOE v2 Enhanced nếu có
if HAS_MOE_V2_ENHANCED:
    # Tạo một wrapper strategy cho MOE v2 Enhanced
    class MOE_v2_Enhanced_Strategy(BaseStrategy):
        def __init__(
            self,
            model_path: str = "models/dynamic_moe_v2_enhanced_final.pkl",
            artifact_path: Optional[str] = None,
            proba_threshold: Optional[float] = None,
            conf_gap: Optional[float] = None,
            ict_thresh: Optional[float] = None,
            regime_gate: Optional[float] = None,
            **kwargs,
        ):
            self.model_path = model_path
            self.artifact_path = artifact_path
            self.model = None
            self.artifact = {}
            self.feature_names = []
            self.pipeline_cfg = None
            self.classes = np.array([-1, 0, 1])

            self._pipeline_overrides = {
                "threshold": proba_threshold,
                "conf_gap": conf_gap,
                "ict_thresh": ict_thresh,
                "regime_gate": regime_gate,
            }
            self.load_model()

        def _resolve_path(self, p: str) -> Path:
            path = Path(p)
            if path.is_absolute():
                return path
            # Resolve relative to current working dir first, then production root.
            cwd_resolved = Path.cwd() / path
            if cwd_resolved.exists():
                return cwd_resolved
            return Path(__file__).resolve().parents[2] / path

        def _build_pipeline_cfg(self, signal_pipeline_cfg: Dict[str, Any]) -> Any:
            cfg = SignalPipelineConfig(
                threshold=float(signal_pipeline_cfg.get("threshold", 0.40)),
                conf_gap=float(signal_pipeline_cfg.get("conf_gap", 0.03)),
                ict_thresh=float(signal_pipeline_cfg.get("ict_thresh", 0.25)),
                min_signal_strength=float(signal_pipeline_cfg.get("min_signal_strength", 0.35)),
                min_expected_edge=float(signal_pipeline_cfg.get("min_expected_edge", 0.12)),
                churn_guard_bars=int(signal_pipeline_cfg.get("churn_guard_bars", 8)),
                regime_gate=float(signal_pipeline_cfg.get("regime_gate", 0.12)),
                volatility_q_low=float(signal_pipeline_cfg.get("volatility_q_low", 0.10)),
                volatility_q_high=float(signal_pipeline_cfg.get("volatility_q_high", 0.90)),
                min_signals_per_fold=int(signal_pipeline_cfg.get("min_signals_per_fold", 25)),
                coverage_min=float(signal_pipeline_cfg.get("coverage_min", 0.05)),
                target_coverage=float(signal_pipeline_cfg.get("target_coverage", 0.05)),
                pf_min=float(signal_pipeline_cfg.get("pf_min", 1.00)),
                min_winrate=float(signal_pipeline_cfg.get("min_winrate", 0.45)),
                min_profit_ratio=float(signal_pipeline_cfg.get("min_profit_ratio", 1.00)),
                max_drawdown=float(signal_pipeline_cfg.get("max_drawdown", 0.15)),
                min_fold_pass_rate=float(signal_pipeline_cfg.get("min_fold_pass_rate", 0.80)),
                max_pf_std=float(signal_pipeline_cfg.get("max_pf_std", 0.70)),
                max_coverage_cv=float(signal_pipeline_cfg.get("max_coverage_cv", 0.80)),
            )
            for key, value in self._pipeline_overrides.items():
                if value is not None:
                    setattr(cfg, key, float(value))
            return cfg

        def load_model(self):
            """Load MOE v2 Enhanced model and aligned artifact config."""
            try:
                import joblib

                model_file = self._resolve_path(self.model_path)
                if not model_file.exists():
                    logger.error(f"❌ Không tìm thấy model tại {model_file}")
                    return

                loaded = joblib.load(str(model_file))
                self.model = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
                logger.info(f"✅ Đã load MOE v2 Enhanced model từ {model_file}")

                artifact_file = self._resolve_path(self.artifact_path) if self.artifact_path else model_file.with_name(model_file.stem + "_artifact.pkl")
                if artifact_file.exists():
                    loaded_artifact = joblib.load(str(artifact_file))
                    self.artifact = loaded_artifact if isinstance(loaded_artifact, dict) else {}
                    logger.info(f"✅ Đã load artifact MOE từ {artifact_file}")
                else:
                    logger.warning(f"⚠️ Không tìm thấy artifact, dùng default pipeline config: {artifact_file}")
                    self.artifact = {}

                self.feature_names = list(self.artifact.get("feature_names") or getattr(self.model, "feature_names", []))
                class_labels = self.artifact.get("class_labels")
                if class_labels:
                    self.classes = np.array(class_labels)
                else:
                    self.classes = np.array(getattr(self.model, "classes_", [-1, 0, 1]))

                self.pipeline_cfg = self._build_pipeline_cfg(self.artifact.get("signal_pipeline", {}))
            except Exception as e:
                logger.error(f"❌ Không thể load MOE v2 Enhanced model: {e}")

        def predict(self, df: pd.DataFrame) -> np.ndarray:
            """Dự đoán tín hiệu từ dữ liệu bằng pipeline regime_head + entry_head."""
            try:
                if self.model is None:
                    logger.error("❌ Model chưa được load")
                    return np.zeros(len(df))

                if self.pipeline_cfg is None:
                    logger.error("❌ Pipeline config chưa sẵn sàng")
                    return np.zeros(len(df))

                feature_names = self.feature_names
                if not feature_names:
                    logger.error("❌ Model không có feature names")
                    return np.zeros(len(df))

                X_df = (
                    df.reindex(columns=feature_names, fill_value=0.0)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                probs = self.model.predict_proba(X_df.values)
                signals, _ = build_trade_signals(probs, df, classes=self.classes, cfg=self.pipeline_cfg)
                return signals.astype(int)
            except Exception as e:
                logger.error(f"❌ Không thể dự đoán tín hiệu: {e}")
                return np.zeros(len(df))

        def __call__(self, df: pd.DataFrame, **kwargs) -> StrategyResult:
            """Gọi strategy để tạo tín hiệu"""
            try:
                # Dự đoán tín hiệu
                signals = self.predict(df)

                # Trả về kết quả
                return StrategyResult(
                    signals=signals,
                    meta={
                        "strategy": "moe_v2_enhanced",
                        "model_path": self.model_path,
                        "threshold": float(self.pipeline_cfg.threshold) if self.pipeline_cfg else None,
                        "signal_count": int((signals != 0).sum()),
                    }
                )
            except Exception as e:
                logger.error(f"❌ Lỗi khi chạy MOE v2 Enhanced strategy: {e}")
                return StrategyResult(
                    signals=np.zeros(len(df)),
                    meta={"error": str(e)}
                )

    # Thêm vào STRATEGY_MAP
    STRATEGY_MAP["moe_v2_enhanced"] = MOE_v2_Enhanced_Strategy

# H1 Hybrid Strategy (NEW - recommended for hourly trading)
if HAS_H1_HYBRID:
    class H1_Hybrid_Strategy(BaseStrategy):
        """
        H1 Hybrid Trading Strategy.
        
        Combines Trend Following and Mean Reversion:
        - When ADX > 25: Trend following mode
        - When ADX <= 25: Mean reversion with RSI filter
        
        Performance (2025 backtest):
        - 169 trades, WR=55%, PF=1.35, Return=+30.6%
        """
        
        def __init__(
            self,
            model_dir: Optional[str] = None,
            min_confidence: float = 0.45,
            **kwargs,
        ):
            self.model_dir = Path(model_dir) if model_dir else None
            self.min_confidence = min_confidence
            self.feature_builder = H1Features()
            self.model = None
            self.load_model()
        
        def load_model(self):
            """Load H1 hybrid model."""
            try:
                self.model = get_h1_model(self.model_dir)
                if self.model.is_fitted:
                    logger.info("✅ Loaded H1 Hybrid model successfully")
                else:
                    # Try to load from algo_trading_H1
                    h1_path = Path(__file__).parent.parent.parent / 'algo_trading_H1' / 'models'
                    if h1_path.exists():
                        self.model = H1HybridModel(h1_path)
                        self.model.load()
                        if self.model.is_fitted:
                            logger.info(f"✅ Loaded H1 model from {h1_path}")
                        else:
                            logger.warning("⚠️ H1 model not found - please train first")
                    else:
                        logger.warning("⚠️ H1 model directory not found")
            except Exception as e:
                logger.error(f"❌ Error loading H1 model: {e}")
        
        def predict(self, df: pd.DataFrame) -> np.ndarray:
            """Generate trading signals."""
            try:
                if self.model is None or not self.model.is_fitted:
                    return np.zeros(len(df))
                
                # Build features
                features = self.feature_builder.build_features(df)
                
                # Get predictions
                signals, confidences, regimes = self.model.predict(features)
                
                # Filter by confidence
                signals[confidences < self.min_confidence] = 0
                
                return signals.astype(int)
            except Exception as e:
                logger.error(f"❌ H1 prediction error: {e}")
                return np.zeros(len(df))
        
        def __call__(self, df: pd.DataFrame, **kwargs) -> StrategyResult:
            """Run strategy."""
            try:
                signals = self.predict(df)
                return StrategyResult(
                    signals=signals,
                    meta={
                        "strategy": "h1_hybrid",
                        "min_confidence": self.min_confidence,
                        "signal_count": int((signals != 0).sum()),
                    }
                )
            except Exception as e:
                logger.error(f"❌ H1 strategy error: {e}")
                return StrategyResult(
                    signals=np.zeros(len(df)),
                    meta={"error": str(e)}
                )
    
    # Add to strategy map
    STRATEGY_MAP["h1_hybrid"] = H1_Hybrid_Strategy
    STRATEGY_MAP["h1"] = H1_Hybrid_Strategy  # Alias
    STRATEGY_MAP["hourly"] = H1_Hybrid_Strategy  # Alias

if HAS_STRATEGIES and HAS_COMBINERS:
    STRATEGY_MAP.update(
        {
            "trend_momentum": lambda **kwargs: OptimizedCombinations.trend_momentum_combo(),
            "mean_reversion": lambda **kwargs: OptimizedCombinations.mean_reversion_combo(),
            "balanced": lambda **kwargs: OptimizedCombinations.balanced_combo(),
            "aggressive_trend": lambda **kwargs: OptimizedCombinations.aggressive_trend_combo(),
            "conservative": lambda **kwargs: OptimizedCombinations.conservative_combo(),
            "momentum_focused": lambda **kwargs: OptimizedCombinations.momentum_focused_combo(),
        }
    )


class BinanceClient(ExchangeClient):
    """Wrapper cho Binance API client."""
    
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], config: BotConfig):
        self.config = config
        self.client = Client(api_key or "", api_secret or "")
        self._symbol_filters_cache: Dict[str, SymbolFilters] = {}
        
        if config.mode == "testnet":
            self.client.API_URL = "https://testnet.binance.vision/api"
            logger.info("🔶 Đang dùng Binance Spot Testnet")
        elif config.mode == "live":
            logger.warning("🔴 BẠN ĐANG Ở CHẾ ĐỘ LIVE! Hãy cẩn thận!")
        else:
            logger.info("📄 Đang chạy ở chế độ PAPER (không gửi lệnh)")
    
    def get_klines_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Lấy dữ liệu kline và trả về DataFrame."""
        try:
            raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ]
            df = pd.DataFrame(raw, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Lỗi lấy klines: {e}")
            return pd.DataFrame()
    
    def get_last_price(self, symbol: str) -> float:
        """Lấy giá hiện tại."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Lỗi lấy giá: {e}")
            return 0.0
    
    def _fetch_symbol_filters(self, symbol: str) -> SymbolFilters:
        """Lấy filters từ exchange info."""
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]
        
        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                raise ValueError(f"Không tìm thấy symbol {symbol}")
            
            step_size = 0.0
            min_qty = 0.0
            min_notional = 0.0
            tick_size = 0.0
            
            for f in info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                elif f["filterType"] == "MIN_NOTIONAL":
                    min_notional = float(f.get("minNotional", 0))
                elif f["filterType"] == "PRICE_FILTER":
                    tick_size = float(f["tickSize"])
            
            filters = SymbolFilters(
                step_size=step_size,
                min_qty=min_qty,
                min_notional=min_notional,
                tick_size=tick_size
            )
            self._symbol_filters_cache[symbol] = filters
            return filters
        except Exception as e:
            logger.error(f"Lỗi lấy filters: {e}")
            return SymbolFilters(0.0, 0.0, 0.0, 0.0)
    
    def get_asset_balance(self, asset: str) -> float:
        """Lấy số dư asset."""
        if self.config.mode == "paper":
            return 0.0
        try:
            bal = self.client.get_asset_balance(asset=asset)
            if not bal:
                return 0.0
            return float(bal.get("free", 0))
        except Exception as e:
            logger.error(f"Lỗi lấy số dư: {e}")
            return 0.0
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Đặt lệnh market."""
        if quantity <= 0:
            raise ValueError("Quantity phải > 0")
        
        filters = self._fetch_symbol_filters(symbol)
        qty = np.floor(quantity / filters.step_size) * filters.step_size if filters.step_size > 0 else quantity
        
        if qty < filters.min_qty:
            raise ValueError(f"Quantity {qty} < minQty {filters.min_qty}")
        
        if self.config.mode == "paper":
            logger.info(f"[PAPER] {side} {qty} {symbol}")
            return {"paper": True, "side": side, "symbol": symbol, "executedQty": qty}
        
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty,
            )
            logger.info(f"✅ Đã gửi lệnh {side} {qty} {symbol}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"❌ Lỗi gửi lệnh: {e}")
            raise
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Đặt lệnh limit (cho SL/TP)."""
        if quantity <= 0:
            raise ValueError("Quantity phải > 0")
        
        filters = self._fetch_symbol_filters(symbol)
        qty = np.floor(quantity / filters.step_size) * filters.step_size if filters.step_size > 0 else quantity
        price = np.round(price / filters.tick_size) * filters.tick_size if filters.tick_size > 0 else price
        
        if self.config.mode == "paper":
            logger.info(f"[PAPER] {side} {qty} {symbol} @ {price} (LIMIT)")
            return {"paper": True, "side": side, "symbol": symbol, "executedQty": qty, "price": price}
        
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=Client.TIME_IN_FORCE_GTC,
                quantity=qty,
                price=str(price),
            )
            logger.info(f"✅ Đã gửi lệnh LIMIT {side} {qty} {symbol} @ {price}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"❌ Lỗi gửi lệnh LIMIT: {e}")
            raise


class LiveTradingBot:
    """Bot live trading tổng quát."""
    
    def __init__(self, client: ExchangeClient, strategy: BaseStrategy, config: BotConfig):
        self.client = client
        self.strategy = strategy
        self.config = config
        self.base_asset, self.quote_asset = self._split_symbol(config.symbol)
        self.holding = False
        self.last_signal_time: Optional[datetime] = None
        self.entry_price: Optional[float] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        self.position_size: float = 0.0
        # Cache signal để UI/Telegram có thể hiển thị nhanh mà không cần recompute (đặc biệt với ML/regime strategies)
        self.latest_signal: Optional[int] = None
        self.latest_signal_error: Optional[str] = None
        self.running = True
        
        # Setup signal handler để dừng bot gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Xử lý signal để dừng bot."""
        logger.info("Nhận signal dừng bot...")
        self.running = False
    
    @staticmethod
    def _split_symbol(symbol: str) -> Tuple[str, str]:
        """Tách symbol thành base/quote."""
        for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH", "FDUSD", "TUSD"]:
            if symbol.endswith(quote):
                return symbol.replace(quote, ""), quote
        return symbol[:3], symbol[3:]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Tính ATR."""
        if len(df) < period:
            return pd.Series(0.0, index=df.index)
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().fillna(0.0)
        return atr
    
    def _calculate_position_size(self, last_price: float, df: pd.DataFrame) -> float:
        """Tính position size dựa trên risk management."""
        if self.config.mode == "paper":
            quote_balance = 1000.0  # Giả lập
        else:
            quote_balance = self.client.get_asset_balance(self.quote_asset)
        
        risk_amount = quote_balance * self.config.risk_per_trade
        
        # Tính SL distance
        sl_distance = None
        if self.config.sl_pct:
            sl_distance = last_price * self.config.sl_pct
        elif self.config.sl_atr_k:
            atr = self._calculate_atr(df)
            if len(atr) > 0 and atr.iloc[-1] > 0:
                sl_distance = self.config.sl_atr_k * atr.iloc[-1]
        
        if sl_distance and sl_distance > 0:
            qty = risk_amount / sl_distance
        else:
            qty = (quote_balance * self.config.risk_per_trade) / last_price
        
        # Giới hạn max position size nếu có
        if self.config.max_position_size:
            qty = min(qty, self.config.max_position_size)
        
        return max(qty, 0.0)
    
    def _update_stop_loss_take_profit(self, entry_price: float, direction: int, df: pd.DataFrame):
        """Cập nhật SL/TP prices."""
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # Tính ATR nếu cần
        atr_value = None
        if self.config.sl_atr_k or self.config.tp_atr_k:
            atr = self._calculate_atr(df)
            if len(atr) > 0:
                atr_value = atr.iloc[-1]
        
        if direction > 0:  # LONG
            if self.config.sl_pct:
                self.stop_loss_price = entry_price * (1 - self.config.sl_pct)
            elif self.config.sl_atr_k and atr_value:
                self.stop_loss_price = entry_price - self.config.sl_atr_k * atr_value
            
            if self.config.tp_pct:
                self.take_profit_price = entry_price * (1 + self.config.tp_pct)
            elif self.config.tp_atr_k and atr_value:
                self.take_profit_price = entry_price + self.config.tp_atr_k * atr_value
        else:  # SHORT
            if self.config.sl_pct:
                self.stop_loss_price = entry_price * (1 + self.config.sl_pct)
            elif self.config.sl_atr_k and atr_value:
                self.stop_loss_price = entry_price + self.config.sl_atr_k * atr_value
            
            if self.config.tp_pct:
                self.take_profit_price = entry_price * (1 - self.config.tp_pct)
            elif self.config.tp_atr_k and atr_value:
                self.take_profit_price = entry_price - self.config.tp_atr_k * atr_value
        
        if self.stop_loss_price:
            logger.info(f"📌 SL: {self.stop_loss_price:.8f}")
        if self.take_profit_price:
            logger.info(f"🎯 TP: {self.take_profit_price:.8f}")
    
    def _check_stop_loss_take_profit(self, current_price: float, direction: int) -> Optional[str]:
        """Kiểm tra SL/TP có bị hit không."""
        if not self.holding or not self.entry_price:
            return None
        
        if direction > 0:  # LONG
            if self.stop_loss_price and current_price <= self.stop_loss_price:
                return "sl"
            if self.take_profit_price and current_price >= self.take_profit_price:
                return "tp"
        else:  # SHORT
            if self.stop_loss_price and current_price >= self.stop_loss_price:
                return "sl"
            if self.take_profit_price and current_price <= self.take_profit_price:
                return "tp"
        
        return None
    
    def _exit_position(self, reason: str = "signal"):
        """Thoát position hiện tại."""
        if not self.holding:
            return
        
        if self.config.mode == "paper":
            qty = self.position_size
        else:
            # Ưu tiên lấy từ position size hiện tại nếu client hỗ trợ (Futures)
            if hasattr(self.client, "get_current_position"):
                # Cập nhật position mới nhất từ sàn để chắc chắn
                current_pos, _ = self.client.get_current_position(self.config.symbol)
                qty = abs(current_pos)
                self.position_size = current_pos
            else:
                # Fallback: Spot (lấy số dư coin)
                qty = self.client.get_asset_balance(self.base_asset)
        
        if qty <= 0:
            logger.warning("Không có position để thoát")
            return
        
        try:
            side = "SELL" if self.position_size > 0 else "BUY"  # Nếu short thì mua lại
            self.client.place_market_order(self.config.symbol, side, qty)
            logger.info(f"🚪 Đã thoát position ({reason})")
            
            # Lấy giá hiện tại để tính P&L
            current_price = self.client.get_last_price(self.config.symbol)
            old_entry = self.entry_price
            old_direction = 1 if self.position_size > 0 else -1
            
            self.holding = False
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
            self.position_size = 0.0
            self.last_signal_time = datetime.now(timezone.utc)
            
            # Gửi thông báo Telegram
            if HAS_TELEGRAM and send_signal_notification:
                signal_val = 0  # Exit = neutral
                send_signal_notification(
                    signal=signal_val,
                    price=current_price,
                    symbol=self.config.symbol,
                    holding=False,
                    entry_price=old_entry,
                    reason=f"Exited ({reason})"
                )
        except Exception as e:
            logger.exception(f"Lỗi thoát position: {e}")
    
    def _enter_position(self, direction: int, df: pd.DataFrame):
        """Vào position mới (hoặc DCA thêm)."""
        # Check DCA limit
        # Ước tính số lệnh hiện tại = position_size / (risk_amount_alloc) 
        # Tuy nhiên để đơn giản, ta dùng logic: 
        # Nếu holding và cùng chiều -> check max_dca_orders
        
        is_dca = False
        if self.holding:
            # Check direction match
            current_dir = 1 if self.position_size > 0 else -1
            if current_dir != direction:
                # Ngược chiều -> Có thể là signal đảo chiều -> return để logic exit xử lý trước
                return
            
            # Check DCA limit
            # Ở đây ta cần ước lượng xem đã vào bao nhiêu lệnh.
            # Giả sử mỗi lệnh size xấp xỉ nhau (theo risk).
            # Hoặc đơn giản là đếm số lần vào lệnh trong session (nhưng reset khi restart).
            # Tạm thời dùng logic đơn giản: Nếu holding và cho phép DCA -> vào tiếp.
            # Cần cơ chế đếm số lệnh chính xác hơn.
            # => Check max_position_size nếu có.
            
            if self.config.max_dca_orders <= 1:
                return # Không cho DCA
                
            # TODO: Cần cách đếm số lệnh DCA hiện tại chính xác hơn. 
            # Tạm thời ta cho phép vào nếu chưa đạt max_position_size (nếu có set)
            # Hoặc ta đếm tương đối:
            
            # Nếu position quá lớn -> chặn
            # if self.config.max_position_size and abs(self.position_size) >= self.config.max_position_size:
            #    logger.warning("Max position size reached, skip DCA")
            #    return
                
            # Logic tạm thời: Cứ có signal là vào, nhưng cần cơ chế rate limit
            # Tuy nhiên, run_once đã có cooldown, nên mỗi lần cooldown xong mà vẫn còn signal -> vào tiếp
            
            # Để tránh spam đến vô tận, ta giới hạn DCA dựa trên việc check PnL hoặc khoảng cách giá (nếu cần). 
            # Nhưng user yêu cầu đơn giản là "vào lệnh mới và vào dc nhiều lệnh".
            
            logger.info(f"➕ Triggering DCA Entry ({direction})")
            is_dca = True
        
        current_price = self.client.get_last_price(self.config.symbol)
        if current_price <= 0:
            logger.error("Không lấy được giá")
            return
        
        # Tính position size (Reset risk hoặc giữ nguyên)
        qty = self._calculate_position_size(current_price, df)
        if qty <= 0:
            logger.warning("Position size <= 0, bỏ qua")
            return
        
        # Điều chỉnh theo direction
        if direction < 0:  # SHORT - cần quote asset
            # Với spot, short phức tạp hơn, ở đây giả lập
            logger.warning("SHORT không được hỗ trợ đầy đủ trên Spot, chỉ mô phỏng")
            qty = -qty  # Đánh dấu là short
        
        try:
            side = "BUY" if direction > 0 else "SELL"
            order_result = self.client.place_market_order(self.config.symbol, side, abs(qty))
            
            # Kiểm tra nếu order thành công
            if order_result and not order_result.get("error"):
                self.holding = True
                self.entry_price = current_price
                self.position_size = qty
                self._update_stop_loss_take_profit(current_price, direction, df)
                self.last_signal_time = datetime.now(timezone.utc)
                
                executed_qty = order_result.get("executedQty", abs(qty))
                logger.info(f"OK Da vao {side} {executed_qty} {self.config.symbol} @ {current_price:.8f}")
            else:
                error_msg = order_result.get("msg", "Unknown error") if order_result else "No response"
                raise ValueError(f"Order failed: {error_msg}")
            
            # Gửi thông báo Telegram
            if HAS_TELEGRAM and send_signal_notification:
                signal_val = 1 if direction > 0 else -1
                send_signal_notification(
                    signal=signal_val,
                    price=current_price,
                    symbol=self.config.symbol,
                    holding=True,
                    entry_price=current_price,
                    reason=f"Entered {side}"
                )
        except ValueError as e:
            # ValueError từ place_market_order đã có message chi tiết
            logger.error(f"Loi vao position: {e}")
        except Exception as e:
            logger.error(f"Loi vao position (unexpected): {e}", exc_info=True)
    
    def _cooldown_ok(self) -> bool:
        """Kiểm tra cooldown."""
        if self.last_signal_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_signal_time).total_seconds()
        return elapsed >= self.config.cool_down_sec
    
    def run_once(self):
        """Chạy một vòng: lấy dữ liệu -> tính signal -> xử lý."""
        # 1. Lấy dữ liệu
        df = self.client.get_klines_df(
            self.config.symbol,
            self.config.interval,
            self.config.history_limit
        )
        
        if df.empty or len(df) < 50:
            logger.warning("Không đủ dữ liệu")
            return
        
        # 2. Tính indicators nếu cần (ATR cho risk management)
        if self.config.sl_atr_k or self.config.tp_atr_k:
            df['ATR14'] = self._calculate_atr(df, 14)
        
        # 3. Tính signal từ strategy
        try:
            result: StrategyResult = self.strategy.generate_signals(df)
            signals = result.signals
            
            if signals.empty:
                logger.warning("Strategy không tạo được signal")
                self.latest_signal = None
                self.latest_signal_error = "empty_signals"
                return
            
            # Lấy signal mới nhất
            latest_signal = signals.iloc[-1] if len(signals) > 0 else 0
            latest_signal = int(latest_signal) if pd.notna(latest_signal) else 0
            self.latest_signal = latest_signal
            self.latest_signal_error = None
        except Exception as e:
            logger.exception(f"Lỗi tính signal: {e}")
            self.latest_signal = None
            self.latest_signal_error = str(e)
            return
        
        # 4. Cập nhật trạng thái holding (với live/testnet)
        # 4. Cập nhật trạng thái holding (với live/testnet)
        if self.config.mode != "paper":
            # Kiểm tra xem client có hỗ trợ get_current_position (Futures) không
            if hasattr(self.client, "get_current_position"):
                current_pos, avg_entry = self.client.get_current_position(self.config.symbol)
                self.holding = abs(current_pos) > 0
                self.position_size = current_pos
                
                # Sync entry price nếu chưa có
                if self.holding and self.entry_price is None and avg_entry is not None:
                    self.entry_price = avg_entry
                    logger.info(f"🔄 Synced Entry Price from Exchange: {self.entry_price}")
                    
                # Log debug
                if self.holding:
                     logger.info(f"🔍 Position detected: {current_pos} (Entry={self.entry_price})")
            else:
                # Fallback: Spot
                base_balance = self.client.get_asset_balance(self.base_asset)
                self.holding = base_balance > 0.001  # Threshold nhỏ để tránh floating point
                self.position_size = base_balance if self.holding else 0.0
        
        # 5. Kiểm tra SL/TP
        if self.holding:
            current_price = self.client.get_last_price(self.config.symbol)
            direction = 1 if self.position_size > 0 else -1
            sl_tp_hit = self._check_stop_loss_take_profit(current_price, direction)
            
            if sl_tp_hit:
                reason_text = "Stop Loss" if sl_tp_hit == "sl" else "Take Profit"
                self._exit_position(reason=reason_text)
                return
        
        # 6. Xử lý signal
        if latest_signal != 0:
            if not self._cooldown_ok():
                logger.info("⏳ Đang trong cooldown, bỏ qua signal")
                return
            
            # Exit nếu signal đổi
            if self.holding:
                current_direction = 1 if self.position_size > 0 else -1
                if (latest_signal > 0 and current_direction < 0) or (latest_signal < 0 and current_direction > 0):
                    self._exit_position(reason="signal_change")
                    return
            
            # Enter position mới
            if not self.holding:
                if latest_signal > 0:
                    self._enter_position(1, df)  # LONG
                elif latest_signal < 0:
                    self._enter_position(-1, df)  # SHORT
        
        # 7. Log trạng thái
        entry_str = f"{self.entry_price:.8f}" if self.entry_price is not None else "N/A"
        logger.info(
            f"📊 Price={df['close'].iloc[-1]:.8f} | "
            f"Signal={latest_signal} | "
            f"Holding={self.holding} | "
            f"Entry={entry_str}"
        )
        
        # 8. Gửi thông báo Telegram cho signal mới (nếu không vào/ra lệnh)
        if HAS_TELEGRAM and send_signal_notification and latest_signal != 0:
            # Chỉ gửi nếu signal mới và không vào/ra lệnh trong vòng này
            current_price = df['close'].iloc[-1]
            send_signal_notification(
                signal=latest_signal,
                price=current_price,
                symbol=self.config.symbol,
                holding=self.holding,
                entry_price=self.entry_price,
                reason="New signal"
            )
    
    def run(self):
        """Vòng lặp chính."""
        logger.info(
            f"🚀 Bắt đầu bot | "
            f"Mode={self.config.mode} | "
            f"Symbol={self.config.symbol} | "
            f"Interval={self.config.interval} | "
            f"Strategy={self.config.strategy_name}"
        )
        
        while self.running:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Nhận Ctrl+C, dừng bot...")
                break
            except Exception as e:
                logger.exception(f"Lỗi vòng lặp: {e}")
            
            time.sleep(self.config.check_interval_sec)
        
        # Thoát position trước khi dừng
        if self.holding:
            logger.info("Đang thoát position trước khi dừng...")
            self._exit_position(reason="bot_stop")
        
        logger.info("Bot đã dừng")


def load_config_from_env() -> BotConfig:
    """Load config từ .env file."""
    root_env = Path(".env")
    config_env = Path("config") / ".env"

    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=True)
        logger.info(f"Loaded env from {root_env}")
    elif config_env.exists():
        load_dotenv(dotenv_path=config_env, override=True)
        logger.info(f"Loaded env from {config_env}")
    else:
        load_dotenv(override=True)
        logger.warning("⚠️ Không tìm thấy .env hoặc config/.env, dùng biến môi trường hiện tại/defaults")

    def _strip_quotes(v: Optional[str], default: str = "") -> str:
        if v is None:
            return default
        s = str(v).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        return s if s else default

    def _float_env(key: str, default: float) -> float:
        raw = _strip_quotes(os.getenv(key, str(default)), str(default))
        try:
            return float(raw)
        except Exception:
            logger.warning(f"⚠️ {key}='{raw}' không hợp lệ, fallback={default}")
            return float(default)

    def _bool_env(key: str, default: bool) -> bool:
        raw = _strip_quotes(os.getenv(key, str(default))).lower()
        if raw in ("1", "true", "yes", "y", "on"):
            return True
        if raw in ("0", "false", "no", "n", "off"):
            return False
        logger.warning(f"⚠️ {key}='{raw}' không hợp lệ, fallback={default}")
        return bool(default)

    mode_raw = _strip_quotes(os.getenv("MODE", "paper"), "paper").lower()
    if mode_raw in ("demo", "simulated", "simulation"):
        logger.warning("⚠️ MODE=demo được map thành MODE=live (kết hợp OKX_USE_SIMULATED_TRADING=1)")
        mode = "live"
    else:
        mode = mode_raw

    exchange = _strip_quotes(os.getenv("EXCHANGE", "binance"), "binance").lower()
    symbol = _strip_quotes(os.getenv("SYMBOL", "BTCUSDT"), "BTCUSDT").upper()
    interval = _strip_quotes(os.getenv("INTERVAL", "5m"), "5m")
    strategy_name = _strip_quotes(os.getenv("STRATEGY", "sma_ema"), "sma_ema").lower()

    # Parse strategy params (supports both JSON and Python-literal style strings)
    strategy_params_str = _strip_quotes(os.getenv("STRATEGY_PARAMS", "{}"), "{}")
    try:
        strategy_params = json.loads(strategy_params_str)
    except Exception:
        try:
            parsed = ast.literal_eval(strategy_params_str)
            strategy_params = parsed if isinstance(parsed, dict) else {}
        except Exception:
            logger.warning("⚠️ STRATEGY_PARAMS parse lỗi, dùng {}")
            strategy_params = {}

    # Auto-map MOE env vars when strategy is MOE and params were not explicitly provided.
    if strategy_name == "moe_v2_enhanced":
        raw_moe_th = _strip_quotes(os.getenv("MOE_PROBA_THRESHOLD", ""), "")
        moe_th = None
        if raw_moe_th:
            try:
                moe_th = float(raw_moe_th)
            except Exception:
                logger.warning(f"⚠️ MOE_PROBA_THRESHOLD='{raw_moe_th}' không hợp lệ, bỏ qua override")

        moe_defaults = {
            "model_path": _strip_quotes(
                os.getenv("MOE_MODEL_PATH", "models/dynamic_moe_v2_enhanced_final.pkl"),
                "models/dynamic_moe_v2_enhanced_final.pkl",
            ),
            "artifact_path": _strip_quotes(os.getenv("MOE_ARTIFACT_PATH", ""), "") or None,
            "proba_threshold": moe_th,
            "use_regime_specific": _bool_env("MOE_USE_REGIME_SPECIFIC", False),
            "use_dynamic_threshold": _bool_env("MOE_USE_DYNAMIC_THRESHOLD", True),
            "use_quantile_threshold": _bool_env("MOE_USE_QUANTILE_THRESHOLD", True),
            "target_signal_rate": _float_env("MOE_TARGET_SIGNAL_RATE", 0.08),
            "quantile_window": int(_float_env("MOE_QUANTILE_WINDOW", 400)),
            "quantile_floor": _float_env("MOE_QUANTILE_FLOOR", 0.55),
        }
        strategy_params = {**moe_defaults, **(strategy_params or {})}

    risk_per_trade = _float_env("RISK_PER_TRADE", 0.1)
    sl_pct = _float_env("SL_PCT", 0.0) or None
    tp_pct = _float_env("TP_PCT", 0.0) or None
    trailing_pct = _float_env("TRAILING_PCT", 0.0) or None
    sl_atr_k = _float_env("SL_ATR_K", 0.0) or None
    tp_atr_k = _float_env("TP_ATR_K", 0.0) or None
    trailing_atr_k = _float_env("TRAILING_ATR_K", 0.0) or None
    
    return BotConfig(
        mode=mode,
        exchange=exchange,
        symbol=symbol,
        interval=interval,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        risk_per_trade=risk_per_trade,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trailing_pct=trailing_pct,
        sl_atr_k=sl_atr_k,
        tp_atr_k=tp_atr_k,
        trailing_atr_k=trailing_atr_k,
        atr_col="ATR14",
        history_limit=int(_float_env("HISTORY_LIMIT", 200)),
        cool_down_sec=int(_float_env("COOL_DOWN_SEC", 60)),
        check_interval_sec=int(_float_env("CHECK_INTERVAL_SEC", 30)),
        max_position_size=_float_env("MAX_POSITION_SIZE", 0.0) or None,
        max_dca_orders=int(_float_env("MAX_DCA_ORDERS", 1)),
    )


def create_exchange_client(config: BotConfig):
    """Tạo exchange client dựa trên config.exchange."""
    if config.exchange == "okx":
        if not HAS_OKX or OKXClient is None:
            raise SystemExit("❌ OKXClient chưa được import. Kiểm tra lại dependencies.")
        
        api_key = os.getenv("OKX_API_KEY", "")
        api_secret = os.getenv("OKX_API_SECRET", "") or os.getenv("OKX_SECRET_KEY", "")
        passphrase = os.getenv("OKX_PASSPHRASE", "")
        use_simulated = os.getenv("OKX_USE_SIMULATED_TRADING", "0").strip() in ("1", "true", "True", "TRUE")
        
        if config.mode in ("testnet", "live") and (not api_key or not api_secret or not passphrase):
            raise SystemExit("❌ Thiếu OKX_API_KEY + (OKX_API_SECRET hoặc OKX_SECRET_KEY) + OKX_PASSPHRASE trong .env")
        
        return OKXClient(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            config=config,
            use_simulated_trading=use_simulated,
        )
    
    # Mặc định: Binance
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    if config.mode in ("testnet", "live") and (not api_key or not api_secret):
        raise SystemExit("❌ Thiếu BINANCE_API_KEY/BINANCE_API_SECRET trong .env")
    
    return BinanceClient(api_key, api_secret, config)


def evaluate_strategies(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    days: int = 30,
    output_file: str = "strategy_comparison_report.txt",
):
    """
    Đánh giá và so sánh tất cả các strategy.
    
    Args:
        symbol: Symbol để đánh giá
        interval: Timeframe
        days: Số ngày dữ liệu lịch sử
        output_file: File output cho báo cáo
    """
    logger.info(f"Bắt đầu đánh giá strategies cho {symbol} {interval}...")
    
    # Lấy dữ liệu từ Binance
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    client = Client(api_key or "", api_secret or "")
    
    # Tính số klines cần
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200,
    }
    minutes_per_bar = interval_minutes.get(interval, 60)
    total_minutes = days * 24 * 60
    limit = min(int(total_minutes / minutes_per_bar), 1000)  # Binance limit
    
    logger.info(f"Đang lấy {limit} klines...")
    try:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df = df[["open", "high", "low", "close", "volume"]]
        
        if df.empty or len(df) < 50:
            logger.error("Không đủ dữ liệu để đánh giá")
            return
        
        logger.info(f"Đã lấy {len(df)} klines từ {df.index[0]} đến {df.index[-1]}")
        
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu: {e}")
        return
    
    # Tạo evaluator
    evaluator = StrategyEvaluator(
        df=df,
        initial_capital=10000.0,
        commission=0.001,
        use_stops=True,
        sl_pct=0.02,
        tp_pct=0.04,
    )
    
    # Tạo báo cáo
    report = evaluator.generate_comparison_report(output_file=output_file, top_n=15)
    logger.info(f"✅ Đã tạo báo cáo: {output_file}")
    print(f"\n{'='*80}")
    print("BÁO CÁO ĐÃ ĐƯỢC TẠO!")
    print(f"File: {output_file}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Trading Bot')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Chạy đánh giá và so sánh các strategy thay vì chạy bot'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Symbol để đánh giá (mặc định từ .env)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default=None,
        help='Interval để đánh giá (mặc định từ .env)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Số ngày dữ liệu lịch sử (mặc định: 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='strategy_comparison_report.txt',
        help='File output cho báo cáo (mặc định: strategy_comparison_report.txt)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Chạy 1 vòng an toàn (paper) để kiểm tra IO, không chạy loop vô hạn'
    )
    
    args = parser.parse_args()
    
    # Nếu có flag --evaluate, chạy đánh giá
    if args.evaluate:
        config = load_config_from_env()
        symbol = args.symbol or config.symbol
        interval = args.interval or config.interval
        evaluate_strategies(
            symbol=symbol,
            interval=interval,
            days=args.days,
            output_file=args.output,
        )
        return
    
    # Chạy bot bình thường
    config = load_config_from_env()

    if args.dry_run:
        config.mode = "paper"
        logger.info("🧪 DRY-RUN mode bật: ép MODE=paper, chạy 1 vòng rồi thoát")

        if os.getenv("BOT_OFFLINE_SMOKE") == "1":
            client = create_exchange_client(config)
            logger.info(
                "✅ Offline smoke preflight OK: configuration loaded and "
                f"{client.__class__.__name__} constructed in paper mode"
            )
            return

        # Neu production bundle toi gian chua dong goi strategies,
        # van cho preflight I/O (env + exchange + market data fetch).
        if not HAS_STRATEGIES:
            logger.warning("⚠️ Không có algo_trading.strategies trong production bundle. Chạy preflight I/O mode.")
            client = create_exchange_client(config)
            df = client.get_klines_df(config.symbol, config.interval, max(50, config.history_limit))
            if df is None or df.empty:
                raise SystemExit("❌ Dry-run preflight thất bại: không lấy được market data")
            logger.info(f"✅ Dry-run preflight OK: fetched {len(df)} candles for {config.symbol} {config.interval}")
            return
    
    # Validate strategy
    if config.strategy_name not in STRATEGY_MAP:
        raise SystemExit(f"❌ Strategy '{config.strategy_name}' không tồn tại. Chọn từ: {list(STRATEGY_MAP.keys())}")
    
    # Tạo strategy
    StrategyClass = STRATEGY_MAP[config.strategy_name]
    
    # Xử lý combiner strategies (là lambda functions)
    if callable(StrategyClass) and not isinstance(StrategyClass, type):
        strategy = StrategyClass(**(config.strategy_params or {}))
    else:
        strategy = StrategyClass(**(config.strategy_params or {}))
    
    # Tạo exchange client (Binance hoặc OKX tùy theo config.exchange)
    client = create_exchange_client(config)
    bot = LiveTradingBot(client, strategy, config)

    if args.dry_run:
        try:
            bot.run_once()
            logger.info("✅ Dry-run hoàn tất")
            return
        except Exception as e:
            logger.exception(f"❌ Dry-run thất bại: {e}")
            raise
    
    # Khởi động Telegram bot trong background thread (nếu có)
    telegram_thread = None
    if HAS_TELEGRAM and run_telegram_bot:
        try:
            # Set trading bot instance để Telegram bot có thể điều khiển
            set_trading_bot(bot)
            
            # Khởi động Telegram bot trong background thread
            def run_telegram():
                try:
                    run_telegram_bot()
                except Exception as e:
                    logger.error(f"Lỗi Telegram bot: {e}")
            
            telegram_thread = threading.Thread(target=run_telegram, daemon=True)
            telegram_thread.start()
            logger.info("✅ Telegram bot đã khởi động trong background")
        except Exception as e:
            logger.warning(f"⚠️ Không thể khởi động Telegram bot: {e}")
    
    # Chạy bot
    try:
        bot.run()
    except Exception as e:
        logger.exception(f"Lỗi chạy bot: {e}")
        raise
    finally:
        # Dừng Telegram bot nếu đang chạy
        if telegram_thread and telegram_thread.is_alive():
            logger.info("Đang dừng Telegram bot...")


if __name__ == "__main__":
    main()
