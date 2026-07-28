"""
Algo trading mẫu sử dụng Binance Spot Testnet (API key miễn phí)
- Chiến lược: Simple Moving Average (SMA) crossover (giao cắt SMA nhanh/chậm)
- Chế độ: paper (không gửi lệnh), testnet (gửi lệnh lên Spot Testnet), live (KHÔNG khuyến nghị)

Lưu ý cực quan trọng:
- Đây là ví dụ mang tính học tập, không phải lời khuyên đầu tư.
- Testnet dùng API miễn phí: https://testnet.binance.vision/ (đăng nhập GitHub để tạo API key)
- Không dùng key thật trên testnet và ngược lại.
- Khi chạy ở chế độ testnet, bạn vẫn cần có số dư testnet USDT/BNB... (tham khảo mục Faucet/Reset balance trên testnet UI nếu có).

Chạy nhanh:
1) pip install -r requirements.txt
2) Tạo file .env cùng thư mục, điền các biến:
   MODE=testnet
   BINANCE_API_KEY=your_testnet_api_key
   BINANCE_API_SECRET=your_testnet_api_secret
   SYMBOL=BTCUSDT
   INTERVAL=5m
   SMA_FAST=20
   SMA_SLOW=50
   RISK_PER_TRADE=0.1
   HISTORY_LIMIT=120
   COOL_DOWN_SEC=60
3) python -m algo_trading.live.binance_sma_bot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv

# -----------------------------
# Cấu hình logging cơ bản
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Dataclass cấu hình chiến lược/bot
# -----------------------------
@dataclass
class Config:
    # Chế độ chạy: "paper", "testnet", hoặc "live"
    mode: str = "paper"
    # Cặp giao dịch, ví dụ BTCUSDT
    symbol: str = "BTCUSDT"
    # Khung thời gian nến cho dữ liệu kline, ví dụ "5m", "1h"
    interval: str = "5m"
    # Tham số SMA
    sma_fast: int = 20
    sma_slow: int = 50
    # Rủi ro trên mỗi lệnh (theo phần trăm số dư quote), ví dụ 0.1 = 10%
    risk_per_trade: float = 0.1
    # Số nến cần để có đủ dữ liệu tính chỉ báo (nên >= sma_slow + đệm)
    history_limit: int = 120
    cool_down_sec: int = 60


# -----------------------------
# Tiện ích định dạng bước/khối lượng theo quy tắc sàn
# -----------------------------
@dataclass
class SymbolFilters:
    step_size: float
    min_qty: float
    min_notional: float
    tick_size: float


def floor_to_step(value: float, step: float) -> float:
    """Làm tròn xuống theo step (LOT_SIZE stepSize)."""
    if step <= 0:
        return value
    return np.floor(value / step) * step


def round_to_tick(value: float, tick: float) -> float:
    """Làm tròn theo tick (PRICE_FILTER tickSize)."""
    if tick <= 0:
        return value
    # làm tròn về bội số gần nhất của tick
    return np.round(value / tick) * tick


# -----------------------------
# Lớp bọc client Binance (paper/testnet/live)
# -----------------------------
class BinanceSpot:
    def __init__(self, api_key: Optional[str], api_secret: Optional[str], cfg: Config):
        self.cfg = cfg
        # Ở paper mode, không cần client thực; vẫn có thể dùng public endpoints bằng key rỗng
        self.client = Client(api_key or "", api_secret or "")
        # Chuyển sang Spot Testnet nếu chọn testnet
        if cfg.mode == "testnet":
            # python-binance: đặt API_URL để dùng Spot Testnet
            # Lưu ý: một số bản python-binance mới có hỗ trợ testnet flag, nhưng cách dưới đây phổ biến và rõ ràng
            self.client.API_URL = "https://testnet.binance.vision/api"
            logger.info("Đang dùng Binance Spot Testnet API")
        elif cfg.mode == "live":
            logger.warning("Bạn đang ở chế độ LIVE. Hãy cẩn thận và chắc chắn bạn hiểu rủi ro!")
        else:
            logger.info("Đang chạy ở chế độ PAPER (không gửi lệnh)")

        self._symbol_filters_cache: Dict[str, SymbolFilters] = {}

    # -------- Public data --------
    def get_klines_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Lấy dữ liệu nến (kline) và trả về DataFrame đã chuẩn hóa."""
        raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        # Chỉ cần cột thời gian & OHLCV cần thiết
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df

    def get_last_price(self, symbol: str) -> float:
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    # -------- Trading utils --------
    def _fetch_symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]
        info = self.client.get_symbol_info(symbol)
        if not info:
            raise ValueError(f"Không tìm thấy thông tin symbol {symbol}")
        step_size = 0.0
        min_qty = 0.0
        min_notional = 0.0
        tick_size = 0.0
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])  # bước khối lượng
                min_qty = float(f["minQty"])      # khối lượng tối thiểu
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f.get("minNotional", 0))
            elif f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])  # bước giá
        filters = SymbolFilters(step_size=step_size, min_qty=min_qty, min_notional=min_notional, tick_size=tick_size)
        self._symbol_filters_cache[symbol] = filters
        return filters

    def get_asset_balance(self, asset: str) -> float:
        # Ở paper mode, không có số dư thật; có thể trả về 0 để bot chỉ mô phỏng tín hiệu
        if self.cfg.mode == "paper":
            return 0.0
        bal = self.client.get_asset_balance(asset=asset)
        if not bal:
            return 0.0
        return float(bal.get("free", 0))

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Gửi lệnh market. Nếu ở paper, chỉ log hành động.
        side: "BUY" hoặc "SELL"
        """
        if quantity <= 0:
            raise ValueError("Khối lượng (quantity) phải > 0")

        filters = self._fetch_symbol_filters(symbol)
        qty = floor_to_step(quantity, filters.step_size)
        if qty < max(filters.min_qty, 0):
            raise ValueError(
                f"Khối lượng sau khi làm tròn ({qty}) < minQty ({filters.min_qty}). Tăng vốn/risk_per_trade hoặc đổi symbol."
            )

        if self.cfg.mode == "paper":
            logger.info(f"[PAPER] {side} {qty} {symbol} (market)")
            return {"paper": True, "side": side, "symbol": symbol, "executedQty": qty}

        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=qty,
            )
            logger.info(f"Đã gửi lệnh {side} {qty} {symbol} (market)")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Lỗi gửi lệnh: {e}")
            raise


# -----------------------------
# Logic chiến lược SMA crossover
# -----------------------------
class SmaCrossoverStrategy:
    """Sinh tín hiệu khi SMA nhanh cắt lên/cắt xuống SMA chậm.
    - Tín hiệu mua (BUY) khi SMA_FAST từ dưới cắt lên SMA_SLOW.
    - Tín hiệu bán (SELL) khi SMA_FAST từ trên cắt xuống SMA_SLOW.

    Ta chỉ giữ tối đa 1 vị thế (nắm giữ asset cơ sở) hoặc flat.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_signal_time: Optional[datetime] = None 
        self.holding: bool = False  

    def compute_signals(self, df: pd.DataFrame) -> Tuple[Optional[str], pd.Series, pd.Series]:
        """Tính SMA và xác định tín hiệu cho cây nến mới nhất.
        Trả về: (signal, sma_fast_series, sma_slow_series)
        signal ∈ {"BUY", "SELL", None}
        """
        if len(df) < max(self.cfg.sma_fast, self.cfg.sma_slow):
            return None, pd.Series(dtype=float), pd.Series(dtype=float)

        df = df.copy()
        df["sma_fast"] = df["close"].rolling(self.cfg.sma_fast).mean()
        df["sma_slow"] = df["close"].rolling(self.cfg.sma_slow).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Điều kiện giao cắt lên: trước đó fast <= slow, hiện tại fast > slow
        golden = prev["sma_fast"] <= prev["sma_slow"] and last["sma_fast"] > last["sma_slow"]
        # Điều kiện giao cắt xuống: trước đó fast >= slow, hiện tại fast < slow
        death = prev["sma_fast"] >= prev["sma_slow"] and last["sma_fast"] < last["sma_slow"]
        signal: Optional[str] = None
        if golden and not self.holding:
            signal = "BUY"
        elif death and self.holding:
            signal = "SELL"

        return signal, df["sma_fast"], df["sma_slow"]

    def cooldown_ok(self) -> bool:
        if self.last_signal_time is None:
            return True
        return (datetime.now(timezone.utc) - self.last_signal_time).total_seconds() >= self.cfg.cool_down_sec

    def mark_signal_time(self):
        self.last_signal_time = datetime.now(timezone.utc)


# -----------------------------
# Bot điều phối: lấy dữ liệu -> tính tín hiệu -> quản trị rủi ro -> đặt lệnh
# -----------------------------
class SmaBot:
    def __init__(self, client: BinanceSpot, strategy: SmaCrossoverStrategy, cfg: Config):
        self.client = client
        self.strategy = strategy
        self.cfg = cfg
        self.base_asset, self.quote_asset = self._split_symbol(cfg.symbol)

    @staticmethod
    def _split_symbol(symbol: str) -> Tuple[str, str]:
        # Heuristic tách base/quote phổ biến (USDT, BUSD, USDC, BTC, ETH...) – tối giản cho ví dụ
        for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH", "FDUSD", "TUSD"]:
            if symbol.endswith(quote):
                return symbol.replace(quote, ""), quote
        # fallback (không chính xác cho mọi trường hợp)
        return symbol[:3], symbol[3:]

    def position_state(self) -> bool:
        """Đơn giản: coi như đang giữ nếu số dư base_asset > 0 (đối với testnet/live).
        Ở paper mode, dùng self.strategy.holding.
        """
        if self.cfg.mode == "paper":
            return self.strategy.holding
        bal = self.client.get_asset_balance(self.base_asset)
        return bal > 0

    def size_position(self, last_price: float) -> float:
        """Tính khối lượng mua theo risk_per_trade dựa trên số dư quote.
        - Khối lượng (số lượng coin) = (quote_balance * risk) / last_price
        - Đảm bảo thoả minNotional và làm tròn theo stepSize.
        """
        if self.cfg.mode == "paper":
            quote_balance = 1000.0
        else:
            quote_balance = self.client.get_asset_balance(self.quote_asset)
        notional = quote_balance * self.cfg.risk_per_trade
        qty_raw = notional / last_price if last_price > 0 else 0

        filters = self.client._fetch_symbol_filters(self.cfg.symbol)
        qty = floor_to_step(qty_raw, filters.step_size)
        if qty * last_price < max(filters.min_notional, 0):
            min_qty_for_notional = (filters.min_notional / last_price) if last_price > 0 else 0
            qty = floor_to_step(max(qty, min_qty_for_notional), filters.step_size)
        return max(qty, 0)

    def run_once(self):
        df = self.client.get_klines_df(self.cfg.symbol, self.cfg.interval, self.cfg.history_limit)
        if df.empty:
            logger.warning("Không lấy được dữ liệu kline.")
            return

        signal, sma_fast, sma_slow = self.strategy.compute_signals(df)
        self.strategy.holding = self.position_state()
        # 4) Kiểm tra cooldown để tránh vào/ra liên tiếp
        if signal:
            if not self.strategy.cooldown_ok():
                logger.info("Trong thời gian cooldown, bỏ qua tín hiệu.")
                return

        # 5) Nếu có tín hiệu thì tính khối lượng và gửi lệnh
        if signal == "BUY" and not self.strategy.holding:
            last_price = self.client.get_last_price(self.cfg.symbol)
            qty = self.size_position(last_price)
            if qty <= 0:
                logger.info("Khối lượng tính được <= 0 hoặc không đạt yêu cầu minNotional. Bỏ qua.")
                return
            try:
                self.client.place_market_order(self.cfg.symbol, "BUY", qty)
                self.strategy.holding = True
                self.strategy.mark_signal_time()
            except Exception:
                logger.exception("Gửi lệnh BUY thất bại")

        elif signal == "SELL" and self.strategy.holding:
            # Với ví dụ này, ta thoát toàn bộ vị thế (bán hết base asset hiện có)
            if self.cfg.mode == "paper":
                # Ở paper: giả lập bán với khối lượng như đã mua (không lưu trữ ước lượng cụ thể ở ví dụ)
                qty = 1.0  # giá trị mô phỏng
            else:
                qty = self.client.get_asset_balance(self.base_asset)
            if qty <= 0:
                logger.info("Không có khối lượng để bán. Bỏ qua.")
                return
            try:
                self.client.place_market_order(self.cfg.symbol, "SELL", qty)
                self.strategy.holding = False
                self.strategy.mark_signal_time()
            except Exception:
                logger.exception("Gửi lệnh SELL thất bại")
        else:
            logger.info("Không có tín hiệu hoặc trạng thái vị thế không phù hợp để hành động.")

        # 6) Log tham khảo SMA cho nến mới nhất
        if not sma_fast.empty and not sma_slow.empty:
            logger.info(
                f"Close={df.iloc[-1]['close']:.2f} | SMA{self.cfg.sma_fast}={sma_fast.iloc[-1]:.2f} | "
                f"SMA{self.cfg.sma_slow}={sma_slow.iloc[-1]:.2f} | holding={self.strategy.holding}"
            )


# -----------------------------
# Hàm main: tải .env -> dựng bot -> vòng lặp theo chu kỳ nến
# -----------------------------

def load_config_from_env() -> Config:
    load_dotenv(override=True)
    mode = os.getenv("MODE", "paper").lower()
    symbol = os.getenv("SYMBOL", "BTCUSDT").upper()
    interval = os.getenv("INTERVAL", "5m")
    sma_fast = int(os.getenv("SMA_FAST", "20"))
    sma_slow = int(os.getenv("SMA_SLOW", "50"))
    risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.1"))
    history_limit = int(os.getenv("HISTORY_LIMIT", "120"))
    cool_down_sec = int(os.getenv("COOL_DOWN_SEC", "60"))

    if sma_fast >= sma_slow:
        logger.warning("SMA_FAST nên nhỏ hơn SMA_SLOW để chiến lược hợp lý. Sẽ vẫn tiếp tục chạy.")

    return Config(
        mode=mode,
        symbol=symbol,
        interval=interval,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        risk_per_trade=risk_per_trade,
        history_limit=history_limit,
        cool_down_sec=cool_down_sec,
    )


def main():
    cfg = load_config_from_env()

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if cfg.mode in ("testnet", "live") and (not api_key or not api_secret):
        raise SystemExit("Thiếu BINANCE_API_KEY/BINANCE_API_SECRET trong .env")

    client = BinanceSpot(api_key, api_secret, cfg)
    strategy = SmaCrossoverStrategy(cfg)
    bot = SmaBot(client, strategy, cfg)

    logger.info(
        f"Bắt đầu chạy bot | mode={cfg.mode} symbol={cfg.symbol} interval={cfg.interval} "
        f"SMA({cfg.sma_fast},{cfg.sma_slow}) risk={cfg.risk_per_trade} history={cfg.history_limit}"
    )

    # Vòng lặp chính: chạy theo chu kỳ nến
    # Nếu interval=5m, bạn có thể đồng bộ chạy mỗi 5 phút sau khi nến đóng.
    # Đơn giản: chạy mỗi 30s để lấy dữ liệu mới và kiểm tra tín hiệu.
    while True:
        try:
            bot.run_once()
        except Exception as e:
            logger.exception(f"Lỗi vòng lặp: {e}")
        # Ngủ một chút trước vòng kế tiếp
        time.sleep(30)


if __name__ == "__main__":
    main()

