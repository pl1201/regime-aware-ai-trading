"""
OKX Exchange Client - Implement OKX API v5
Hỗ trợ paper trading (không dùng tiền thật)
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from urllib.parse import urlencode
import logging
import requests
import pandas as pd
import numpy as np

from algo_trading.live.exchange_base import ExchangeClient, SymbolFilters
from algo_trading.config import BotConfig

logger = logging.getLogger(__name__)


class OKXClient(ExchangeClient):
    """OKX API v5 client."""
    
    BASE_URL = "https://www.okx.com"
    
    def __init__(
        self,
        api_key: Optional[str],
        api_secret: Optional[str],
        passphrase: Optional[str],
        config: BotConfig,
        use_simulated_trading: bool = False
    ):
        """
        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX passphrase (bắt buộc)
            config: BotConfig với exchange="okx"
            use_simulated_trading: Nếu True, thêm header x-simulated-trading: 1 để dùng Demo Account
        """
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.passphrase = passphrase or ""
        self.config = config
        self.use_simulated_trading = use_simulated_trading
        self._symbol_filters_cache: Dict[str, SymbolFilters] = {}
        self.inst_type = "SWAP"
        
        if use_simulated_trading:
            logger.info("🎮 Đang dùng OKX Simulated Trading (Demo Account) - Header x-simulated-trading: 1")
        elif config.mode == "live":
            logger.warning("🔴 BẠN ĐANG Ở CHẾ ĐỘ LIVE! Hãy cẩn thận!")
        else:
            logger.info("📄 Đang chạy ở chế độ PAPER (không gửi lệnh thật)")
    
    def _convert_symbol(self, symbol: str) -> str:
        # Tách base và quote
        for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH", "FDUSD", "TUSD"]:
            if symbol.endswith(quote):
                base = symbol.replace(quote, "")
                inst = f"{base}-{quote}"
                if self.inst_type == "SWAP":
                    return f"{inst}-SWAP"
                return inst
        # Nếu không match, thử tách 3 ký tự đầu
        if len(symbol) >= 6:
            base, quote = symbol[:3], symbol[3:]
            inst = f"{base}-{quote}"
            if self.inst_type == "SWAP":
                return f"{inst}-SWAP"
            return inst
        return symbol
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval từ 1h -> 1H, 1d -> 1D."""
        # OKX yêu cầu H và D viết hoa
        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
            "1d": "1D", "1w": "1W", "1M": "1M",
        }
        return interval_map.get(interval.lower(), interval.upper())
    
    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        """Chuyển đổi sang float một cách an toàn, xử lý chuỗi rỗng."""
        if val is None or val == "":
            return default
        try:
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    return default
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def _make_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """
        Tạo signature cho OKX API.
        Theo docs OKX v5: prehash = timestamp + method.upper() + request_path + body_json
        """
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(prehash, encoding='utf8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """
        Gửi request đến OKX API.
        - Với GET private: query string phải nằm trong cả URL và request_path để ký.
        - Với POST private: body là JSON string, không có query.
        """
        method = method.upper()
        params = params or {}

        # Timestamp ISO 8601
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        request_path = endpoint
        body_str = ""

        if method == "GET" and params:
            query = urlencode(params)
            request_path = f"{endpoint}?{query}"
            url = f"{self.BASE_URL}{request_path}"
        else:
            url = f"{self.BASE_URL}{endpoint}"

        if method == "POST":
            body_str = json.dumps(params) if params else ""

        # Headers
        headers = {
            "Content-Type": "application/json",
        }

        if self.use_simulated_trading:
            headers["x-simulated-trading"] = "1"

        # Private endpoints cần ký
        if signed:
            if not self.api_key or not self.api_secret or not self.passphrase:
                logger.warning("⚠️ Thiếu OKX credentials, chỉ có thể dùng public endpoints")
                return {}

            signature = self._make_signature(timestamp, method, request_path, body_str)
            headers.update(
                {
                    "OK-ACCESS-KEY": self.api_key,
                    "OK-ACCESS-SIGN": signature,
                    "OK-ACCESS-TIMESTAMP": timestamp,
                    "OK-ACCESS-PASSPHRASE": self.passphrase,
                }
            )

        try:
            if method == "GET":
                # params đã nằm trong request_path nếu có
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                json_body = json.loads(body_str) if body_str else {}
                response = requests.post(url, json=json_body, headers=headers, timeout=10)
            else:
                raise ValueError(f"Method {method} không được hỗ trợ")

            # Không raise_for_status ngay, để đọc body kể cả khi 4xx/5xx
            try:
                data = response.json()
            except ValueError:
                logger.error(
                    f"Lỗi request OKX API (non-JSON): status={response.status_code}, body={response.text}"
                )
                response.raise_for_status()
                return {}

            if response.status_code != 200:
                okx_code = data.get("code")
                okx_msg = data.get("msg")
                logger.error(
                    f"OKX HTTP error status={response.status_code}, code={okx_code}, msg={okx_msg}"
                )
                return {}

            # OKX wrap response trong {"code": "0", "data": [...]}
            if data.get("code") != "0":
                error_code = data.get("code", "unknown")
                error_msg = data.get("msg", "Unknown error")
                error_data = data.get("data", [])
                logger.error(
                    f"OKX API error: code={error_code}, msg={error_msg}"
                    + (f", data={error_data}" if error_data else "")
                )
                # Trả về dict với error info để caller có thể xử lý
                return {"error": True, "code": error_code, "msg": error_msg, "data": error_data}

            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi request OKX API: {e}")
            return {}
        except Exception as e:
            logger.error(f"Lỗi xử lý response OKX: {e}")
            return {}
    
    def get_klines_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Lấy dữ liệu kline và trả về DataFrame."""
        try:
            okx_symbol = self._convert_symbol(symbol)
            okx_interval = self._convert_interval(interval)
            
            endpoint = "/api/v5/market/candles"
            params = {
                "instId": okx_symbol,
                "bar": okx_interval,
                "limit": min(limit, 300)  # OKX limit 300
            }
            
            data = self._make_request("GET", endpoint, params=params, signed=False)
            
            if not data:
                logger.warning(f"Không lấy được klines cho {okx_symbol}")
                return pd.DataFrame()
            
            data.reverse()
            
            rows = []
            for candle in data:
                rows.append({
                    "open_time": pd.to_datetime(int(candle[0]), unit="ms", utc=True),
                    "open": self._safe_float(candle[1]),
                    "high": self._safe_float(candle[2]),
                    "low": self._safe_float(candle[3]),
                    "close": self._safe_float(candle[4]),
                    "volume": self._safe_float(candle[5]),
                })
            
            df = pd.DataFrame(rows)
            if df.empty:
                return pd.DataFrame()
            
            df.set_index("open_time", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Lỗi lấy klines: {e}")
            return pd.DataFrame()
    
    def get_last_price(self, symbol: str) -> float:
        """Lấy giá hiện tại."""
        try:
            okx_symbol = self._convert_symbol(symbol)
            
            endpoint = "/api/v5/market/ticker"
            params = {"instId": okx_symbol}
            
            data = self._make_request("GET", endpoint, params=params, signed=False)
            
            if not data or len(data) == 0:
                logger.warning(f"Không lấy được giá cho {okx_symbol}")
                return 0.0
            
            # OKX ticker format: {"last": "50000.0", ...}
            last_price = self._safe_float(data[0].get("last", 0))
            return last_price
        except Exception as e:
            logger.error(f"Lỗi lấy giá: {e}")
            return 0.0
    
    def _fetch_symbol_filters(self, symbol: str) -> SymbolFilters:
        """Lấy filters từ exchange info."""
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]
        
        try:
            okx_symbol = self._convert_symbol(symbol)
            
            endpoint = "/api/v5/public/instruments"
            # Dùng SWAP để lấy thông tin hợp đồng futures
            params = {"instType": self.inst_type, "instId": okx_symbol}
            
            data = self._make_request("GET", endpoint, params=params, signed=False)
            
            if not data or len(data) == 0:
                logger.warning(f"Không tìm thấy symbol {okx_symbol}")
                return SymbolFilters(0.0, 0.0, 0.0, 0.0)
            
            info = data[0]
            
            # OKX format
            step_size = float(info.get("lotSz", 0))  # Lot size
            min_qty = float(info.get("minSz", 0))  # Min size
            tick_size = float(info.get("tickSz", 0))  # Tick size
            min_notional = float(info.get("minSz", 0)) * float(info.get("tickSz", 0))  # Approximate
            
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
            endpoint = "/api/v5/account/balance"
            params = {"ccy": asset}
            
            data = self._make_request("GET", endpoint, params=params, signed=True)
            
            if not data or len(data) == 0:
                return 0.0
            
            # OKX balance format: {"details": [{"ccy": "USDT", "availBal": "1000.0", ...}]}
            details = data[0].get("details", [])
            for detail in details:
                if detail.get("ccy") == asset:
                    return self._safe_float(detail.get("availBal", 0))
            
            return 0.0
        except Exception as e:
            logger.error(f"Lỗi lấy số dư: {e}")
            return 0.0

    def get_current_position(self, symbol: str) -> Tuple[float, Optional[float]]:
        """
        Lấy vị thế hiện tại (Position Size) và Entry Price.
        Trả về: (position_size, avg_entry_price)
          - position_size: Số dương (Long), âm (Short), 0 (None)
          - avg_entry_price: Giá vào lệnh trung bình (hoặc None)
        """
        if self.config.mode == "paper":
            return 0.0, None
            
        try:
            okx_symbol = self._convert_symbol(symbol)
            endpoint = "/api/v5/account/positions"
            params = {"instId": okx_symbol}
            
            data = self._make_request("GET", endpoint, params=params, signed=True)
            
            if not data or len(data) == 0:
                return 0.0, None
            
            # OKX position data
            total_pos = 0.0
            avg_px = 0.0
            total_sz = 0.0
            
            for pos_data in data:
                sz = self._safe_float(pos_data.get("pos", 0))
                # Side: long/short/net
                side = pos_data.get("posSide", "net")
                px = self._safe_float(pos_data.get("avgPx", 0))
                
                if side == "short":
                    signed_sz = -abs(sz)
                elif side == "long":
                    signed_sz = abs(sz)
                else: # net
                    signed_sz = sz
                
                total_pos += signed_sz
                
                # Tính weighted average price nếu có nhiều positions (ít khi xảy ra với bot đơn)
                if abs(signed_sz) > 0:
                     avg_px = (avg_px * total_sz + px * abs(signed_sz)) / (total_sz + abs(signed_sz))
                     total_sz += abs(signed_sz)

            return total_pos, avg_px if total_pos != 0 else None
        except Exception as e:
            logger.error(f"Lỗi lấy position: {e}")
            return 0.0, None
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Đặt lệnh market."""
        if quantity <= 0:
            raise ValueError("Quantity phải > 0")
        
        if self.config.mode == "paper":
            logger.info(f"[PAPER] {side} {quantity} {symbol}")
            return {"paper": True, "side": side, "symbol": symbol, "executedQty": quantity}
        
        try:
            okx_symbol = self._convert_symbol(symbol)
            filters = self._fetch_symbol_filters(symbol)
            
            # Điều chỉnh quantity theo step_size
            if filters.step_size > 0:
                qty = np.floor(quantity / filters.step_size) * filters.step_size
            else:
                qty = quantity
            
            # Validate quantity
            if qty < filters.min_qty:
                logger.warning(
                    f"Quantity {qty} < minQty {filters.min_qty} cho {symbol}. "
                    f"Tăng lên minQty: {filters.min_qty}"
                )
                qty = filters.min_qty
            
            # Kiểm tra balance trước khi gửi lệnh
            if side.upper() == "BUY":
                # Với SWAP futures, cần kiểm tra margin available
                try:
                    balance = self.get_asset_balance("USDT")
                    # Ước tính margin cần: qty * price (lấy giá hiện tại)
                    current_price = self.get_last_price(symbol)
                    if current_price > 0:
                        estimated_margin = qty * current_price
                        if estimated_margin > balance * 0.95:  # Giữ 5% buffer
                            logger.warning(
                                f"Estimated margin {estimated_margin:.2f} > available balance {balance:.2f}. "
                                f"Giảm quantity từ {qty} xuống {balance * 0.95 / current_price:.6f}"
                            )
                            # Giảm quantity để fit với balance
                            qty = min(qty, (balance * 0.95 / current_price))
                            # Điều chỉnh lại theo step_size
                            if filters.step_size > 0:
                                qty = np.floor(qty / filters.step_size) * filters.step_size
                            if qty < filters.min_qty:
                                raise ValueError(
                                    f"Không đủ balance. Cần ít nhất {filters.min_qty} contracts "
                                    f"nhưng chỉ có margin cho {qty} contracts"
                                )
                except Exception as e:
                    logger.warning(f"Không thể kiểm tra balance: {e}")
            
            # OKX side: "buy" hoặc "sell"
            okx_side = "buy" if side.upper() == "BUY" else "sell"
            
            endpoint = "/api/v5/trade/order"
            params = {
                "instId": okx_symbol,
                # Dùng cross margin cho SWAP futures
                "tdMode": "cross",
                "side": okx_side,
                "ordType": "market",
                "sz": str(qty),  # Size (quantity/contracts)
            }
            
            logger.info(f"Gửi lệnh {side} {qty} {symbol} (OKX: {okx_symbol})")
            data = self._make_request("POST", endpoint, params=params, signed=True)
            
            # Kiểm tra nếu có lỗi trong response
            if isinstance(data, dict) and data.get("error"):
                error_code = data.get("code", "unknown")
                error_msg = data.get("msg", "Unknown error")
                error_data = data.get("data", [])
                
                # Log chi tiết lỗi
                logger.error(
                    f"OKX order failed: code={error_code}, msg={error_msg}"
                    + (f", data={error_data}" if error_data else "")
                )
                
                # Một số lỗi có thể retry với quantity nhỏ hơn
                if error_code in ["1", "51000", "51001"]:  # All operations failed, insufficient margin, etc.
                    if qty > filters.min_qty * 2:
                        # Thử lại với quantity nhỏ hơn 50%
                        retry_qty = max(filters.min_qty, qty * 0.5)
                        if filters.step_size > 0:
                            retry_qty = np.floor(retry_qty / filters.step_size) * filters.step_size
                        logger.info(f"Retry với quantity nhỏ hơn: {retry_qty}")
                        params["sz"] = str(retry_qty)
                        data = self._make_request("POST", endpoint, params=params, signed=True)
                        if isinstance(data, dict) and data.get("error"):
                            raise ValueError(f"OKX order failed sau retry: {data.get('msg')}")
                        qty = retry_qty
                    else:
                        raise ValueError(f"OKX order failed: {error_msg} (code={error_code})")
                else:
                    raise ValueError(f"OKX order failed: {error_msg} (code={error_code})")
            
            if not data or len(data) == 0:
                raise ValueError("OKX không trả về order response")
            
            order = data[0]
            logger.info(f"OK Da gui lenh {side} {qty} {symbol}")
            
            # Convert OKX format về format tương tự Binance
            return {
                "orderId": order.get("ordId"),
                "symbol": symbol,
                "side": side,
                "executedQty": qty,
                "status": "FILLED" if order.get("state") == "filled" else "PENDING",
            }
        except ValueError as e:
            # ValueError đã có message chi tiết từ code trên
            logger.error(f"Loi gui lenh: {e}")
            raise
        except Exception as e:
            logger.error(f"Loi gui lenh (unexpected): {e}", exc_info=True)
            raise ValueError(f"Loi gui lenh OKX: {str(e)}") from e
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Đặt lệnh limit (cho SL/TP)."""
        if quantity <= 0:
            raise ValueError("Quantity phải > 0")
        
        if self.config.mode == "paper":
            logger.info(f"[PAPER] {side} {quantity} {symbol} @ {price} (LIMIT)")
            return {"paper": True, "side": side, "symbol": symbol, "executedQty": quantity, "price": price}
        
        try:
            okx_symbol = self._convert_symbol(symbol)
            filters = self._fetch_symbol_filters(symbol)
            
            # Điều chỉnh quantity và price
            qty = np.floor(quantity / filters.step_size) * filters.step_size if filters.step_size > 0 else quantity
            price = np.round(price / filters.tick_size) * filters.tick_size if filters.tick_size > 0 else price
            
            if qty < filters.min_qty:
                raise ValueError(f"Quantity {qty} < minQty {filters.min_qty}")
            
            okx_side = "buy" if side.upper() == "BUY" else "sell"
            
            endpoint = "/api/v5/trade/order"
            params = {
                "instId": okx_symbol,
                "tdMode": "cross",
                "side": okx_side,
                "ordType": "limit",
                "sz": str(qty),
                "px": str(price),  # Price
            }
            
            data = self._make_request("POST", endpoint, params=params, signed=True)
            
            if not data or len(data) == 0:
                raise ValueError("OKX không trả về order response")
            
            order = data[0]
            logger.info(f"✅ Đã gửi lệnh LIMIT {side} {qty} {symbol} @ {price}")
            
            return {
                "orderId": order.get("ordId"),
                "symbol": symbol,
                "side": side,
                "executedQty": qty,
                "price": price,
                "status": "PENDING",
            }
        except Exception as e:
            logger.error(f"❌ Lỗi gửi lệnh LIMIT: {e}")
            raise
