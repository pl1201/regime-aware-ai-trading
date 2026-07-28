"""
Backward compatibility layer for common_strategies

This module re-exports all strategies from the new modular structure.
Old code importing from algo_trading.strategies.common_strategies will continue to work.
"""
# Re-export all strategies for backward compatibility
from .trend import (
    SMAEMACrossStrategy,
    RenkoTrendStrategy,
    KalmanFilterForecastStrategy,
    ARIMAStrategy,
)
from .momentum import (
    RSIDivergenceStrategy,
    MACDMomentumStrategy,
    BollingerBreakoutStrategy,
    VolumeProfileImbalanceStrategy,
)
from .mean_reversion import (
    VWAPMeanReversionStrategy,
    OUProcessMeanReversionStrategy,
    StatArbCointegrationStrategy,
)
from .ml import (
    LSTMTransformerStrategy,
)
from .volatility import (
    GARCHVolatilityStrategy,
)

__all__ = [
    'SMAEMACrossStrategy', 'RSIDivergenceStrategy', 'MACDMomentumStrategy', 'BollingerBreakoutStrategy',
    'VWAPMeanReversionStrategy', 'RenkoTrendStrategy', 'VolumeProfileImbalanceStrategy', 'OUProcessMeanReversionStrategy',
    'KalmanFilterForecastStrategy', 'ARIMAStrategy', 'LSTMTransformerStrategy', 'StatArbCointegrationStrategy',
    'GARCHVolatilityStrategy',
]


class SMAEMACrossStrategy(BaseStrategy):
    """
    SMA/EMA Crossover
    Nguyên lý: mua khi đường trung bình ngắn hạn cắt lên dài hạn, bán khi cắt xuống.
    Tham số: fast=20, slow=50, ma_type in {sma, ema}
    """
    name = "SMA/EMA Crossover"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        fast = int(self.params.get('fast', 20))
        slow = int(self.params.get('slow', 50))
        ma_type = self.params.get('ma_type', 'ema')
        ma_fast = ema(close, fast) if ma_type == 'ema' else sma(close, fast)
        ma_slow = ema(close, slow) if ma_type == 'ema' else sma(close, slow)
        sig = cross_over(ma_fast, ma_slow)
        # giữ vị thế theo tín hiệu cắt
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'fast': fast, 'slow': slow, 'ma_type': ma_type})


class RSIDivergenceStrategy(BaseStrategy):
    """
    RSI + Divergence (đơn giản)
    Nguyên lý: quá mua/quá bán theo RSI, cộng thêm kiểm tra phân kỳ đơn giản: giá tạo đỉnh cao hơn nhưng RSI tạo đỉnh thấp hơn (bearish), ngược lại là bullish.
    Tham số: period=14, ob=70, os=30, lookback=5
    """
    name = "RSI Divergence"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        period = int(self.params.get('period', 14))
        ob = float(self.params.get('overbought', 70))
        os_ = float(self.params.get('oversold', 30))
        look = int(self.params.get('lookback', 5))
        r = rsi(close, period)
        # tín hiệu nền tảng
        base = pd.Series(0, index=df.index)
        base[r < os_] = 1
        base[r > ob] = -1
        # phân kỳ đơn giản: so sánh hai pivot gần nhất trong lookback
        price_high = close.rolling(look).max()
        price_low = close.rolling(look).min()
        rsi_high = r.rolling(look).max()
        rsi_low = r.rolling(look).min()
        bearish = ((price_high > price_high.shift(look)) & (rsi_high < rsi_high.shift(look))).astype(int) * -1
        bullish = ((price_low < price_low.shift(look)) & (rsi_low > rsi_low.shift(look))).astype(int) * 1
        div = bullish + bearish
        sig = base.where(div == 0, div)
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'period': period, 'overbought': ob, 'oversold': os_, 'lookback': look})


class MACDMomentumStrategy(BaseStrategy):
    """
    MACD Momentum
    Nguyên lý: giao dịch theo hướng MACD trên signal, và histogram mở rộng thu hẹp.
    Tham số: fast=12, slow=26, signal=9
    """
    name = "MACD Momentum"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        fast = int(self.params.get('fast', 12))
        slow = int(self.params.get('slow', 26))
        sigp = int(self.params.get('signal', 9))
        macd_line, signal_line, hist = macd(close, fast, slow, sigp)
        sig = pd.Series(0, index=df.index)
        sig[macd_line > signal_line] = 1
        sig[macd_line < signal_line] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'fast': fast, 'slow': slow, 'signal': sigp})


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout
    Nguyên lý: breakout trên dải trên mua, dưới dải dưới bán; thoát khi trở lại dải giữa.
    Tham số: window=20, k=2
    """
    name = "Bollinger Breakout"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        window = int(self.params.get('window', 20))
        k = float(self.params.get('k', 2.0))
        m, u, l = bollinger_bands(close, window, k)
        sig = pd.Series(0, index=df.index)
        sig[close > u] = 1
        sig[close < l] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        # exit rule: when price crosses middle back
        exit_long = (pos.shift(1) > 0) & (close < m)
        exit_short = (pos.shift(1) < 0) & (close > m)
        pos[exit_long | exit_short] = 0
        pos = pos.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'window': window, 'k': k})


class VWAPMeanReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion
    Nguyên lý: giá lệch xa VWAP -> kỳ vọng hồi về VWAP.
    Tham số: thr=1.5 (đơn vị ATR), atr_window=14
    """
    name = "VWAP Mean Reversion"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        v = vwap(df)
        a = atr(df, 14)
        thr = float(self.params.get('thr', 1.5))
        close = df['close']
        dist = close - v
        sig = pd.Series(0, index=df.index)
        sig[dist < -thr * a] = 1  # dưới xa -> mua
        sig[dist > thr * a] = -1  # trên xa -> bán
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'thr_atr': thr})


class RenkoTrendStrategy(BaseStrategy):
    """
    Renko Trend Following (xấp xỉ)
    Nguyên lý: chuyển chuỗi giá sang brick kích thước theo ATR; theo dõi hướng bricks.
    Tham số: brick_atr=14, brick_k=1.0
    """
    name = "Renko Trend"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        a = atr(df, int(self.params.get('brick_atr', 14)))
        k = float(self.params.get('brick_k', 1.0))
        brick = k * a
        direction = pd.Series(0, index=df.index)
        ref = close.iloc[0]
        dir_ = 0
        for i in range(1, len(close)):
            c = close.iloc[i]
            b = brick.iloc[i] if not np.isnan(brick.iloc[i]) else brick.iloc[:i].dropna().iloc[-1] if i>1 else np.nan
            if np.isnan(b):
                direction.iloc[i] = 0
                continue
            if c - ref >= b:
                dir_ = 1
                ref = c
            elif ref - c >= b:
                dir_ = -1
                ref = c
            direction.iloc[i] = dir_
        pos = direction.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'brick_k': k})


class VolumeProfileImbalanceStrategy(BaseStrategy):
    """
    Volume Profile Imbalance (xấp xỉ)
    Nguyên lý: trong cửa sổ N, tạo histogram giá theo bins và tổng hợp volume; khi giá vượt ra khỏi vùng giá trị cao (HVN) -> breakout theo hướng đó.
    Tham số: window=200, bins=20
    """
    name = "Volume Profile Imbalance"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        vol = df.get('volume', pd.Series(0, index=df.index))
        window = int(self.params.get('window', 200))
        bins = int(self.params.get('bins', 20))
        hvn_low = pd.Series(np.nan, index=df.index)
        hvn_high = pd.Series(np.nan, index=df.index)
        for i in range(window, len(df)):
            c = close.iloc[i-window:i]
            v = vol.iloc[i-window:i]
            counts, edges = np.histogram(c, bins=bins, weights=v)
            idx = counts.argmax()
            hvn_low.iloc[i] = edges[max(0, idx-1)]
            hvn_high.iloc[i] = edges[min(len(edges)-1, idx+1)]
        sig = pd.Series(0, index=df.index)
        sig[close > hvn_high] = 1
        sig[close < hvn_low] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'window': window, 'bins': bins})


class OUProcessMeanReversionStrategy(BaseStrategy):
    """
    Ornstein–Uhlenbeck Mean Reversion
    Nguyên lý: coi giá/chuỗi spread là quy trình OU; giao dịch khi z-score lệch xa mức cân bằng.
    Tham số: lookback=100, z=1.5
    """
    name = "OU Mean Reversion"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        x = df['close']
        look = int(self.params.get('lookback', 100))
        zthr = float(self.params.get('z', 1.5))
        # ước lượng OU qua AR(1): x_t = a + b x_{t-1} + e
        x1 = x.shift(1)
        # rolling regression
        zscores = pd.Series(np.nan, index=x.index)
        for i in range(look, len(x)):
            y = x.iloc[i-look+1:i+1]
            X = np.vstack([np.ones(len(y)-1), y.shift(1).iloc[1:].values]).T
            yy = y.iloc[1:].values
            try:
                beta = np.linalg.lstsq(X, yy, rcond=None)[0]
            except Exception:
                continue
            a, b = beta
            mu = a / (1 - b) if (1-b)!=0 else y.mean()
            resid = y - mu
            zscores.iloc[i] = (y.iloc[-1] - mu) / (resid.std(ddof=1) + 1e-12)
        sig = pd.Series(0, index=df.index)
        sig[zscores < -zthr] = 1
        sig[zscores > zthr] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'lookback': look, 'z': zthr})


class KalmanFilterForecastStrategy(BaseStrategy):
    """
    Kalman Filter Price Forecasting (mô hình 2 trạng thái: level + trend)
    Nguyên lý: bộ lọc Kalman ẩn 2D (local linear trend). Mua khi xu hướng (trend) > 0, bán khi < 0.
    Tham số: q=0.0001, r=0.001
    """
    name = "Kalman Forecast"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        y = df['close'].values.astype(float)
        n = len(y)
        q = float(self.params.get('q', 1e-4))  # process noise
        r = float(self.params.get('r', 1e-3))  # observation noise
        # state: [level, trend]
        F = np.array([[1, 1], [0, 1]], dtype=float)
        H = np.array([[1, 0]], dtype=float)
        Q = q * np.eye(2)
        R = np.array([[r]], dtype=float)
        x = np.array([[y[0]], [0.0]])
        P = np.eye(2)
        level = np.zeros(n)
        trend = np.zeros(n)
        for i in range(n):
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            # update
            z = np.array([[y[i]]])
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(2) - K @ H) @ P
            level[i], trend[i] = x.flatten()
        sig = pd.Series(0, index=df.index)
        sig[trend > 0] = 1
        sig[trend < 0] = -1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'q': q, 'r': r})


class ARIMAStrategy(BaseStrategy):
    """
    ARIMA/SARIMA Forecast
    Nguyên lý: dự báo bước kế tiếp bằng ARIMA. Nếu dự báo tăng -> mua, giảm -> bán.
    Tham số: order=(1,1,1) (SARIMA có seasonal_order)
    """
    name = "ARIMA/SARIMA"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        order = self.params.get('order', (1,1,1))
        seasonal_order = self.params.get('seasonal_order', None)
        try:
            import statsmodels.api as sm
        except Exception:
            # fallback: dùng EMA crossover nhẹ
            ema_fast = ema(close, 10)
            ema_slow = ema(close, 30)
            sig = cross_over(ema_fast, ema_slow)
            pos = sig.replace(0, np.nan).ffill().fillna(0)
            return StrategyResult(signals=pos, meta={'fallback':'ema'})
        # rolling one-step forecast
        sig = pd.Series(0, index=df.index)
        window = int(self.params.get('window', 200))
        for i in range(window, len(close)):
            y = close.iloc[i-window:i]
            try:
                if seasonal_order is None:
                    model = sm.tsa.ARIMA(y, order=order)
                else:
                    model = sm.tsa.SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                f = res.forecast(1)
                sig.iloc[i] = 1 if f.iloc[-1] > y.iloc[-1] else -1
            except Exception:
                continue
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'order': order, 'seasonal_order': seasonal_order})


class LSTMTransformerStrategy(BaseStrategy):
    """
    LSTM/Transformer Prediction (đơn giản)
    Ghi chú: yêu cầu torch. Nếu không có, fallback sang EMA.
    Tham số: lookback=50
    """
    name = "LSTM/Transformer"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close'].astype(float)
        look = int(self.params.get('lookback', 50))
        try:
            import torch
            import torch.nn as nn
        except Exception:
            # fallback ema
            ema_fast = ema(close, 10)
            ema_slow = ema(close, 30)
            sig = cross_over(ema_fast, ema_slow)
            pos = sig.replace(0, np.nan).ffill().fillna(0)
            return StrategyResult(signals=pos, meta={'fallback':'ema'})

        class TinyLSTM(nn.Module):
            def __init__(self, input_size=1, hidden=16):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                o,_ = self.lstm(x)
                return self.fc(o[:,-1,:])
        # prepare data (quick train)
        x = close.values
        X, Y = [], []
        for i in range(look, len(x)):
            X.append(x[i-look:i])
            Y.append(x[i])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
        model = TinyLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(10):  # few epochs for speed
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(look, len(x)):
                seq = torch.tensor(x[i-look:i], dtype=torch.float32).view(1, look, 1)
                p = model(seq).item()
                preds.append(p)
        sig = pd.Series(0, index=df.index)
        sig.iloc[look:] = np.where(np.array(preds) > x[look-1:-1], 1, -1)
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'lookback': look})


class StatArbCointegrationStrategy(BaseStrategy):
    """
    Statistical Arbitrage (Cointegration) – Pairs Trading
    Cần 2 chuỗi giá: close_X và close_Y trong df (cột 'close_Y' tồn tại) hoặc truyền vào params 'other'.
    Nguyên lý: kiểm định đồng liên kết; tạo spread = X - beta*Y; trade mean-reversion theo z-score.
    Tham số: lookback=250, z=2
    """
    name = "StatArb Cointegration"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        x = df['close']
        if 'close_Y' in df.columns:
            y = df['close_Y']
        elif 'other' in self.params:
            y = self.params['other'].reindex(df.index)['close']
        else:
            # không đủ dữ liệu -> flat
            return StrategyResult(signals=pd.Series(0, index=df.index), meta={'error':'need second series'})
        look = int(self.params.get('lookback', 250))
        zthr = float(self.params.get('z', 2.0))
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.stattools import coint
        except Exception:
            # fallback: OLS beta rolling
            pass
        spread = pd.Series(np.nan, index=df.index)
        zscores = pd.Series(np.nan, index=df.index)
        for i in range(look, len(df)):
            X = y.iloc[i-look:i]
            Y = x.iloc[i-look:i]
            X1 = np.vstack([X.values, np.ones(len(X))]).T
            beta, alpha = np.linalg.lstsq(X1, Y.values, rcond=None)[0]
            sp = Y - (alpha + beta*X)
            spread.iloc[i] = sp.iloc[-1]
            zscores.iloc[i] = (sp.iloc[-1] - sp.mean())/(sp.std(ddof=1)+1e-12)
        sig = pd.Series(0, index=df.index)
        sig[zscores > zthr] = -1
        sig[zscores < -zthr] = 1
        pos = sig.replace(0, np.nan).ffill().fillna(0)
        return StrategyResult(signals=pos, meta={'lookback': look, 'z': zthr})


class GARCHVolatilityStrategy(BaseStrategy):
    """
    GARCH Volatility (quy mô vị thế theo vol)
    Nguyên lý: ước lượng volatility bằng GARCH(1,1); khi vol thấp và động lượng dương -> tăng vị thế; vol cao -> giảm.
    Tham số: window=250
    """
    name = "GARCH Volatility"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        close = df['close']
        ret = np.log(close).diff().fillna(0)
        vol = pd.Series(np.nan, index=df.index)
        try:
            from arch import arch_model
            window = int(self.params.get('window', 500))
            for i in range(window, len(ret)):
                r = ret.iloc[i-window:i]
                try:
                    am = arch_model(r*100, vol='Garch', p=1, o=0, q=1, dist='normal')
                    res = am.fit(disp='off')
                    f = res.forecast(horizon=1).variance.iloc[-1,0]
                    vol.iloc[i] = np.sqrt(f)/100
                except Exception:
                    continue
        except Exception:
            # fallback: rolling std
            window = int(self.params.get('window', 250))
            vol = ret.rolling(window).std().reindex(df.index)
        mom = close.diff(5)
        raw = np.sign(mom)
        # scale by inverse vol
        v = (vol - vol.min())/(vol.max()-vol.min()+1e-12)
        scale = (1 - v).fillna(0.5)
        pos = (raw * scale).fillna(0).clip(-1,1)
        return StrategyResult(signals=pos, meta={'window': window})

