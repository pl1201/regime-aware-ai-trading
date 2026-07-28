from __future__ import annotations
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import Dict, Any, Optional
import pandas as pd

from algo_trading.data_loader.loader import load_data
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
from algo_trading.backtest.vectorized import run_backtest, BacktestConfig, RiskConfig
from algo_trading.backtest.event_driven import run_event_backtest, EventConfig
from algo_trading.utils.session_analysis import (
    session_return_stats,
    hour_of_day_return_stats,
)
from algo_trading.utils.trade_formatter import format_trades_csv
from algo_trading.viz.plots import (
    plot_candlestick,
    plot_equity_curve,
    plot_drawdown,
    plot_volatility,
    plot_correlation_heatmap,
    alpha_beta_scatter,
    quick_dashboard,
)


STRATEGY_MAP = {
    'sma_ema': SMAEMACrossStrategy,
    'rsi_div': RSIDivergenceStrategy,
    'macd': MACDMomentumStrategy,
    'bb_breakout': BollingerBreakoutStrategy,
    'vwap_mr': VWAPMeanReversionStrategy,
    'renko_trend': RenkoTrendStrategy,
    'vol_profile': VolumeProfileImbalanceStrategy,
    'ou_mr': OUProcessMeanReversionStrategy,
    'kalman': KalmanFilterForecastStrategy,
    'arima': ARIMAStrategy,
    'lstm': LSTMTransformerStrategy,
    'stat_arb': StatArbCointegrationStrategy,
    'garch_vol': GARCHVolatilityStrategy,
}


def parse_params(params_json: Optional[str]) -> Dict[str, Any]:
    if not params_json:
        return {}
    try:
        return json.loads(params_json)
    except Exception:
        print("[WARN] Không parse được --params, dùng rỗng.")
        return {}


def detect_freq_from_interval(interval: Optional[str]) -> Optional[str]:
    if not interval:
        return None
    s = interval.lower()
    if 'min' in s or 'm' in s and 'mo' not in s:
        return '1min'
    if 'h' in s:
        return '1H'
    if 'd' in s:
        return '1D'
    if 'w' in s:
        return '1W'
    return None


def build_overlays(df: pd.DataFrame) -> Dict[str, pd.Series]:
    overlays = {}
    for col in ['SMA20','EMA20','BB_MID','BB_UPPER','BB_LOWER','VWAP']:
        if col in df.columns:
            overlays[col] = df[col]
    return overlays


def main():
    parser = argparse.ArgumentParser(description='Algo Trading Demo End-to-End')
    # Data source
    parser.add_argument('--source', type=str, default='yfinance', choices=['csv','parquet','yfinance','binance'], help='Nguồn dữ liệu')
    parser.add_argument('--path', type=str, default=None, help='Đường dẫn CSV/Parquet')
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Ticker cho yfinance (vd: BTC-USD)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol cho Binance (vd: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='Khung thời gian yfinance/binance (vd: 1h, 1d, 1m)')
    parser.add_argument('--start', type=str, default=None, help='Ngày bắt đầu (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='Ngày kết thúc (YYYY-MM-DD)')

    # Analysis-only mode (không backtest, chỉ phân tích phiên/giờ)
    parser.add_argument(
        '--analysis',
        type=str,
        default=None,
        choices=['session', 'hour', 'both'],
        help='Chế độ phân tích thay vì backtest: session/hour/both (bỏ qua chiến lược & backtest)',
    )

    # Strategy
    parser.add_argument('--strategy', type=str, default='sma_ema', choices=list(STRATEGY_MAP.keys()))
    parser.add_argument('--params', type=str, default=None, help='JSON tham số chiến lược, vd: {"fast":20,"slow":50,"ma_type":"ema"}')

    # Backtest mode
    parser.add_argument('--mode', type=str, default='vectorized', choices=['vectorized','event'], help='Chế độ backtest')
    parser.add_argument('--allow_short', action='store_true', help='Cho phép short')
    parser.add_argument('--leverage', type=float, default=1.0)
    parser.add_argument('--commission', type=float, default=0.0005)
    parser.add_argument('--slippage_bps', type=float, default=1.0)
    parser.add_argument('--use_next_open', action='store_true', help='Vào/ra tại open kế tiếp')

    # Risk (SL/TP/Trailing)
    parser.add_argument('--sl_pct', type=float, default=None)
    parser.add_argument('--tp_pct', type=float, default=None)
    parser.add_argument('--trailing_pct', type=float, default=None)
    parser.add_argument('--sl_atr_k', type=float, default=None)
    parser.add_argument('--tp_atr_k', type=float, default=None)
    parser.add_argument('--trailing_atr_k', type=float, default=None)
    parser.add_argument('--atr_col', type=str, default='ATR14')

    # Plot
    parser.add_argument('--plot', action='store_true', help='Hiển thị biểu đồ')
    parser.add_argument('--plotly', action='store_true', help='Dùng plotly cho candlestick nếu có')
    parser.add_argument('--save_dir', type=str, default=None, help='Lưu hình vào thư mục')
    parser.add_argument('--export_csv', type=str, default=None, help='Thư mục để lưu các file CSV backtest')

    args = parser.parse_args()

    # Load data
    load_kwargs: Dict[str, Any] = {}
    if args.source == 'csv':
        if not args.path:
            raise SystemExit('--path là bắt buộc cho source=csv')
        load_kwargs = {'path': args.path, 'timeframe': None}
    elif args.source == 'parquet':
        if not args.path:
            raise SystemExit('--path là bắt buộc cho source=parquet')
        load_kwargs = {'path': args.path, 'timeframe': None}
    elif args.source == 'yfinance':
        load_kwargs = {'ticker': args.ticker, 'interval': args.interval, 'start': args.start, 'end': args.end}
    elif args.source == 'binance':
        load_kwargs = {'symbol': args.symbol, 'interval': args.interval, 'start': args.start, 'end': args.end, 'market': 'spot'}

    try:
        df = load_data(args.source, **load_kwargs)
    except Exception as e:
        raise SystemExit(f'Lỗi load dữ liệu: {e}')

    if df.empty:
        raise SystemExit('DataFrame rỗng sau khi load dữ liệu')

    # Nếu chọn chế độ phân tích phiên/giờ -> bỏ qua chiến lược & backtest
    if args.analysis:
        print("===== Session / Hour Analysis =====")
        tz = 'UTC'  
        if args.analysis in ('session', 'both'):
            sess_stats = session_return_stats(df, price_col='close', tz=tz)
            print("\n--- Session stats (Asia/Europe/US) ---")
            print(sess_stats.to_string(float_format=lambda x: f"{x:.6f}"))
        if args.analysis in ('hour', 'both'):
            hour_stats = hour_of_day_return_stats(df, price_col='close', tz=tz)
            print("\n--- Hour-of-day stats (0–23) ---")
            print(hour_stats.to_string(float_format=lambda x: f"{x:.6f}"))
        return

    # Strategy
    StrategyCls = STRATEGY_MAP[args.strategy]
    params = parse_params(args.params)
    strat = StrategyCls(**params)
    sig_res = strat.generate_signals(df)
    signals = sig_res.signals

    # Backtest
    freq = detect_freq_from_interval(args.interval) or '1D'
    if args.mode == 'vectorized':
        cfg = BacktestConfig(
            initial_capital=1.0,
            leverage=args.leverage,
            allow_short=args.allow_short,
            commission=args.commission,
            slippage_bps=args.slippage_bps,
            use_next_open=args.use_next_open,
            freq=freq,
        )
        risk = RiskConfig(
            sl_pct=args.sl_pct, tp_pct=args.tp_pct, trailing_pct=args.trailing_pct,
            sl_atr_k=args.sl_atr_k, tp_atr_k=args.tp_atr_k, trailing_atr_k=args.trailing_atr_k,
            atr_col=args.atr_col,
        ) if any(v is not None for v in [args.sl_pct, args.tp_pct, args.trailing_pct, args.sl_atr_k, args.tp_atr_k, args.trailing_atr_k]) else None
        res = run_backtest(df, signals, cfg=cfg, risk=risk, max_trades=100)
    else:
        cfg = EventConfig(
            initial_cash=10000.0,
            leverage=args.leverage,
            allow_short=args.allow_short,
            commission=args.commission,
            slippage_bps=args.slippage_bps,
            use_next_open=args.use_next_open,
            price_col='close', open_col='open', high_col='high', low_col='low',
            freq=freq,
        )
        risk = RiskConfig(
            sl_pct=args.sl_pct, tp_pct=args.tp_pct, trailing_pct=args.trailing_pct,
            sl_atr_k=args.sl_atr_k, tp_atr_k=args.tp_atr_k, trailing_atr_k=args.trailing_atr_k,
            atr_col=args.atr_col,
        ) if any(v is not None for v in [args.sl_pct, args.tp_pct, args.trailing_pct, args.sl_atr_k, args.tp_atr_k, args.trailing_atr_k]) else None
        res = run_event_backtest(df, signals, cfg=cfg, risk=risk, max_trades=100)

    print("===== Metrics =====")
    for k, v in res['summary'].items():
        try:
            if isinstance(v, float):
                print(f"{k:>12}: {v:.4f}")
            else:
                print(f"{k:>12}: {v}")
        except Exception:
            print(f"{k:>12}: {v}")

    # Export CSV
    if args.export_csv:
        os.makedirs(args.export_csv, exist_ok=True)
        
        # Export CSV tổng hợp (OHLCV + Signals + Equity + Returns)
        export_data = {
            'timestamp': df.index,
            'open': df['open'] if 'open' in df.columns else df['close'],
            'high': df['high'] if 'high' in df.columns else df['close'],
            'low': df['low'] if 'low' in df.columns else df['close'],
            'close': df['close'],
            'volume': df['volume'] if 'volume' in df.columns else 0,
            'signal': signals.reindex(df.index, fill_value=0),
            'equity': res['equity'].reindex(df.index, method='ffill'),
            'returns': res['returns'].reindex(df.index, fill_value=0),
        }
        export_df = pd.DataFrame(export_data)
        export_df = export_df.fillna(0)
        csv_path = os.path.join(args.export_csv, 'backtest_full.csv')
        export_df.to_csv(csv_path, index=True)
        print(f"[INFO] Đã lưu CSV tổng hợp: {csv_path}")
        
        # Export CSV Equity Curve
        equity_df = pd.DataFrame({
            'timestamp': res['equity'].index,
            'equity': res['equity'].values,
            'returns': res['returns'].values,
        })
        equity_path = os.path.join(args.export_csv, 'equity_curve.csv')
        equity_df.to_csv(equity_path, index=False)
        print(f"[INFO] Đã lưu CSV Equity Curve: {equity_path}")
        
        # Export CSV Summary Metrics
        summary_df = pd.DataFrame([res['summary']])
        summary_path = os.path.join(args.export_csv, 'summary_metrics.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Đã lưu CSV Summary Metrics: {summary_path}")
        
        if 'trades' in res and isinstance(res['trades'], pd.DataFrame) and not res['trades'].empty:
            symbol_name = args.symbol if args.symbol else (args.ticker if args.ticker else "UNKNOWN")
            formatted_trades = format_trades_csv(
                res['trades'],
                df,
                symbol=symbol_name,
                timeframe=args.interval
            )
            trades_path = os.path.join(args.export_csv, 'trades.csv')
            formatted_trades.to_csv(trades_path, index=False)
            print(f"[INFO] Đã lưu CSV Trades (định dạng chuẩn): {trades_path}")
            
            # Also save raw trades
            trades_raw_path = os.path.join(args.export_csv, 'trades_raw.csv')
            res['trades'].to_csv(trades_raw_path, index=False)
            print(f"[INFO] Đã lưu CSV Trades (raw): {trades_raw_path}")

    # Plot
    if args.plot or args.save_dir:
        overlays = build_overlays(df)
        figs = quick_dashboard(df, overlays, res['equity'], signals=signals, use_plotly=args.plotly)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            # Lưu các figure matplotlib
            if isinstance(figs, tuple):
                for i, fg in enumerate(figs, start=1):
                    try:
                        # plotly figure
                        if hasattr(fg, 'to_image'):
                            p = os.path.join(args.save_dir, f'fig_{i}.png')
                            fg.write_image(p)
                        else:
                            p = os.path.join(args.save_dir, f'fig_{i}.png')
                            fg.savefig(p, dpi=150)
                    except Exception:
                        pass
        if args.plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()


