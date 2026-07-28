
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging

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
from algo_trading.core.backtest_vectorized import (
    BacktestConfig,
    RiskConfig,
    vectorized_pnl,
    barwise_with_stops,
)
from algo_trading.core.metrics import (
    performance_summary,
    to_returns,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    compound_annual_growth_rate,
    calmar_ratio,
    infer_freq_label_from_index,
    has_min_bars_for_freq,
)

logger = logging.getLogger(__name__)


# Định nghĩa các strategy và tham số mặc định
STRATEGY_CONFIGS = {
    'SMA/EMA Crossover': {
        'class': SMAEMACrossStrategy,
        'params': [
            {'fast': 20, 'slow': 50, 'ma_type': 'ema'},
            {'fast': 10, 'slow': 30, 'ma_type': 'sma'},
            {'fast': 12, 'slow': 26, 'ma_type': 'ema'},
        ],
        'category': 'Trend Following',
    },
    'RSI Divergence': {
        'class': RSIDivergenceStrategy,
        'params': [
            {'period': 14, 'overbought': 70, 'oversold': 30, 'lookback': 5},
            {'period': 21, 'overbought': 75, 'oversold': 25, 'lookback': 7},
        ],
        'category': 'Momentum',
    },
    'MACD Momentum': {
        'class': MACDMomentumStrategy,
        'params': [
            {'fast': 12, 'slow': 26, 'signal': 9},
            {'fast': 8, 'slow': 21, 'signal': 5},
        ],
        'category': 'Momentum',
    },
    'Bollinger Breakout': {
        'class': BollingerBreakoutStrategy,
        'params': [
            {'window': 20, 'k': 2.0},
            {'window': 15, 'k': 2.5},
        ],
        'category': 'Momentum',
    },
    'VWAP Mean Reversion': {
        'class': VWAPMeanReversionStrategy,
        'params': [
            {'thr': 1.5},
            {'thr': 2.0},
        ],
        'category': 'Mean Reversion',
    },
    'Renko Trend': {
        'class': RenkoTrendStrategy,
        'params': [
            {'brick_atr': 14, 'brick_k': 1.0},
            {'brick_atr': 20, 'brick_k': 1.5},
        ],
        'category': 'Trend Following',
    },
    'Volume Profile': {
        'class': VolumeProfileImbalanceStrategy,
        'params': [
            {'window': 200, 'bins': 20},
            {'window': 100, 'bins': 15},
        ],
        'category': 'Momentum',
    },
    'OU Mean Reversion': {
        'class': OUProcessMeanReversionStrategy,
        'params': [
            {'lookback': 100, 'z': 1.5},
            {'lookback': 150, 'z': 2.0},
        ],
        'category': 'Mean Reversion',
    },
    'Kalman Forecast': {
        'class': KalmanFilterForecastStrategy,
        'params': [
            {'q': 0.0001, 'r': 0.001},
            {'q': 0.0005, 'r': 0.002},
        ],
        'category': 'Trend Following',
    },
    'ARIMA': {
        'class': ARIMAStrategy,
        'params': [
            {'order': (1, 1, 1), 'window': 200},
        ],
        'category': 'Trend Following',
    },
    'LSTM/Transformer': {
        'class': LSTMTransformerStrategy,
        'params': [
            {'lookback': 50},
        ],
        'category': 'ML',
    },
    'GARCH Volatility': {
        'class': GARCHVolatilityStrategy,
        'params': [
            {'window': 250},
        ],
        'category': 'Volatility',
    },
}


class StrategyEvaluator:
    """Đánh giá và so sánh các strategy."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        use_stops: bool = True,
        sl_pct: Optional[float] = 0.02,
        tp_pct: Optional[float] = 0.04,
    ):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.use_stops = use_stops
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Kiểm tra số lượng dữ liệu tối thiểu (một số strategy cần nhiều dữ liệu)
        min_data_points = 50  
        if len(self.df) < min_data_points:
            logger.warning(f"DataFrame chỉ có {len(self.df)} điểm dữ liệu, một số strategy có thể không hoạt động tốt")
        
        # Tính ATR nếu cần
        if use_stops:
            from algo_trading.indicators import atr
            self.df['ATR14'] = atr(self.df, 14)
    
    @staticmethod
    def _default_error_result(
        strategy_name: str,
        params: Dict[str, Any],
        error: str,
        extended: bool = False,
    ) -> Dict[str, Any]:
        """Trả về kết quả lỗi mặc định cho strategy evaluation."""
        result = {
            'strategy_name': strategy_name,
            'params': params,
            'error': error,
            'total_return': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_drawdown': 0.0,
            'cagr': 0.0,
            'calmar': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
        }
        if extended:
            result.update({
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'rr_ratio': 0.0,
                'profit_factor': 0.0,
                'long_trades': 0,
                'short_trades': 0,
                'avg_holding_time': 0.0,
            })
        return result
    
    def evaluate_strategy(
        self,
        strategy_class: type,
        strategy_params: Dict[str, Any],
        strategy_name: str = "Strategy",
    ) -> Dict[str, Any]:
        try:
            # Validate DataFrame before strategy execution
            if self.df.empty:
                logger.warning(f"DataFrame is empty for {strategy_name}")
                return self._default_error_result(strategy_name, strategy_params, 'DataFrame is empty')
            
            # Tạo strategy instance
            try:
                strategy = strategy_class(**strategy_params)
            except Exception as e:
                logger.error(f"Lỗi khởi tạo {strategy_name} với params {strategy_params}: {e}")
                return self._default_error_result(strategy_name, strategy_params, f'Strategy initialization failed: {str(e)}')
            
            # Generate signals
            try:
                result = strategy.generate_signals(self.df)
            except Exception as e:
                logger.error(f"Lỗi generate_signals cho {strategy_name}: {e}")
                return self._default_error_result(strategy_name, strategy_params, f'Signal generation failed: {str(e)}')
            
            # Kiểm tra result
            if result is None:
                return self._default_error_result(strategy_name, strategy_params, 'Strategy returned None', extended=True)
            
            # Kiểm tra signals
            signals = result.signals if hasattr(result, 'signals') else None
            if signals is None:
                return self._default_error_result(strategy_name, strategy_params, 'Strategy result has no signals attribute', extended=True)
            
            if not isinstance(signals, pd.Series):
                return self._default_error_result(strategy_name, strategy_params, f'Signals is not a Series (got {type(signals)})', extended=True)
            
            if signals.empty or signals.isna().all():
                return self._default_error_result(strategy_name, strategy_params, 'No signals generated (empty or all NaN)', extended=True)
            
            freq_label = infer_freq_label_from_index(self.df.index)

            cfg = BacktestConfig(
                initial_capital=self.initial_capital,
                commission=self.commission,
                allow_short=True,
                freq=freq_label,
            )
            
            risk_cfg = None
            if self.use_stops:
                risk_cfg = RiskConfig(
                    sl_pct=self.sl_pct,
                    tp_pct=self.tp_pct,
                )
            
            if self.use_stops and risk_cfg:
                equity, returns, trades_df = barwise_with_stops(
                    self.df, signals, cfg, risk_cfg
                )
            else:
                equity, returns = vectorized_pnl(self.df, signals, cfg)
                trades_df = pd.DataFrame()
            
            # Tính metrics
            if equity.empty or len(equity) < 2:
                return self._default_error_result(strategy_name, strategy_params, 'Insufficient data', extended=True)
            
            # Tính các chỉ số
            perf = performance_summary(equity, returns, freq=freq_label)
            perf['HasSufficientData'] = has_min_bars_for_freq(len(self.df), freq_label)
            
            # Tính win rate và các metrics từ trades
            win_rate = 0.0
            total_trades = 0
            avg_win = 0.0
            avg_loss = 0.0
            rr_ratio = 0.0
            profit_factor = 0.0
            total_pnl = 0.0
            long_trades = 0
            short_trades = 0
            avg_holding_time = 0.0  # trong giờ
            
            if not trades_df.empty and 'pnl' in trades_df.columns:
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                total_trades = len(trades_df)
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
                
                # Tính PnL
                total_pnl = trades_df['pnl'].sum()
                
                # Tính average win/loss và RR ratio
                if len(winning_trades) > 0:
                    avg_win = winning_trades['pnl'].mean()
                if len(losing_trades) > 0:
                    avg_loss = abs(losing_trades['pnl'].mean())
                    if avg_loss > 0:
                        rr_ratio = avg_win / avg_loss if avg_win > 0 else 0.0
                
                # Profit factor = tổng lợi nhuận / tổng lỗ
                total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
                total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
                profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)
                
                # Phân tích lệnh vào (long/short)
                if 'position' in trades_df.columns:
                    long_trades = (trades_df['position'] > 0).sum()
                    short_trades = (trades_df['position'] < 0).sum()
                elif 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns:
                    # Ước tính từ giá: nếu exit > entry thì là long
                    long_trades = (trades_df['exit_price'] > trades_df['entry_price']).sum()
                    short_trades = total_trades - long_trades
                
                # Tính thời gian giữ lệnh trung bình
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    holding_times = []
                    for idx, row in trades_df.iterrows():
                        if pd.notna(row.get('entry_time')) and pd.notna(row.get('exit_time')):
                            try:
                                entry_time = pd.to_datetime(row['entry_time'])
                                exit_time = pd.to_datetime(row['exit_time'])
                                holding_time = (exit_time - entry_time).total_seconds() / 3600  # chuyển sang giờ
                                if holding_time > 0:
                                    holding_times.append(holding_time)
                            except:
                                pass
                    if holding_times:
                        avg_holding_time = np.mean(holding_times)
            
            # Tính số lần đổi signal (proxy cho số trades nếu không có trades_df)
            if total_trades == 0:
                signal_changes = (signals.diff() != 0).sum()
                total_trades = max(1, signal_changes)
            
            return {
                'strategy_name': strategy_name,
                'params': strategy_params,
                'total_return': perf.get('TotalReturn', 0.0),
                'sharpe': perf.get('Sharpe', 0.0),
                'sortino': perf.get('Sortino', 0.0),
                'max_drawdown': perf.get('MaxDrawdown', 0.0),
                'cagr': perf.get('CAGR', 0.0),
                'calmar': perf.get('Calmar', 0.0),
                'volatility': perf.get('Volatility', 0.0),
                'has_sufficient_data': perf.get('HasSufficientData', True),
                'win_rate': win_rate,
                'total_trades': total_trades,
                'final_equity': equity.iloc[-1] if len(equity) > 0 else self.initial_capital,
                # Metrics mới
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'rr_ratio': rr_ratio,
                'profit_factor': profit_factor,
                'long_trades': long_trades,
                'short_trades': short_trades,
                'avg_holding_time': avg_holding_time,
            }
        except Exception as e:
            logger.error(f"Lỗi đánh giá {strategy_name}: {e}")
            return self._default_error_result(strategy_name, strategy_params, str(e), extended=True)
    
    def evaluate_all(self) -> pd.DataFrame:
        results = []
        
        for strategy_name, config in STRATEGY_CONFIGS.items():
            strategy_class = config['class']
            params_list = config['params']
            category = config.get('category', 'Unknown')
            
            for params in params_list:
                result = self.evaluate_strategy(
                    strategy_class,
                    params,
                    strategy_name
                )
                result['category'] = category
                results.append(result)
        
        df_results = pd.DataFrame(results)
        
        # Tính composite_score nếu có dữ liệu hợp lệ
        if not df_results.empty:
            # Tạo mask cho các strategy không có lỗi
            has_error = df_results.get('error', pd.Series([False] * len(df_results)))
            if isinstance(has_error, pd.Series):
                valid_mask = ~has_error.astype(bool)
            else:
                valid_mask = pd.Series([True] * len(df_results), index=df_results.index)
            
            df_valid = df_results[valid_mask].copy()
            
            if not df_valid.empty:
                # Tính điểm tổng hợp (composite score)
                # Normalize các metrics về scale 0-100
                metrics_to_score = ['sharpe', 'sortino', 'cagr', 'calmar', 'win_rate']
                for metric in metrics_to_score:
                    if metric in df_valid.columns:
                        col = df_valid[metric].fillna(0)
                        max_val = col.abs().max()
                        if max_val > 0:
                            df_valid[f'{metric}_score'] = (col / max_val * 50).clip(-50, 50)
                        else:
                            df_valid[f'{metric}_score'] = 0
                    else:
                        df_valid[f'{metric}_score'] = 0
                
                # Tính điểm tổng hợp (trọng số)
                df_valid['composite_score'] = (
                    df_valid['sharpe_score'] * 0.25 +
                    df_valid['sortino_score'] * 0.25 +
                    df_valid['cagr_score'] * 0.20 +
                    df_valid['calmar_score'] * 0.15 +
                    df_valid['win_rate_score'] * 0.15
                )
                
                # Gán composite_score vào df_results
                df_results['composite_score'] = 0.0
                df_results.loc[df_valid.index, 'composite_score'] = df_valid['composite_score']
            else:
                # Nếu không có strategy hợp lệ nào, thêm composite_score = 0
                df_results['composite_score'] = 0.0
        
        return df_results
    
    def generate_comparison_report(
        self,
        output_file: str = "strategy_comparison_report.txt",
        top_n: int = 10,
    ) -> str:
        logger.info("Bắt đầu đánh giá tất cả strategies...")
        df_results = self.evaluate_all()
        
        if df_results.empty:
            report = "Không có kết quả để so sánh."
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            return report
        
        # Loại bỏ các strategy có lỗi
        df_results = df_results[~df_results.get('error', pd.Series([False] * len(df_results))).astype(bool)]
        
        # Kiểm tra lại sau khi lọc lỗi
        if df_results.empty:
            report = "Không có kết quả hợp lệ để so sánh (tất cả strategies đều có lỗi)."
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            return report
        
        # Tính điểm tổng hợp (composite score)
        # Normalize các metrics về scale 0-100
        metrics_to_score = ['sharpe', 'sortino', 'cagr', 'calmar', 'win_rate']
        for metric in metrics_to_score:
            if metric in df_results.columns:
                col = df_results[metric].fillna(0)
                max_val = col.abs().max()
                if max_val > 0:
                    df_results[f'{metric}_score'] = (col / max_val * 50).clip(-50, 50)
                else:
                    df_results[f'{metric}_score'] = 0
        
        # Tính điểm tổng hợp (trọng số)
        df_results['composite_score'] = (
            df_results.get('sharpe_score', 0) * 0.25 +
            df_results.get('sortino_score', 0) * 0.25 +
            df_results.get('cagr_score', 0) * 0.20 +
            df_results.get('calmar_score', 0) * 0.15 +
            df_results.get('win_rate_score', 0) * 0.15
        )
        
        # Sắp xếp theo composite score
        df_results = df_results.sort_values('composite_score', ascending=False)
        
        # Tạo báo cáo
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("BÁO CÁO SO SÁNH CÁC PHƯƠNG PHÁP INDICATOR")
        report_lines.append("=" * 100)
        report_lines.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Vốn ban đầu: ${self.initial_capital:,.2f}")
        report_lines.append(f"Phí giao dịch: {self.commission*100:.2f}%")
        report_lines.append(f"SL/TP: {'Có' if self.use_stops else 'Không'}")
        if self.use_stops:
            report_lines.append(f"  - Stop Loss: {self.sl_pct*100:.1f}%")
            report_lines.append(f"  - Take Profit: {self.tp_pct*100:.1f}%")
        report_lines.append("")
        report_lines.append(f"Tổng số strategy được đánh giá: {len(df_results)}")
        report_lines.append("")
        
        # Top strategies
        report_lines.append("-" * 100)
        report_lines.append(f"TOP {top_n} STRATEGY HIỆU QUẢ NHẤT")
        report_lines.append("-" * 100)
        report_lines.append("")
        
        for idx, row in df_results.head(top_n).iterrows():
            report_lines.append(f"#{df_results.index.get_loc(idx) + 1}. {row['strategy_name']}")
            report_lines.append(f"   Category: {row.get('category', 'Unknown')}")
            report_lines.append(f"   Parameters: {row['params']}")
            report_lines.append(f"   Composite Score: {row['composite_score']:.2f}")
            report_lines.append(f"   Total Return: {row['total_return']*100:.2f}%")
            report_lines.append(f"   Sharpe Ratio: {row['sharpe']:.3f}")
            report_lines.append(f"   Sortino Ratio: {row['sortino']:.3f}")
            report_lines.append(f"   CAGR: {row['cagr']*100:.2f}%")
            report_lines.append(f"   Calmar Ratio: {row['calmar']:.3f}")
            report_lines.append(f"   Max Drawdown: {row['max_drawdown']*100:.2f}%")
            report_lines.append(f"   Win Rate: {row['win_rate']*100:.1f}%")
            report_lines.append(f"   Total Trades: {int(row['total_trades'])}")
            report_lines.append(f"   Final Equity: ${row['final_equity']:,.2f}")
            report_lines.append("")
        
        # So sánh theo category
        report_lines.append("-" * 100)
        report_lines.append("SO SÁNH THEO CATEGORY")
        report_lines.append("-" * 100)
        report_lines.append("")
        
        categories = df_results['category'].unique()
        for category in categories:
            cat_df = df_results[df_results['category'] == category].sort_values('composite_score', ascending=False)
            if len(cat_df) > 0:
                report_lines.append(f"Category: {category}")
                best = cat_df.iloc[0]
                report_lines.append(f"  Best: {best['strategy_name']} (Score: {best['composite_score']:.2f}, Return: {best['total_return']*100:.2f}%)")
                report_lines.append(f"  Average Return: {cat_df['total_return'].mean()*100:.2f}%")
                report_lines.append(f"  Average Sharpe: {cat_df['sharpe'].mean():.3f}")
                report_lines.append("")
        
        # Bảng tổng hợp
        report_lines.append("-" * 100)
        report_lines.append("BẢNG TỔNG HỢP TẤT CẢ STRATEGIES")
        report_lines.append("-" * 100)
        report_lines.append("")
        
        # Tạo bảng dạng text
        cols = ['strategy_name', 'category', 'total_return', 'sharpe', 'sortino', 'cagr', 'max_drawdown', 'win_rate', 'composite_score']
        # Chỉ lấy các cột có sẵn
        available_cols = [col for col in cols if col in df_results.columns]
        display_df = df_results[available_cols].copy()
        
        # Format các cột số nếu tồn tại
        if 'total_return' in display_df.columns:
            display_df['total_return'] = (display_df['total_return'] * 100).round(2)
        if 'cagr' in display_df.columns:
            display_df['cagr'] = (display_df['cagr'] * 100).round(2)
        if 'max_drawdown' in display_df.columns:
            display_df['max_drawdown'] = (display_df['max_drawdown'] * 100).round(2)
        if 'win_rate' in display_df.columns:
            display_df['win_rate'] = (display_df['win_rate'] * 100).round(1)
        if 'sharpe' in display_df.columns:
            display_df['sharpe'] = display_df['sharpe'].round(3)
        if 'sortino' in display_df.columns:
            display_df['sortino'] = display_df['sortino'].round(3)
        if 'calmar' in display_df.columns:
            display_df['calmar'] = display_df['calmar'].round(3)
        if 'composite_score' in display_df.columns:
            display_df['composite_score'] = display_df['composite_score'].round(2)
        
        # Format bảng
        report_lines.append(display_df.to_string(index=False))
        report_lines.append("")
        
        # Kết luận
        report_lines.append("-" * 100)
        report_lines.append("KẾT LUẬN VÀ KHUYẾN NGHỊ")
        report_lines.append("-" * 100)
        report_lines.append("")
        
        # Kiểm tra lại trước khi truy cập best strategy
        if len(df_results) > 0:
            best_strategy = df_results.iloc[0]
            report_lines.append(f"1. Strategy hiệu quả nhất: {best_strategy['strategy_name']}")
            report_lines.append(f"   - Với tham số: {best_strategy['params']}")
            report_lines.append(f"   - Đạt được: {best_strategy['total_return']*100:.2f}% return, Sharpe {best_strategy['sharpe']:.3f}")
            report_lines.append("")
        else:
            report_lines.append("1. Không có strategy hợp lệ để đánh giá.")
            report_lines.append("")
        
        # Tìm strategy ổn định nhất (thấp drawdown, cao Sharpe)
        stable_df = df_results[
            (df_results['max_drawdown'] > -0.2) &  # Drawdown < 20%
            (df_results['sharpe'] > 0.5)  # Sharpe > 0.5
        ]
        if len(stable_df) > 0:
            stable_best = stable_df.iloc[0]
            report_lines.append(f"2. Strategy ổn định nhất (Drawdown < 20%, Sharpe > 0.5):")
            report_lines.append(f"   - {stable_best['strategy_name']} với {stable_best['params']}")
            report_lines.append(f"   - Max Drawdown: {stable_best['max_drawdown']*100:.2f}%, Sharpe: {stable_best['sharpe']:.3f}")
            report_lines.append("")
        
        # Category tốt nhất
        cat_perf = df_results.groupby('category')['composite_score'].mean().sort_values(ascending=False)
        if len(cat_perf) > 0:
            best_cat = cat_perf.index[0]
            report_lines.append(f"3. Category hiệu quả nhất: {best_cat}")
            report_lines.append(f"   - Average Score: {cat_perf[best_cat]:.2f}")
            report_lines.append("")
        
        report_lines.append("=" * 100)
        
        # Ghi file
        report = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Đã tạo báo cáo: {output_file}")
        return report
    
    def generate_detailed_report(
        self,
        output_file: str = "strategy_detailed_report.txt",
        top_n: int = 10,
    ) -> str:
        """
        Tạo báo cáo chi tiết với phân tích PnL, RR ratio, entry signals, và time analysis.
        """
        logger.info("Bắt đầu tạo báo cáo chi tiết...")
        df_results = self.evaluate_all()
        
        if df_results.empty:
            report = "Không có kết quả để phân tích."
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            return report
        
        # Loại bỏ các strategy có lỗi
        df_results = df_results[~df_results.get('error', pd.Series([False] * len(df_results))).astype(bool)]
        
        if df_results.empty:
            report = "Không có kết quả hợp lệ để phân tích (tất cả strategies đều có lỗi)."
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            return report
        
        # Sắp xếp theo composite_score hoặc total_return
        if 'composite_score' in df_results.columns:
            df_results = df_results.sort_values('composite_score', ascending=False)
        elif 'total_return' in df_results.columns:
            df_results = df_results.sort_values('total_return', ascending=False)
        
        # Lấy top N strategies
        top_strategies = df_results.head(top_n)
        
        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append("BÁO CÁO CHI TIẾT CÁC STRATEGY - PHÂN TÍCH PNL, RR RATIO, ENTRY SIGNALS & TIME ANALYSIS")
        report_lines.append("=" * 120)
        report_lines.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Vốn ban đầu: ${self.initial_capital:,.2f}")
        report_lines.append(f"Phí giao dịch: {self.commission*100:.2f}%")
        report_lines.append(f"SL/TP: {'Có' if self.use_stops else 'Không'}")
        if self.use_stops:
            report_lines.append(f"  - Stop Loss: {self.sl_pct*100:.1f}%")
            report_lines.append(f"  - Take Profit: {self.tp_pct*100:.1f}%")
        report_lines.append("")
        report_lines.append(f"Phân tích Top {len(top_strategies)} strategies:")
        report_lines.append("")
        
        # Phân tích từng strategy
        for idx, (_, row) in enumerate(top_strategies.iterrows(), 1):
            strategy_name = row['strategy_name']
            params = row.get('params', {})
            
            report_lines.append("-" * 120)
            report_lines.append(f"{idx}. {strategy_name}")
            report_lines.append(f"   Parameters: {params}")
            report_lines.append("-" * 120)
            report_lines.append("")
            
            # Thông tin cơ bản
            report_lines.append("📊 THÔNG TIN CƠ BẢN:")
            report_lines.append(f"   - Total Return: {row.get('total_return', 0)*100:.2f}%")
            report_lines.append(f"   - Final Equity: ${row.get('final_equity', 0):,.2f}")
            report_lines.append(f"   - Sharpe Ratio: {row.get('sharpe', 0):.3f}")
            report_lines.append(f"   - Max Drawdown: {row.get('max_drawdown', 0)*100:.2f}%")
            report_lines.append(f"   - Win Rate: {row.get('win_rate', 0)*100:.1f}%")
            report_lines.append("")
            
            # Phân tích PnL
            report_lines.append("💰 PHÂN TÍCH PNL:")
            total_pnl = row.get('total_pnl', 0)
            avg_win = row.get('avg_win', 0)
            avg_loss = row.get('avg_loss', 0)
            profit_factor = row.get('profit_factor', 0)
            
            report_lines.append(f"   - Total PnL: ${total_pnl:,.2f}")
            report_lines.append(f"   - Average Win: ${avg_win:,.2f}")
            report_lines.append(f"   - Average Loss: ${avg_loss:,.2f}")
            report_lines.append(f"   - Profit Factor: {profit_factor:.2f}")
            report_lines.append("")
            
            # Phân tích RR Ratio
            report_lines.append("📈 PHÂN TÍCH RISK-REWARD RATIO:")
            rr_ratio = row.get('rr_ratio', 0)
            if rr_ratio > 0:
                report_lines.append(f"   - RR Ratio: {rr_ratio:.2f}")
                if rr_ratio >= 2.0:
                    report_lines.append(f"   - Đánh giá: ⭐⭐⭐ Tuyệt vời (RR >= 2.0)")
                elif rr_ratio >= 1.5:
                    report_lines.append(f"   - Đánh giá: ⭐⭐ Tốt (RR >= 1.5)")
                elif rr_ratio >= 1.0:
                    report_lines.append(f"   - Đánh giá: ⭐ Khá (RR >= 1.0)")
                else:
                    report_lines.append(f"   - Đánh giá: ⚠️ Cần cải thiện (RR < 1.0)")
            else:
                report_lines.append(f"   - RR Ratio: Không có dữ liệu (chưa có lệnh thắng/thua)")
            report_lines.append("")
            
            # Phân tích Entry Signals
            report_lines.append("🎯 PHÂN TÍCH LỆNH VÀO:")
            total_trades = row.get('total_trades', 0)
            long_trades = row.get('long_trades', 0)
            short_trades = row.get('short_trades', 0)
            
            report_lines.append(f"   - Tổng số lệnh: {total_trades}")
            if total_trades > 0:
                report_lines.append(f"   - Lệnh Long: {long_trades} ({long_trades/total_trades*100:.1f}%)")
                report_lines.append(f"   - Lệnh Short: {short_trades} ({short_trades/total_trades*100:.1f}%)")
                
                # Phân tích chiến lược
                if long_trades > short_trades * 1.5:
                    report_lines.append(f"   - Chiến lược: Chủ yếu Long (Bullish)")
                elif short_trades > long_trades * 1.5:
                    report_lines.append(f"   - Chiến lược: Chủ yếu Short (Bearish)")
                else:
                    report_lines.append(f"   - Chiến lược: Cân bằng Long/Short")
            else:
                report_lines.append(f"   - Không có lệnh được thực hiện")
            report_lines.append("")
            
            # Phân tích thời gian
            report_lines.append("⏱️ PHÂN TÍCH THỜI GIAN:")
            avg_holding_time = row.get('avg_holding_time', 0)
            if avg_holding_time > 0:
                if avg_holding_time < 1:
                    report_lines.append(f"   - Thời gian giữ lệnh TB: {avg_holding_time*60:.1f} phút")
                elif avg_holding_time < 24:
                    report_lines.append(f"   - Thời gian giữ lệnh TB: {avg_holding_time:.1f} giờ")
                else:
                    report_lines.append(f"   - Thời gian giữ lệnh TB: {avg_holding_time/24:.1f} ngày")
                
                # Phân loại theo thời gian
                if avg_holding_time < 1:
                    report_lines.append(f"   - Loại: Scalping (giữ lệnh < 1 giờ)")
                elif avg_holding_time < 24:
                    report_lines.append(f"   - Loại: Day Trading (giữ lệnh < 1 ngày)")
                elif avg_holding_time < 168:  # 7 ngày
                    report_lines.append(f"   - Loại: Swing Trading (giữ lệnh vài ngày)")
                else:
                    report_lines.append(f"   - Loại: Position Trading (giữ lệnh dài hạn)")
            else:
                report_lines.append(f"   - Thời gian giữ lệnh TB: Không có dữ liệu")
            report_lines.append("")
            
            # Đánh giá tổng thể
            report_lines.append("📋 ĐÁNH GIÁ TỔNG THỂ:")
            score_parts = []
            
            # Điểm PnL
            if total_pnl > 0:
                score_parts.append("✅ PnL dương")
            else:
                score_parts.append("❌ PnL âm")
            
            # Điểm RR
            if rr_ratio >= 2.0:
                score_parts.append("✅ RR tốt (>=2.0)")
            elif rr_ratio >= 1.0:
                score_parts.append("⚠️ RR trung bình")
            elif rr_ratio > 0:
                score_parts.append("❌ RR thấp (<1.0)")
            
            # Điểm Win Rate
            win_rate = row.get('win_rate', 0)
            if win_rate >= 0.6:
                score_parts.append("✅ Win rate cao (>=60%)")
            elif win_rate >= 0.5:
                score_parts.append("⚠️ Win rate trung bình (50-60%)")
            elif win_rate > 0:
                score_parts.append("❌ Win rate thấp (<50%)")
            
            # Điểm Profit Factor
            if profit_factor >= 2.0:
                score_parts.append("✅ Profit Factor tốt (>=2.0)")
            elif profit_factor >= 1.5:
                score_parts.append("⚠️ Profit Factor khá (1.5-2.0)")
            elif profit_factor >= 1.0:
                score_parts.append("⚠️ Profit Factor trung bình (1.0-1.5)")
            else:
                score_parts.append("❌ Profit Factor thấp (<1.0)")
            
            for part in score_parts:
                report_lines.append(f"   {part}")
            
            report_lines.append("")
            report_lines.append("")
        
        # So sánh tổng hợp
        report_lines.append("=" * 120)
        report_lines.append("SO SÁNH TỔNG HỢP")
        report_lines.append("=" * 120)
        report_lines.append("")
        
        # Strategy tốt nhất theo từng tiêu chí
        if 'total_pnl' in top_strategies.columns:
            best_pnl = top_strategies.loc[top_strategies['total_pnl'].idxmax()]
            report_lines.append(f"💰 Strategy có PnL cao nhất: {best_pnl['strategy_name']}")
            report_lines.append(f"   - Total PnL: ${best_pnl['total_pnl']:,.2f}")
            report_lines.append("")
        
        if 'rr_ratio' in top_strategies.columns:
            valid_rr = top_strategies[top_strategies['rr_ratio'] > 0]
            if not valid_rr.empty:
                best_rr = valid_rr.loc[valid_rr['rr_ratio'].idxmax()]
                report_lines.append(f"📈 Strategy có RR Ratio tốt nhất: {best_rr['strategy_name']}")
                report_lines.append(f"   - RR Ratio: {best_rr['rr_ratio']:.2f}")
                report_lines.append("")
        
        if 'profit_factor' in top_strategies.columns:
            valid_pf = top_strategies[top_strategies['profit_factor'] > 0]
            if not valid_pf.empty:
                best_pf = valid_pf.loc[valid_pf['profit_factor'].idxmax()]
                report_lines.append(f"💎 Strategy có Profit Factor tốt nhất: {best_pf['strategy_name']}")
                report_lines.append(f"   - Profit Factor: {best_pf['profit_factor']:.2f}")
                report_lines.append("")
        
        if 'win_rate' in top_strategies.columns:
            best_wr = top_strategies.loc[top_strategies['win_rate'].idxmax()]
            report_lines.append(f"🎯 Strategy có Win Rate cao nhất: {best_wr['strategy_name']}")
            report_lines.append(f"   - Win Rate: {best_wr['win_rate']*100:.1f}%")
            report_lines.append("")
        
        report_lines.append("=" * 120)
        
        # Ghi file
        report = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Đã tạo báo cáo chi tiết: {output_file}")
        return report


































