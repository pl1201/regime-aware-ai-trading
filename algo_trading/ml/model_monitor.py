"""
Model Performance Monitor - Theo dõi performance và tự động trigger retraining.

Tính năng:
- Log trades và tính performance metrics (winrate, Sharpe, drawdown)
- So sánh với baseline (backtest metrics)
- Tự động phát hiện model drift
- Trigger retraining khi cần
- Gửi alerts qua Telegram
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """
    Monitor model performance và tự động trigger retraining khi cần.
    """
    
    def __init__(
        self,
        model_path: str,
        min_winrate: float = 0.45,
        min_sharpe: float = 0.5,
        lookback_days: int = 30,
        retrain_threshold: float = 0.15,  # Performance giảm >15%
        baseline_winrate: float = 0.52,  # Từ backtest
        baseline_sharpe: float = 1.0,
        min_trades_for_evaluation: int = 20,
        history_file: str = "data/model_performance_history.json"
    ):
        """
        Args:
            model_path: Đường dẫn đến model file
            min_winrate: Winrate tối thiểu chấp nhận được
            min_sharpe: Sharpe ratio tối thiểu
            lookback_days: Số ngày để đánh giá performance
            retrain_threshold: Ngưỡng giảm performance để trigger retrain (0.15 = 15%)
            baseline_winrate: Winrate baseline từ backtest
            baseline_sharpe: Sharpe baseline từ backtest
            min_trades_for_evaluation: Số trades tối thiểu để đánh giá
            history_file: File để lưu performance history
        """
        self.model_path = model_path
        self.min_winrate = min_winrate
        self.min_sharpe = min_sharpe
        self.lookback_days = lookback_days
        self.retrain_threshold = retrain_threshold
        self.baseline_winrate = baseline_winrate
        self.baseline_sharpe = baseline_sharpe
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.history_file = Path(history_file)
        
        # Tạo thư mục nếu chưa có
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Trade history
        self.trade_history: List[Dict[str, Any]] = []
        
        # Load history nếu có
        self._load_history()
    
    def _load_history(self):
        """Load performance history từ file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trade_history = data.get('trades', [])
                    logger.info(f"✅ Đã load {len(self.trade_history)} trades từ history")
            except Exception as e:
                logger.warning(f"⚠️ Không load được history: {e}")
                self.trade_history = []
    
    def _save_history(self):
        """Lưu performance history vào file."""
        try:
            data = {
                'trades': self.trade_history,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Lỗi lưu history: {e}")
    
    def log_trade(
        self,
        entry_price: float,
        exit_price: float,
        direction: int,  # 1 = LONG, -1 = SHORT
        entry_time: datetime,
        exit_time: datetime,
        symbol: str = "BTCUSDT",
        exit_reason: str = "signal",
        regime: Optional[str] = None,
        model_version: Optional[str] = None
    ):
        """
        Log mỗi trade để tính performance metrics.
        
        Args:
            entry_price: Giá vào lệnh
            exit_price: Giá ra lệnh
            direction: 1 = LONG, -1 = SHORT
            entry_time: Thời gian vào lệnh
            exit_time: Thời gian ra lệnh
            symbol: Symbol (mặc định BTCUSDT)
            exit_reason: Lý do thoát (sl, tp, signal_change, etc.)
            regime: Regime tại thời điểm trade
            model_version: Version của model
        """
        # Tính PnL
        pnl = (exit_price - entry_price) * direction
        pnl_pct = (pnl / entry_price) * 100
        
        # Tính holding time
        holding_time = (exit_time - entry_time).total_seconds() / 3600  # hours
        
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'holding_time_hours': holding_time,
            'symbol': symbol,
            'exit_reason': exit_reason,
            'regime': regime,
            'model_version': model_version or Path(self.model_path).stem,
            'logged_at': datetime.now().isoformat()
        }
        
        self.trade_history.append(trade)
        self._save_history()
        
        logger.debug(f"📊 Đã log trade: {symbol} {direction} | PnL: {pnl_pct:.2f}%")
    
    def get_recent_trades(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        Lấy trades gần đây.
        
        Args:
            days: Số ngày gần đây (None = dùng lookback_days)
        
        Returns:
            DataFrame với trades
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_history)
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        if days is None:
            days = self.lookback_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = df[df['exit_time'] >= cutoff_date]
        
        return recent_trades
    
    def calculate_performance_metrics(
        self,
        days: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """
        Tính winrate, Sharpe, drawdown từ trade history.
        
        Args:
            days: Số ngày để tính metrics (None = dùng lookback_days)
        
        Returns:
            Dict với metrics hoặc None nếu không đủ data
        """
        recent_trades = self.get_recent_trades(days)
        
        if len(recent_trades) < self.min_trades_for_evaluation:
            logger.debug(
                f"⚠️ Chưa đủ trades để đánh giá "
                f"(cần {self.min_trades_for_evaluation}, có {len(recent_trades)})"
            )
            return None
        
        # Tính winrate
        winning_trades = recent_trades[recent_trades['pnl'] > 0]
        losing_trades = recent_trades[recent_trades['pnl'] < 0]
        
        winrate = len(winning_trades) / len(recent_trades) if len(recent_trades) > 0 else 0.0
        
        # Tính avg win/loss
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = abs(losing_trades['pnl_pct'].mean()) if len(losing_trades) > 0 else 0.0
        
        # Tính risk/reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Tính Sharpe ratio (giả sử trades là daily)
        returns = recent_trades['pnl_pct'].values
        if len(returns) > 1 and np.std(returns) > 0:
            # Annualized Sharpe (giả sử ~252 trading days/year)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / len(returns))
        else:
            sharpe = 0.0
        
        # Tính max drawdown
        cumulative_pnl = recent_trades['pnl_pct'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # Tính profit factor
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Tính total PnL
        total_pnl = recent_trades['pnl'].sum()
        total_pnl_pct = recent_trades['pnl_pct'].sum()
        
        metrics = {
            'winrate': winrate,
            'sharpe': sharpe,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'risk_reward': risk_reward,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor,
            'total_trades': len(recent_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
        }
        
        return metrics
    
    def check_retrain_needed(self) -> tuple[bool, Dict[str, Any]]:
        """
        Kiểm tra xem có cần retrain không.
        
        Returns:
            (needs_retrain: bool, reasons: Dict)
        """
        metrics = self.calculate_performance_metrics()
        
        if metrics is None:
            return False, {'reason': 'insufficient_data'}
        
        reasons = []
        needs_retrain = False
        
        # Check winrate
        if metrics['winrate'] < self.min_winrate:
            reasons.append(f"Winrate thấp: {metrics['winrate']:.2%} < {self.min_winrate:.2%}")
            needs_retrain = True
        
        # Check Sharpe
        if metrics['sharpe'] < self.min_sharpe:
            reasons.append(f"Sharpe thấp: {metrics['sharpe']:.2f} < {self.min_sharpe:.2f}")
            needs_retrain = True
        
        # Check so với baseline
        winrate_drop = (self.baseline_winrate - metrics['winrate']) / self.baseline_winrate
        if winrate_drop > self.retrain_threshold:
            reasons.append(
                f"Winrate giảm {winrate_drop:.1%} so với baseline "
                f"({metrics['winrate']:.2%} vs {self.baseline_winrate:.2%})"
            )
            needs_retrain = True
        
        sharpe_drop = (self.baseline_sharpe - metrics['sharpe']) / self.baseline_sharpe
        if sharpe_drop > self.retrain_threshold:
            reasons.append(
                f"Sharpe giảm {sharpe_drop:.1%} so với baseline "
                f"({metrics['sharpe']:.2f} vs {self.baseline_sharpe:.2f})"
            )
            needs_retrain = True
        
        return needs_retrain, {
            'needs_retrain': needs_retrain,
            'reasons': reasons,
            'metrics': metrics,
            'baseline_winrate': self.baseline_winrate,
            'baseline_sharpe': self.baseline_sharpe,
        }
    
    def get_performance_report(self) -> str:
        """
        Tạo báo cáo performance dạng text.
        
        Returns:
            String báo cáo
        """
        metrics = self.calculate_performance_metrics()
        
        if metrics is None:
            return "⚠️ Chưa đủ data để đánh giá performance."
        
        needs_retrain, retrain_info = self.check_retrain_needed()
        
        report = f"""
📊 MODEL PERFORMANCE REPORT
{'=' * 50}
📅 Period: {self.lookback_days} days
📈 Total Trades: {metrics['total_trades']}
✅ Winning: {metrics['winning_trades']} | ❌ Losing: {metrics['losing_trades']}

📊 METRICS:
  • Winrate: {metrics['winrate']:.2%} (baseline: {self.baseline_winrate:.2%})
  • Sharpe Ratio: {metrics['sharpe']:.2f} (baseline: {self.baseline_sharpe:.2f})
  • Avg Win: {metrics['avg_win_pct']:.2f}%
  • Avg Loss: {metrics['avg_loss_pct']:.2f}%
  • Risk/Reward: {metrics['risk_reward']:.2f}
  • Profit Factor: {metrics['profit_factor']:.2f}
  • Max Drawdown: {metrics['max_drawdown_pct']:.2f}%
  • Total PnL: {metrics['total_pnl_pct']:.2f}%

{'=' * 50}
"""
        
        if needs_retrain:
            report += f"⚠️ CẦN RETRAIN MODEL!\n"
            report += "Lý do:\n"
            for reason in retrain_info['reasons']:
                report += f"  • {reason}\n"
        else:
            report += "✅ Model performance OK, chưa cần retrain.\n"
        
        return report
    
    def trigger_retrain_notification(self):
        """
        Gửi notification khi cần retrain.
        Có thể tích hợp với Telegram bot.
        """
        needs_retrain, info = self.check_retrain_needed()
        
        if needs_retrain:
            report = self.get_performance_report()
            logger.warning(f"⚠️ MODEL DRIFT DETECTED!\n{report}")
            
            # Có thể gửi qua Telegram ở đây
            try:
                from algo_trading.live.telegram_bot import send_signal_notification
                send_signal_notification(
                    f"⚠️ MODEL CẦN RETRAIN!\n\n{report}"
                )
            except ImportError:
                pass
            except Exception as e:
                logger.error(f"Lỗi gửi Telegram notification: {e}")


# Example usage:
if __name__ == "__main__":
    # Tạo monitor
    monitor = ModelPerformanceMonitor(
        model_path="models/regime_ensemble_optimized.pkl",
        min_winrate=0.45,
        min_sharpe=0.5,
        lookback_days=30,
        baseline_winrate=0.52,
        baseline_sharpe=1.0
    )
    
    # Simulate một số trades
    now = datetime.now()
    monitor.log_trade(
        entry_price=50000,
        exit_price=51000,
        direction=1,
        entry_time=now - timedelta(hours=2),
        exit_time=now,
        exit_reason="tp"
    )
    
    # Check performance
    report = monitor.get_performance_report()
    print(report)
    
    # Check retrain
    needs_retrain, info = monitor.check_retrain_needed()
    print(f"Needs retrain: {needs_retrain}")
