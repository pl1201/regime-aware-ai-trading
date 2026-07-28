"""
Module tính toán thống kê trades: winrate, profit factor, average win/loss, v.v.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def calculate_trade_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Tính toán các thống kê từ DataFrame trades.
    
    Giả định trades DataFrame có các cột:
    - entry_time, exit_time: thời gian vào/ra
    - entry_price, exit_price: giá vào/ra
    - quantity hoặc size: khối lượng
    - pnl hoặc profit: lợi nhuận
    - direction hoặc side: 'long'/'short' hoặc 1/-1
    
    Hoặc có thể có:
    - return: tỷ lệ lợi nhuận (%)
    - pnl: lợi nhuận tuyệt đối
    
    Returns:
        Dict chứa các metrics: winrate, total_trades, winning_trades, losing_trades,
        avg_win, avg_loss, profit_factor, largest_win, largest_loss, v.v.
    """
    if trades is None or trades.empty:
        return {
            "total_trades": 0,
            "winrate": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade": 0.0,
            "expectancy": 0.0,
        }
    
    # Xác định cột PnL
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        # Thử tính từ entry/exit price
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            if 'direction' in trades.columns:
                direction = trades['direction'].replace({'long': 1, 'short': -1, 'buy': 1, 'sell': -1})
            elif 'side' in trades.columns:
                direction = trades['side'].replace({'long': 1, 'short': -1, 'buy': 1, 'sell': -1})
            else:
                direction = 1  # mặc định long
            
            if 'quantity' in trades.columns:
                size = trades['quantity']
            elif 'size' in trades.columns:
                size = trades['size']
            else:
                size = 1.0
            
            pnl = (trades['exit_price'] - trades['entry_price']) * direction * size
            trades = trades.copy()
            trades['pnl'] = pnl
            pnl_col = 'pnl'
        else:
            return {
                "total_trades": len(trades),
                "winrate": 0.0,
                "error": "Không tìm thấy cột PnL hoặc entry/exit price",
            }
    
    pnl = trades[pnl_col]
    
    # Phân loại trades
    winning = pnl > 0
    losing = pnl < 0
    breakeven = pnl == 0
    
    winning_trades = pnl[winning]
    losing_trades = pnl[losing]
    
    total_trades = len(trades)
    n_win = winning.sum()
    n_loss = losing.sum()
    n_breakeven = breakeven.sum()
    
    winrate = (n_win / total_trades * 100) if total_trades > 0 else 0.0
    
    # Tính các metrics
    total_pnl = pnl.sum()
    avg_win = winning_trades.mean() if n_win > 0 else 0.0
    avg_loss = losing_trades.mean() if n_loss > 0 else 0.0
    avg_trade = pnl.mean() if total_trades > 0 else 0.0
    
    # Profit factor = tổng lợi nhuận từ trades thắng / tổng lỗ từ trades thua
    gross_profit = winning_trades.sum() if n_win > 0 else 0.0
    gross_loss = abs(losing_trades.sum()) if n_loss > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    
    # Expectancy = (winrate * avg_win) - (loss_rate * avg_loss)
    loss_rate = (n_loss / total_trades) if total_trades > 0 else 0.0
    expectancy = (winrate / 100 * avg_win) - (loss_rate * abs(avg_loss))
    
    largest_win = winning_trades.max() if n_win > 0 else 0.0
    largest_loss = losing_trades.min() if n_loss > 0 else 0.0
    
    # Thống kê theo direction nếu có
    direction_stats = {}
    if 'direction' in trades.columns or 'side' in trades.columns:
        dir_col = 'direction' if 'direction' in trades.columns else 'side'
        for direction in trades[dir_col].unique():
            dir_trades = trades[trades[dir_col] == direction]
            dir_pnl = dir_trades[pnl_col]
            dir_win = (dir_pnl > 0).sum()
            dir_total = len(dir_trades)
            direction_stats[str(direction)] = {
                "total": dir_total,
                "wins": dir_win,
                "winrate": (dir_win / dir_total * 100) if dir_total > 0 else 0.0,
                "total_pnl": dir_pnl.sum(),
            }
    
    return {
        "total_trades": int(total_trades),
        "winrate": float(winrate),
        "winning_trades": int(n_win),
        "losing_trades": int(n_loss),
        "breakeven_trades": int(n_breakeven),
        "total_pnl": float(total_pnl),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_trade": float(avg_trade),
        "profit_factor": float(profit_factor) if profit_factor != np.inf else float('inf'),
        "expectancy": float(expectancy),
        "largest_win": float(largest_win),
        "largest_loss": float(largest_loss),
        "direction_stats": direction_stats,
    }


def get_winning_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Trả về DataFrame chỉ chứa các trades thắng."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        return pd.DataFrame()
    
    return trades[trades[pnl_col] > 0].copy()


def get_losing_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Trả về DataFrame chỉ chứa các trades thua."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        return pd.DataFrame()
    
    return trades[trades[pnl_col] < 0].copy()


def get_trade_summary_table(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng tóm tắt trades với các cột quan trọng và highlight win/loss.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    
    trades = trades.copy()
    
    # Xác định cột PnL
    pnl_col = None
    for col in ['pnl', 'profit', 'net_pnl', 'return_pct', 'return']:
        if col in trades.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        return trades
    
    # Thêm cột status
    trades['status'] = trades[pnl_col].apply(
        lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Breakeven')
    )
    
    # Sắp xếp theo thời gian exit (hoặc entry nếu không có exit)
    time_col = 'exit_time' if 'exit_time' in trades.columns else 'entry_time'
    if time_col in trades.columns:
        trades = trades.sort_values(time_col, ascending=False)
    
    return trades




























































