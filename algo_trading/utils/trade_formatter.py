from __future__ import annotations
"""
Utility functions to format trades DataFrame for CSV export.
"""

import pandas as pd
from typing import Optional, Dict, Any


def format_trades_csv(
    trades_df: pd.DataFrame,
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> pd.DataFrame:

    if trades_df.empty:
        return pd.DataFrame(columns=[
            'trade_id', 'date', 'symbol', 'timeframe', 'direction', 'entry_price',
            'stop_loss', 'take_profit', 'exit_price', 'position_size', 'risk_percent',
            'rr', 'result_pnl', 'result_r'
        ])
    
    if not isinstance(trades_df, pd.DataFrame):
        return pd.DataFrame(columns=[
            'trade_id', 'date', 'symbol', 'timeframe', 'direction', 'entry_price',
            'stop_loss', 'take_profit', 'exit_price', 'position_size', 'risk_percent',
            'rr', 'result_pnl', 'result_r'
        ])
    
    formatted_trades = []
    
    trades_df = trades_df.copy().reset_index(drop=True)
    
    for trade_idx, (_, trade) in enumerate(trades_df.iterrows()):
        # Get entry/exit times
        entry_time = None
        if 'entry_time' in trade and pd.notna(trade['entry_time']):
            entry_time = trade['entry_time']
        elif 'entry_idx' in trade and pd.notna(trade['entry_idx']):
            try:
                entry_idx = int(trade['entry_idx'])
                if 0 <= entry_idx < len(df):
                    entry_time = df.index[entry_idx]
            except (ValueError, TypeError, IndexError):
                pass
        
        exit_time = None
        if 'exit_time' in trade and pd.notna(trade['exit_time']):
            exit_time = trade['exit_time']
        elif 'exit_idx' in trade and pd.notna(trade['exit_idx']):
            try:
                exit_idx = int(trade['exit_idx'])
                if 0 <= exit_idx < len(df):
                    exit_time = df.index[exit_idx]
            except (ValueError, TypeError, IndexError):
                pass
        
        # Get values safely
        entry_price = float(trade.get('entry_price', 0)) if pd.notna(trade.get('entry_price')) else 0
        exit_price = float(trade.get('exit_price', 0)) if pd.notna(trade.get('exit_price')) else 0
        position = float(trade.get('position', 0)) if pd.notna(trade.get('position')) else 0
        position_size = float(trade.get('position_size', abs(position))) if pd.notna(trade.get('position_size')) else abs(position)
        stop_loss = float(trade.get('stop_loss')) if pd.notna(trade.get('stop_loss')) else None
        take_profit = float(trade.get('take_profit')) if pd.notna(trade.get('take_profit')) else None
        risk_percent = float(trade.get('risk_percent')) if pd.notna(trade.get('risk_percent')) else None
        pnl = float(trade.get('pnl', 0)) if pd.notna(trade.get('pnl')) else 0
        
        # Determine direction
        if position > 0:
            direction = 'LONG'
        elif position < 0:
            direction = 'SHORT'
        else:
            direction = 'FLAT'
        
        # Calculate risk/reward ratio
        rr = None
        if stop_loss is not None and take_profit is not None and entry_price > 0:
            if direction == 'LONG':
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
            else:  # SHORT
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - take_profit)
            
            if risk > 0:
                rr = reward / risk
        
        # Calculate return percentage
        result_r = None
        if entry_price > 0:
            if direction == 'LONG':
                result_r = ((exit_price - entry_price) / entry_price) * 100
            else: 
                result_r = ((entry_price - exit_price) / entry_price) * 100
        
        # Format date
        date_str = None
        if entry_time:
            if isinstance(entry_time, pd.Timestamp):
                date_str = entry_time.strftime('%Y-%m-%d')
            else:
                try:
                    date_str = pd.Timestamp(entry_time).strftime('%Y-%m-%d')
                except:
                    date_str = str(entry_time)
        
        formatted_trades.append({
            'trade_id': trade_idx + 1, 
            'date': date_str,
            'symbol': symbol or 'UNKNOWN',
            'timeframe': timeframe or 'UNKNOWN',
            'direction': direction,
            'entry_price': round(entry_price, 8) if entry_price else None,
            'stop_loss': round(stop_loss, 8) if stop_loss is not None else None,
            'take_profit': round(take_profit, 8) if take_profit is not None else None,
            'exit_price': round(exit_price, 8) if exit_price else None,
            'position_size': round(position_size, 8) if position_size else None,
            'risk_percent': round(risk_percent * 100, 4) if risk_percent is not None else None,
            'rr': round(rr, 4) if rr is not None else None,
            'result_pnl': round(pnl, 8) if pnl else 0,
            'result_r': round(result_r, 4) if result_r is not None else None,
        })
    
    # Tạo DataFrame từ list of dicts
    if not formatted_trades:
        return pd.DataFrame(columns=[
            'trade_id', 'date', 'symbol', 'timeframe', 'direction', 'entry_price',
            'stop_loss', 'take_profit', 'exit_price', 'position_size', 'risk_percent',
            'rr', 'result_pnl', 'result_r'
        ])
    
    result_df = pd.DataFrame(formatted_trades)
    
    # Đảm bảo có đủ 14 cột
    expected_columns = [
        'trade_id', 'date', 'symbol', 'timeframe', 'direction', 'entry_price',
        'stop_loss', 'take_profit', 'exit_price', 'position_size', 'risk_percent',
        'rr', 'result_pnl', 'result_r'
    ]
    
    # Kiểm tra và đảm bảo tất cả cột có mặt
    for col in expected_columns:
        if col not in result_df.columns:
            result_df[col] = None
    
    # Sắp xếp lại cột theo thứ tự đúng
    result_df = result_df[expected_columns]
    
    return result_df

