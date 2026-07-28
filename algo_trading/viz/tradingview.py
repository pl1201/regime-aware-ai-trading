"""
TradingView Charting Library integration for Streamlit

This module provides functions to create TradingView-compatible charts
that can be embedded in Streamlit UI to display indicators and trading signals.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json


def prepare_tradingview_data(
    df: pd.DataFrame,
    signals: Optional[pd.Series] = None,
    indicators: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Chuẩn bị dữ liệu cho TradingView Charting Library.
    
    Args:
        df: DataFrame với OHLCV và index là DatetimeIndex
        signals: Series với giá trị {-1, 0, 1} cho buy/sell signals
        indicators: Dict tên indicator -> Series (ví dụ {'SMA20': df['SMA20']})
    
    Returns:
        Dict chứa data và config cho TradingView
    """
    # Đảm bảo index là DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Chuẩn bị OHLCV data
    ohlcv_data = []
    for idx, row in df.iterrows():
        timestamp = int(idx.timestamp())
        ohlcv_data.append({
            'time': timestamp,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
        })
    
    # Chuẩn bị indicators data
    indicators_data = {}
    if indicators:
        for name, series in indicators.items():
            if series is not None and len(series) > 0:
                indicator_points = []
                for idx, val in series.items():
                    if pd.notna(val):
                        timestamp = int(pd.to_datetime(idx).timestamp())
                        indicator_points.append({
                            'time': timestamp,
                            'value': float(val),
                        })
                if indicator_points:
                    indicators_data[name] = indicator_points
    
    # Chuẩn bị signals (buy/sell markers)
    signals_data = {'buy': [], 'sell': []}
    if signals is not None:
        sig_diff = signals.diff().fillna(0)
        for idx, change in sig_diff.items():
            timestamp = int(pd.to_datetime(idx).timestamp())
            price = float(df.loc[idx, 'close'])
            if change > 0:  # Buy signal
                signals_data['buy'].append({
                    'time': timestamp,
                    'price': price,
                    'text': 'BUY',
                })
            elif change < 0:  # Sell signal
                signals_data['sell'].append({
                    'time': timestamp,
                    'price': price,
                    'text': 'SELL',
                })
    
    return {
        'ohlcv': ohlcv_data,
        'indicators': indicators_data,
        'signals': signals_data,
    }


def create_tradingview_html(
    data: Dict[str, Any],
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    theme: str = "dark",
    height: int = 600,
) -> str:
    """
    Tạo HTML code để embed TradingView Charting Library.
    
    Args:
        data: Dict từ prepare_tradingview_data()
        symbol: Tên symbol (ví dụ "BTCUSDT")
        interval: Timeframe (ví dụ "1h", "1d")
        theme: "light" hoặc "dark"
        height: Chiều cao chart (pixels)
    
    Returns:
        HTML string để embed vào Streamlit
    """
    # Escape JSON để tránh lỗi trong HTML
    ohlcv_json_escaped = json.dumps(data['ohlcv']).replace('</', '<\\/')
    indicators_json_escaped = json.dumps(data['indicators']).replace('</', '<\\/')
    signals_json_escaped = json.dumps(data['signals']).replace('</', '<\\/')
    
    text_color = '#191919' if theme == 'light' else '#d1d4dc'
    grid_color = '#e0e3eb' if theme == 'light' else '#2B2B43'
    chart_id = symbol.replace('/', '_').replace('-', '_')
    
    html_template = f"""
    <div id="tradingview_chart_{chart_id}" style="height: {height}px; width: 100%;"></div>
    <script type="text/javascript" src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
    <script type="text/javascript">
        (function() {{
            const chartId = 'tradingview_chart_{chart_id}';
            
            // Function to initialize chart after library loads
            function initChart() {{
                try {{
                    const chartContainer = document.getElementById(chartId);
                    if (!chartContainer) {{
                        console.error('Chart container not found:', chartId);
                        return;
                    }}
                    
                    // Check if library is loaded
                    if (typeof LightweightCharts === 'undefined') {{
                        console.error('LightweightCharts library not loaded');
                        chartContainer.innerHTML = '<p style="color: red; padding: 20px;">Error: TradingView library không load được. Kiểm tra kết nối internet.</p>';
                        return;
                    }}
                    
                    console.log('LightweightCharts loaded');
                    console.log('LightweightCharts methods:', Object.keys(LightweightCharts));
                    console.log('Initializing TradingView chart...');
                    
                    const chart = LightweightCharts.createChart(chartContainer, {{
                        layout: {{
                            background: {{ type: '{theme}' }},
                            textColor: '{text_color}',
                        }},
                        grid: {{
                            vertLines: {{ color: '{grid_color}' }},
                            horzLines: {{ color: '{grid_color}' }},
                        }},
                        width: chartContainer.clientWidth,
                        height: {height},
                        timeScale: {{
                            timeVisible: true,
                            secondsVisible: false,
                        }},
                    }});
                    
                    // Add candlestick series - version 3.8.0 uses addCandlestickSeries method
                    let candlestickSeries;
                    console.log('Attempting to add candlestick series...');
                    console.log('LightweightCharts available:', typeof LightweightCharts !== 'undefined');
                    
                    try {{
                        // Version 3.8.0 has addCandlestickSeries method
                        if (typeof chart.addCandlestickSeries === 'function') {{
                            console.log('Using addCandlestickSeries method (v3.8.0)');
                            candlestickSeries = chart.addCandlestickSeries({{
                                upColor: '#26a69a',
                                downColor: '#ef5350',
                                borderVisible: false,
                                wickUpColor: '#26a69a',
                                wickDownColor: '#ef5350',
                            }});
                        }}
                        // Fallback: try addSeries if available
                        else if (typeof chart.addSeries === 'function') {{
                            console.log('Trying addSeries method');
                            // Check if SeriesType exists
                            if (LightweightCharts.SeriesType) {{
                                candlestickSeries = chart.addSeries(LightweightCharts.SeriesType.Candlestick, {{
                                    upColor: '#26a69a',
                                    downColor: '#ef5350',
                                    borderVisible: false,
                                    wickUpColor: '#26a69a',
                                    wickDownColor: '#ef5350',
                                }});
                            }} else {{
                                candlestickSeries = chart.addSeries('Candlestick', {{
                                    upColor: '#26a69a',
                                    downColor: '#ef5350',
                                    borderVisible: false,
                                    wickUpColor: '#26a69a',
                                    wickDownColor: '#ef5350',
                                }});
                            }}
                        }}
                        else {{
                            throw new Error('No method to add candlestick series found');
                        }}
                        
                        console.log('Successfully created candlestick series');
                    }} catch (e) {{
                        console.error('Error creating candlestick series:', e);
                        console.error('Error message:', e.message);
                        console.error('Error stack:', e.stack);
                        chartContainer.innerHTML = '<p style="color: red; padding: 20px;">Error: Không thể tạo candlestick series.<br>Error: ' + e.message + '<br><br>Vui lòng:<br>1. Kiểm tra Console (F12) để xem chi tiết<br>2. Thử refresh trang (Ctrl+F5)<br>3. Hoặc tắt TradingView Chart và dùng Matplotlib</p>';
                        return;
                    }}
                    
                    if (!candlestickSeries) {{
                        chartContainer.innerHTML = '<p style="color: red; padding: 20px;">Error: Không thể tạo candlestick series.</p>';
                        return;
                    }}
                    
                    console.log('Candlestick series created successfully:', typeof candlestickSeries);
                    
                    // Load OHLCV data
                    const ohlcvData = {ohlcv_json_escaped};
                    if (!ohlcvData || ohlcvData.length === 0) {{
                        console.error('No OHLCV data');
                        chartContainer.innerHTML = '<p style="color: orange; padding: 20px;">Warning: Không có dữ liệu OHLCV.</p>';
                        return;
                    }}
                    
                    candlestickSeries.setData(ohlcvData.map(d => ({{
                        time: d.time,
                        open: d.open,
                        high: d.high,
                        low: d.low,
                        close: d.close,
                    }})));
                    
                    // Add indicators
                    const indicatorsData = {indicators_json_escaped};
                    const indicatorSeries = {{}};
                    
                    Object.keys(indicatorsData).forEach(name => {{
                        if (indicatorsData[name] && indicatorsData[name].length > 0) {{
                            const color = getIndicatorColor(name);
                            indicatorSeries[name] = chart.addLineSeries({{
                                color: color,
                                lineWidth: 2,
                                title: name,
                            }});
                            indicatorSeries[name].setData(indicatorsData[name].map(d => ({{
                                time: d.time,
                                value: d.value,
                            }})));
                        }}
                    }});
                    
                    // Add buy/sell signals
                    const signalsData = {signals_json_escaped};
                    const markers = [];
                    
                    // Buy signals (green markers)
                    if (signalsData.buy && signalsData.buy.length > 0) {{
                        signalsData.buy.forEach(signal => {{
                            markers.push({{
                                time: signal.time,
                                position: 'belowBar',
                                color: '#26a69a',
                                shape: 'arrowUp',
                                text: signal.text || 'BUY',
                            }});
                        }});
                    }}
                    
                    // Sell signals (red markers)
                    if (signalsData.sell && signalsData.sell.length > 0) {{
                        signalsData.sell.forEach(signal => {{
                            markers.push({{
                                time: signal.time,
                                position: 'aboveBar',
                                color: '#ef5350',
                                shape: 'arrowDown',
                                text: signal.text || 'SELL',
                            }});
                        }});
                    }}
                    
                    if (markers.length > 0) {{
                        candlestickSeries.setMarkers(markers);
                    }}
                    
                    // Helper function to assign colors to indicators
                    function getIndicatorColor(name) {{
                        const colors = {{
                            'SMA20': '#2196F3',
                            'EMA20': '#FF9800',
                            'BB_UPPER': '#9C27B0',
                            'BB_MID': '#607D8B',
                            'BB_LOWER': '#9C27B0',
                            'VWAP': '#4CAF50',
                        }};
                        return colors[name] || '#FFC107';
                    }}
                    
                    // Auto-resize
                    chart.timeScale().fitContent();
                    
                    // Handle window resize
                    window.addEventListener('resize', () => {{
                        chart.applyOptions({{ width: chartContainer.clientWidth }});
                    }});
                    
                    console.log('TradingView chart initialized successfully');
                }} catch (error) {{
                    console.error('Error initializing TradingView chart:', error);
                    const chartContainer = document.getElementById(chartId);
                    if (chartContainer) {{
                        chartContainer.innerHTML = '<p style="color: red; padding: 20px;">Error: ' + error.message + '<br>Stack: ' + (error.stack || 'N/A') + '</p>';
                    }}
                }}
            }}
            
            // Wait for library to load (script tag is loaded before this)
            function waitForLibrary() {{
                if (typeof LightweightCharts !== 'undefined') {{
                    console.log('Library ready, initializing chart...');
                    initChart();
                }} else {{
                    console.log('Waiting for library to load...');
                    setTimeout(waitForLibrary, 50);
                }}
            }}
            
            // Start waiting for library (with timeout)
            let attempts = 0;
            const maxAttempts = 100; // 5 seconds max wait
            
            function waitForLibraryWithTimeout() {{
                if (typeof LightweightCharts !== 'undefined') {{
                    console.log('Library ready, initializing chart...');
                    initChart();
                }} else {{
                    attempts++;
                    if (attempts < maxAttempts) {{
                        setTimeout(waitForLibraryWithTimeout, 50);
                    }} else {{
                        const chartContainer = document.getElementById(chartId);
                        if (chartContainer) {{
                            chartContainer.innerHTML = '<p style="color: red; padding: 20px;">Error: TradingView library không load được sau 5 giây.<br>Kiểm tra kết nối internet hoặc thử refresh trang.</p>';
                        }}
                    }}
                }}
            }}
            
            waitForLibraryWithTimeout();
        }})();
    </script>
    """
    return html_template


def create_tradingview_pinescript(
    strategy_name: str,
    indicators: List[str],
    buy_condition: str,
    sell_condition: str,
) -> str:
    """
    Tạo Pine Script code để publish lên TradingView.
    
    Args:
        strategy_name: Tên strategy
        indicators: List tên indicators cần dùng
        buy_condition: Điều kiện mua (Pine Script code)
        sell_condition: Điều kiện bán (Pine Script code)
    
    Returns:
        Pine Script code string
    """
    pinescript = f"""//@version=5
indicator("{strategy_name}", overlay=true)

// Indicators
"""
    
    # Generate indicator code based on list
    if 'SMA20' in indicators:
        pinescript += "sma20 = ta.sma(close, 20)\nplot(sma20, title='SMA20', color=color.blue)\n\n"
    if 'EMA20' in indicators:
        pinescript += "ema20 = ta.ema(close, 20)\nplot(ema20, title='EMA20', color=color.orange)\n\n"
    if 'BB' in str(indicators):
        pinescript += """[bb_upper, bb_mid, bb_lower] = ta.bb(close, 20, 2)
plot(bb_upper, title='BB Upper', color=color.purple)
plot(bb_mid, title='BB Mid', color=color.gray)
plot(bb_lower, title='BB Lower', color=color.purple)

"""
    if 'VWAP' in indicators:
        pinescript += "vwap = ta.vwap(hlc3)\nplot(vwap, title='VWAP', color=color.green)\n\n"
    
    pinescript += f"""// Buy/Sell Conditions
buy_signal = {buy_condition}
sell_signal = {sell_condition}

// Plot signals
plotshape(buy_signal, title='Buy', location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(sell_signal, title='Sell', location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)

// Alerts
alertcondition(buy_signal, title='Buy Signal', message='Buy signal detected')
alertcondition(sell_signal, title='Sell Signal', message='Sell signal detected')
"""
    
    return pinescript
