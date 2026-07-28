"""
Telegram Bot cho Trading Bot
- Nhận thông báo signal
- Xem kết quả backtest
- Điều khiển bot từ Telegram

Cách sử dụng:
1. Tạo bot với @BotFather trên Telegram
2. Lấy BOT_TOKEN
3. Lấy CHAT_ID (dùng @userinfobot hoặc gửi message cho bot)
4. Thêm vào .env:
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
5. Chạy bot: python -m algo_trading.live.telegram_bot
"""

from __future__ import annotations

import os
import json
import logging
import signal
import subprocess
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Global bot instance (sẽ được set từ universal_bot)
_trading_bot_instance = None
_backtest_results_cache = {}


def set_trading_bot(bot_instance):
    """Set trading bot instance để điều khiển từ Telegram."""
    global _trading_bot_instance
    _trading_bot_instance = bot_instance


def send_signal_notification(
    signal: int,
    price: float,
    symbol: str,
    holding: bool,
    entry_price: Optional[float] = None,
    reason: Optional[str] = None
):
    """
    Gửi thông báo signal đến Telegram.
    
    Args:
        signal: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
        price: Giá hiện tại
        symbol: Mã giao dịch (BTCUSDT)
        holding: Đang giữ position hay không
        entry_price: Giá vào lệnh (nếu có)
        reason: Lý do (nếu có, ví dụ: "SL hit", "TP hit")
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        return  # Không có cấu hình Telegram
    
    try:
        from telegram import Bot
        
        bot = Bot(token=bot_token)
        
        # Format message
        signal_emoji = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
        signal_text = "LONG" if signal > 0 else "SHORT" if signal < 0 else "NEUTRAL"
        
        message = f"{signal_emoji} Signal mới\n\n"
        message += f"Symbol: {symbol}\n"
        message += f"Signal: {signal_text} ({signal})\n"
        message += f"Price: ${price:.8f}\n"
        message += f"Holding: {'✅ Có' if holding else '❌ Không'}\n"
        
        if entry_price:
            # Tính P&L: LONG (signal > 0) = (current - entry) / entry, SHORT (signal < 0) = (entry - current) / entry
            if signal > 0:  # LONG
                pnl_pct = ((price - entry_price) / entry_price) * 100
            elif signal < 0:  # SHORT
                pnl_pct = ((entry_price - price) / entry_price) * 100
            else:
                pnl_pct = 0.0
            
            message += f"Entry Price: ${entry_price:.8f}\n"
            message += f"P&L: {pnl_pct:+.2f}%\n"
        
        if reason:
            message += f"Lý do: {reason}\n"
        
        message += f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        bot.send_message(
            chat_id=chat_id,
            text=message
        )
        logger.info(f"✅ Đã gửi thông báo signal đến Telegram")
    except Exception as e:
        logger.error(f"❌ Lỗi gửi thông báo Telegram: {e}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /start - Chào mừng và hiển thị menu."""
    try:
        logger.info(f"📨 Nhận lệnh /start từ user {update.effective_user.id}")
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("📈 Signals", callback_data="signals"),
            ],
            [
                InlineKeyboardButton("🛑 Stop Bot", callback_data="stop"),
                InlineKeyboardButton("▶️ Start Bot", callback_data="start"),
            ],
            [
                InlineKeyboardButton("📉 Backtest", callback_data="backtest"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_msg = (
            "🤖 Trading Bot Control Panel\n\n"
            "Chào mừng! Tôi là bot điều khiển trading bot của bạn.\n\n"
            "Các lệnh có sẵn:\n"
            "/start - Hiển thị menu này\n"
            "/status - Xem trạng thái bot\n"
            "/signals - Xem signals gần đây\n"
            "/backtest - Xem kết quả backtest\n"
            "/stop - Dừng bot\n"
            "/start_bot - Khởi động bot\n"
            "/help - Xem hướng dẫn\n\n"
            "Hoặc dùng các nút bên dưới ⬇️"
        )
        
        await update.message.reply_text(
            welcome_msg,
            reply_markup=reply_markup
        )
        logger.info("✅ Đã gửi phản hồi /start")
    except Exception as e:
        logger.exception(f"❌ Lỗi trong start_command: {e}")
        try:
            await update.message.reply_text(
                f"❌ Lỗi: {str(e)}\n\n"
                "Vui lòng thử lại hoặc liên hệ admin."
            )
        except:
            pass


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /status - Xem trạng thái bot."""
    if _trading_bot_instance is None:
        await update.message.reply_text("❌ Bot chưa được khởi động hoặc chưa kết nối.")
        return
    
    # Gửi thông báo "Đang xử lý..." trước
    processing_msg = await update.message.reply_text("⏳ Đang lấy trạng thái...")
    
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        bot = _trading_bot_instance
        config = bot.config
        
        # Chạy các tác vụ có thể timeout trong thread pool với timeout
        def get_status_sync():
            """Lấy trạng thái trong thread riêng để tránh block."""
            try:
                # Lấy giá với timeout
                current_price = bot.client.get_last_price(config.symbol)
                
                # Lấy dữ liệu với timeout (giảm số lượng bars để nhanh hơn)
                df = bot.client.get_klines_df(config.symbol, config.interval, 20)  # Giảm từ 50 xuống 20
                
                cached_signal = getattr(bot, "latest_signal", None)
                cached_error = getattr(bot, "latest_signal_error", None)

                latest_signal = cached_signal if cached_signal is not None else 0
                signal_error = cached_error if cached_error else None
                
                # Tính signal với timeout (bỏ qua nếu quá lâu)
                # Chỉ recompute nếu chưa có cached signal (hoặc cached đang lỗi) và strategy đủ nhẹ.
                if (
                    (cached_signal is None or cached_error)
                    and not df.empty
                    and hasattr(bot.strategy, 'generate_signals')
                ):
                    try:
                        # Chỉ tính signal nếu không phải ML strategy phức tạp
                        strategy_name = config.strategy_name.lower()
                        if 'regime' not in strategy_name or len(df) < 20:
                            result = bot.strategy.generate_signals(df)
                            if not result.signals.empty:
                                latest_signal = int(result.signals.iloc[-1]) if pd.notna(result.signals.iloc[-1]) else 0
                                signal_error = None
                        else:
                            # Với regime/ml strategies: /status không recompute để tránh timeout.
                            # Nếu chưa có cached signal, báo lý do rõ ràng.
                            if cached_signal is None:
                                signal_error = "Signal calculation skipped (too slow). Wait for next bot cycle or use /signals"
                    except Exception as e:
                        signal_error = str(e)[:50]
                        logger.debug(f"Không thể lấy signal: {e}")
                
                return {
                    'current_price': current_price,
                    'latest_signal': latest_signal,
                    'signal_error': signal_error,
                    'success': True
                }
            except Exception as e:
                return {
                    'error': str(e),
                    'success': False
                }
        
        # Chạy với timeout 10 giây
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(executor, get_status_sync)
                result = await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            # Timeout - hiển thị thông tin cơ bản không cần API
            result = {'success': False, 'error': 'Timeout khi lấy dữ liệu từ API'}
        
        if not result.get('success', False):
            # Hiển thị thông tin cơ bản không cần API
            status_msg = "📊 Trạng thái Bot\n\n"
            status_msg += f"Mode: {config.mode.upper()}\n"
            status_msg += f"Symbol: {config.symbol}\n"
            status_msg += f"Interval: {config.interval}\n"
            status_msg += f"Strategy: {config.strategy_name}\n"
            status_msg += f"Holding: {'✅ Có' if bot.holding else '❌ Không'}\n"
            
            if bot.entry_price:
                status_msg += f"Entry Price: ${bot.entry_price:.8f}\n"
                status_msg += f"Position Size: {abs(bot.position_size):.8f}\n"
            
            status_msg += f"\n⚠️ Không thể lấy giá hiện tại: {result.get('error', 'Timeout')}\n"
            status_msg += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            await processing_msg.edit_text(status_msg)
            return
        
        # Có dữ liệu, hiển thị đầy đủ
        current_price = result['current_price']
        latest_signal = result['latest_signal']
        signal_error = result.get('signal_error')
        
        def format_price(price: float) -> str:
            """Format giá với 2 số thập phân cho BTC."""
            if price >= 1000:
                return f"${price:,.2f}"
            else:
                return f"${price:.8f}"
        
        def format_signal(signal: int) -> str:
            """Format signal với emoji."""
            if signal > 0:
                return f"🟢 LONG ({signal})"
            elif signal < 0:
                return f"🔴 SHORT ({signal})"
            else:
                return f"⚪ NEUTRAL ({signal})"
        
        status_msg = "📊 Trạng thái Bot\n"
        status_msg += "━━━━━━━━━━━━━━━━━━\n\n"
        
        # Thông tin cơ bản
        mode_emoji = "🔴" if config.mode == "live" else "🟡" if config.mode == "testnet" else "📄"
        status_msg += f"{mode_emoji} Mode: {config.mode.upper()}\n"
        status_msg += f"💰 Symbol: {config.symbol}\n"
        status_msg += f"⏱️ Interval: {config.interval}\n"
        status_msg += f"🎯 Strategy: {config.strategy_name}\n"
        status_msg += f"💵 Current Price: {format_price(current_price)}\n\n"
        
        # Signal
        status_msg += "━━━━━━━━━━━━━━━━━━\n"
        status_msg += "📡 Signal\n"
        if signal_error:
            status_msg += f"⚠️ N/A ({signal_error})\n"
        else:
            status_msg += f"{format_signal(latest_signal)}\n"
        status_msg += "\n"
        
        # Position Status
        status_msg += "━━━━━━━━━━━━━━━━━━\n"
        status_msg += "📦 Position\n"
        if bot.holding:
            status_msg += "✅ Holding: Có\n"
        
        if bot.entry_price:
                if bot.position_size > 0:
                    pnl_pct = ((current_price - bot.entry_price) / bot.entry_price) * 100
                    position_type = "🟢 LONG"
                elif bot.position_size < 0:  # SHORT position
                    pnl_pct = ((bot.entry_price - current_price) / bot.entry_price) * 100
                    position_type = "🔴 SHORT"
                else:
                    # position_size = 0 hoặc không xác định, dùng holding để xác định
                    # Nếu holding = True nhưng position_size = 0, giả định là LONG
                    pnl_pct = ((current_price - bot.entry_price) / bot.entry_price) * 100 if bot.holding else 0.0
                    position_type = "⚪ UNKNOWN"
                
                # Tính P&L absolute
                pnl_absolute = (current_price - bot.entry_price) * abs(bot.position_size) if bot.position_size > 0 else (bot.entry_price - current_price) * abs(bot.position_size)
                
                status_msg += f"{position_type}\n"
                status_msg += f"📥 Entry Price: {format_price(bot.entry_price)}\n"
                status_msg += f"📊 Position Size: {abs(bot.position_size):.8f} {config.symbol.replace('USDT', '')}\n"
                
                # P&L với emoji và màu
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                status_msg += f"{pnl_emoji} P&L: {pnl_pct:+.2f}% ({format_price(pnl_absolute)})\n"
        
                # Stop Loss / Take Profit nếu có
                if bot.stop_loss_price:
                    status_msg += f"🛑 Stop Loss: {format_price(bot.stop_loss_price)}\n"
                if bot.take_profit_price:
                    status_msg += f"🎯 Take Profit: {format_price(bot.take_profit_price)}\n"
        else:
            status_msg += "❌ Holding: Không\n"
            status_msg += "💤 Đang chờ signal...\n"
        
        status_msg += "\n"
        status_msg += "━━━━━━━━━━━━━━━━━━\n"
        status_msg += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        await processing_msg.edit_text(status_msg)
        
    except Exception as e:
        error_msg = f"❌ Lỗi lấy trạng thái: {str(e)}"
        try:
            await processing_msg.edit_text(error_msg)
        except:
            await update.message.reply_text(error_msg)
        logger.exception("Lỗi status command")


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /signals - Xem signals gần đây."""
    if _trading_bot_instance is None:
        await update.message.reply_text("❌ Bot chưa được khởi động.")
        return
    
    try:
        bot = _trading_bot_instance
        config = bot.config
        
        # Lấy dữ liệu gần đây
        df = bot.client.get_klines_df(config.symbol, config.interval, 50)
        if df.empty:
            await update.message.reply_text("❌ Không có dữ liệu.")
            return
        
        # Generate signals
        result = bot.strategy.generate_signals(df)
        signals = result.signals
        
        if signals.empty:
            await update.message.reply_text("❌ Không có signals.")
            return
        
        # Lấy 10 signals cuối cùng
        recent_signals = signals.tail(10)
        signal_list = []
        
        for idx, (timestamp, signal_val) in enumerate(recent_signals.items(), 1):
            price = df.loc[timestamp, 'close'] if timestamp in df.index else df['close'].iloc[-1]
            signal_emoji = "🟢" if signal_val > 0 else "🔴" if signal_val < 0 else "⚪"
            signal_text = "LONG" if signal_val > 0 else "SHORT" if signal_val < 0 else "NEUTRAL"
            signal_list.append(f"{idx}. {signal_emoji} {signal_text} @ ${price:.8f}")
        
        signals_msg = "📈 Signals gần đây\n\n"
        signals_msg += "\n".join(signal_list)
        signals_msg += f"\n\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        await update.message.reply_text(signals_msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi lấy signals: {str(e)}")
        logger.exception("Lỗi signals command")


async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /backtest - Xem kết quả backtest."""
    logger.info("📉 Nhận lệnh /backtest")
    
    # Kiểm tra cache hoặc file backtest results
    backtest_file = Path("backtest_results.json")
    
    if not backtest_file.exists():
        await update.message.reply_text(
            "❌ Chưa có kết quả backtest.\n\n"
            "Chạy backtest từ Streamlit UI trước, sau đó dùng lệnh này để xem kết quả.\n\n"
            "Lưu ý: Kết quả sẽ được lưu tự động sau khi backtest hoàn thành."
        )
        return
    
    try:
        with open(backtest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Format message
        msg = "📉 Kết quả Backtest\n\n"
        
        if "total_return" in results:
            msg += f"Total Return: {results['total_return']:.2f}%\n"
        if "sharpe" in results:
            msg += f"Sharpe Ratio: {results['sharpe']:.2f}\n"
        if "max_drawdown" in results:
            msg += f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
        if "cagr" in results:
            msg += f"CAGR: {results['cagr']:.2f}%\n"
        if "profit_factor" in results:
            msg += f"Profit Factor: {results['profit_factor']:.2f}\n"
        
        if "winrate" in results:
            msg += f"\nWinrate: {results['winrate']:.2f}%\n"
        if "total_trades" in results:
            msg += f"Total Trades: {results['total_trades']}\n"
        if "winning_trades" in results:
            msg += f"Winning Trades: {results['winning_trades']}\n"
        if "losing_trades" in results:
            msg += f"Losing Trades: {results['losing_trades']}\n"
        
        if "expectancy" in results:
            msg += f"Expectancy: {results['expectancy']:.4f}\n"
        if "avg_win" in results:
            msg += f"Avg Win: {results['avg_win']:.4f}\n"
        if "avg_loss" in results:
            msg += f"Avg Loss: {results['avg_loss']:.4f}\n"
        
        # Thêm timestamp nếu có
        if "timestamp" in results:
            from datetime import datetime
            try:
                ts = datetime.fromisoformat(results["timestamp"].replace('Z', '+00:00'))
                msg += f"\n⏰ Thời gian: {ts.strftime('%Y-%m-%d %H:%M:%S')}"
            except:
                pass
        
        await update.message.reply_text(msg)
        logger.info("✅ Đã gửi kết quả backtest")
    except Exception as e:
        error_msg = f"❌ Lỗi đọc kết quả backtest: {str(e)}"
        await update.message.reply_text(error_msg)
        logger.exception("Lỗi backtest command")


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /stop - Dừng bot."""
    # Xử lý cả trường hợp gọi từ command và button callback
    message = update.message
    if message is None and update.callback_query:
        # Được gọi từ button callback
        query = update.callback_query
        if _trading_bot_instance is None:
            await query.edit_message_text("❌ Bot chưa được khởi động.")
            return
        
        try:
            if hasattr(_trading_bot_instance, 'stop_flag'):
                _trading_bot_instance.stop_flag = True
            
            await query.edit_message_text(
                "🛑 Bot đã được yêu cầu dừng.\n\n"
                "Bot sẽ dừng sau khi hoàn thành vòng lặp hiện tại."
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Lỗi dừng bot: {str(e)}")
        return
    
    # Được gọi từ command
    if message is None:
        return
    
    if _trading_bot_instance is None:
        await message.reply_text("❌ Bot chưa được khởi động.")
        return
    
    try:
        # Set flag để bot dừng
        if hasattr(_trading_bot_instance, 'stop_flag'):
            _trading_bot_instance.stop_flag = True
        
        await message.reply_text(
            "🛑 Bot đã được yêu cầu dừng.\n\n"
            "Bot sẽ dừng sau khi hoàn thành vòng lặp hiện tại."
        )
    except Exception as e:
        await message.reply_text(f"❌ Lỗi dừng bot: {str(e)}")


async def start_bot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /start_bot - Khởi động bot."""
    import subprocess
    import sys
    import time
    from pathlib import Path
    
    # Xử lý cả trường hợp gọi từ command và button callback
    message = update.message
    query = update.callback_query
    
    # Kiểm tra xem bot đã chạy chưa
    if _trading_bot_instance is not None:
        response_msg = "✅ Trading bot đã đang chạy!\n\nDùng /status để xem trạng thái."
        if query:
            await query.edit_message_text(response_msg)
        elif message:
            await message.reply_text(response_msg)
        return
    
    # Kiểm tra qua PID file
    pid_file = Path("trading_bot.pid")
    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Signal 0 chỉ kiểm tra
                response_msg = f"✅ Trading bot đã đang chạy (PID {pid})!\n\nDùng /status để xem trạng thái."
                if query:
                    await query.edit_message_text(response_msg)
                elif message:
                    await message.reply_text(response_msg)
                return
            except OSError:
                # Process không còn sống, xóa file PID
                pid_file.unlink()
        except (ValueError, FileNotFoundError):
            pass
    
    # Thử khởi động bot
    try:
        # Dùng script helper
        script_path = Path(__file__).parent.parent.parent / "start_trading_bot.py"
        if not script_path.exists():
            # Fallback: chạy trực tiếp
            cmd = [sys.executable, "-m", "algo_trading.live.universal_bot"]
        else:
            cmd = [sys.executable, str(script_path), "start"]
        
        # Khởi động trong background
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                cwd=Path(__file__).parent.parent.parent
            )
        else:  # Unix/Linux
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                cwd=Path(__file__).parent.parent.parent
            )
        
        # Đợi một chút để kiểm tra
        time.sleep(2)
        
        # Kiểm tra process còn sống không
        if process.poll() is None:  # Process vẫn chạy
            response_msg = (
                f"✅ Đã khởi động trading bot!\n\n"
                f"PID: {process.pid}\n"
                f"Bot đang khởi động...\n\n"
                f"Dùng /status sau vài giây để kiểm tra trạng thái."
            )
        else:
            # Process đã dừng (có thể do lỗi)
            # Đọc log file nếu có
            log_file = Path(__file__).parent.parent.parent / "trading_bot.log"
            error_info = ""
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                        log_content = f.read()
                        if log_content:
                            # Lấy 300 ký tự cuối (thường chứa lỗi)
                            error_info = log_content[-300:].strip()
                except:
                    pass
            
            # Nếu không có log, thử đọc stderr
            if not error_info:
                try:
                    stderr_output = process.stderr.read().decode('utf-8', errors='ignore')[:200]
                    if stderr_output:
                        error_info = stderr_output
                except:
                    pass
            
            if error_info:
                response_msg = (
                    f"⚠️ Bot đã khởi động nhưng dừng ngay.\n\n"
                    f"Có thể có lỗi. Kiểm tra file trading_bot.log hoặc chạy thủ công:\n"
                    f"python -m algo_trading.live.universal_bot\n\n"
                    f"Lỗi:\n{error_info}"
                )
            else:
                response_msg = (
                    f"⚠️ Bot đã khởi động nhưng dừng ngay.\n\n"
                    f"Có thể có lỗi. Kiểm tra file trading_bot.log hoặc chạy thủ công:\n"
                    f"python -m algo_trading.live.universal_bot"
                )
        
        if query:
            await query.edit_message_text(response_msg)
        elif message:
            await message.reply_text(response_msg)
            
    except Exception as e:
        error_msg = (
            f"❌ Không thể khởi động bot tự động.\n\n"
            f"Lỗi: {str(e)}\n\n"
            f"Vui lòng chạy thủ công:\n"
            f"python -m algo_trading.live.universal_bot"
        )
        if query:
            await query.edit_message_text(error_msg)
        elif message:
            await message.reply_text(error_msg)
        logger.exception("Lỗi khởi động bot từ Telegram")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command /help - Hướng dẫn."""
    help_msg = (
        "📖 Hướng dẫn sử dụng\n\n"
        "Các lệnh có sẵn:\n"
        "/start - Hiển thị menu chính\n"
        "/status - Xem trạng thái bot hiện tại\n"
        "/signals - Xem 10 signals gần đây\n"
        "/backtest - Xem kết quả backtest\n"
        "/stop - Dừng bot\n"
        "/start_bot - Hướng dẫn khởi động bot\n"
        "/help - Xem hướng dẫn này\n\n"
        "Tự động:\n"
        "Bot sẽ tự động gửi thông báo khi:\n"
        "• Có signal mới (LONG/SHORT)\n"
        "• Vào/ra lệnh\n"
        "• Hit SL/TP\n"
        "• Có lỗi xảy ra\n\n"
        "Cấu hình:\n"
        "Thêm vào file .env:\n"
        "TELEGRAM_BOT_TOKEN=your_bot_token\n"
        "TELEGRAM_CHAT_ID=your_chat_id"
    )
    
    await update.message.reply_text(help_msg)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý callback từ inline buttons."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "status":
        # Tạo message status tương tự command
        if _trading_bot_instance is None:
            await query.edit_message_text("❌ Bot chưa được khởi động hoặc chưa kết nối.")
            return
        # ... (tương tự status_command)
        await query.edit_message_text("📊 Đang lấy trạng thái...\n\nDùng /status để xem chi tiết.")
    
    elif query.data == "signals":
        await query.edit_message_text("📈 Đang lấy signals...\n\nDùng /signals để xem chi tiết.")
    
    elif query.data == "backtest":
        # Xử lý backtest từ button
        backtest_file = Path("backtest_results.json")
        
        if not backtest_file.exists():
            await query.edit_message_text(
                "❌ Chưa có kết quả backtest.\n\n"
                "Chạy backtest từ Streamlit UI trước, sau đó dùng lệnh này để xem kết quả.\n\n"
                "Lưu ý: Kết quả sẽ được lưu tự động sau khi backtest hoàn thành."
            )
            return
        
        try:
            with open(backtest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Format message
            msg = "📉 Kết quả Backtest\n\n"
            
            if "total_return" in results:
                msg += f"Total Return: {results['total_return']:.2f}%\n"
            if "sharpe" in results:
                msg += f"Sharpe Ratio: {results['sharpe']:.2f}\n"
            if "max_drawdown" in results:
                msg += f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
            if "cagr" in results:
                msg += f"CAGR: {results['cagr']:.2f}%\n"
            if "profit_factor" in results:
                msg += f"Profit Factor: {results['profit_factor']:.2f}\n"
            
            if "winrate" in results:
                msg += f"\nWinrate: {results['winrate']:.2f}%\n"
            if "total_trades" in results:
                msg += f"Total Trades: {results['total_trades']}\n"
            if "winning_trades" in results:
                msg += f"Winning Trades: {results['winning_trades']}\n"
            if "losing_trades" in results:
                msg += f"Losing Trades: {results['losing_trades']}\n"
            
            if "expectancy" in results:
                msg += f"Expectancy: {results['expectancy']:.4f}\n"
            if "avg_win" in results:
                msg += f"Avg Win: {results['avg_win']:.4f}\n"
            if "avg_loss" in results:
                msg += f"Avg Loss: {results['avg_loss']:.4f}\n"
            
            # Thêm timestamp nếu có
            if "timestamp" in results:
                from datetime import datetime
                try:
                    ts = datetime.fromisoformat(results["timestamp"].replace('Z', '+00:00'))
                    msg += f"\n⏰ Thời gian: {ts.strftime('%Y-%m-%d %H:%M:%S')}"
                except:
                    pass
            
            await query.edit_message_text(msg)
            logger.info("✅ Đã gửi kết quả backtest từ button")
        except Exception as e:
            error_msg = f"❌ Lỗi đọc kết quả backtest: {str(e)}"
            await query.edit_message_text(error_msg)
            logger.exception("Lỗi backtest button callback")
    
    elif query.data == "stop":
        # Tạo update mới với callback_query để stop_command biết được gọi từ button
        # update_id thuộc về Update, không phải CallbackQuery
        fake_update = Update(
            update_id=update.update_id,
            callback_query=query,
            message=None,
            edited_message=None,
            channel_post=None,
            edited_channel_post=None,
            inline_query=None,
            chosen_inline_result=None,
            shipping_query=None,
            pre_checkout_query=None,
            poll=None,
            poll_answer=None,
            my_chat_member=None,
            chat_member=None,
            chat_join_request=None,
        )
        await stop_command(fake_update, context)
    
    elif query.data == "start":
        # Tạo update mới với callback_query để start_bot_command biết được gọi từ button
        # update_id thuộc về Update, không phải CallbackQuery
        fake_update = Update(
            update_id=update.update_id,
            callback_query=query,
            message=None,
            edited_message=None,
            channel_post=None,
            edited_channel_post=None,
            inline_query=None,
            chosen_inline_result=None,
            shipping_query=None,
            pre_checkout_query=None,
            poll=None,
            poll_answer=None,
            my_chat_member=None,
            chat_member=None,
            chat_join_request=None,
        )
        await start_bot_command(fake_update, context)
    
    elif query.data == "settings":
        await query.edit_message_text(
            "⚙️ **Settings**\n\n"
            "Cấu hình trong file `.env`:\n"
            "• MODE: paper/testnet/live\n"
            "• STRATEGY: strategy name\n"
            "• SYMBOL: trading pair\n"
            "• RISK_PER_TRADE: risk per trade\n"
            "• SL_ATR_K, TP_ATR_K: stop loss/take profit\n\n"
            "Sau khi thay đổi, restart bot."
        )


def run_telegram_bot():
    """Chạy Telegram bot."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN không được set. Telegram bot sẽ không chạy.")
        return None
    
    # Kiểm tra và kill instance cũ nếu có
    pid_file = Path("telegram_bot.pid")
    if pid_file.exists():
        try:
            with open(pid_file, 'r', encoding='utf-8') as f:
                old_pid = int(f.read().strip())
            
            # Kiểm tra process còn sống không
            try:
                os.kill(old_pid, 0)  # Signal 0 chỉ kiểm tra
                logger.warning(f"⚠️ Phát hiện Telegram bot instance cũ (PID {old_pid}). Đang dừng...")
                try:
                    if os.name == 'nt':  # Windows
                        os.kill(old_pid, signal.CTRL_BREAK_EVENT)
                    else:  # Unix/Linux
                        os.kill(old_pid, signal.SIGTERM)
                    import time
                    time.sleep(1)
                    # Kiểm tra lại
                    try:
                        os.kill(old_pid, 0)
                        if os.name == 'nt':
                            subprocess.run(["taskkill", "/F", "/PID", str(old_pid)], check=False, capture_output=True)
                        else:
                            os.kill(old_pid, signal.SIGKILL)
                    except OSError:
                        pass
                except Exception as e:
                    logger.warning(f"Không thể kill process cũ: {e}")
            except OSError:
                # Process không còn sống
                logger.info(f"Process cũ (PID {old_pid}) đã không còn tồn tại.")
            
            # Xóa file PID cũ
            try:
                pid_file.unlink()
            except:
                pass
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Không thể đọc PID file: {e}")
    
    # Lưu PID hiện tại
    try:
        with open(pid_file, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.warning(f"Không thể lưu PID file: {e}")
    
    # Tạo application với error handler
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log lỗi và thông báo cho user."""
        error_str = str(context.error) if context.error else "Unknown error"
        logger.error(f"Exception while handling an update: {error_str}")
        
        # Bỏ qua lỗi conflict (nhiều instance)
        if "Conflict" in error_str or "terminated by other getUpdates" in error_str:
            logger.warning("⚠️ Phát hiện conflict - có thể có instance khác đang chạy. Dừng bot này...")
            # Xóa PID file và dừng
            try:
                pid_file.unlink()
            except:
                pass
            return
        
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    f"❌ Đã xảy ra lỗi: {error_str[:100]}\n\n"
                    "Vui lòng thử lại sau."
                )
            except:
                pass
    
    application = Application.builder().token(bot_token).build()
    
    # Đăng ký error handler
    application.add_error_handler(error_handler)
    
    # Đăng ký handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("backtest", backtest_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("start_bot", start_bot_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start bot
    logger.info("🤖 Telegram bot đã khởi động. Dùng /start để bắt đầu.")
    logger.info(f"📱 Bot token: {bot_token[:20]}...")
    logger.info(f"📝 PID: {os.getpid()} (đã lưu vào telegram_bot.pid)")
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Bỏ qua các update cũ
        )
    except KeyboardInterrupt:
        logger.info("👋 Đang dừng Telegram bot...")
    finally:
        # Xóa PID file khi dừng
        try:
            if pid_file.exists():
                pid_file.unlink()
        except:
            pass
    
    return application


if __name__ == "__main__":
    run_telegram_bot()
    