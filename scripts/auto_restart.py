"""
Script auto-restart khi bot tắt
"""
import os
import time
import logging
import psutil
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_restart.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Đường dẫn đến PID file
PID_FILE = "trading_bot.pid"
# Đường dẫn đến script chạy bot
START_SCRIPT = "start_trading_bot.py"
# Khoảng thời gian kiểm tra (giây)
CHECK_INTERVAL = 1800  # 30 phút

# Kiểm tra xem bot có đang chạy không
def is_bot_running():
    """Kiểm tra xem bot có đang chạy không"""
    if not os.path.exists(PID_FILE):
        return False

    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        # Kiểm tra xem process có tồn tại không
        process = psutil.Process(pid)
        return process.is_running() and "python" in process.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, FileNotFoundError):
        return False

# Khởi động lại bot
def start_bot():
    """Khởi động lại bot"""
    try:
        # Kiểm tra xem script có tồn tại không
        if not os.path.exists(START_SCRIPT):
            logger.error(f"❌ Script {START_SCRIPT} không tồn tại")
            return

        # Chạy bot trong background
        os.system(f"python {START_SCRIPT} start > bot_output.log 2>&1 &")
        logger.info("✅ Đã khởi động lại bot")
    except Exception as e:
        logger.error(f"❌ Lỗi khi khởi động bot: {e}")

# Hàm chính
def main():
    """Chạy script auto-restart"""
    logger.info("🚀 Bắt đầu script auto-restart")

    while True:
        try:
            if not is_bot_running():
                logger.warning("⚠️  Bot không đang chạy. Đang khởi động lại...")
                start_bot()
            else:
                logger.info("✅ Bot đang chạy bình thường")

            # Chờ trước khi kiểm tra lần tiếp theo
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("🛑 Người dùng dừng script")
            break
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình chạy script: {e}")
            time.sleep(60)  # Chờ 1 phút rồi thử lại

if __name__ == "__main__":
    main()