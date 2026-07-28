"""
Script helper để khởi động trading bot từ Telegram bot
"""

import os
import sys
import subprocess
import signal
from pathlib import Path

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent

def start_bot():
    """Khởi động trading bot."""
    try:
        # Chạy universal_bot trong subprocess
        cmd = [sys.executable, "-m", "algo_trading.live.universal_bot"]
        
        # Tạo file PID để track process
        pid_file = Path("trading_bot.pid")
        
        # Kiểm tra xem bot đã chạy chưa
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                # Kiểm tra process còn sống không
                try:
                    os.kill(old_pid, 0)  # Signal 0 chỉ kiểm tra, không kill
                    print(f"⚠️ Bot đã chạy với PID {old_pid}")
                    return False, old_pid
                except OSError:
                    # Process không còn sống, xóa file PID cũ
                    pid_file.unlink()
            except (ValueError, FileNotFoundError):
                pass
        
        # Khởi động bot mới
        # Tạo log file để capture output
        log_file = project_root / "trading_bot.log"
        
        if os.name == 'nt':  # Windows
            # Dùng CREATE_NEW_PROCESS_GROUP để tách process
            # Redirect output vào log file
            with open(log_file, 'w', encoding='utf-8') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    cwd=project_root,
                    env=os.environ.copy()
                )
        else:  # Unix/Linux
            with open(log_file, 'w', encoding='utf-8') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=project_root,
                    env=os.environ.copy()
                )
        
        # Lưu PID
        with open(pid_file, 'w', encoding='utf-8') as f:
            f.write(str(process.pid))
        
        # Đợi một chút để kiểm tra process
        import time
        time.sleep(1)
        
        # Kiểm tra process còn sống không
        if process.poll() is None:
            # Process vẫn chạy
            print(f"OK: Da khoi dong bot voi PID {process.pid}")
            return True, process.pid
        else:
            # Process đã dừng, đọc log để xem lỗi
            error_msg = "Bot da dung ngay sau khi khoi dong."
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                        log_content = f.read()
                        if log_content:
                            error_msg += f"\nLog:\n{log_content[-500:]}"  # 500 ký tự cuối
                except:
                    pass
            print(f"ERROR: {error_msg}")
            return False, None
        
    except Exception as e:
        print(f"❌ Lỗi khởi động bot: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def dry_run_bot():
    """Chạy bot một vòng dry-run để verify IO mà không vào lệnh thật."""
    try:
        cmd = [sys.executable, "-m", "algo_trading.live.universal_bot", "--dry-run"]
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        print(result.stdout or "")
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print("OK: Dry-run thanh cong")
            return True

        print(f"ERROR: Dry-run that bai (code={result.returncode})")
        return False
    except Exception as e:
        print(f"❌ Lỗi dry-run bot: {e}")
        return False


def stop_bot():
    """Dừng trading bot."""
    pid_file = Path("trading_bot.pid")
    
    if not pid_file.exists():
        print("⚠️ Không tìm thấy file PID. Bot có thể chưa chạy.")
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Kiểm tra process còn sống không
        try:
            if os.name == 'nt':  # Windows
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:  # Unix/Linux
                os.kill(pid, signal.SIGTERM)
            
            # Đợi một chút
            import time
            time.sleep(1)
            
            # Kiểm tra lại
            try:
                os.kill(pid, 0)
                print(f"⚠️ Process {pid} vẫn còn sống, thử kill mạnh hơn...")
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
                else:
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass  # Process đã dừng
            
            # Xóa file PID
            pid_file.unlink()
            print(f"✅ Đã dừng bot (PID {pid})")
            return True
            
        except OSError:
            print(f"⚠️ Process {pid} không còn tồn tại")
            pid_file.unlink()
            return False
            
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Lỗi đọc file PID: {e}")
        return False


def check_bot_status():
    """Kiểm tra trạng thái bot."""
    pid_file = Path("trading_bot.pid")
    
    if not pid_file.exists():
        return False, None
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Kiểm tra process còn sống không
        try:
            os.kill(pid, 0)  # Signal 0 chỉ kiểm tra
            return True, pid
        except OSError:
            # Process không còn sống
            pid_file.unlink()
            return False, None
            
    except (ValueError, FileNotFoundError):
        return False, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot Manager')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'dry-run'], help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        success, pid = start_bot()
        sys.exit(0 if success else 1)
    elif args.action == 'stop':
        success = stop_bot()
        sys.exit(0 if success else 1)
    elif args.action == 'status':
        running, pid = check_bot_status()
        if running:
            print(f"✅ Bot đang chạy với PID {pid}")
            sys.exit(0)
        else:
            print("❌ Bot không chạy")
            sys.exit(1)
    elif args.action == 'dry-run':
        success = dry_run_bot()
        sys.exit(0 if success else 1)
