"""
Script để điền TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID vào .env
"""

import os
from pathlib import Path

def setup_telegram_env():
    """Hướng dẫn và điền Telegram config vào .env"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ Không tìm thấy file .env")
        return
    
    # Đọc file hiện tại
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Kiểm tra xem đã có config chưa
    has_token = "TELEGRAM_BOT_TOKEN=" in content
    has_chat_id = "TELEGRAM_CHAT_ID=" in content
    
    print("=" * 60)
    print("🔧 Thiết lập Telegram Bot Configuration")
    print("=" * 60)
    print()
    
    # Lấy token
    if has_token:
        # Tìm dòng hiện tại
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('TELEGRAM_BOT_TOKEN='):
                current_value = line.split('=', 1)[1].strip()
                if current_value:
                    print(f"✅ TELEGRAM_BOT_TOKEN đã có: {current_value[:20]}...")
                    use_current = input("Bạn có muốn thay đổi? (y/n): ").lower() == 'y'
                    if not use_current:
                        token = current_value
                        break
                break
        
        if 'token' not in locals() or (has_token and use_current):
            token = input("Nhập TELEGRAM_BOT_TOKEN (từ @BotFather): ").strip()
    else:
        token = input("Nhập TELEGRAM_BOT_TOKEN (từ @BotFather): ").strip()
    
    # Lấy chat_id
    if has_chat_id:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('TELEGRAM_CHAT_ID='):
                current_value = line.split('=', 1)[1].strip()
                if current_value:
                    print(f"✅ TELEGRAM_CHAT_ID đã có: {current_value}")
                    use_current = input("Bạn có muốn thay đổi? (y/n): ").lower() == 'y'
                    if not use_current:
                        chat_id = current_value
                        break
                break
        
        if 'chat_id' not in locals() or (has_chat_id and use_current):
            chat_id = input("Nhập TELEGRAM_CHAT_ID (từ @userinfobot): ").strip()
    else:
        chat_id = input("Nhập TELEGRAM_CHAT_ID (từ @userinfobot): ").strip()
    
    # Cập nhật file
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if line.startswith('TELEGRAM_BOT_TOKEN='):
            lines[i] = f'TELEGRAM_BOT_TOKEN={token}'
            updated = True
        elif line.startswith('TELEGRAM_CHAT_ID='):
            lines[i] = f'TELEGRAM_CHAT_ID={chat_id}'
            updated = True
    
    # Nếu chưa có, thêm vào cuối
    if not updated:
        lines.append('')
        lines.append('# Telegram Bot Configuration')
        lines.append(f'TELEGRAM_BOT_TOKEN={token}')
        lines.append(f'TELEGRAM_CHAT_ID={chat_id}')
    
    # Ghi lại file
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print()
    print("✅ Đã cập nhật file .env!")
    print()
    print("📝 Thông tin đã lưu:")
    print(f"   TELEGRAM_BOT_TOKEN={token[:20]}..." if len(token) > 20 else f"   TELEGRAM_BOT_TOKEN={token}")
    print(f"   TELEGRAM_CHAT_ID={chat_id}")
    print()
    print("🚀 Bây giờ bạn có thể chạy:")
    print("   python run_telegram_bot_only.py")


if __name__ == "__main__":
    try:
        setup_telegram_env()
    except KeyboardInterrupt:
        print("\n\n👋 Đã hủy.")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
