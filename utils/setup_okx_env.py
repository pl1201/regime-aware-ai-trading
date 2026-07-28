"""
Script giúp cấu hình OKX API credentials vào file .env
Chạy script này để điền thông tin API key OKX một cách dễ dàng
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def setup_okx_env():
    """Hướng dẫn user điền thông tin OKX API và lưu vào .env"""
    
    print("=" * 60)
    print("🚀 CẤU HÌNH OKX API CHO TRADING BOT")
    print("=" * 60)
    print()
    
    # Load .env hiện tại nếu có
    env_path = Path(".env")
    load_dotenv(env_path, override=True)
    
    # Lấy giá trị hiện tại (nếu có)
    current_exchange = os.getenv("EXCHANGE", "")
    current_mode = os.getenv("MODE", "")
    current_okx_key = os.getenv("OKX_API_KEY", "")
    current_okx_secret = os.getenv("OKX_API_SECRET", "")
    current_okx_passphrase = os.getenv("OKX_PASSPHRASE", "")
    current_okx_simulated = os.getenv("OKX_USE_SIMULATED_TRADING", "")
    
    print("📝 Điền thông tin OKX API:")
    print()
    
    # 1. Exchange
    print("1️⃣ Chọn Exchange:")
    if current_exchange:
        print(f"   (Hiện tại: {current_exchange})")
    exchange = input("   Nhập 'okx' hoặc 'binance' [okx]: ").strip().lower() or "okx"
    
    # 2. Mode
    print()
    print("2️⃣ Chọn Mode:")
    print("   - 'paper': Mô phỏng (không gửi lệnh thật)")
    print("   - 'live': Gửi lệnh thật lên OKX")
    if current_mode:
        print(f"   (Hiện tại: {current_mode})")
    mode = input("   Nhập 'paper' hoặc 'live' [live]: ").strip().lower() or "live"
    
    # 3. OKX API Key
    print()
    print("3️⃣ OKX API Key:")
    if current_okx_key:
        print(f"   (Hiện tại: {current_okx_key[:10]}...{current_okx_key[-5:]})")
        use_current = input("   Dùng giá trị hiện tại? (y/n) [n]: ").strip().lower()
        if use_current == 'y':
            okx_key = current_okx_key
        else:
            okx_key = input("   Nhập OKX API Key: ").strip()
    else:
        okx_key = input("   Nhập OKX API Key: ").strip()
    
    # 4. OKX Secret Key
    print()
    print("4️⃣ OKX Secret Key:")
    if current_okx_secret:
        print(f"   (Hiện tại: {current_okx_secret[:5]}...{current_okx_secret[-3:]})")
        use_current = input("   Dùng giá trị hiện tại? (y/n) [n]: ").strip().lower()
        if use_current == 'y':
            okx_secret = current_okx_secret
        else:
            okx_secret = input("   Nhập OKX Secret Key: ").strip()
    else:
        okx_secret = input("   Nhập OKX Secret Key: ").strip()
    
    # 5. OKX Passphrase
    print()
    print("5️⃣ OKX Passphrase:")
    print("   ⚠️ Đây là passphrase BẠN TỰ TẠO khi tạo API key (KHÔNG phải mật khẩu đăng nhập OKX)")
    if current_okx_passphrase:
        print(f"   (Hiện tại: {'*' * len(current_okx_passphrase)})")
        use_current = input("   Dùng giá trị hiện tại? (y/n) [n]: ").strip().lower()
        if use_current == 'y':
            okx_passphrase = current_okx_passphrase
        else:
            okx_passphrase = input("   Nhập OKX Passphrase: ").strip()
    else:
        okx_passphrase = input("   Nhập OKX Passphrase: ").strip()
    
    # 5.5. Simulated Trading (Demo Account)
    print()
    print("5️⃣.5 OKX Simulated Trading (Demo Account):")
    print("   💡 OKX hỗ trợ Simulated Trading qua header x-simulated-trading: 1")
    print("   - Nếu bạn có Demo Account và muốn dùng tiền ảo → Chọn 'y'")
    print("   - Nếu bạn muốn dùng tiền thật → Chọn 'n'")
    if current_okx_simulated:
        print(f"   (Hiện tại: {current_okx_simulated})")
    use_simulated = input("   Dùng Simulated Trading (Demo Account)? (y/n) [n]: ").strip().lower() or "n"
    okx_use_simulated = "1" if use_simulated == "y" else "0"
    
    # 6. Trading Configuration (optional)
    print()
    print("6️⃣ Cấu hình Trading (có thể bỏ qua, dùng giá trị mặc định):")
    
    symbol = input("   Symbol (BTCUSDT, ETHUSDT...) [BTCUSDT]: ").strip().upper() or "BTCUSDT"
    interval = input("   Interval (1m, 5m, 15m, 1h, 1d) [5m]: ").strip() or "5m"
    strategy = input("   Strategy (regime_specific, sma_ema...) [regime_specific]: ").strip() or "regime_specific"
    
    # Risk management
    print()
    print("7️⃣ Risk Management (có thể bỏ qua, dùng giá trị mặc định):")
    risk_per_trade = input("   Risk per trade (0.1 = 10%) [0.1]: ").strip() or "0.1"
    sl_pct = input("   Stop Loss % (0.02 = 2%) [0.02]: ").strip() or "0.02"
    tp_pct = input("   Take Profit % (0.04 = 4%) [0.04]: ").strip() or "0.04"
    
    # Validate
    if not okx_key or not okx_secret or not okx_passphrase:
        print()
        print("❌ Lỗi: Bạn phải điền đầy đủ OKX_API_KEY, OKX_API_SECRET và OKX_PASSPHRASE!")
        return False
    
    # Đọc file .env hiện tại (nếu có)
    env_content = ""
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
    
    # Tạo nội dung .env mới
    new_env_lines = []
    
    # Parse các dòng hiện tại và cập nhật
    lines = env_content.split('\n') if env_content else []
    found_sections = {
        'EXCHANGE': False,
        'MODE': False,
        'OKX_API_KEY': False,
        'OKX_API_SECRET': False,
        'OKX_PASSPHRASE': False,
        'OKX_USE_SIMULATED_TRADING': False,
        'SYMBOL': False,
        'INTERVAL': False,
        'STRATEGY': False,
        'RISK_PER_TRADE': False,
        'SL_PCT': False,
        'TP_PCT': False,
    }
    
    # Xử lý các dòng hiện có
    for line in lines:
        stripped = line.strip()
        
        # Bỏ qua comment và dòng trống
        if not stripped or stripped.startswith('#'):
            new_env_lines.append(line)
            continue
        
        # Cập nhật các giá trị
        if stripped.startswith('EXCHANGE='):
            new_env_lines.append(f"EXCHANGE={exchange}")
            found_sections['EXCHANGE'] = True
        elif stripped.startswith('MODE='):
            new_env_lines.append(f"MODE={mode}")
            found_sections['MODE'] = True
        elif stripped.startswith('OKX_API_KEY='):
            new_env_lines.append(f"OKX_API_KEY={okx_key}")
            found_sections['OKX_API_KEY'] = True
        elif stripped.startswith('OKX_API_SECRET='):
            new_env_lines.append(f"OKX_API_SECRET={okx_secret}")
            found_sections['OKX_API_SECRET'] = True
        elif stripped.startswith('OKX_PASSPHRASE='):
            new_env_lines.append(f"OKX_PASSPHRASE={okx_passphrase}")
            found_sections['OKX_PASSPHRASE'] = True
        elif stripped.startswith('OKX_USE_SIMULATED_TRADING='):
            new_env_lines.append(f"OKX_USE_SIMULATED_TRADING={okx_use_simulated}")
            found_sections['OKX_USE_SIMULATED_TRADING'] = True
        elif stripped.startswith('SYMBOL='):
            new_env_lines.append(f"SYMBOL={symbol}")
            found_sections['SYMBOL'] = True
        elif stripped.startswith('INTERVAL='):
            new_env_lines.append(f"INTERVAL={interval}")
            found_sections['INTERVAL'] = True
        elif stripped.startswith('STRATEGY='):
            new_env_lines.append(f"STRATEGY={strategy}")
            found_sections['STRATEGY'] = True
        elif stripped.startswith('RISK_PER_TRADE='):
            new_env_lines.append(f"RISK_PER_TRADE={risk_per_trade}")
            found_sections['RISK_PER_TRADE'] = True
        elif stripped.startswith('SL_PCT='):
            new_env_lines.append(f"SL_PCT={sl_pct}")
            found_sections['SL_PCT'] = True
        elif stripped.startswith('TP_PCT='):
            new_env_lines.append(f"TP_PCT={tp_pct}")
            found_sections['TP_PCT'] = True
        else:
            # Giữ nguyên các dòng khác
            new_env_lines.append(line)
    
    # Thêm các giá trị mới nếu chưa có
    if not found_sections['EXCHANGE']:
        new_env_lines.append(f"\n# Exchange Configuration\nEXCHANGE={exchange}")
    if not found_sections['MODE']:
        new_env_lines.append(f"MODE={mode}")
    
    if not found_sections['OKX_API_KEY']:
        new_env_lines.append(f"\n# OKX API Credentials\nOKX_API_KEY={okx_key}")
    if not found_sections['OKX_API_SECRET']:
        new_env_lines.append(f"OKX_API_SECRET={okx_secret}")
    if not found_sections['OKX_PASSPHRASE']:
        new_env_lines.append(f"OKX_PASSPHRASE={okx_passphrase}")
    if not found_sections['OKX_USE_SIMULATED_TRADING']:
        new_env_lines.append(f"OKX_USE_SIMULATED_TRADING={okx_use_simulated}")
    
    if not found_sections['SYMBOL']:
        new_env_lines.append(f"\n# Trading Configuration\nSYMBOL={symbol}")
    if not found_sections['INTERVAL']:
        new_env_lines.append(f"INTERVAL={interval}")
    if not found_sections['STRATEGY']:
        new_env_lines.append(f"STRATEGY={strategy}")
    
    if not found_sections['RISK_PER_TRADE']:
        new_env_lines.append(f"\n# Risk Management\nRISK_PER_TRADE={risk_per_trade}")
    if not found_sections['SL_PCT']:
        new_env_lines.append(f"SL_PCT={sl_pct}")
    if not found_sections['TP_PCT']:
        new_env_lines.append(f"TP_PCT={tp_pct}")
    
    # Thêm các cấu hình mặc định khác nếu chưa có
    if 'STRATEGY_PARAMS' not in env_content:
        new_env_lines.append(f'STRATEGY_PARAMS={{"use_regime_specific":true,"proba_threshold":0.45,"use_sequence_features":true}}')
    
    if 'HISTORY_LIMIT' not in env_content:
        new_env_lines.append(f"\n# Bot Settings\nHISTORY_LIMIT=200")
    if 'COOL_DOWN_SEC' not in env_content:
        new_env_lines.append(f"COOL_DOWN_SEC=60")
    if 'CHECK_INTERVAL_SEC' not in env_content:
        new_env_lines.append(f"CHECK_INTERVAL_SEC=30")
    
    # Ghi file .env
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_env_lines))
        
        print()
        print("=" * 60)
        print("✅ ĐÃ CẤU HÌNH THÀNH CÔNG!")
        print("=" * 60)
        print()
        print(f"📁 File .env đã được cập nhật tại: {env_path.absolute()}")
        print()
        print("📋 Tóm tắt cấu hình:")
        print(f"   Exchange: {exchange}")
        print(f"   Mode: {mode}")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {interval}")
        print(f"   Strategy: {strategy}")
        print(f"   Risk per trade: {risk_per_trade} ({float(risk_per_trade)*100}%)")
        print(f"   Stop Loss: {sl_pct} ({float(sl_pct)*100}%)")
        print(f"   Take Profit: {tp_pct} ({float(tp_pct)*100}%)")
        print()
        print("🚀 Bạn có thể chạy bot bằng lệnh:")
        print("   python -m algo_trading.live.universal_bot")
        print()
        
        return True
        
    except Exception as e:
        print()
        print(f"❌ Lỗi khi ghi file .env: {e}")
        return False


if __name__ == "__main__":
    try:
        setup_okx_env()
    except KeyboardInterrupt:
        print("\n\n❌ Đã hủy bởi user")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
