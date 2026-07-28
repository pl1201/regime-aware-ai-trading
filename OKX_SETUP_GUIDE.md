# 🚀 HƯỚNG DẪN CẤU HÌNH BOT TRÊN OKX

## 📋 TỔNG QUAN

Hướng dẫn này giúp bạn cấu hình bot để chạy trên OKX và **xem lệnh trực tiếp trên OKX**.

---

## 1️⃣ TẠO OKX API KEY

### Bước 1: Đăng nhập OKX
1. Truy cập: https://www.okx.com
2. Đăng nhập tài khoản của bạn

### Bước 2: Tạo API Key
1. Vào **User Center** (góc trên bên phải)
2. Chọn **API** → **Create API Key**
3. Điền thông tin:
   - **API Key Name**: `Trading Bot` (tên tùy ý)
   - **Passphrase**: **Tạo passphrase mới** (⚠️ BẠN TỰ NGHĨ RA, KHÔNG PHẢI MẬT KHẨU ĐĂNG NHẬP!)
     - Ví dụ: `MyBot2024!` hoặc `TradingBot123`
     - ⚠️ **LƯU LẠI CẨN THẬN** - không thể lấy lại nếu quên!
   - **Permissions**: Chọn:
     - ✅ **Read** (đọc dữ liệu)
     - ✅ **Trade** (giao dịch)
     - ❌ **Withdraw** (KHÔNG chọn, tránh rủi ro)

4. **Lưu 3 thông tin quan trọng**:
   - `OKX_API_KEY`: API Key (hiển thị ngay)
   - `OKX_API_SECRET`: Secret Key (chỉ hiển thị 1 lần, copy ngay!)
   - `OKX_PASSPHRASE`: **Passphrase bạn vừa tự tạo** (KHÔNG PHẢI mật khẩu đăng nhập OKX!)
     - Đây là chuỗi mật khẩu BẠN TỰ NGHĨ RA khi tạo API key
     - Dùng để xác thực API requests, không phải để đăng nhập website

### ⚠️ QUAN TRỌNG: Kiểm tra quyền truy cập

Sau khi tạo API key, bạn sẽ thấy thông tin như:
```
apikey = 
secretkey = 
IP = ""
Tên mã API = "bot1"
Quyền = "Đọc/Giao dịch"
```

**✅ NẾU CÓ QUYỀN "Đọc/Giao dịch" → ĐÃ ĐỦ!**

Bot cần **2 quyền**:
- ✅ **Đọc** (Read) - để lấy giá, dữ liệu klines, số dư
- ✅ **Giao dịch** (Trade) - để đặt lệnh mua/bán

**🎉 Nếu bạn thấy "Quyền = Đọc/Giao dịch" → API key đã đủ quyền!**

### Cách sửa: Thêm quyền "Giao dịch"

**Option 1: Edit API Key (nếu OKX cho phép)**
1. Vào **User Center** → **API**
2. Tìm API key vừa tạo
3. Click **Edit** hoặc **Manage**
4. Thêm quyền **Trade** (Giao dịch)
5. Lưu lại

**Option 2: Tạo lại API Key (nếu không edit được)**
1. Xóa API key cũ (hoặc giữ lại nếu muốn)
2. Tạo API key mới với **cả 2 quyền**:
   - ✅ **Read** (Đọc)
   - ✅ **Trade** (Giao dịch)
3. Lưu lại 3 thông tin mới

⚠️ **CẢNH BÁO**: 
- Secret Key và Passphrase **KHÔNG THỂ LẤY LẠI** nếu mất
- Lưu backup ở nơi an toàn
- Không chia sẻ với ai

---

## 2️⃣ CẤU HÌNH .env FILE

Tạo hoặc cập nhật file `.env` trong thư mục gốc của project:

```env
# ============================================
# EXCHANGE CONFIGURATION
# ============================================
EXCHANGE=okx
MODE=live

# ============================================
# OKX API CREDENTIALS
# ============================================
OKX_API_KEY=your_okx_api_key_here
OKX_API_SECRET=your_okx_secret_key_here
OKX_PASSPHRASE=your_okx_passphrase_here

# ============================================
# TRADING CONFIGURATION
# ============================================
SYMBOL=BTCUSDT
INTERVAL=5m
STRATEGY=regime_specific

# Strategy parameters (JSON format)
STRATEGY_PARAMS={"use_regime_specific":true,"proba_threshold":0.45,"use_sequence_features":true}

# ============================================
# RISK MANAGEMENT
# ============================================
RISK_PER_TRADE=0.1
SL_PCT=0.02
TP_PCT=0.04

# ============================================
# BOT SETTINGS
# ============================================
HISTORY_LIMIT=200
COOL_DOWN_SEC=60
CHECK_INTERVAL_SEC=30
```

### Giải thích các tham số:

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `EXCHANGE` | `okx` | Chọn exchange (okx hoặc binance) |
| `MODE` | `live` | `live` = gửi lệnh thật, `paper` = mô phỏng |
| `OKX_API_KEY` | `your_key` | API Key từ OKX |
| `OKX_API_SECRET` | `your_secret` | Secret Key từ OKX |
| `OKX_PASSPHRASE` | `your_passphrase` | Passphrase bạn tạo |
| `SYMBOL` | `BTCUSDT` | Cặp giao dịch |
| `INTERVAL` | `5m` | Timeframe (1m, 5m, 15m, 1h, 1d) |
| `STRATEGY` | `regime_specific` | Strategy name |
| `RISK_PER_TRADE` | `0.1` | Rủi ro mỗi lệnh (10% số dư) |
| `SL_PCT` | `0.02` | Stop Loss (2%) |
| `TP_PCT` | `0.04` | Take Profit (4%) |

---

## 3️⃣ CHẠY BOT

### Cách 1: Chạy trực tiếp
```bash
python -m algo_trading.live.universal_bot
```

### Cách 2: Chạy với Telegram bot
```bash
python start_trading_bot.py
```

### Cách 3: Chạy như service (Windows)
```bash
python run.bat
```

---

## 4️⃣ XEM LỆNH TRỰC TIẾP TRÊN OKX

### 4.1. Xem trên OKX Web

1. **Đăng nhập OKX**: https://www.okx.com
2. Vào **Trading** → **Spot Trading**
3. Xem tab **Orders**:
   - **Open Orders**: Lệnh đang chờ khớp
   - **Order History**: Lịch sử lệnh đã khớp/hủy
4. Filter theo:
   - **Symbol**: BTC-USDT
   - **Time**: Chọn khoảng thời gian bot chạy
   - **Type**: Market, Limit

### 4.2. Xem trên OKX App

1. Mở app OKX
2. Vào **Trading** → **Spot**
3. Tab **Orders** → Xem tất cả lệnh
4. Tab **History** → Xem lịch sử giao dịch

### 4.3. Xem Positions (Vị thế)

1. **OKX Web**: **Assets** → **Funding Account** → **Spot Holdings**
2. **OKX App**: **Assets** → **Spot**

Bạn sẽ thấy:
- Số dư BTC (nếu bot mua BTC)
- Số dư USDT (nếu bot bán BTC)
- Giá trị USD
- P&L (nếu có)

---

## 5️⃣ MONITOR BOT

### 5.1. Logs

Bot sẽ ghi log vào file `live_trading.log`:

```bash
# Xem log real-time
tail -f live_trading.log

# Windows PowerShell
Get-Content live_trading.log -Wait -Tail 50
```

### 5.2. Telegram Notifications

Nếu đã setup Telegram bot, bạn sẽ nhận thông báo:
- 🟢 Signal mới
- 🚀 Đã vào lệnh
- 🚪 Đã thoát lệnh
- 🛑 Stop Loss / Take Profit hit

### 5.3. Check Status qua Telegram

Gửi lệnh `/status` trong Telegram bot để xem:
- Giá hiện tại
- Signal mới nhất
- Trạng thái holding
- Entry price
- P&L

---

## 6️⃣ PAPER TRADING (TEST KHÔNG DÙNG TIỀN THẬT)

Nếu muốn test trước khi dùng tiền thật:

### Cách 1: Dùng MODE=paper
```env
MODE=paper
EXCHANGE=okx
```

Bot sẽ **không gửi lệnh thật**, chỉ log ra console.

### Cách 2: Dùng OKX Simulated Trading (Demo Account) ⭐ **KHUYẾN NGHỊ**

**✅ ĐÂY LÀ CÁCH TỐT NHẤT ĐỂ TEST VỚI OKX!**

OKX hỗ trợ **Simulated Trading** qua header `x-simulated-trading: 1`. Đây là cách chính thức để dùng Demo Account.

#### Cách hoạt động:

1. **Bạn có thể dùng BẤT KỲ API key nào** (Live Account hoặc Demo Account)
2. **Thêm header `x-simulated-trading: 1`** vào mọi request
3. OKX sẽ tự động chuyển tất cả lệnh vào **Simulated Trading Account** (tiền ảo)
4. **KHÔNG ảnh hưởng tiền thật**, dù bạn dùng API key từ Live Account

#### Bước 1: Cấu hình trong `.env`

**Cách 1: Dùng script tự động**
```bash
python setup_okx_env.py
```
Khi script hỏi "Dùng Simulated Trading (Demo Account)?", chọn **`y`**.

**Cách 2: Sửa thủ công file `.env`**

Thêm hoặc cập nhật:
```env
# OKX API Credentials (có thể dùng API key từ Live Account hoặc Demo Account)
OKX_API_KEY=your_api_key_here
OKX_API_SECRET=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here

# ⭐ QUAN TRỌNG: Bật Simulated Trading
OKX_USE_SIMULATED_TRADING=1

# VẪN PHẢI SET MODE=live (vì bot vẫn gửi lệnh lên OKX API)
MODE=live
EXCHANGE=okx
```

**⚠️ Lưu ý:**
- `OKX_USE_SIMULATED_TRADING=1` → Bot sẽ thêm header `x-simulated-trading: 1` vào mọi request
- `OKX_USE_SIMULATED_TRADING=0` hoặc không có → Bot gửi lệnh thật (tiền thật)
- Khi dùng Simulated Trading, **VẪN phải set `MODE=live`** (vì bot vẫn gửi lệnh lên OKX API)

#### Bước 2: Test kết nối

```bash
python test_okx_connection.py
```

Nếu thành công, bạn sẽ thấy:
```
✅ Kết nối OKX thành công!
🎮 Đang dùng OKX Simulated Trading (Demo Account) - Header x-simulated-trading: 1
📊 Symbol: BTC-USDT
💰 Giá hiện tại: 43250.5 USDT
```

#### Bước 3: Chạy bot

```bash
python -m algo_trading.live.universal_bot
```

Bot sẽ:
- Thêm header `x-simulated-trading: 1` vào mọi request
- Gửi lệnh vào Simulated Trading Account (tiền ảo)
- Bạn có thể thấy lệnh trên OKX → **Simulated Trading** (không phải Live Trading)

#### Ưu điểm của Simulated Trading:

✅ **Test thực tế với OKX API**  
✅ **Thấy lệnh thật trên OKX** (trong Simulated Trading)  
✅ **Biết được phí giao dịch**  
✅ **KHÔNG ảnh hưởng tiền thật** (dù dùng API key từ Live Account)  
✅ **Có thể test với số tiền lớn** (số dư ảo)  
✅ **Không cần Demo Account riêng** (có thể dùng API key từ Live Account)  

#### So sánh:

| Cách | OKX_USE_SIMULATED_TRADING | MODE | Thấy lệnh trên OKX? | Ảnh hưởng tiền thật? |
|------|---------------------------|------|---------------------|----------------------|
| **Paper (mô phỏng)** | Không cần | `paper` | ❌ Không | ❌ Không |
| **Simulated Trading** | `1` | `live` | ✅ Có (Simulated) | ❌ Không |
| **Live Trading** | `0` hoặc không có | `live` | ✅ Có (Live) | ✅ Có |

**Khuyến nghị:** Dùng `OKX_USE_SIMULATED_TRADING=1` để test an toàn!

---

## 7️⃣ TROUBLESHOOTING

### Lỗi: "OKX API error: Invalid API Key"
- ✅ Kiểm tra `OKX_API_KEY` trong `.env`
- ✅ Đảm bảo API key chưa bị xóa trên OKX

### Lỗi: "OKX API error: Invalid Passphrase"
- ✅ Kiểm tra `OKX_PASSPHRASE` trong `.env`
- ✅ Đảm bảo passphrase đúng (case-sensitive)

### Lỗi: "Insufficient balance"
- ✅ Kiểm tra số dư USDT trên OKX
- ✅ Giảm `RISK_PER_TRADE` nếu số dư nhỏ

### Lỗi: "Symbol not found"
- ✅ Kiểm tra `SYMBOL` format: `BTCUSDT` (không có dấu gạch)
- ✅ Bot sẽ tự convert sang `BTC-USDT` khi gọi OKX API

### Không thấy lệnh trên OKX
- ✅ Kiểm tra `MODE=live` (không phải `paper`)
- ✅ Kiểm tra log file `live_trading.log`
- ✅ Kiểm tra API permissions có **Trade** không

---

## 8️⃣ BEST PRACTICES

### 8.1. Bắt đầu với số tiền nhỏ
- Test với `RISK_PER_TRADE=0.05` (5%)
- Test với số tiền nhỏ trước khi scale up

### 8.2. Monitor kỹ trong giai đoạn đầu
- Xem logs mỗi ngày
- Check OKX orders thường xuyên
- So sánh với backtest results

### 8.3. Setup Risk Management
- Luôn set `SL_PCT` và `TP_PCT`
- Không trade quá nhiều % số dư
- Có kế hoạch dừng bot nếu thua lỗ

### 8.4. Backup API Keys
- Lưu API keys ở nơi an toàn
- Không commit `.env` vào git
- Dùng `.gitignore` để bỏ qua `.env`

---

## 9️⃣ TÓM TẮT CÁC BƯỚC

1. ✅ Tạo OKX API Key (lưu 3 thông tin: key, secret, passphrase)
2. ✅ Cấu hình `.env` với OKX credentials
3. ✅ Set `EXCHANGE=okx` và `MODE=live`
4. ✅ Chạy bot: `python -m algo_trading.live.universal_bot`
5. ✅ Xem lệnh trên OKX Web/App: **Trading** → **Spot** → **Orders**

---

## 🔟 LIÊN HỆ & HỖ TRỢ

Nếu gặp vấn đề:
1. Check logs: `live_trading.log`
2. Check OKX API status: https://www.okx.com/status
3. Xem tài liệu: `docs/OKX_INTEGRATION_PLAN.md`
4. Xem monitoring guide: `docs/OKX_MONITORING_GUIDE.md`

---

**🎉 Chúc bạn trading thành công!**
