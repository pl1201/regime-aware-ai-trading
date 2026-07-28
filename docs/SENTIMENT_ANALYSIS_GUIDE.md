# Hướng Dẫn Train AI Phân Tích Sentiment & Tích Hợp Vào Trading Bot

## Tổng Quan

Tài liệu này hướng dẫn chi tiết:
1. **Format dữ liệu** cần cho training sentiment model
2. **Quy trình train** AI phân tích sentiment
3. **Tích hợp sentiment** vào pipeline Regime Ensemble hiện tại

---

## 1. FORMAT DỮ LIỆU CHO TRAINING SENTIMENT

### 1.1. Cấu Trúc Dữ Liệu Chuẩn

Dữ liệu training cho sentiment analysis cần có **ít nhất** các cột sau:

| Cột | Kiểu | Bắt buộc | Mô tả |
|-----|------|----------|-------|
| `text` | string | ✅ | Nội dung văn bản (tin tức, tweet, post...) |
| `label` | int/string | ✅ | Nhãn sentiment: 0=negative, 1=neutral, 2=positive HOẶC -1/0/1 |
| `timestamp` | datetime | Khuyến nghị | Thời gian phát hành (để align với price data) |
| `source` | string | Tùy chọn | Nguồn: news, twitter, reddit, telegram... |
| `symbol` | string | Tùy chọn | Mã tài sản liên quan: BTCUSDT, ETH, VNM... |
| `title` | string | Tùy chọn | Tiêu đề (nếu có, có thể concat với text) |

### 1.2. Format CSV Mẫu

**File: `data/sentiment_train.csv`**

```csv
text,label,timestamp,source,symbol
"Bitcoin surges past $100k as institutional adoption grows",2,2024-01-15 10:30:00,news,BTCUSDT
"Market volatility increases amid Fed rate decision",1,2024-01-15 11:00:00,news,BTCUSDT
"Major exchange faces liquidity crisis, users withdraw funds",0,2024-01-15 12:00:00,twitter,BTCUSDT
"Ethereum upgrade successful, fees drop significantly",2,2024-01-15 14:00:00,reddit,ETHUSDT
"Crypto market consolidates after rally",1,2024-01-15 15:00:00,news,BTCUSDT
```

**Label encoding:**
- `0` hoặc `negative` hoặc `-1` = Tiêu cực (bearish)
- `1` hoặc `neutral` hoặc `0` = Trung tính
- `2` hoặc `positive` hoặc `1` = Tích cực (bullish)

### 1.3. Format JSON Mẫu (Cho dữ liệu phức tạp hơn)

**File: `data/sentiment_train.json`**

```json
[
  {
    "text": "Bitcoin surges past $100k as institutional adoption grows",
    "title": "BTC breaks key resistance",
    "label": 2,
    "label_text": "positive",
    "timestamp": "2024-01-15T10:30:00Z",
    "source": "news",
    "symbol": "BTCUSDT",
    "url": "https://...",
    "confidence": 0.95
  },
  {
    "text": "Major exchange faces liquidity crisis",
    "label": 0,
    "timestamp": "2024-01-15T12:00:00Z",
    "source": "twitter",
    "symbol": "BTCUSDT"
  }
]
```

### 1.4. Yêu Cầu Về Số Lượng Dữ Liệu

| Giai đoạn | Số lượng tối thiểu | Chất lượng |
|-----------|-------------------|------------|
| **Proof of concept** | 500-1,000 mẫu | Gán nhãn thủ công |
| **Fine-tune model** | 3,000-5,000 mẫu | Cân bằng 3 classes |
| **Production-ready** | 10,000+ mẫu | Đa nguồn, đa thời gian |

**Phân bố labels khuyến nghị:**
- Negative: 30-40%
- Neutral: 30-40%
- Positive: 30-40%

Tránh imbalance quá mạnh (ví dụ 80% neutral).

---

## 2. NGUỒN DỮ LIỆU

### 2.1. Dữ Liệu Công Khai (Miễn Phí)

| Nguồn | Ngôn ngữ | Format | Ghi chú |
|-------|----------|--------|---------|
| **Financial PhraseBank** | English | CSV | ~5,000 câu tin tức tài chính, 3 labels |
| **FiQA** | English | JSON | Financial Q&A + sentiment |
| **StockTwits** | English | API | Real-time sentiment |
| **Reddit (r/cryptocurrency, WSB)** | English | API/Scrape | Sentiment cộng đồng |
| **Twitter/X** | Đa ngôn ngữ | API | Cần API key |

### 2.2. API Trả Tin Tức Có Sẵn

| API | Endpoint | Cost |
|-----|----------|------|
| **Finnhub** | /news | Free tier |
| **Alpha Vantage** | NEWS_SENTIMENT | Free tier |
| **NewsAPI** | /everything | Free tier (limited) |
| **CryptoPanic** | /news | Free (crypto focus) |

### 2.3. Thu Thập Từ Nguồn Riêng

**Crypto (phù hợp với bot hiện tại):**
- Tin tức: CoinDesk, Cointelegraph, Bitcoin Magazine
- Social: Twitter hashtags (#Bitcoin #BTC #crypto), Reddit r/cryptocurrency
- Telegram: Các nhóm crypto, channel tin tức

**Chứng khoán VN (nếu mở rộng):**
- CafeF, VnExpress Kinh doanh, VietStock
- SSI, VNDirect research

---

## 3. QUY TRÌNH TRAINING SENTIMENT MODEL

### 3.1. Pipeline Tổng Quan

```
[Bước 1] Thu thập dữ liệu (crawl/API)
    ↓
[Bước 2] Tiền xử lý (clean, normalize, tokenize)
    ↓
[Bước 3] Gán nhãn (thủ công / semi-auto / active learning)
    ↓
[Bước 4] Train/Val/Test split (theo thời gian!)
    ↓
[Bước 5] Fine-tune model (FinBERT/PhoBERT)
    ↓
[Bước 6] Đánh giá & lưu model
    ↓
[Bước 7] Tích hợp vào pipeline inference
```

### 3.2. Lựa Chọn Model

| Model | Ngôn ngữ | Use case | Ghi chú |
|-------|----------|----------|---------|
| **FinBERT** | English | Tin tức tài chính | Pre-trained trên financial text, tốt nhất cho English |
| **PhoBERT** | Tiếng Việt | Tin VN, social VN | Tốt cho tiếng Việt |
| **BERT multilingual** | Đa ngôn ngữ | Mixed content | Khi có cả EN + VI |

**Khuyến nghị cho Crypto Bot (BTCUSDT):**
- Dùng **FinBERT** nếu tin tức chủ yếu tiếng Anh
- Dùng **PhoBERT** nếu có nhiều tin tiếng Việt

### 3.3. Output Model

Sau training, model trả về:
- **Logits** hoặc **probabilities** cho 3 classes: [P(neg), P(neutral), P(pos)]
- **Sentiment score** chuẩn hóa: `score = P(pos) - P(neg)` → range [-1, 1]
- **Label dự đoán**: argmax của probabilities

---

## 4. TÍCH HỢP SENTIMENT VÀO REGIME ENSEMBLE

### 4.1. Sentiment Features Cần Thêm

Khi tích hợp vào pipeline train Regime Ensemble, cần thêm các **sentiment features**:

| Feature | Mô tả | Cách tính |
|---------|-------|-----------|
| `sentiment_score` | Score tổng hợp [-1, 1] | Mean sentiment của N tin gần nhất |
| `sentiment_score_1h` | Sentiment 1h gần nhất | Aggregate tin trong 1h |
| `sentiment_score_4h` | Sentiment 4h gần nhất | Aggregate tin trong 4h |
| `sentiment_score_1d` | Sentiment 24h | Aggregate tin trong 24h |
| `sentiment_volatility` | Độ phân tán sentiment | Std của sentiment scores |
| `sentiment_surprise` | Thay đổi đột ngột | sentiment_now - sentiment_ma_7d |
| `sentiment_positive_ratio` | Tỷ lệ tin tích cực | Count(positive) / total |
| `sentiment_negative_ratio` | Tỷ lệ tin tiêu cực | Count(negative) / total |

### 4.2. Cách Align Sentiment Với Price Data

**Vấn đề:** Tin tức có timestamp, price data có OHLCV theo timeframe (1h, 4h...).

**Giải pháp:**
1. Aggregate sentiment theo **time window** trùng với candle:
   - Candle 1h lúc 10:00 → lấy sentiment của tin từ 09:00-10:00
2. Forward-fill nếu không có tin trong window (dùng sentiment gần nhất)
3. Lưu sentiment theo `(timestamp, symbol)` để join với OHLCV

**Schema lưu sentiment aggregated:**

```csv
timestamp,symbol,sentiment_score,sentiment_positive_ratio,sentiment_negative_ratio,sentiment_volatility,count_news
2024-01-15 10:00:00,BTCUSDT,0.35,0.45,0.20,0.25,12
2024-01-15 11:00:00,BTCUSDT,-0.12,0.30,0.45,0.31,8
```

### 4.3. Flow Tích Hợp

```
[Live/Backtest]
    │
    ├─→ Load OHLCV data (1h)
    │
    ├─→ Load/Compute sentiment features (theo timestamp)
    │       └─→ sentiment_score_1h, sentiment_score_4h, sentiment_volatility...
    │
    ├─→ Merge với indicators (RSI, MACD, regime...)
    │
    └─→ Feed vào Regime Ensemble Model → Signal
```

---

## 5. CÁC BƯỚC CỤ THỂ BẠN NÊN LÀM

### Phase 1: Chuẩn Bị Data (1-2 tuần)

1. **Thu thập dữ liệu:**
   - Dùng Finnhub/Alpha Vantage/CryptoPanic API lấy tin crypto
   - Hoặc crawl từ 2-3 nguồn tin (CoinDesk, Cointelegraph)
   - Lưu: `text`, `timestamp`, `source`, `symbol`

2. **Gán nhãn:**
   - Option A: Gán thủ công 500-1000 mẫu (dùng tool Label Studio, hoặc Google Sheet)
   - Option B: Dùng FinBERT pre-trained gán nhãn trước, sau đó người kiểm tra/sửa (semi-supervised)
   - Option C: Gán nhãn từ giá (tin trước khi giá tăng → positive, giá giảm → negative) - **cẩn thận look-ahead bias!**

3. **Chuẩn hóa format:**
   - Export CSV/JSON đúng schema ở mục 1
   - Train/Val/Test split **theo thời gian** (không shuffle random!) ví dụ: train=2022-2023, val=2024 Q1, test=2024 Q2

### Phase 2: Train Model (3-5 ngày)

4. **Setup environment:**
   ```bash
   pip install transformers torch pandas scikit-learn
   ```

5. **Chạy training script** (xem `scripts/train_sentiment_model.py`):
   - Load FinBERT/PhoBERT
   - Fine-tune trên data của bạn
   - Save model + tokenizer

6. **Đánh giá:**
   - Accuracy, F1-macro trên test set
   - Kiểm tra confusion matrix (có bias class nào không)

### Phase 3: Tích Hợp (1 tuần)

7. **Tạo sentiment aggregator:**
   - Script đọc tin mới → inference sentiment → aggregate theo time window
   - Lưu sentiment features vào DB hoặc CSV

8. **Thêm sentiment vào feature matrix:**
   - Trong `train_regime_ensemble_models_advanced.py`: thêm bước load sentiment, merge với df theo timestamp
   - Trong `RegimeEnsembleStrategy`: thêm sentiment features khi build feature matrix

9. **Retrain Regime Ensemble** với sentiment features mới

10. **Backtest** so sánh: có sentiment vs không có sentiment

---

## 6. LƯU Ý QUAN TRỌNG

### 6.1. Look-Ahead Bias
- **Không** dùng tin tức tương lai khi train/predict
- Khi backtest: sentiment của tin lúc 10:30 chỉ được dùng cho candle từ 11:00 trở đi

### 6.2. Data Leakage
- Train/Val/Test split phải theo thời gian
- Không dùng scaler fit trên toàn bộ data (chỉ fit trên train)

### 6.3. Sentiment Có Thể Không Predictive
- Một số nghiên cứu chỉ ra sentiment lag behind price (giá đi trước tin)
- Nên **validate** bằng backtest: strategy có sentiment có tốt hơn không?
- Có thể sentiment chỉ hữu ích trong một số regime (ví dụ: volatile)

### 6.4. Domain Shift
- Model train trên tin 2022-2023 có thể kém trên tin 2024 (từ vựng, phong cách thay đổi)
- Định kỳ thu thập data mới, đánh giá lại, retrain nếu cần

---

## 7. TÍCH HỢP VÀO TRAINING REGIME ENSEMBLE

### 7.1. Chuẩn Bị Sentiment Data Đã Aggregate

Trước khi train Regime Ensemble, cần có file `data/sentiment_aggregated.csv`:

```csv
timestamp,symbol,sentiment_mean_1h,sentiment_mean_4h,sentiment_mean_1d,sentiment_volatility_1d
2024-01-15 10:00:00,BTCUSDT,0.35,-0.12,0.20,0.25
2024-01-15 11:00:00,BTCUSDT,-0.12,0.10,0.15,0.31
```

Cách tạo file này:
- Dùng `SentimentAggregator.aggregate_from_raw_news()` từ tin tức raw
- Hoặc dùng API (Finnhub, CryptoPanic) + SentimentModel để tính và lưu

### 7.2. Thêm Sentiment Vào Feature Matrix

Trong `build_feature_matrix_enhanced()` hoặc tương đương, thêm:

```python
# Load sentiment data (nếu có)
sentiment_path = Path("data/sentiment_aggregated.csv")
if sentiment_path.exists():
    sent_df = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
    sent_df = sent_df[sent_df["symbol"] == symbol]  # Filter theo symbol
    # Align với df theo index
    for col in ["sentiment_mean_1h", "sentiment_mean_4h", "sentiment_mean_1d"]:
        if col in sent_df.columns:
            feats[f"sent_{col.replace('sentiment_', '')}"] = sent_df[col].reindex(df.index).ffill().bfill()
```

### 7.3. Feature Names Cho Inference

Khi train với sentiment, cần lưu `feature_names` bao gồm cả sentiment. Khi inference (live), phải có sentiment data real-time hoặc forward-fill từ lần update gần nhất.

---

## 8. FILE STRUCTURE ĐỀ XUẤT

```
Bot_Trading/
├── data/
│   ├── sentiment/
│   │   ├── train.csv          # Dữ liệu training đã gán nhãn
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── raw/               # Tin thô chưa gán nhãn
│   │       └── news_*.json
│   └── sentiment_aggregated.csv  # Sentiment theo (timestamp, symbol)
├── models/
│   └── sentiment/
│       ├── finbert_finetuned/  # Model + tokenizer
│       └── config.json
├── algo_trading/
│   └── sentiment/
│       ├── __init__.py
│       ├── model.py           # Load model, inference
│       ├── aggregator.py      # Aggregate sentiment theo time window
│       └── data_loader.py     # Load sentiment data
├── scripts/
│   ├── train_sentiment_model.py
│   ├── collect_news.py        # Thu thập tin
│   └── label_sentiment.py     # Tool gán nhãn (optional)
```

---

## 9. TÀI LIỆU THAM KHẢO

- **FinBERT**: Araci (2019) - Financial Sentiment Analysis with Pre-trained Language Models
- **Financial PhraseBank**: Malo et al. - Dataset cho financial sentiment
- **PhoBERT**: Nguyễn et al. - PhoBERT: Pre-trained language models for Vietnamese
- Document nội bộ: `phan_tich_tam_ly_thi_truong_da_phuong_tien.doc`
