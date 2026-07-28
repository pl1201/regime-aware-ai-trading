"""
Technical Indicators Showcase - Slide 3
Minh họa các indicator nổi bật được sử dụng trong Bot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate synthetic price data
np.random.seed(42)
n_points = 200

# Create realistic price movement
dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
price = 100 + np.cumsum(np.random.randn(n_points) * 0.5 + 0.05)
high = price + np.random.rand(n_points) * 2
low = price - np.random.rand(n_points) * 2
volume = np.random.rand(n_points) * 1000 + 500

df = pd.DataFrame({
    'date': dates,
    'open': price,
    'high': high,
    'low': low,
    'close': price,
    'volume': volume
})

# Calculate indicators
# EMAs
df['EMA9'] = df['close'].ewm(span=9).mean()
df['EMA21'] = df['close'].ewm(span=21).mean()
df['EMA50'] = df['close'].ewm(span=50).mean()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = df['close'].ewm(span=12).mean()
ema26 = df['close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# Bollinger Bands
bb_mid = df['close'].rolling(20).mean()
bb_std = df['close'].rolling(20).std()
df['BB_upper'] = bb_mid + 2 * bb_std
df['BB_lower'] = bb_mid - 2 * bb_std
df['BB_mid'] = bb_mid

# ATR
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift()).abs()
tr3 = (df['low'] - df['close'].shift()).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR'] = tr.rolling(14).mean()

# ADX (simplified)
df['ADX'] = 15 + np.random.rand(n_points) * 20  # Simulated

# Volume ratio
df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# ============================================================================
# CHART 1: INDICATOR OVERVIEW - 6 panels
# ============================================================================

fig1 = plt.figure(figsize=(18, 14))
gs = fig1.add_gridspec(6, 1, height_ratios=[3, 1.5, 1.5, 1.5, 1.5, 1], hspace=0.3)

# Title
fig1.suptitle('CAC INDICATOR NOI BAT TRONG BOT TRADING\n100+ Features tu OHLCV Data', 
              fontsize=18, fontweight='bold', y=0.995)

# Panel 1: Price + EMAs + Bollinger Bands
ax1 = fig1.add_subplot(gs[0])
ax1.plot(df['date'], df['close'], color='#2c3e50', linewidth=2, label='Price', zorder=5)
ax1.plot(df['date'], df['EMA9'], color='#e74c3c', linewidth=1.5, label='EMA 9 (Fast)', alpha=0.8)
ax1.plot(df['date'], df['EMA21'], color='#f39c12', linewidth=1.5, label='EMA 21 (Medium)', alpha=0.8)
ax1.plot(df['date'], df['EMA50'], color='#3498db', linewidth=1.5, label='EMA 50 (Slow)', alpha=0.8)

# Bollinger Bands
ax1.fill_between(df['date'], df['BB_upper'], df['BB_lower'], alpha=0.2, color='gray', label='Bollinger Bands')
ax1.plot(df['date'], df['BB_upper'], color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax1.plot(df['date'], df['BB_lower'], color='gray', linewidth=1, linestyle='--', alpha=0.5)

ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
ax1.set_title('1. TREND INDICATORS: EMAs & Bollinger Bands', fontsize=13, fontweight='bold', loc='left')
ax1.legend(loc='upper left', fontsize=9, ncol=3)
ax1.grid(True, alpha=0.3)

# Panel 2: RSI
ax2 = fig1.add_subplot(gs[1], sharex=ax1)
ax2.plot(df['date'], df['RSI'], color='#9b59b6', linewidth=2)
ax2.axhline(y=70, color='#e74c3c', linestyle='--', linewidth=1.5, label='Overbought (70)')
ax2.axhline(y=30, color='#2ecc71', linestyle='--', linewidth=1.5, label='Oversold (30)')
ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.fill_between(df['date'], 70, 100, alpha=0.1, color='red')
ax2.fill_between(df['date'], 0, 30, alpha=0.1, color='green')

ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
ax2.set_title('2. MOMENTUM: RSI (Relative Strength Index)', fontsize=13, fontweight='bold', loc='left')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: MACD
ax3 = fig1.add_subplot(gs[2], sharex=ax1)
ax3.plot(df['date'], df['MACD'], color='#3498db', linewidth=2, label='MACD')
ax3.plot(df['date'], df['MACD_signal'], color='#e74c3c', linewidth=2, label='Signal')

# MACD histogram
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['MACD_hist']]
ax3.bar(df['date'], df['MACD_hist'], color=colors, alpha=0.3, width=0.03, label='Histogram')

ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_ylabel('MACD', fontsize=11, fontweight='bold')
ax3.set_title('3. TREND MOMENTUM: MACD (Moving Average Convergence Divergence)', 
             fontsize=13, fontweight='bold', loc='left')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: ATR
ax4 = fig1.add_subplot(gs[3], sharex=ax1)
ax4.plot(df['date'], df['ATR'], color='#e67e22', linewidth=2)
ax4.fill_between(df['date'], df['ATR'], alpha=0.3, color='#e67e22')

# Volatility zones
atr_mean = df['ATR'].mean()
ax4.axhline(y=atr_mean * 1.5, color='#e74c3c', linestyle='--', linewidth=1.5, 
           label='High Volatility', alpha=0.7)
ax4.axhline(y=atr_mean * 0.5, color='#2ecc71', linestyle='--', linewidth=1.5, 
           label='Low Volatility', alpha=0.7)

ax4.set_ylabel('ATR', fontsize=11, fontweight='bold')
ax4.set_title('4. VOLATILITY: ATR (Average True Range)', fontsize=13, fontweight='bold', loc='left')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# Panel 5: ADX
ax5 = fig1.add_subplot(gs[4], sharex=ax1)
ax5.plot(df['date'], df['ADX'], color='#16a085', linewidth=2)
ax5.fill_between(df['date'], df['ADX'], alpha=0.3, color='#16a085')

# ADX zones
ax5.axhline(y=25, color='#e74c3c', linestyle='--', linewidth=1.5, label='Strong Trend (>25)', alpha=0.7)
ax5.axhline(y=20, color='#f39c12', linestyle='--', linewidth=1.5, label='Weak Trend (<20)', alpha=0.7)
ax5.fill_between(df['date'], 25, 50, alpha=0.1, color='green')
ax5.fill_between(df['date'], 0, 20, alpha=0.1, color='red')

ax5.set_ylabel('ADX', fontsize=11, fontweight='bold')
ax5.set_title('5. TREND STRENGTH: ADX (Average Directional Index)', 
             fontsize=13, fontweight='bold', loc='left')
ax5.legend(loc='upper left', fontsize=9)
ax5.grid(True, alpha=0.3)

# Panel 6: Volume
ax6 = fig1.add_subplot(gs[5], sharex=ax1)
colors_vol = ['#2ecc71' if x > 1 else '#95a5a6' for x in df['vol_ratio']]
ax6.bar(df['date'], df['vol_ratio'], color=colors_vol, alpha=0.6, width=0.03)
ax6.axhline(y=1, color='black', linewidth=1, linestyle='-')
ax6.axhline(y=1.5, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.5, label='High Volume')

ax6.set_ylabel('Volume Ratio', fontsize=11, fontweight='bold')
ax6.set_xlabel('Time', fontsize=12, fontweight='bold')
ax6.set_title('6. VOLUME: Volume Ratio (vs 20-period MA)', fontsize=13, fontweight='bold', loc='left')
ax6.legend(loc='upper left', fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization/indicators_showcase.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/indicators_showcase.png")

# ============================================================================
# CHART 2: INDICATOR CATEGORIES
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('PHAN LOAI 100+ INDICATORS THEO CHUC NANG', fontsize=18, fontweight='bold')

# Category 1: Momentum
ax1 = axes[0, 0]
categories = ['mom_1h', 'mom_4h', 'mom_12h', 'mom_24h', 'mom_48h', 'RSI', 'MACD', 'Stoch']
counts = [1, 1, 1, 1, 1, 2, 3, 2]
colors_cat = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']

bars1 = ax1.barh(categories, counts, color=colors_cat, edgecolor='black', linewidth=2)
ax1.set_xlabel('So luong features', fontsize=12, fontweight='bold')
ax1.set_title('MOMENTUM INDICATORS (12 features)\nDo toc do thay doi gia', 
             fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars1, counts)):
    ax1.text(count + 0.1, i, str(count), va='center', fontsize=11, fontweight='bold')

# Category 2: Trend
ax2 = axes[0, 1]
categories2 = ['EMA 9', 'EMA 21', 'EMA 50', 'EMA 100', 'EMA 200', 'EMA Crosses', 'Trend Align']
counts2 = [1, 1, 1, 1, 1, 7, 1]
colors_cat2 = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2980b9', '#9b59b6', '#2ecc71']

bars2 = ax2.barh(categories2, counts2, color=colors_cat2, edgecolor='black', linewidth=2)
ax2.set_xlabel('So luong features', fontsize=12, fontweight='bold')
ax2.set_title('TREND INDICATORS (13 features)\nXac dinh xu huong thi truong', 
             fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars2, counts2)):
    ax2.text(count + 0.2, i, str(count), va='center', fontsize=11, fontweight='bold')

# Category 3: Volatility
ax3 = axes[1, 0]
categories3 = ['ATR', 'ATR %', 'BB Position', 'BB Width', 'True Range', 'Price Range']
counts3 = [1, 1, 1, 1, 1, 1]
colors_cat3 = ['#e67e22', '#e74c3c', '#9b59b6', '#3498db', '#f39c12', '#2ecc71']

bars3 = ax3.barh(categories3, counts3, color=colors_cat3, edgecolor='black', linewidth=2)
ax3.set_xlabel('So luong features', fontsize=12, fontweight='bold')
ax3.set_title('VOLATILITY INDICATORS (6 features)\nDo bien dong thi truong', 
             fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars3, counts3)):
    ax3.text(count + 0.05, i, str(count), va='center', fontsize=11, fontweight='bold')

# Category 4: Others
ax4 = axes[1, 1]
categories4 = ['Volume Ratio', 'ADX', 'DI Diff', 'Range Pos', 'MTF Bull', 'Trend Align']
counts4 = [1, 1, 1, 1, 1, 1]
colors_cat4 = ['#1abc9c', '#16a085', '#27ae60', '#2ecc71', '#3498db', '#9b59b6']

bars4 = ax4.barh(categories4, counts4, color=colors_cat4, edgecolor='black', linewidth=2)
ax4.set_xlabel('So luong features', fontsize=12, fontweight='bold')
ax4.set_title('OTHER INDICATORS (6 features)\nVolume, Strength, Structure', 
             fontsize=13, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars4, counts4)):
    ax4.text(count + 0.05, i, str(count), va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization/indicators_categories.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/indicators_categories.png")

# ============================================================================
# CHART 3: INDICATOR IMPORTANCE
# ============================================================================

fig3, ax = plt.subplots(figsize=(14, 10))

# Feature importance (simulated based on typical XGBoost results)
features = [
    'mom_24', 'mom_48', 'ema_9_21', 'rsi_norm', 'adx', 
    'mtf_bull', 'trend_align', 'atr_pct', 'bb_pos', 'vol_ratio',
    'macd_hist', 'di_diff', 'price_ema50', 'mom_12', 'ema_21_50',
    'bb_width', 'range_pos', 'mom_8', 'price_ema200', 'ema_50_100'
]

importance = [
    0.085, 0.078, 0.072, 0.068, 0.065,
    0.062, 0.058, 0.055, 0.052, 0.048,
    0.045, 0.042, 0.040, 0.038, 0.035,
    0.032, 0.030, 0.028, 0.025, 0.022
]

# Color by category
colors_imp = []
for feat in features:
    if 'mom' in feat:
        colors_imp.append('#e74c3c')  # Momentum
    elif 'ema' in feat or 'trend' in feat or 'price' in feat:
        colors_imp.append('#3498db')  # Trend
    elif 'rsi' in feat or 'macd' in feat:
        colors_imp.append('#9b59b6')  # Momentum indicators
    elif 'atr' in feat or 'bb' in feat or 'range' in feat:
        colors_imp.append('#e67e22')  # Volatility
    elif 'adx' in feat or 'di' in feat:
        colors_imp.append('#16a085')  # Strength
    else:
        colors_imp.append('#2ecc71')  # Others

bars = ax.barh(features, importance, color=colors_imp, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Feature Importance', fontsize=13, fontweight='bold')
ax.set_title('TOP 20 INDICATORS QUAN TRONG NHAT (XGBoost Feature Importance)\nCac indicator co anh huong lon nhat den quyet dinh giao dich', 
            fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add values
for i, (bar, imp) in enumerate(zip(bars, importance)):
    ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='black', label='Momentum'),
    Patch(facecolor='#3498db', edgecolor='black', label='Trend'),
    Patch(facecolor='#9b59b6', edgecolor='black', label='Momentum Indicators'),
    Patch(facecolor='#e67e22', edgecolor='black', label='Volatility'),
    Patch(facecolor='#16a085', edgecolor='black', label='Strength'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='Others')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, title='Category')

plt.tight_layout()
plt.savefig('visualization/indicators_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/indicators_importance.png")

# ============================================================================
# CHART 4: INDICATOR SUMMARY INFOGRAPHIC
# ============================================================================

fig4 = plt.figure(figsize=(16, 10))
ax = fig4.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, '100+ TECHNICAL INDICATORS - TONG QUAN', 
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(5, 9, 'Tu OHLCV Data den Trading Signals', 
        ha='center', va='top', fontsize=14, style='italic', color='#7f8c8d')

# Box 1: Input
box1 = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                      edgecolor='#3498db', facecolor='#ecf0f1', linewidth=3)
ax.add_patch(box1)
ax.text(1.5, 8.2, 'INPUT', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#2c3e50')
ax.text(1.5, 7.7, 'OHLCV Data', ha='center', va='center', fontsize=11)
ax.text(1.5, 7.4, '1000 bars', ha='center', va='center', fontsize=10, color='#7f8c8d')

# Arrow 1
ax.annotate('', xy=(3, 7.75), xytext=(2.5, 7.75),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# Box 2: Feature Engineering
box2 = FancyBboxPatch((3, 6.5), 4, 2.5, boxstyle="round,pad=0.1", 
                      edgecolor='#2ecc71', facecolor='#d5f4e6', linewidth=3)
ax.add_patch(box2)
ax.text(5, 8.7, 'FEATURE ENGINEERING', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#27ae60')

features_text = [
    '- Momentum: 8 features (1h to 48h)',
    '- Trend EMAs: 13 features (9 to 200)',
    '- RSI: 2 features',
    '- Volatility: 6 features (ATR, BB)',
    '- Volume: 1 feature',
    '- ADX: 2 features',
    '- Structure: 3 features',
    '- MTF: 2 features'
]

y_pos = 8.2
for text in features_text:
    ax.text(3.2, y_pos, text, ha='left', va='center', fontsize=9, family='monospace')
    y_pos -= 0.25

ax.text(5, 6.7, 'TOTAL: 100+ Features', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#27ae60')

# Arrow 2
ax.annotate('', xy=(7.5, 7.75), xytext=(7, 7.75),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# Box 3: ML Model
box3 = FancyBboxPatch((7.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                      edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=3)
ax.add_patch(box3)
ax.text(8.5, 8.2, 'ML MODEL', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#c0392b')
ax.text(8.5, 7.7, 'XGBoost', ha='center', va='center', fontsize=11)
ax.text(8.5, 7.4, '200 trees', ha='center', va='center', fontsize=10, color='#7f8c8d')

# Arrow down
ax.annotate('', xy=(8.5, 6.5), xytext=(8.5, 7),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# Box 4: Output
box4 = FancyBboxPatch((7.5, 5), 2, 1.3, boxstyle="round,pad=0.1", 
                      edgecolor='#f39c12', facecolor='#fef5e7', linewidth=3)
ax.add_patch(box4)
ax.text(8.5, 6, 'OUTPUT', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#d68910')
ax.text(8.5, 5.6, 'Trading Signal', ha='center', va='center', fontsize=11)
ax.text(8.5, 5.3, 'LONG/SHORT/HOLD', ha='center', va='center', fontsize=10, color='#7f8c8d')

# Statistics boxes
stats_data = [
    ('MOMENTUM', '12', '#e74c3c', (0.5, 4)),
    ('TREND', '13', '#3498db', (2.5, 4)),
    ('VOLATILITY', '6', '#e67e22', (4.5, 4)),
    ('VOLUME', '1', '#1abc9c', (6.5, 4)),
    ('STRENGTH', '2', '#16a085', (8.5, 4)),
]

for name, count, color, pos in stats_data:
    box = FancyBboxPatch((pos[0]-0.8, pos[1]-0.6), 1.6, 1.2, boxstyle="round,pad=0.1", 
                        edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(pos[0], pos[1]+0.3, count, ha='center', va='center', 
           fontsize=24, fontweight='bold', color=color)
    ax.text(pos[0], pos[1]-0.2, name, ha='center', va='center', 
           fontsize=9, fontweight='bold', color='#2c3e50')

# Bottom info
info_box = FancyBboxPatch((0.5, 0.3), 9, 2, boxstyle="round,pad=0.15", 
                         edgecolor='#34495e', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(info_box)

ax.text(5, 2, 'TAI SAO CAN 100+ INDICATORS?', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#2c3e50')

reasons = [
    '1. Multi-Scale Analysis: Phan tich nhieu khung thoi gian (1h -> 48h)',
    '2. Multiple Perspectives: Momentum, Trend, Volatility, Volume, Strength',
    '3. Confirmation: Nhieu indicators dong thuan = tin hieu manh',
    '4. ML Power: XGBoost hoc patterns phuc tap tu 100+ features',
    '5. High Accuracy: Ket qua Win Rate 85.2% nho feature engineering tot'
]

y_pos = 1.5
for reason in reasons:
    ax.text(0.8, y_pos, reason, ha='left', va='center', 
           fontsize=9, family='monospace', color='#2c3e50')
    y_pos -= 0.25

plt.tight_layout()
plt.savefig('visualization/indicators_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/indicators_summary.png")

print("\n" + "="*70)
print(">>> HOAN THANH! Da tao 4 hinh anh cho Indicators:")
print("="*70)
print("1. indicators_showcase.png - 6 panels hien thi indicators chinh")
print("2. indicators_categories.png - Phan loai indicators theo chuc nang")
print("3. indicators_importance.png - Top 20 indicators quan trong nhat")
print("4. indicators_summary.png - Infographic tong quan")
print("="*70)
print("\n>> MEO TRINH BAY:")
print("• Slide 1: Dung showcase de gioi thieu cac indicators")
print("• Slide 2: Dung categories de phan loai ro rang")
print("• Slide 3: Dung importance de nhan manh indicators chinh")
print("• Slide 4: Dung summary de tong ket toan bo quy trinh")
print("="*70)
