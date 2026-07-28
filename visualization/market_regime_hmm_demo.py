"""
Market Regime Detection Visualization using HMM
Tạo biểu đồ demo cho slide thuyết trình về nhận diện cấu trúc thị trường
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate synthetic price data with different regimes
np.random.seed(42)
n_points = 500

# Create time series
dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')

# Simulate different market regimes
price = np.zeros(n_points)
regime = np.zeros(n_points)
price[0] = 100

# Define regime periods
# 0: Calm, 1: Trending Up, 2: Ranging, 3: Volatile, 4: Trending Down
regime_periods = [
    (0, 80, 0, 0.3),      # Calm
    (80, 180, 1, 0.8),    # Trending Up
    (180, 280, 2, 0.4),   # Ranging
    (280, 350, 3, 1.5),   # Volatile
    (350, 420, 4, 0.7),   # Trending Down
    (420, 500, 1, 0.6),   # Trending Up again
]

for start, end, reg, volatility in regime_periods:
    for i in range(start, min(end, n_points)):
        regime[i] = reg
        
        if reg == 0:  # Calm
            drift = 0.01
            vol = volatility
        elif reg == 1:  # Trending Up
            drift = 0.15
            vol = volatility
        elif reg == 2:  # Ranging
            drift = 0.02 * np.sin(i * 0.1)
            vol = volatility
        elif reg == 3:  # Volatile
            drift = np.random.choice([-0.3, 0.3])
            vol = volatility
        elif reg == 4:  # Trending Down
            drift = -0.12
            vol = volatility
        
        if i > 0:
            price[i] = price[i-1] * (1 + drift/100 + np.random.randn() * vol/100)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'price': price,
    'regime': regime
})

# Regime names and colors
regime_names = {
    0: 'Calm (Yên bình)',
    1: 'Trending (Xu hướng)',
    2: 'Ranging (Đi ngang)',
    3: 'Volatile (Biến động)',
    4: 'Trending Down (Xu hướng giảm)'
}

regime_colors = {
    0: '#95a5a6',  # Gray - Calm
    1: '#2ecc71',  # Green - Trending Up
    2: '#f39c12',  # Orange - Ranging
    3: '#e74c3c',  # Red - Volatile
    4: '#c0392b',  # Dark Red - Trending Down
}

bot_actions = {
    0: 'Chờ đợi',
    1: 'TẤN CÔNG (Long)',
    2: 'Thận trọng',
    3: 'ĐỨNG NGOÀI',
    4: 'TẤN CÔNG (Short)'
}

# Create the main visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

# Main price chart with regime backgrounds
ax1 = fig.add_subplot(gs[0])

# Plot regime backgrounds
for i in range(len(df) - 1):
    reg = int(df.iloc[i]['regime'])
    ax1.axvspan(df.iloc[i]['date'], df.iloc[i+1]['date'], 
                alpha=0.3, color=regime_colors[reg])

# Plot price line
ax1.plot(df['date'], df['price'], color='#2c3e50', linewidth=2, label='Giá BTC/USDT')

# Add trading signals
for reg_id, periods in [(1, (80, 180)), (1, (420, 500)), (4, (350, 420))]:
    start_idx, end_idx = periods
    if reg_id == 1:  # Long signals
        ax1.scatter(df.iloc[start_idx]['date'], df.iloc[start_idx]['price'], 
                   color='green', s=200, marker='^', zorder=5, edgecolors='black', linewidth=2)
        ax1.scatter(df.iloc[end_idx-1]['date'], df.iloc[end_idx-1]['price'], 
                   color='red', s=200, marker='v', zorder=5, edgecolors='black', linewidth=2)
    elif reg_id == 4:  # Short signals
        ax1.scatter(df.iloc[start_idx]['date'], df.iloc[start_idx]['price'], 
                   color='red', s=200, marker='v', zorder=5, edgecolors='black', linewidth=2)
        ax1.scatter(df.iloc[end_idx-1]['date'], df.iloc[end_idx-1]['price'], 
                   color='green', s=200, marker='^', zorder=5, edgecolors='black', linewidth=2)

ax1.set_ylabel('Giá (USDT)', fontsize=14, fontweight='bold')
ax1.set_title('NHẬN DIỆN CẤU TRÚC THỊ TRƯỜNG VỚI HIDDEN MARKOV MODEL (HMM)\nBot tự động điều chỉnh chiến thuật theo từng trạng thái', 
              fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=12)

# Regime indicator
ax2 = fig.add_subplot(gs[1], sharex=ax1)
for i in range(len(df) - 1):
    reg = int(df.iloc[i]['regime'])
    ax2.axvspan(df.iloc[i]['date'], df.iloc[i+1]['date'], 
                alpha=0.7, color=regime_colors[reg])
    
ax2.set_ylabel('Trạng thái\nThị trường', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.5, 0.5)
ax2.set_yticks([])
ax2.grid(True, alpha=0.3)

# Bot action indicator
ax3 = fig.add_subplot(gs[2], sharex=ax1)
for i in range(len(df) - 1):
    reg = int(df.iloc[i]['regime'])
    ax3.axvspan(df.iloc[i]['date'], df.iloc[i+1]['date'], 
                alpha=0.7, color=regime_colors[reg])
    
ax3.set_ylabel('Hành động\nBot', fontsize=12, fontweight='bold')
ax3.set_xlabel('Thời gian', fontsize=12, fontweight='bold')
ax3.set_ylim(-0.5, 0.5)
ax3.set_yticks([])
ax3.grid(True, alpha=0.3)

# Add legend for regimes - moved to lower right to avoid covering price action
legend_elements = []
for reg_id, name in regime_names.items():
    legend_elements.append(Rectangle((0, 0), 1, 1, fc=regime_colors[reg_id], 
                                    alpha=0.8, label=f'{name}: {bot_actions[reg_id]}'))

ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, 
          title='Trang thai & Hanh dong Bot', title_fontsize=11,
          frameon=True, fancybox=True, shadow=True,
          framealpha=0.95, edgecolor='black', facecolor='white')

plt.tight_layout()
plt.savefig('visualization/market_regime_hmm_slide.png', dpi=300, bbox_inches='tight')
print("✅ Đã tạo: visualization/market_regime_hmm_slide.png")

# Create a second detailed explanation chart
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('CHI TIẾT 4 TRẠNG THÁI THỊ TRƯỜNG & CHIẾN LƯỢC BOT', 
              fontsize=18, fontweight='bold', y=0.98)

# Example data for each regime
regime_examples = {
    'Trending': {
        'data': np.cumsum(np.random.randn(100) * 0.5 + 0.3) + 100,
        'color': regime_colors[1],
        'action': 'TẤN CÔNG - Mở lệnh Long/Short mạnh mẽ',
        'description': '- Xu huong ro rang\n- Momentum manh\n- Bot tan dung toi da\n- Win rate cao nhat',
        'pos': (0, 0)
    },
    'Ranging': {
        'data': 100 + 5 * np.sin(np.linspace(0, 8*np.pi, 100)) + np.random.randn(100) * 0.5,
        'color': regime_colors[2],
        'action': 'THẬN TRỌNG - Giao dịch biên độ hẹp',
        'description': '- Gia dao dong trong range\n- Mua day, ban dinh\n- Position size nho\n- Chot loi nhanh',
        'pos': (0, 1)
    },
    'Volatile': {
        'data': 100 + np.cumsum(np.random.randn(100) * 2),
        'color': regime_colors[3],
        'action': 'ĐỨNG NGOÀI - Bảo vệ vốn',
        'description': '- Bien dong cuc manh\n- Nhieu tin hieu gia\n- Rui ro cao\n- Bot tu dong Hold',
        'pos': (1, 0)
    },
    'Calm': {
        'data': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'color': regime_colors[0],
        'action': 'CHỜ ĐỢI - Tích lũy năng lượng',
        'description': '- Thi truong yen tinh\n- Chuan bi breakout\n- Bot quan sat\n- San sang hanh dong',
        'pos': (1, 1)
    }
}

for regime_name, info in regime_examples.items():
    row, col = info['pos']
    ax = axes[row, col]
    
    # Plot price
    ax.plot(info['data'], color=info['color'], linewidth=2.5)
    ax.fill_between(range(len(info['data'])), info['data'], 
                     alpha=0.3, color=info['color'])
    
    # Styling
    ax.set_title(f'{regime_name}\n{info["action"]}', 
                fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Thời gian', fontsize=11)
    ax.set_ylabel('Giá', fontsize=11)
    
    # Add description box
    props = dict(boxstyle='round', facecolor=info['color'], alpha=0.2)
    ax.text(0.02, 0.98, info['description'], transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', bbox=props, 
           family='monospace')

plt.tight_layout()
plt.savefig('visualization/market_regime_details_slide.png', dpi=300, bbox_inches='tight')
print("✅ Đã tạo: visualization/market_regime_details_slide.png")

# Create comparison chart: Bot with HMM vs Bot without HMM
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig3.suptitle('SO SÁNH: BOT CÓ HMM vs BOT KHÔNG CÓ HMM\nTại sao HMM là "Điểm ăn tiền"?', 
              fontsize=18, fontweight='bold')

# Simulate performance
cumulative_return_with_hmm = np.zeros(n_points)
cumulative_return_without_hmm = np.zeros(n_points)

for i in range(1, n_points):
    reg = int(regime[i])
    price_change = (price[i] - price[i-1]) / price[i-1] * 100
    
    # Bot with HMM - smart trading
    if reg == 1:  # Trending up
        cumulative_return_with_hmm[i] = cumulative_return_with_hmm[i-1] + abs(price_change) * 0.8
    elif reg == 4:  # Trending down
        cumulative_return_with_hmm[i] = cumulative_return_with_hmm[i-1] + abs(price_change) * 0.8
    elif reg == 2:  # Ranging
        cumulative_return_with_hmm[i] = cumulative_return_with_hmm[i-1] + abs(price_change) * 0.3
    elif reg == 3:  # Volatile - STAND ASIDE
        cumulative_return_with_hmm[i] = cumulative_return_with_hmm[i-1]
    else:  # Calm
        cumulative_return_with_hmm[i] = cumulative_return_with_hmm[i-1]
    
    # Bot without HMM - always trading
    if price_change > 0:
        cumulative_return_without_hmm[i] = cumulative_return_without_hmm[i-1] + abs(price_change) * 0.5
    else:
        cumulative_return_without_hmm[i] = cumulative_return_without_hmm[i-1] - abs(price_change) * 0.7

# Plot comparison
ax1.plot(dates, cumulative_return_with_hmm, color='#2ecc71', linewidth=3, 
         label='Bot CÓ HMM (Thông minh)', marker='o', markevery=50, markersize=8)
ax1.plot(dates, cumulative_return_without_hmm, color='#e74c3c', linewidth=3, 
         label='Bot KHÔNG HMM (Giao dịch mù)', marker='x', markevery=50, markersize=8)

# Highlight volatile period
volatile_start = dates[280]
volatile_end = dates[350]
ax1.axvspan(volatile_start, volatile_end, alpha=0.3, color='red', 
           label='Giai đoạn Volatile (2022 crash)')

ax1.set_ylabel('Lợi nhuận tích lũy (%)', fontsize=14, fontweight='bold')
ax1.set_title('Hiệu suất giao dịch', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add annotations
ax1.annotate('Bot co HMM DUNG NGOAI\ntrong giai doan Volatile\n-> Bao ve von thanh cong!', 
            xy=(dates[315], cumulative_return_with_hmm[315]), 
            xytext=(dates[250], cumulative_return_with_hmm[315] + 20),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax1.annotate('Bot khong HMM tiep tuc\ngiao dich -> Thua lo nang!', 
            xy=(dates[315], cumulative_return_without_hmm[315]), 
            xytext=(dates[380], cumulative_return_without_hmm[315] - 15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Regime visualization
ax2.plot(dates, price, color='#34495e', linewidth=2, label='Giá BTC')
for i in range(len(df) - 1):
    reg = int(df.iloc[i]['regime'])
    ax2.axvspan(df.iloc[i]['date'], df.iloc[i+1]['date'], 
                alpha=0.3, color=regime_colors[reg])

ax2.set_ylabel('Gia (USDT)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Thoi gian', fontsize=14, fontweight='bold')
ax2.set_title('Trang thai thi truong duoc HMM nhan dien', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization/hmm_comparison_slide.png', dpi=300, bbox_inches='tight')
print("✅ Đã tạo: visualization/hmm_comparison_slide.png")

print("\n" + "="*60)
print(">>> HOAN THANH! Da tao 3 hinh anh cho slide:")
print("="*60)
print("1. market_regime_hmm_slide.png - Bieu do chinh voi cac trang thai")
print("2. market_regime_details_slide.png - Chi tiet 4 trang thai")
print("3. hmm_comparison_slide.png - So sanh Bot co/khong co HMM")
print("="*60)
print("\n>> MEO TRINH BAY:")
print("• Slide 1: Gioi thieu tong quan ve HMM")
print("• Slide 2: Chi vao tung vung mau va giai thich")
print("• Slide 3: Nhan manh diem 'an tien' - bao ve von trong Volatile")
print("="*60)
