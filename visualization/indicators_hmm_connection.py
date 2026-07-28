"""
Visualization: Mối liên kết giữa Indicators và HMM
Giải thích cách indicators được sử dụng làm input cho HMM để detect regime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CHART 1: INDICATORS -> HMM FLOW
# ============================================================================

fig1 = plt.figure(figsize=(18, 12))
ax = fig1.add_subplot(111)
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(6, 11.5, 'MOI LIEN KET GIUA INDICATORS VA HMM', 
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(6, 11, 'Indicators la INPUT cho HMM de nhan dien Market Regime', 
        ha='center', va='top', fontsize=14, style='italic', color='#7f8c8d')

# ===== LAYER 1: OHLCV DATA =====
box_ohlcv = FancyBboxPatch((0.5, 9), 2, 1.5, boxstyle="round,pad=0.1", 
                           edgecolor='#3498db', facecolor='#ecf0f1', linewidth=3)
ax.add_patch(box_ohlcv)
ax.text(1.5, 10.2, 'OHLCV DATA', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#2c3e50')
ax.text(1.5, 9.7, 'Price, Volume', ha='center', va='center', fontsize=10)
ax.text(1.5, 9.4, '1000 bars', ha='center', va='center', fontsize=9, color='#7f8c8d')

# Arrow to indicators
ax.annotate('', xy=(3, 9.75), xytext=(2.5, 9.75),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# ===== LAYER 2: ALL INDICATORS (100+) =====
box_all_ind = FancyBboxPatch((3, 8.5), 3, 2.5, boxstyle="round,pad=0.1", 
                             edgecolor='#95a5a6', facecolor='#ecf0f1', linewidth=2, linestyle='--')
ax.add_patch(box_all_ind)
ax.text(4.5, 10.7, '100+ INDICATORS', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#7f8c8d')

indicators_list = [
    'Momentum (8)', 'EMAs (13)', 'RSI (2)',
    'Volatility (6)', 'Volume (1)', 'ADX (2)',
    'Structure (3)', 'MTF (2)', '...'
]
y_pos = 10.2
for ind in indicators_list:
    ax.text(3.2, y_pos, f'- {ind}', ha='left', va='center', 
           fontsize=8, color='#7f8c8d', family='monospace')
    y_pos -= 0.2

# ===== LAYER 3: HMM INPUT INDICATORS (5 key ones) =====
# Arrow down to HMM inputs
ax.annotate('', xy=(4.5, 8), xytext=(4.5, 8.5),
           arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c'))
ax.text(4.8, 8.25, 'Chon 5 indicators\nquan trong nhat', ha='left', va='center',
       fontsize=9, fontweight='bold', color='#e74c3c')

# 5 HMM input indicators
hmm_indicators = [
    ('RSI', '#9b59b6', (1, 6)),
    ('MACD Hist', '#3498db', (2.5, 6)),
    ('BB Width', '#e67e22', (4, 6)),
    ('ATR', '#e74c3c', (5.5, 6)),
    ('Volume Ratio', '#1abc9c', (7, 6))
]

for name, color, pos in hmm_indicators:
    box = FancyBboxPatch((pos[0]-0.6, pos[1]-0.5), 1.2, 1, boxstyle="round,pad=0.1", 
                        edgecolor=color, facecolor='white', linewidth=3)
    ax.add_patch(box)
    ax.text(pos[0], pos[1]+0.2, name, ha='center', va='center', 
           fontsize=10, fontweight='bold', color=color)
    ax.text(pos[0], pos[1]-0.2, 'HMM Input', ha='center', va='center', 
           fontsize=8, color='#7f8c8d')

# ===== LAYER 4: HMM MODEL =====
# Arrows from indicators to HMM
for name, color, pos in hmm_indicators:
    ax.annotate('', xy=(4.5, 4.5), xytext=(pos[0], pos[1]-0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.6))

# HMM box
box_hmm = FancyBboxPatch((2.5, 3), 4, 2, boxstyle="round,pad=0.15", 
                        edgecolor='#2ecc71', facecolor='#d5f4e6', linewidth=4)
ax.add_patch(box_hmm)
ax.text(4.5, 4.7, 'HIDDEN MARKOV MODEL', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#27ae60')
ax.text(4.5, 4.3, 'Gaussian HMM (4 states)', ha='center', va='center', fontsize=10)

hmm_details = [
    '- Forward-Backward Algorithm',
    '- Emission Probabilities',
    '- Transition Matrix'
]
y_pos = 3.9
for detail in hmm_details:
    ax.text(2.7, y_pos, detail, ha='left', va='center', 
           fontsize=8, family='monospace', color='#27ae60')
    y_pos -= 0.25

# ===== LAYER 5: REGIME OUTPUT =====
# Arrow to regimes
ax.annotate('', xy=(4.5, 2.5), xytext=(4.5, 3),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# 4 Regimes
regimes = [
    ('TRENDING', '#2ecc71', (1.5, 1)),
    ('RANGING', '#f39c12', (3.5, 1)),
    ('VOLATILE', '#e74c3c', (5.5, 1)),
    ('CALM', '#95a5a6', (7.5, 1))
]

for name, color, pos in regimes:
    box = FancyBboxPatch((pos[0]-0.7, pos[1]-0.4), 1.4, 0.8, boxstyle="round,pad=0.1", 
                        edgecolor=color, facecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(pos[0], pos[1], name, ha='center', va='center', 
           fontsize=10, fontweight='bold', color=color)

# ===== RIGHT SIDE: EXPLANATION =====
explain_box = FancyBboxPatch((8, 3), 3.5, 7.5, boxstyle="round,pad=0.15", 
                            edgecolor='#34495e', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(explain_box)

ax.text(9.75, 10.2, 'TAI SAO DUNG 5 INDICATORS NAY?', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#2c3e50')

explanations = [
    ('1. RSI (Momentum)', '#9b59b6', [
        '- Do suc manh xu huong',
        '- >70: Overbought',
        '- <30: Oversold',
        '- ~50: Neutral'
    ]),
    ('2. MACD Hist (Trend)', '#3498db', [
        '- Huong va suc manh trend',
        '- Duong: Bullish',
        '- Am: Bearish',
        '- Gan 0: Sideways'
    ]),
    ('3. BB Width (Volatility)', '#e67e22', [
        '- Do bien dong gia',
        '- Rong: High volatility',
        '- Hep: Low volatility',
        '- Squeeze -> Breakout'
    ]),
    ('4. ATR (True Volatility)', '#e74c3c', [
        '- Bien dong thuc te',
        '- Cao: Volatile regime',
        '- Thap: Calm regime',
        '- Normalize: ATR/Price'
    ]),
    ('5. Volume Ratio', '#1abc9c', [
        '- Xac nhan move',
        '- >1.5: High volume',
        '- <0.5: Low volume',
        '- Ratio vs MA(20)'
    ])
]

y_pos = 9.7
for title, color, points in explanations:
    ax.text(8.3, y_pos, title, ha='left', va='center', 
           fontsize=9, fontweight='bold', color=color)
    y_pos -= 0.2
    for point in points:
        ax.text(8.5, y_pos, point, ha='left', va='center', 
               fontsize=7, family='monospace', color='#2c3e50')
        y_pos -= 0.15
    y_pos -= 0.1

plt.tight_layout()
plt.savefig('visualization/indicators_hmm_connection.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/indicators_hmm_connection.png")

# ============================================================================
# CHART 2: HMM WORKING MECHANISM
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('CACH HMM SU DUNG INDICATORS DE NHAN DIEN REGIME', 
              fontsize=18, fontweight='bold')

# Panel 1: Indicator Values by Regime
ax1 = axes[0, 0]
regimes_data = ['Trending', 'Ranging', 'Volatile', 'Calm']
rsi_values = [65, 50, 45, 52]
macd_values = [0.08, 0.01, -0.05, 0.02]
bb_width_values = [0.025, 0.015, 0.045, 0.012]
atr_values = [0.018, 0.012, 0.035, 0.008]

x = np.arange(len(regimes_data))
width = 0.2

bars1 = ax1.bar(x - 1.5*width, rsi_values, width, label='RSI', color='#9b59b6', edgecolor='black')
bars2 = ax1.bar(x - 0.5*width, [m*100 for m in macd_values], width, label='MACD*100', color='#3498db', edgecolor='black')
bars3 = ax1.bar(x + 0.5*width, [b*100 for b in bb_width_values], width, label='BB Width*100', color='#e67e22', edgecolor='black')
bars4 = ax1.bar(x + 1.5*width, [a*100 for a in atr_values], width, label='ATR*100', color='#e74c3c', edgecolor='black')

ax1.set_ylabel('Gia tri (normalized)', fontsize=12, fontweight='bold')
ax1.set_title('Gia tri Indicators dac trung cho tung Regime', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(regimes_data)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Emission Probabilities
ax2 = axes[0, 1]

# Simulated emission probabilities for each regime given certain indicator values
# Example: If RSI=65, MACD=0.08, BB_width=0.025, ATR=0.018
emission_probs = [0.75, 0.10, 0.10, 0.05]  # [Trending, Ranging, Volatile, Calm]
colors_regime = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']

bars = ax2.bar(regimes_data, emission_probs, color=colors_regime, edgecolor='black', linewidth=2)
ax2.set_ylabel('Emission Probability', fontsize=12, fontweight='bold')
ax2.set_title('P(Indicators | Regime)\nVi du: RSI=65, MACD=0.08, BB=0.025, ATR=0.018', 
             fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1)

for bar, prob in zip(bars, emission_probs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{prob:.2f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.grid(axis='y', alpha=0.3)

# Panel 3: Transition Matrix
ax3 = axes[1, 0]

transition_matrix = np.array([
    [0.70, 0.15, 0.10, 0.05],  # From Trending
    [0.20, 0.60, 0.15, 0.05],  # From Ranging
    [0.25, 0.20, 0.50, 0.05],  # From Volatile
    [0.15, 0.20, 0.10, 0.55]   # From Calm
])

im = ax3.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax3.set_xticks(range(4))
ax3.set_yticks(range(4))
ax3.set_xticklabels(regimes_data, fontsize=10)
ax3.set_yticklabels(regimes_data, fontsize=10)
ax3.set_xlabel('To Regime', fontsize=12, fontweight='bold')
ax3.set_ylabel('From Regime', fontsize=12, fontweight='bold')
ax3.set_title('Transition Matrix: P(Regime_t+1 | Regime_t)', fontsize=13, fontweight='bold')

# Add text annotations
for i in range(4):
    for j in range(4):
        text = ax3.text(j, i, f'{transition_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax3, label='Probability')

# Panel 4: Forward-Backward Algorithm
ax4 = axes[1, 1]
ax4.axis('off')

ax4.text(0.5, 0.95, 'FORWARD-BACKWARD ALGORITHM', ha='center', va='top', 
        fontsize=13, fontweight='bold', transform=ax4.transAxes)

algorithm_text = """
1. FORWARD PASS (α):
   α_t(i) = P(regime_i at t, obs 1..t)
   
   α_t(i) = P(obs_t | regime_i) × 
            Σ[α_t-1(j) × P(i|j)]
   
2. BACKWARD PASS (β):
   β_t(i) = P(obs t+1..T | regime_i at t)
   
   β_t(i) = Σ[P(obs_t+1 | regime_j) × 
             P(j|i) × β_t+1(j)]

3. POSTERIOR PROBABILITY:
   P(regime_i at t | all obs) = 
       α_t(i) × β_t(i) / P(all obs)

4. KET QUA:
   - Moi thoi diem t co 4 probabilities
   - Tong = 1.0
   - Chon regime co prob cao nhat
   
VI DU:
   P(Trending) = 0.75  <- MAX!
   P(Ranging)  = 0.15
   P(Volatile) = 0.05
   P(Calm)     = 0.05
   
   -> Current Regime: TRENDING
"""

ax4.text(0.05, 0.85, algorithm_text, ha='left', va='top', 
        fontsize=9, family='monospace', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('visualization/hmm_mechanism.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/hmm_mechanism.png")

# ============================================================================
# CHART 3: COMPLETE PIPELINE
# ============================================================================

fig3 = plt.figure(figsize=(18, 10))
ax = fig3.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(7, 9.5, 'PIPELINE HOAN CHINH: INDICATORS -> HMM -> FILTERS -> SIGNAL', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# Step 1: OHLCV
box1 = FancyBboxPatch((0.5, 7), 1.5, 1.5, boxstyle="round,pad=0.1", 
                      edgecolor='#3498db', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(box1)
ax.text(1.25, 8, 'OHLCV', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.25, 7.5, '1000 bars', ha='center', va='center', fontsize=9)

# Arrow
ax.annotate('', xy=(2.5, 7.75), xytext=(2, 7.75),
           arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

# Step 2: 100+ Indicators
box2 = FancyBboxPatch((2.5, 7), 1.5, 1.5, boxstyle="round,pad=0.1", 
                      edgecolor='#9b59b6', facecolor='#f4ecf7', linewidth=2)
ax.add_patch(box2)
ax.text(3.25, 8, '100+', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(3.25, 7.5, 'Indicators', ha='center', va='center', fontsize=9)

# Arrow split
ax.annotate('', xy=(4.5, 8.5), xytext=(4, 7.75),
           arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
ax.text(4.2, 8.2, '5 key', ha='left', va='center', fontsize=8, color='#e74c3c')

ax.annotate('', xy=(5, 7.75), xytext=(4, 7.75),
           arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
ax.text(4.5, 7.95, 'All', ha='center', va='center', fontsize=8)

# Step 3a: HMM (top path)
box3a = FancyBboxPatch((4.5, 8.2), 1.5, 1, boxstyle="round,pad=0.1", 
                       edgecolor='#2ecc71', facecolor='#d5f4e6', linewidth=2)
ax.add_patch(box3a)
ax.text(5.25, 8.7, 'HMM', ha='center', va='center', fontsize=11, fontweight='bold', color='#27ae60')

# Step 3b: XGBoost (bottom path)
box3b = FancyBboxPatch((5, 7), 1.5, 1.5, boxstyle="round,pad=0.1", 
                       edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(box3b)
ax.text(5.75, 8, 'XGBoost', ha='center', va='center', fontsize=11, fontweight='bold', color='#c0392b')
ax.text(5.75, 7.5, 'ML Model', ha='center', va='center', fontsize=9)

# Arrows to merge
ax.annotate('', xy=(7, 8.5), xytext=(6, 8.7),
           arrowprops=dict(arrowstyle='->', lw=2, color='#2ecc71'))
ax.text(6.5, 8.8, 'Regime', ha='center', va='center', fontsize=8, color='#27ae60')

ax.annotate('', xy=(7, 7.75), xytext=(6.5, 7.75),
           arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
ax.text(6.75, 7.95, 'Signal', ha='center', va='center', fontsize=8, color='#e74c3c')

# Step 4: Filters
box4 = FancyBboxPatch((7, 7), 2, 2, boxstyle="round,pad=0.1", 
                      edgecolor='#f39c12', facecolor='#fef5e7', linewidth=3)
ax.add_patch(box4)
ax.text(8, 8.7, '4 FILTERS', ha='center', va='center', fontsize=12, fontweight='bold', color='#d68910')

filters = ['1. Regime Check', '2. Confidence', '3. ADX', '4. MTF']
y_pos = 8.3
for f in filters:
    ax.text(7.2, y_pos, f, ha='left', va='center', fontsize=9)
    y_pos -= 0.3

# Arrow
ax.annotate('', xy=(9.5, 8), xytext=(9, 8),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# Step 5: Final Signal
box5 = FancyBboxPatch((9.5, 7), 1.5, 2, boxstyle="round,pad=0.15", 
                      edgecolor='#27ae60', facecolor='#abebc6', linewidth=3)
ax.add_patch(box5)
ax.text(10.25, 8.5, 'SIGNAL', ha='center', va='center', fontsize=13, fontweight='bold', color='#27ae60')
ax.text(10.25, 8, 'LONG', ha='center', va='center', fontsize=11)
ax.text(10.25, 7.6, 'SHORT', ha='center', va='center', fontsize=11)
ax.text(10.25, 7.2, 'HOLD', ha='center', va='center', fontsize=11)

# Bottom explanation
explain_box = FancyBboxPatch((0.5, 0.5), 10, 5.5, boxstyle="round,pad=0.15", 
                            edgecolor='#34495e', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(explain_box)

ax.text(5.5, 5.7, 'VAI TRO CUA TUNG THANH PHAN', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#2c3e50')

roles = [
    ('100+ INDICATORS:', [
        '- Phan tich toan dien thi truong',
        '- Nhieu goc nhin: Momentum, Trend, Volatility, Volume, Strength',
        '- Cung cap du lieu cho ca HMM va XGBoost'
    ]),
    ('HMM (5 indicators):', [
        '- Nhan dien BOI CANH thi truong (Regime)',
        '- Input: RSI, MACD, BB Width, ATR, Volume',
        '- Output: Trending/Ranging/Volatile/Calm',
        '- Giup Bot biet NEN hay KHONG NEN giao dich'
    ]),
    ('XGBOOST (100+ indicators):', [
        '- Du bao HUONG di cua gia (Signal)',
        '- Input: Tat ca 100+ indicators',
        '- Output: LONG/SHORT/HOLD + Confidence',
        '- Hoc patterns phuc tap tu historical data'
    ]),
    ('4 FILTERS:', [
        '- Ket hop ket qua tu HMM va XGBoost',
        '- Loc tin hieu chat luong cao',
        '- Dam bao dieu kien DU de vao lenh',
        '- Ket qua: Win Rate 85.2%'
    ])
]

y_pos = 5.2
for title, points in roles:
    ax.text(0.8, y_pos, title, ha='left', va='center', 
           fontsize=10, fontweight='bold', color='#2c3e50')
    y_pos -= 0.25
    for point in points:
        ax.text(1.0, y_pos, point, ha='left', va='center', 
               fontsize=8, family='monospace', color='#34495e')
        y_pos -= 0.2
    y_pos -= 0.15

# Key insight box
insight_box = FancyBboxPatch((0.8, 0.7), 9.4, 0.8, boxstyle="round,pad=0.1", 
                            edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(insight_box)
ax.text(5.5, 1.1, '>> DIEM QUAN TRONG: HMM va XGBoost lam viec SONG SONG, khong phu thuoc nhau', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='#c0392b')
ax.text(5.5, 0.85, 'HMM nhan dien BOI CANH (nen trade hay khong), XGBoost du bao HUONG (long hay short)', 
        ha='center', va='center', fontsize=9, style='italic', color='#c0392b')

plt.tight_layout()
plt.savefig('visualization/complete_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/complete_pipeline.png")

print("\n" + "="*70)
print(">>> HOAN THANH! Da tao 3 hinh anh ve moi lien ket Indicators-HMM:")
print("="*70)
print("1. indicators_hmm_connection.png - Flow tu indicators den HMM")
print("2. hmm_mechanism.png - Cach HMM hoat dong voi indicators")
print("3. complete_pipeline.png - Pipeline hoan chinh")
print("="*70)
print("\n>> TOM TAT MOI LIEN KET:")
print("- 100+ indicators duoc tinh tu OHLCV data")
print("- 5 indicators chinh (RSI, MACD, BB, ATR, Volume) lam INPUT cho HMM")
print("- HMM nhan dien REGIME (Trending/Ranging/Volatile/Calm)")
print("- XGBoost dung TAT CA 100+ indicators de du bao SIGNAL")
print("- Filters ket hop ket qua HMM + XGBoost de loc tin hieu")
print("="*70)
