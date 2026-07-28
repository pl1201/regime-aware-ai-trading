"""
XGBoost Model Visualization - Chi tiet
Giai thich cach XGBoost hoat dong: Training, Prediction, Feature Importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CHART 1: XGBOOST ARCHITECTURE
# ============================================================================

fig1 = plt.figure(figsize=(18, 12))
ax = fig1.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(7, 11.5, 'XGBOOST MODEL - KIEN TRUC VA HOAT DONG', 
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(7, 11, 'Gradient Boosting Decision Trees cho Trading Signal Classification', 
        ha='center', va='top', fontsize=13, style='italic', color='#7f8c8d')

# ===== INPUT LAYER =====
box_input = FancyBboxPatch((0.5, 9), 2, 1.5, boxstyle="round,pad=0.1", 
                           edgecolor='#3498db', facecolor='#ecf0f1', linewidth=3)
ax.add_patch(box_input)
ax.text(1.5, 10.2, 'INPUT', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#2c3e50')
ax.text(1.5, 9.7, '100+ Features', ha='center', va='center', fontsize=10)
ax.text(1.5, 9.4, 'Normalized', ha='center', va='center', fontsize=9, color='#7f8c8d')

# Arrow
ax.annotate('', xy=(3, 9.75), xytext=(2.5, 9.75),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# ===== XGBOOST ENSEMBLE =====
ensemble_box = FancyBboxPatch((3, 7.5), 8, 3, boxstyle="round,pad=0.15", 
                              edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=3)
ax.add_patch(ensemble_box)
ax.text(7, 10.2, 'XGBOOST ENSEMBLE (200 Trees)', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#c0392b')

# Draw 5 trees as examples
tree_positions = [(3.5, 8.5), (5, 8.5), (6.5, 8.5), (8, 8.5), (9.5, 8.5)]
tree_labels = ['Tree 1', 'Tree 2', 'Tree 3', '...', 'Tree 200']

for i, (pos, label) in enumerate(zip(tree_positions, tree_labels)):
    x_pos = pos[0]  # Extract x coordinate
    y_pos = pos[1]  # Extract y coordinate
    
    if label == '...':
        ax.text(x_pos, 9, '...', ha='center', va='center', fontsize=20, fontweight='bold')
        continue
    
    # Tree structure (simplified) - using rectangles
    # Root
    root = Rectangle((x_pos-0.15, 9.15), 0.3, 0.3, facecolor='#3498db', edgecolor='black', linewidth=1.5)
    ax.add_patch(root)
    
    # Level 1
    left1 = Rectangle((x_pos-0.42, 8.78), 0.24, 0.24, facecolor='#2ecc71', edgecolor='black', linewidth=1)
    right1 = Rectangle((x_pos+0.18, 8.78), 0.24, 0.24, facecolor='#2ecc71', edgecolor='black', linewidth=1)
    ax.add_patch(left1)
    ax.add_patch(right1)
    
    # Level 2 (leaves)
    leaf_x_positions = [x_pos-0.45, x_pos-0.15, x_pos+0.15, x_pos+0.45]
    for leaf_x in leaf_x_positions:
        leaf = Rectangle((leaf_x-0.08, 8.42), 0.16, 0.16, 
                        facecolor='#f39c12', edgecolor='black', linewidth=1)
        ax.add_patch(leaf)
    
    # Connections
    ax.plot([x_pos, x_pos-0.3], [9.15, 9.02], 'k-', linewidth=1, alpha=0.5)
    ax.plot([x_pos, x_pos+0.3], [9.15, 9.02], 'k-', linewidth=1, alpha=0.5)
    ax.plot([x_pos-0.3, x_pos-0.45], [8.78, 8.58], 'k-', linewidth=0.8, alpha=0.5)
    ax.plot([x_pos-0.3, x_pos-0.15], [8.78, 8.58], 'k-', linewidth=0.8, alpha=0.5)
    ax.plot([x_pos+0.3, x_pos+0.15], [8.78, 8.58], 'k-', linewidth=0.8, alpha=0.5)
    ax.plot([x_pos+0.3, x_pos+0.45], [8.78, 8.58], 'k-', linewidth=0.8, alpha=0.5)
    
    # Label
    ax.text(x_pos, 8.2, label, ha='center', va='center', fontsize=8, fontweight='bold')

# Boosting explanation
ax.text(7, 7.8, 'Gradient Boosting: Moi tree hoc tu loi cua tree truoc', 
        ha='center', va='center', fontsize=9, style='italic', color='#c0392b')

# Arrow
ax.annotate('', xy=(11.5, 9.75), xytext=(11, 9.75),
           arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

# ===== OUTPUT LAYER =====
output_box = FancyBboxPatch((11.5, 8.5), 2, 2.5, boxstyle="round,pad=0.1", 
                            edgecolor='#2ecc71', facecolor='#d5f4e6', linewidth=3)
ax.add_patch(output_box)
ax.text(12.5, 10.7, 'OUTPUT', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='#27ae60')

# 3 classes with probabilities
classes = [
    ('HOLD', 0.20, '#95a5a6', 9.8),
    ('LONG', 0.75, '#2ecc71', 9.3),
    ('SHORT', 0.05, '#e74c3c', 8.8)
]

for label, prob, color, y_pos in classes:
    # Bar
    bar_width = prob * 1.5
    bar = Rectangle((11.7, y_pos-0.1), bar_width, 0.2, 
                    facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(bar)
    
    # Label and probability
    ax.text(11.65, y_pos, label, ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(11.7 + bar_width + 0.05, y_pos, f'{prob:.0%}', ha='left', va='center', 
           fontsize=9, fontweight='bold', color=color)

ax.text(12.5, 8.6, 'Prediction: LONG', ha='center', va='center', 
        fontsize=11, fontweight='bold', color='#27ae60')

# ===== BOTTOM: KEY PARAMETERS =====
params_box = FancyBboxPatch((0.5, 0.5), 13, 6.5, boxstyle="round,pad=0.15", 
                            edgecolor='#34495e', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(params_box)

ax.text(7, 6.7, 'XGBOOST PARAMETERS VA TRAINING PROCESS', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#2c3e50')

# Left column: Parameters
ax.text(1, 6.2, 'MODEL PARAMETERS:', ha='left', va='center', 
        fontsize=11, fontweight='bold', color='#2c3e50')

params = [
    'n_estimators = 200',
    'max_depth = 6',
    'learning_rate = 0.05',
    'subsample = 0.8',
    'colsample_bytree = 0.8',
    'objective = multi:softprob',
    'num_class = 3'
]

y_pos = 5.8
for param in params:
    ax.text(1.2, y_pos, f'- {param}', ha='left', va='center', 
           fontsize=9, family='monospace', color='#34495e')
    y_pos -= 0.25

# Middle column: Training Process
ax.text(5, 6.2, 'TRAINING PROCESS:', ha='left', va='center', 
        fontsize=11, fontweight='bold', color='#2c3e50')

training_steps = [
    '1. Tao labels tu future returns',
    '   - Return > +0.8% -> LONG',
    '   - Return < -0.8% -> SHORT',
    '   - Else -> HOLD',
    '',
    '2. Normalize features (StandardScaler)',
    '',
    '3. Train XGBoost:',
    '   - Tree 1: Hoc tu data',
    '   - Tree 2: Hoc tu loi cua Tree 1',
    '   - Tree 3: Hoc tu loi cua Tree 2',
    '   - ... (200 trees)',
    '',
    '4. Validation:',
    '   - Walk-forward validation',
    '   - Time-series split'
]

y_pos = 5.8
for step in training_steps:
    ax.text(5.2, y_pos, step, ha='left', va='center', 
           fontsize=8, family='monospace', color='#34495e')
    y_pos -= 0.2

# Right column: Prediction Process
ax.text(9.5, 6.2, 'PREDICTION PROCESS:', ha='left', va='center', 
        fontsize=11, fontweight='bold', color='#2c3e50')

prediction_steps = [
    '1. Tinh 100+ features',
    '',
    '2. Normalize features',
    '',
    '3. Chay qua 200 trees:',
    '   - Moi tree vote',
    '   - Aggregate votes',
    '',
    '4. Softmax -> Probabilities:',
    '   P(HOLD) = 0.20',
    '   P(LONG) = 0.75',
    '   P(SHORT) = 0.05',
    '',
    '5. Chon class co prob cao nhat:',
    '   -> LONG (75%)',
    '',
    '6. Return:',
    '   - Signal: LONG',
    '   - Confidence: 75%'
]

y_pos = 5.8
for step in prediction_steps:
    ax.text(9.7, y_pos, step, ha='left', va='center', 
           fontsize=8, family='monospace', color='#34495e')
    y_pos -= 0.2

# Bottom highlight
highlight_box = FancyBboxPatch((1, 0.7), 12, 0.6, boxstyle="round,pad=0.1", 
                              edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(highlight_box)
ax.text(7, 1, '>> GRADIENT BOOSTING: Moi tree hoc tu SAI SO cua tree truoc -> Ensemble manh hon', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='#c0392b')

plt.tight_layout()
plt.savefig('visualization/xgboost_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/xgboost_architecture.png")

# ============================================================================
# CHART 2: DECISION TREE EXAMPLE
# ============================================================================

fig2 = plt.figure(figsize=(16, 10))
ax = fig2.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(8, 9.5, 'VI DU: MOT DECISION TREE TRONG XGBOOST', 
        ha='center', va='top', fontsize=18, fontweight='bold')
ax.text(8, 9, 'Cach tree quyet dinh LONG/SHORT/HOLD dua tren features', 
        ha='center', va='top', fontsize=13, style='italic', color='#7f8c8d')

# Root node
root_box = FancyBboxPatch((6.5, 7.5), 3, 0.8, boxstyle="round,pad=0.1", 
                          edgecolor='#3498db', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(root_box)
ax.text(8, 8.1, 'mom_24 > 0.02?', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8, 7.8, '(Momentum 24h > 2%?)', ha='center', va='center', fontsize=8, color='#7f8c8d')

# Left branch (Yes)
ax.annotate('', xy=(4.5, 6.5), xytext=(6.8, 7.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='#2ecc71'))
ax.text(5.5, 7, 'Yes', ha='center', va='center', fontsize=10, fontweight='bold', color='#2ecc71')

left_box = FancyBboxPatch((3, 5.7), 3, 0.8, boxstyle="round,pad=0.1", 
                         edgecolor='#2ecc71', facecolor='#d5f4e6', linewidth=2)
ax.add_patch(left_box)
ax.text(4.5, 6.3, 'rsi_norm > 0.2?', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(4.5, 6, '(RSI bullish?)', ha='center', va='center', fontsize=8, color='#7f8c8d')

# Right branch (No)
ax.annotate('', xy=(11.5, 6.5), xytext=(9.2, 7.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
ax.text(10.5, 7, 'No', ha='center', va='center', fontsize=10, fontweight='bold', color='#e74c3c')

right_box = FancyBboxPatch((10, 5.7), 3, 0.8, boxstyle="round,pad=0.1", 
                          edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(right_box)
ax.text(11.5, 6.3, 'adx > 20?', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(11.5, 6, '(Trend strong?)', ha='center', va='center', fontsize=8, color='#7f8c8d')

# Level 2 - Left side
ax.annotate('', xy=(2.5, 4.5), xytext=(3.5, 5.7),
           arrowprops=dict(arrowstyle='->', lw=1.5, color='#2ecc71'))
ax.text(2.8, 5.1, 'Yes', ha='center', va='center', fontsize=9, color='#2ecc71')

leaf1 = FancyBboxPatch((1.5, 4), 2, 0.6, boxstyle="round,pad=0.08", 
                       edgecolor='#27ae60', facecolor='#abebc6', linewidth=2)
ax.add_patch(leaf1)
ax.text(2.5, 4.3, 'LONG', ha='center', va='center', fontsize=12, fontweight='bold', color='#27ae60')

ax.annotate('', xy=(5.5, 4.5), xytext=(4.5, 5.7),
           arrowprops=dict(arrowstyle='->', lw=1.5, color='#f39c12'))
ax.text(5.2, 5.1, 'No', ha='center', va='center', fontsize=9, color='#f39c12')

leaf2 = FancyBboxPatch((4.5, 4), 2, 0.6, boxstyle="round,pad=0.08", 
                       edgecolor='#d68910', facecolor='#fef5e7', linewidth=2)
ax.add_patch(leaf2)
ax.text(5.5, 4.3, 'HOLD', ha='center', va='center', fontsize=12, fontweight='bold', color='#d68910')

# Level 2 - Right side
ax.annotate('', xy=(9.5, 4.5), xytext=(10.5, 5.7),
           arrowprops=dict(arrowstyle='->', lw=1.5, color='#e74c3c'))
ax.text(9.8, 5.1, 'Yes', ha='center', va='center', fontsize=9, color='#e74c3c')

leaf3 = FancyBboxPatch((8.5, 4), 2, 0.6, boxstyle="round,pad=0.08", 
                       edgecolor='#c0392b', facecolor='#fadbd8', linewidth=2)
ax.add_patch(leaf3)
ax.text(9.5, 4.3, 'SHORT', ha='center', va='center', fontsize=12, fontweight='bold', color='#c0392b')

ax.annotate('', xy=(12.5, 4.5), xytext=(11.5, 5.7),
           arrowprops=dict(arrowstyle='->', lw=1.5, color='#95a5a6'))
ax.text(12.2, 5.1, 'No', ha='center', va='center', fontsize=9, color='#95a5a6')

leaf4 = FancyBboxPatch((11.5, 4), 2, 0.6, boxstyle="round,pad=0.08", 
                       edgecolor='#7f8c8d', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(leaf4)
ax.text(12.5, 4.3, 'HOLD', ha='center', va='center', fontsize=12, fontweight='bold', color='#7f8c8d')

# Example path
example_box = FancyBboxPatch((1, 0.5), 14, 2.8, boxstyle="round,pad=0.15", 
                            edgecolor='#34495e', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(example_box)

ax.text(8, 3, 'VI DU PREDICTION:', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='#2c3e50')

example_text = """
Input Features:
- mom_24 = 0.025 (2.5% up in 24h)
- rsi_norm = 0.30 (RSI = 65, bullish)
- adx = 28 (strong trend)

Decision Path:
1. mom_24 > 0.02? -> YES (0.025 > 0.02) -> Go LEFT
2. rsi_norm > 0.2? -> YES (0.30 > 0.2) -> Go LEFT
3. Reach leaf: LONG

Output: This tree votes for LONG

Note: Day chi la 1 trong 200 trees. Ket qua cuoi cung la tong hop vote cua tat ca trees.
"""

ax.text(1.5, 2.5, example_text, ha='left', va='top', 
        fontsize=9, family='monospace', color='#2c3e50')

plt.tight_layout()
plt.savefig('visualization/xgboost_tree_example.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/xgboost_tree_example.png")

# ============================================================================
# CHART 3: TRAINING VS PREDICTION
# ============================================================================

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
fig3.suptitle('XGBOOST: TRAINING vs PREDICTION', fontsize=18, fontweight='bold')

# Left: Training
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('TRAINING PHASE', fontsize=14, fontweight='bold', pad=20)

# Training flow
train_steps = [
    ('Historical Data\n15 months', '#3498db', 9),
    ('Calculate 100+ Features', '#9b59b6', 7.5),
    ('Create Labels\n(Future Returns)', '#f39c12', 6),
    ('Train XGBoost\n200 Trees', '#e74c3c', 4.5),
    ('Validate\nWalk-Forward', '#2ecc71', 3),
    ('Save Model', '#34495e', 1.5)
]

for i, (text, color, y_pos) in enumerate(train_steps):
    box = FancyBboxPatch((3, y_pos-0.4), 4, 0.8, boxstyle="round,pad=0.1", 
                        edgecolor=color, facecolor='white', linewidth=2)
    ax1.add_patch(box)
    ax1.text(5, y_pos, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    if i < len(train_steps) - 1:
        ax1.annotate('', xy=(5, train_steps[i+1][2]+0.4), xytext=(5, y_pos-0.4),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

# Right: Prediction
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('PREDICTION PHASE (Real-time)', fontsize=14, fontweight='bold', pad=20)

# Prediction flow
pred_steps = [
    ('New Data\nCurrent Bar', '#3498db', 9),
    ('Calculate 100+ Features', '#9b59b6', 7.5),
    ('Load Model', '#34495e', 6.5),
    ('Normalize Features', '#1abc9c', 5.5),
    ('Run Through 200 Trees', '#e74c3c', 4),
    ('Aggregate Votes', '#f39c12', 2.5),
    ('Output Signal\n+ Confidence', '#2ecc71', 1)
]

for i, (text, color, y_pos) in enumerate(pred_steps):
    box = FancyBboxPatch((3, y_pos-0.4), 4, 0.8, boxstyle="round,pad=0.1", 
                        edgecolor=color, facecolor='white', linewidth=2)
    ax2.add_patch(box)
    ax2.text(5, y_pos, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    if i < len(pred_steps) - 1:
        ax2.annotate('', xy=(5, pred_steps[i+1][2]+0.4), xytext=(5, y_pos-0.4),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

plt.tight_layout()
plt.savefig('visualization/xgboost_train_vs_predict.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/xgboost_train_vs_predict.png")

# ============================================================================
# CHART 4: LABEL CREATION
# ============================================================================

fig4, ax = plt.subplots(figsize=(16, 10))

# Generate sample price data
np.random.seed(42)
n_points = 100
price = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')

# Calculate future returns
threshold = 0.008  # 0.8%
holding_period = 12  # 12 hours

future_returns = np.zeros(n_points)
labels = np.zeros(n_points)

for i in range(n_points - holding_period):
    future_returns[i] = (price[i + holding_period] / price[i]) - 1
    if future_returns[i] > threshold:
        labels[i] = 1  # LONG
    elif future_returns[i] < -threshold:
        labels[i] = -1  # SHORT
    else:
        labels[i] = 0  # HOLD

# Plot
ax.plot(dates, price, color='#2c3e50', linewidth=2, label='Price', zorder=3)

# Color background by label
for i in range(len(dates)-1):
    if labels[i] == 1:  # LONG
        ax.axvspan(dates[i], dates[i+1], alpha=0.3, color='#2ecc71')
    elif labels[i] == -1:  # SHORT
        ax.axvspan(dates[i], dates[i+1], alpha=0.3, color='#e74c3c')
    else:  # HOLD
        ax.axvspan(dates[i], dates[i+1], alpha=0.1, color='#95a5a6')

# Add some example annotations
long_idx = np.where(labels == 1)[0][0]
short_idx = np.where(labels == -1)[0][0]

ax.annotate(f'LONG Label\nFuture return: +{future_returns[long_idx]*100:.1f}%', 
           xy=(dates[long_idx], price[long_idx]), 
           xytext=(dates[long_idx+10], price[long_idx]+5),
           arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
           fontsize=10, fontweight='bold', color='#27ae60',
           bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#2ecc71', linewidth=2))

ax.annotate(f'SHORT Label\nFuture return: {future_returns[short_idx]*100:.1f}%', 
           xy=(dates[short_idx], price[short_idx]), 
           xytext=(dates[short_idx+10], price[short_idx]-5),
           arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
           fontsize=10, fontweight='bold', color='#c0392b',
           bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=2))

ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USDT)', fontsize=13, fontweight='bold')
ax.set_title('LABEL CREATION: Future Returns -> LONG/SHORT/HOLD\nThreshold = 0.8%, Holding Period = 12 hours', 
            fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.3, label='LONG (Future return > +0.8%)'),
    Patch(facecolor='#e74c3c', alpha=0.3, label='SHORT (Future return < -0.8%)'),
    Patch(facecolor='#95a5a6', alpha=0.1, label='HOLD (Future return between -0.8% and +0.8%)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('visualization/xgboost_label_creation.png', dpi=300, bbox_inches='tight', facecolor='white')
print(">>> Da tao: visualization/xgboost_label_creation.png")

print("\n" + "="*70)
print(">>> HOAN THANH! Da tao 4 hinh anh ve XGBoost:")
print("="*70)
print("1. xgboost_architecture.png - Kien truc va parameters")
print("2. xgboost_tree_example.png - Vi du mot decision tree")
print("3. xgboost_train_vs_predict.png - Training vs Prediction flow")
print("4. xgboost_label_creation.png - Cach tao labels tu future returns")
print("="*70)
print("\n>> MEO TRINH BAY:")
print("• Slide 1: Gioi thieu kien truc XGBoost ensemble")
print("• Slide 2: Giai thich cach mot tree quyet dinh")
print("• Slide 3: So sanh quy trinh training vs prediction")
print("• Slide 4: Giai thich cach tao labels (supervised learning)")
print("="*70)
