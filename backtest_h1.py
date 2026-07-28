import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load
df = pd.read_csv("data/okx_1h.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df.index = df.index.tz_localize(None)

# Features
data = df.copy()
for p in [1, 4, 8, 12, 24]:
    data[f'mom_{p}'] = data['close'].pct_change(p)
data['ema_cross'] = (data['close'].ewm(9).mean() - data['close'].ewm(21).mean()) / data['close']
delta = data['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
data['atr'] = (data['high'] - data['low']).rolling(14).mean() / data['close']

# Labels
future = data['close'].shift(-6) / data['close'] - 1
data['label'] = np.where(future > 0.006, 1, np.where(future < -0.006, 2, 0))
data = data.dropna()

feats = ['mom_1', 'mom_4', 'mom_8', 'mom_12', 'mom_24', 'ema_cross', 'rsi', 'atr']

# Split
train = data.loc[:'2024-12-31']
test = data.loc['2025-01-01':]
print(f"Train: {len(train)}, Test: {len(test)}")

# Train
model = GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_leaf=50, random_state=42)
model.fit(train[feats], train['label'])
print(f"Train acc: {model.score(train[feats], train['label']):.1%}")

# Backtest
pred = model.predict(test[feats])
proba = model.predict_proba(test[feats]).max(axis=1)

print("\nBACKTEST RESULTS:")
for conf in [0.35, 0.40, 0.45, 0.50]:
    signals = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
    signals[proba < conf] = 0
    
    rets = []
    for i in range(len(signals) - 6):
        if signals[i] == 0: continue
        ret = (test['close'].iloc[i+6] / test['close'].iloc[i] - 1) * signals[i] - 0.001
        rets.append(ret)
    
    rets = np.array(rets)
    wins = rets > 0
    pf = rets[wins].sum() / abs(rets[~wins].sum()) if len(rets) > 0 and (~wins).any() else 0
    wr = wins.mean() if len(rets) > 0 else 0
    print(f"Conf>={conf}: {len(rets)} trades, WR={wr:.1%}, PF={pf:.2f}, Ret={rets.sum()*100:+.1f}%")
