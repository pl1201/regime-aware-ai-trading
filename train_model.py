import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
X = pd.read_csv('datasets/X_features.csv')
y = pd.read_csv('datasets/y_labels.csv')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'models/dynamic_moe_v2_enhanced_final_v2.pkl')

print("Model trained and saved as 'models/dynamic_moe_v2_enhanced_final_v2.pkl'")