import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# 1. Load Data
print("Loading data...")
df = pd.read_csv(r"E:\ML\predictive_maintenance_dataset.csv")
df['date'] = pd.to_datetime(df['date'])

# 2. Sort by device and date to prepare for time-series features
df = df.sort_values(by=['device', 'date']).reset_index(drop=True)

# 3. Comprehensive Time-Series Feature Engineering
# Create rolling averages and change metrics for ALL metrics to capture degradation patterns
print("Engineering time-series features (Rolling averages, changes, and trends)...")
all_metrics = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric9']

for metric in all_metrics:
    # Rolling averages (3-day and 7-day windows)
    df[f'{metric}_rolling_3d'] = df.groupby('device')[metric].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df[f'{metric}_rolling_7d'] = df.groupby('device')[metric].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Change metrics (1-day and 3-day deltas)
    df[f'{metric}_change_1d'] = df.groupby('device')[metric].transform(lambda x: x.diff().fillna(0))
    df[f'{metric}_change_pct_1d'] = df.groupby('device')[metric].transform(lambda x: x.pct_change().fillna(0))
    
    # Trend: is the metric going up or down over the last 3 days
    df[f'{metric}_trend_3d'] = df.groupby('device')[metric].transform(
        lambda x: (x.rolling(window=3, min_periods=1).mean().diff().fillna(0) > 0).astype(int)
    )

# Build complete feature list: base metrics + all engineered features
all_features = [col for col in df.columns if col.startswith('metric') and col != 'metric8']
X = df[all_features]
y = df['failure']

# Handle NaN and infinity values
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# 4. Stratified Split
print(f"Total features created: {len(all_features)}")
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Apply SMOTE to balance training data
print("Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Original training set - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"Resampled training set - Class 0: {(y_train_resampled == 0).sum()}, Class 1: {(y_train_resampled == 1).sum()}")

# 6. Train Model with SMOTE-balanced data
print("\nTraining Random Forest with SMOTE-balanced data (this may take a moment)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)

# 7. Evaluate with Optimal Threshold
y_prob = rf.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold: {optimal_threshold:.4f} (F1-Score: {f1_scores[optimal_idx]:.4f})")

y_pred = rf.predict(X_test)
y_pred_opt = (y_prob >= optimal_threshold).astype(int)

print("\n--- Model Evaluation (Default 50% Threshold) ---")
print(classification_report(y_test, y_pred))

print("\n--- Model Evaluation (Optimal Threshold) ---")
print(classification_report(y_test, y_pred_opt))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# 8. Visualizations
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix with SMOTE + Time-Series')
plt.xlabel('Predicted Label (0 = Healthy, 1 = Failure)')
plt.ylabel('True Label (0 = Healthy, 1 = Failure)')
plt.show()

plt.figure(figsize=(10, 8))
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features
top_features = [all_features[i] for i in indices]
plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
plt.yticks(range(len(indices)), top_features)
plt.xlabel('Relative Importance')
plt.title('Top 20 Features Driving Failure Predictions')
plt.tight_layout()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
optimal_recall = recall[optimal_idx]
optimal_precision = precision[optimal_idx]
plt.plot(optimal_recall, optimal_precision, 'go', markersize=10, label=f'Optimal ({optimal_threshold:.4f})')
plt.xlabel('Recall (Caught Failures)')
plt.ylabel('Precision (True Alarm Rate)')
plt.title('Precision-Recall Curve (SMOTE + Time-Series Model)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()