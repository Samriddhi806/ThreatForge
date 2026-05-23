# src/baseline.py — Phase 1 Final: XGBoost Baseline Classifier
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, ConfusionMatrixDisplay)
import time, os

print("=" * 55)
print("  Phase 1 — XGBoost Baseline Classifier")
print("=" * 55)

# ── 1. Load splits ────────────────────────────────────────
print("\n[1/5] Loading train/val/test splits...")
train = pd.read_parquet("data/processed/train.parquet")
val   = pd.read_parquet("data/processed/val.parquet")
test  = pd.read_parquet("data/processed/test.parquet")

feature_cols = [c for c in train.columns
                if c not in ['binary_label', 'attack_type']]

X_train = train[feature_cols].values
y_train = train['binary_label'].values

X_val   = val[feature_cols].values
y_val   = val['binary_label'].values

X_test  = test[feature_cols].values
y_test  = test['binary_label'].values

print(f"      Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"      Features: {len(feature_cols)}")

# ── 2. Train baseline XGBoost ─────────────────────────────
print("\n[2/5] Training XGBoost baseline (this may take 3-5 mins)...")
t0 = time.time()

model = xgb.XGBClassifier(
    n_estimators     = 300,
    max_depth        = 7,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder= False,
    eval_metric      = 'logloss',
    random_state     = 42,
    n_jobs           = -1,     # use all CPU cores
    early_stopping_rounds = 20,
    verbosity        = 1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

elapsed = time.time() - t0
print(f"\n      Training complete in {elapsed:.1f}s")

# ── 3. Evaluate on test set ───────────────────────────────
print("\n[3/5] Evaluating on held-out test set...")
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
tn  = ((y_pred == 0) & (y_test == 0)).sum()
fp  = ((y_pred == 1) & (y_test == 0)).sum()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\n  ── Baseline Metrics (RECORD THESE) ──────────────")
print(f"  F1 Score       : {f1:.4f}")
print(f"  ROC-AUC        : {auc:.4f}")
print(f"  False Pos Rate : {fpr:.4f}  ({fp:,} false alarms)")
print(f"  ─────────────────────────────────────────────────")
print(f"\n{classification_report(y_test, y_pred, target_names=['Benign','Attack'])}")

# Per-class F1 on multiclass
print("\n  ── Per Attack Class F1 ──────────────────────────")
test_multi = test.copy()
test_multi['pred'] = y_pred
for cls in test['attack_type'].unique():
    subset = test_multi[test_multi['attack_type'] == cls]
    if len(subset) == 0: continue
    correct = (subset['pred'] == subset['binary_label']).sum()
    acc = correct / len(subset) * 100
    bar = "█" * int(acc // 5)
    print(f"  {cls:15s}: {acc:5.1f}%  {bar}")

# ── 4. Save model and results ─────────────────────────────
print("\n[4/5] Saving model and plots...")
os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

joblib.dump(model, "models/baseline_xgb.pkl")
print("      Saved → models/baseline_xgb.pkl")

# Save metrics to file (we'll compare with GAN-augmented later)
metrics = {
    "model": "baseline",
    "f1":    round(f1, 4),
    "auc":   round(auc, 4),
    "fpr":   round(fpr, 4),
    "n_train": len(X_train),
}
import json
with open("logs/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("      Saved → logs/baseline_metrics.json")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Benign", "Attack"],
    cmap="Blues", ax=ax
)
ax.set_title("Baseline XGBoost — Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("logs/baseline_confusion_matrix.png", dpi=120)
plt.show()
print("      Saved → logs/baseline_confusion_matrix.png")

# Feature importance plot (top 20)
fig2, ax2 = plt.subplots(figsize=(8, 6))
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.nlargest(20).sort_values().plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_title("Top 20 Feature Importances — Baseline XGBoost")
ax2.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("logs/baseline_feature_importance.png", dpi=120)
plt.show()
print("      Saved → logs/baseline_feature_importance.png")

# ── 5. Summary ────────────────────────────────────────────
print("\n[5/5] ✅ Baseline complete!")
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  BASELINE SCORE TO BEAT WITH GAN        │")
print(f"  │  F1  : {f1:.4f}                           │")
print(f"  │  AUC : {auc:.4f}                           │")
print(f"  │  FPR : {fpr:.4f}                           │")
print(f"  └─────────────────────────────────────────┘")
print("\n  Next: run python src/train_gan.py  (Phase 2)")