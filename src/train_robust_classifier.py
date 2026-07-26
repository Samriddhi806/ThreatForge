# Phase 3: Robust XGBoost + SHAP
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import json, os, time
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score,
                              ConfusionMatrixDisplay)

print("=" * 55)
print("  Phase 3 — Robust Classifier + SHAP Explainability")
print("=" * 55)

os.makedirs("models",     exist_ok=True)
os.makedirs("logs",       exist_ok=True)
os.makedirs("logs/shap",  exist_ok=True)

# 1. Load augmented training data
print("\n[1/6] Loading augmented training data...")

augmented = pd.read_parquet("data/processed/augmented_train.parquet")
val       = pd.read_parquet("data/processed/val.parquet")
test      = pd.read_parquet("data/processed/test.parquet")

feature_cols = [c for c in augmented.columns
                if c not in ['binary_label', 'attack_type']]

X_train = augmented[feature_cols].values
y_train = augmented['binary_label'].values

X_val   = val[feature_cols].values
y_val   = val['binary_label'].values

X_test  = test[feature_cols].values
y_test  = test['binary_label'].values

print(f"      Augmented train : {X_train.shape}")
print(f"      Val             : {X_val.shape}")
print(f"      Test            : {X_test.shape}")
print(f"      Features        : {len(feature_cols)}")

# Show class balance in augmented set
real_count  = (augmented['attack_type'] != 'Synthetic-Attack').sum()
synth_count = (augmented['attack_type'] == 'Synthetic-Attack').sum()
print(f"\n      Real rows      : {real_count:,}")
print(f"      Synthetic rows : {synth_count:,}")
print(f"      Total          : {len(augmented):,}")

# 2. Train robust XGBoost 
print("\n[2/6] Training robust XGBoost on augmented data...")
print("      This may take 5–10 minutes...\n")

t0 = time.time()

model = xgb.XGBClassifier(
    n_estimators          = 500,
    max_depth             = 7,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    gamma                 = 0.1,
    use_label_encoder     = False,
    eval_metric           = 'logloss',
    random_state          = 42,
    n_jobs                = -1,
    early_stopping_rounds = 30,
    verbosity             = 1
)

model.fit(
    X_train, y_train,
    eval_set    = [(X_val, y_val)],
    verbose     = 50
)

elapsed = time.time() - t0
print(f"\n      Training complete in {elapsed:.1f}s")

# 3. Evaluate on test set
print("\n[3/6] Evaluating on held-out test set...")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
tn  = ((y_pred == 0) & (y_test == 0)).sum()
fp  = ((y_pred == 1) & (y_test == 0)).sum()
fn  = ((y_pred == 0) & (y_test == 1)).sum()
tp  = ((y_pred == 1) & (y_test == 1)).sum()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# Load baseline metrics to compare
baseline_f1  = 0.913
baseline_fpr = 0.731
try:
    with open("logs/baseline_metrics.json") as f:
        bm = json.load(f)
        baseline_f1  = bm.get("f1",  0.913)
        baseline_fpr = bm.get("fpr", 0.731)
except:
    pass

print(f"\n  Results Comparison: ")
print(f"  Metric      Baseline    GAN-Augmented   Change")
print(f"  ─────────────────────────────────────────────────")
print(f"  F1 Score    {baseline_f1:.4f}      {f1:.4f}          "
      f"{'▲' if f1 > baseline_f1 else '▼'} {abs(f1-baseline_f1):.4f}")
print(f"  FPR         {baseline_fpr:.4f}      {fpr:.4f}          "
      f"{'▼' if fpr < baseline_fpr else '▲'} {abs(fpr-baseline_fpr):.4f}")
print(f"  ROC-AUC     —           {auc:.4f}")
print(f"  ─────────────────────────────────────────────────")
print(f"  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Benign','Attack'])}")

# Per-class accuracy
print(" Per attack class accuracy:")
test_copy = test.copy()
test_copy['pred'] = y_pred
for cls in sorted(test['attack_type'].unique()):
    subset = test_copy[test_copy['attack_type'] == cls]
    if len(subset) == 0: continue
    acc = (subset['pred'] == subset['binary_label']).mean() * 100
    # bar = "█" * int(acc // 5)
    print(f"  {cls:15s}: {acc:5.1f}%  ")

# 4. Save model and metrics 
print("\n Saving model and metrics...")

joblib.dump(model, "models/model_final.pkl")
print("      Saved → models/model_final.pkl")

metrics = {
    "model":          "gan_augmented",
    "f1":             round(f1, 4),
    "auc":            round(auc, 4),
    "fpr":            round(fpr, 4),
    "tp":             int(tp),
    "fp":             int(fp),
    "fn":             int(fn),
    "tn":             int(tn),
    "n_train":        len(X_train),
    "baseline_f1":    baseline_f1,
    "baseline_fpr":   baseline_fpr,
    "f1_improvement": round(f1 - baseline_f1, 4),
    "fpr_reduction":  round(baseline_fpr - fpr, 4),
}
with open("logs/robust_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("      Saved → logs/robust_metrics.json")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Benign", "Attack"],
    cmap="Blues", ax=ax
)
ax.set_title("GAN-Augmented XGBoost — Confusion Matrix")
plt.tight_layout()
plt.savefig("logs/robust_confusion_matrix.png", dpi=120)
plt.close()
print("      Saved → logs/robust_confusion_matrix.png")

# Feature importance
fig2, ax2 = plt.subplots(figsize=(8, 6))
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.nlargest(20).sort_values().plot(
    kind='barh', ax=ax2, color='steelblue')
ax2.set_title("Top 20 Feature Importances — GAN-Augmented Model")
ax2.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("logs/robust_feature_importance.png", dpi=120)
plt.close()
print("      Saved → logs/robust_feature_importance.png")

# 5. SHAP explainability 
print("\n[5/6] Computing SHAP explanations (sample of 2000 rows)...")

sample_idx  = np.random.choice(len(X_test), size=2000, replace=False)
X_shap      = X_test[sample_idx]
shap_labels = y_test[sample_idx]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# Global summary plot
plt.figure()
shap.summary_plot(
    shap_values, X_shap,
    feature_names=feature_cols,
    show=False, max_display=20
)
plt.tight_layout()
plt.savefig("logs/shap/shap_summary.png", dpi=120, bbox_inches='tight')
plt.close()
print("      Saved → logs/shap/shap_summary.png")

# Save SHAP values for dashboard use
np.save("logs/shap/shap_values.npy",    shap_values)
np.save("logs/shap/shap_X_sample.npy",  X_shap)
joblib.dump(feature_cols, "logs/shap/feature_cols.pkl")
print("      Saved → logs/shap/shap_values.npy")
print("      Saved → logs/shap/shap_X_sample.npy")

# Per-alert SHAP waterfall for first 5 attack detections
print("\n      Generating per-alert SHAP waterfall plots...")
attack_indices = np.where(
    (model.predict(X_shap) == 1) & (shap_labels == 1)
)[0][:5]

for i, idx in enumerate(attack_indices):
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = explainer.expected_value,
            data          = X_shap[idx],
            feature_names = feature_cols
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"logs/shap/alert_{i+1}_waterfall.png",
                dpi=120, bbox_inches='tight')
    plt.close()

print(f"      Saved {len(attack_indices)} waterfall plots → logs/shap/")

# 6. Summary 
print("\n[6/6] Phase 3 classifier complete!")
print(f"""

  ARIA — Phase 3 Results                              
-------------------------------------------------------------------------------------------------
  F1  : {baseline_f1:.4f} → {f1:.4f}  ({'▲ improved' if f1>baseline_f1 else '▼ check GAN'})               
  FPR : {baseline_fpr:.4f} → {fpr:.4f}  ({'▼ improved' if fpr<baseline_fpr else '▲ check GAN'})               
  AUC : {auc:.4f}                                      

  Next: python src/llm_rule_gen.py
""")