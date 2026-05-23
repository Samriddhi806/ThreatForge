# src/train_gan.py — Phase 2: CTGAN with Checkpointing
import pandas as pd
import numpy as np
import json, os, time, joblib
import matplotlib.pyplot as plt
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

print("=" * 55)
print("  Phase 2 — CTGAN Training with Checkpointing")
print("=" * 55)

os.makedirs("models",           exist_ok=True)
os.makedirs("models/gan_ckpt",  exist_ok=True)
os.makedirs("data/synthetic",   exist_ok=True)
os.makedirs("logs",             exist_ok=True)

# ── 1. Load attack-only rows ──────────────────────────────
print("\n[1/6] Loading attack-only training data...")
attack_df = pd.read_parquet("data/processed/attack_only.parquet")
print(f"      Attack rows : {len(attack_df):,}")
print(f"      Features    : {attack_df.shape[1]}")

train_full = pd.read_parquet("data/processed/train.parquet")
attack_with_type = train_full[train_full['binary_label'] == 1]
print(f"\n      Class breakdown:")
for cls, cnt in attack_with_type['attack_type'].value_counts().items():
    bar = "█" * (cnt // 15000)
    print(f"      {cls:15s}: {cnt:>7,}  {bar}")

# ── 2. Check for existing checkpoint ─────────────────────
CKPT_DIR    = "models/gan_ckpt"
FINAL_MODEL = "models/ctgan_model.pkl"
CKPT_LOG    = os.path.join(CKPT_DIR, "ckpt_log.json")

TOTAL_EPOCHS     = 300
CHECKPOINT_EVERY = 50   # save every 50 epochs

def find_latest_checkpoint():
    """Return (epoch, path) of latest checkpoint or (0, None)"""
    if not os.path.exists(CKPT_LOG):
        return 0, None
    with open(CKPT_LOG) as f:
        log = json.load(f)
    if not log.get("checkpoints"):
        return 0, None
    latest = log["checkpoints"][-1]
    path = latest["path"]
    if os.path.exists(path):
        print(f"\n  ✅ Found checkpoint at epoch {latest['epoch']}: {path}")
        return latest["epoch"], path
    return 0, None

def save_checkpoint_log(epoch, path):
    log = {}
    if os.path.exists(CKPT_LOG):
        with open(CKPT_LOG) as f:
            log = json.load(f)
    log.setdefault("checkpoints", [])
    log["checkpoints"].append({
        "epoch": epoch,
        "path": path,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    with open(CKPT_LOG, "w") as f:
        json.dump(log, f, indent=2)

# ── 3. Train CTGAN in segments with checkpointing ────────
print("\n[2/6] Setting up CTGAN with checkpoint resume...")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(attack_df)

start_epoch, ckpt_path = find_latest_checkpoint()
remaining_epochs = TOTAL_EPOCHS - start_epoch

if remaining_epochs <= 0:
    print("      Training already complete! Loading final model...")
    gan = CTGANSynthesizer.load(FINAL_MODEL)
else:
    if ckpt_path:
        print(f"      Resuming from epoch {start_epoch}...")
        print(f"      Remaining epochs: {remaining_epochs}")
        gan = CTGANSynthesizer.load(ckpt_path)
        # Re-fit remaining epochs in chunks
    else:
        print(f"      Starting fresh — {TOTAL_EPOCHS} epochs total")
        gan = CTGANSynthesizer(
            metadata,
            epochs            = CHECKPOINT_EVERY,  # train in chunks
            batch_size        = 500,
            generator_dim     = (256, 256),
            discriminator_dim = (256, 256),
            verbose           = True,
            enable_gpu        = True
        )

    # Train in chunks of CHECKPOINT_EVERY epochs
    current_epoch = start_epoch
    t_total = time.time()

    while current_epoch < TOTAL_EPOCHS:
        epochs_this_round = min(CHECKPOINT_EVERY, TOTAL_EPOCHS - current_epoch)
        print(f"\n  ── Training epochs {current_epoch+1}"
              f"–{current_epoch + epochs_this_round} ──")

        t0 = time.time()

        if current_epoch == 0:
            # First run — fit fresh
            gan._epochs = epochs_this_round
            gan.fit(attack_df)
        else:
            # Subsequent runs — update epochs and continue fitting
            gan._model._epochs = epochs_this_round
            gan.fit(attack_df)

        elapsed = time.time() - t0
        current_epoch += epochs_this_round

        # Save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"gan_epoch_{current_epoch}.pkl")
        gan.save(ckpt_path)
        save_checkpoint_log(current_epoch, ckpt_path)

        total_elapsed = time.time() - t_total
        pct = current_epoch / TOTAL_EPOCHS * 100
        eta  = (total_elapsed / current_epoch) * (TOTAL_EPOCHS - current_epoch)

        print(f"  ✅ Checkpoint saved → {ckpt_path}")
        print(f"     Progress : {current_epoch}/{TOTAL_EPOCHS} epochs ({pct:.0f}%)")
        print(f"     Elapsed  : {total_elapsed/60:.1f} min")
        print(f"     ETA      : {eta/60:.1f} min remaining")

    # Save final model
    gan.save(FINAL_MODEL)
    print(f"\n  ✅ Final model saved → {FINAL_MODEL}")

# ── 4. Generate synthetic samples ────────────────────────
print("\n[3/6] Generating synthetic adversarial flows...")
n_real  = len(attack_df)
n_synth = n_real * 2
synthetic = gan.sample(num_rows=n_synth)
print(f"      Generated : {len(synthetic):,} synthetic flows")
synthetic.to_parquet("data/synthetic/synthetic_attacks.parquet", index=False)
print(f"      Saved     → data/synthetic/synthetic_attacks.parquet")

# ── 5. Adversarial validation ─────────────────────────────
print("\n[4/6] Adversarial Validation...")
real_s  = attack_df.sample(n=min(50000, len(attack_df)), random_state=42).copy()
synth_s = synthetic.sample(n=min(50000, len(synthetic)), random_state=42).copy()
real_s['is_real']  = 1
synth_s['is_real'] = 0

combined = pd.concat([real_s, synth_s], ignore_index=True)
X_adv    = combined.drop(columns=['is_real'])
y_adv    = combined['is_real']

X_tr, X_te, y_tr, y_te = train_test_split(
    X_adv, y_adv, test_size=0.3, stratify=y_adv, random_state=42)

clf     = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_tr, y_tr)
adv_auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

if   adv_auc < 0.60: verdict = "EXCELLENT ✅"
elif adv_auc < 0.65: verdict = "GOOD ✅"
elif adv_auc < 0.72: verdict = "ACCEPTABLE ⚠️ — consider more epochs"
else:                verdict = "POOR ❌ — retrain with more epochs"

print(f"      Adversarial AUC : {adv_auc:.4f}  →  {verdict}")

# ── 6. KS-test + build augmented dataset ─────────────────
print("\n[5/6] KS-test statistical check...")
feature_cols = attack_df.columns.tolist()
failed = []
for col in feature_cols:
    stat, _ = stats.ks_2samp(attack_df[col].values, synthetic[col].values)
    if stat > 0.2:
        failed.append((col, round(stat, 3)))

passed = len(feature_cols) - len(failed)
print(f"      KS pass rate: {passed}/{len(feature_cols)} features")

print("\n[6/6] Building augmented training set...")
real_train  = pd.read_parquet("data/processed/train.parquet")
synth_clean = synthetic.copy()
synth_clean['binary_label'] = 1
synth_clean['attack_type']  = 'Synthetic-Attack'
synth_clean = synth_clean.reindex(columns=real_train.columns, fill_value=0)

augmented = pd.concat([real_train, synth_clean], ignore_index=True)
augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)
augmented.to_parquet("data/processed/augmented_train.parquet", index=False)

results = {
    "adversarial_auc": round(adv_auc, 4),
    "verdict":         verdict,
    "n_real":          n_real,
    "n_synthetic":     n_synth,
    "ks_pass_rate":    f"{passed}/{len(feature_cols)}",
}
with open("logs/gan_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"""
╔══════════════════════════════════════════════════╗
║  Phase 2 Complete                                ║
╠══════════════════════════════════════════════════╣
║  Adversarial AUC : {adv_auc:.4f}                       ║
║  KS Pass Rate    : {passed}/{len(feature_cols)} features              ║
║  Augmented rows  : {len(augmented):,}                ║
╚══════════════════════════════════════════════════╝
  Checkpoints saved in: models/gan_ckpt/
  Next: python src/train_robust_classifier.py
""")