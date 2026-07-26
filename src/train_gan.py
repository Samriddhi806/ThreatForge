#Phase 2: CTGAN (bulletproof version)
import pandas as pd
import numpy as np
import json, os, time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

print("=" * 55)
print("  Phase 2 — CTGAN Training (bulletproof version)")
print("=" * 55)

os.makedirs("models",          exist_ok=True)
os.makedirs("models/gan_ckpt", exist_ok=True)
os.makedirs("data/synthetic",  exist_ok=True)
os.makedirs("logs",            exist_ok=True)

# 1. Load and sample attack rows 
print("\n[1/6] Loading and sampling attack data...")

train_full = pd.read_parquet("data/processed/train.parquet")
attack_df  = train_full[train_full['binary_label'] == 1].copy()
feature_cols = [c for c in attack_df.columns
                if c not in ['binary_label', 'attack_type']]
attack_df  = attack_df[feature_cols].reset_index(drop=True)

print(f"      Full attack rows : {len(attack_df):,}")

# Sample 100K rows — sufficient for CTGAN
attack_sample = attack_df.sample(
    n=min(100_000, len(attack_df)),
    random_state=42
).reset_index(drop=True)

print(f"      Sample size      : {len(attack_sample):,}")
print(f"      Features         : {attack_sample.shape[1]}")

# Ensure all columns are float32 — avoids metadata issues
attack_sample = attack_sample.astype('float32')

# 2. Setup CTGAN 
print("\n[2/6] Setting up CTGAN...")

import sdv
print(f"      SDV version: {sdv.__version__}")

TOTAL_EPOCHS     = 300
CHECKPOINT_EVERY = 50
CKPT_DIR         = "models/gan_ckpt"
CKPT_LOG         = os.path.join(CKPT_DIR, "ckpt_log.json")
FINAL_MODEL      = "models/ctgan_model.pkl"

def find_latest_checkpoint():
    if not os.path.exists(CKPT_LOG):
        return 0, None
    with open(CKPT_LOG) as f:
        log = json.load(f)
    if not log.get("checkpoints"):
        return 0, None
    latest = log["checkpoints"][-1]
    if os.path.exists(latest["path"]):
        print(f"      Resuming from epoch {latest['epoch']}")
        return latest["epoch"], latest["path"]
    return 0, None

def save_ckpt_log(epoch, path):
    log = {}
    if os.path.exists(CKPT_LOG):
        with open(CKPT_LOG) as f:
            log = json.load(f)
    log.setdefault("checkpoints", [])
    log["checkpoints"].append({
        "epoch": epoch, "path": path,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    with open(CKPT_LOG, "w") as f:
        json.dump(log, f, indent=2)

# ── Build metadata safely ─────────────────────────────────
from sdv.metadata import SingleTableMetadata

def build_metadata(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    metadata.validate()
    return metadata

# ── Train CTGAN ───────────────────────────────────────────
start_epoch, ckpt_path = find_latest_checkpoint()
remaining = TOTAL_EPOCHS - start_epoch

if remaining <= 0:
    print("      Already complete — loading final model")
    from sdv.single_table import CTGANSynthesizer
    gan = CTGANSynthesizer.load(FINAL_MODEL)
else:
    from sdv.single_table import CTGANSynthesizer

    if ckpt_path:
        gan = CTGANSynthesizer.load(ckpt_path)
    else:
        print(f"      Starting fresh — {TOTAL_EPOCHS} epochs")

        try:
            meta = build_metadata(attack_sample)
            gan = CTGANSynthesizer(
                meta,
                epochs            = CHECKPOINT_EVERY,
                batch_size        = 500,
                generator_dim     = (256, 256),
                discriminator_dim = (256, 256),
                verbose           = True,
                cuda              = False
            )
        except Exception as e:
            print(f"      Metadata approach failed: {e}")
            print("      Trying direct CTGAN (no metadata)...")

            # Last resort — use ctgan library directly
           

    current_epoch = start_epoch
    t_total       = time.time()
    first_run     = (ckpt_path is None)

    # Check if we fell back to raw ctgan
    if 'CTGAN' in str(type(gan)) and 'sdv' not in str(type(gan)):
        # Raw ctgan library fallback
        print("\n  Using raw CTGAN library (no SDV wrapper)...")
        from ctgan import CTGAN

        gan = CTGAN(
            epochs     = TOTAL_EPOCHS,
            batch_size = 500,
            verbose    = True
        )
        print(f"\n  ── Training all {TOTAL_EPOCHS} epochs at once ──")
        t0 = time.time()
        gan.fit(attack_sample, discrete_columns=[])
        elapsed = time.time() - t0
        print(f"\n  Training complete in {elapsed/60:.1f} min")

        import pickle
        with open(FINAL_MODEL, 'wb') as f:
            pickle.dump(gan, f)
        print(f"  Saved → {FINAL_MODEL}")

    else:
        while current_epoch < TOTAL_EPOCHS:
            chunk = min(CHECKPOINT_EVERY, TOTAL_EPOCHS - current_epoch)
            print(f"\n  ── Epochs {current_epoch+1}–{current_epoch+chunk} ──")

            t0 = time.time()
            if first_run:
                gan.fit(attack_sample)
                first_run = False
            else:
                try:
                    gan._model._epochs = chunk
                    gan.fit(attack_sample)
                except Exception:
                    gan.fit(attack_sample)

            elapsed       = time.time() - t0
            current_epoch += chunk
            total_elapsed  = time.time() - t_total
            pct            = current_epoch / TOTAL_EPOCHS * 100
            eta            = (total_elapsed / current_epoch) * \
                             (TOTAL_EPOCHS - current_epoch)

            path = os.path.join(CKPT_DIR, f"gan_epoch_{current_epoch}.pkl")
            gan.save(path)
            save_ckpt_log(current_epoch, path)

            print(f"\n  ✅ Checkpoint → {path}")
            print(f"     Progress : {current_epoch}/{TOTAL_EPOCHS} ({pct:.0f}%)")
            print(f"     Elapsed  : {total_elapsed/60:.1f} min")
            print(f"     ETA      : {eta/60:.1f} min remaining")

        gan.save(FINAL_MODEL)
        print(f"\n  ✅ Final model → {FINAL_MODEL}")

# 3. Generate synthetic samples 
print("\n[3/6] Generating synthetic flows...")

try:
    synthetic = gan.sample(num_rows=200_000)
except Exception:
    import pickle
    with open(FINAL_MODEL, 'rb') as f:
        gan = pickle.load(f)
    synthetic = gan.sample(200_000)

print(f"      Generated : {len(synthetic):,} rows")
synthetic.to_parquet("data/synthetic/synthetic_attacks.parquet", index=False)
print(f"      Saved     → data/synthetic/synthetic_attacks.parquet")

# 4. Adversarial validation 
print("\n[4/6] Adversarial validation...")

real_s  = attack_df.sample(n=min(25000, len(attack_df)),
                            random_state=42).copy()
synth_s = synthetic.sample(n=min(25000, len(synthetic)),
                            random_state=42).copy()

# Align columns
common_cols   = [c for c in real_s.columns if c in synth_s.columns]
real_s        = real_s[common_cols]
synth_s       = synth_s[common_cols]

real_s['is_real']  = 1
synth_s['is_real'] = 0

combined = pd.concat([real_s, synth_s], ignore_index=True).fillna(0)
X_adv    = combined.drop(columns=['is_real'])
y_adv    = combined['is_real']

X_tr, X_te, y_tr, y_te = train_test_split(
    X_adv, y_adv, test_size=0.3, stratify=y_adv, random_state=42)

clf     = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_tr, y_tr)
adv_auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

if   adv_auc < 0.60: verdict = "EXCELLENT "
elif adv_auc < 0.65: verdict = "GOOD "
elif adv_auc < 0.72: verdict = "ACCEPTABLE "
else:                verdict = "POOR "

print(f"      Adversarial AUC : {adv_auc:.4f}  →  {verdict}")

# 5. KS-test
print("\n[5/6] KS-test per feature...")

failed = []
for col in common_cols:
    stat, _ = stats.ks_2samp(
        attack_df[col].values,
        synthetic[col].values
    )
    if stat > 0.2:
        failed.append((col, round(stat, 3)))

passed = len(common_cols) - len(failed)
print(f"      KS pass rate : {passed}/{len(common_cols)}")

# 6. Build augmented dataset
print("\n[6/6] Building augmented training set...")

real_train  = pd.read_parquet("data/processed/train.parquet")
synth_clean = synthetic.copy()
synth_clean['binary_label'] = 1
synth_clean['attack_type']  = 'Synthetic-Attack'
synth_clean = synth_clean.reindex(
    columns=real_train.columns, fill_value=0)

augmented = pd.concat(
    [real_train, synth_clean], ignore_index=True)
augmented = augmented.sample(
    frac=1, random_state=42).reset_index(drop=True)
augmented.to_parquet(
    "data/processed/augmented_train.parquet", index=False)

results = {
    "adversarial_auc": round(adv_auc, 4),
    "verdict":         verdict,
    "n_real":          len(attack_df),
    "n_synthetic":     len(synthetic),
    "ks_pass_rate":    f"{passed}/{len(common_cols)}",
}
with open("logs/gan_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"""

  Phase 2 Complete                                
--------------------------------------------------------
  Adversarial AUC : {adv_auc:.4f}                      
  KS Pass Rate    : {passed}/{len(common_cols)}                       
  Synthetic rows  : {len(synthetic):,}               
  Augmented total : {len(augmented):,}             

  Next: python src/train_robust_classifier.py
""")