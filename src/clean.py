# src/clean.py — Phase 1: Clean, preprocess, and split CICIDS data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

print("=" * 55)
print("  Phase 1 — Data Cleaning & Preprocessing")
print("=" * 55)

# ── 1. Load merged dataset ─────────────────────────────────
print("\n[1/7] Loading full_cicids.parquet...")
df = pd.read_parquet("data/processed/full_cicids.parquet")
print(f"      Loaded: {len(df):,} rows × {df.shape[1]} cols")

# ── 2. Drop non-feature columns ───────────────────────────
print("\n[2/7] Dropping non-feature columns...")
drop_cols = ['attack_type', 'binary_label']
# Also drop any timestamp or ID-like columns if present
for col in df.columns:
    if any(x in col.lower() for x in ['timestamp', 'flow id', 'src ip',
                                        'dst ip', 'src port', 'dst port']):
        drop_cols.append(col)

drop_cols = list(set([c for c in drop_cols if c in df.columns]))
X = df.drop(columns=drop_cols)
y_binary    = df['binary_label']
y_multiclass = df['attack_type']
print(f"      Feature matrix: {X.shape}")
print(f"      Dropped: {drop_cols}")

# ── 3. Fix impossible values ──────────────────────────────
print("\n[3/7] Fixing negative values...")
num_cols = X.select_dtypes(include='number').columns

neg_before = (X[num_cols] < 0).sum().sum()
X[num_cols] = X[num_cols].clip(lower=0)  # no negative flow stats
neg_after  = (X[num_cols] < 0).sum().sum()
print(f"      Negative values fixed: {neg_before:,} → {neg_after}")

# ── 4. Remove constant / near-zero variance columns ───────
print("\n[4/7] Removing constant columns...")
std = X[num_cols].std()
const_cols = std[std == 0].index.tolist()
X = X.drop(columns=const_cols)
print(f"      Removed {len(const_cols)} constant columns: {const_cols}")

# ── 5. Clip extreme outliers (99.5th percentile) ──────────
print("\n[5/7] Clipping outliers at 99.5th percentile...")
num_cols = X.select_dtypes(include='number').columns
upper = X[num_cols].quantile(0.995)
X[num_cols] = X[num_cols].clip(upper=upper, axis=1)
print(f"      Outliers clipped across {len(num_cols)} features")

# ── 6. Normalize features ─────────────────────────────────
print("\n[6/7] Normalizing with MinMaxScaler...")
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X[num_cols]),
    columns=num_cols,
    index=X.index
)
print(f"      All features now in range [0, 1]")

# Save scaler for later use in pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("      Scaler saved → models/scaler.pkl")

# ── 7. Train / Val / Test split ───────────────────────────
print("\n[7/7] Splitting: 70% train / 15% val / 15% test ...")

# First split off test (15%)
X_temp, X_test, y_temp, y_test, ym_temp, ym_test = train_test_split(
    X_scaled, y_binary, y_multiclass,
    test_size=0.15, stratify=y_binary, random_state=42
)

# Then split remaining into train (70%) and val (15%)
X_train, X_val, y_train, y_val, ym_train, ym_val = train_test_split(
    X_temp, y_temp, ym_temp,
    test_size=0.1765,  # 0.1765 × 0.85 ≈ 0.15
    stratify=y_temp, random_state=42
)

print(f"      Train : {len(X_train):>7,} rows")
print(f"      Val   : {len(X_val):>7,} rows")
print(f"      Test  : {len(X_test):>7,} rows")

# ── 8. Save splits ────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

train = X_train.copy(); train['binary_label'] = y_train.values; train['attack_type'] = ym_train.values
val   = X_val.copy();   val['binary_label']   = y_val.values;   val['attack_type']   = ym_val.values
test  = X_test.copy();  test['binary_label']  = y_test.values;  test['attack_type']  = ym_test.values

train.to_parquet("data/processed/train.parquet", index=False)
val.to_parquet("data/processed/val.parquet",     index=False)
test.to_parquet("data/processed/test.parquet",   index=False)

# Also save attack-only rows for GAN training later
attack_only = train[train['binary_label'] == 1].drop(
    columns=['binary_label', 'attack_type'])
attack_only.to_parquet("data/processed/attack_only.parquet", index=False)

print("\n── Saved files ──────────────────────────────────────")
print("  data/processed/train.parquet")
print("  data/processed/val.parquet")
print("  data/processed/test.parquet")
print("  data/processed/attack_only.parquet  ← GAN input")
print("  models/scaler.pkl")

# ── 9. Summary ────────────────────────────────────────────
print("\n── Final class balance (train set) ─────────────────")
vc = train['attack_type'].value_counts()
for name, count in vc.items():
    bar = "█" * (count // 20000)
    print(f"  {name:15s} {count:>7,}  {bar}")

print("\n✅ Phase 1 complete! Ready for baseline model.")
print("   Next: run python src/baseline.py")