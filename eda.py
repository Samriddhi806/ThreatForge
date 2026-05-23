
# eda.py — Phase 1: Load and explore CICIDS data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load all parquet files ──────────────────────────────────────────
CICIDS_DIR = "CICIDS/"

files = {
    "Benign":       "Benign-Monday-no-metadata.parquet",
    "Botnet":       "Botnet-Friday-no-metadata.parquet",
    "Bruteforce":   "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS":         "DDoS-Friday-no-metadata.parquet",
    "DoS":          "DoS-Wednesday-no-metadata.parquet",
    "Infiltration": "Infiltration-Thursday-no-metadata.parquet",
    "Portscan":     "Portscan-Friday-no-metadata.parquet",
    "WebAttacks":   "WebAttacks-Thursday-no-metadata.parquet",
}

dfs = []
for label, fname in files.items():
    path = os.path.join(CICIDS_DIR, fname)
    df = pd.read_parquet(path)
    df['attack_type'] = label          # add attack type column
    df['binary_label'] = 0 if label == "Benign" else 1
    dfs.append(df)
    print(f"Loaded {label:15s} → {len(df):>7,} rows, {df.shape[1]} cols")

# ── 2. Merge all files ────────────────────────────────────────────────
full_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal dataset: {len(full_df):,} rows × {full_df.shape[1]} columns")

# ── 3. Class distribution ─────────────────────────────────────────────
print("\n── Class distribution ──")
print(full_df['attack_type'].value_counts())

plt.figure(figsize=(10, 4))
full_df['attack_type'].value_counts().plot(kind='bar', color='steelblue', edgecolor='white')
plt.title("Sample count per attack type")
plt.ylabel("Count")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("logs/class_distribution.png")
plt.show()
print("Plot saved → logs/class_distribution.png")

# ── 4. Check for NaN and infinite values ─────────────────────────────
print("\n── Data quality ──")
nan_count = full_df.isnull().sum().sum()
inf_count = np.isinf(full_df.select_dtypes(include='number')).sum().sum()
print(f"NaN values  : {nan_count:,}")
print(f"Inf values  : {inf_count:,}")

# ── 5. Basic stats ────────────────────────────────────────────────────
print("\n── Numeric feature stats ──")
print(full_df.describe().T[['mean','std','min','max']].head(15))

# ── 6. Save merged dataset ────────────────────────────────────────────
full_df.to_parquet("data/processed/full_cicids.parquet", index=False)
print("\n✅ Saved → data/processed/full_cicids.parquet")
print("Phase 1 Step 1 complete!")