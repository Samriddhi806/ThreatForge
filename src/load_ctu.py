# src/load_ctu.py — Load and parse CTU-13 for GAN use in Phase 2
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

print("=" * 55)
print("  CTU-13 Loader & Label Parser")
print("=" * 55)

CTU_DIR = "CTU/"
files = sorted(glob.glob(CTU_DIR + "*.parquet"))
print(f"\nFound {len(files)} CTU scenario files")

def parse_label(label_str):
    """Convert CTU label string to binary: 1=botnet, 0=benign, -1=skip"""
    l = str(label_str).lower()
    if 'botnet' in l:
        return 1
    elif 'normal' in l:
        return 0
    else:
        return -1  # Background — skip

dfs = []
for fpath in files:
    fname = os.path.basename(fpath)
    scenario = fname.split('-')[0]
    family   = fname.split('-')[1] if '-' in fname else 'Unknown'

    df = pd.read_parquet(fpath)
    df['binary_label'] = df['label'].astype(str).apply(parse_label)
    df['scenario']     = scenario
    df['family']       = family

    # Skip background flows
    df = df[df['binary_label'] != -1]

    botnet_count = (df['binary_label'] == 1).sum()
    normal_count = (df['binary_label'] == 0).sum()
    print(f"  Scenario {scenario:>2s} ({family:8s}): "
          f"{botnet_count:>6,} botnet | {normal_count:>6,} normal")
    dfs.append(df)

full_ctu = pd.concat(dfs, ignore_index=True)
print(f"\nTotal CTU rows (after removing background): {len(full_ctu):,}")
print(f"Botnet: {(full_ctu['binary_label']==1).sum():,}")
print(f"Normal: {(full_ctu['binary_label']==0).sum():,}")

# ── Select numeric features only ──────────────────────────
feature_cols = ['dur', 'stos', 'dtos', 'tot_pkts', 'tot_bytes', 'src_bytes']
# Encode protocol as numeric
full_ctu['proto_num'] = full_ctu['proto'].astype('category').cat.codes
feature_cols.append('proto_num')

X_ctu = full_ctu[feature_cols].copy()

# Fix negatives and nulls
X_ctu = X_ctu.fillna(0).clip(lower=0)

# Clip outliers
upper = X_ctu.quantile(0.995)
X_ctu = X_ctu.clip(upper=upper, axis=1)

# Normalize
ctu_scaler = MinMaxScaler()
X_ctu_scaled = pd.DataFrame(
    ctu_scaler.fit_transform(X_ctu),
    columns=feature_cols
)
X_ctu_scaled['binary_label'] = full_ctu['binary_label'].values
X_ctu_scaled['family']       = full_ctu['family'].values

# Save
os.makedirs("data/processed", exist_ok=True)
X_ctu_scaled.to_parquet("data/processed/ctu_clean.parquet", index=False)
joblib.dump(ctu_scaler, "models/ctu_scaler.pkl")

# Save botnet-only rows for GAN Phase 2
ctu_botnet = X_ctu_scaled[X_ctu_scaled['binary_label'] == 1].drop(
    columns=['binary_label', 'family'])
ctu_botnet.to_parquet("data/processed/ctu_botnet_only.parquet", index=False)

print("\n── Saved ────────────────────────────────────────────")
print("  data/processed/ctu_clean.parquet")
print("  data/processed/ctu_botnet_only.parquet  ← GAN input")
print("  models/ctu_scaler.pkl")
print(f"\n  CTU botnet rows for GAN: {len(ctu_botnet):,}")
print("\n── Botnet families breakdown ────────────────────────")
fam = X_ctu_scaled[X_ctu_scaled['binary_label']==1]['family'].value_counts()
for name, count in fam.items():
    bar = "█" * (count // 500)
    print(f"  {name:10s} {count:>6,}  {bar}")

print("\n✅ CTU loading complete!")
print("   Next: run python src/baseline.py")