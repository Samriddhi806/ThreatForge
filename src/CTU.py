
import pandas as pd
df = pd.read_parquet('CTU/1-Neris-20110810.binetflow.parquet')
print('Shape:', df.shape)
print()
print('Columns and dtypes:')
for col in df.columns:
    print(f'  {col:35s} {str(df[col].dtype):10s}')
print()
print('Sample label values:')
for col in df.columns:
    if 'label' in col.lower():
        print(f'  {col}:', df[col].unique()[:5])
