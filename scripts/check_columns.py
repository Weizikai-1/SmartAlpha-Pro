import pandas as pd
df = pd.read_parquet("data/processed/factors_neutral.parquet", engine="fastparquet")
print("Columns:", list(df.columns))
print("Has fwd_ret_5d:", "fwd_ret_5d" in df.columns)
print("Label-like cols:", [c for c in df.columns if "ret" in c.lower() or "fwd" in c.lower() or "label" in c.lower()])
