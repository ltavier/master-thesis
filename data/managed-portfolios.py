"""
  In this file, we compute the characteristic managed-portfolios, we notate them as X2
  
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

# ─── Configuration ───────────────────────────────────────────────────────
RZ_FILES       = [
    "/cluster/home/ltavier/data/RZ/split1984-1989-RZ.parquet",
    "/cluster/home/ltavier/data/RZ/split1990-1995-RZ.parquet",
    "/cluster/home/ltavier/data/RZ/split1996-2001-RZ.parquet",
    "/cluster/home/ltavier/data/RZ/split2002-2007-RZ.parquet",
    "/cluster/home/ltavier/data/RZ/split2008-2013-RZ.parquet",
    "/cluster/home/ltavier/data/RZ/split2014-2020-RZ.parquet",   
]

OUTPUT_FILE    = "/cluster/home/ltavier/data/X2_monthly.parquet"

# Automatically detect number of CPUs
N_JOBS = multiprocessing.cpu_count()

# ─── Load and concatenate RZ data ─────────────────────────────────────────
dfs = [pd.read_parquet(path) for path in RZ_FILES]
RZ  = pd.concat(dfs)

# Identify characteristic columns (all except the target 'Y')
char_cols = [c for c in RZ.columns if c != 'Y']

# Unique sorted months
months = sorted(RZ.index.get_level_values('end_date').unique())

# Function to compute x2 for a single month
def compute_x2_for_month(month):
    sub = RZ.xs(month, level='end_date')
    Z   = sub[char_cols].values     # (n_i, p)
    R   = sub['Y'].values           # (n_i,)

    # —— pseudoinverse approach with cutoff rcond ——  
    rcond    = 1e-3  # drop singular values < 0.1% of max
    Z_pinv   = np.linalg.pinv(Z, rcond=rcond)
    x2_char  = Z_pinv @ R           # shape (p,)

    print(f"Computed X2 for {month} via pseudoinverse (rcond={rcond})")
    return month, x2_char

# Parallel computation across months
results = Parallel(n_jobs=N_JOBS, backend='loky')(
    delayed(compute_x2_for_month)(m) for m in months
)

# Assemble results into DataFrame (only char portfolios)
records = [
    pd.Series(x2, index=char_cols, name=month)
    for month, x2 in results
]
X2_df = pd.DataFrame(records)
X2_df.index.name = 'month'
X2_df = X2_df.sort_index()

# Save to disk
X2_df.to_parquet(OUTPUT_FILE)
print(f"Saved X2 for {len(months)} months to {OUTPUT_FILE} using {N_JOBS} CPUs")

print("Finished!")
