import os
import re
import cudf
import json
import pandas as pd
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────
INPUT_FOLDER   = '/cluster/home/ltavier/data/monthly_data'
RF_CSV         = '/cluster/home/ltavier/data/macro_data_sorted.csv'
OUTPUT_FOLDER  = '/cluster/home/ltavier/data/normalized_monthly'
OUTPUT_PREFIX  = 'RZ'
# Maximum number of zero‐valued characteristics allowed per row
ZERO_THRESHOLD = 60

# ─── Helpers ──────────────────────────────────────────────────────────
def extract_date(fn):

    #Extract date string of format YYYYMMDD from filename.

    m = re.search(r"_(\d{8})\.csv$", fn)
    return m.group(1) if m else None


def load_and_concat(input_folder: str) -> cudf.DataFrame:

    #Read all CSVs in `input_folder`, cleanup text fields, and concatenate into a single cudf DataFrame.
    #Adds a 'DATE' column parsed from filenames.

    #Returns:
    #    cudf.DataFrame sorted by ['permno', 'DATE']

    dfs = []
    for fn in tqdm(sorted(os.listdir(input_folder), key=extract_date),
                   desc=f"Loading {input_folder}"):
        if not fn.endswith('.csv'):
            continue
        path = os.path.join(input_folder, fn)
        df = cudf.read_csv(path, dtype=str)  # keep as strings for now; don't inject zeros
        for c in df.columns:
            df[c] = df[c].str.strip()
        # standardize placeholder strings to missing
        df = df.replace({'<NA>': None, '': None})
        date_str = extract_date(fn)
        df['DATE'] = cudf.to_datetime(date_str, format="%Y%m%d")
        dfs.append(df)
    full = cudf.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} files into DataFrame, total rows: {full.shape[0]}")
    print(f"Unique permnos: {full['permno'].nunique()}")
    return full.sort_values(['permno', 'DATE'])


def monthly_rank_normalize(
    cdf: cudf.DataFrame,
    feature_cols: list,
    date_col: str = 'DATE',
) -> cudf.DataFrame:

    #Rank‐normalize each feature in `feature_cols` per month.
    #Stashes the raw target in a new column and normalizes in‐place.

    #Returns:
    #  - cudf.DataFrame with normalized features and raw target appended

    pdf = cdf.to_pandas()
    print(f"Monthly rank‐normalization: processing {len(feature_cols)} features")
    for col in feature_cols:
        ranks = pdf.groupby(date_col)[col].rank(method='first')
        counts = pdf.groupby(date_col)[col].transform('count')
        pdf[col] = 2 * ranks.div(counts + 1) - 1
    sample_feats = feature_cols[:3]
    print("Sample normalized values for first 3 features on first 5 rows:")
    print(pdf[sample_feats].head().to_string(index=False))
    return cudf.from_pandas(pdf)


def main(
    input_folder: str,
    rf_csv: str,
    output_folder: str,
    output_prefix: str,
    zero_thresh: int
):
    # 1) Load RF rates
    rf_pd = pd.read_csv(rf_csv, dtype={'yyyymm': str})
    rf_dict = rf_pd.set_index('yyyymm')['Rfree'].astype(float).to_dict()
    print(f"Loaded RF rates for {len(rf_dict)} months")

    # 2) Load and concatenate monthly data
    df = load_and_concat(input_folder)

    # 3) Compute ExcessRET
    pdf = df.to_pandas()
    extras = pd.DataFrame({
        'yyyymm_next': (pdf['DATE'] + pd.DateOffset(months=1)).dt.strftime('%Y%m'),
    })
    extras['Rfree'] = extras['yyyymm_next'].map(rf_dict).fillna(0).astype('float32')
    pdf['ExcessRET'] = pdf['RET'].astype('float32') - extras['Rfree']
    df = cudf.from_pandas(pdf)
    print(f"ExcessRET computed: min={df['ExcessRET'].min()}, max={df['ExcessRET'].max()}, mean={df['ExcessRET'].mean():.4f}")

    # 4) Define feature columns
    feat_cols = [c for c in df.columns
                 if c not in ['Unnamed: 0','permno','DATE','RET','mve0','prc','SHROUT','SHRCD','sic2','ExcessRET']]
    print(f"Identified {len(feat_cols)} feature columns")

    # 5) Cast features to numeric and impute missing with month cross-section medians
    pdf_imp = df.to_pandas()

    # make sure feature columns are numeric; coerce non-numeric to NaN
    pdf_imp[feat_cols] = pdf_imp[feat_cols].apply(pd.to_numeric, errors='coerce')

    # compute per-month medians for all features at once and fill
    medians = pdf_imp.groupby('DATE')[feat_cols].transform('median')
    pdf_imp[feat_cols] = pdf_imp[feat_cols].fillna(medians)

    # move back to cuDF
    df = cudf.from_pandas(pdf_imp)

    # 5) Filter rows
    # Count how many features are missing (None/NaN) in each row
    missing_counts = df[feat_cols].isna().sum(axis=1)
    before = df.shape[0]

    # Keep rows with fewer missing values than threshold
    mask = (df['ExcessRET'] >= -1) & (missing_counts <= zero_thresh)
    df = df[mask]
    after = df.shape[0]
    print(f"Filtered rows: before={before}, after={after}, removed={before-after}")
    print(f"ExcessRET stats after filtering: min={df['ExcessRET'].min()}, max={df['ExcessRET'].max()}, mean={df['ExcessRET'].mean():.4f}")


    # 5bis) Cast features to numeric and impute missing with month cross-section medians
    pdf_imp = df.to_pandas()

    # make sure feature columns are numeric; coerce non-numeric to NaN
    pdf_imp[feat_cols] = pdf_imp[feat_cols].apply(pd.to_numeric, errors='coerce')

    # compute per-month medians for all features at once and fill
    medians = pdf_imp.groupby('DATE')[feat_cols].transform('median')
    pdf_imp[feat_cols] = pdf_imp[feat_cols].fillna(medians)

    # move back to cuDF
    df = cudf.from_pandas(pdf_imp)

    # 6) Rank‐normalize features
    df_norm = monthly_rank_normalize(df, feat_cols)

    # 7) Sort final frame
    df_norm = df_norm.sort_values(['DATE', 'mvel1'])

    # 8) Reorder and drop unwanted cols, keep DATE & permno
    keep = ['DATE','permno','ExcessRET'] + [c for c in df_norm.columns
                                            if c not in ['Unnamed: 0','permno','DATE','RET',
                                                         'mve0','prc','SHROUT','sic2','SHRCD','ExcessRET']]
    df_final = df_norm[keep]

    # 9) Save features list
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "features.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print(f"Features list saved")

    # 10) Write each date's CSV into new folder
    pdf_final = df_final.to_pandas()
    for date, month_df in pdf_final.groupby('DATE'):
        stamp = date.strftime('%Y%m%d')
        fn = f"{output_prefix}_{stamp}.csv"
        path = os.path.join(output_folder, fn)
        month_df.to_csv(path, index=False)
        print(f"  → Wrote {fn} ({len(month_df)} rows)")

    print(f"All files written to {output_folder}")

if __name__ == "__main__":
    main(INPUT_FOLDER, RF_CSV, OUTPUT_FOLDER, OUTPUT_PREFIX, ZERO_THRESHOLD)


"""
import os
import re
import cudf
import json
import pandas as pd
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────
INPUT_FOLDER   = '/cluster/home/ltavier/data/monthly_data'
RF_CSV         = '/cluster/home/ltavier/data/macro_data_sorted.csv'
OUTPUT_FOLDER  = '/cluster/home/ltavier/data/normalized_monthly'
OUTPUT_PREFIX  = 'RZ'
# Maximum number of zero‐valued characteristics allowed per row
ZERO_THRESHOLD = 60

# ─── Helpers ──────────────────────────────────────────────────────────
def extract_date(fn):

    #Extract date string of format YYYYMMDD from filename.

    m = re.search(r"_(\d{8})\.csv$", fn)
    return m.group(1) if m else None


def load_and_concat(input_folder: str) -> cudf.DataFrame:

    #Read all CSVs in `input_folder`, cleanup text fields, and concatenate into a single cudf DataFrame.
    #Adds a 'DATE' column parsed from filenames.

    #Returns:
    #    cudf.DataFrame sorted by ['permno', 'DATE']

    dfs = []
    for fn in tqdm(sorted(os.listdir(input_folder), key=extract_date),
                   desc=f"Loading {input_folder}"):
        if not fn.endswith('.csv'):
            continue
        path = os.path.join(input_folder, fn)
        df = cudf.read_csv(path, dtype=str).fillna('0')
        for c in df.columns:
            df[c] = df[c].str.strip().str.replace(r'^<NA>$', '0', regex=True)
        date_str = extract_date(fn)
        df['DATE'] = cudf.to_datetime(date_str, format="%Y%m%d")
        dfs.append(df)
    full = cudf.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} files into DataFrame, total rows: {full.shape[0]}")
    print(f"Unique permnos: {full['permno'].nunique()}")
    return full.sort_values(['permno', 'DATE'])


def monthly_rank_normalize(
    cdf: cudf.DataFrame,
    feature_cols: list,
    date_col: str = 'DATE',
) -> cudf.DataFrame:

    #Rank‐normalize each feature in `feature_cols` per month.
    #Stashes the raw target in a new column and normalizes in‐place.

    #Returns:
    #  - cudf.DataFrame with normalized features and raw target appended

    pdf = cdf.to_pandas()
    print(f"Monthly rank‐normalization: processing {len(feature_cols)} features")
    for col in feature_cols:
        ranks = pdf.groupby(date_col)[col].rank(method='first')
        counts = pdf.groupby(date_col)[col].transform('count')
        pdf[col] = 2 * ranks.div(counts + 1) - 1
    sample_feats = feature_cols[:3]
    print("Sample normalized values for first 3 features on first 5 rows:")
    print(pdf[sample_feats].head().to_string(index=False))
    return cudf.from_pandas(pdf)


def main(
    input_folder: str,
    rf_csv: str,
    output_folder: str,
    output_prefix: str,
    zero_thresh: int
):
    # 1) Load RF rates
    rf_pd = pd.read_csv(rf_csv, dtype={'yyyymm': str})
    rf_dict = rf_pd.set_index('yyyymm')['Rfree'].astype(float).to_dict()
    print(f"Loaded RF rates for {len(rf_dict)} months")

    # 2) Load and concatenate monthly data
    df = load_and_concat(input_folder)

    # 3) Compute ExcessRET
    pdf = df.to_pandas()
    extras = pd.DataFrame({
        'yyyymm_next': (pdf['DATE'] + pd.DateOffset(months=1)).dt.strftime('%Y%m'),
    })
    extras['Rfree'] = extras['yyyymm_next'].map(rf_dict).fillna(0).astype('float32')
    pdf['ExcessRET'] = pdf['RET'].astype('float32') - extras['Rfree']
    df = cudf.from_pandas(pdf)
    print(f"ExcessRET computed: min={df['ExcessRET'].min()}, max={df['ExcessRET'].max()}, mean={df['ExcessRET'].mean():.4f}")

    # 4) Define feature columns
    feat_cols = [c for c in df.columns
                 if c not in ['Unnamed: 0','permno','DATE','RET','mve0','prc','SHROUT','SHRCD','sic2','ExcessRET']]
    print(f"Identified {len(feat_cols)} feature columns")

    # 5) Filter rows
    zero_counts = (df[feat_cols] == 0).sum(axis=1)
    before = df.shape[0]
    mask = (df['ExcessRET'] >= -1) & (zero_counts <= zero_thresh)
    df = df[mask]
    after = df.shape[0]
    print(f"Filtered rows: before={before}, after={after}, removed={before-after}")
    print(f"ExcessRET stats after filtering: min={df['ExcessRET'].min()}, max={df['ExcessRET'].max()}, mean={df['ExcessRET'].mean():.4f}")

    # 6) Rank‐normalize features
    df_norm = monthly_rank_normalize(df, feat_cols)

    # 7) Sort final frame
    df_norm = df_norm.sort_values(['DATE', 'mvel1'])

    # 8) Reorder and drop unwanted cols, keep DATE & permno
    keep = ['DATE','permno','ExcessRET'] + [c for c in df_norm.columns
                                            if c not in ['Unnamed: 0','permno','DATE','RET',
                                                         'mve0','prc','SHROUT','sic2','SHRCD','ExcessRET']]
    df_final = df_norm[keep]

    # 9) Save features list
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "features.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print(f"Features list saved")

    # 10) Write each date's CSV into new folder
    pdf_final = df_final.to_pandas()
    for date, month_df in pdf_final.groupby('DATE'):
        stamp = date.strftime('%Y%m%d')
        fn = f"{output_prefix}_{stamp}.csv"
        path = os.path.join(output_folder, fn)
        month_df.to_csv(path, index=False)
        print(f"  → Wrote {fn} ({len(month_df)} rows)")

    print(f"All files written to {output_folder}")

if __name__ == "__main__":
    main(INPUT_FOLDER, RF_CSV, OUTPUT_FOLDER, OUTPUT_PREFIX, ZERO_THRESHOLD)


import os
import re
import cudf
import json
import pandas as pd
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────
INPUT_FOLDER   = '/cluster/home/ltavier/data/monthly_data'
RF_CSV         = '/cluster/home/ltavier/data/macro_data_sorted.csv'
OUTPUT_FOLDER  = '/cluster/home/ltavier/data/normalized_monthly'
OUTPUT_PREFIX  = 'RZ'
# Maximum number of zero‐valued characteristics allowed per row
ZERO_THRESHOLD = 60

# ─── Helpers ──────────────────────────────────────────────────────────
def extract_date(fn):
    
    #Extract date string of format YYYYMMDD from filename.
    
    m = re.search(r"_(\d{8})\.csv$", fn)
    return m.group(1) if m else None


def load_and_concat(input_folder: str) -> cudf.DataFrame:
    
    #Read all CSVs in `input_folder`, clean text fields, preserve real zeros
    #and keep genuine missing values as NaN. Concatenate into a single cudf DataFrame.
    #Adds a 'DATE' column parsed from filenames.

    #Returns:
    #    cudf.DataFrame sorted by ['permno', 'DATE']
    
    dfs = []
    for fn in tqdm(sorted(os.listdir(input_folder), key=extract_date),
                   desc=f"Loading {input_folder}"):
        if not fn.endswith('.csv'):
            continue
        path = os.path.join(input_folder, fn)
        # 1) Read all columns as strings
        df = cudf.read_csv(path, dtype=str)

        # 2) Strip whitespace and turn exact "<NA>" into true missing
        for c in df.columns:
            # 1) strip whitespace
            col = df[c].str.strip()
            # 2) turn exact "<NA>" into empty string
            col = col.str.replace(r'^<NA>$', '', regex=True)
            # 3) mask empty strings back to true missing (None)
            df[c] = col.mask(col == '', None)

        # 3) Cast numeric‐looking columns to float32, but leave permno as-is
        for c in df.columns:
            if c == 'permno':
                continue
            try:
                df[c] = df[c].astype('float32')
            except Exception:
                pass  # non‐numeric columns stay as they are


        # 4) Parse DATE from filename
        date_str = extract_date(fn)
        df['DATE'] = cudf.to_datetime(date_str, format="%Y%m%d")
        dfs.append(df)

    full = cudf.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} files into DataFrame, total rows: {full.shape[0]}")
    print(f"Unique permnos: {full['permno'].nunique()}")
    return full.sort_values(['permno', 'DATE'])


def monthly_rank_normalize(
    cdf: cudf.DataFrame,
    feature_cols: list,
    date_col: str = 'DATE',
) -> cudf.DataFrame:
    
    #Rank-normalize each feature in `feature_cols` per month.
    #Genuine NaNs are left untouched.

    #Returns:
    #  - cudf.DataFrame with normalized features and raw target appended
    
    pdf = cdf.to_pandas()
    print(f"Monthly rank-normalization: processing {len(feature_cols)} features")

    for col in feature_cols:
        # ignore NaNs when ranking; they stay NaN
        ranks  = pdf.groupby(date_col)[col] \
                    .rank(method='first', na_option='keep')
        counts = pdf.groupby(date_col)[col].transform('count')
        pdf[col] = 2 * ranks.div(counts + 1) - 1

    sample_feats = feature_cols[:3]
    print("Sample normalized values for first 3 features on first 5 rows:")
    print(pdf[sample_feats].head().to_string(index=False))
    return cudf.from_pandas(pdf)


def main(
    input_folder: str,
    rf_csv: str,
    output_folder: str,
    output_prefix: str,
    zero_thresh: int
):
    # 1) Load RF rates
    rf_pd = pd.read_csv(rf_csv, dtype={'yyyymm': str})
    rf_dict = rf_pd.set_index('yyyymm')['Rfree'].astype(float).to_dict()
    print(f"Loaded RF rates for {len(rf_dict)} months")

    # 2) Load and concatenate monthly data
    df = load_and_concat(input_folder)

    # 3) Compute ExcessRET
    pdf = df.to_pandas()
    extras = pd.DataFrame({
        'yyyymm_next': (pdf['DATE'] + pd.DateOffset(months=1)).dt.strftime('%Y%m'),
    })
    extras['Rfree'] = extras['yyyymm_next'].map(rf_dict).fillna(0).astype('float32')
    pdf['ExcessRET'] = pdf['RET'].astype('float32') - extras['Rfree']
    df = cudf.from_pandas(pdf)
    print(f"ExcessRET computed: min={df['ExcessRET'].min()}, "
          f"max={df['ExcessRET'].max()}, mean={df['ExcessRET'].mean():.4f}")

    # 4) Define feature columns
    feat_cols = [c for c in df.columns
                 if c not in ['Unnamed: 0','permno','DATE','RET','mve0',
                              'prc','SHROUT','SHRCD','sic2','ExcessRET']]
    print(f"Identified {len(feat_cols)} feature columns")

    # 5) Rank-normalize features (NaNs are kept)
    df_norm = monthly_rank_normalize(df, feat_cols)

    # 6) Fill only those genuine NaNs with 0 (real zeros untouched)
    print("Filling missing values with 0 (real zeros are untouched)...")
    pdf_norm = df_norm.to_pandas()
    pdf_norm[feat_cols] = pdf_norm[feat_cols].fillna(0)
    df_norm = cudf.from_pandas(pdf_norm)

    # 7) Filter rows: keep ExcessRET ≥ -1 and drop rows with too many zeros if desired
    zero_counts = (df_norm[feat_cols] == 0).sum(axis=1)
    before = df_norm.shape[0]
    mask = (df_norm['ExcessRET'] >= -1) & (zero_counts <= zero_thresh)
    df_filt = df_norm[mask]
    after = df_filt.shape[0]
    print(f"Filtered rows: before={before}, after={after}, removed={before-after}")
    print(f"ExcessRET stats after filtering: min={df_filt['ExcessRET'].min()}, "
          f"max={df_filt['ExcessRET'].max()}, mean={df_filt['ExcessRET'].mean():.4f}")

    # 8) Sort final frame
    df_sorted = df_filt.sort_values(['DATE', 'mvel1'])

    # 9) Reorder and drop unwanted cols, keep DATE & permno
    keep = ['DATE','permno','ExcessRET'] + [
        c for c in df_sorted.columns
        if c not in ['Unnamed: 0','permno','DATE','RET',
                     'mve0','prc','SHROUT','sic2','SHRCD','ExcessRET']
    ]
    df_final = df_sorted[keep]

    # 10) Save features list
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "features.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print("Features list saved")

    # 11) Write each date's CSV into new folder
    pdf_final = df_final.to_pandas()
    for date, month_df in pdf_final.groupby('DATE'):
        stamp = date.strftime('%Y%m%d')
        fn = f"{output_prefix}_{stamp}.csv"
        path = os.path.join(output_folder, fn)
        month_df.to_csv(path, index=False)
        print(f"  → Wrote {fn} ({len(month_df)} rows)")

    print(f"All files written to {output_folder}")


if __name__ == "__main__":
    main(INPUT_FOLDER, RF_CSV, OUTPUT_FOLDER, OUTPUT_PREFIX, ZERO_THRESHOLD)







"""