"""

In this file, we create the lookback windows for the asset characteristics

"""


import os
import glob
import re
import random
import cudf
import cupy as cp
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import shutil
from typing import List



# ─── Helpers & params ──────────────────────────────────────────────────────────
def extract_date(fn):
    m = re.search(r"_(\d{8})\.csv$", fn)
    return m.group(1) if m else ""


def split_csv_by_decade(input_folder, sequence_length, output_base, start_years, length):
    """
    Copy CSVs from `input_folder` into decade‐buckets under `output_base`.
    Returns a list of the three created folder paths.
    """
    def extract_date(fn):
        m = re.search(r"_(\d{8})\.csv$", fn)
        return pd.to_datetime(m.group(1), format="%Y%m%d") if m else None

    created_dirs = []
    n = len(start_years)
    for idx, start_year in enumerate(start_years):
        if idx == n - 1:
            end_year = 2020
        else:
            end_year = start_year + length - 1

        out_dir = os.path.join(output_base, f"data{start_year}-{end_year}")
        os.makedirs(out_dir, exist_ok=True)
        created_dirs.append(out_dir)

        # compute lower/upper bounds
        first_bound = pd.to_datetime(f"{start_year:04d}0131", format="%Y%m%d")
        lower = first_bound - pd.DateOffset(months=sequence_length)
        upper = pd.to_datetime(f"{end_year:04d}1231", format="%Y%m%d")

        print(f"→ Bucket {start_year}-{end_year}: copying files in [{lower.date()} → {upper.date()}]")
        for fn in tqdm(os.listdir(input_folder), desc=f"{start_year}s"):
            if not fn.endswith(".csv"):
                continue
            dt = extract_date(fn)
            if dt is None:
                continue
            if lower <= dt <= upper:
                shutil.copy2(
                    os.path.join(input_folder, fn),
                    os.path.join(out_dir, fn)
                )

    return created_dirs

# ─── Core windowing functions ───────────────────────────────────────────────────
def load_and_concat(input_folder):
    dfs = []
    for fn in tqdm(sorted(os.listdir(input_folder), key=extract_date),
                   desc=f"Loading {input_folder}"):
        if not fn.endswith('.csv'): continue
        path = os.path.join(input_folder, fn)
        df = cudf.read_csv(path, dtype=str).fillna('0')
        for c in df.columns:
            df[c] = df[c].str.strip()
            df[c] = df[c].str.replace(r'^<NA>$', '0', regex=True)
        df['DATE'] = extract_date(fn)
        dfs.append(df)
    full = cudf.concat(dfs, ignore_index=True)
    full['DATE'] = cudf.to_datetime(full['DATE'], format="%Y%m%d")
    print(f"\nLoaded {input_folder}: shape={full.shape}")
    print("Unique permnos:", full['permno'].nunique())
    nulls = full.isnull().sum().to_pandas()
    print("Missing values per column:", nulls[nulls > 0])
    return full.sort_values(['permno', 'DATE'])


def process_stock_windows(item,
                          window_length, stride,
                          target_index, label_col,
                          ts_cols, sequence_length,   # pass it in
                          debug_permno=None):
    permno, grp = item
    grp = grp.sort_values('DATE').reset_index(drop=True)
    out, first = [], True   
    for start in range(0, len(grp) - window_length + 1, stride):
        win = grp.iloc[start:start + window_length]
        # select in ts_cols order, and only the first seq_length rows
        raw = win.iloc[:sequence_length][ts_cols].to_numpy(dtype=np.float32)
        Y   = float(win.iloc[target_index][label_col])
        sd  = str(win.iloc[0]['DATE'])[:10]
        ed  = str(win.iloc[target_index]['DATE'])[:10]

        if debug_permno and permno == debug_permno and first:
            first = False
            print(f"DEBUG permno={permno} window@{sd}→{ed}")
            print("X1 raw shape:", raw.shape)
            print("Sample X1 values:", raw[:, :3])
            print("Y:", Y)
        
        out.append({'permno': int(permno), 'start_date': sd,
                    'end_date': ed, 'X1': raw, 'Y': Y})
    return out

def extract_windows(pdf, ts_cols, sequence_length, label_col, debug_permno=None):
    groups = list(pdf.groupby('permno'))
    print(f"Extracting windows from {len(groups)} permnos...")
    target_index = sequence_length -1
    window_length = sequence_length
    stride = 1
    worker = partial(
        process_stock_windows,
        window_length=window_length,
        stride=stride,
        target_index=target_index,
        label_col=label_col,
        ts_cols=ts_cols,
        sequence_length=sequence_length,
        debug_permno=debug_permno
    )
    samples = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as exe:
        for res in tqdm(exe.map(worker, groups), total=len(groups), desc="Sliding windows"):
            samples.extend(res)
    print(f"Total windows extracted: {len(samples)}")
    return samples

def save_buckets(
    samples,
    output_folder,
    ts_cols,
    SAMPLE_SIZE,
    selection_mode="topmv",  # "random", "topmv", "botmv", "all"
    min_end=None,
    max_end=None,
    debug_date=None,
):
    """
    Save sliding-window buckets as parquet files, optionally filtering by end-date.
    """
    mvel1_idx = ts_cols.index('mvel1')

    # group samples by end_date
    buckets = defaultdict(list)
    for s in samples:
        buckets[s['end_date']].append(s)

    first = True
    written_idx = 0
    skipped = 0

    for i, (wend, bucket) in enumerate(sorted(buckets.items())):
        # filter by allowed end-date range
        if min_end or max_end:
            wend_dt = pd.to_datetime(wend)
            if (min_end is not None and wend_dt < min_end) or \
               (max_end is not None and wend_dt > max_end):
                skipped += 1
                continue

        do_debug = (debug_date == wend) if debug_date else first
        if first:
            first = False

        print(f"Window {written_idx:03d} @ {wend}: {len(bucket)} permnos")

        # select entries
        if selection_mode == "all":
            sel = bucket
            print(f"Saving all {len(sel)} entries")

        elif selection_mode == "random":
            sel = random.sample(bucket, SAMPLE_SIZE) if len(bucket) > SAMPLE_SIZE else bucket
            print(f"Randomly sampled {len(sel)} entries")

        elif selection_mode in ("topmv", "botmv"):
            if len(bucket) > SAMPLE_SIZE:
                bucket_sorted = sorted(
                    bucket,
                    key=lambda s: s['X1'][-1, mvel1_idx],
                    reverse=(selection_mode == "topmv")
                )
                sel = bucket_sorted[:SAMPLE_SIZE]
            else:
                sel = bucket
            mode_str = "highest" if selection_mode == "topmv" else "lowest"
            print(f"Selected {len(sel)} {mode_str}-mvel1 entries")
        else:
            raise ValueError(f"Unknown selection_mode '{selection_mode}'")

        # flatten X1 for saving
        flat_X1 = [s['X1'].flatten().tolist() for s in sel]

        df_out = pd.DataFrame({
            'permno':     [s['permno']    for s in sel],
            'start_date': [s['start_date'] for s in sel],
            'end_date':   [s['end_date']   for s in sel],
            'X1':         flat_X1,
            'Y':          [s['Y']          for s in sel],
        })

        out_path = os.path.join(output_folder, f"window_{written_idx:03d}_{wend}.parquet")
        df_out.to_parquet(
            out_path,
            index=False,
            engine='pyarrow',
            compression='zstd',
            compression_level=5
        )
        print(f"Saved {out_path}")
        written_idx += 1

    if skipped:
        print(f"Skipped {skipped} bucket(s) outside [{min_end} .. {max_end}]")


"""
def save_buckets(samples, output_folder, ts_cols, SAMPLE_SIZE, TOPMV = True, debug_date=None):
    # ─── load your ordered feature list ─────────────────────

    mvel1_idx    = ts_cols.index('mvel1')         # column index of mvel1 in each row

    buckets = defaultdict(list)
    for s in samples:
        buckets[s['end_date']].append(s)
    first = True
    for i, (wend, bucket) in enumerate(sorted(buckets.items())):
        do_debug = (debug_date == wend) if debug_date else first
        if first: first = False
        print(f"Window {i:03d} @ {wend}: {len(bucket)} permnos")
        if TOPMV == False:
            sel = random.sample(bucket, SAMPLE_SIZE) if len(bucket) > SAMPLE_SIZE else bucket
            print(f"Sampling {len(sel)} entries")
        else:
            if len(bucket) > SAMPLE_SIZE:
                # sort descending by the last timestep’s mve0
                bucket = sorted(
                    bucket,
                    key=lambda s: s['X1'][-1, mvel1_idx],
                    reverse=True
                )
                sel = bucket[:SAMPLE_SIZE]
            else:
                sel = bucket

            print(f"Selected {len(sel)} highest‐mve0 entries")
        
        flat_X1 = [s['X1'].flatten().tolist() for s in sel]

        pdf = pd.DataFrame({
            'permno':     [s['permno']    for s in sel],
            'start_date': [s['start_date'] for s in sel],
            'end_date':   [s['end_date']   for s in sel],
            'X1':         flat_X1,
            'Y':          [s['Y']          for s in sel],
        })

        # write with snappy compression (default engine is pyarrow)
        out_path = os.path.join(output_folder, f"window_{i:03d}_{wend}.parquet")
        pdf.to_parquet(
        out_path,
        index=False,
        engine='pyarrow',
        compression='zstd',
        compression_level=5
        )
        print(f"Saved {out_path}")
"""       

def merge_buckets(folder, out_path, rz_path, sequence_length):
    gc.collect()
    
    files = sorted(glob.glob(os.path.join(folder, "window_*.parquet")))
    print(f"Merging {len(files)} bucket files...")

    dfs = []
    for f in tqdm(files, desc="Reading buckets"):
        df = cudf.read_parquet(f).to_pandas()  # convert to CPU memory
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(by=['end_date', 'start_date', 'permno'])
    merged.to_parquet(
    out_path,
    index=False,
    engine='pyarrow',
    compression='zstd',
    compression_level=5
    )   
    print(f"Merged rows: {len(merged)} → {out_path}")


    total_len = len(merged['X1'].iloc[0])
    n_chars   = total_len // sequence_length

    # 2b) Stack & reshape X1, then take only the last row of each window
    X1_flat = np.stack(merged['X1'].to_numpy())  # (N, sequence_length * n_chars)
    Z_array = X1_flat.reshape(-1, sequence_length, n_chars)[:, -1, :]  # (N, n_chars)

    # 2c) Build a MultiIndex (end_date, permno)
    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(merged['end_date']), merged['permno']],
        names=['end_date', 'permno']
    )

    # 2d) Assemble the RZ DataFrame
    char_cols = [f"char_{i}" for i in range(n_chars)]
    R_series  = pd.Series(merged['Y'].values, index=idx, name='Y')
    Z_df      = pd.DataFrame(Z_array, index=idx, columns=char_cols)
    RZ        = pd.concat([R_series.to_frame(), Z_df], axis=1)


    RZ.to_parquet(
    rz_path,
    index=True,
    engine='pyarrow',
    compression='zstd',
    compression_level=5
    )

    print(f"Saved RZ to {rz_path}")

    del dfs, merged
    gc.collect()


def process_split(
    name,
    in_dir,
    out_dir,
    sequence_length,
    drop_cols,
    ts_cols,
    raw_label,
    selection_mode,
    sample_size=500,
    debug_permno=None,
    debug_date=None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ── derive [min_end, max_end] from the split name, e.g. "split1984-1989"
    m = re.search(r"split(\d{4})-(\d{4})", name)
    if not m:
        raise ValueError(f"Could not parse start/end years from split name: {name}")
    start_year, end_year = int(m.group(1)), int(m.group(2))
    min_end = pd.to_datetime(f"{start_year}-01-31")
    max_end = pd.to_datetime(f"{end_year}-12-31")

    # ─── load & clip ────────────────────────────────────────
    df = load_and_concat(in_dir)

    # on the first pass you decide ts_cols once
    if ts_cols is None:
        ts_cols = [c for c in df.columns if c not in drop_cols]

    # ─── sliding windows & save ──────────────────────────────
    samples = extract_windows(
        df.to_pandas(),
        ts_cols=ts_cols,
        sequence_length=sequence_length,
        label_col=raw_label,
        debug_permno=debug_permno,
    )

    save_buckets(
        samples,
        out_dir,
        ts_cols=ts_cols,
        SAMPLE_SIZE=sample_size,
        debug_date=debug_date,
        selection_mode=selection_mode,
        min_end=min_end,      # <-- NEW
        max_end=max_end,      # <-- NEW
    )

    # ─── 3) free CPU & GPU memory before merging ────────────────
    del df, samples
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    base_out = os.path.dirname(out_dir)
    parquet_path = os.path.join(base_out, f"TS_RZ")
    RZ_path = os.path.join(base_out, f"RZ")

    os.makedirs(parquet_path, exist_ok=True)
    os.makedirs(RZ_path,     exist_ok=True)

    merged_path = os.path.join(parquet_path, f"{name}-{sequence_length}.parquet")
    rz_path     = os.path.join(RZ_path, f"{name}-RZ.parquet")

    merge_buckets(out_dir, merged_path, rz_path, sequence_length)

    # ─── 5) final cleanup if you do more iterations later ────────
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()











