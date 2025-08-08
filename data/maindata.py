from data_sorting import split_csv_by_decade, process_split
import os
import numpy as np
import shutil


if __name__ == "__main__":
    base = "/cluster/home/ltavier/data"
    raw_folder = os.path.join(base, "normalized_monthly")
    seq_len = 48
    start_years = (1984,1990,1996,2002,2008,2014)
    length=start_years[1]-start_years[0]

    # 1) split your raw CSVs into decades
    decade_dirs = split_csv_by_decade(raw_folder, seq_len, base, start_years,length)

    splits = {
        (f"split{start_years[idx]}-{start_years[idx+1]-1}"
        if idx < len(start_years) - 1
        else f"split{start_years[idx]}-2020"): decade_dirs[idx]
        for idx in range(len(start_years))
    }


    out_dirs = {
        k: os.path.join(base, f"monthly_data{('-'.join(d[-9:].split(os.sep)[-1].split('data')[-1]))}")
        for k, d in splits.items()
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    drop_cols = ["permno", "DATE", "ExcessRET"]
    target_col = "ExcessRET"
    SAMPLE_SIZE = 1024


    raw_label, ts_cols = None, None
    for i, (name, in_dir) in enumerate(splits.items()):
        out_dir = out_dirs[name]
        process_split(
            name=name,
            in_dir=in_dir,
            out_dir=out_dir,
            sequence_length=seq_len,
            drop_cols=drop_cols + ([raw_label] if raw_label else []),
            ts_cols=ts_cols,
            raw_label=target_col,
            sample_size=SAMPLE_SIZE,
            selection_mode="topmv",
            # carry over the old “debug only for first block” logic:
            debug_permno="10107" if i == 0 else None,
            debug_date ="1990-01-31" if i == 0 else None,
        )

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
            print(f"Removed intermediate folder: {out_dir}")
        if os.path.isdir(in_dir):
            shutil.rmtree(in_dir)
            print(f"Removed input CSV folder:    {in_dir}")


    print("All done!")