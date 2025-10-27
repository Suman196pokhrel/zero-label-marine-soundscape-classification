#!/usr/bin/env python3
"""
sampler_stratified.py

Stratified sampling from clip_manifest.parquet into temporal bins with:
- Target allocations per bin (scaled to --n-clips, default 50k)
- Optional RMS-based weighting (if 'rms' exists)
- Exclude heavily clipped (clip_fraction) if available
- Per-day caps inside each bin to avoid domination by long runs
- Reproducible selection (seeded)

Output: Parquet (recommended) or CSV list of selected clips, based on --out path.
"""

import argparse
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Bin definitions and baseline proportions
# --------------------------------------------------------------------------- #

BINS = {
    "pre_dawn":   [0, 1, 2, 3],                 # 00:00–03:59
    "dawn":       [4, 5, 6],                    # 04:00–06:59
    "morning":    [7, 8, 9, 10, 11],            # 07:00–11:59
    "midday":     [12, 13],                     # 12:00–13:59
    "dusk":       [17, 18, 19],                 # 17:00–19:59
    "night":      [14, 15, 16, 20, 21, 22, 23], # 14–16 & 20–23
}

# Baseline proportions (sum = 20k); these will be scaled to --n-clips
BASE_TARGETS_20K = {
    "dawn": 5000,
    "dusk": 5000,
    "morning": 3000,
    "night": 3000,
    "pre_dawn": 2000,
    "midday": 2000,
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def assign_bins(hour_series: pd.Series) -> pd.Series:
    def map_hour(h):
        h = int(h)
        for name, hours in BINS.items():
            if h in hours:
                return name
        return "other"
    return hour_series.astype(int).map(map_hour)

def weighted_choice_idx(idx: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sample without replacement over given index array."""
    weights = np.clip(weights, 0, None)
    s = weights.sum()
    if s <= 0 or np.all(weights == 0):
        return rng.choice(idx, size=k, replace=False)
    probs = weights / s
    return rng.choice(idx, size=k, replace=False, p=probs)

def cap_by_day(bin_df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Apply a per-day cap to widen temporal spread within a bin."""
    if cap <= 0 or bin_df.empty:
        return bin_df
    parts = []
    for day, g in bin_df.groupby("day"):
        if len(g) <= cap:
            parts.append(g)
        else:
            idx = weighted_choice_idx(g.index.values, g["w"].to_numpy(), cap, rng)
            parts.append(g.loc[idx])
    return pd.concat(parts, ignore_index=False)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Stratified sampler for reef clips (Parquet-first output)")
    ap.add_argument("--manifest", default="data/manifests/clip_manifest.parquet",
                    help="Input manifest Parquet/CSV (Parquet recommended).")
    ap.add_argument("--out", default="data/manifests/sample_stratified.parquet",
                    help="Output path (.parquet recommended; falls back to CSV if extension is .csv).")
    ap.add_argument("--n-clips", type=int, default=50000,
                    help="Total clips to sample (default: 50,000).")
    # Optional per-bin overrides (if None, scaled from baseline proportions)
    ap.add_argument("--alloc-dawn", type=int, default=None)
    ap.add_argument("--alloc-dusk", type=int, default=None)
    ap.add_argument("--alloc-morning", type=int, default=None)
    ap.add_argument("--alloc-night", type=int, default=None)
    ap.add_argument("--alloc-pre-dawn", type=int, default=None, dest="alloc_pre_dawn")
    ap.add_argument("--alloc-midday", type=int, default=None)
    ap.add_argument("--clip-frac-max", type=float, default=0.01,
                    help="Drop clips with clip_fraction > this (if present).")
    ap.add_argument("--per-day-cap", type=int, default=60,
                    help="Max clips per day inside a bin before final sampling.")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_clips = int(args.n_clips)

    # Load manifest (Parquet preferred)
    print("Reading manifest…")
    if args.manifest.lower().endswith(".parquet"):
        df = pd.read_parquet(args.manifest)
    else:
        df = pd.read_csv(args.manifest)

    # Parse time
    df["start_time_dt"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df = df[df["start_time_dt"].notna()].copy()
    df["hour"] = df["start_time_dt"].dt.hour
    df["day"]  = df["start_time_dt"].dt.floor("D")

    # Optional: drop heavily clipped
    if "clip_fraction" in df.columns:
        before = len(df)
        df = df[df["clip_fraction"].fillna(0.0) <= args.clip_frac_max].copy()
        print(f"Clipping filter: kept {len(df):,}/{before:,} ({len(df)/before*100:.1f}%)")

    # Assign bins
    df["bin"] = assign_bins(df["hour"])

    # Prepare RMS weights if present
    if "rms" in df.columns:
        rms = df["rms"].astype(float).clip(lower=0)
        lo, hi = np.percentile(rms, [10, 99.5])
        w = (rms - lo) / max(hi - lo, 1e-9)
        df["w"] = np.clip(0.1 + 0.9*w, 0, 1)  # floor so quiet clips still have a chance
        print("Using RMS-based weighting.")
    else:
        df["w"] = 1.0
        print("RMS not found; sampling uniformly within bins.")

    # Compute per-bin allocations
    alloc = {
        "dawn":      args.alloc_dawn,
        "dusk":      args.alloc_dusk,
        "morning":   args.alloc_morning,
        "night":     args.alloc_night,
        "pre_dawn":  args.alloc_pre_dawn,
        "midday":    args.alloc_midday,
    }
    base_sum = sum(BASE_TARGETS_20K.values())  # 20000
    scale = n_clips / base_sum if base_sum > 0 else 1.0
    for k, v in alloc.items():
        if v is None:
            alloc[k] = int(round(BASE_TARGETS_20K[k] * scale))

    # Adjust to exactly n_clips
    diff = n_clips - sum(alloc.values())
    if diff != 0:
        # distribute across bins with most availability
        avail = df["bin"].value_counts()
        order = [b for b,_ in sorted(avail.items(), key=lambda kv: -kv[1])]
        si = 0
        while diff != 0 and order:
            b = order[si % len(order)]
            alloc[b] = alloc.get(b, 0) + (1 if diff > 0 else -1)
            diff += -1 if diff > 0 else 1
            si += 1

    print("\nBin allocations (target total = {:,}):".format(n_clips))
    for k in ["dawn","dusk","morning","midday","night","pre_dawn"]:
        print(f"  {k:9s}: {alloc.get(k,0):6d}  (available={int((df['bin']==k).sum()):,})")

    # Sample each bin
    out_parts = []
    for bname, target in alloc.items():
        sub = df[df["bin"] == bname].copy()
        if sub.empty or target <= 0:
            if target > 0:
                print(f"[warn] Bin '{bname}' has no data; requested {target}, picking 0.")
            continue

        # Stage A: widen spread with per-day cap
        sub_cap = cap_by_day(sub, args.per_day_cap, rng)

        # Stage B: weighted pick up to target
        if len(sub_cap) <= target:
            pick = sub_cap
        else:
            idx = weighted_choice_idx(sub_cap.index.values, sub_cap["w"].to_numpy(), target, rng)
            pick = sub_cap.loc[idx]

        out_parts.append(pick)

    out = pd.concat(out_parts, ignore_index=False) if out_parts else pd.DataFrame(columns=df.columns)

    # Top-up/trim to exact n_clips
    short = n_clips - len(out)
    if short > 0:
        pool = df.loc[~df.index.isin(out.index)].copy()
        if not pool.empty:
            k = min(short, len(pool))
            idx = weighted_choice_idx(pool.index.values, pool["w"].to_numpy(), k, rng)
            out = pd.concat([out, pool.loc[idx]], ignore_index=False)
    elif short < 0:
        # Trim by inverse weight to keep high-weight items
        k = n_clips
        out = out.sample(n=k, weights=out["w"], random_state=args.seed)

    out = out.sort_values(["day", "hour", "clip_index"]).reset_index(drop=True)

    # Columns to persist
    cols = [
        "clip_id","filepath","clip_index",
        "start_time","day","hour","bin",
        "start_s","duration_s","sr","channels",
        "start_frame","n_frames","timestamp_source",
    ]
    for opt in [
        "rms","clip_fraction","overload_flags",
        "first_data_time_utc","finalised_time_utc","rec_start_time_utc"
    ]:
        if opt in out.columns and opt not in cols:
            cols.append(opt)

    # Save (Parquet recommended)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_path = args.out
    if out_path.lower().endswith(".parquet"):
        # Requires pyarrow (recommended) or fastparquet
        out[cols].to_parquet(out_path, index=False)  # default compression is snappy when using pyarrow
    else:
        out[cols].to_csv(out_path, index=False)

    print(f"\n✓ Wrote stratified sample: {out_path} (n={len(out):,})")

    # Small report
    print("\nDistribution by bin:")
    print(out["bin"].value_counts().rename("n"))

    print("\nDistribution by hour:")
    print(out.groupby("hour").size())

    per_day = out.groupby("day").size()
    print(f"\nDays covered: {per_day.index.nunique()}, median clips/day: {int(per_day.median())}")

if __name__ == "__main__":
    main()
