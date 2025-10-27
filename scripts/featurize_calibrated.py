
#!/usr/bin/env python3
"""
featurize_calibrated.py
-----------------------
Streaming feature extractor that:
  - Reads a clip manifest (from slice_manifest.py)
  - Applies per-logger calibration (db_volts_per_uPa) to convert voltage spectra -> pressure (µPa) spectra
  - Computes calibrated log-mel features (mean & std over time) + simple eco stats
  - Writes per-day Parquet shards in float16 to save disk

Usage:
  pip install numpy pandas pyarrow soundfile librosa
  python scripts/featurize_calibrated.py \
    --manifest data/manifests/clip_manifest.parquet \
    --out-root data/features/mels \
    --n-mels 64 --fmin 10 --fmax 6000 \
    --dtype float16 \
    --per-day-shards

Notes:
  - We DO NOT save audio clips; we read windows directly from big WAVs.
  - Calibration is taken from each row's json_meta['calibration_ref'].
  - If calibration is missing, we proceed without it (volts-based features), tagging 'calibrated=False'.
"""

import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import Dict, Any, Optional, Tuple

def load_calibration(calib_json_path: str):
    with open(calib_json_path, "r") as f:
        C = json.load(f)
    freq_hz = np.asarray(C["freq_hz"], dtype=float)
    db_v_per_uPa = np.asarray(C["db_volts_per_uPa"], dtype=float)
    return freq_hz, db_v_per_uPa

def interp_calibration_to_fft(db_v_per_uPa, calib_freqs, fft_freqs):
    # Extrapolate by edge values (clip)
    return np.interp(fft_freqs, calib_freqs, db_v_per_uPa, left=db_v_per_uPa[0], right=db_v_per_uPa[-1])

def stft_power(y, sr, n_fft=2048, hop_length=512, win_length=2048, center=False):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=center)
    P = (np.abs(S))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return P, freqs

def apply_calibration_power(P_volts, db_v_per_uPa_on_bins):
    # Power calibration: multiply by 10^(-dB/10)
    gain = np.power(10.0, -db_v_per_uPa_on_bins / 10.0)  # shape [F]
    return (P_volts.T * gain).T  # broadcast over time

def mel_from_power(P_power, sr, n_fft, n_mels=64, fmin=10.0, fmax=None):
    # Use librosa mel filter on a linear-frequency power spectrogram
    if fmax is None:
        fmax = sr / 2.0
    mel = librosa.feature.melspectrogram(S=P_power, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, power=1.0)
    # Convert to dB for stability
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def eco_stats(y, sr):
    # Lightweight indices
    rms = float(np.sqrt(np.mean(y**2) + 1e-20))
    # Spectral entropy
    S, _ = stft_power(y, sr, n_fft=1024, hop_length=256, win_length=1024, center=False)
    ps = S / (np.sum(S, axis=0, keepdims=True) + 1e-20)
    spec_entropy = float(-np.mean(np.sum(ps * np.log(ps + 1e-20), axis=0)))
    # Spectral centroid/bandwidth
    cent = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    bw   = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    return {"rms": rms, "spectral_entropy": spec_entropy, "centroid": cent, "bandwidth": bw}

def process_window(row, n_fft, hop_length, win_length, n_mels, fmin, fmax, dtype_out, cache):
    wav_path = row["wav_path"]
    start_s  = float(row["start_s"])
    end_s    = float(row["end_s"])
    dur_s    = end_s - start_s

    # Read window
    with sf.SoundFile(wav_path, "r") as rf:
        sr = rf.samplerate
        start_frame = int(round(start_s * sr))
        n_frames    = int(round(dur_s   * sr))
        rf.seek(start_frame)
        y = rf.read(frames=n_frames, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1).astype(np.float32)  # mono mixdown

    # STFT power
    P_volts, fft_freqs = stft_power(y, sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)

    # Calibration
    calibrated = False
    db_bins = None
    # Expect calibration ref in json_meta
    calib_ref = None
    try:
        jm = row["json_meta"]
        if isinstance(jm, dict):
            calib_ref = jm.get("calibration_ref", None)
    except Exception:
        calib_ref = None

    if calib_ref and os.path.isfile(calib_ref):
        # Cache calibration per path
        if calib_ref not in cache["calib"]:
            cf, dbv = load_calibration(calib_ref)
            cache["calib"][calib_ref] = (cf, dbv)
        cf, dbv = cache["calib"][calib_ref]
        db_bins = interp_calibration_to_fft(dbv, cf, fft_freqs)
        P_uPa = apply_calibration_power(P_volts, db_bins)  # now in µPa^2
        calibrated = True
    else:
        P_uPa = P_volts  # fallback: volts-based

    # Mel features (on pressure-calibrated power if available)
    mel_db = mel_from_power(P_uPa, sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Summaries
    mel_mean = mel_db.mean(axis=1).astype(dtype_out)
    mel_std  = mel_db.std(axis=1).astype(dtype_out)

    stats = eco_stats(y, sr)
    out = {
        "wav_path": wav_path,
        "start_s": start_s,
        "end_s": end_s,
        "sr": sr,
        "calibrated": calibrated,
        "n_mels": int(n_mels),
        "fmin": float(fmin),
        "fmax": float(fmax) if fmax is not None else None,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "rms": np.array([stats["rms"]], dtype=dtype_out)[0],
        "spectral_entropy": np.array([stats["spectral_entropy"]], dtype=dtype_out)[0],
        "centroid": np.array([stats["centroid"]], dtype=dtype_out)[0],
        "bandwidth": np.array([stats["bandwidth"]], dtype=dtype_out)[0],
    }
    # Optional: include logger/date if present
    for k in ("logger", "date"):
        if k in row:
            out[k] = row[k]
    return out

def day_shard_path(out_root, logger, date):
    return os.path.join(out_root, "PAPCA", logger or "unknown", date or "unknown", "features.parquet")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", default="data/features/mels")
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--win-length", type=int, default=2048)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--fmin", type=float, default=10.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--dtype", choices=["float16","float32"], default="float16")
    ap.add_argument("--per-day-shards", action="store_true", help="Write per-day parquet files instead of a single file")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of windows (for pilot runs)")
    args = ap.parse_args()

    dtype_out = np.float16 if args.dtype == "float16" else np.float32

    # Load manifest (parquet or csv)
    if args.manifest.lower().endswith(".csv"):
        df = pd.read_csv(args.manifest)
    else:
        df = pd.read_parquet(args.manifest)
    # Ensure dict in json_meta
    if "json_meta" in df.columns:
        df["json_meta"] = df["json_meta"].apply(lambda x: x if isinstance(x, dict) else x)

    if args.limit:
        df = df.iloc[:args.limit]

    os.makedirs(args.out_root, exist_ok=True)

    cache = {"calib": {}}
    rows_out = []
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        row_dict = row._asdict() if hasattr(row, "_asdict") else row._asdict()  # pandas namedtuple
        try:
            feat = process_window(
                row_dict,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
                dtype_out=dtype_out,
                cache=cache
            )
            if args.per_day_shards:
                logger = feat.get("logger", "unknown")
                date   = feat.get("date", "unknown")
                shard = day_shard_path(args.out_root, logger, date)
                os.makedirs(os.path.dirname(shard), exist_ok=True)
                # Append row to shard
                pd.DataFrame([feat]).to_parquet(shard, index=False, append=os.path.exists(shard))
            else:
                rows_out.append(feat)
        except Exception as e:
            print(f"[warn] failed on window {i}/{total}: {e}")

        if i % 200 == 0:
            print(f"Processed {i}/{total} windows")

    if not args.per_day_shards:
        out_path = os.path.join(args.out_root, "features.parquet")
        pd.DataFrame(rows_out).to_parquet(out_path, index=False)
        print(f"Wrote features -> {out_path}")
    else:
        print(f"Done writing per-day shards under {args.out_root}")

if __name__ == "__main__":
    main()
