#!/usr/bin/env python3
"""
Slice WAVs into a clip manifest with reliable UTC timestamps.

Key improvements:
- Parse tz-aware UTC from sidecar['rec_start_time_utc'] (preferred) and normalize to ISO8601 'Z'.
- Stable clip_id: md5(abs_path + start_frame).
- Include clip_index, start_frame/end_frame/n_frames, and inherit overload/clip stats from sidecar.
- Optional --compute-rms to add quick energy for sampling.
"""

import argparse
import os
import re
import glob
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import soundfile as sf


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def find_wavs(in_root: str) -> List[str]:
    # Case-insensitive *.wav under **/wav/
    pattern = os.path.join(in_root, "**", "wav", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


LOGGER_DATE_RX = re.compile(r"(?i)/PAPCA/(\d{4})/(\d{8})/wav/")

def detect_logger_and_date(wav_path: str):
    m = LOGGER_DATE_RX.search(wav_path.replace("\\", "/"))
    if m:
        return m.group(1), m.group(2)
    # fallback using path tokens
    parts = wav_path.replace("\\", "/").split("/")
    if "PAPCA" in parts:
        i = parts.index("PAPCA")
        if i + 2 < len(parts):
            return parts[i+1], parts[i+2]
    return None, None


# -----------------------------------------------------------------------------
# Timestamp parsing / normalization
# -----------------------------------------------------------------------------
def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # Assume already UTC if naive (your pipeline writes UTC)
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def _isoz(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return _to_utc(dt).isoformat().replace("+00:00", "Z")

def extract_timestamp_from_json(sidecar: dict) -> Optional[datetime]:
    """
    Prefer tz-aware UTC 'rec_start_time_utc'. Fallback to other keys.
    """
    candidates = ['rec_start_time_utc', 'start_time', 'timestamp', 'recording_start', 'datetime_start']
    for key in candidates:
        if key in sidecar and sidecar[key] is not None:
            val = sidecar[key]
            # ISO strings
            if isinstance(val, str):
                s = val.strip()
                # Normalize trailing 'Z'
                if s.endswith("Z"):
                    try:
                        return datetime.fromisoformat(s.replace("Z", "+00:00"))
                    except ValueError:
                        pass
                # Try flexible ISO
                try:
                    return datetime.fromisoformat(s)
                except ValueError:
                    pass
                # Try common explicit formats
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y%m%d_%H%M%S",
                ):
                    try:
                        dt = datetime.strptime(s, fmt)
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            # Unix timestamp
            if isinstance(val, (int, float)):
                try:
                    return datetime.fromtimestamp(val, tz=timezone.utc)
                except Exception:
                    pass
    return None

def extract_timestamp_from_filename(wav_path: str, date_str: str) -> Optional[datetime]:
    """
    Fallback: try YYYYMMDD_HHMMSS in filename, combined with date folder.
    """
    base = os.path.basename(wav_path)
    if '_' in base:
        parts = base.split('_')
        if len(parts) >= 2:
            time_part = parts[1].split('.')[0]
            if len(time_part) >= 6 and date_str and len(date_str) == 8:
                try:
                    y = int(date_str[0:4]); m = int(date_str[4:6]); d = int(date_str[6:8])
                    hh = int(time_part[0:2]); mm = int(time_part[2:4]); ss = int(time_part[4:6])
                    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)
                except Exception:
                    return None
    return None


# -----------------------------------------------------------------------------
# IDs, dirs, features
# -----------------------------------------------------------------------------
def generate_clip_id(wav_abs_path: str, start_frame: int) -> str:
    # Stable across OS: absolute path + integer start_frame
    unique_str = f"{wav_abs_path}|{start_frame}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]

def desired_clip_dir(clip_root: str, logger: str, date: str) -> str:
    return os.path.join(clip_root, logger or "unknown", date or "unknown", "clips")

def compute_rms(block: np.ndarray) -> float:
    if block.ndim == 2:
        # mean across channels before RMS, or channel-agnostic RMS—choose one
        block = block.mean(axis=1)
    # avoid overflow for int types
    block = block.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(block**2) + 1e-20))


# -----------------------------------------------------------------------------
# Slicing
# -----------------------------------------------------------------------------
def slice_file(
    wav_path: str,
    clip_len: float,
    hop: float,
    write_audio: Optional[str],
    clip_root: str,
    compute_rms_flag: bool = False,
) -> List[Dict[str, Any]]:
    """Slice WAV into clips with timestamp extraction"""
    wav_abs = os.path.abspath(wav_path)
    meta_path = os.path.splitext(wav_abs)[0] + ".json"

    # Load JSON sidecar (non-fatal)
    sidecar: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                sidecar = json.load(f)
        except Exception as e:
            print(f"[warn] Failed to load JSON for {wav_path}: {e}")

    # Info
    info = sf.info(wav_abs)
    sr = int(info.samplerate)
    ch = int(info.channels)
    frames_total = int(info.frames)
    total_sec = frames_total / sr

    # Logger/date
    logger, date = detect_logger_and_date(wav_abs)

    # Base timestamp
    base_dt = extract_timestamp_from_json(sidecar)
    timestamp_source = "json"
    if base_dt is None and date:
        base_dt = extract_timestamp_from_filename(wav_abs, date)
        if base_dt:
            timestamp_source = "filename"
    if base_dt is None and date:
        # Date-at-midnight fallback (UTC)
        try:
            y = int(date[0:4]); m = int(date[4:6]); d = int(date[6:8])
            base_dt = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
            timestamp_source = "date_midnight"
            print(f"[warn] No precise timestamp for {os.path.basename(wav_abs)}, using date@00:00Z")
        except Exception:
            timestamp_source = "none"

    # Normalize to UTC
    base_dt_utc = _to_utc(base_dt) if base_dt else None

    # Iteration
    rows: List[Dict[str, Any]] = []
    i = 0
    start_s = 0.0

    # Pre-open file if we need per-clip write or RMS
    rf = None
    if write_audio or compute_rms_flag:
        rf = sf.SoundFile(wav_abs, "r")

    try:
        while start_s + 1e-9 < total_sec:
            end_s = min(start_s + clip_len, total_sec)
            dur_s = end_s - start_s
            start_frame = int(round(start_s * sr))
            n_frames = int(round(dur_s * sr))
            end_frame = start_frame + n_frames

            # Timestamp per clip
            start_iso = None
            if base_dt_utc is not None:
                clip_dt = base_dt_utc + timedelta(seconds=start_s)
                start_iso = _isoz(clip_dt)

            clip_id = generate_clip_id(wav_abs, start_frame)

            row = {
                "clip_id": clip_id,
                "filepath": wav_abs,
                "logger": logger,
                "date": date,
                "clip_index": i,
                "start_time": start_iso,              # ISO8601 with Z or None
                "timestamp_source": timestamp_source, # json | filename | date_midnight | none
                "start_s": float(round(start_s, 6)),
                "end_s": float(round(end_s, 6)),
                "duration_s": float(round(dur_s, 6)),
                "sr": sr,
                "channels": ch,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_frames": n_frames,
            }

            # Inherit useful sidecar bits for sampling
            for k in ("overload_flags", "clip_count", "clip_fraction",
                      "peak_value_per_channel", "min_value_per_channel",
                      "first_data_time_utc", "finalised_time_utc", "rec_start_time_utc"):
                if k in sidecar:
                    row[k] = sidecar[k]

            # Optional RMS (fast)
            if compute_rms_flag:
                rf.seek(start_frame)
                block = rf.read(frames=n_frames, dtype="float32", always_2d=True)
                row["rms"] = compute_rms(block)

            # Optional write
            if write_audio:
                out_dir = desired_clip_dir(clip_root, logger or "unknown", date or "unknown")
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(wav_abs))[0]
                out_name = f"{base}__i={i:06d}__t={start_iso or 'NA'}.{write_audio.lower()}"
                out_path = os.path.join(out_dir, out_name)
                # reuse block if computed; otherwise stream-read
                if not compute_rms_flag:
                    with sf.SoundFile(wav_abs, "r") as rfr:
                        rfr.seek(start_frame)
                        block = rfr.read(frames=n_frames, dtype="float32", always_2d=False)
                sf.write(out_path, block, sr, format=write_audio.upper(),
                         subtype=None if write_audio.lower()=="flac" else "PCM_16")
                row["clip_path"] = os.path.abspath(out_path)

            rows.append(row)
            i += 1
            start_s += hop
    finally:
        if rf is not None:
            rf.close()

    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Slice WAV files into clips with UTC timestamps")
    ap.add_argument("--in-root", default="data/wav/PAPCA_test", help="Root directory with WAV files")
    ap.add_argument("--out-manifest", default="data/manifests/clip_manifest.parquet", help="Output manifest path")
    ap.add_argument("--clip-len", type=float, default=10.0, help="Clip length in seconds")
    ap.add_argument("--hop", type=float, default=10.0, help="Hop between clips in seconds")
    ap.add_argument("--write-audio", choices=["flac","wav"], default=None, help="Write physical clip files")
    ap.add_argument("--clip-root", default="data/clips/PAPCA_test", help="Root for physical clips")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of WAV files (for testing)")
    ap.add_argument("--compute-rms", action="store_true", help="Compute and store per-clip RMS (fast)")
    args = ap.parse_args()

    print("=" * 80)
    print("SLICING WAV FILES INTO MANIFEST (UTC)")
    print("=" * 80)
    print(f"Input root:         {args.in_root}")
    print(f"Output manifest:    {args.out_manifest}")
    print(f"Clip length:        {args.clip_len}s")
    print(f"Hop:                {args.hop}s")
    print(f"Write audio:        {args.write_audio}")
    print(f"Compute RMS:        {args.compute_rms}")
    print()

    wavs = find_wavs(args.in_root)
    print(f"Found {len(wavs)} WAV files")
    if args.limit:
        wavs = wavs[:args.limit]
        print(f"Limited to {len(wavs)} files for testing")

    rows: List[Dict[str, Any]] = []
    missing_ts = 0

    for i, wp in enumerate(wavs, 1):
        try:
            clips = slice_file(
                wp, args.clip_len, args.hop,
                write_audio=args.write_audio,
                clip_root=args.clip_root,
                compute_rms_flag=args.compute_rms
            )
            for c in clips:
                if c["start_time"] is None:
                    missing_ts += 1
            rows.extend(clips)
        except Exception as e:
            print(f"[error] Failed on {wp}: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(wavs)} wavs, {len(rows)} clips so far")

    print()
    print("=" * 80)
    print("WRITING MANIFEST")
    print("=" * 80)
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)

    df = pd.DataFrame(rows)

    total = len(df)
    print(f"Total clips:        {total:,}")
    print(f"Missing timestamps: {missing_ts:,} ({(missing_ts/max(total,1))*100:.1f}%)")

    if "start_time" in df.columns and total:
        valid = df["start_time"].notna().sum()
        print(f"Valid timestamps:   {valid:,} ({valid/max(total,1)*100:.1f}%)")
        if valid:
            tmp = df[df["start_time"].notna()].copy()
            tmp["start_time_dt"] = pd.to_datetime(tmp["start_time"], utc=True)
            print(f"\nTimestamp range:")
            print(f"  From: {tmp['start_time_dt'].min()}")
            print(f"  To:   {tmp['start_time_dt'].max()}")

            tmp["hour"] = tmp["start_time_dt"].dt.hour
            dawn = tmp[tmp["hour"].between(4, 6)].shape[0]
            dusk = tmp[tmp["hour"].between(17, 19)].shape[0]
            print(f"\nHourly coverage:")
            print(f"  Hours with data: {tmp['hour'].nunique()}/24")
            print(f"  Dawn clips (04-06h): {dawn:,} ({dawn/max(valid,1)*100:.1f}%)")
            print(f"  Dusk clips (17-19h): {dusk:,} ({dusk/max(valid,1)*100:.1f}%)")

    # Save
    out = args.out_manifest
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)  # default snappy

    print(f"\n✓ Manifest written to: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
