#!/usr/bin/env python3
"""
Quick check: Do we have WAV files from different dates/times?
"""

import os
import glob
from pathlib import Path
from collections import Counter

# Find all WAV files
wav_root = Path.home() / "Uni-stuff/semester-2/applied_Ml/reef_zmsc/data/wav/PAPCA_test/"
pattern = str(wav_root / "**" / "wav" / "*.wav")
wavs = sorted(glob.glob(pattern, recursive=True))

print(f"Found {len(wavs):,} WAV files\n")

# Extract dates from paths
dates = []
for wav_path in wavs:
    parts = wav_path.split('/')
    if 'PAPCA' in parts:
        i = parts.index('PAPCA')
        if i + 2 < len(parts):
            logger = parts[i+1]
            date = parts[i+2]
            dates.append((logger, date))

# Count by logger and date
logger_counts = Counter([l for l, d in dates])
date_counts = Counter([d for l, d in dates])

print("=" * 80)
print("LOGGER DISTRIBUTION")
print("=" * 80)
for logger, count in sorted(logger_counts.items()):
    print(f"Logger {logger}: {count:,} files")

print("\n" + "=" * 80)
print("DATE DISTRIBUTION (First 50)")
print("=" * 80)
for date, count in sorted(date_counts.items())[:50]:
    print(f"{date}: {count:>6,} files")

print(f"\n... and {len(date_counts) - 50} more dates" if len(date_counts) > 50 else "")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total dates: {len(date_counts)}")
print(f"Total loggers: {len(logger_counts)}")
print(f"Date range: {min(date_counts.keys())} to {max(date_counts.keys())}")

# Estimate clips
total_clips = len(wavs) * 20  # Assuming 200s WAVs = 20 clips each
print(f"\nEstimated total clips: {total_clips:,}")

# Check if dates span multiple times of day
print("\n" + "=" * 80)
print("TIME-OF-DAY DISTRIBUTION POSSIBILITY")
print("=" * 80)
print(f"If recordings were continuous 24/7:")
print(f"  Expected clips per hour: {total_clips // 24:,}")
print(f"  Dawn clips (3 hours): ~{total_clips * 3 // 24:,}")
print(f"  Dusk clips (3 hours): ~{total_clips * 3 // 24:,}")