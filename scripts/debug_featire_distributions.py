"""
Debug: Check actual feature distributions (FAST - uses sampling)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HOME = Path(os.environ["HOME"])
REPO_ROOT = HOME / "Uni-stuff/semester-2/applied_Ml/reef_zmsc"

CLUSTERS_PATH = REPO_ROOT / "data/clusters/clusters_hdbscan.parquet"
FUSED_PATH = REPO_ROOT / "data/features/embeds_fused_pilot/PAPCA"
AUTO_LABELS_PATH = REPO_ROOT / "data/auto_labels/cluster_auto_labels.csv"

print("Loading cluster assignments...")
clusters_df = pd.read_parquet(CLUSTERS_PATH)
print(f"Loaded {len(clusters_df):,} clips")

# Sample clusters first
sample_size = min(10000, len(clusters_df))
sample_clusters = clusters_df.sample(n=sample_size, random_state=42)

print("\nLoading fused features (sampling)...")
# Load only one file for speed
sample_file = next(FUSED_PATH.rglob("features.parquet"))
fused_df = pd.read_parquet(sample_file)
print(f"Loaded sample features from {sample_file.parent.name}/{sample_file.parent.parent.name}")

print("\nMerging...")
merged = sample_clusters.merge(
    fused_df,
    on=['logger', 'date', 'start_s'],
    how='inner'
)
print(f"Merged {len(merged):,} clips with features")

if len(merged) == 0:
    print("\nERROR: No clips matched! Checking merge keys...")
    print(f"Sample cluster keys: logger={sample_clusters['logger'].iloc[0]}, date={sample_clusters['date'].iloc[0]}")
    print(f"Sample fused keys: logger={fused_df['logger'].iloc[0]}, date={fused_df['date'].iloc[0]}")
    exit(1)

# Ecoacoustic features (these should already be in the dataframe)
features = [
    'low_freq_energy', 'mid_freq_energy', 'high_freq_energy',
    'spectral_flatness_mean', 'aci', 'temporal_entropy',
    'snr_db', 'dynamic_range_db', 'spectral_centroid_mean'
]

# Check if features exist
available_features = [f for f in features if f in merged.columns]

if not available_features:
    print("\nERROR: No ecoacoustic features found after merge!")
    print(f"Available columns: {list(merged.columns)}")
    exit(1)

print(f"\nFound {len(available_features)} ecoacoustic features")

print("\n" + "="*60)
print("FEATURE STATISTICS")
print("="*60)

for feat in available_features:
    data = merged[feat].dropna()
    
    print(f"\n{feat}:")
    print(f"  Min: {data.min():.4f}")
    print(f"  Q1: {data.quantile(0.25):.4f}")
    print(f"  Median: {data.quantile(0.5):.4f}")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Q3: {data.quantile(0.75):.4f}")
    print(f"  Max: {data.max():.4f}")

plt.tight_layout()
output_path = REPO_ROOT / "analysis/feature_distributions.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
print(f"\nSaving plot...")
plt.savefig(output_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"Plot saved: {output_path}")

# Check cluster-level auto-labels if they exist
if AUTO_LABELS_PATH.exists():
    print("\n" + "="*60)
    print("AUTO-LABEL SCORES (first 10 clusters)")
    print("="*60)
    
    labels_df = pd.read_csv(AUTO_LABELS_PATH)
    print(labels_df.head(10)[['cluster_id', 'auto_label', 'confidence', 
                                'bio_score', 'anthro_score', 'ambient_score']])
    
    print("\nLabel distribution:")
    print(labels_df['auto_label'].value_counts())
    
    print("\nScore statistics:")
    print(labels_df[['bio_score', 'anthro_score', 'ambient_score']].describe())

print("\n" + "="*60)
print("DONE - Check the plot and statistics above")
print("="*60)