"""
Diagnostic Script for Auto-Labeling Issues
==========================================
Analyzes why acoustic rules aren't matching your data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path("~/Uni-stuff/semester-2/applied_Ml/reef_zmsc").expanduser()
CLUSTERED_DATA = ROOT_DIR / "data/clustering/results_50k/clustered_data_kmeans.parquet"
FUSED_FEATURES = ROOT_DIR / "data/features/embeds_fused_50k"

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA FOR DIAGNOSIS")
print("=" * 80)

# Load clustered data
print(f"\nLoading: {CLUSTERED_DATA}")
clustered_df = pd.read_parquet(CLUSTERED_DATA)
print(f"  ✅ {len(clustered_df):,} clips")
print(f"  Columns: {list(clustered_df.columns)}")

# Load a sample of fused features to analyze
print(f"\nLoading ecoacoustic features from: {FUSED_FEATURES}")
feature_files = list(FUSED_FEATURES.rglob("features.parquet"))
print(f"  Found {len(feature_files)} files")

# Load first few files
sample_dfs = []
for f in feature_files[:50]:  # Sample 50 files
    try:
        df = pd.read_parquet(f)
        sample_dfs.append(df)
    except Exception as e:
        continue

if sample_dfs:
    features_df = pd.concat(sample_dfs, ignore_index=True)
    print(f"  ✅ Loaded {len(features_df):,} clips for analysis")
    print(f"  Columns: {list(features_df.columns)[:20]}...")  # First 20 columns
else:
    print("  ❌ Failed to load features!")
    exit(1)

# Merge with cluster assignments
merged_df = clustered_df.merge(
    features_df,
    on=['filepath', 'logger', 'date'],
    how='inner'
)

print(f"\n  ✅ Merged: {len(merged_df):,} clips with both clusters and features")

# ============================================================================
# ANALYZE FEATURE DISTRIBUTIONS
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 80)

# Key features for rules
key_features = [
    'spectral_centroid_mean',
    'spectral_entropy_mean', 
    'temporal_entropy_mean',
    'aci',
    'zero_crossing_rate_mean',
    'rms_energy_mean',
    'spectral_rolloff_mean'
]

# Check which features exist
available_features = [f for f in key_features if f in merged_df.columns]
print(f"\nAvailable ecoacoustic features: {len(available_features)}/{len(key_features)}")
for f in available_features:
    print(f"  ✅ {f}")

missing_features = [f for f in key_features if f not in merged_df.columns]
if missing_features:
    print(f"\n⚠️  Missing features:")
    for f in missing_features:
        print(f"  ❌ {f}")

# ============================================================================
# ANALYZE BY CLUSTER
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER-SPECIFIC FEATURE ANALYSIS")
print("=" * 80)

for cluster_id in sorted(merged_df['cluster'].unique()):
    cluster_data = merged_df[merged_df['cluster'] == cluster_id]
    
    print(f"\n{'─' * 80}")
    print(f"CLUSTER {cluster_id} ({len(cluster_data):,} clips)")
    print(f"{'─' * 80}")
    
    for feature in available_features:
        values = cluster_data[feature].dropna()
        if len(values) > 0:
            print(f"\n{feature}:")
            print(f"  Min:    {values.min():.3f}")
            print(f"  25%:    {values.quantile(0.25):.3f}")
            print(f"  Median: {values.median():.3f}")
            print(f"  75%:    {values.quantile(0.75):.3f}")
            print(f"  Max:    {values.max():.3f}")
            print(f"  Mean:   {values.mean():.3f}")
            print(f"  Std:    {values.std():.3f}")

# ============================================================================
# CHECK RULE MATCHING WITH ACTUAL DATA
# ============================================================================

print("\n" + "=" * 80)
print("RULE MATCHING ANALYSIS")
print("=" * 80)

if 'spectral_centroid_mean' in merged_df.columns:
    print("\nChecking spectral_centroid_mean ranges:")
    
    # Original BIO rule: >1000 Hz
    bio_fish = (merged_df['spectral_centroid_mean'] > 1000).sum()
    bio_fish_pct = (bio_fish / len(merged_df)) * 100
    print(f"  Fish rule (>1000 Hz):        {bio_fish:,} clips ({bio_fish_pct:.1f}%)")
    
    # Original AMBIENT rule: <500 Hz  
    ambient_wave = (merged_df['spectral_centroid_mean'] < 500).sum()
    ambient_wave_pct = (ambient_wave / len(merged_df)) * 100
    print(f"  Wave rule (<500 Hz):         {ambient_wave:,} clips ({ambient_wave_pct:.1f}%)")
    
    # Original HUMAN rule: 50-500 Hz
    human_vessel = ((merged_df['spectral_centroid_mean'] >= 50) & 
                    (merged_df['spectral_centroid_mean'] <= 500)).sum()
    human_vessel_pct = (human_vessel / len(merged_df)) * 100
    print(f"  Vessel rule (50-500 Hz):     {human_vessel:,} clips ({human_vessel_pct:.1f}%)")
    
    # Show actual distribution
    print(f"\n  Actual spectral_centroid_mean distribution:")
    print(f"    < 100 Hz:     {(merged_df['spectral_centroid_mean'] < 100).sum():,} clips")
    print(f"    100-500 Hz:   {((merged_df['spectral_centroid_mean'] >= 100) & (merged_df['spectral_centroid_mean'] < 500)).sum():,} clips")
    print(f"    500-1000 Hz:  {((merged_df['spectral_centroid_mean'] >= 500) & (merged_df['spectral_centroid_mean'] < 1000)).sum():,} clips")
    print(f"    1000-5000 Hz: {((merged_df['spectral_centroid_mean'] >= 1000) & (merged_df['spectral_centroid_mean'] < 5000)).sum():,} clips")
    print(f"    > 5000 Hz:    {(merged_df['spectral_centroid_mean'] >= 5000).sum():,} clips")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

output_dir = ROOT_DIR / "data/diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Spectral centroid by cluster
if 'spectral_centroid_mean' in merged_df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for cluster_id in sorted(merged_df['cluster'].unique()):
        cluster_data = merged_df[merged_df['cluster'] == cluster_id]
        ax.hist(cluster_data['spectral_centroid_mean'], 
                bins=50, 
                alpha=0.5, 
                label=f'Cluster {cluster_id}')
    
    ax.axvline(x=500, color='red', linestyle='--', label='AMBIENT threshold (<500)')
    ax.axvline(x=1000, color='green', linestyle='--', label='BIO threshold (>1000)')
    ax.set_xlabel('Spectral Centroid (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Spectral Centroid Distribution by Cluster')
    ax.legend()
    ax.set_xlim(0, 10000)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_centroid_by_cluster.png", dpi=300)
    print(f"\n✅ Saved: {output_dir / 'spectral_centroid_by_cluster.png'}")
    plt.close()

# Plot 2: All features by cluster (boxplot)
if len(available_features) > 0:
    n_features = len(available_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(available_features):
        data_to_plot = []
        labels = []
        for cluster_id in sorted(merged_df['cluster'].unique()):
            cluster_values = merged_df[merged_df['cluster'] == cluster_id][feature].dropna()
            if len(cluster_values) > 0:
                data_to_plot.append(cluster_values)
                labels.append(f'Cluster {cluster_id}')
        
        if data_to_plot:
            axes[idx].boxplot(data_to_plot, labels=labels)
            axes[idx].set_title(feature)
            axes[idx].set_ylabel('Value')
            axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_features_by_cluster.png", dpi=300)
    print(f"✅ Saved: {output_dir / 'all_features_by_cluster.png'}")
    plt.close()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if 'spectral_centroid_mean' in merged_df.columns:
    median_centroid = merged_df['spectral_centroid_mean'].median()
    mean_centroid = merged_df['spectral_centroid_mean'].mean()
    
    print(f"\nYour data characteristics:")
    print(f"  Median spectral centroid: {median_centroid:.1f} Hz")
    print(f"  Mean spectral centroid:   {mean_centroid:.1f} Hz")
    
    print(f"\n💡 RECOMMENDED RULE ADJUSTMENTS:")
    
    # If most data is below 1000 Hz, adjust rules
    if median_centroid < 500:
        print(f"\n  ⚠️  Your data is dominated by low-frequency sounds!")
        print(f"  Suggested rules:")
        print(f"    AMBIENT (wave/wind): < {median_centroid*0.7:.0f} Hz")
        print(f"    HUMAN (vessels):     {median_centroid*0.7:.0f} - {median_centroid*1.5:.0f} Hz")
        print(f"    BIO (fish/chorus):   > {median_centroid*1.5:.0f} Hz")
    
    elif median_centroid < 1000:
        print(f"\n  Suggested rules:")
        print(f"    AMBIENT:  < {median_centroid*0.8:.0f} Hz")
        print(f"    HUMAN:    {median_centroid*0.8:.0f} - {median_centroid*1.2:.0f} Hz")
        print(f"    BIO:      > {median_centroid*1.2:.0f} Hz")
    
    else:
        print(f"\n  ✅ Your data range looks reasonable for standard rules!")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
print(f"\nCheck visualizations in: {output_dir}")
print(f"\nNext steps:")
print(f"  1. Review the feature distributions above")
print(f"  2. Check the visualization plots")
print(f"  3. Adjust rule thresholds based on your data")
print(f"  4. Re-run auto-labeling with updated rules")