import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

# === Paths ===
HOME = Path(os.environ["HOME"])
REPO_ROOT = HOME / "Uni-stuff/semester-2/applied_Ml/reef_zmsc"

YAMNET_BASE = REPO_ROOT / "data/features/embeds_yamnet_50k/PAPCA"
ECOACOUSTIC_BASE = REPO_ROOT / "data/features/embeds_ecoacoustic_50k/PAPCA"
OUTPUT_BASE = REPO_ROOT / "data/features/embeds_fused_50k/PAPCA"


def find_matching_files(yamnet_base, ecoacoustic_base):
    """
    Find all logger/date combinations that exist in BOTH YAMNet and Ecoacoustic.
    Returns list of (logger, date, yamnet_path, ecoacoustic_path) tuples.
    """
    matches = []
    
    # Iterate through YAMNet structure
    for logger_dir in yamnet_base.iterdir():
        if not logger_dir.is_dir():
            continue
        
        logger = logger_dir.name
        
        for date_dir in logger_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            date = date_dir.name
            
            # Check if corresponding ecoacoustic file exists
            yamnet_file = date_dir / "features.parquet"
            ecoacoustic_file = ecoacoustic_base / logger / date / "features.parquet"
            
            if yamnet_file.exists() and ecoacoustic_file.exists():
                matches.append((logger, date, yamnet_file, ecoacoustic_file))
    
    return matches


def load_and_validate(yamnet_path, ecoacoustic_path):
    """
    Load both feature files and validate they can be merged.
    Returns (yamnet_df, ecoacoustic_df, merge_key)
    """
    yamnet_df = pd.read_parquet(yamnet_path)
    eco_df = pd.read_parquet(ecoacoustic_path)
    
    # Print columns for debugging
    # print(f"   YAMNet columns: {list(yamnet_df.columns)}")
    # print(f"   Ecoacoustic columns: {list(eco_df.columns)}")
    
    # Identify merge key
    yamnet_cols = set(yamnet_df.columns)
    eco_cols = set(eco_df.columns)
    
    common_cols = yamnet_cols & eco_cols
    
    # Use filepath and start_s as composite merge key for precision
    merge_key = ['filepath', 'start_s']
    
    return yamnet_df, eco_df, merge_key


def expand_yamnet_embeddings(yamnet_df):
    """
    Expand the yamnet_1024 list column into individual columns.
    Returns DataFrame with yamnet_0, yamnet_1, ..., yamnet_1023 columns.
    """
    # Extract the yamnet_1024 column (which contains lists)
    embeddings = yamnet_df['yamnet_1024'].tolist()
    
    # Convert to numpy array and then to DataFrame
    embeddings_array = np.array(embeddings)
    
    # Create column names: yamnet_0, yamnet_1, ..., yamnet_1023
    embed_cols = [f'yamnet_{i}' for i in range(embeddings_array.shape[1])]
    
    # Create DataFrame with expanded embeddings
    embed_df = pd.DataFrame(embeddings_array, columns=embed_cols)
    
    # Combine with metadata from original DataFrame
    metadata_cols = ['filepath', 'start_s', 'end_s', 'logger', 'date']
    metadata_df = yamnet_df[metadata_cols].reset_index(drop=True)
    
    result_df = pd.concat([metadata_df, embed_df], axis=1)
    
    return result_df


def fuse_features(yamnet_df, eco_df, merge_key):
    """
    Merge YAMNet and Ecoacoustic features.
    Returns fused DataFrame with combined embeddings.
    """
    # Expand YAMNet embeddings from list to individual columns
    print(f"   Expanding YAMNet embeddings...")
    yamnet_expanded = expand_yamnet_embeddings(yamnet_df)
    
    # Identify YAMNet embedding columns
    yamnet_embed_cols = [c for c in yamnet_expanded.columns if c.startswith('yamnet_')]
    
    # Identify ecoacoustic feature columns (exclude metadata)
    eco_meta_cols = {'filepath', 'logger', 'date', 'start_s', 'end_s'}
    eco_feature_cols = [c for c in eco_df.columns if c not in eco_meta_cols]
    
    print(f"   YAMNet: {len(yamnet_embed_cols)} dimensions")
    print(f"   Ecoacoustic: {len(eco_feature_cols)} dimensions")
    
    # Merge on common key
    fused = pd.merge(
        yamnet_expanded,
        eco_df,
        on=merge_key,
        how='inner',
        suffixes=('_yamnet', '_eco')
    )
    
    # Handle duplicate logger/date columns if they exist
    if 'logger_yamnet' in fused.columns and 'logger_eco' in fused.columns:
        fused['logger'] = fused['logger_yamnet']
        fused = fused.drop(['logger_yamnet', 'logger_eco'], axis=1)
    
    if 'date_yamnet' in fused.columns and 'date_eco' in fused.columns:
        fused['date'] = fused['date_yamnet']
        fused = fused.drop(['date_yamnet', 'date_eco'], axis=1)
    
    if 'end_s_yamnet' in fused.columns and 'end_s_eco' in fused.columns:
        fused['end_s'] = fused['end_s_yamnet']
        fused = fused.drop(['end_s_yamnet', 'end_s_eco'], axis=1)
    
    print(f"   Matched clips: {len(fused)}")
    
    # Construct final DataFrame: metadata + yamnet features + eco features
    output_cols = ['filepath', 'start_s', 'end_s', 'logger', 'date'] + yamnet_embed_cols + eco_feature_cols
    
    # Only include columns that exist in fused
    output_cols = [c for c in output_cols if c in fused.columns]
    
    output_df = fused[output_cols].copy()
    
    return output_df, len(yamnet_embed_cols), len(eco_feature_cols)


def main():
    print("=" * 70)
    print("FEATURE FUSION: YAMNet + Ecoacoustic (50K Sample)")
    print("=" * 70)
    
    # Find matching files
    print(f"\n🔍 Scanning for matching files...")
    print(f"   YAMNet base: {YAMNET_BASE}")
    print(f"   Ecoacoustic base: {ECOACOUSTIC_BASE}")
    
    matches = find_matching_files(YAMNET_BASE, ECOACOUSTIC_BASE)
    
    if not matches:
        print("\n❌ No matching logger/date combinations found!")
        print("   Make sure both YAMNet and Ecoacoustic features exist for the same data.")
        return
    
    print(f"\n✅ Found {len(matches)} matching logger/date combinations")
    
    # Process each match
    print(f"\n🔧 Fusing features...")
    
    total_clips = 0
    total_yamnet_dims = 0
    total_eco_dims = 0
    
    for logger, date, yamnet_path, eco_path in tqdm(matches, desc="Fusing"):
        # print(f"\n📦 Processing {logger}/{date}...")
        
        try:
            # Load and validate
            yamnet_df, eco_df, merge_key = load_and_validate(yamnet_path, eco_path)
            
            # Fuse features
            fused_df, yamnet_dims, eco_dims = fuse_features(yamnet_df, eco_df, merge_key)
            
            # Save fused features
            output_dir = OUTPUT_BASE / logger / date
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "features.parquet"
            fused_df.to_parquet(output_path, index=False, compression='snappy')
            
            # print(f"   ✓ Saved {len(fused_df)} clips → {output_path.relative_to(REPO_ROOT)}")
            
            total_clips += len(fused_df)
            total_yamnet_dims = yamnet_dims
            total_eco_dims = eco_dims
            
        except Exception as e:
            print(f"\n⚠️ Error processing {logger}/{date}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ FUSION COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Total clips fused: {total_clips:,}")
    print(f"   YAMNet dimensions: {total_yamnet_dims}")
    print(f"   Ecoacoustic dimensions: {total_eco_dims}")
    print(f"   Fusion vector size: {total_yamnet_dims + total_eco_dims} dimensions")
    
    print(f"\n💾 Output saved to: {OUTPUT_BASE.relative_to(REPO_ROOT)}")
    
    # Show output structure
    print(f"\n📂 Output structure by logger/date:")
    logger_summary = {}
    for logger_dir in sorted(OUTPUT_BASE.iterdir()):
        if logger_dir.is_dir():
            logger = logger_dir.name
            date_count = 0
            clip_count = 0
            
            for date_dir in sorted(logger_dir.iterdir()):
                if date_dir.is_dir():
                    parquet_file = date_dir / "features.parquet"
                    if parquet_file.exists():
                        df = pd.read_parquet(parquet_file)
                        clip_count += len(df)
                        date_count += 1
            
            logger_summary[logger] = (date_count, clip_count)
            print(f"   ✓ {logger}: {date_count} dates, {clip_count:,} clips")
    
    # Load one sample to show structure
    print(f"\n🔍 Sample fused feature structure:")
    sample_file = next(OUTPUT_BASE.rglob("features.parquet"))
    sample_df = pd.read_parquet(sample_file)
    
    print(f"\n   Metadata columns: {[c for c in sample_df.columns if c in ['filepath', 'start_s', 'end_s', 'logger', 'date']]}")
    print(f"   YAMNet columns: yamnet_0 ... yamnet_{total_yamnet_dims - 1}")
    
    eco_cols = [c for c in sample_df.columns if not c.startswith('yamnet_') and c not in ['filepath', 'start_s', 'end_s', 'logger', 'date']]
    print(f"   Ecoacoustic columns ({len(eco_cols)}): {eco_cols[:5]}... (showing first 5)")
    
    print(f"\n   Total columns: {len(sample_df.columns)}")
    print(f"   Shape: {sample_df.shape}")
    
    # Show first row sample (just metadata and first few features)
    print(f"\n   Sample row (metadata + first 3 yamnet + first 3 eco):")
    sample_cols = ['filepath', 'logger', 'date', 'start_s', 'yamnet_0', 'yamnet_1', 'yamnet_2'] + eco_cols[:3]
    sample_cols = [c for c in sample_cols if c in sample_df.columns]
    print(sample_df[sample_cols].head(1).T)
    

if __name__ == "__main__":
    main()