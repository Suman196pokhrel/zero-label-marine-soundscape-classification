def plot_variance_explained(explained_var, output_dir):
    """Plot cumulative variance explained by PCA components."""
    cumsum_var = np.cumsum(explained_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('Variance Explained by Each Component')
    ax1.set_xlim(0, min(50, len(explained_var)))
    
    # Cumulative variance
    ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o', markersize=3)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_variance_explained.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved variance plot")

def plot_umap_by_logger(umap_df, metadata, output_dir):
    """Plot UMAP colored by logger."""
    combined = pd.concat([metadata[['logger']], umap_df], axis=1)
    
    plt.figure(figsize=(10, 8))
    
    for logger in combined['logger'].unique():
        mask = combined['logger'] == logger
        plt.scatter(
            combined.loc[mask, 'umap_x'],
            combined.loc[mask, 'umap_y'],
            alpha=0.5,
            s=1,
            label=f'Logger {logger}'
        )
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Visualization colored by Logger')
    plt.legend(markerscale=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_by_logger.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved UMAP visualization")

def save_preprocessed_data(metadata, features_scaled, features_pca, umap_df, output_dir):
    """Save all preprocessed data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving preprocessed data...")
    
    # Combine everything
    final_df = pd.concat([
        metadata.reset_index(drop=True),
        features_scaled.reset_index(drop=True),
        features_pca.reset_index(drop=True),
        umap_df.reset_index(drop=True)
    ], axis=1)
    
    # Save full dataset
    output_file = output_dir / "preprocessed_features_full.parquet"
    final_df.to_parquet(output_file, index=False, compression='snappy')
    print(f"   ✓ Full dataset: {output_file.relative_to(REPO_ROOT)}")
    
    # Save PCA-only version (most useful for clustering)
    pca_cols = [c for c in final_df.columns if c.startswith('pca_')]
    pca_df = pd.concat([
        metadata.reset_index(drop=True),
        final_df[pca_cols + ['umap_x', 'umap_y']].reset_index(drop=True)
    ], axis=1)
    
    output_file_pca = output_dir / "preprocessed_features_pca.parquet"
    pca_df.to_parquet(output_file_pca, index=False, compression='snappy')
    print(f"   ✓ PCA version: {output_file_pca.relative_to(REPO_ROOT)}")
    
    print(f"\n   Saved {len(final_df):,} clips")
    print(f"   Full shape: {final_df.shape}")
    print(f"   PCA shape: {pca_df.shape}")

def main():
    print("=" * 70)
    print("FEATURE PREPROCESSING: Standardization + PCA + UMAP")
    print("=" * 70)
    
    # Load data
    df = load_all_fused_features()
    
    # Separate metadata and features
    metadata, features, yamnet_cols, eco_cols = separate_metadata_features(df)
    
    # Check for missing values
    missing = features.isnull().sum().sum()
    if missing > 0:
        print(f"\n⚠️ Found {missing} missing values, filling with 0")
        features = features.fillna(0)
    
    # 1. Standardization
    features_scaled, scaler = standardize_features(features)
    
    # 2. PCA
    features_pca, pca, explained_var = apply_pca(features_scaled, variance_threshold=PCA_VARIANCE)
    
    # 3. UMAP for visualization
    umap_df, umap_reducer = apply_umap_for_viz(features_pca)
    
    # 4. Visualizations
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Creating visualizations...")
    plot_variance_explained(explained_var, OUTPUT_BASE)
    plot_umap_by_logger(umap_df, metadata, OUTPUT_BASE)
    
    # 5. Save preprocessed data
    save_preprocessed_data(metadata, features_scaled, features_pca, umap_df, OUTPUT_BASE)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Original dimensions: {len(features.columns)}")
    print(f"   PCA dimensions: {len(features_pca.columns)}")
    print(f"   Reduction: {(1 - len(features_pca.columns)/len(features.columns))*100:.1f}%")
    print(f"   Variance preserved: {np.sum(explained_var)*100:.2f}%")
    
    print(f"\n💾 Output files:")
    print(f"   • preprocessed_features_full.parquet - All features (scaled + PCA + UMAP)")
    print(f"   • preprocessed_features_pca.parquet - Metadata + PCA + UMAP (use this for clustering)")
    print(f"   • pca_variance_explained.png - PCA analysis plot")
    print(f"   • umap_by_logger.png - UMAP visualization")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Use 'preprocessed_features_pca.parquet' for clustering")
    print(f"   2. Try different clustering algorithms (K-means, HDBSCAN, GMM)")
    print(f"   3. Use UMAP coordinates for visualization")
    

if __name__ == "__main__":
    main()