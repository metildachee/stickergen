# ============================================================================
# SIMPLE STICKER ORGANIZATION FOR LoRA TRAINING
# ============================================================================
import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    STYLE_LABEL_CSV = "/data/<>/projects/sticker_gen/style_labeling/sticker_style_label_k30.csv"
    STICKERS_PNG_PATH = "/data/<>/data/stickerqueries/stickers_png"
    OUTPUT_DIR = "/data/<>/projects/sticker_gen/style_labeling/lora_training_data"
    
    # Processing settings
    COPY_MODE = True  # True to copy files, False to create symlinks (saves space)
    VERIFY_EXISTENCE = True  # Verify files exist before copying
    MIN_CLUSTER_SIZE = 1  # Minimum number of stickers per cluster (1 = include all)
    
    # Data selection
    USE_ONLY_INCLUDED = True  # Only use stickers with included_in_revision=True
    
    # Organization
    CLEAN_FOLDER_NAMES = True  # Clean special characters from folder names
    CREATE_SUMMARY_FILES = True  # Create summary CSV and JSON files

config = Config()

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================
def load_and_validate_data():
    """Load the style label CSV and validate sticker existence"""
    print(f"📂 Loading style labels from {config.STYLE_LABEL_CSV}...")
    
    df = pd.read_csv(config.STYLE_LABEL_CSV)
    
    # Filter data if needed
    if config.USE_ONLY_INCLUDED and 'included_in_revision' in df.columns:
        original_count = len(df)
        df = df[df['included_in_revision'] == True]
        print(f"  ✅ Filtered to included stickers: {len(df)}/{original_count} ({len(df)/original_count*100:.1f}%)")
    
    # Remove stickers with no cluster name
    df = df[df['new_cluster_name'].notna() & (df['new_cluster_name'] != '')]
    
    # Remove stickers with new_cluster_id = -1
    df = df[df['new_cluster_id'] != -1]
    
    print(f"\n📊 Data Summary:")
    print(f"  Total stickers: {len(df)}")
    print(f"  Unique clusters: {df['new_cluster_id'].nunique()}")
    
    # Verify sticker files exist
    valid_data = []
    missing_files = []
    
    if config.VERIFY_EXISTENCE:
        print(f"\n🔍 Verifying sticker files...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
            sticker_path = row['sticker_path']
            
            # Handle different path formats
            if pd.isna(sticker_path):
                missing_files.append(row['sticker_id'])
                continue
                
            # If path is just filename, prepend base path
            if not os.path.isabs(str(sticker_path)):
                full_path = os.path.join(config.STICKERS_PNG_PATH, os.path.basename(str(sticker_path)))
            else:
                full_path = str(sticker_path)
            
            # Check if file exists
            if os.path.exists(full_path):
                valid_data.append({
                    'sticker_id': row['sticker_id'],
                    'sticker_path': full_path,
                    'cluster_id': row['new_cluster_id'],
                    'cluster_name': row['new_cluster_name'],
                    'distance': row.get('distance_to_centroid', 0),
                    'original_cluster': row.get('cluster_id', -1)
                })
            else:
                missing_files.append(row['sticker_id'])
        
        print(f"  ✅ Valid files: {len(valid_data)}")
        if missing_files:
            print(f"  ⚠️ Missing files: {len(missing_files)}")
            if len(missing_files) <= 5:
                for f in missing_files[:5]:
                    print(f"    - {f}")
    else:
        # Assume all files exist
        for _, row in df.iterrows():
            sticker_path = row['sticker_path']
            if not os.path.isabs(str(sticker_path)):
                full_path = os.path.join(config.STICKERS_PNG_PATH, os.path.basename(str(sticker_path)))
            else:
                full_path = str(sticker_path)
            
            valid_data.append({
                'sticker_id': row['sticker_id'],
                'sticker_path': full_path,
                'cluster_id': row['new_cluster_id'],
                'cluster_name': row['new_cluster_name'],
                'distance': row.get('distance_to_centroid', 0),
                'original_cluster': row.get('cluster_id', -1)
            })
    
    df_valid = pd.DataFrame(valid_data)
    
    # Filter out small clusters
    if config.MIN_CLUSTER_SIZE > 1:
        cluster_sizes = df_valid.groupby('cluster_id').size()
        valid_clusters = cluster_sizes[cluster_sizes >= config.MIN_CLUSTER_SIZE].index
        df_valid = df_valid[df_valid['cluster_id'].isin(valid_clusters)]
        
        print(f"\n📉 Filtered out clusters with < {config.MIN_CLUSTER_SIZE} stickers")
        print(f"  Remaining clusters: {len(valid_clusters)}")
        print(f"  Remaining stickers: {len(df_valid)}")
    
    return df_valid

# ============================================================================
# FOLDER NAME CLEANING
# ============================================================================
def clean_folder_name(name):
    """Clean cluster name to be a valid folder name"""
    if not config.CLEAN_FOLDER_NAMES:
        return str(name)
    
    # Remove invalid characters for Windows/Linux/Mac
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove multiple underscores
    while '__' in name:
        name = name.replace('__', '_')
    
    # Strip leading/trailing underscores and dots
    name = name.strip('_. ')
    
    # Limit length
    if len(name) > 50:
        name = name[:50]
    
    # Ensure it's not empty
    if not name:
        name = 'unknown_cluster'
    
    return name

# ============================================================================
# FILE ORGANIZATION
# ============================================================================
def organize_stickers_simple(df):
    """Simply organize stickers into cluster folders"""
    print(f"\n📁 Organizing stickers into cluster folders...")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Group by cluster
    clusters = df.groupby(['cluster_id', 'cluster_name'])
    
    # Track statistics
    stats = {
        'total_processed': 0,
        'total_copied': 0,
        'total_errors': 0,
        'clusters_created': 0,
        'clusters_skipped': 0
    }
    
    # Process each cluster
    for (cluster_id, cluster_name), cluster_data in clusters:
        # Clean folder name
        folder_name = clean_folder_name(cluster_name)
        cluster_dir = os.path.join(config.OUTPUT_DIR, folder_name)
        
        # Skip if folder already exists and has files
        if os.path.exists(cluster_dir) and len(os.listdir(cluster_dir)) > 0:
            print(f"  ⚠️ Folder '{folder_name}' already exists, skipping...")
            stats['clusters_skipped'] += 1
            continue
        
        # Create cluster directory
        os.makedirs(cluster_dir, exist_ok=True)
        stats['clusters_created'] += 1
        
        print(f"\n  📂 Cluster: '{cluster_name}' -> folder: '{folder_name}'")
        print(f"    Stickers: {len(cluster_data)}")
        print(f"    Directory: {cluster_dir}")
        
        # Copy each sticker
        cluster_processed = 0
        cluster_copied = 0
        cluster_errors = 0
        
        for _, row in tqdm(cluster_data.iterrows(), total=len(cluster_data), desc=f"  Copying stickers"):
            src_path = row['sticker_path']
            sticker_filename = os.path.basename(src_path)
            dst_path = os.path.join(cluster_dir, sticker_filename)
            
            stats['total_processed'] += 1
            cluster_processed += 1
            
            try:
                if config.COPY_MODE:
                    shutil.copy2(src_path, dst_path)
                else:
                    # Create relative symlink
                    rel_path = os.path.relpath(src_path, os.path.dirname(dst_path))
                    os.symlink(rel_path, dst_path)
                
                stats['total_copied'] += 1
                cluster_copied += 1
                
            except Exception as e:
                stats['total_errors'] += 1
                cluster_errors += 1
                print(f"    ❌ Error copying {sticker_filename}: {e}")
        
        print(f"    ✅ Copied: {cluster_copied}/{cluster_processed} | Errors: {cluster_errors}")
    
    return stats

# ============================================================================
# SUMMARY FILES CREATION
# ============================================================================
def create_summary_files(df, stats):
    """Create summary CSV and JSON files"""
    if not config.CREATE_SUMMARY_FILES:
        return
    
    print(f"\n📝 Creating summary files...")
    
    # Save the full dataset CSV
    output_csv = os.path.join(config.OUTPUT_DIR, 'all_stickers.csv')
    df.to_csv(output_csv, index=False)
    print(f"  ✅ Saved sticker list: {output_csv}")
    
    # Create cluster summary
    cluster_summary = df.groupby(['cluster_id', 'cluster_name']).agg(
        sticker_count=('sticker_id', 'count'),
        avg_distance=('distance', 'mean'),
        min_distance=('distance', 'min'),
        max_distance=('distance', 'max'),
        sample_stickers=('sticker_id', lambda x: list(x[:3]))  # First 3 sticker IDs
    ).reset_index()
    
    # Add folder name
    cluster_summary['folder_name'] = cluster_summary['cluster_name'].apply(clean_folder_name)
    
    cluster_csv = os.path.join(config.OUTPUT_DIR, 'cluster_summary.csv')
    cluster_summary.to_csv(cluster_csv, index=False)
    print(f"  ✅ Saved cluster summary: {cluster_csv}")
    
    # Create JSON summary
    import json
    summary_data = {
        'organization_date': pd.Timestamp.now().isoformat(),
        'total_stickers': len(df),
        'total_clusters': df['cluster_id'].nunique(),
        'operation_stats': stats,
        'copy_mode': 'copy' if config.COPY_MODE else 'symlink',
        'clusters': []
    }
    
    for _, row in cluster_summary.iterrows():
        summary_data['clusters'].append({
            'cluster_id': int(row['cluster_id']),
            'cluster_name': row['cluster_name'],
            'folder_name': row['folder_name'],
            'sticker_count': int(row['sticker_count']),
            'avg_distance': float(row['avg_distance']),
            'folder_path': os.path.join(config.OUTPUT_DIR, row['folder_name'])
        })
    
    summary_json = os.path.join(config.OUTPUT_DIR, 'dataset_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"  ✅ Saved JSON summary: {summary_json}")
    
    # Create a simple README
    readme_content = f"""# LoRA Training Dataset - Sticker Styles

## Dataset Information
- **Created**: {summary_data['organization_date']}
- **Total Stickers**: {summary_data['total_stickers']}
- **Total Styles/Clusters**: {summary_data['total_clusters']}
- **Organization**: {'File copies' if config.COPY_MODE else 'Symbolic links'}
"""
    
def organize_stickers_for_lora():
    """Main function to organize stickers into cluster folders"""
    print("="*80)
    print("🎨 ORGANIZING STICKERS FOR LoRA TRAINING")
    print("="*80)
    
    # Step 1: Load data
    print(f"\n1. LOADING DATA")
    print("-"*40)
    df = load_and_validate_data()
    
    if len(df) == 0:
        print("❌ No valid data found! Exiting...")
        return None
    
    # Show cluster distribution
    print(f"\n📈 Cluster Distribution:")
    cluster_counts = df.groupby('cluster_name').size().sort_values(ascending=False)
    for cluster_name, count in cluster_counts.items():
        folder_name = clean_folder_name(cluster_name)
        print(f"  {folder_name}: {count} stickers")
    
    # Step 2: Organize stickers
    print(f"\n2. ORGANIZING STICKERS")
    print("-"*40)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Mode: {'Copying files' if config.COPY_MODE else 'Creating symlinks'}")
    
    stats = organize_stickers_simple(df)
    
    # Step 3: Create summary files
    print(f"\n3. CREATING SUMMARY FILES")
    print("-"*40)
    create_summary_files(df, stats)
    
    # Step 4: Display final results
    print(f"\n4. FINAL RESULTS")
    print("-"*40)
    
    print(f"\n✅ ORGANIZATION COMPLETE!")
    print(f"="*80)
    print(f"📊 Statistics:")
    print(f"  Total stickers processed: {stats['total_processed']}")
    print(f"  Successfully copied: {stats['total_copied']}")
    print(f"  Errors: {stats['total_errors']}")
    print(f"  Clusters created: {stats['clusters_created']}")
    print(f"  Clusters skipped: {stats['clusters_skipped']}")
    
    # Show folder structure
    print(f"\n📁 Folder Structure:")
    print(f"{config.OUTPUT_DIR}/")
    
    # List first 10 folders
    folders = sorted([f for f in os.listdir(config.OUTPUT_DIR) if os.path.isdir(os.path.join(config.OUTPUT_DIR, f))])
    for folder in folders:
        folder_path = os.path.join(config.OUTPUT_DIR, folder)
        sticker_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
        print(f"  ├── {folder}/ ({sticker_count} stickers)")
    
    # if len(folders) > 10:
    #     print(f"  └── ... and {len(folders) - 10} more folders")
    
    print(f"\n🎯 Ready for LoRA Training!")
    print(f"  All stickers are organized in: {config.OUTPUT_DIR}")
    print(f"  Each folder is a different style/cluster")
    print(f"="*80)
    
    return config.OUTPUT_DIR

# ============================================================================
# QUICK UTILITY FUNCTIONS
# ============================================================================
def quick_organize(csv_path=None, output_dir=None, copy_mode=True, min_cluster_size=1):
    """Quick organization function"""
    if csv_path:
        config.STYLE_LABEL_CSV = csv_path
    if output_dir:
        config.OUTPUT_DIR = output_dir
    config.COPY_MODE = copy_mode
    config.MIN_CLUSTER_SIZE = min_cluster_size
    
    return organize_stickers_for_lora()

def check_disk_space_requirements(df):
    """Estimate disk space needed"""
    print(f"\n💾 Estimating disk space requirements...")
    
    total_size = 0
    sample_files = 0
    
    # Sample 100 files to estimate average size
    sample_df = df.sample(min(100, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        try:
            total_size += os.path.getsize(row['sticker_path'])
            sample_files += 1
        except:
            pass
    
    if sample_files > 0:
        avg_file_size = total_size / sample_files
        estimated_total = avg_file_size * len(df)
        
        print(f"  Average file size: {avg_file_size / 1024:.2f} KB")
        print(f"  Estimated total: {estimated_total / 1024 / 1024:.2f} MB")
        print(f"  Estimated total (copy mode): {estimated_total / 1024 / 1024:.2f} MB")
        print(f"  Estimated total (symlink mode): ~1-2 MB (just symlinks)")
        
        return estimated_total
    return None

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run the organization
    output_dir = organize_stickers_for_lora()
    
    if output_dir:
        print(f"\n✨ All done! Your stickers are organized and ready for LoRA training!")
        print(f"📁 Location: {output_dir}")
        print(f"\nNext steps:")
        print(f"1. Check the README.md file in the output directory")
        print(f"2. Use the folders directly in Kohya SS or Automatic1111")
        print(f"3. Start training your style LoRA! 🎨")

# ============================================================================
# FOR NOTEBOOK USE
# ============================================================================
def organize_from_notebook(csv_path, output_base_dir=None, use_symlinks=False):
    """Easy function to call from notebook"""
    # Set config
    config.STYLE_LABEL_CSV = csv_path
    
    if output_base_dir:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        config.OUTPUT_DIR = os.path.join(output_base_dir, f"lora_{base_name}")
    
    config.COPY_MODE = not use_symlinks
    
    print(f"📂 Input CSV: {config.STYLE_LABEL_CSV}")
    print(f"📁 Output directory: {config.OUTPUT_DIR}")
    print(f"🔗 Using symlinks: {use_symlinks}")
    
    return organize_stickers_for_lora()