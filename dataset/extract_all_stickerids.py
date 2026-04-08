import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def extract_all_unique_sticker_ids(csv_path, output_dir=None):
    """
    Extract ALL unique sticker IDs from the dataset including:
    1. Stickers in the 'search_sticker_id' column
    2. Stickers in the 'history' column
    """
    
    print(f"Loading data from {csv_path}...")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None
    
    # Load the data
    df = pd.read_csv(csv_path, dtype={'user_id': str, 'search_sticker_id': str})
    
    print(f"Dataset loaded: {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Method 1: Extract from search_sticker_id column (easy)
    print("\n" + "="*60)
    print("METHOD 1: Extracting from search_sticker_id column")
    print("="*60)
    
    search_sticker_ids = df['search_sticker_id'].dropna().unique()
    print(f"Unique stickers from search_sticker_id: {len(search_sticker_ids):,}")
    
    # Method 2: Extract from history column (more complex)
    print("\n" + "="*60)
    print("METHOD 2: Extracting from history column")
    print("="*60)
    
    def extract_stickers_from_history(history_str):
        """Extract sticker IDs from history string"""
        if pd.isna(history_str) or history_str == '':
            return []
        
        sticker_ids = []
        items = history_str.split(',')
        for item in items:
            if '|' in item:
                sticker_id = item.split('|')[0].strip()
                if sticker_id:
                    sticker_ids.append(sticker_id)
        return sticker_ids
    
    # Extract all stickers from history
    all_history_stickers = []
    with tqdm(total=len(df), desc="Parsing history column") as pbar:
        for history in df['history'].dropna():
            sticker_ids = extract_stickers_from_history(history)
            all_history_stickers.extend(sticker_ids)
            pbar.update(1)
    
    history_sticker_ids = set(all_history_stickers)
    print(f"Unique stickers from history: {len(history_sticker_ids):,}")
    
    # Method 3: Combine both sources
    print("\n" + "="*60)
    print("METHOD 3: Combining all sources")
    print("="*60)
    
    # Combine all unique stickers
    all_unique_stickers = set(search_sticker_ids).union(history_sticker_ids)
    
    print(f"Total unique stickers found: {len(all_unique_stickers):,}")
    print(f"From search_sticker_id only: {len(set(search_sticker_ids) - history_sticker_ids):,}")
    print(f"From history only: {len(history_sticker_ids - set(search_sticker_ids)):,}")
    print(f"In both search and history: {len(set(search_sticker_ids).intersection(history_sticker_ids)):,}")
    
    # Convert to sorted list for consistent output
    all_unique_stickers_list = sorted(list(all_unique_stickers))
    
    # Save to files
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined list
        combined_path = os.path.join(output_dir, "all_unique_stickers.txt")
        with open(combined_path, 'w') as f:
            for sticker_id in all_unique_stickers_list:
                f.write(f"{sticker_id}\n")
        print(f"\nSaved combined list to: {combined_path}")
        
        # Save separate lists
        search_only_path = os.path.join(output_dir, "search_only_stickers.txt")
        search_only = sorted(list(set(search_sticker_ids) - history_sticker_ids))
        with open(search_only_path, 'w') as f:
            for sticker_id in search_only:
                f.write(f"{sticker_id}\n")
        print(f"Saved search-only stickers to: {search_only_path}")
        
        history_only_path = os.path.join(output_dir, "history_only_stickers.txt")
        history_only = sorted(list(history_sticker_ids - set(search_sticker_ids)))
        with open(history_only_path, 'w') as f:
            for sticker_id in history_only:
                f.write(f"{sticker_id}\n")
        print(f"Saved history-only stickers to: {history_only_path}")
        
        # Save as CSV with additional info
        csv_output_path = os.path.join(output_dir, "sticker_analysis.csv")
        sticker_data = []
        
        # Count frequencies
        from collections import Counter
        all_stickers_counter = Counter(all_history_stickers + list(search_sticker_ids))
        
        for sticker_id in all_unique_stickers_list:
            sticker_data.append({
                'sticker_id': sticker_id,
                'total_frequency': all_stickers_counter.get(sticker_id, 0),
                'in_search': 1 if sticker_id in set(search_sticker_ids) else 0,
                'in_history': 1 if sticker_id in history_sticker_ids else 0,
                'search_count': list(search_sticker_ids).count(sticker_id),
                'history_count': all_history_stickers.count(sticker_id)
            })
        
        sticker_df = pd.DataFrame(sticker_data)
        sticker_df.to_csv(csv_output_path, index=False)
        print(f"Saved detailed analysis to: {csv_output_path}")
        
        # Summary statistics
        print(f"\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print(f"Most frequent stickers:")
        top_stickers = sticker_df.sort_values('total_frequency', ascending=False).head(10)
        for idx, row in top_stickers.iterrows():
            print(f"  {row['sticker_id'][:20]}...: {row['total_frequency']:,} times")
        
        print(f"\nStickers by category:")
        print(f"  Only in searches: {len(search_only):,}")
        print(f"  Only in history: {len(history_only):,}")
        print(f"  In both: {len(set(search_sticker_ids).intersection(history_sticker_ids)):,}")
        
        print(f"\nFrequency distribution:")
        print(f"  Min frequency: {sticker_df['total_frequency'].min()}")
        print(f"  Max frequency: {sticker_df['total_frequency'].max()}")
        print(f"  Average frequency: {sticker_df['total_frequency'].mean():.1f}")
        print(f"  Median frequency: {sticker_df['total_frequency'].median():.1f}")
        
        # Stickers that appear only once
        once_stickers = sticker_df[sticker_df['total_frequency'] == 1]
        print(f"  Stickers appearing only once: {len(once_stickers):,} ({len(once_stickers)/len(sticker_df)*100:.1f}%)")
    
    return {
        'all_unique_stickers': all_unique_stickers_list,
        'search_stickers': sorted(list(search_sticker_ids)),
        'history_stickers': sorted(list(history_sticker_ids)),
        'search_only': sorted(list(set(search_sticker_ids) - history_sticker_ids)),
        'history_only': sorted(list(history_sticker_ids - set(search_sticker_ids))),
        'overlap': sorted(list(set(search_sticker_ids).intersection(history_sticker_ids)))
    }

# Simple version if you just want the list
def extract_simple_unique_sticker_ids(csv_path, output_file=None):
    """Simple version that just extracts and saves unique sticker IDs"""
    
    df = pd.read_csv(csv_path, dtype={'user_id': str, 'search_sticker_id': str})
    
    # Extract from search_sticker_id
    search_ids = set(df['search_sticker_id'].dropna().unique())
    
    # Extract from history
    history_ids = set()
    for history in df['history'].dropna():
        if history:
            items = history.split(',')
            for item in items:
                if '|' in item:
                    sticker_id = item.split('|')[0].strip()
                    if sticker_id:
                        history_ids.add(sticker_id)
    
    # Combine
    all_ids = sorted(list(search_ids.union(history_ids)))
    
    print(f"Total unique sticker IDs: {len(all_ids):,}")
    print(f"From searches: {len(search_ids):,}")
    print(f"From history: {len(history_ids):,}")
    
    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            for sticker_id in all_ids:
                f.write(f"{sticker_id}\n")
        print(f"Saved to: {output_file}")
    
    return all_ids

# Alternative: Extract with frequencies
def extract_stickers_with_frequencies(csv_path, output_file=None):
    """Extract stickers with their frequencies"""
    
    df = pd.read_csv(csv_path, dtype={'user_id': str, 'search_sticker_id': str})
    
    from collections import Counter
    
    # Count search stickers
    search_counter = Counter(df['search_sticker_id'].dropna())
    
    # Count history stickers
    history_counter = Counter()
    for history in df['history'].dropna():
        if history:
            items = history.split(',')
            for item in items:
                if '|' in item:
                    sticker_id = item.split('|')[0].strip()
                    if sticker_id:
                        history_counter[sticker_id] += 1
    
    # Combine counters
    total_counter = search_counter + history_counter
    
    # Create DataFrame
    sticker_data = []
    for sticker_id, total_count in total_counter.most_common():
        sticker_data.append({
            'sticker_id': sticker_id,
            'total_frequency': total_count,
            'search_frequency': search_counter.get(sticker_id, 0),
            'history_frequency': history_counter.get(sticker_id, 0),
            'in_search': 1 if sticker_id in search_counter else 0,
            'in_history': 1 if sticker_id in history_counter else 0
        })
    
    result_df = pd.DataFrame(sticker_data)
    
    print(f"Total unique stickers: {len(result_df):,}")
    print(f"Total occurrences: {result_df['total_frequency'].sum():,}")
    
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
    
    return result_df

# Main execution
if __name__ == "__main__":
    # File path
    csv_path = "/data/<>/projects/sticker_gen/dataset/new_all/release/search_based_sequences_dedup_duplicates_final_gt_not_in_history.csv"
    
    print("="*70)
    print("EXTRACTING UNIQUE STICKER IDs")
    print("="*70)
    
    # Option 1: Full analysis with multiple outputs
    output_dir = "/data/<>/projects/sticker_gen/dataset/new_all/sticker_ids"
    results = extract_all_unique_sticker_ids(csv_path, output_dir)
    
    # Option 2: Simple extraction (just the list)
    print("\n" + "="*70)
    print("SIMPLE EXTRACTION")
    print("="*70)
    
    simple_output = "/data/<>/projects/sticker_gen/dataset/new_all/all_unique_sticker_ids.txt"
    simple_list = extract_simple_unique_sticker_ids(csv_path, simple_output)
    
    # Option 3: With frequencies
    print("\n" + "="*70)
    print("EXTRACTION WITH FREQUENCIES")
    print("="*70)
    
    # freq_output = "/data/metchee/projects/sticker_gen/dataset/new_all/sticker_frequencies.csv"
    # freq_df = extract_stickers_with_frequencies(csv_path, freq_output)
    
    # Show some samples
    print("\n" + "="*70)
    print("SAMPLE STICKER IDs")
    print("="*70)
    
    if results and 'all_unique_stickers' in results:
        sample_size = min(20, len(results['all_unique_stickers']))
        print(f"First {sample_size} sticker IDs:")
        for i in range(sample_size):
            print(f"  {i+1:3d}. {results['all_unique_stickers'][i]}")