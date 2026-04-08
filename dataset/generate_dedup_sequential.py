import pandas as pd
import os

def remove_duplicate_rows(input_file, output_file):
    """
    Remove duplicate rows where ENTIRE row is the same.
    Save to new file.
    """
    
    print(f"Loading data from {input_file}...")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return
    
    # Load the data
    df = pd.read_csv(input_file, dtype={'user_id': str})
    
    print(f"Original dataset size: {len(df):,} rows")
    print(f"Original unique rows: {df.drop_duplicates().shape[0]:,} rows")
    
    # Count duplicates before removal
    duplicate_count = len(df) - len(df.drop_duplicates())
    print(f"Number of duplicate rows to remove: {duplicate_count:,}")
    
    if duplicate_count == 0:
        print("No duplicates found. Copying file as-is...")
        df.to_csv(output_file, index=False)
        print(f"File saved to: {output_file}")
        return df
    
    # Remove duplicates (keeping first occurrence)
    df_dedup = df.drop_duplicates(keep='first')
    
    print(f"\nAfter removing duplicates:")
    print(f"  New dataset size: {len(df_dedup):,} rows")
    print(f"  Rows removed: {duplicate_count:,} rows")
    print(f"  Reduction: {duplicate_count/len(df)*100:.2f}%")
    
    # Show some examples of removed duplicates if there are any
    if duplicate_count > 0 and duplicate_count <= 10:
        print(f"\nExamples of duplicate rows that were removed:")
        duplicates = df[df.duplicated(keep=False)]
        sample_duplicates = duplicates.drop_duplicates().head(3)
        for idx, row in sample_duplicates.iterrows():
            print(f"\nDuplicate example {idx}:")
            for col in df.columns:
                print(f"  {col}: {str(row[col])[:50]}{'...' if len(str(row[col])) > 50 else ''}")
    
    # Analyze which columns have the most duplicates
    print(f"\nDuplicate analysis by column combinations:")
    
    # Check for duplicates in specific column combinations
    column_combinations = [
        ['user_id', 'search_sticker_id', 'search_query', 'search_timestamp'],  # All columns
        ['user_id', 'search_sticker_id', 'search_query'],  # Without timestamp
        ['user_id', 'search_sticker_id'],  # Just user + sticker
        ['search_sticker_id', 'search_query'],  # Just sticker + query
    ]
    
    for cols in column_combinations:
        if all(col in df.columns for col in cols):
            dup_count = len(df) - len(df.drop_duplicates(subset=cols))
            if dup_count > 0:
                print(f"  Duplicates in {cols}: {dup_count:,} rows")
    
    # Save to new file
    print(f"\nSaving deduplicated data to {output_file}...")
    df_dedup.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df_dedup):,} rows to {output_file}")
    
    # Verify the deduplication
    print(f"\nVerification:")
    df_verify = pd.read_csv(output_file)
    verify_duplicates = len(df_verify) - len(df_verify.drop_duplicates())
    print(f"  Duplicates in output file: {verify_duplicates}")
    
    if verify_duplicates == 0:
        print("  ✓ Successfully removed all duplicates!")
    else:
        print("  ⚠ Warning: Some duplicates remain")
    
    return df_dedup

# Alternative: Remove duplicates but keep count of occurrences
def remove_duplicates_with_counts(input_file, output_file):
    """
    Remove duplicates and add a column showing how many times each row occurred.
    """
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, dtype={'user_id': str})
    
    print(f"Original size: {len(df):,} rows")
    
    # Count duplicates before removal
    duplicate_counts = df.groupby(list(df.columns)).size().reset_index(name='occurrence_count')
    
    print(f"Unique rows: {len(duplicate_counts):,}")
    print(f"Total duplicates: {len(df) - len(duplicate_counts):,}")
    
    # Show distribution of occurrence counts
    print(f"\nOccurrence distribution:")
    occurrence_stats = duplicate_counts['occurrence_count'].value_counts().sort_index()
    for count, freq in occurrence_stats.items():
        print(f"  {count} occurrence{'s' if count > 1 else ''}: {freq:,} rows")
    
    # Save with count column
    duplicate_counts.to_csv(output_file, index=False)
    print(f"\nSaved {len(duplicate_counts):,} unique rows to {output_file}")
    print(f"Added 'occurrence_count' column showing duplicate frequency")
    
    return duplicate_counts

# Alternative: Remove duplicates based on specific columns only
def remove_duplicates_specific_columns(input_file, output_file, columns_to_check):
    """
    Remove duplicates based on specific columns only.
    """
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, dtype={'user_id': str})
    
    print(f"Original size: {len(df):,} rows")
    print(f"Checking duplicates in columns: {columns_to_check}")
    
    # Count duplicates in specified columns
    before = len(df)
    df_dedup = df.drop_duplicates(subset=columns_to_check, keep='first')
    after = len(df_dedup)
    
    print(f"\nAfter removing duplicates in specified columns:")
    print(f"  New size: {after:,} rows")
    print(f"  Rows removed: {before - after:,}")
    print(f"  Reduction: {(before - after)/before*100:.2f}%")
    
    # Save
    df_dedup.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    return df_dedup

# Main execution
if __name__ == "__main__":
    # File paths
    input_file = "/data/<>/projects/sticker_gen/dataset/new_all/search_based_sequences_duplicates.csv"
    output_file = "/data/<>/projects/sticker_gen/dataset/new_all/search_based_sequences_dedup_duplicates.csv"
    
    print("=" * 70)
    print("REMOVING DUPLICATE ROWS")
    print("=" * 70)
    
    # Option 1: Remove all duplicates (entire row must match)
    print("\n[Option 1] Removing all duplicate rows (entire row must match):")
    dedup_df = remove_duplicate_rows(input_file, output_file)

# If you want to also see which specific rows are most duplicated:
def analyze_top_duplicates(input_file, top_n=10):
    """Show the most frequently duplicated rows"""
    
    print(f"\nAnalyzing top {top_n} most duplicated rows...")
    df = pd.read_csv(input_file, dtype={'user_id': str})
    
    # Group by all columns to find duplicates
    grouped = df.groupby(list(df.columns)).size().reset_index(name='count')
    grouped = grouped.sort_values('count', ascending=False)
    
    print(f"\nTop {top_n} most duplicated rows:")
    print("-" * 80)
    
    for i in range(min(top_n, len(grouped))):
        row = grouped.iloc[i]
        print(f"\n#{i+1}: Appears {int(row['count'])} times")
        print(f"  User: {row['user_id'][:20]}...")
        print(f"  Query: {str(row['search_query'])[:50]}...")
        print(f"  Sticker: {row['search_sticker_id']}")
        print(f"  Timestamp: {row['search_timestamp']}")
        history_items = len(str(row['history']).split(',')) if pd.notna(row['history']) and row['history'] != '' else 0
        print(f"  History items: {history_items}")
    
    return grouped

# Run the duplicate analysis
top_duplicates = analyze_top_duplicates(input_file, top_n=5)