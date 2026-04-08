import pandas as pd

# Load your CSV file
df = pd.read_csv('/data/metchee/projects/sticker_gen/dataset/new_all/release/search_based_sequences_dedup_duplicates_final.csv')

def remove_search_sticker_from_history(row):
    """
    Remove the search_sticker_id occurrence from the history column.
    History format: "sticker_id1|timestamp1,sticker_id2|timestamp2,..."
    """
    if pd.isna(row['history']) or pd.isna(row['search_sticker_id']):
        return row['history']
    
    search_id = str(row['search_sticker_id'])
    history_items = str(row['history']).split(',')
    
    # Filter out items where the sticker_id matches search_sticker_id
    filtered_items = []
    for item in history_items:
        if '|' in item:
            sticker_id = item.split('|')[0]
            if sticker_id != search_id:
                filtered_items.append(item)
        else:
            # Handle case where item doesn't have timestamp
            if item != search_id:
                filtered_items.append(item)
    
    # Join back with comma
    return ','.join(filtered_items)

# Apply the function to each row
df['history'] = df.apply(remove_search_sticker_from_history, axis=1)

# Save the original shape for comparison
original_shape = df.shape
original_row_count = len(df)

# Remove rows where history is empty or NaN after cleaning
# First, handle cases where history might be empty string after cleaning
df['history_cleaned'] = df['history'].fillna('')
df['history_cleaned'] = df['history_cleaned'].astype(str)

# Remove rows where history is empty or contains only commas/spaces
df = df[df['history_cleaned'].str.strip().astype(bool)]

# Drop the helper column
df = df.drop('history_cleaned', axis=1)

# Save the cleaned DataFrame
output_path = '/data/metchee/projects/sticker_gen/dataset/new_all/release/search_based_sequences_dedup_duplicates_final_gt_not_in_history.csv'
df.to_csv(output_path, index=False)

print(f"Cleaning complete! File saved to: {output_path}")
print(f"Original shape: {original_shape}")
print(f"Rows removed due to empty history: {original_row_count - len(df)}")
print(f"New shape: {df.shape}")
print(f"Remaining rows: {len(df)}")