import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Simple user-based split
csv_path = "/data/metchee/projects/sticker_gen/dataset/new_all/release/search_based_sequences_dedup_duplicates_final_gt_not_in_history.csv"
output_dir = "/data/metchee/projects/sticker_gen/dataset/new_all/release"

# Load data
df = pd.read_csv(csv_path, dtype={'user_id': str})
print(f"Loaded {len(df):,} rows")
print(f"Unique users: {df['user_id'].nunique():,}")

# Get unique users
unique_users = df['user_id'].unique()

# Split users: 70% train, 15% validation, 15% test
train_users, temp_users = train_test_split(unique_users, test_size=0.3, random_state=42)
val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)

# Create train/val/test sets
train_df = df[df['user_id'].isin(train_users)].copy()
val_df = df[df['user_id'].isin(val_users)].copy()
test_df = df[df['user_id'].isin(test_users)].copy()

print(f"\nTrain set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
print(f"\nTrain users: {len(train_users):,}")
print(f"Validation users: {len(val_users):,}")
print(f"Test users: {len(test_users):,}")

# Save files
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(f"{output_dir}/train.csv", index=False)
val_df.to_csv(f"{output_dir}/val.csv", index=False)
test_df.to_csv(f"{output_dir}/test.csv", index=False)

print(f"\nFiles saved to {output_dir}:")
print(f"  train.csv: {len(train_df):,} rows")
print(f"  val.csv: {len(val_df):,} rows")
print(f"  test.csv: {len(test_df):,} rows")