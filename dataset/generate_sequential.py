import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
import ast
from tqdm import tqdm

def parse_history(history_str: str) -> List[Tuple[str, datetime]]:
    """Parse history string into list of (sticker_id, timestamp) tuples"""
    if pd.isna(history_str) or not history_str:
        return []
    
    items = []
    for item in history_str.split(','):
        if '|' in item:
            sticker_id, timestamp = item.strip().split('|')
            try:
                # Parse and make timezone-naive (remove timezone info)
                dt = pd.to_datetime(timestamp, errors='coerce')
                if pd.notna(dt):
                    dt = dt.tz_localize(None) if dt.tz is not None else dt
                    items.append((sticker_id, dt))
            except:
                continue
    return items

def create_sequential_training_examples(history_path: str, searches_path: str, output_path: str, min_sequence_length: int = 2):
    """
    Create sequential training examples where each example predicts the next sticker.
    
    For user with stickers [1, 2, 3, 4]:
    - Example 1: history=[1], predict sticker 2
    - Example 2: history=[1,2], predict sticker 3  
    - Example 3: history=[1,2,3], predict sticker 4
    """
    
    print("Loading data...")
    
    # Load data
    history_df = pd.read_csv(history_path, dtype={'user_id': str})
    searches_df = pd.read_csv(searches_path, dtype={'user_id': str, 'sticker_id': str})
    
    print(f"Users with history: {len(history_df)}")
    print(f"Search records: {len(searches_df)}")
    
    # Create a mapping from user_id to search queries for quick lookup
    print("\nBuilding search query index...")
    user_searches = {}
    for _, row in tqdm(searches_df.iterrows(), total=len(searches_df), desc="Indexing searches"):
        user_id = row['user_id']
        sticker_id = row['sticker_id']
        query = row['query']
        
        # Parse search timestamp (use first from datetime_list)
        datetime_objs = []
        if pd.notna(row['datetime_list']):
            clean_str = str(row['datetime_list']).strip('[]"\'')
            timestamps = [ts.strip() for ts in clean_str.split(',') if ts.strip()]
            for ts in timestamps:
                try:
                    dt = pd.to_datetime(ts.strip(" '\""), errors='coerce')
                    if pd.notna(dt):
                        dt = dt.tz_localize(None) if dt.tz is not None else dt
                        datetime_objs.append(dt)
                except:
                    continue
        
        search_time = datetime_objs[0] if datetime_objs else pd.NaT
        
        if pd.notna(search_time):
            if user_id not in user_searches:
                user_searches[user_id] = []
            user_searches[user_id].append({
                'sticker_id': sticker_id,
                'query': query,
                'timestamp': search_time
            })
    
    print(f"Users with search queries: {len(user_searches)}")
    
    # Process each user to create sequential examples
    print("\nCreating sequential training examples...")
    sequential_data = []
    stats = {
        'total_users': 0,
        'users_with_sequences': 0,
        'total_sequences': 0,
        'sequences_with_search': 0
    }
    
    for user_idx, row in tqdm(history_df.iterrows(), total=len(history_df), desc="Processing users"):
        user_id = row['user_id']
        stats['total_users'] += 1
        
        # Parse user's sticker history
        history_items = parse_history(row['history'])
        
        # Sort by timestamp
        history_items.sort(key=lambda x: x[1])
        
        # Need at least min_sequence_length stickers to create examples
        if len(history_items) < min_sequence_length:
            continue
        
        stats['users_with_sequences'] += 1
        
        # Get user's searches sorted by timestamp
        user_search_items = user_searches.get(user_id, [])
        user_search_items.sort(key=lambda x: x['timestamp'])
        
        # Create a combined timeline of stickers and searches
        timeline = []
        for sticker_id, sticker_time in history_items:
            timeline.append({
                'type': 'sticker',
                'id': sticker_id,
                'timestamp': sticker_time,
                'query': None
            })
        
        for search_item in user_search_items:
            timeline.append({
                'type': 'search',
                'id': search_item['sticker_id'],
                'timestamp': search_item['timestamp'],
                'query': search_item['query']
            })
        
        # Sort combined timeline by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        # Find search events and create training examples
        for i in range(len(timeline)):
            item = timeline[i]
            
            # We're looking for search events
            if item['type'] != 'search':
                continue
            
            # Get all stickers BEFORE this search
            prior_stickers = []
            for j in range(i):
                prior_item = timeline[j]
                if prior_item['type'] == 'sticker':
                    # Format: sticker_id|timestamp
                    formatted_item = f"{prior_item['id']}|{prior_item['timestamp'].strftime('%Y-%m-%dT%H:%M:%S')}"
                    prior_stickers.append(formatted_item)
            
            # Only create example if there are prior stickers
            if prior_stickers:
                history_str = ','.join(prior_stickers)
                search_sticker_id = item['id']
                search_query = item['query']
                search_timestamp = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                sequential_data.append({
                    'user_id': user_id,
                    'history': history_str,
                    'search_sticker_id': search_sticker_id,
                    'search_query': search_query,
                    'search_timestamp': search_timestamp
                })
                stats['sequences_with_search'] += 1
        
        # Also create sequences where we predict next sticker based on previous stickers
        # (even without search queries)
        for i in range(1, len(history_items)):
            # history: stickers[0:i], target: stickers[i]
            prior_stickers = []
            for j in range(i):
                sticker_id, sticker_time = history_items[j]
                formatted_item = f"{sticker_id}|{sticker_time.strftime('%Y-%m-%dT%H:%M:%S')}"
                prior_stickers.append(formatted_item)
            
            history_str = ','.join(prior_stickers)
            target_sticker_id, target_time = history_items[i]
            
            # Try to find if there was a search before this sticker
            search_query = None
            search_before_sticker = None
            
            # Look for searches that happened before this sticker
            for search_item in user_search_items:
                if search_item['timestamp'] < target_time:
                    # Check if this search led to this sticker
                    if search_item['sticker_id'] == target_sticker_id:
                        search_query = search_item['query']
                        search_before_sticker = search_item['timestamp']
                        break
            
            # Create example (even if no search found)
            sequential_data.append({
                'user_id': user_id,
                'history': history_str,
                'search_sticker_id': target_sticker_id,
                'search_query': search_query if search_query else "",
                'search_timestamp': search_before_sticker.strftime('%Y-%m-%d %H:%M:%S') if search_before_sticker else ""
            })
            stats['total_sequences'] += 1
    
    # Create DataFrame
    result_df = pd.DataFrame(sequential_data)
    
    # Remove duplicates and empty histories
    result_df = result_df[result_df['history'] != ''].copy()
    result_df = result_df.drop_duplicates()
    
    # Save to CSV
    print(f"\nSaving to {output_path}...")
    result_df.to_csv(output_path, index=False)
    
    print(f"\n=== Summary ===")
    print(f"Total users processed: {stats['total_users']}")
    print(f"Users with valid sequences: {stats['users_with_sequences']}")
    print(f"Total sequences created: {len(result_df)}")
    print(f"Sequences with search queries: {stats['sequences_with_search']}")
    print(f"Sequences without search queries: {len(result_df) - stats['sequences_with_search']}")
    
    # Show statistics
    if len(result_df) > 0:
        history_lengths = result_df['history'].apply(lambda x: len(x.split(',')) if x else 0)
        print(f"\nHistory length statistics:")
        print(f"Average history length: {history_lengths.mean():.1f} stickers")
        print(f"Median history length: {history_lengths.median():.1f} stickers")
        print(f"Min history length: {history_lengths.min()} stickers")
        print(f"Max history length: {history_lengths.max()} stickers")
        
        # Show sample
        print("\nSample of sequential training examples:")
        for i in range(min(5, len(result_df))):
            row = result_df.iloc[i]
            hist_items = row['history'].split(',')
            print(f"\nExample {i+1}:")
            print(f"  User: {row['user_id'][:15]}...")
            print(f"  History length: {len(hist_items)} stickers")
            print(f"  Target sticker: {row['search_sticker_id']}")
            print(f"  Search query: {row['search_query'][:50] if row['search_query'] else 'None'}")
            if hist_items:
                print(f"  Last sticker in history: {hist_items[-1][:30]}...")
    
    return result_df

# Alternative simpler version if you only want to predict stickers after searches
def create_search_based_sequences(history_path: str, searches_path: str, output_path: str):
    """
    Create sequences where we predict the sticker chosen after a search
    based on all stickers used before that search.
    """
    
    print("Creating search-based sequences...")
    
    # Load data
    history_df = pd.read_csv(history_path, dtype={'user_id': str})
    searches_df = pd.read_csv(searches_path, dtype={'user_id': str, 'sticker_id': str})
    
    # Parse search timestamps
    def get_search_timestamp(datetime_str):
        if pd.isna(datetime_str):
            return pd.NaT
        try:
            # Try to get first timestamp from the list
            clean_str = str(datetime_str).strip('[]"\'')
            timestamps = [ts.strip() for ts in clean_str.split(',') if ts.strip()]
            if timestamps:
                dt = pd.to_datetime(timestamps[0].strip(" '\""), errors='coerce')
                if pd.notna(dt):
                    return dt.tz_localize(None) if dt.tz is not None else dt
        except:
            pass
        return pd.NaT
    
    searches_df['search_timestamp'] = searches_df['datetime_list'].apply(get_search_timestamp)
    searches_df = searches_df[~searches_df['search_timestamp'].isna()]
    
    # Group searches by user
    user_searches = {}
    for _, row in searches_df.iterrows():
        user_id = row['user_id']
        if user_id not in user_searches:
            user_searches[user_id] = []
        user_searches[user_id].append({
            'sticker_id': row['sticker_id'],
            'query': row['query'],
            'timestamp': row['search_timestamp']
        })
    
    # Sort searches for each user by timestamp
    for user_id in user_searches:
        user_searches[user_id].sort(key=lambda x: x['timestamp'])
    
    # Process each user
    sequential_data = []
    
    for _, row in tqdm(history_df.iterrows(), total=len(history_df), desc="Processing users"):
        user_id = row['user_id']
        
        # Get user's sticker history
        history_items = parse_history(row['history'])
        history_items.sort(key=lambda x: x[1])
        
        # Get user's searches
        searches = user_searches.get(user_id, [])
        if not searches:
            continue
        
        # For each search, find all stickers used before it
        for search in searches:
            search_time = search['timestamp']
            prior_stickers = []
            
            for sticker_id, sticker_time in history_items:
                if sticker_time < search_time:
                    formatted_item = f"{sticker_id}|{sticker_time.strftime('%Y-%m-%dT%H:%M:%S')}"
                    prior_stickers.append(formatted_item)
                else:
                    break  # History is sorted by time
            
            if prior_stickers:
                sequential_data.append({
                    'user_id': user_id,
                    'history': ','.join(prior_stickers),
                    'search_sticker_id': search['sticker_id'],
                    'search_query': search['query'],
                    'search_timestamp': search_time.strftime('%Y-%m-%d %H:%M:%S')
                })
    
    result_df = pd.DataFrame(sequential_data)
    result_df.to_csv(output_path, index=False)
    
    print(f"\nCreated {len(result_df)} search-based sequences")
    return result_df

# Paths
history_path = "/data/metchee/projects/sticker_gen/dataset/all/history_all.csv"
searches_path = "/data/metchee/projects/sticker_gen/dataset/all/searches_all_duplicates.csv"

# Option 1: Create comprehensive sequential examples (predict every next sticker)
# output_path1 = "/data/metchee/projects/sticker_gen/dataset/new_allsequential_training_examples.csv"
# print("=" * 60)
# print("Creating Comprehensive Sequential Training Examples")
# print("=" * 60)
# df1 = create_sequential_training_examples(history_path, searches_path, output_path1, min_sequence_length=2)

# Option 2: Create only search-based sequences
output_path2 = "/data/metchee/projects/sticker_gen/dataset/new_all/search_based_sequences_duplicates.csv"
print("\n" + "=" * 60)
print("Creating Search-Based Sequences Only")
print("=" * 60)
df2 = create_search_based_sequences(history_path, searches_path, output_path2)

print("\nDone! Both datasets have been created.")