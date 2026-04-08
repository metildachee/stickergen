import pandas as pd
from zhipuai import ZhipuAI
import json
import os

# Initialize ZhipuAI client
api_key = ""
client = ZhipuAI(api_key=api_key)  # Replace with your actual API key

# Load dataset
df = pd.read_csv('/data/metchee/projects/sticker_gen/dataset/new_all/release/test.csv')

# System prompt for query rewriting
system_prompt = """You are an assistant that rewrites short, incomplete, or noisy user search queries into clean image generation prompts by conservatively inferring the user's intended meaning.

Your task:
- Elaborate the query only to the extent needed to make the intent explicit and usable.
- Infer what the user is most likely trying to express, but be conservative.
- Replace the main subject with the literal placeholder token: TRIGGER_WORD.
- Preserve entities, attributes, actions, and relationships implied by the query.
- Do NOT introduce new people, objects, events, or opinions beyond what is strongly implied.
- Do NOT include any style, aesthetic, artistic, rendering, camera, or mood descriptors.
- Do NOT include real names or specific identities.
- Avoid explanations, meta commentary, or safety disclaimers.
- Output ONLY the rewritten prompt, no quotes, no extra text.

Assume the prompt will be used directly by an image generation model."""

def rewrite_query(user_query, history=None):
    """Rewrite user query using GLM-4 model"""
    try:
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Call GLM-4 API
        response = client.chat.completions.create(
            model="glm-4.7",
            messages=messages,
            stream=False,
            temperature=1
        )
        
        # Extract the rewritten prompt
        rewritten_prompt = response.choices[0].message.content.strip()
        print(f"Original: {user_query}")
        print(f"Rewritten: {rewritten_prompt}")
        print("-" * 50)
        
        return rewritten_prompt
        
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return user_query  # Return original query if error occurs

# Output file path
output_file = './short_v2.csv'

# Check if output file exists to determine starting point
if os.path.exists(output_file):
    try:
        # Load existing results to continue from where we left off
        existing_df = pd.read_csv(output_file)
        processed_count = len(existing_df)
        print(f"Found existing results with {processed_count} rows. Resuming...")
        
        # Start from the next row if not all rows are processed
        if processed_count < len(df):
            # Create a new DataFrame with all original columns
            results_df = df.copy()
            results_df['query_rewrite'] = None
            
            # Copy already processed results from existing file
            for idx in range(processed_count):
                if idx < len(results_df):
                    results_df.at[idx, 'query_rewrite'] = existing_df.at[idx, 'query_rewrite']
            
            start_index = processed_count
        else:
            print("All rows already processed!")
            exit()
    except Exception as e:
        print(f"Error loading existing file: {e}. Starting fresh...")
        results_df = df.copy()
        results_df['query_rewrite'] = None
        start_index = 0
        processed_count = 0
else:
    # Create a new DataFrame with all columns plus query_rewrite
    results_df = df.copy()
    results_df['query_rewrite'] = None
    start_index = 0
    processed_count = 0

# Process rows from start_index onwards
for index in range(start_index, len(df)):
    row = df.iloc[index]
    user_query = row['search_query']
    user_query = user_query.split(",")[0]
    
    # Optional: Use history if available
    history = None
    if 'history' in row and pd.notna(row['history']):
        history = row['history']
    
    # Rewrite the query
    rewritten_query = rewrite_query(user_query, history)
    
    # Update the results DataFrame with all original data
    # Copy all columns from the original row
    for col in df.columns:
        results_df.at[index, col] = row[col]
    
    # Add the rewritten query
    results_df.at[index, 'query_rewrite'] = rewritten_query
    
    # Create a dictionary with all data for this row
    row_data = {}
    for col in df.columns:
        row_data[col] = row[col]
    row_data['query_rewrite'] = rewritten_query
    
    # Convert to DataFrame for appending
    row_to_append = pd.DataFrame([row_data])
    
    # Write to CSV
    if index == start_index and not os.path.exists(output_file):
        # Write with header for first row
        row_to_append.to_csv(output_file, index=False, mode='w')
    else:
        # Append without header
        row_to_append.to_csv(output_file, index=False, mode='a', header=False)
    
    # Print progress
    processed_count = index - start_index + 1
    total_processed = start_index + processed_count
    if total_processed % 10 == 0 or total_processed == len(df):
        print(f"Processed {total_processed}/{len(df)} rows ({(total_processed/len(df))*100:.1f}%)")

print("\n" + "="*50)
print("Query rewriting completed!")
print(f"Processed all {len(df)} rows")
print("\nSample results:")
print(results_df[['search_query', 'query_rewrite']].head())
print("\nAll columns preserved:")
print(results_df.columns.tolist())
print(f"\nOutput saved to: {output_file}")