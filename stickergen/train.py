#!/usr/bin/env python3
"""
Batch Sticker Generator for Test Dataset
Process test.csv to generate stickers for all search queries using style from search_sticker_id
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import json
import argparse
from pathlib import Path
import subprocess
from PIL import Image
import re

# ================= CONFIGURATION =================
TEST_CSV_PATH = "/data/<>/projects/sticker_gen/dataset/new_all/release/test.csv"
STYLE_CSV_PATH = "/data/<>/projects/sticker_gen/style_prediction/single_query_only/baseline4b_predictions_add.csv"
STICKER_DIR = "/data/<>/data/sticker-gen/sticker-gen_png"
LORA_BASE_DIR = "/data/<>/repos/sd-scripts/train_scripts"
SDXL_CKPT = "/data/<>/pretrained_models/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "./test_generation_results"  # Will be overridden by command line argument
GENERATION_SCRIPT = "/data/<>/repos/sd-scripts/sdxl_gen_img.py"
QUERY_REWRITE_PATH = "/data/<>/projects/sticker_gen/query_rewrite/short_v2.csv"

# Trigger word mapping
# TRIGGER_WORDS = {
#     'animal': '8ehfk animal',
#     'anime_girl': '839d anime_girl',
#     'anime_guy': '9e0j anime_guy',
#     'bald_cartoon_guy': '9s8cn bald_cartoon_guy',
#     'cartoon_cat_bear': '7e8fn cartoon_cat_bear',
#     'cartoon_cat': 'shd8 cat',
#     'cartoon_rabbit': 's29d rabbit',
#     'chibi': '13lk chibi',
#     'cute_animal': '8s9f cute_animal',
#     'korean_baby': '738dhd korean_baby',
#     'pepe': 'sdf89 pepe',
#     'san_mao': '98dj san_mao',
#     'white_baby': '9s0a white_baby'
# }

TRIGGER_WORDS = {
    'animal': '8ehfk',
    'anime_girl': '839d',
    'anime_guy': '9e0j',
    'bald_cartoon_guy': '9s8cn',
    'cartoon_cat_bear': '7e8fn',
    'cartoon_cat': 'shd8',
    'cartoon_rabbit': 's29d',
    'chibi': '13lk',
    'cute_animal': '8s9f',
    'korean_baby': '738dhd',
    'pepe': 'sdf89',
    'san_mao': '98dj',
    'white_baby': '9s0a'
}

# Add pred_ versions to trigger words
for key in list(TRIGGER_WORDS.keys()):
    TRIGGER_WORDS[f'pred_{key}'] = TRIGGER_WORDS[key]

# ================= UTILITY FUNCTIONS =================
def ensure_png_extension(filename):
    """Ensure filename has .png extension, replacing other extensions."""
    # Remove any existing extension
    name_without_ext = os.path.splitext(filename)[0]
    # Return with .png extension
    return f"{name_without_ext}.png"


def setup_directories(output_dir):
    """Create necessary directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

def find_lora_path(style_name):
    """Find LoRA file for a given style category."""
    # Clean style name
    if "pred_" in style_name:
        style_name = style_name.split("pred_")[-1]
    patterns = [
        f"{style_name}/loras/*-000029.safetensors",
        # f"{style_name}*/**/*.safetensors",
        # f"*{style_name}*/*.safetensors",
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(LORA_BASE_DIR, pattern), recursive=True)
        if matches:
            for match in matches:
                if "lora" in match.lower() or "lora" in os.path.basename(match).lower():
                    return match
    
    # Try specific known patterns as fallback
    known_patterns = {
        'cartoon_cat': 'cartoon_cat_train/lora_output_xdsl/cartoon_cat.safetensors',
        'cartoon_rabbit': 'cartoon_rabbit_train/xsdl_lora_rabbit/cartoon_rabbit.safetensors',
        'chibi': 'chibi_train/lora_output_xdsl/chibi.safetensors',
        'anime_girl': 'anime_girl_train/lora_output_xdsl/anime_girl.safetensors',
        'anime_guy': 'anime_guy_train/lora_output_xdsl/anime_guy.safetensors',
    }
    
    if style_name in known_patterns:
        path = os.path.join(LORA_BASE_DIR, known_patterns[style_name])
        if os.path.exists(path):
            return path
    
    return None

# def distances_to_weights(distances, available_styles, temperature=0.3, top_k=5, min_weight=0.05):
#     """
#     Convert distances to blending weights.
#     """
#     # Filter distances to available styles
#     filtered_distances = {s: d for s, d in distances.items() if s in available_styles}
    
#     if not filtered_distances:
#         return {s: 1.0/len(available_styles) for s in available_styles[:top_k]}
    
#     # Convert distances to similarities
#     similarities = {s: np.exp(-d / temperature) for s, d in filtered_distances.items()}
    
#     # Get top-k styles by similarity
#     sorted_styles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
#     # Apply softmax
#     exp_sims = np.array([sim for _, sim in sorted_styles])
#     weights = np.exp(exp_sims) / np.sum(np.exp(exp_sims))
    
#     # Apply minimum weight threshold
#     weights = np.maximum(weights, min_weight)
#     weights = weights / weights.sum()  # Renormalize
    
#     result = {style: weight for (style, _), weight in zip(sorted_styles, weights)}
#     return result

def distances_to_weights(distances, available_styles, temperature=0.3, top_k=5, min_weight=0.05):
    """
    Convert distances to blending weights.
    MODIFIED: Always uses top-2 styles with 0.9 and 0.1 weights
    """
    # Filter distances to available styles
    filtered_distances = {s: d for s, d in distances.items() if s in available_styles}
    
    if not filtered_distances:
        return {s: 1.0/len(available_styles) for s in available_styles[:top_k]}
    
    # Convert distances to similarities (lower distance = higher similarity)
    similarities = {s: 1.0 / (d + 1e-8) for s, d in filtered_distances.items()}  # Add small epsilon to avoid division by zero
    
    # Get top-2 styles by similarity
    sorted_styles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:2]
    
    result = {}
    
    if len(sorted_styles) >= 2:
        # Assign fixed weights to top 2
        result[sorted_styles[0][0]] = 0.7
        result[sorted_styles[1][0]] = 0.3
        print(f"Top styles: {sorted_styles[0][0]} (0.9), {sorted_styles[1][0]} (0.1)")
    elif len(sorted_styles) == 1:
        # Only one style available
        result[sorted_styles[0][0]] = 1.0
        print(f"Only one style: {sorted_styles[0][0]} (1.0)")
    
    return result

def replace_trigger_word_in_query(original_query, styles_weights):
    """Replace TRIGGER_WORD in query with actual trigger word from dominant style."""
    if not styles_weights:
        return original_query
    
    dominant_style = get_dominant_style(styles_weights)
    if not dominant_style:
        return original_query
    
    # Get actual trigger word for dominant style
    if dominant_style in TRIGGER_WORDS:
        # Extract just the trigger word part (e.g., "8ehfk animal" -> "animal")
        trigger_text = TRIGGER_WORDS[dominant_style]
        # Get the last word as the trigger word
        trigger_word = trigger_text.split()[-1] if ' ' in trigger_text else trigger_text
    else:
        # Fallback: use style name without 'pred_' prefix
        trigger_word = dominant_style.replace('pred_', '')
    
    # Replace TRIGGER_WORD in query (case insensitive)
    query_lower = original_query.lower()
    if 'trigger_word' in query_lower:
        # Find the exact case used in the original
        import re
        pattern = re.compile(r'trigger_word', re.IGNORECASE)
        replaced_query = pattern.sub(trigger_word, original_query)
        return replaced_query
    
    return original_query

def enhance_prompt_with_trigger_words(prompt, styles_weights):
    """Enhance prompt with trigger words for activated LoRAs."""
    # Sort styles by weight
    sorted_styles = sorted(styles_weights.items(), key=lambda x: x[1], reverse=True)
    
    # Add trigger words for top styles (weight > 0.1)
    trigger_texts = []
    for style, weight in sorted_styles:
        if weight > 0.1 and style in TRIGGER_WORDS:
            style = style.split("pred_")[1]
            trigger_text = TRIGGER_WORDS[style]
            trigger_texts.append(trigger_text)
    
    # Combine with original prompt
    if trigger_texts:
        # enhanced_prompt = ", ".join(trigger_texts) + ", " + prompt
        enhanced_prompt = prompt.replace("TRIGGER_WORD", ", ".join(trigger_texts))
    else:
        enhanced_prompt = prompt
    
    # Add quality suffixes
    enhanced_prompt += ", high quality sticker, clean lines, white background, no text"
    
    # Add character consistency guidance if multiple styles
    if len([w for w in styles_weights.values() if w > 0.2]) > 1:
        enhanced_prompt += ", single coherent character, no mixed features"
    
    return enhanced_prompt

def clean_query(query):
    """Clean and format the search query for use as a prompt."""
    # Remove special characters but keep basic punctuation
    query = str(query).strip()
    
    # Replace ?? with appropriate text
    # query = query.replace("??", "question mark")
    
    # Remove excessive whitespace
    query = re.sub(r'\s+', ' ', query)
    
    # Ensure it ends properly
    # if not query.endswith(('.', '!', '?')):
    #     query = query + '.'
    
    return query

def find_sticker_row_index(sticker_id, style_df):
    """Find the row index in style_df for a given sticker_id."""
    # Ensure sticker_id has .png extension
    # if not sticker_id.endswith('.png'):
    #     sticker_id = sticker_id + '.png'
    sticker_id = ensure_png_extension(sticker_id)    
    # Find matching row
    matches = style_df[style_df['sticker_id'] == sticker_id]
    
    if len(matches) > 0:
        # Return the first match's index in the dataframe
        return matches.index[0]
    else:
        # Try without .png if it has it
        if sticker_id.endswith('.png'):
            sticker_id_no_ext = sticker_id[:-4]
            matches = style_df[style_df['sticker_id'].str.startswith(sticker_id_no_ext)]
            if len(matches) > 0:
                return matches.index[0]
        
        # Return None if not found
        return None

def generate_sticker_for_test_sample(output_dir, test_sample, style_df, style_columns, 
                                     lora_mapping, available_styles, temperature=0.3, 
                                     top_k=1, seed=42, sample_index=0):
    """Generate sticker for a single test sample - simplified version."""
    
    user_id = test_sample['user_id']
    search_sticker_id = test_sample['search_sticker_id']
    search_query = test_sample['search_query']
    
    print(f"\nProcessing Sample {sample_index + 1}: {search_sticker_id}")
    
    # Clean the query
    cleaned_query = clean_query(search_query)
    
    # Find the row index for the search_sticker_id in style_df
    row_index = find_sticker_row_index(search_sticker_id, style_df)
    
    if row_index is None:
        print(f"✗ ERROR: Sticker ID {search_sticker_id} not found")
        return False
    
    # Get row data
    row = style_df.iloc[row_index]
    distances = row[style_columns].to_dict()
    
    # Convert distances to weights,
    weights = distances_to_weights(distances, available_styles, temperature, top_k)
    
    # Enhance prompt with trigger words
    enhanced_prompt = enhance_prompt_with_trigger_words(cleaned_query, weights)
    
    # Prepare LoRA paths and weights
    lora_paths = []
    lora_weights = []
    
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_weights)
    for style, weight in sorted_weights:
        if weight > 0.01 and style in lora_mapping:
            lora_paths.append(lora_mapping[style])
            lora_weights.append(str(weight))
    
    if not lora_paths:
        print("Warning: No LoRAs with significant weight")
        return False
    
    # Create output directory
    sticker_output_dir = os.path.join(output_dir, "generated")
    os.makedirs(sticker_output_dir, exist_ok=True)
    
    # Create a temp directory to avoid conflicts
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix=f"temp_{sample_index}_")
    
    try:
        # Build command - output to temp directory
        cmd = [
            sys.executable, GENERATION_SCRIPT,
            "--ckpt", SDXL_CKPT,
            "--prompt", enhanced_prompt,
            "--outdir", temp_dir,
            "--H", "512",
            "--W", "512",
            "--scale", "7.5",
            "--steps", "30",
            "--sampler", "euler_a",
            "--seed", str(seed),
            "--network_merge",
        ]
        
        # Add LoRA networks
        if lora_paths:
            # Create a list with "networks.lora" repeated for each path
            module_args = ["networks.lora"] * len(lora_paths)
            cmd.extend(["--network_module"] + module_args)
            
            # For network_weights, join paths with space (not comma)
            # cmd.extend(["--network_weights", " ".join(lora_paths)])
        if lora_paths:
            cmd.extend(["--network_weights"] + [str(w) for w in lora_paths])
            
        if lora_weights:
            cmd.extend(["--network_mul"] + [str(w) for w in lora_weights])
                
        print(f"Generating: {search_sticker_id}")
        print(f"{cmd}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr[:500]}")
            return False
        
        # Check what files were generated in temp directory
        print(f"Checking files in temp directory: {temp_dir}")
        all_files = os.listdir(temp_dir)
        print(f"Generated files: {all_files}")
        
        # Find image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"✗ No image files generated")
            return False
        
        # Use the first image file found
        source_file = os.path.join(temp_dir, image_files[0])
        
        # Define target filename (always use .png extension)
        target_filename = f"{ensure_png_extension(str(sample_index))}"
        
        target_file = os.path.join(sticker_output_dir, target_filename)
        
        # Copy the file
        import shutil
        shutil.copy2(source_file, target_file)
        print(f"✓ Saved as: {search_sticker_id}, {target_filename}. Enhanced prompt: {enhanced_prompt}, Topk: {top_k}")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp directory
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return False
        
    except Exception as e:
        print(f"Exception: {e}")
        return False

def get_sticker_query_mapping(test_df, query_rewrite_df=None):
    """Create a mapping of sticker_id to search_query from test_df with optional rewrite."""
    print("Creating sticker_id to search_query mapping...")
    
    sticker_queries = {}
    
    for _, row in test_df.iterrows():
        sticker_id = row['search_sticker_id']
        original_query = row['search_query'].split(",")[0]  # Take first query
        
        # Use rewrite if available
        if query_rewrite_df is not None:
            rewrite_match = query_rewrite_df[query_rewrite_df['search_sticker_id'] == sticker_id]
            if not rewrite_match.empty:
                rewritten_query = rewrite_match.iloc[0]['query_rewrite']
                if pd.notna(rewritten_query) and rewritten_query.strip():
                    sticker_queries[sticker_id] = rewritten_query
                    continue
        
        # Fall back to original query
        sticker_queries[sticker_id] = original_query
    
    print(f"Created mapping for {len(sticker_queries)} unique stickers")
    return sticker_queries

    
def main():
    """Main function - process only unique search_sticker_id values."""
    parser = argparse.ArgumentParser(description='Generate stickers for unique test stickers')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated stickers')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for style blending (default: 0.3)')
    parser.add_argument('--top_k', type=int, default=1,
                       help='Number of top styles to blend (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of unique stickers to process (default: all)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip stickers that already exist in output directory')
    parser.add_argument('--start', type=int, default=0,
                       help='Skip stickers that already exist in output directory')

    args = parser.parse_args()
    
    # Set output directory
    OUTPUT_DIR = args.output_dir
    generated_dir = os.path.join(OUTPUT_DIR, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    print("=== Unique Sticker Generator ===")
    print(f"Output directory: {generated_dir}/")
    print(f"Images will be saved as: sticker_id.png")
    print(f"Skip existing: {args.skip_existing}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"Test dataset loaded: {len(test_df)} samples")

    query_rewrite_df = None
    try:
        query_rewrite_df = pd.read_csv(QUERY_REWRITE_PATH)
        print(f"Query rewrite dataset loaded: {len(query_rewrite_df)} entries")
        
        # Check required columns
        if 'search_sticker_id' not in query_rewrite_df.columns or 'query_rewrite' not in query_rewrite_df.columns:
            print("✗ Query rewrite dataset missing required columns. Using original queries.")
            query_rewrite_df = None
        else:
            print(f"  Found {query_rewrite_df['search_sticker_id'].nunique()} unique stickers in rewrite data")
    except Exception as e:
        print(f"✗ Error loading query rewrite dataset: {e}. Using original queries.")
        query_rewrite_df = None
    
    # Get unique stickers with their associated queries (with rewrite if available)
    sticker_query_map = get_sticker_query_mapping(test_df, query_rewrite_df)
    unique_stickers = list(sticker_query_map.keys())
    print(f"Found {len(unique_stickers)} unique sticker IDs")
    
    # Load style classification dataset
    print("\nLoading style classification dataset...")
    style_df = pd.read_csv(STYLE_CSV_PATH)
    
    # Get style categories
    style_columns = [col for col in style_df.columns if col != 'sticker_id']
    
    # Build LoRA mapping
    print("\nBuilding LoRA mapping...")
    lora_mapping = {}
    available_styles = []
    
    for style in style_columns:
        lora_path = find_lora_path(style)
        if lora_path:
            lora_mapping[style] = lora_path
            available_styles.append(style)
            print(f"✓ {style}")
    
    # Apply limit if specified
    if args.limit:
        unique_stickers = unique_stickers[:args.limit]
    
    print(f"\nProcessing {len(unique_stickers)} unique stickers...")
    
    successful_stickers = 0
    skipped_stickers = 0
    failed_stickers = 0
    
    # Get sample queries for each sticker (first occurrence)
    sticker_to_query = {}
    # finished = 0
    for i, row in test_df.iterrows():
        sticker_id = row['search_sticker_id']
        if sticker_id not in sticker_to_query:
            search_query = row['search_query'].split(",")[0]
            sticker_id = row['search_sticker_id']
            sticker_to_query[sticker_id] = search_query

    # for i, sticker_id in enumerate(unique_stickers):
    #     print(f"\n[{i+1}/{len(unique_stickers)}] ", end="")
        
        # Check if file already exists and skip if requested
        # expected_file = os.path.join(generated_dir, sticker_id)
        # if args.skip_existing and os.path.exists(expected_file):
        #     print(f"Skipping (already exists): {sticker_id}")
        #     skipped_stickers += 1
        #     continue
        
    for i, row in test_df.iterrows():
        if args.start > i:
            print(f"Skipping {i}")
            continue
        sticker_id = row['search_sticker_id']
        # Get a sample query for this sticker
        if sticker_id in sticker_to_query:
            search_query = sticker_to_query[sticker_id]
            search_query = sticker_query_map[sticker_id]
        # else:
        #     print(f"✗ No query found for {sticker_id}, using default")
        #     sample_query = "sticker"
        
        # Create a dummy test sample row
        test_sample = pd.Series({
            'user_id': 'dummy_user',
            'search_sticker_id': sticker_id,
            'search_query': search_query
        })

        print(available_styles)
        print(lora_mapping)
        
        success = generate_sticker_for_test_sample(
            OUTPUT_DIR, test_sample, style_df, style_columns, 
            lora_mapping, available_styles, args.temperature, 
            args.top_k, args.seed, i
        )
        
        if success:
            successful_stickers += 1
        else:
            failed_stickers += 1
        print(f"generated: {successful_stickers}, failed {failed_stickers}, total {len(unique_stickers)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    # print(f"Unique stickers processed: {len(unique_stickers)}")
    print(f"Successfully generated: {successful_stickers}")
    print(f"Skipped (already existed): {skipped_stickers}")
    print(f"Failed: {failed_stickers}")
    print(f"\nImages saved in: {generated_dir}")

def test_single_sample():
    """Test with a single sample (for debugging)."""
    # Load datasets
    test_df = pd.read_csv(TEST_CSV_PATH)
    style_df = pd.read_csv(STYLE_CSV_PATH)
    
    # Take first sample
    sample = test_df.iloc[0]
    
    print("Test Sample:")
    print(f"User ID: {sample['user_id']}")
    print(f"Search Sticker ID: {sample['search_sticker_id']}")
    print(f"Search Query: {sample['search_query']}")
    
    # Clean query
    cleaned_query = clean_query(sample['search_query'])
    print(f"Cleaned Query: {cleaned_query}")
    
    # Find sticker in style dataset
    row_index = find_sticker_row_index(sample['search_sticker_id'], style_df)
    
    if row_index is not None:
        print(f"\nFound sticker at row index: {row_index}")
        
        # Get style info
        row = style_df.iloc[row_index]
        style_columns = [col for col in style_df.columns if col != 'sticker_id']
        distances = row[style_columns].to_dict()
        
        print(f"\nDistances to style centroids:")
        for style, distance in sorted(distances.items(), key=lambda x: x[1]):
            print(f"  {style}: {distance}")
        
        # Build LoRA mapping
        available_styles = []
        for style in style_columns:
            lora_path = find_lora_path(style)
            if lora_path:
                available_styles.append(style)
        
        # Test weight conversion
        weights = distances_to_weights(distances, available_styles, temperature=0.3, top_k=1)
        print(f"\nCalculated weights (temp=0.3, top_k=1):")
        for style, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {style}: {weight:.3f}")
        
        # Test prompt enhancement
        enhanced_prompt = enhance_prompt_with_trigger_words(cleaned_query, weights)
        print(f"\nEnhanced prompt: {enhanced_prompt}")
        
    else:
        print(f"\nERROR: Sticker {sample['search_sticker_id']} not found in style dataset")

if __name__ == "__main__":
    # For testing a single sample
    # test_single_sample()
    
    # Run main processing
    main()