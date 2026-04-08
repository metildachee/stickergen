"""
Baseline 4b: Query + History Style Vectors - FIXED for long histories
Uses pre-computed style vectors from style_classification.csv instead of CLIP embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import time
warnings.filterwarnings('ignore')

# ======================
# Configuration
# ======================
class Config:
    # Paths
    STYLE_LABEL_PATH = "/data/metchee/projects/sticker_gen/style_labeling/sticker_style_classification.csv"
    TRAIN_CSV_PATH = "/data/metchee/projects/sticker_gen/dataset/new_all/release/train.csv"
    TEST_CSV_PATH = "/data/metchee/projects/sticker_gen/dataset/new_all/release/val.csv"
    
    # Model parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    TEXT_EMBEDDING_DIM = 384  # Sentence-BERT dimension
    STYLE_DIM = 13  # Number of style classes
    MAX_HISTORY_LEN = 20  # Increased from 10 to capture more history
    
    # Aggregation strategy for long histories
    HISTORY_AGGREGATION = 'mean'  # 'mean', 'attention', or 'last_k'
    LAST_K = 10  # If using 'last_k', how many recent items to use
    
    # Style column names (13 styles)
    STYLE_COLUMNS = [
        'animal', 'anime_girl', 'anime_guy', 'bald_cartoon_guy',
        'cartoon_cat', 'cartoon_cat_bear', 'cartoon_rabbit',
        'chibi', 'cute_animal', 'korean_baby', 'pepe', 'san_mao', 'white_baby'
    ]
    
    # Numerical stability
    EPSILON = 1e-8

# ======================
# Utility Functions
# ======================
def ensure_png_extension(sticker_id: str) -> str:
    """Ensure sticker_id has .png extension."""
    sticker_id = str(sticker_id).strip()
    if not sticker_id.endswith('.png'):
        return f"{sticker_id}.png"
    return sticker_id

def clean_sticker_id(sticker_id: str) -> str:
    """Clean sticker_id by removing any extensions and returning base ID."""
    sticker_id = str(sticker_id).strip()
    # Remove any extension
    for ext in ['.png', '.jpg', '.jpeg', '.webm', '.gif']:
        if sticker_id.endswith(ext):
            sticker_id = sticker_id[:-len(ext)]
            break
    return sticker_id

# ======================
# Style Label Loader
# ======================
class StyleLabelLoader:
    """Loads and caches style labels for stickers."""
    
    def __init__(self, label_path: str):
        self.label_path = label_path
        print(f"Loading style labels from: {label_path}")
        self.style_df = pd.read_csv(label_path)
        
        # Ensure sticker_id is string and has .png extension for consistency
        self.style_df['sticker_id'] = self.style_df['sticker_id'].astype(str).apply(ensure_png_extension)
        self.style_df.set_index('sticker_id', inplace=True)
        self.label_cache = {}
        
        # Create a mapping from base ID (without extension) to full ID
        self.base_id_to_full = {}
        for full_id in self.style_df.index:
            base_id = clean_sticker_id(full_id)
            self.base_id_to_full[base_id] = full_id
        
        # Precompute statistics for normalization
        all_vectors = self.style_df[Config.STYLE_COLUMNS].values.astype(np.float32)
        self.mean = np.mean(all_vectors, axis=0)
        self.std = np.std(all_vectors, axis=0) + Config.EPSILON
        
        print(f"Loaded {len(self.style_df)} sticker style vectors")
        print(f"Style value range: [{np.min(all_vectors):.4f}, {np.max(all_vectors):.4f}]")
        
    def get_style_vector(self, sticker_id: str, normalize: bool = False) -> np.ndarray:
        """Get style vector for a sticker_id."""
        sticker_id = str(sticker_id).strip()
        
        # Check cache first
        if sticker_id in self.label_cache:
            return self.label_cache[sticker_id]
        
        # Try with .png extension
        png_id = ensure_png_extension(sticker_id)
        if png_id in self.style_df.index:
            vector = self.style_df.loc[png_id, Config.STYLE_COLUMNS].values.astype(np.float32)
            
            # Optional normalization
            if normalize:
                vector = (vector - self.mean) / self.std
            
            self.label_cache[sticker_id] = vector
            return vector
        
        # Try base ID (without extension) lookup
        base_id = clean_sticker_id(sticker_id)
        if base_id in self.base_id_to_full:
            full_id = self.base_id_to_full[base_id]
            vector = self.style_df.loc[full_id, Config.STYLE_COLUMNS].values.astype(np.float32)
            
            if normalize:
                vector = (vector - self.mean) / self.std
            
            self.label_cache[sticker_id] = vector
            return vector
        
        # Return zeros for missing stickers
        return np.zeros(len(Config.STYLE_COLUMNS), dtype=np.float32)
    
    def get_all_style_vectors(self, sticker_ids: List[str], normalize: bool = False) -> np.ndarray:
        """Get style vectors for multiple stickers."""
        vectors = [self.get_style_vector(sid, normalize) for sid in sticker_ids]
        return np.array(vectors)

# ======================
# Text Encoder (Sentence-BERT)
# ======================
class TextEncoder:
    """Pretrained text encoder using Sentence-BERT."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded Sentence-BERT model: {model_name}")
        except ImportError:
            print("⚠ sentence-transformers not installed. Using dummy embeddings.")
            print("  Install with: pip install sentence-transformers")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            return np.random.randn(len(texts), Config.TEXT_EMBEDDING_DIM).astype(np.float32)
        
        # Filter out empty strings
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            return np.zeros((len(texts), Config.TEXT_EMBEDDING_DIM), dtype=np.float32)
        
        with torch.no_grad():
            embeddings = self.model.encode(valid_texts, convert_to_tensor=True, device=self.device)
            embeddings = embeddings.cpu().numpy()
        
        # Reconstruct full array with zeros for invalid texts
        result = np.zeros((len(texts), Config.TEXT_EMBEDDING_DIM), dtype=np.float32)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb
        
        return result

# ======================
# Dataset Class - Fixed for long histories
# ======================
class StickerDatasetWithHistoryStyles(Dataset):
    """Dataset that provides history style vectors directly - fixed for long histories."""
    
    def __init__(self, csv_path: str, 
                 style_loader: StyleLabelLoader,
                 text_encoder: Optional[TextEncoder] = None,
                 max_history_len: int = 20):
        
        self.df = pd.read_csv(csv_path)
        self.style_loader = style_loader
        self.text_encoder = text_encoder
        self.max_history_len = max_history_len
        
        print(f"Loading dataset from: {csv_path}")
        print(f"Total samples: {len(self.df)}")
        
        # Precompute text embeddings
        self.text_embeddings = {}
        if self.text_encoder is not None:
            print("Precomputing text embeddings...")
            all_queries = self.df['search_query'].fillna("").apply(lambda x: x.split(",")[0].strip()).tolist()
            
            # Process in batches
            batch_size = 128
            all_embeddings = []
            for i in range(0, len(all_queries), batch_size):
                batch_queries = all_queries[i:i+batch_size]
                batch_embeddings = self.text_encoder.encode(batch_queries)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.concatenate(all_embeddings, axis=0)
            
            for idx, (_, row) in enumerate(self.df.iterrows()):
                query = row['search_query'].split(",")[0].strip() if pd.notna(row['search_query']) else ""
                key = f"{query}_{row['search_sticker_id']}"
                self.text_embeddings[key] = embeddings[idx]
        
        # Parse histories - handle long histories properly
        print("Parsing histories...")
        self.histories = []
        empty_history_count = 0
        total_history_items = 0
        history_lengths = []
        
        for history_str in self.df['history']:
            if pd.isna(history_str) or history_str == '':
                self.histories.append([])
                empty_history_count += 1
                history_lengths.append(0)
                continue
            
            # Split by comma
            items = history_str.split(',')
            history = []
            for item in items:
                if '|' in item:
                    parts = item.split('|')
                    if len(parts) >= 2:
                        sticker_id = parts[0].strip()
                        # Clean up sticker_id
                        clean_id = clean_sticker_id(sticker_id)
                        history.append(clean_id)
            
            history_lengths.append(len(history))
            total_history_items += len(history)
            
            # Strategy 1: Take most recent items
            if Config.HISTORY_AGGREGATION == 'last_k':
                history = history[-Config.LAST_K:]
            else:  # 'mean' or 'attention' - keep all but we'll aggregate later
                # Still truncate to max_history_len to avoid OOM
                if len(history) > self.max_history_len:
                    # Keep most recent
                    history = history[-self.max_history_len:]
            
            self.histories.append(history)
        
        # Print statistics
        history_lengths = np.array(history_lengths)
        print(f"  - Samples with history: {len(self.df) - empty_history_count}")
        print(f"  - Samples without history: {empty_history_count}")
        print(f"  - Average history length: {total_history_items / max(len(self.df) - empty_history_count, 1):.2f}")
        print(f"  - Max history length: {np.max(history_lengths)}")
        print(f"  - Min history length (non-zero): {np.min(history_lengths[history_lengths > 0]) if np.any(history_lengths > 0) else 0}")
        print(f"  - Using aggregation: {Config.HISTORY_AGGREGATION}")
        if Config.HISTORY_AGGREGATION == 'last_k':
            print(f"  - Keeping last {Config.LAST_K} items")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        history = self.histories[idx]
        target_sticker = clean_sticker_id(row['search_sticker_id'])
        
        # Target style vector
        target = self.style_loader.get_style_vector(target_sticker)
        
        # Text embedding for query
        text_embedding = None
        if self.text_encoder is not None:
            query = row['search_query'].split(",")[0].strip() if pd.notna(row['search_query']) else ""
            key = f"{query}_{row['search_sticker_id']}"
            text_embedding = self.text_embeddings.get(key, np.zeros(Config.TEXT_EMBEDDING_DIM, dtype=np.float32))
        
        # Get style vectors for history stickers
        history_style_vectors = []
        for sticker_id in history:
            style_vec = self.style_loader.get_style_vector(sticker_id)
            history_style_vectors.append(style_vec)
        
        # For aggregation methods that need all items, we'll keep them
        # Pad to fixed length for batching
        if len(history_style_vectors) < self.max_history_len:
            padding = [np.zeros(Config.STYLE_DIM, dtype=np.float32)] * (self.max_history_len - len(history_style_vectors))
            history_style_vectors.extend(padding)
        else:
            history_style_vectors = history_style_vectors[:self.max_history_len]
        
        history_style_vectors = np.array(history_style_vectors)
        
        # Create a mask indicating which positions are real (non-zero)
        valid_mask = np.array([1 if i < len(history) else 0 for i in range(self.max_history_len)], dtype=np.float32)
        
        return {
            'text_embedding': text_embedding,
            'history_style_vectors': history_style_vectors,
            'history_mask': valid_mask,
            'target': target,
            'history_length': len(history),
            'query': row['search_query'].split(",")[0].strip() if pd.notna(row['search_query']) else "",
            'sticker_id': target_sticker,
            'user_id': row['user_id'],
            'search_timestamp': row['search_timestamp'] if 'search_timestamp' in row else 0
        }

# ======================
# Collate Function
# ======================
def collate_history_styles(batch):
    """Collate for history style vector model."""
    text_embeddings = []
    history_style_vectors = []
    history_masks = []
    targets = []
    history_lengths = []
    queries = []
    sticker_ids = []
    user_ids = []
    timestamps = []
    
    for item in batch:
        if item['text_embedding'] is not None:
            text_embeddings.append(item['text_embedding'])
        else:
            text_embeddings.append(np.zeros(Config.TEXT_EMBEDDING_DIM, dtype=np.float32))
        
        history_style_vectors.append(item['history_style_vectors'])
        history_masks.append(item['history_mask'])
        targets.append(item['target'])
        history_lengths.append(item['history_length'])
        queries.append(item['query'])
        sticker_ids.append(item['sticker_id'])
        user_ids.append(item['user_id'])
        timestamps.append(item['search_timestamp'])
    
    return {
        'text_embeddings': torch.FloatTensor(np.array(text_embeddings)),
        'history_style_vectors': torch.FloatTensor(np.array(history_style_vectors)),
        'history_masks': torch.FloatTensor(np.array(history_masks)),
        'targets': torch.FloatTensor(np.array(targets)),
        'history_lengths': torch.LongTensor(history_lengths),
        'queries': queries,
        'sticker_ids': sticker_ids,
        'user_ids': user_ids,
        'timestamps': timestamps
    }

# ======================
# Model Definition - Fixed for long histories
# ======================
class QueryHistoryStyleMLP(nn.Module):
    """Combined model using query embeddings + history style vectors - fixed for long histories."""
    
    def __init__(self, 
                 text_dim: int = 384,
                 style_dim: int = 13,
                 history_size: int = 20,
                 fusion_method: str = 'concat',
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 output_dim: int = 13):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.history_size = history_size
        self.style_dim = style_dim
        self.aggregation = Config.HISTORY_AGGREGATION
        
        # Project text and style history
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.style_history_proj = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # For handling variable-length history
        if self.aggregation == 'attention':
            self.history_attention = nn.MultiheadAttention(
                embed_dim=256, num_heads=4, batch_first=True, dropout=dropout_rate
            )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = 512
            self.fusion_proj = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        elif fusion_method == 'gated':
            self.gate_text = nn.Linear(256, 256)
            self.gate_history = nn.Linear(256, 256)
            fusion_dim = 256
            self.fusion_proj = nn.Identity()
        else:
            fusion_dim = 256
            self.fusion_proj = nn.Identity()
        
        # MLP layers
        layers = []
        prev_dim = 256
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"Model initialized with fusion: {fusion_method}, aggregation: {self.aggregation}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def aggregate_history(self, history_proj, mask):
        """Aggregate history vectors based on selected strategy."""
        batch_size = history_proj.shape[0]
        
        if self.aggregation == 'mean':
            # Simple mean pooling
            masked_sum = (history_proj * mask.unsqueeze(-1)).sum(dim=1)
            history_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return masked_sum / history_lengths
        
        elif self.aggregation == 'last_k':
            # Just take the last valid items (already truncated in dataset)
            # But we still need to handle padding
            last_valid = []
            for i in range(batch_size):
                # Find last non-padded position
                valid_indices = torch.where(mask[i] > 0)[0]
                if len(valid_indices) > 0:
                    last_idx = valid_indices[-1]
                    last_valid.append(history_proj[i, last_idx])
                else:
                    last_valid.append(torch.zeros(256, device=history_proj.device))
            return torch.stack(last_valid)
        
        elif self.aggregation == 'attention':
            # Attention over history
            attended_history, _ = self.history_attention(
                history_proj, history_proj, history_proj,
                key_padding_mask=(mask == 0)
            )
            masked_sum = (attended_history * mask.unsqueeze(-1)).sum(dim=1)
            history_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return masked_sum / history_lengths
        
        else:
            # Default to mean
            masked_sum = (history_proj * mask.unsqueeze(-1)).sum(dim=1)
            history_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return masked_sum / history_lengths
    
    def forward(self, text_embeddings, history_style_vectors, history_masks):
        """
        Args:
            text_embeddings: [batch_size, text_dim]
            history_style_vectors: [batch_size, history_size, style_dim]
            history_masks: [batch_size, history_size] - 1 for real, 0 for padding
        """
        batch_size = text_embeddings.shape[0]
        
        # Project text
        text_proj = self.text_proj(text_embeddings)
        
        # Project history styles
        history_proj = self.style_history_proj(history_style_vectors)
        
        # Aggregate history
        history_pooled = self.aggregate_history(history_proj, history_masks)
        
        # Fuse text and history
        if self.fusion_method == 'concat':
            fused = torch.cat([text_proj, history_pooled], dim=1)
            fused = self.fusion_proj(fused)
        elif self.fusion_method == 'gated':
            gate = torch.sigmoid(self.gate_text(text_proj) + self.gate_history(history_pooled))
            fused = gate * text_proj + (1 - gate) * history_pooled
        else:
            fused = text_proj + history_pooled
        
        # Final prediction
        return self.mlp(fused)

# ======================
# Trainer Class
# ======================
class CombinedStyleTrainer:
    """Trainer for combined query + history style model."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print(f"Using device: {self.device}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move data to device
                text_emb = batch['text_embeddings'].to(self.device)
                history_styles = batch['history_style_vectors'].to(self.device)
                history_masks = batch['history_masks'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Check for extreme values
                if torch.isnan(text_emb).any() or torch.isnan(history_styles).any():
                    print(f"  Warning: NaN in inputs, skipping batch {batch_idx}")
                    continue
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(text_emb, history_styles, history_masks)
                
                # Check outputs
                if torch.isnan(outputs).any():
                    print(f"  Warning: NaN in outputs, skipping batch {batch_idx}")
                    continue
                
                loss = self.criterion(outputs, targets)
                
                # Check loss
                if torch.isnan(loss) or loss.item() > 1e4:
                    print(f"  Warning: Invalid loss {loss.item():.2f}, skipping batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 50 == 0:
                    print(f"    Batch {batch_idx}: Loss = {loss.item():.6f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(batch_count, 1)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_emb = batch['text_embeddings'].to(self.device)
                history_styles = batch['history_style_vectors'].to(self.device)
                history_masks = batch['history_masks'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(text_emb, history_styles, history_masks)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        mse = total_loss / len(val_loader)
        outputs_tensor = torch.cat(all_outputs, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        
        rmse_per_style = torch.sqrt(torch.mean((outputs_tensor - targets_tensor) ** 2, dim=0))
        overall_rmse = torch.sqrt(torch.mean((outputs_tensor - targets_tensor) ** 2))
        
        return mse, rmse_per_style.numpy(), overall_rmse.item()
    
    def predict(self, test_loader, output_file=None):
        """Generate predictions for test set."""
        self.model.eval()
        all_predictions = []
        all_queries = []
        all_sticker_ids = []
        all_user_ids = []
        all_timestamps = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                text_emb = batch['text_embeddings'].to(self.device)
                history_styles = batch['history_style_vectors'].to(self.device)
                history_masks = batch['history_masks'].to(self.device)
                
                outputs = self.model(text_emb, history_styles, history_masks)
                
                all_predictions.append(outputs.cpu().numpy())
                all_queries.extend(batch['queries'])
                all_sticker_ids.extend(batch['sticker_ids'])
                all_user_ids.extend(batch['user_ids'])
                all_timestamps.extend(batch['timestamps'])
                all_targets.append(batch['targets'].numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        pred_df = pd.DataFrame({
            'user_id': all_user_ids,
            'query': all_queries,
            'sticker_id': all_sticker_ids,
            'timestamp': all_timestamps
        })
        
        for i, style in enumerate(Config.STYLE_COLUMNS):
            pred_df[f'pred_{style}'] = predictions[:, i]
            pred_df[f'true_{style}'] = targets[:, i]
        
        if output_file:
            pred_df.to_csv(output_file, index=False)
            print(f"✓ Predictions saved to: {output_file}")
        
        return pred_df

# ======================
# Main Execution
# ======================
def main():
    print("="*70)
    print("BASELINE 4b: Query + History Style Vectors - FIXED for long histories")
    print("="*70)
    
    # Set aggregation strategy
    Config.HISTORY_AGGREGATION = 'mean'  # Try 'mean', 'last_k', or 'attention'
    Config.MAX_HISTORY_LEN = 20
    
    print(f"\nConfiguration:")
    print(f"  - History aggregation: {Config.HISTORY_AGGREGATION}")
    print(f"  - Max history length: {Config.MAX_HISTORY_LEN}")
    
    # Initialize components
    print("\n1. Initializing components...")
    style_loader = StyleLabelLoader(Config.STYLE_LABEL_PATH)
    text_encoder = TextEncoder()
    
    # Create datasets
    print("\n2. Loading datasets...")
    train_dataset = StickerDatasetWithHistoryStyles(
        Config.TRAIN_CSV_PATH, 
        style_loader, 
        text_encoder, 
        max_history_len=Config.MAX_HISTORY_LEN
    )
    test_dataset = StickerDatasetWithHistoryStyles(
        Config.TEST_CSV_PATH, 
        style_loader, 
        text_encoder, 
        max_history_len=Config.MAX_HISTORY_LEN
    )
    
    print(f"\n   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_history_styles,
        num_workers=0,  # Set to 0 to avoid forking issues
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_history_styles,
        num_workers=0
    )
    
    # Test different fusion methods
    fusion_methods = ['concat', 'attention', 'gated']
    results = {}
    
    for fusion_method in fusion_methods:
        print(f"\n" + "="*50)
        print(f"Training with fusion method: {fusion_method}")
        print("="*50)
        
        # Create model
        model = QueryHistoryStyleMLP(
            text_dim=Config.TEXT_EMBEDDING_DIM,
            style_dim=Config.STYLE_DIM,
            history_size=Config.MAX_HISTORY_LEN,
            fusion_method=fusion_method,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.3,
            output_dim=Config.STYLE_DIM
        )
        
        # Print model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Create trainer
        trainer = CombinedStyleTrainer(model)
        
        # Training loop
        print(f"\nTraining for {Config.EPOCHS} epochs...")
        best_val_mse = float('inf')
        best_epoch = 0
        
        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Evaluate
            val_mse, rmse_per_style, overall_rmse = trainer.evaluate(test_loader)
            
            # Learning rate scheduling
            trainer.scheduler.step(val_mse)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val MSE: {val_mse:.6f}")
                print(f"  Val RMSE: {overall_rmse:.6f}")
                
                # Print top/bottom styles
                style_rmse = [(Config.STYLE_COLUMNS[i], rmse_per_style[i]) 
                             for i in range(len(Config.STYLE_COLUMNS))]
                style_rmse.sort(key=lambda x: x[1])
                print("  Best 3 styles:")
                for style, rmse in style_rmse[:3]:
                    print(f"    - {style}: {rmse:.4f}")
            
            # Save best model
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f"best_model_{fusion_method}_fixed.pt")
                print(f"  ✓ New best model! Val MSE: {val_mse:.6f}")
        
        print(f"\n✓ Best {fusion_method} model at epoch {best_epoch} with Val MSE: {best_val_mse:.6f}")
        
        # Final evaluation
        val_mse, rmse_per_style, overall_rmse = trainer.evaluate(test_loader)
        results[fusion_method] = {
            'mse': val_mse,
            'rmse': overall_rmse,
            'rmse_per_style': rmse_per_style,
            'best_mse': best_val_mse
        }
        
        # Generate predictions
        output_file = f"baseline4b_predictions_{fusion_method}_val_fixed.csv"
        trainer.predict(test_loader, output_file)
    
    # Summary
    print("\n" + "="*70)
    print("FUSION METHOD COMPARISON")
    print("="*70)
    
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Best MSE: {metrics['best_mse']:.6f}")
        print(f"  Final MSE: {metrics['mse']:.6f}")
        print(f"  Overall RMSE: {metrics['rmse']:.6f}")

if __name__ == "__main__":
    main()