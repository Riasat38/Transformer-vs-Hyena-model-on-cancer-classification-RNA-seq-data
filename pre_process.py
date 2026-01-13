import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_DIR = 'processed_data'
OUTPUT_DIR = 'final_processed_data'

ZERO_THRESHOLD = 0.5  # Remove genes with >50% zeros
VARIANCE_PERCENTILE = 95 

X_train = np.load(f'{INPUT_DIR}/X_train.npy')
X_val = np.load(f'{INPUT_DIR}/X_val.npy')
X_test = np.load(f'{INPUT_DIR}/X_test.npy')

y_train = np.load(f'{INPUT_DIR}/y_train.npy')
y_val = np.load(f'{INPUT_DIR}/y_val.npy')
y_test = np.load(f'{INPUT_DIR}/y_test.npy')

gene_ids = np.load(f'{INPUT_DIR}/gene_ids.npy', allow_pickle=True)

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Genes:   {len(gene_ids)}")

print("FILTERING LOW-EXPRESSION GENES")
print("="*70)

# Count zeros per gene (only on training data)
zero_ratio = (X_train == 0).sum(axis=0) / X_train.shape[0]
keep_mask = zero_ratio < ZERO_THRESHOLD

n_removed = (~keep_mask).sum()
n_kept = keep_mask.sum()

print(f"Genes with ≥{ZERO_THRESHOLD*100}% zeros: {n_removed}")
print(f"Genes kept: {n_kept}")

# Apply filter to all datasets
X_train = X_train[:, keep_mask]
X_val = X_val[:, keep_mask]
X_test = X_test[:, keep_mask]
gene_ids = gene_ids[keep_mask]

print(f"New shape: {X_train.shape}")

print("SELECTING HIGH-VARIANCE GENES")
print("="*70)

# Calculate variance per gene (only on training data)
variances = np.var(X_train, axis=0)
threshold = np.percentile(variances, VARIANCE_PERCENTILE)
variance_mask = variances >= threshold
variance_indices = np.where(variance_mask)[0]

n_selected = len(variance_indices)

print(f"Variance threshold (p{VARIANCE_PERCENTILE}): {threshold:.6f}")
print(f"Genes selected: {n_selected}")

# Apply selection to all datasets
X_train = X_train[:, variance_indices]
X_val = X_val[:, variance_indices]
X_test = X_test[:, variance_indices]
gene_ids = gene_ids[variance_indices]
gene_variances = variances[variance_indices]

print(f"Final shape: {X_train.shape}")
print(f"Variance range: [{gene_variances.min():.6f}, {gene_variances.max():.6f}]")

print("PREPARING FOR TRANSFORMER")
print("="*70)

# Reshape: (samples, genes) → (samples, genes, 1)
X_train_transformer = X_train[..., np.newaxis]
X_val_transformer = X_val[..., np.newaxis]
X_test_transformer = X_test[..., np.newaxis]

print(f"ML format:          {X_train.shape}")
print(f"Transformer format: {X_train_transformer.shape}")

print("SAVING PROCESSED DATA")
print("="*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save ML-ready data (2D)
np.save(f'{OUTPUT_DIR}/X_train_ml.npy', X_train.astype(np.float32))
np.save(f'{OUTPUT_DIR}/X_val_ml.npy', X_val.astype(np.float32))
np.save(f'{OUTPUT_DIR}/X_test_ml.npy', X_test.astype(np.float32))

# Save Transformer-ready data (3D)
np.save(f'{OUTPUT_DIR}/X_train_transformer.npy', X_train_transformer.astype(np.float32))
np.save(f'{OUTPUT_DIR}/X_val_transformer.npy', X_val_transformer.astype(np.float32))
np.save(f'{OUTPUT_DIR}/X_test_transformer.npy', X_test_transformer.astype(np.float32))

# Save labels
np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
np.save(f'{OUTPUT_DIR}/y_val.npy', y_val)
np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)

# Save gene information
np.save(f'{OUTPUT_DIR}/selected_gene_ids.npy', gene_ids)
np.save(f'{OUTPUT_DIR}/gene_variances.npy', gene_variances)

# Save preprocessing info
np.save(f'{OUTPUT_DIR}/zero_filter_mask.npy', keep_mask)
np.save(f'{OUTPUT_DIR}/variance_indices.npy', variance_indices)

# Save metadata
metadata = {
    'zero_threshold': ZERO_THRESHOLD,
    'variance_percentile': VARIANCE_PERCENTILE,
    'n_genes_selected': len(gene_ids),
    'train_shape_ml': list(X_train.shape),
    'train_shape_transformer': list(X_train_transformer.shape),
    'n_classes': len(np.unique(y_train)),
    'train_samples': int(len(y_train)),
    'val_samples': int(len(y_val)),
    'test_samples': int(len(y_test))
}

with open(f'{OUTPUT_DIR}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved to '{OUTPUT_DIR}/' directory")

print("\n" + "="*70)
print("COMPLETE!")