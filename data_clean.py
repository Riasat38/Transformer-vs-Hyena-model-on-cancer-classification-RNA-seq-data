import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json

df = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', sep='\t', index_col=0)
labels = pd.read_csv('annotations.csv', index_col=0)

print("Data loaded.")

df_id = ['-'.join(col.split('-')[:3]) for col in df.columns]
meta_dict = dict(zip(labels['id'], labels['CANCER_TYPE']))

matching_indices = []
matching_cancer_types = []

for i, sample_id in enumerate(df_id):
    if sample_id in meta_dict:
        matching_indices.append(i)
        matching_cancer_types.append(meta_dict[sample_id])

print(f"\n✓ Matching samples: {len(matching_indices)}/{df.shape[1]}")

expr_df_filtered = df.iloc[:, matching_indices]

print(f"✓ Filtered shape: {expr_df_filtered.shape}")

df_clean = expr_df_filtered.fillna(df.median())

labels_array = np.array(matching_cancer_types)
unique_labels, label_counts = np.unique(labels_array, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print(f"  {label}: {count}")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_array)
print(f"\n✓ Encoded labels: {len(unique_labels)} cancer types")
print(f"  Label mapping: {dict(enumerate(label_encoder.classes_))}")

X = df_clean.T.values  # Transpose
y = labels_encoded

print(f"\n✓ Data matrix shape: {X.shape}")
print(f"  (samples × genes)")
print(f"✓ Labels shape: {y.shape}")

gene_ids = df_clean.index.values
sample_ids = np.array([df.columns[i] for i in matching_indices])

print(f"\n✓ Gene IDs: {len(gene_ids)}")
print(f"✓ Sample IDs: {len(sample_ids)}")

# 5. Convert to float32 to save memory
X = X.astype(np.float32)

print(f"\n✓ Memory usage: {X.nbytes / 1e9:.3f} GB")


X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
    X, y, np.arange(len(y)), 
    test_size=0.20, 
    random_state=42,
    stratify=y  # Maintain class distribution
)

X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_train_val, y_train_val, np.arange(len(y_train_val)),
    test_size=.20,  
    random_state=42,
    stratify=y_train_val
)

print(f"Train set: {X_train.shape} - {len(y_train)} samples")
print(f"Val set:   {X_val.shape} - {len(y_val)} samples")
print(f"Test set:  {X_test.shape} - {len(y_test)} samples")

print("\nClass distribution:")
for split_name, split_labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    unique, counts = np.unique(split_labels, return_counts=True)
    print(f"\n{split_name}:")
    for u, c in zip(unique, counts):
        cancer_type = label_encoder.classes_[u]
        print(f"  {cancer_type}: {c}")



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Normalized with StandardScaler")
print(f"  Train - mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
print(f"  Val   - mean: {X_val_scaled.mean():.3f}, std: {X_val_scaled.std():.3f}")
print(f"  Test  - mean: {X_test_scaled.mean():.3f}, std: {X_test_scaled.std():.3f}")

output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

np.save(f'{output_dir}/X_train.npy', X_train_scaled.astype(np.float32))
np.save(f'{output_dir}/X_val.npy', X_val_scaled.astype(np.float32))
np.save(f'{output_dir}/X_test.npy', X_test_scaled.astype(np.float32))

np.save(f'{output_dir}/y_train.npy', y_train)
np.save(f'{output_dir}/y_val.npy', y_val)
np.save(f'{output_dir}/y_test.npy', y_test)

np.save(f'{output_dir}/gene_ids.npy', gene_ids)
np.save(f'{output_dir}/train_sample_ids.npy', sample_ids[idx_train])
np.save(f'{output_dir}/val_sample_ids.npy', sample_ids[idx_val])
np.save(f'{output_dir}/test_sample_ids.npy', sample_ids[idx_test])
print(f"\n✓ Data saved to '{output_dir}' directory.")



with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save scaler
with open(f'{output_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metadata

metadata = {
    'n_samples': len(y),
    'n_genes': len(gene_ids),
    'n_classes': len(unique_labels),
    'class_names': label_encoder.classes_.tolist(),
    'train_size': len(y_train),
    'val_size': len(y_val),
    'test_size': len(y_test),
    'normalization': 'StandardScaler',
    'dtype': 'float32'
}

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)