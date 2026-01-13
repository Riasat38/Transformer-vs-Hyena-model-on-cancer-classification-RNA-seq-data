import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns



from transformer import RNASeqTransformer
from utils import EarlyStopping, calculate_metrics

DATA_DIR = 'final_processed_data'
OUTPUT_DIR = 'model_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training parameters
NUM_EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3
PATIENCE = 8

# Model parameters
NHEAD = 4
NUM_LAYERS = 3
D_MODEL = 192
DROPOUT = 0.4
DIM_FEEDFORWARD = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train = np.load(f'{DATA_DIR}/X_train_transformer.npy')
X_val = np.load(f'{DATA_DIR}/X_val_transformer.npy')
X_test = np.load(f'{DATA_DIR}/X_test_transformer.npy')

y_train = np.load(f'{DATA_DIR}/y_train.npy')
y_val = np.load(f'{DATA_DIR}/y_val.npy')
y_test = np.load(f'{DATA_DIR}/y_test.npy')

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")

# Auto-detect number of classes
NUM_CLASSES = len(np.unique(y_train))
INPUT_DIM = X_train.shape[1]  # Number of genes

print(f"\nDataset info:")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Training samples: {len(y_train)}")
print(f"  Validation samples: {len(y_val)}")
print(f"  Test samples: {len(y_test)}")

X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)
X_test_tensor = torch.FloatTensor(X_test)

y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")


model = RNASeqTransformer(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)

# History tracking
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    
}
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS): 
    # ===== TRAINING PHASE =====
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    train_loss = total_loss / len(train_loader)
    train_metrics = calculate_metrics(all_labels, all_preds)
    
    # ===== VALIDATION PHASE =====
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    val_loss = total_loss / len(val_loader)
    val_metrics = calculate_metrics(all_labels, all_preds)
    
    # ===== RECORD HISTORY =====
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_metrics['accuracy'])
    history['val_acc'].append(val_metrics['accuracy'])
    
    # ===== LEARNING RATE SCHEDULING =====
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # ===== SAVE BEST MODEL =====
    if val_metrics['accuracy'] > best_val_acc:
        best_val_acc = val_metrics['accuracy']
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/transformer_best_model.pth')
        print(f"✓ Saved new best model (Val Acc: {best_val_acc:.4f})")
    
    # ===== PRINT PROGRESS =====
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print(f"  Train Acc:  {train_metrics['accuracy']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.4f} | LR: {current_lr:.6f}")
    
    # ===== EARLY STOPPING CHECK =====
    if early_stopping(val_loss, epoch):
        print(f"\n✓ Early stopping at epoch {epoch+1}")
        print(f"  Best validation loss was {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch+1}")
        break

    np.save(f'{OUTPUT_DIR}/training_history.npy', history)

with open(f'{OUTPUT_DIR}/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ Saved training history to {OUTPUT_DIR}/")



sns.set_style("whitegrid")

print("\n" + "="*60)
print("GENERATING TRAINING VISUALIZATIONS")
print("="*60)

# Create figure with train/val comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training & Validation Analysis', fontsize=16, fontweight='bold')

epochs = range(1, len(history['train_loss']) + 1)

# ===== Loss Curves =====
ax1 = axes[0, 0]
ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Find best epoch
best_epoch = np.argmin(history['val_loss']) + 1
ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
ax1.legend(fontsize=10)

# =====  Accuracy Curves =====
ax2 = axes[0, 1]
ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# ===== Final Metrics Comparison =====
ax3 = axes[1, 0]
final_epoch = len(history['train_loss'])
metrics = ['Accuracy', 'Loss']
train_vals = [history['train_acc'][-1], history['train_loss'][-1]]
val_vals = [history['val_acc'][-1], history['val_loss'][-1]]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, train_vals, width, label='Train', color='skyblue', edgecolor='black')
bars2 = ax3.bar(x + width/2, val_vals, width, label='Validation', color='lightcoral', edgecolor='black')

ax3.set_ylabel('Value', fontsize=12)
ax3.set_title(f'Final Metrics Comparison (Epoch {final_epoch})', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# =====  Overfitting Analysis =====
ax4 = axes[1, 1]
gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
ax4.plot(epochs, gap, 'purple', linewidth=2, marker='o', markersize=5)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.fill_between(epochs, gap, 0, where=[g > 0 for g in gap], 
                color='red', alpha=0.3, label='Train > Val (Overfitting)')
ax4.fill_between(epochs, gap, 0, where=[g <= 0 for g in gap], 
                color='green', alpha=0.3, label='Val ≥ Train (Good)')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Train Acc - Val Acc', fontsize=12)
ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_validation_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved training/validation analysis to {OUTPUT_DIR}/training_validation_analysis.png")
plt.close()

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Total Epochs Trained: {len(history['train_loss'])}")
print(f"Best Validation Accuracy: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc']) + 1})")
print(f"Best Validation Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
print(f"Final Accuracy Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.4f}")
print("="*60)
print("\n✅ Training complete! Run result.py to evaluate on test set.")
