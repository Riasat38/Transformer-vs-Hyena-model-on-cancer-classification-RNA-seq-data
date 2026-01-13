import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model and utils
from hyena import RNASeqHyena
from utils import EarlyStopping, calculate_metrics

DATA_DIR = '../final_processed_data'
OUTPUT_DIR = 'hyena_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 5e-4
PATIENCE = 10

# Model parameters
NUM_LAYERS = 3
D_MODEL = 128
DROPOUT = 0.4
DIM_FEEDFORWARD = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#same data that was used for transformer is being used for hyena model
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

X_train_tensor = torch.FloatTensor(X_train).squeeze(-1)  # Remove last dimension
X_val_tensor = torch.FloatTensor(X_val).squeeze(-1)
X_test_tensor = torch.FloatTensor(X_test).squeeze(-1)

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


model = RNASeqHyena(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)


criterion = nn.CrossEntropyLoss()

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
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/hyena_best_model.pth')
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

    np.save(f'{OUTPUT_DIR}/training_history_hyena.npy', history)

with open(f'{OUTPUT_DIR}/training_history_hyena.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f" Saved training history to {OUTPUT_DIR}/")

