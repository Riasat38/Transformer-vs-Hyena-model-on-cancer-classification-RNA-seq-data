import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import RNASeqTransformer  # Assuming your model is defined in model.py
import torch.optim as optim
from torch.utils.data import TensorDataset


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
OUTPUT_DIR = 'model_outputs'  
DATA_DIR = 'final_processed_data' 
BATCH_SIZE = 32  


with open('processed_data/metadata.json', 'r') as f:
    metadata = json.load(f)
    class_names = metadata['class_names']

print(f"Loaded {len(class_names)} cancer type classes")

# Load training history
with open(f'{OUTPUT_DIR}/training_history.json', 'r') as f:
    history = json.load(f)

# Load test results
with open(f'{OUTPUT_DIR}/test_results.json', 'r') as f:
    test_results = json.load(f)


X_train = np.load(f'{DATA_DIR}/X_train_transformer.npy')
X_test = np.load(f'{DATA_DIR}/X_test_transformer.npy')
y_test = np.load(f'{DATA_DIR}/y_test.npy')
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 1e-4
PATIENCE = 10

# Model parameters
NUM_CLASSES = 30  
NHEAD = 4
NUM_LAYERS = 3
D_MODEL = 192
DROPOUT = 0.3
DIM_FEEDFORWARD = 768

INPUT_DIM = X_train.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNASeqTransformer(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/transformer_best_model.pth'))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Save predictions for future use
np.save(f'{OUTPUT_DIR}/test_predictions.npy', all_preds)
np.save(f'{OUTPUT_DIR}/test_labels.npy', all_labels)


# Load predictions (if already generated)
try:
    all_preds = np.load(f'{OUTPUT_DIR}/test_predictions.npy')
    all_labels = np.load(f'{OUTPUT_DIR}/test_labels.npy')
    has_predictions = True
except:
    print("Warning: Predictions not found. Skipping confusion matrix and classification report.")
    has_predictions = False

print("\n" + "="*60)
print("GENERATING TEST SET VISUALIZATIONS")
print("="*60)

# ===== TEST PERFORMANCE METRICS FIGURE =====
fig_test, ax_test = plt.subplots(1, 1, figsize=(10, 6))
test_metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
test_values = [
    test_results['test_accuracy'],
    test_results['test_precision'],
    test_results['test_recall'],
    test_results['test_f1']
]

colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = ax_test.barh(test_metric_names, test_values, color=colors, edgecolor='black', linewidth=1.5)
ax_test.set_xlabel('Score', fontsize=12)
ax_test.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
ax_test.set_xlim([0, 1])
ax_test.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, test_values)):
    ax_test.text(val + 0.02, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/test_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved test performance metrics to {OUTPUT_DIR}/test_performance.png")
plt.close()

# ===== CONFUSION MATRIX =====
if has_predictions:
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = len(np.unique(all_labels))
    
    # Plot confusion matrix
    fig3, ax = plt.subplots(1, 1, figsize=(20, 18))
    
    # Normalize confusion matrix by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Create abbreviated class names for better readability
    class_names_abbrev = []
    for name in class_names:
        if len(name) > 25:
            # Abbreviate long names
            words = name.split()
            if len(words) > 2:
                abbrev = ' '.join([w[0] for w in words[:-1]]) + ' ' + words[-1]
            else:
                abbrev = name[:25]
            class_names_abbrev.append(abbrev)
        else:
            class_names_abbrev.append(name)
    
    # Set labels
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names_abbrev,
           yticklabels=class_names_abbrev,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix (Normalized) - Transformer Model')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # Add text annotations (only for values > 0.01 to avoid clutter)
    thresh = cm_normalized.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            if cm_normalized[i, j] > 0.01:  # Only show significant values
                text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if cm_normalized[i, j] > thresh else "black",
                             fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")
    plt.show()
    
    # ===== PER-CLASS PERFORMANCE =====
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    # Extract per-class metrics
    classes = list(range(num_classes))
    precisions = [report[class_names[i]]['precision'] for i in classes]
    recalls = [report[class_names[i]]['recall'] for i in classes]
    f1_scores = [report[class_names[i]]['f1-score'] for i in classes]
    supports = [report[class_names[i]]['support'] for i in classes]
    
    # Plot per-class performance
    fig4, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig4.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Precision per class
    ax1 = axes[0, 0]
    bars = ax1.bar(classes, precisions, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axhline(y=np.mean(precisions), color='r', linestyle='--', 
                label=f'Mean: {np.mean(precisions):.4f}', linewidth=2)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision by Class', fontsize=14, fontweight='bold')
    ax1.set_xticks(classes)
    ax1.set_xticklabels(class_names_abbrev, rotation=45, ha='right', fontsize=8)
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Recall per class
    ax2 = axes[0, 1]
    bars = ax2.bar(classes, recalls, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax2.axhline(y=np.mean(recalls), color='r', linestyle='--', 
                label=f'Mean: {np.mean(recalls):.4f}', linewidth=2)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Recall by Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(classes)
    ax2.set_xticklabels(class_names_abbrev, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: F1-Score per class
    ax3 = axes[1, 0]
    bars = ax3.bar(classes, f1_scores, color='#f39c12', edgecolor='black', alpha=0.7)
    ax3.axhline(y=np.mean(f1_scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(f1_scores):.4f}', linewidth=2)
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    ax3.set_xticks(classes)
    ax3.set_xticklabels(class_names_abbrev, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim([0, 1.05])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Support (sample count) per class
    ax4 = axes[1, 1]
    bars = ax4.bar(classes, supports, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax4.axhline(y=np.mean(supports), color='r', linestyle='--', 
                label=f'Mean: {np.mean(supports):.1f}', linewidth=2)
    ax4.set_xlabel('Class', fontsize=12)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('Sample Distribution by Class', fontsize=14, fontweight='bold')
    ax4.set_xticks(classes)
    ax4.set_xticklabels(class_names_abbrev, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-class performance to {OUTPUT_DIR}/per_class_performance.png")
    plt.show()
    
    # Save classification report to text file
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    with open(f'{OUTPUT_DIR}/classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
    print(f"✓ Saved classification report to {OUTPUT_DIR}/classification_report.txt")

# ===== SUMMARY STATISTICS =====
print("\n" + "="*60)
print("TEST SET PERFORMANCE")
print("="*60)
print(f"Test Accuracy:  {test_results['test_accuracy']:.4f}")
print(f"Test Precision: {test_results['test_precision']:.4f}")
print(f"Test Recall:    {test_results['test_recall']:.4f}")
print(f"Test F1-Score:  {test_results['test_f1']:.4f}")
print("="*60)

if has_predictions:
    print("\n" + "="*60)
    print("PER-CLASS STATISTICS")
    print("="*60)
    print(f"Mean Precision: {np.mean(precisions):.4f}")
    print(f"Mean Recall: {np.mean(recalls):.4f}")
    print(f"Mean F1-Score: {np.mean(f1_scores):.4f}")
    print(f"Std Precision: {np.std(precisions):.4f}")
    print(f"Std Recall: {np.std(recalls):.4f}")
    print(f"Std F1-Score: {np.std(f1_scores):.4f}")

    print("="*60)
