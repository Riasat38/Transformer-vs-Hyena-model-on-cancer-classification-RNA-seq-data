import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model and utils
from hyena import RNASeqHyena
from utils import calculate_metrics

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Configuration
DATA_DIR = '../final_processed_data'
OUTPUT_DIR = 'hyena_outputs'
BATCH_SIZE = 32

# Model parameters (must match training configuration)
NUM_LAYERS = 3
D_MODEL = 128
DROPOUT = 0.4
DIM_FEEDFORWARD = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
X_train = np.load(f'{DATA_DIR}/X_train_transformer.npy')
X_test = np.load(f'{DATA_DIR}/X_test_transformer.npy')
y_test = np.load(f'{DATA_DIR}/y_test.npy')

# Load class names from metadata
with open('../processed_data/metadata.json', 'r') as f:
    metadata = json.load(f)
    class_names = metadata['class_names']

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

# Auto-detect number of classes and input dimension
NUM_CLASSES = len(np.unique(y_test))
INPUT_DIM = X_test.shape[1]

print(f"\nDataset info:")
print(f"  Input dimension: {INPUT_DIM}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Test samples: {len(y_test)}")

# Prepare test data
X_test_tensor = torch.FloatTensor(X_test).squeeze(-1)  # Remove last dimension
y_test_tensor = torch.LongTensor(y_test)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test batches: {len(test_loader)}")

# Load model
model = RNASeqHyena(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)

# Load trained weights
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/hyena_best_model.pth'))
model.eval()

print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

# Generate predictions and calculate test loss
all_preds = []
all_labels = []
test_loss_total = 0.0

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        test_loss_total += loss.item()
        
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Calculate average test loss
test_loss = test_loss_total / len(test_loader)

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
test_metrics = calculate_metrics(all_labels, all_preds)

print(f"\nTest Set Results:")
print(f"  Loss:      {test_loss:.6f}")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1-Score:  {test_metrics['f1']:.4f}")

# Save predictions and labels
np.save(f'{OUTPUT_DIR}/test_predictions.npy', all_preds)
np.save(f'{OUTPUT_DIR}/test_labels.npy', all_labels)

# Load training history
try:
    with open(f'{OUTPUT_DIR}/training_history_hyena.json', 'r') as f:
        history = json.load(f)
    has_history = True
    print(f"\n✓ Loaded training history from {OUTPUT_DIR}/training_history_hyena.json")
except:
    print("\nWarning: Training history not found. Skipping loss comparison plots.")
    has_history = False

# Save test results
test_results = {
    'test_loss': test_loss,
    'test_accuracy': test_metrics['accuracy'],
    'test_precision': test_metrics['precision'],
    'test_recall': test_metrics['recall'],
    'test_f1': test_metrics['f1']
}

with open(f'{OUTPUT_DIR}/test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n✓ Saved test results to {OUTPUT_DIR}/test_results.json")

# ===== GENERATE CLASSIFICATION REPORT =====
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

report_dict = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   output_dict=True)
report_text = classification_report(all_labels, all_preds, 
                                   target_names=class_names)

print(report_text)

# Save classification report to text file
with open(f'{OUTPUT_DIR}/classification_report.txt', 'w') as f:
    f.write("HYENA MODEL - CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(report_text)
    f.write("\n\n")
    f.write("TEST SET PERFORMANCE\n")
    f.write("="*60 + "\n")
    f.write(f"Test Loss:      {test_results['test_loss']:.6f}\n")
    f.write(f"Test Accuracy:  {test_results['test_accuracy']:.4f}\n")
    f.write(f"Test Precision: {test_results['test_precision']:.4f}\n")
    f.write(f"Test Recall:    {test_results['test_recall']:.4f}\n")
    f.write(f"Test F1-Score:  {test_results['test_f1']:.4f}\n")

print(f"✓ Saved classification report to {OUTPUT_DIR}/classification_report.txt")

# ===== VISUALIZATIONS =====
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# ===== FIGURE 1: Training Analysis (if history available) =====
if has_history:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hyena Model Training Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss Curves
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
    
    # Plot 2: Accuracy Curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    ax2.axhline(y=test_results['test_accuracy'], color='g', linestyle='--', 
                label=f"Test Acc: {test_results['test_accuracy']:.4f}", linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Final Metrics Comparison
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
    
    # Plot 4: Test Performance Metrics
    ax4 = axes[1, 1]
    test_metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    test_values = [
        test_results['test_accuracy'],
        test_results['test_precision'],
        test_results['test_recall'],
        test_results['test_f1']
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax4.barh(test_metric_names, test_values, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Score', fontsize=12)
    ax4.set_title('Test Set Performance', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, test_values)):
        ax4.text(val + 0.02, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved training analysis to {OUTPUT_DIR}/training_analysis.png")
    plt.close()
    
    # ===== OVERFITTING ANALYSIS =====
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    ax.plot(epochs, gap, 'purple', linewidth=2, marker='o', markersize=5)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(epochs, gap, 0, where=[g > 0 for g in gap], 
                    color='red', alpha=0.3, label='Train > Val (Overfitting)')
    ax.fill_between(epochs, gap, 0, where=[g <= 0 for g in gap], 
                    color='green', alpha=0.3, label='Val ≥ Train (Good)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Acc - Val Acc', fontsize=12)
    ax.set_title('Overfitting Analysis (Accuracy Gap)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved overfitting analysis to {OUTPUT_DIR}/overfitting_analysis.png")
    plt.close()

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(all_labels, all_preds)
num_classes = len(np.unique(all_labels))

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
       title='Confusion Matrix (Normalized) - Hyena Model')

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
plt.close()

# ===== PER-CLASS PERFORMANCE =====
classes = list(range(num_classes))
precisions = [report_dict[class_names[i]]['precision'] for i in classes]
recalls = [report_dict[class_names[i]]['recall'] for i in classes]
f1_scores = [report_dict[class_names[i]]['f1-score'] for i in classes]
supports = [report_dict[class_names[i]]['support'] for i in classes]

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
plt.close()

# ===== TRAIN/VAL/TEST COMPARISON =====
if has_history:
    # Create a comprehensive comparison figure
    fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Train vs Validation vs Test - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # Get final train and val metrics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    test_acc = test_results['test_accuracy']
    
    # Plot 1: Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    splits = ['Train', 'Validation', 'Test']
    accuracies = [final_train_acc, final_val_acc, test_acc]
    colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax1.bar(splits, accuracies, color=colors_bar, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison Across Splits', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Loss Comparison Bar Chart
    ax2 = axes[0, 1]
    losses = [final_train_loss, final_val_loss, test_loss]
    bars = ax2.bar(['Train', 'Validation', 'Test'], losses, color=colors_bar, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Comparison (Final Epoch)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: All Test Metrics
    ax3 = axes[0, 2]
    test_metric_names_comp = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    test_values_comp = [
        test_results['test_accuracy'],
        test_results['test_precision'],
        test_results['test_recall'],
        test_results['test_f1']
    ]
    colors_metrics = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax3.bar(range(len(test_metric_names_comp)), test_values_comp, 
                   color=colors_metrics, edgecolor='black', linewidth=2)
    ax3.set_xticks(range(len(test_metric_names_comp)))
    ax3.set_xticklabels(test_metric_names_comp, rotation=0)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Test Set - All Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, test_values_comp):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Accuracy Over Epochs with Test Line
    ax4 = axes[1, 0]
    epochs_comp = range(1, len(history['train_acc']) + 1)
    ax4.plot(epochs_comp, history['train_acc'], 'b-o', label='Train', linewidth=2.5, markersize=5)
    ax4.plot(epochs_comp, history['val_acc'], 'r-s', label='Validation', linewidth=2.5, markersize=5)
    ax4.axhline(y=test_acc, color='g', linestyle='--', linewidth=3, 
                label=f'Test: {test_acc:.4f}', alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Accuracy Curves + Test Baseline', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Plot 5: Loss Over Epochs
    ax5 = axes[1, 1]
    ax5.plot(epochs_comp, history['train_loss'], 'b-o', label='Train', linewidth=2.5, markersize=5)
    ax5.plot(epochs_comp, history['val_loss'], 'r-s', label='Validation', linewidth=2.5, markersize=5)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Loss', fontsize=12)
    ax5.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Generalization Gap Analysis
    ax6 = axes[1, 2]
    
    # Calculate gaps
    train_test_gap = final_train_acc - test_acc
    val_test_gap = final_val_acc - test_acc
    train_val_gap = final_train_acc - final_val_acc
    
    gap_names = ['Train-Test', 'Val-Test', 'Train-Val']
    gap_values = [train_test_gap, val_test_gap, train_val_gap]
    gap_colors = ['red' if g > 0.05 else 'orange' if g > 0.02 else 'green' for g in [abs(g) for g in gap_values]]
    
    bars = ax6.barh(gap_names, gap_values, color=gap_colors, edgecolor='black', linewidth=2)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Accuracy Gap', fontsize=12)
    ax6.set_title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, gap_values):
        width = bar.get_width()
        label_x = width + (0.002 if width > 0 else -0.002)
        ha = 'left' if width > 0 else 'right'
        ax6.text(label_x, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha=ha, va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/train_val_test_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved train/val/test comparison to {OUTPUT_DIR}/train_val_test_comparison.png")
    plt.close()

# ===== SUMMARY STATISTICS =====
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

if has_history:
    print("\nTRAINING SUMMARY")
    print("-"*60)
    print(f"Total Epochs Trained: {len(history['train_loss'])}")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc']) + 1})")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Final Accuracy Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.4f}")

print("\nTEST SET PERFORMANCE")
print("-"*60)
print(f"Test Loss:      {test_results['test_loss']:.6f}")
print(f"Test Accuracy:  {test_results['test_accuracy']:.4f}")
print(f"Test Precision: {test_results['test_precision']:.4f}")
print(f"Test Recall:    {test_results['test_recall']:.4f}")
print(f"Test F1-Score:  {test_results['test_f1']:.4f}")

print("\nPER-CLASS STATISTICS")
print("-"*60)
print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f}")
print(f"Mean F1-Score: {np.mean(f1_scores):.4f}")
print(f"Std Precision: {np.std(precisions):.4f}")
print(f"Std Recall: {np.std(recalls):.4f}")
print(f"Std F1-Score: {np.std(f1_scores):.4f}")
print("="*60)

print("\n Testing and visualization complete!")
print(f"All outputs saved to: {OUTPUT_DIR}/")
