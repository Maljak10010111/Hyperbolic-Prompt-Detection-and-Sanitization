# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up the style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# Current data from classification results
accuracy_data = {
    'test_visu_em': {'benign_acc': 98.44, 'malicious_acc': 97.50, 'benign_samples': 5000, 'malicious_samples': 5000},
    'coco': {'benign_acc': 98.24, 'malicious_acc': None, 'benign_samples': 5000, 'malicious_samples': 0},
    '4chain': {'benign_acc': 60.80, 'malicious_acc': 99.60, 'benign_samples': 500, 'malicious_samples': 500},
    'mma': {'benign_acc': 96.60, 'malicious_acc': 81.20, 'benign_samples': 1000, 'malicious_samples': 1000}
}

def calculate_metrics_from_accuracy(benign_acc, malicious_acc, benign_samples, malicious_samples):
    """
    Calculate Precision, Recall, F1, and AUC from accuracy data
    """
    if malicious_acc is None:
        return None
    
    # Convert percentages to ratios
    benign_acc_ratio = benign_acc / 100.0
    malicious_acc_ratio = malicious_acc / 100.0
    
    # Calculate confusion matrix elements
    TN = int(benign_acc_ratio * benign_samples)  # True Negatives
    FP = benign_samples - TN  # False Positives
    TP = int(malicious_acc_ratio * malicious_samples)  # True Positives
    FN = malicious_samples - TP  # False Negatives
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    auc = (recall + specificity) / 2
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

# Calculate all metrics for each dataset
all_metrics = {}
for dataset, data in accuracy_data.items():
    metrics = calculate_metrics_from_accuracy(
        data['benign_acc'], 
        data['malicious_acc'], 
        data['benign_samples'], 
        data['malicious_samples']
    )
    all_metrics[dataset] = metrics

# Prepare data for visualization
datasets = ['test_visu_em', 'coco', '4chain', 'mma']
metrics_df_data = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1_Score': [],
    'AUC': [],
    'Specificity': []
}

for dataset in datasets:
    if all_metrics[dataset] is not None:
        metrics_df_data['Accuracy'].append(accuracy_data[dataset]['malicious_acc'] / 100.0)
        metrics_df_data['Precision'].append(all_metrics[dataset]['precision'])
        metrics_df_data['Recall'].append(all_metrics[dataset]['recall'])
        metrics_df_data['F1_Score'].append(all_metrics[dataset]['f1'])
        metrics_df_data['AUC'].append(all_metrics[dataset]['auc'])
        metrics_df_data['Specificity'].append(all_metrics[dataset]['specificity'])
    else:
        # COCO dataset - only benign data available
        metrics_df_data['Accuracy'].append(np.nan)
        metrics_df_data['Precision'].append(np.nan)
        metrics_df_data['Recall'].append(np.nan)
        metrics_df_data['F1_Score'].append(np.nan)
        metrics_df_data['AUC'].append(np.nan)
        metrics_df_data['Specificity'].append(accuracy_data[dataset]['benign_acc'] / 100.0)

# Create DataFrame
df_metrics = pd.DataFrame(metrics_df_data, index=datasets)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Comprehensive Classification Metrics Analysis', fontsize=16, fontweight='bold', y=0.98)

# 1. All Metrics Bar Chart
ax1 = axes[0, 0]
df_metrics.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Datasets', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.05)

# 2. Precision-Recall Scatter Plot
ax2 = axes[0, 1]
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
markers = ['o', 's', '^', 'D']

for i, dataset in enumerate(datasets):
    precision = df_metrics.loc[dataset, 'Precision']
    recall = df_metrics.loc[dataset, 'Recall']
    
    if not np.isnan(precision) and not np.isnan(recall):
        ax2.scatter(recall, precision, s=200, alpha=0.8, 
                   color=colors[i], marker=markers[i], 
                   edgecolors='black', linewidth=1.5, label=dataset)
        ax2.annotate(dataset, (recall, precision), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')

ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 1.05)
ax2.set_ylim(0, 1.05)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

# 3. F1 Score and AUC Comparison
ax3 = axes[0, 2]
f1_auc_data = df_metrics[['F1_Score', 'AUC']].dropna()
f1_auc_data.plot(kind='bar', ax=ax3, width=0.7)
ax3.set_title('F1 Score vs AUC', fontsize=14, fontweight='bold')
ax3.set_xlabel('Datasets', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(title='Metrics')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1.05)

# Add value labels on bars
for container in ax3.containers:
    ax3.bar_label(container, fmt='%.3f', padding=3)

# 4. Metrics Heatmap (FIXED)
ax4 = axes[1, 0]
# Create the mask with the correct shape
mask = df_metrics.T.isnull()  # Transpose the mask to match transposed data
sns.heatmap(df_metrics.T, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f', 
            mask=mask, cbar_kws={'label': 'Score'}, ax=ax4,
            linewidths=0.5, square=True)
ax4.set_title('All Metrics Heatmap', fontsize=14, fontweight='bold')
ax4.set_xlabel('Datasets', fontsize=12)
ax4.set_ylabel('Metrics', fontsize=12)

# 5. ROC Space Visualization
ax5 = axes[1, 1]
for i, dataset in enumerate(datasets):
    sensitivity = df_metrics.loc[dataset, 'Recall']
    specificity = df_metrics.loc[dataset, 'Specificity']
    
    if not np.isnan(sensitivity) and not np.isnan(specificity):
        fpr = 1 - specificity  # False Positive Rate
        tpr = sensitivity      # True Positive Rate
        
        ax5.scatter(fpr, tpr, s=200, alpha=0.8, 
                   color=colors[i], marker=markers[i], 
                   edgecolors='black', linewidth=1.5, label=dataset)
        ax5.annotate(dataset, (fpr, tpr), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')

ax5.set_title('ROC Space Visualization', fontsize=14, fontweight='bold')
ax5.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax5.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.set_xlim(0, 1.05)
ax5.set_ylim(0, 1.05)
ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')

# 6. Confusion Matrix Summary
ax6 = axes[1, 2]
summary_text = "Classification Summary:\n\n"
for dataset in datasets:
    if all_metrics[dataset] is not None:
        metrics = all_metrics[dataset]
        summary_text += f"{dataset}:\n"
        summary_text += f"  Precision: {metrics['precision']:.3f}\n"
        summary_text += f"  Recall:    {metrics['recall']:.3f}\n"
        summary_text += f"  F1-Score:  {metrics['f1']:.3f}\n"
        summary_text += f"  AUC:       {metrics['auc']:.3f}\n\n"
    else:
        summary_text += f"{dataset}:\n"
        summary_text += f"  Benign only\n"
        summary_text += f"  Spec: {accuracy_data[dataset]['benign_acc']/100:.3f}\n\n"

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
ax6.set_title('Detailed Metrics Summary', fontsize=14, fontweight='bold')
ax6.axis('off')

plt.tight_layout()
plt.show()

# Print comprehensive analysis
print("="*80)
print("COMPREHENSIVE CLASSIFICATION METRICS ANALYSIS")
print("="*80)

for dataset in datasets:
    print(f"\n{dataset.upper()}:")
    print("-" * 40)
    
    if all_metrics[dataset] is not None:
        metrics = all_metrics[dataset]
        print(f"Accuracy (Malicious): {accuracy_data[dataset]['malicious_acc']:.2f}%")
        print(f"Precision:           {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"F1-Score:            {metrics['f1']:.4f}")
        print(f"AUC:                 {metrics['auc']:.4f}")
        print(f"Specificity:         {metrics['specificity']:.4f}")
    else:
        print(f"Accuracy (Benign only): {accuracy_data[dataset]['benign_acc']:.2f}%")
        print("Other metrics: Not available (no malicious samples)")

print(f"\n" + "="*80)
print("PERFORMANCE RANKING:")
print("="*80)

# Rank by F1 score
valid_f1 = {k: v['f1'] for k, v in all_metrics.items() if v is not None}
sorted_f1 = sorted(valid_f1.items(), key=lambda x: x[1], reverse=True)

print("By F1-Score:")
for i, (dataset, f1) in enumerate(sorted_f1, 1):
    print(f"{i}. {dataset}: {f1:.4f}")

print(f"\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("• test_visu_em: Most balanced performance across all metrics")
print("• 4chain: High precision but poor recall balance")
print("• mma: Moderate performance across all metrics")
print("• coco: Cannot evaluate full classification performance (benign only)")
# %%