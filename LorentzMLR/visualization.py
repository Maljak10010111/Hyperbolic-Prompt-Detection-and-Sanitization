# %%import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up the style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# Data from the classification results
data = {
    'test_visu_em': [98.44, 97.50],
    'coco': [98.24, np.nan],  # Only benign classification available
    '4chain': [60.80, 99.60],
    'mma': [96.60, 81.20]
}

classification_types = ['Benign Accuracy (%)', 'Malicious Accuracy (%)']
datasets = list(data.keys())

# Create DataFrame
df = pd.DataFrame(data, index=classification_types)

# Sample sizes for reference
sample_sizes = {
    'test_visu_em': '5000 + 5000',
    'coco': '5000 (benign only)',
    '4chain': '500 + 500', 
    'mma': '1000 + 1000'
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Classification Results: Benign vs Malicious Detection', fontsize=16, fontweight='bold', y=0.98)

# 1. Grouped Bar Chart - Direct comparison
ax1 = axes[0, 0]
df.plot(kind='bar', ax=ax1, width=0.7)
ax1.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Classification Type', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.tick_params(axis='x', rotation=0)
ax1.legend(title='Datasets', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 105)

# Add value labels on bars
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.1f%%', padding=3)

# 2. Heatmap - Visual pattern recognition
ax2 = axes[0, 1]
mask = df.isnull()
sns.heatmap(df.T, annot=True, cmap='RdYlGn', center=80, fmt='.1f', 
            mask=mask.T, cbar_kws={'label': 'Accuracy (%)'}, ax=ax2,
            linewidths=0.5, square=True, vmin=50, vmax=100)
ax2.set_title('Accuracy Heatmap', fontsize=14, fontweight='bold')
ax2.set_xlabel('Classification Type', fontsize=12)
ax2.set_ylabel('Datasets', fontsize=12)

# 3. Scatter Plot - Benign vs Malicious relationship
ax3 = axes[1, 0]
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
markers = ['o', 's', '^', 'D']

for i, dataset in enumerate(datasets):
    benign_acc = df.loc['Benign Accuracy (%)', dataset]
    malicious_acc = df.loc['Malicious Accuracy (%)', dataset]
    
    if not np.isnan(malicious_acc):  # Only plot if malicious data exists
        ax3.scatter(benign_acc, malicious_acc, s=200, alpha=0.8, 
                   color=colors[i], marker=markers[i], 
                   edgecolors='black', linewidth=1.5, label=dataset)
        
        # Add dataset labels
        ax3.annotate(dataset, (benign_acc, malicious_acc), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')

# Add COCO point on x-axis since it has no malicious data
coco_benign = df.loc['Benign Accuracy (%)', 'coco']
ax3.scatter(coco_benign, 50, s=200, alpha=0.6, color=colors[1], 
           marker='x', linewidth=3, label='coco (benign only)')
ax3.annotate('coco\n(no malicious)', (coco_benign, 50), 
            xytext=(5, 10), textcoords='offset points', 
            fontsize=10, fontweight='bold', ha='center')

ax3.set_title('Benign vs Malicious Detection Trade-off', fontsize=14, fontweight='bold')
ax3.set_xlabel('Benign Accuracy (%)', fontsize=12)
ax3.set_ylabel('Malicious Accuracy (%)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(55, 100)
ax3.set_ylim(45, 105)

# Add diagonal line for reference (equal performance)
ax3.plot([55, 100], [55, 100], 'k--', alpha=0.5, linewidth=1, label='Equal Performance Line')
ax3.legend(loc='lower right')

# 4. Individual Dataset Detailed View
ax4 = axes[1, 1]
x_pos = np.arange(len(datasets))
width = 0.35

# Prepare data for plotting
benign_values = [df.loc['Benign Accuracy (%)', dataset] for dataset in datasets]
malicious_values = [df.loc['Malicious Accuracy (%)', dataset] if not np.isnan(df.loc['Malicious Accuracy (%)', dataset]) else 0 for dataset in datasets]

bars1 = ax4.bar(x_pos - width/2, benign_values, width, label='Benign Detection', 
                color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax4.bar(x_pos + width/2, malicious_values, width, label='Malicious Detection', 
                color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    
    ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
            f'{height1:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if datasets[i] != 'coco':  # Don't show malicious label for COCO
        ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
                f'{height2:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax4.text(bar2.get_x() + bar2.get_width()/2., 2,
                'N/A', ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

# Add sample size annotations
for i, dataset in enumerate(datasets):
    ax4.text(i, -8, f'n={sample_sizes[dataset]}', ha='center', va='top', 
            fontsize=9, style='italic', color='gray')

ax4.set_title('Dataset-wise Classification Performance', fontsize=14, fontweight='bold')
ax4.set_xlabel('Datasets', fontsize=12)
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(datasets)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 105)

plt.tight_layout()
plt.show()

# Print analysis
print("="*60)
print("CLASSIFICATION ANALYSIS")
print("="*60)

print("\nDataset Performance Summary:")
for dataset in datasets:
    benign = df.loc['Benign Accuracy (%)', dataset]
    malicious = df.loc['Malicious Accuracy (%)', dataset]
    
    print(f"\n{dataset.upper()}:")
    print(f"  Benign Detection: {benign:.2f}%")
    if not np.isnan(malicious):
        print(f"  Malicious Detection: {malicious:.2f}%")
        print(f"  Balance Score: {abs(benign - malicious):.2f}% difference")
    else:
        print(f"  Malicious Detection: Not tested")
    print(f"  Sample Size: {sample_sizes[dataset]}")

print(f"\nKey Observations:")
print(f"• Best Balanced: test_visu_em (98.44% vs 97.50% - only 0.94% difference)")
print(f"• Highest Malicious Detection: 4chain (99.60%)")
print(f"• Most Problematic: 4chain (poor benign detection at 60.80%)")
print(f"• COCO: Only benign samples tested (98.24% accuracy)")
print(f"• MMA: Good benign detection but lower malicious detection (96.60% vs 81.20%)")
# %%
