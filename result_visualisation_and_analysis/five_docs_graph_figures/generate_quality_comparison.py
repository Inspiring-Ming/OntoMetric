"""
Generate comparison diagram of three quality metrics (Schema Compliance, Semantic Accuracy,
Relationship Retention) across five documents for Baseline vs Ontology-Guided methods.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from validation reports
documents = ['SASB Banks', 'SASB Semi.', 'IFRS S2', 'Australia AASB', 'TCFD Report']
documents_short = ['SASB\nBanks', 'SASB\nSemi.', 'IFRS\nS2', 'Australia\nAASB', 'TCFD\nReport']

# Baseline metrics (from batch_baseline_validation.txt)
baseline_schema = [66.7, 0.0, 0.0, 83.3, 0.0]
baseline_semantic = [9.8, 4.7, 0.0, 3.1, 0.0]
baseline_relationship = [0.0, 0.0, 0.0, 0.0, 0.0]

# Ontology-Guided metrics (from validation_quality_metrics.csv)
ontology_schema = [82.72, 82.02, 81.48, 83.33, 80.09]
ontology_semantic = [79.25, 89.86, 85.00, 89.19, 64.77]
ontology_relationship = [79.25, 90.14, 89.23, 86.67, 62.79]

# Create figure with 3 subplots (one per metric)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

x = np.arange(len(documents))
width = 0.35

colors_baseline = '#E57373'  # Red
colors_ontology = '#64B5F6'  # Blue

# Plot 1: Schema Compliance
ax1 = axes[0]
bars1_base = ax1.bar(x - width/2, baseline_schema, width, label='Baseline', color=colors_baseline, edgecolor='white')
bars1_onto = ax1.bar(x + width/2, ontology_schema, width, label='Ontology-Guided', color=colors_ontology, edgecolor='white')
ax1.set_ylabel('Percentage (%)', fontsize=11)
ax1.set_title('Schema Compliance', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(documents_short, fontsize=9)
ax1.set_ylim(0, 105)
ax1.legend(loc='upper right', fontsize=9)
ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

# Add value labels
for bar in bars1_base:
    height = bar.get_height()
    if height > 0:
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for bar in bars1_onto:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Plot 2: Semantic Accuracy
ax2 = axes[1]
bars2_base = ax2.bar(x - width/2, baseline_semantic, width, label='Baseline', color=colors_baseline, edgecolor='white')
bars2_onto = ax2.bar(x + width/2, ontology_semantic, width, label='Ontology-Guided', color=colors_ontology, edgecolor='white')
ax2.set_ylabel('Percentage (%)', fontsize=11)
ax2.set_title('Semantic Accuracy', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(documents_short, fontsize=9)
ax2.set_ylim(0, 105)
ax2.legend(loc='upper right', fontsize=9)

# Add value labels
for bar in bars2_base:
    height = bar.get_height()
    if height > 0:
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for bar in bars2_onto:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Plot 3: Relationship Retention
ax3 = axes[2]
bars3_base = ax3.bar(x - width/2, baseline_relationship, width, label='Baseline', color=colors_baseline, edgecolor='white')
bars3_onto = ax3.bar(x + width/2, ontology_relationship, width, label='Ontology-Guided', color=colors_ontology, edgecolor='white')
ax3.set_ylabel('Percentage (%)', fontsize=11)
ax3.set_title('Relationship Retention', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(documents_short, fontsize=9)
ax3.set_ylim(0, 105)
ax3.legend(loc='upper right', fontsize=9)

# Add value labels
for bar in bars3_base:
    height = bar.get_height()
    if height > 0:
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for bar in bars3_onto:
    height = bar.get_height()
    ax3.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Add "0%" labels for baseline relationship retention
for i, bar in enumerate(bars3_base):
    ax3.annotate('0',
                xy=(bar.get_x() + bar.get_width() / 2, 2),
                ha='center', va='bottom', fontsize=8, color=colors_baseline)

plt.tight_layout()

# Save figure in same directory as script
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'quality_metrics_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved to: {output_path}")
print(f"PDF saved to: {output_path.replace('.png', '.pdf')}")
plt.close()
