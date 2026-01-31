#!/usr/bin/env python3
"""
Generate high-fidelity publication-quality figures for conference paper
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_method_comparison_figure():
    """
    Figure 1: Method Comparison - Using Figure 3 style
    """
    fig, ax = plt.subplots(figsize=(5, 3.2))

    methods = ['Baseline', 'Ontology-Guided']
    cost_per_entity = [0.747, 0.0155]  # Fixed: baseline should be 0.747, not 0.1594
    waste_ratio = [97.23, 18.80]

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width/2, cost_per_entity, width, label='Cost per Entity (USD)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.7)

    # Create second y-axis
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, waste_ratio, width, label='Cost Waste Ratio (%)',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Cost per Entity (USD)', fontweight='bold', color='#3498db')
    ax2.set_ylabel('Cost Waste Ratio (%)', fontweight='bold', color='#e74c3c')
    ax.set_ylim(0, 0.85)  # Increased from 0.18 to accommodate 0.747
    ax2.set_ylim(0, 105)
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # Legend - inside box at upper right
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
             framealpha=0.95, handletextpad=0.5, labelspacing=0.3)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    for bar, val in zip(bars1, cost_per_entity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'${val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#3498db')

    for bar, val in zip(bars2, waste_ratio):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e74c3c')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig1_method_comparison.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig1_method_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 1 - Method Comparison")


def create_document_performance_figure():
    """
    Figure 2: Per-Document Performance - Using Figure 3 style
    """
    fig, ax = plt.subplots(figsize=(7, 3.3))

    documents = ['SASB\nBanks', 'SASB\nSemicond.', 'IFRS\nS2', 'Australia\nAASB', 'TCFD\nReport']
    cost_per_entity = [0.0176, 0.0142, 0.0126, 0.0138, 0.0207]
    waste_ratio = [20.75, 10.14, 15.00, 10.81, 35.23]

    x = np.arange(len(documents))
    width = 0.25

    bars1 = ax.bar(x - width/2, [c*1000 for c in cost_per_entity], width,
                   label='Cost per Entity (×$0.001)', color='#3498db', alpha=0.8,
                   edgecolor='black', linewidth=0.7)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, waste_ratio, width,
                    label='Cost Waste Ratio (%)', color='#e67e22', alpha=0.8,
                    edgecolor='black', linewidth=0.7)

    ax.set_xlabel('Document', fontweight='bold')
    ax.set_ylabel('Cost per Entity (×$0.001)', fontweight='bold', color='#3498db')
    ax2.set_ylabel('Cost Waste Ratio (%)', fontweight='bold', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(documents)
    ax.set_ylim(0, 23)
    ax2.set_ylim(0, 40)
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e67e22')

    # Legend - positioned with more space above box line
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
             bbox_to_anchor=(0.5, 1.17), ncol=2, framealpha=0.95,
             columnspacing=1.0, handletextpad=0.5)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}', ha='center', va='bottom', fontsize=7, color='#3498db', fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7, color='#e67e22', fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig2_document_performance.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig2_document_performance.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 2 - Document Performance")


def create_validation_quality_figure():
    """
    Figure 3: Validation Quality Metrics by Document
    Grouped bar chart for schema compliance, semantic accuracy, relationship retention
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    documents = ['SASB\nBanks', 'SASB\nSemicond.', 'IFRS\nS2', 'Australia\nAASB', 'TCFD\nReport']
    schema_compliance = [82.72, 82.02, 81.48, 83.33, 80.09]  # Fixed: SASB Banks was 89.7, should be 82.72
    semantic_accuracy = [79.25, 89.86, 85.00, 89.19, 64.77]
    relationship_retention = [79.25, 90.14, 89.23, 86.67, 62.79]

    x = np.arange(len(documents))
    width = 0.25

    bars1 = ax.bar(x - width, schema_compliance, width, label='Schema Compliance',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x, semantic_accuracy, width, label='Semantic Accuracy',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.7)
    bars3 = ax.bar(x + width, relationship_retention, width, label='Relationship Retention',
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.7)

    ax.set_xlabel('Document', fontweight='bold')
    ax.set_ylabel('Quality Score (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(documents)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower left', framealpha=0.9, ncol=3, bbox_to_anchor=(0, 1.02, 1, 0.2),
             mode='expand')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(OUTPUT_DIR / "fig3_validation_quality.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig3_validation_quality.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 3 - Validation Quality")


def create_stage2_entity_figure():
    """
    Figure 4a: Stage 2 Entity Extraction
    """
    fig, ax = plt.subplots(figsize=(3.5, 3))

    entities = ['Industry', 'Reporting\nFramework', 'Category', 'Metric', 'Model']
    stage2_counts = [20, 5, 44, 266, 29]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.barh(entities, stage2_counts, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Entity Count', fontweight='bold')
    ax.set_title('Stage 2: Extraction', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 300)

    for bar, val in zip(bars, stage2_counts):
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2.,
                f'{val}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig4a_stage2_entities.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig4a_stage2_entities.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 4a - Stage 2 Entity Extraction")


def create_stage3_entity_figure():
    """
    Figure 4b: Stage 3 Entity Validation
    """
    fig, ax = plt.subplots(figsize=(3.5, 3))

    entities = ['Industry', 'Reporting\nFramework', 'Category', 'Metric', 'Model']
    stage3_counts = [14, 5, 40, 207, 29]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.barh(entities, stage3_counts, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Entity Count', fontweight='bold')
    ax.set_title('Stage 3: Validation', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 300)

    for bar, val in zip(bars, stage3_counts):
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2.,
                f'{val}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig4b_stage3_entities.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig4b_stage3_entities.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 4b - Stage 3 Entity Validation")




def create_baseline_stage2_entity_figure():
    """
    Figure 4c: Baseline Stage 2 Entity Extraction
    """
    fig, ax = plt.subplots(figsize=(3.5, 3))

    entities = ['Industry', 'Reporting\nFramework', 'Category', 'Metric', 'Model']
    stage2_counts = [14, 0, 0, 9, 0]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.barh(entities, stage2_counts, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Entity Count', fontweight='bold')
    ax.set_title('Baseline Stage 2: Extraction', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 300)

    for bar, val in zip(bars, stage2_counts):
        width = bar.get_width()
        if val > 0:
            ax.text(width + 5, bar.get_y() + bar.get_height()/2.,
                    f'{val}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig4c_baseline_stage2_entities.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig4c_baseline_stage2_entities.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 4c - Baseline Stage 2 Entity Extraction")


def create_baseline_stage3_entity_figure():
    """
    Figure 4d: Baseline Stage 3 Entity Validation
    """
    fig, ax = plt.subplots(figsize=(3.5, 3))

    entities = ['Industry', 'Reporting\nFramework', 'Category', 'Metric', 'Model']
    stage3_counts = [3, 0, 0, 3, 0]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.barh(entities, stage3_counts, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Entity Count', fontweight='bold')
    ax.set_title('Baseline Stage 3: Validation', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 300)

    for bar, val in zip(bars, stage3_counts):
        width = bar.get_width()
        if val > 0:
            ax.text(width + 5, bar.get_y() + bar.get_height()/2.,
                    f'{val}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig4d_baseline_stage3_entities.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig4d_baseline_stage3_entities.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 4d - Baseline Stage 3 Entity Validation")


def create_2x2_entity_comparison_figure():
    """
    Figure 4 Alternative: 2x2 Grid Comparison of Entity Distributions
    Shows Baseline vs Ontology-Guided, Stage 2 vs Stage 3
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    entities = ['Industry', 'Reporting\nFramework', 'Category', 'Metric', 'Model']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Baseline Stage 2
    ax1 = axes[0, 0]
    baseline_s2 = [14, 0, 0, 9, 0]
    bars1 = ax1.barh(entities, baseline_s2, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.7)
    ax1.set_title('Baseline - Stage 2 (n=23)', fontweight='bold', fontsize=10)
    ax1.set_xlim(0, 295)  # Match ontology-guided Stage 2
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, baseline_s2):
        if val > 0:
            ax1.text(val + 5, bar.get_y() + bar.get_height()/2.,
                    f'{val}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Baseline Stage 3
    ax2 = axes[1, 0]
    baseline_s3 = [3, 0, 0, 3, 0]
    bars2 = ax2.barh(entities, baseline_s3, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.7)
    ax2.set_title('Baseline - Stage 3 (n=6)', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Entity Count', fontweight='bold', fontsize=9)
    ax2.set_xlim(0, 240)  # Match ontology-guided Stage 3
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, baseline_s3):
        if val > 0:
            ax2.text(val + 5, bar.get_y() + bar.get_height()/2.,
                    f'{val}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Ontology-Guided Stage 2
    ax3 = axes[0, 1]
    ontology_s2 = [20, 5, 44, 266, 29]
    bars3 = ax3.barh(entities, ontology_s2, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.7)
    ax3.set_title('Ontology-Guided - Stage 2 (n=364)', fontweight='bold', fontsize=10)
    ax3.set_xlim(0, 295)  # Increased from 280 to 295 to prevent label overlap
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.set_yticklabels([])  # Hide y-axis labels on right side
    for bar, val in zip(bars3, ontology_s2):
        ax3.text(val + 5, bar.get_y() + bar.get_height()/2.,
                f'{val}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Ontology-Guided Stage 3
    ax4 = axes[1, 1]
    ontology_s3 = [14, 5, 40, 207, 29]
    bars4 = ax4.barh(entities, ontology_s3, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.7)
    ax4.set_title('Ontology-Guided - Stage 3 (n=295)', fontweight='bold', fontsize=10)
    ax4.set_xlabel('Entity Count', fontweight='bold', fontsize=9)
    ax4.set_xlim(0, 240)  # Increased from 280 to 240 (207+5+buffer) to prevent label overlap
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)
    ax4.set_yticklabels([])  # Hide y-axis labels on right side
    for bar, val in zip(bars4, ontology_s3):
        ax4.text(val + 5, bar.get_y() + bar.get_height()/2.,
                f'{val}', ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "fig4_entity_comparison_2x2.png", dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "fig4_entity_comparison_2x2.pdf", bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Figure 4 (2x2 Grid) - Entity Type Comparison")


def main():
    """Generate all paper figures"""
    print("=" * 80)
    print("GENERATING PUBLICATION-QUALITY FIGURES FOR CONFERENCE PAPER")
    print("=" * 80)

    create_method_comparison_figure()
    create_document_performance_figure()
    create_validation_quality_figure()
    create_stage2_entity_figure()
    create_stage3_entity_figure()
    create_baseline_stage2_entity_figure()
    create_baseline_stage3_entity_figure()
    create_2x2_entity_comparison_figure()

    print("=" * 80)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("Available formats: PNG (300 DPI) and PDF (vector)")
    print("=" * 80)


if __name__ == "__main__":
    main()
