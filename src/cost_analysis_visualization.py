#!/usr/bin/env python3
"""
Cost Analysis Visualization Script

Generates two types of cost analysis:
1. Ontology-Guided Focused: Stage 2 extraction + Stage 3 validation costs
2. Comparison: Baseline vs Ontology-Guided + "Wasted Money" on filtered entities

Author: Pipeline Team
Date: 2025-11-20
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from datetime import datetime
import csv

# ====================================================================================================
# Configuration
# ====================================================================================================

PROJECT_ROOT = Path(__file__).parent.parent
COST_TRACKING_DIR = PROJECT_ROOT / "outputs" / "cost_tracking"
STAGE3_ONTOLOGY_DIR = PROJECT_ROOT / "outputs" / "stage3_ontology_guided_validation"
STAGE3_BASELINE_DIR = PROJECT_ROOT / "outputs" / "stage3_baseline_validation"
OUTPUT_DIR = PROJECT_ROOT / "result_visualisation_and_analysis" / "cost_analysis"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pricing (per million tokens in USD) - from cost reports
PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
}

# ====================================================================================================
# Data Loading Functions
# ====================================================================================================

def load_ontology_guided_stage2_costs() -> Dict:
    """Load Stage 2 ontology-guided extraction costs for all documents."""
    costs = {}

    for file_path in COST_TRACKING_DIR.glob("*_ontology_guided_cost_report.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract document name (remove suffix)
        doc_name = data['document_name'].replace('_ontology_guided', '')

        costs[doc_name] = {
            'total_cost': data['summary']['total_cost_usd'],
            'input_cost': data['summary']['input_cost_usd'],
            'output_cost': data['summary']['output_cost_usd'],
            'total_segments': data['summary']['total_segments'],
            'input_tokens': data['summary']['total_input_tokens'],
            'output_tokens': data['summary']['total_output_tokens'],
        }

    return costs


def load_baseline_stage2_costs() -> Dict:
    """Load Stage 2 baseline extraction costs for all documents."""
    costs = {}

    for file_path in COST_TRACKING_DIR.glob("*_baseline_cost_report.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract document name (remove suffix)
        doc_name = data['document_name'].replace('_baseline', '')

        costs[doc_name] = {
            'total_cost': data['summary']['total_cost_usd'],
            'input_cost': data['summary']['input_cost_usd'],
            'output_cost': data['summary']['output_cost_usd'],
            'total_segments': data['summary']['total_segments'],
            'input_tokens': data['summary']['total_input_tokens'],
            'output_tokens': data['summary']['total_output_tokens'],
        }

    return costs


def load_stage3_validation_data(validation_dir: Path) -> Dict:
    """Load Stage 3 validation data including entity counts and semantic validation costs."""
    validation_data = {}

    for file_path in validation_dir.glob("*_validated.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract document name from file
        doc_name = file_path.stem.replace('_validated', '')

        metadata = data.get('validation_metadata', {})

        validation_data[doc_name] = {
            'original_entities': metadata.get('original_entity_count', 0),
            'validated_entities': metadata.get('validated_entity_count', 0),
            'semantic_validation_cost': metadata.get('semantic_validation', {}).get('llm_cost', 0.0),
            'semantic_accuracy': metadata.get('quality_metrics', {}).get('semantic_type_accuracy', 0.0),
            'schema_compliance': metadata.get('quality_metrics', {}).get('schema_compliance_weighted', 0.0),
        }

    return validation_data


# ====================================================================================================
# Analysis Functions
# ====================================================================================================

def calculate_wasted_cost(extraction_cost: float, original_entities: int, filtered_entities: int) -> float:
    """
    Calculate "wasted cost" spent on entities that were later filtered out.

    Assumption: Cost is proportional to number of entities extracted.
    Formula: (filtered_entities / original_entities) × extraction_cost
    """
    if original_entities == 0:
        return 0.0

    return (filtered_entities / original_entities) * extraction_cost


def aggregate_costs(costs_dict: Dict) -> Dict:
    """Aggregate costs across all documents."""
    total_cost = sum(doc['total_cost'] for doc in costs_dict.values())
    total_input_cost = sum(doc['input_cost'] for doc in costs_dict.values())
    total_output_cost = sum(doc['output_cost'] for doc in costs_dict.values())
    total_segments = sum(doc['total_segments'] for doc in costs_dict.values())
    total_input_tokens = sum(doc['input_tokens'] for doc in costs_dict.values())
    total_output_tokens = sum(doc['output_tokens'] for doc in costs_dict.values())

    return {
        'total_cost': total_cost,
        'input_cost': total_input_cost,
        'output_cost': total_output_cost,
        'total_segments': total_segments,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'avg_cost_per_segment': total_cost / total_segments if total_segments > 0 else 0,
    }


# ====================================================================================================
# Visualization 1: Ontology-Guided Focused Cost Analysis
# ====================================================================================================

def create_ontology_guided_cost_diagram():
    """Create focused cost diagram for ontology-guided method."""

    # Load data
    stage2_costs = load_ontology_guided_stage2_costs()
    stage3_data = load_stage3_validation_data(STAGE3_ONTOLOGY_DIR)

    # Aggregate
    stage2_agg = aggregate_costs(stage2_costs)
    total_stage3_cost = sum(doc['semantic_validation_cost'] for doc in stage3_data.values())
    total_original_entities = sum(doc['original_entities'] for doc in stage3_data.values())
    total_validated_entities = sum(doc['validated_entities'] for doc in stage3_data.values())
    total_filtered_entities = total_original_entities - total_validated_entities

    # Calculate wasted cost
    wasted_cost = calculate_wasted_cost(
        stage2_agg['total_cost'],
        total_original_entities,
        total_filtered_entities
    )

    effective_cost = stage2_agg['total_cost'] - wasted_cost + total_stage3_cost

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Ontology-Guided Cost Analysis', fontsize=16, fontweight='bold', y=0.98)

    # ================================================================================================
    # Left: Cost Breakdown Stacked Bar
    # ================================================================================================

    categories = ['Stage 2\nExtraction', 'Stage 3\nValidation', 'Total Pipeline']
    stage2_cost = [stage2_agg['total_cost'], 0, stage2_agg['total_cost']]
    stage3_cost = [0, total_stage3_cost, total_stage3_cost]

    x = np.arange(len(categories))
    width = 0.6

    bars1 = ax1.bar(x, stage2_cost, width, label='Stage 2 Extraction', color='#3498db', alpha=0.9)
    bars2 = ax1.bar(x, stage3_cost, width, bottom=stage2_cost, label='Stage 3 Validation',
                    color='#e74c3c', alpha=0.9)

    # Add value labels
    for i, (s2, s3) in enumerate(zip(stage2_cost, stage3_cost)):
        if s2 > 0:
            ax1.text(i, s2/2, f'${s2:.4f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')
        if s3 > 0:
            ax1.text(i, s2 + s3/2, f'${s3:.4f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')

        # Total on top
        total = s2 + s3
        if total > 0:
            ax1.text(i, total + total * 0.02, f'${total:.4f}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax1.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Cost Breakdown by Stage', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # ================================================================================================
    # Right: Entity Retention & Cost Efficiency
    # ================================================================================================

    retention_rate = (total_validated_entities / total_original_entities * 100) if total_original_entities > 0 else 0
    cost_per_extracted = (stage2_agg['total_cost'] / total_original_entities) if total_original_entities > 0 else 0
    cost_per_validated = (effective_cost / total_validated_entities) if total_validated_entities > 0 else 0

    # Bar chart for cost efficiency
    metrics = ['Cost per\nExtracted Entity', 'Cost per\nValidated Entity']
    costs = [cost_per_extracted, cost_per_validated]
    colors = ['#95a5a6', '#27ae60']

    bars = ax2.barh(metrics, costs, color=colors, alpha=0.9)

    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        ax2.text(cost + cost * 0.05, i, f'${cost:.6f}', va='center', fontsize=12, fontweight='bold')

    ax2.set_xlabel('Cost per Entity (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost Efficiency Analysis', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add annotation box
    textstr = f'Entity Retention: {retention_rate:.1f}%\n'
    textstr += f'Extracted: {total_original_entities} entities\n'
    textstr += f'Validated: {total_validated_entities} entities\n'
    textstr += f'Filtered: {total_filtered_entities} entities\n\n'
    textstr += f'Wasted Cost: ${wasted_cost:.4f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.98, 0.02, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "ontology_guided_cost_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()

    # ================================================================================================
    # Save CSV Tables
    # ================================================================================================

    # CSV 1: Cost Breakdown by Stage
    csv_path = OUTPUT_DIR / "ontology_guided_cost_breakdown.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Stage 2 Extraction Cost (USD)', 'Stage 3 Validation Cost (USD)', 'Total Cost (USD)'])
        writer.writerow(['Stage 2 Extraction', f'{stage2_agg["total_cost"]:.4f}', '0.0000', f'{stage2_agg["total_cost"]:.4f}'])
        writer.writerow(['Stage 3 Validation', '0.0000', f'{total_stage3_cost:.4f}', f'{total_stage3_cost:.4f}'])
        writer.writerow(['Total Pipeline', f'{stage2_agg["total_cost"]:.4f}', f'{total_stage3_cost:.4f}', f'{stage2_agg["total_cost"] + total_stage3_cost:.4f}'])
    print(f"✅ Saved: {csv_path}")

    # CSV 2: Cost Efficiency
    csv_path = OUTPUT_DIR / "ontology_guided_cost_efficiency.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Cost per Extracted Entity (USD)', f'{cost_per_extracted:.6f}'])
        writer.writerow(['Cost per Validated Entity (USD)', f'{cost_per_validated:.6f}'])
        writer.writerow(['Entity Retention Rate (%)', f'{retention_rate:.2f}'])
        writer.writerow(['Total Entities Extracted', total_original_entities])
        writer.writerow(['Total Entities Validated', total_validated_entities])
        writer.writerow(['Total Entities Filtered', total_filtered_entities])
        writer.writerow(['Wasted Cost (USD)', f'{wasted_cost:.4f}'])
    print(f"✅ Saved: {csv_path}")

    # CSV 3: Per-Document Details
    csv_path = OUTPUT_DIR / "ontology_guided_per_document_cost.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Document', 'Stage 2 Cost (USD)', 'Stage 3 Cost (USD)', 'Total Cost (USD)',
                        'Entities Extracted', 'Entities Validated', 'Retention Rate (%)', 'Wasted Cost (USD)'])

        for doc_name in sorted(stage2_costs.keys()):
            s2_cost = stage2_costs[doc_name]['total_cost']
            s3_cost = stage3_data.get(doc_name, {}).get('semantic_validation_cost', 0.0)
            total_cost = s2_cost + s3_cost

            original = stage3_data.get(doc_name, {}).get('original_entities', 0)
            validated = stage3_data.get(doc_name, {}).get('validated_entities', 0)
            filtered = original - validated
            retention = (validated / original * 100) if original > 0 else 0
            wasted = calculate_wasted_cost(s2_cost, original, filtered)

            writer.writerow([doc_name, f'{s2_cost:.4f}', f'{s3_cost:.4f}', f'{total_cost:.4f}',
                           original, validated, f'{retention:.2f}', f'{wasted:.4f}'])
    print(f"✅ Saved: {csv_path}")

    return {
        'stage2_total_cost': stage2_agg['total_cost'],
        'stage3_total_cost': total_stage3_cost,
        'total_cost': stage2_agg['total_cost'] + total_stage3_cost,
        'wasted_cost': wasted_cost,
        'effective_cost': effective_cost,
        'total_original_entities': total_original_entities,
        'total_validated_entities': total_validated_entities,
        'retention_rate': retention_rate,
        'cost_per_extracted': cost_per_extracted,
        'cost_per_validated': cost_per_validated,
    }


def create_ontology_guided_cost_tables(summary: Dict):
    """Create detailed cost tables for ontology-guided method."""

    # Load data
    stage2_costs = load_ontology_guided_stage2_costs()
    stage3_data = load_stage3_validation_data(STAGE3_ONTOLOGY_DIR)

    # Create text report
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("ONTOLOGY-GUIDED METHOD: DETAILED COST ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Overall Summary
    report_lines.append("-" * 100)
    report_lines.append("1. OVERALL COST SUMMARY")
    report_lines.append("-" * 100)
    report_lines.append("")
    report_lines.append(f"  Stage 2 Extraction Cost:       ${summary['stage2_total_cost']:.4f}")
    report_lines.append(f"  Stage 3 Validation Cost:       ${summary['stage3_total_cost']:.4f}")
    report_lines.append(f"  Total Pipeline Cost:           ${summary['total_cost']:.4f}")
    report_lines.append("")
    report_lines.append(f"  Entities Extracted (Stage 2):  {summary['total_original_entities']}")
    report_lines.append(f"  Entities Validated (Stage 3):  {summary['total_validated_entities']}")
    report_lines.append(f"  Entities Filtered:             {summary['total_original_entities'] - summary['total_validated_entities']}")
    report_lines.append(f"  Entity Retention Rate:         {summary['retention_rate']:.2f}%")
    report_lines.append("")
    report_lines.append(f"  Cost per Extracted Entity:     ${summary['cost_per_extracted']:.6f}")
    report_lines.append(f"  Cost per Validated Entity:     ${summary['cost_per_validated']:.6f}")
    report_lines.append(f"  Wasted Cost (on filtered):     ${summary['wasted_cost']:.4f}")
    report_lines.append("")

    # Per-Document Breakdown
    report_lines.append("-" * 100)
    report_lines.append("2. PER-DOCUMENT COST BREAKDOWN")
    report_lines.append("-" * 100)
    report_lines.append("")

    # Header
    header = f"{'Document':<50} {'Stage2 Cost':<15} {'Stage3 Cost':<15} {'Total Cost':<15} {'Entities':<20}"
    report_lines.append(header)
    report_lines.append("-" * 100)

    # Sort by document name
    sorted_docs = sorted(stage2_costs.keys())

    for doc_name in sorted_docs:
        stage2_cost = stage2_costs[doc_name]['total_cost']
        stage3_cost = stage3_data.get(doc_name, {}).get('semantic_validation_cost', 0.0)
        total_cost = stage2_cost + stage3_cost

        original = stage3_data.get(doc_name, {}).get('original_entities', 0)
        validated = stage3_data.get(doc_name, {}).get('validated_entities', 0)
        entity_str = f"{original} → {validated}"

        row = f"{doc_name:<50} ${stage2_cost:<14.4f} ${stage3_cost:<14.4f} ${total_cost:<14.4f} {entity_str:<20}"
        report_lines.append(row)

    report_lines.append("")

    # Token Usage Summary
    report_lines.append("-" * 100)
    report_lines.append("3. TOKEN USAGE SUMMARY")
    report_lines.append("-" * 100)
    report_lines.append("")

    stage2_agg = aggregate_costs(stage2_costs)

    report_lines.append(f"  Total Segments Processed:      {stage2_agg['total_segments']}")
    report_lines.append(f"  Total Input Tokens:            {stage2_agg['input_tokens']:,}")
    report_lines.append(f"  Total Output Tokens:           {stage2_agg['output_tokens']:,}")
    report_lines.append(f"  Total Tokens:                  {stage2_agg['input_tokens'] + stage2_agg['output_tokens']:,}")
    report_lines.append("")
    report_lines.append(f"  Average Tokens per Segment:    {(stage2_agg['input_tokens'] + stage2_agg['output_tokens']) / stage2_agg['total_segments']:.0f}")
    report_lines.append(f"  Average Cost per Segment:      ${stage2_agg['avg_cost_per_segment']:.4f}")
    report_lines.append("")

    # Wasted Cost Analysis
    report_lines.append("-" * 100)
    report_lines.append("4. WASTED COST ANALYSIS")
    report_lines.append("-" * 100)
    report_lines.append("")
    report_lines.append("  Methodology:")
    report_lines.append("    • Assumption: Extraction cost is proportional to number of entities extracted")
    report_lines.append("    • Formula: Wasted Cost = (Filtered Entities / Total Entities) × Extraction Cost")
    report_lines.append("")

    for doc_name in sorted_docs:
        stage2_cost = stage2_costs[doc_name]['total_cost']
        original = stage3_data.get(doc_name, {}).get('original_entities', 0)
        validated = stage3_data.get(doc_name, {}).get('validated_entities', 0)
        filtered = original - validated

        wasted = calculate_wasted_cost(stage2_cost, original, filtered)
        retention = (validated / original * 100) if original > 0 else 0

        report_lines.append(f"  {doc_name}:")
        report_lines.append(f"    Extraction Cost: ${stage2_cost:.4f}")
        report_lines.append(f"    Entities: {original} → {validated} (retention: {retention:.1f}%)")
        report_lines.append(f"    Wasted Cost: ${wasted:.4f} (spent on {filtered} filtered entities)")
        report_lines.append("")

    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)

    # Save
    output_path = OUTPUT_DIR / "ontology_guided_cost_analysis.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Saved: {output_path}")


# ====================================================================================================
# Visualization 2: Comparison Cost Analysis (Baseline vs Ontology-Guided)
# ====================================================================================================

def create_comparison_cost_diagram():
    """Create comparison diagram showing baseline vs ontology-guided costs."""

    # Load data
    onto_stage2 = load_ontology_guided_stage2_costs()
    onto_stage3 = load_stage3_validation_data(STAGE3_ONTOLOGY_DIR)

    base_stage2 = load_baseline_stage2_costs()
    base_stage3 = load_stage3_validation_data(STAGE3_BASELINE_DIR)

    # Aggregate
    onto_s2_agg = aggregate_costs(onto_stage2)
    onto_s3_cost = sum(doc['semantic_validation_cost'] for doc in onto_stage3.values())
    onto_original = sum(doc['original_entities'] for doc in onto_stage3.values())
    onto_validated = sum(doc['validated_entities'] for doc in onto_stage3.values())
    onto_filtered = onto_original - onto_validated

    base_s2_agg = aggregate_costs(base_stage2)
    base_s3_cost = sum(doc['semantic_validation_cost'] for doc in base_stage3.values())
    base_original = sum(doc['original_entities'] for doc in base_stage3.values())
    base_validated = sum(doc['validated_entities'] for doc in base_stage3.values())
    base_filtered = base_original - base_validated

    # Calculate wasted costs
    onto_wasted = calculate_wasted_cost(onto_s2_agg['total_cost'], onto_original, onto_filtered)
    base_wasted = calculate_wasted_cost(base_s2_agg['total_cost'], base_original, base_filtered)

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cost Comparison: Baseline vs Ontology-Guided', fontsize=16, fontweight='bold', y=0.98)

    # ================================================================================================
    # Top Left: Total Cost Comparison
    # ================================================================================================

    methods = ['Baseline', 'Ontology-Guided']
    stage2_costs = [base_s2_agg['total_cost'], onto_s2_agg['total_cost']]
    stage3_costs = [base_s3_cost, onto_s3_cost]

    x = np.arange(len(methods))
    width = 0.5

    bars1 = ax1.bar(x, stage2_costs, width, label='Stage 2 Extraction', color='#3498db', alpha=0.9)
    bars2 = ax1.bar(x, stage3_costs, width, bottom=stage2_costs, label='Stage 3 Validation',
                    color='#e74c3c', alpha=0.9)

    # Add value labels
    for i, (s2, s3) in enumerate(zip(stage2_costs, stage3_costs)):
        ax1.text(i, s2/2, f'${s2:.2f}', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        ax1.text(i, s2 + s3/2, f'${s3:.3f}', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

        # Total on top
        total = s2 + s3
        ax1.text(i, total + total * 0.02, f'${total:.2f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax1.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Pipeline Cost by Method', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # ================================================================================================
    # Top Right: Wasted Cost Comparison
    # ================================================================================================

    wasted_costs = [base_wasted, onto_wasted]
    effective_costs = [base_s2_agg['total_cost'] - base_wasted + base_s3_cost,
                       onto_s2_agg['total_cost'] - onto_wasted + onto_s3_cost]

    bars1 = ax2.bar(x, effective_costs, width, label='Effective Cost (on validated)',
                    color='#27ae60', alpha=0.9)
    bars2 = ax2.bar(x, wasted_costs, width, bottom=effective_costs, label='Wasted Cost (on filtered)',
                    color='#e67e22', alpha=0.9)

    # Add value labels
    for i, (eff, wst) in enumerate(zip(effective_costs, wasted_costs)):
        ax2.text(i, eff/2, f'${eff:.3f}', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        ax2.text(i, eff + wst/2, f'${wst:.3f}', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

        # Total on top
        total = eff + wst
        ax2.text(i, total + total * 0.02, f'${total:.2f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax2.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Effective vs Wasted Cost', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # ================================================================================================
    # Bottom Left: Entity Extraction & Retention
    # ================================================================================================

    entities_extracted = [base_original, onto_original]
    entities_validated = [base_validated, onto_validated]

    x_pos = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, entities_extracted, width, label='Extracted (Stage 2)',
                    color='#95a5a6', alpha=0.9)
    bars2 = ax3.bar(x_pos + width/2, entities_validated, width, label='Validated (Stage 3)',
                    color='#27ae60', alpha=0.9)

    # Add value labels
    for i, (ext, val) in enumerate(zip(entities_extracted, entities_validated)):
        ax3.text(i - width/2, ext + ext * 0.02, str(ext), ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        ax3.text(i + width/2, val + val * 0.02, str(val), ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        # Retention rate
        retention = (val / ext * 100) if ext > 0 else 0
        ax3.text(i, max(ext, val) + max(ext, val) * 0.15, f'{retention:.1f}%', ha='center',
                va='bottom', fontsize=10, style='italic', color='darkred')

    ax3.set_ylabel('Number of Entities', fontsize=12, fontweight='bold')
    ax3.set_title('Entity Extraction & Retention', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)

    # ================================================================================================
    # Bottom Right: Cost Efficiency (per validated entity)
    # ================================================================================================

    base_cost_per_validated = (base_s2_agg['total_cost'] + base_s3_cost) / base_validated if base_validated > 0 else 0
    onto_cost_per_validated = (onto_s2_agg['total_cost'] + onto_s3_cost) / onto_validated if onto_validated > 0 else 0

    costs_per_validated = [base_cost_per_validated, onto_cost_per_validated]

    bars = ax4.barh(methods, costs_per_validated, color=['#e74c3c', '#3498db'], alpha=0.9)

    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, costs_per_validated)):
        if cost > 0:
            ax4.text(cost + cost * 0.05, i, f'${cost:.5f}', va='center', fontsize=12,
                    fontweight='bold')
        else:
            ax4.text(0.001, i, 'N/A (0 entities)', va='center', fontsize=10, style='italic')

    ax4.set_xlabel('Cost per Validated Entity (USD)', fontsize=12, fontweight='bold')
    ax4.set_title('Cost Efficiency Comparison', fontsize=13, fontweight='bold', pad=15)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    # Add savings annotation
    if base_cost_per_validated > 0 and onto_cost_per_validated > 0:
        savings_pct = ((base_cost_per_validated - onto_cost_per_validated) / base_cost_per_validated * 100)
        savings_text = f'Ontology-Guided is {abs(savings_pct):.1f}% '
        savings_text += 'cheaper' if savings_pct > 0 else 'more expensive'

        props = dict(boxstyle='round', facecolor='lightgreen' if savings_pct > 0 else 'lightcoral', alpha=0.5)
        ax4.text(0.98, 0.02, savings_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "cost_comparison_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()

    # ================================================================================================
    # Save CSV Tables
    # ================================================================================================

    # CSV 1: Total Pipeline Cost Comparison
    csv_path = OUTPUT_DIR / "comparison_total_cost.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Stage 2 Extraction Cost (USD)', 'Stage 3 Validation Cost (USD)', 'Total Cost (USD)'])
        writer.writerow(['Baseline', f'{base_s2_agg["total_cost"]:.4f}', f'{base_s3_cost:.4f}', f'{base_s2_agg["total_cost"] + base_s3_cost:.4f}'])
        writer.writerow(['Ontology-Guided', f'{onto_s2_agg["total_cost"]:.4f}', f'{onto_s3_cost:.4f}', f'{onto_s2_agg["total_cost"] + onto_s3_cost:.4f}'])
    print(f"✅ Saved: {csv_path}")

    # CSV 2: Effective vs Wasted Cost
    base_effective = base_s2_agg['total_cost'] - base_wasted + base_s3_cost
    onto_effective = onto_s2_agg['total_cost'] - onto_wasted + onto_s3_cost

    csv_path = OUTPUT_DIR / "comparison_wasted_cost.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Effective Cost (USD)', 'Wasted Cost (USD)', 'Total Cost (USD)', 'Wasted Percentage (%)'])
        base_wasted_pct = (base_wasted / (base_s2_agg['total_cost'] + base_s3_cost) * 100) if (base_s2_agg['total_cost'] + base_s3_cost) > 0 else 0
        onto_wasted_pct = (onto_wasted / (onto_s2_agg['total_cost'] + onto_s3_cost) * 100) if (onto_s2_agg['total_cost'] + onto_s3_cost) > 0 else 0
        writer.writerow(['Baseline', f'{base_effective:.4f}', f'{base_wasted:.4f}', f'{base_s2_agg["total_cost"] + base_s3_cost:.4f}', f'{base_wasted_pct:.2f}'])
        writer.writerow(['Ontology-Guided', f'{onto_effective:.4f}', f'{onto_wasted:.4f}', f'{onto_s2_agg["total_cost"] + onto_s3_cost:.4f}', f'{onto_wasted_pct:.2f}'])
    print(f"✅ Saved: {csv_path}")

    # CSV 3: Entity Extraction & Retention
    csv_path = OUTPUT_DIR / "comparison_entity_retention.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Entities Extracted', 'Entities Validated', 'Entities Filtered', 'Retention Rate (%)'])
        base_retention = (base_validated / base_original * 100) if base_original > 0 else 0
        onto_retention = (onto_validated / onto_original * 100) if onto_original > 0 else 0
        writer.writerow(['Baseline', base_original, base_validated, base_filtered, f'{base_retention:.2f}'])
        writer.writerow(['Ontology-Guided', onto_original, onto_validated, onto_filtered, f'{onto_retention:.2f}'])
    print(f"✅ Saved: {csv_path}")

    # CSV 4: Cost Efficiency
    csv_path = OUTPUT_DIR / "comparison_cost_efficiency.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Cost per Validated Entity (USD)', 'Total Cost (USD)', 'Validated Entities'])
        writer.writerow(['Baseline', f'{base_cost_per_validated:.6f}', f'{base_s2_agg["total_cost"] + base_s3_cost:.4f}', base_validated])
        writer.writerow(['Ontology-Guided', f'{onto_cost_per_validated:.6f}', f'{onto_s2_agg["total_cost"] + onto_s3_cost:.4f}', onto_validated])

        # Add efficiency comparison
        if base_cost_per_validated > 0 and onto_cost_per_validated > 0:
            savings_pct = ((base_cost_per_validated - onto_cost_per_validated) / base_cost_per_validated * 100)
            writer.writerow([])
            writer.writerow(['Efficiency Improvement', f'{savings_pct:.2f}%', '', ''])
    print(f"✅ Saved: {csv_path}")

    return {
        'baseline': {
            'stage2_cost': base_s2_agg['total_cost'],
            'stage3_cost': base_s3_cost,
            'total_cost': base_s2_agg['total_cost'] + base_s3_cost,
            'wasted_cost': base_wasted,
            'entities_extracted': base_original,
            'entities_validated': base_validated,
            'retention_rate': (base_validated / base_original * 100) if base_original > 0 else 0,
            'cost_per_validated': base_cost_per_validated,
        },
        'ontology_guided': {
            'stage2_cost': onto_s2_agg['total_cost'],
            'stage3_cost': onto_s3_cost,
            'total_cost': onto_s2_agg['total_cost'] + onto_s3_cost,
            'wasted_cost': onto_wasted,
            'entities_extracted': onto_original,
            'entities_validated': onto_validated,
            'retention_rate': (onto_validated / onto_original * 100) if onto_original > 0 else 0,
            'cost_per_validated': onto_cost_per_validated,
        }
    }


def create_comparison_cost_tables(summary: Dict):
    """Create detailed comparison tables."""

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("COST COMPARISON: BASELINE vs ONTOLOGY-GUIDED")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Side-by-side comparison
    report_lines.append("-" * 100)
    report_lines.append("1. OVERALL COMPARISON")
    report_lines.append("-" * 100)
    report_lines.append("")

    header = f"{'Metric':<40} {'Baseline':<25} {'Ontology-Guided':<25} {'Difference':<10}"
    report_lines.append(header)
    report_lines.append("-" * 100)

    metrics = [
        ('Stage 2 Extraction Cost', 'stage2_cost', '.4f', '$'),
        ('Stage 3 Validation Cost', 'stage3_cost', '.4f', '$'),
        ('Total Pipeline Cost', 'total_cost', '.4f', '$'),
        ('Wasted Cost (filtered)', 'wasted_cost', '.4f', '$'),
        ('Entities Extracted', 'entities_extracted', 'd', ''),
        ('Entities Validated', 'entities_validated', 'd', ''),
        ('Entity Retention Rate (%)', 'retention_rate', '.2f', ''),
        ('Cost per Validated Entity', 'cost_per_validated', '.6f', '$'),
    ]

    for metric_name, key, fmt, prefix in metrics:
        base_val = summary['baseline'][key]
        onto_val = summary['ontology_guided'][key]

        if 'd' in fmt:
            base_str = f"{prefix}{base_val:{fmt}}"
            onto_str = f"{prefix}{onto_val:{fmt}}"
            diff = onto_val - base_val
            diff_str = f"{diff:+d}"
        elif '%' in metric_name:
            base_str = f"{base_val:{fmt}}%"
            onto_str = f"{onto_val:{fmt}}%"
            diff = onto_val - base_val
            diff_str = f"{diff:+.2f}%"
        else:
            base_str = f"{prefix}{base_val:{fmt}}"
            onto_str = f"{prefix}{onto_val:{fmt}}"
            diff = onto_val - base_val
            if prefix == '$':
                diff_str = f"${diff:+.4f}" if diff >= 0 else f"-${abs(diff):.4f}"
            else:
                diff_str = f"{diff:+.4f}"

        row = f"{metric_name:<40} {base_str:<25} {onto_str:<25} {diff_str:<10}"
        report_lines.append(row)

    report_lines.append("")

    # Key Insights
    report_lines.append("-" * 100)
    report_lines.append("2. KEY INSIGHTS")
    report_lines.append("-" * 100)
    report_lines.append("")

    # Cost efficiency
    base_cost_per = summary['baseline']['cost_per_validated']
    onto_cost_per = summary['ontology_guided']['cost_per_validated']

    if base_cost_per > 0 and onto_cost_per > 0:
        savings_pct = ((base_cost_per - onto_cost_per) / base_cost_per * 100)
        if savings_pct > 0:
            report_lines.append(f"  ✅ Ontology-Guided is {savings_pct:.1f}% more cost-efficient per validated entity")
        else:
            report_lines.append(f"  ⚠️  Baseline is {abs(savings_pct):.1f}% more cost-efficient per validated entity")

    # Retention rate
    base_retention = summary['baseline']['retention_rate']
    onto_retention = summary['ontology_guided']['retention_rate']

    if onto_retention > base_retention:
        report_lines.append(f"  ✅ Ontology-Guided has {onto_retention - base_retention:.1f}% better entity retention")
    else:
        report_lines.append(f"  ⚠️  Baseline has {base_retention - onto_retention:.1f}% better entity retention")

    # Wasted cost
    base_wasted = summary['baseline']['wasted_cost']
    onto_wasted = summary['ontology_guided']['wasted_cost']

    report_lines.append("")
    report_lines.append(f"  💰 Baseline wasted ${base_wasted:.4f} on filtered entities")
    report_lines.append(f"  💰 Ontology-Guided wasted ${onto_wasted:.4f} on filtered entities")

    if onto_wasted < base_wasted:
        report_lines.append(f"  ✅ Ontology-Guided reduced wasted cost by ${base_wasted - onto_wasted:.4f}")
    else:
        report_lines.append(f"  ⚠️  Ontology-Guided increased wasted cost by ${onto_wasted - base_wasted:.4f}")

    report_lines.append("")

    # Explanation
    report_lines.append("-" * 100)
    report_lines.append("3. WASTED COST METHODOLOGY")
    report_lines.append("-" * 100)
    report_lines.append("")
    report_lines.append("  Definition:")
    report_lines.append("    'Wasted Cost' represents money spent on extracting entities that were")
    report_lines.append("    later filtered out during Stage 3 validation.")
    report_lines.append("")
    report_lines.append("  Calculation:")
    report_lines.append("    Wasted Cost = (Filtered Entities / Total Entities) × Extraction Cost")
    report_lines.append("")
    report_lines.append("  Assumption:")
    report_lines.append("    We assume extraction cost is proportional to the number of entities extracted.")
    report_lines.append("    This is a simplification but provides a reasonable estimate.")
    report_lines.append("")
    report_lines.append("  Interpretation:")
    report_lines.append("    • Lower wasted cost indicates better extraction quality (fewer invalid entities)")
    report_lines.append("    • Higher retention rate means less money wasted on filtering")
    report_lines.append("    • Ontology-guided extraction should reduce wasted cost through better quality")
    report_lines.append("")

    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)

    # Save
    output_path = OUTPUT_DIR / "cost_comparison_analysis.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Saved: {output_path}")


# ====================================================================================================
# Main Execution
# ====================================================================================================

def main():
    """Main execution function."""

    print("\n" + "=" * 100)
    print("COST ANALYSIS VISUALIZATION")
    print("=" * 100 + "\n")

    # Visualization 1: Ontology-Guided Focused
    print("📊 Creating Ontology-Guided Focused Cost Analysis...")
    onto_summary = create_ontology_guided_cost_diagram()
    create_ontology_guided_cost_tables(onto_summary)
    print()

    # Visualization 2: Comparison
    print("📊 Creating Comparison Cost Analysis...")
    comp_summary = create_comparison_cost_diagram()
    create_comparison_cost_tables(comp_summary)
    print()

    print("=" * 100)
    print("✅ COST ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. ontology_guided_cost_analysis.png - Focused cost diagram")
    print("  2. ontology_guided_cost_analysis.txt - Detailed cost tables")
    print("  3. cost_comparison_analysis.png - Comparison diagram")
    print("  4. cost_comparison_analysis.txt - Comparison tables")
    print()


if __name__ == "__main__":
    main()
