#!/usr/bin/env python3
"""
Cost Efficiency Analysis: Wasted vs Useful Token Spending
Analyzes how much cost was spent on entities/relationships that were filtered in Stage 3 validation
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class CostEfficiencyAnalyzer:
    """Analyzes cost efficiency by comparing Stage 2 extraction costs with Stage 3 validation results"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.stage2_ontology_dir = self.base_dir / "outputs" / "stage2_ontology_guided_extraction"
        self.stage2_baseline_dir = self.base_dir / "outputs" / "stage2_baseline_llm_extraction"
        self.stage3_ontology_dir = self.base_dir / "outputs" / "stage3_ontology_guided_validation"
        self.stage3_baseline_dir = self.base_dir / "outputs" / "stage3_baseline_llm_comparison"
        self.cost_dir = self.base_dir / "outputs" / "cost_tracking"
        self.output_dir = self.base_dir / "result_visualisation_and_analysis" / "cost_analysis"

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Document names mapping
        self.doc_names = {
            "1.SASB-semiconductors-standard_en-gb": "SASB Semiconductors",
            "1. SASB-commercial-banks-standard_en-gb": "SASB Banks",
            "1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures": "IFRS S2",
            "2.FINAL-2017-TCFD-Report": "TCFD Report",
            "2.Australia-AASBS2_09-24": "Australia AASB"
        }

    def load_efficiency_data(self):
        """Load Stage 2 extraction counts, Stage 3 validation counts, and cost data"""
        efficiency_data = []

        # Get all document names
        doc_names = set()
        for file_path in self.stage2_ontology_dir.glob("*_ontology_guided.json"):
            doc_name = file_path.stem.replace("_ontology_guided", "")
            doc_names.add(doc_name)

        for doc_name in sorted(doc_names):
            # Load Stage 2 ontology-guided data
            stage2_ont_path = self.stage2_ontology_dir / f"{doc_name}_ontology_guided.json"
            if stage2_ont_path.exists():
                with open(stage2_ont_path, 'r') as f:
                    stage2_ont = json.load(f)
                    stage2_ont_entities = len(stage2_ont.get('entities', []))
                    stage2_ont_relationships = len(stage2_ont.get('relationships', []))
            else:
                stage2_ont_entities = 0
                stage2_ont_relationships = 0

            # Load Stage 3 ontology-guided validation data
            stage3_ont_path = self.stage3_ontology_dir / f"{doc_name}_validated.json"
            if stage3_ont_path.exists():
                with open(stage3_ont_path, 'r') as f:
                    stage3_ont = json.load(f)
                    stage3_ont_entities = len(stage3_ont.get('entities', []))
                    stage3_ont_relationships = len(stage3_ont.get('relationships', []))
            else:
                stage3_ont_entities = 0
                stage3_ont_relationships = 0

            # Load Stage 2 baseline data
            stage2_base_path = self.stage2_baseline_dir / f"{doc_name}_baseline_llm.json"
            if stage2_base_path.exists():
                with open(stage2_base_path, 'r') as f:
                    stage2_base = json.load(f)
                    # Handle nested structure
                    if 'experiments' in stage2_base and '2_baseline_llm' in stage2_base['experiments']:
                        exp_data = stage2_base['experiments']['2_baseline_llm']
                    else:
                        exp_data = stage2_base
                    stage2_base_entities = len(exp_data.get('entities', []))
                    stage2_base_relationships = len(exp_data.get('relationships', []))
            else:
                stage2_base_entities = 0
                stage2_base_relationships = 0

            # Load Stage 3 baseline validation data
            stage3_base_path = self.stage3_baseline_dir / f"{doc_name}_validated.json"
            if stage3_base_path.exists():
                with open(stage3_base_path, 'r') as f:
                    stage3_base = json.load(f)
                    stage3_base_entities = len(stage3_base.get('entities', []))
                    stage3_base_relationships = len(stage3_base.get('relationships', []))
            else:
                stage3_base_entities = 0
                stage3_base_relationships = 0

            # Load cost data (ontology-guided)
            cost_ont_path = self.cost_dir / f"{doc_name}_ontology_guided_cost_report.json"
            if cost_ont_path.exists():
                with open(cost_ont_path, 'r') as f:
                    cost_ont = json.load(f)
                    summary = cost_ont.get('summary', {})
                    ont_total_cost = summary.get('total_cost_usd', 0.0)
                    ont_input_tokens = summary.get('total_input_tokens', 0)
                    ont_output_tokens = summary.get('total_output_tokens', 0)
                    ont_total_tokens = ont_input_tokens + ont_output_tokens
            else:
                ont_total_cost = 0.0
                ont_input_tokens = 0
                ont_output_tokens = 0
                ont_total_tokens = 0

            # Load cost data (baseline) - try both naming conventions
            cost_base_path = self.cost_dir / f"{doc_name}_baseline_llm_cost_report.json"
            if not cost_base_path.exists():
                cost_base_path = self.cost_dir / f"{doc_name}_baseline_cost_report.json"

            if cost_base_path.exists():
                with open(cost_base_path, 'r') as f:
                    cost_base = json.load(f)
                    summary = cost_base.get('summary', {})
                    base_total_cost = summary.get('total_cost_usd', 0.0)
                    base_input_tokens = summary.get('total_input_tokens', 0)
                    base_output_tokens = summary.get('total_output_tokens', 0)
                    base_total_tokens = base_input_tokens + base_output_tokens
            else:
                base_total_cost = 0.0
                base_input_tokens = 0
                base_output_tokens = 0
                base_total_tokens = 0

            # Calculate filtered (wasted) entities and relationships
            ont_filtered_entities = stage2_ont_entities - stage3_ont_entities
            ont_filtered_relationships = stage2_ont_relationships - stage3_ont_relationships
            base_filtered_entities = stage2_base_entities - stage3_base_entities
            base_filtered_relationships = stage2_base_relationships - stage3_base_relationships

            # Calculate retention rates
            ont_entity_retention = (stage3_ont_entities / stage2_ont_entities * 100) if stage2_ont_entities > 0 else 0
            ont_rel_retention = (stage3_ont_relationships / stage2_ont_relationships * 100) if stage2_ont_relationships > 0 else 0
            base_entity_retention = (stage3_base_entities / stage2_base_entities * 100) if stage2_base_entities > 0 else 0
            base_rel_retention = (stage3_base_relationships / stage2_base_relationships * 100) if stage2_base_relationships > 0 else 0

            # Calculate wasted costs (proportional to filtered entities)
            ont_wasted_cost = (ont_filtered_entities / stage2_ont_entities * ont_total_cost) if stage2_ont_entities > 0 else 0
            ont_useful_cost = ont_total_cost - ont_wasted_cost
            base_wasted_cost = (base_filtered_entities / stage2_base_entities * base_total_cost) if stage2_base_entities > 0 else 0
            base_useful_cost = base_total_cost - base_wasted_cost

            # Calculate wasted tokens (proportional to filtered entities)
            ont_wasted_tokens = (ont_filtered_entities / stage2_ont_entities * ont_total_tokens) if stage2_ont_entities > 0 else 0
            ont_useful_tokens = ont_total_tokens - ont_wasted_tokens
            base_wasted_tokens = (base_filtered_entities / stage2_base_entities * base_total_tokens) if stage2_base_entities > 0 else 0
            base_useful_tokens = base_total_tokens - base_wasted_tokens

            # Cost per validated entity
            ont_cost_per_validated = (ont_total_cost / stage3_ont_entities) if stage3_ont_entities > 0 else 0
            base_cost_per_validated = (base_total_cost / stage3_base_entities) if stage3_base_entities > 0 else 0

            # Efficiency ratio (% of cost spent on validated entities)
            ont_efficiency_ratio = (ont_useful_cost / ont_total_cost * 100) if ont_total_cost > 0 else 0
            base_efficiency_ratio = (base_useful_cost / base_total_cost * 100) if base_total_cost > 0 else 0

            # Store ontology-guided data
            efficiency_data.append({
                'document': doc_name,
                'doc_short': self.doc_names.get(doc_name, doc_name),
                'method': 'Ontology-Guided',
                'stage2_entities': stage2_ont_entities,
                'stage3_entities': stage3_ont_entities,
                'filtered_entities': ont_filtered_entities,
                'entity_retention_pct': ont_entity_retention,
                'stage2_relationships': stage2_ont_relationships,
                'stage3_relationships': stage3_ont_relationships,
                'filtered_relationships': ont_filtered_relationships,
                'relationship_retention_pct': ont_rel_retention,
                'total_cost': ont_total_cost,
                'wasted_cost': ont_wasted_cost,
                'useful_cost': ont_useful_cost,
                'total_tokens': ont_total_tokens,
                'wasted_tokens': ont_wasted_tokens,
                'useful_tokens': ont_useful_tokens,
                'cost_per_validated_entity': ont_cost_per_validated,
                'efficiency_ratio_pct': ont_efficiency_ratio
            })

            # Store baseline data
            efficiency_data.append({
                'document': doc_name,
                'doc_short': self.doc_names.get(doc_name, doc_name),
                'method': 'Baseline',
                'stage2_entities': stage2_base_entities,
                'stage3_entities': stage3_base_entities,
                'filtered_entities': base_filtered_entities,
                'entity_retention_pct': base_entity_retention,
                'stage2_relationships': stage2_base_relationships,
                'stage3_relationships': stage3_base_relationships,
                'filtered_relationships': base_filtered_relationships,
                'relationship_retention_pct': base_rel_retention,
                'total_cost': base_total_cost,
                'wasted_cost': base_wasted_cost,
                'useful_cost': base_useful_cost,
                'total_tokens': base_total_tokens,
                'wasted_tokens': base_wasted_tokens,
                'useful_tokens': base_useful_tokens,
                'cost_per_validated_entity': base_cost_per_validated,
                'efficiency_ratio_pct': base_efficiency_ratio
            })

        return pd.DataFrame(efficiency_data)

    def plot_wasted_vs_useful_cost(self, df):
        """Plot stacked bar chart showing wasted vs useful cost"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Prepare data
        ontology_df = df[df['method'] == 'Ontology-Guided'].sort_values('doc_short')
        baseline_df = df[df['method'] == 'Baseline'].sort_values('doc_short')

        documents = ontology_df['doc_short'].values
        x = np.arange(len(documents))
        width = 0.35

        # Ontology-Guided subplot
        ont_useful = ontology_df['useful_cost'].values
        ont_wasted = ontology_df['wasted_cost'].values

        ax1.bar(x, ont_useful, width, label='Useful Cost', color='#06A77D', alpha=0.8)
        ax1.bar(x, ont_wasted, width, bottom=ont_useful, label='Wasted Cost', color='#E63946', alpha=0.8)

        ax1.set_title('Ontology-Guided: Wasted vs Useful Cost', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Document', fontsize=12)
        ax1.set_ylabel('Cost (USD)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(documents, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add efficiency ratio labels
        for i, (useful, wasted, ratio) in enumerate(zip(ont_useful, ont_wasted, ontology_df['efficiency_ratio_pct'].values)):
            total = useful + wasted
            if total > 0:
                ax1.text(i, total + 0.02, f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Baseline subplot
        base_useful = baseline_df['useful_cost'].values
        base_wasted = baseline_df['wasted_cost'].values

        ax2.bar(x, base_useful, width, label='Useful Cost', color='#06A77D', alpha=0.8)
        ax2.bar(x, base_wasted, width, bottom=base_useful, label='Wasted Cost', color='#E63946', alpha=0.8)

        ax2.set_title('Baseline: Wasted vs Useful Cost', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Document', fontsize=12)
        ax2.set_ylabel('Cost (USD)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(documents, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add efficiency ratio labels
        for i, (useful, wasted, ratio) in enumerate(zip(base_useful, base_wasted, baseline_df['efficiency_ratio_pct'].values)):
            total = useful + wasted
            if total > 0:
                ax2.text(i, total + 0.02, f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "wasted_vs_useful_cost.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def save_wasted_vs_useful_cost_table(self, df):
        """Save wasted vs useful cost data table as CSV"""
        # Create table with cost breakdown per document
        table_data = df[['doc_short', 'method', 'total_cost', 'useful_cost', 'wasted_cost', 'efficiency_ratio_pct']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'total_cost': 'Total Cost (USD)',
            'useful_cost': 'Useful Cost (USD)',
            'wasted_cost': 'Wasted Cost (USD)',
            'efficiency_ratio_pct': 'Efficiency Ratio (%)'
        })

        # Round values
        table_data['Total Cost (USD)'] = table_data['Total Cost (USD)'].round(3)
        table_data['Useful Cost (USD)'] = table_data['Useful Cost (USD)'].round(3)
        table_data['Wasted Cost (USD)'] = table_data['Wasted Cost (USD)'].round(3)
        table_data['Efficiency Ratio (%)'] = table_data['Efficiency Ratio (%)'].round(1)

        csv_path = self.output_dir / "wasted_vs_useful_cost.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_wasted_vs_useful_cost_explanation(self, df):
        """Save explanation for wasted vs useful cost visualization"""
        explanation_path = self.output_dir / "wasted_vs_useful_cost_EXPLANATION.txt"

        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WASTED vs USEFUL COST BREAKDOWN\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This stacked bar chart shows the cost breakdown for each document and method.\n")
            f.write("Green bars represent useful cost (spent on validated entities).\n")
            f.write("Red bars represent wasted cost (spent on filtered entities).\n")
            f.write("Percentage labels above bars show the efficiency ratio.\n\n")

            f.write("CALCULATION FORMULAS\n")
            f.write("-" * 80 + "\n\n")
            f.write("1. Wasted Cost (USD) = (Filtered Entities / Stage 2 Entities) × Total Cost\n")
            f.write("   - Represents cost spent on entities that didn't pass Stage 3 validation\n")
            f.write("   - Filtered Entities = Stage 2 Entities - Stage 3 Entities\n\n")

            f.write("2. Useful Cost (USD) = Total Cost - Wasted Cost\n")
            f.write("   - Represents cost spent on entities that passed validation\n")
            f.write("   - These entities remain in the final knowledge graph\n\n")

            f.write("3. Efficiency Ratio (%) = (Useful Cost / Total Cost) × 100\n")
            f.write("   - Shows what percentage of cost was productive\n")
            f.write("   - Higher values indicate better cost efficiency\n\n")

            f.write("KEY STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write("Ontology-Guided:\n")
            f.write(f"  • Total Cost: ${ont_df['total_cost'].sum():.3f}\n")
            f.write(f"  • Useful Cost: ${ont_df['useful_cost'].sum():.3f}\n")
            f.write(f"  • Wasted Cost: ${ont_df['wasted_cost'].sum():.3f}\n")
            f.write(f"  • Average Efficiency: {ont_df['efficiency_ratio_pct'].mean():.1f}%\n\n")

            f.write("Baseline:\n")
            f.write(f"  • Total Cost: ${base_df['total_cost'].sum():.3f}\n")
            f.write(f"  • Useful Cost: ${base_df['useful_cost'].sum():.3f}\n")
            f.write(f"  • Wasted Cost: ${base_df['wasted_cost'].sum():.3f}\n")
            f.write(f"  • Average Efficiency: {base_df['efficiency_ratio_pct'].mean():.1f}%\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("• Efficiency Ratio close to 100% = Most cost is useful (minimal waste)\n")
            f.write("• Efficiency Ratio far from 100% = Significant wasted cost on filtered entities\n")
            f.write("• Lower wasted cost = Better extraction quality and validation alignment\n\n")

        print(f"✓ Saved: {explanation_path}")

    def plot_wasted_vs_useful_tokens(self, df):
        """Plot stacked bar chart showing wasted vs useful tokens"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Prepare data
        ontology_df = df[df['method'] == 'Ontology-Guided'].sort_values('doc_short')
        baseline_df = df[df['method'] == 'Baseline'].sort_values('doc_short')

        documents = ontology_df['doc_short'].values
        x = np.arange(len(documents))
        width = 0.35

        # Ontology-Guided subplot
        ont_useful = ontology_df['useful_tokens'].values / 1000  # Convert to thousands
        ont_wasted = ontology_df['wasted_tokens'].values / 1000

        ax1.bar(x, ont_useful, width, label='Useful Tokens', color='#2E86AB', alpha=0.8)
        ax1.bar(x, ont_wasted, width, bottom=ont_useful, label='Wasted Tokens', color='#F77F00', alpha=0.8)

        ax1.set_title('Ontology-Guided: Wasted vs Useful Tokens', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Document', fontsize=12)
        ax1.set_ylabel('Tokens (thousands)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(documents, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add efficiency ratio labels
        for i, (useful, wasted, ratio) in enumerate(zip(ont_useful, ont_wasted, ontology_df['efficiency_ratio_pct'].values)):
            total = useful + wasted
            if total > 0:
                ax1.text(i, total + 2, f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Baseline subplot
        base_useful = baseline_df['useful_tokens'].values / 1000
        base_wasted = baseline_df['wasted_tokens'].values / 1000

        ax2.bar(x, base_useful, width, label='Useful Tokens', color='#2E86AB', alpha=0.8)
        ax2.bar(x, base_wasted, width, bottom=base_useful, label='Wasted Tokens', color='#F77F00', alpha=0.8)

        ax2.set_title('Baseline: Wasted vs Useful Tokens', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Document', fontsize=12)
        ax2.set_ylabel('Tokens (thousands)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(documents, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add efficiency ratio labels
        for i, (useful, wasted, ratio) in enumerate(zip(base_useful, base_wasted, baseline_df['efficiency_ratio_pct'].values)):
            total = useful + wasted
            if total > 0:
                ax2.text(i, total + 2, f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "wasted_vs_useful_tokens.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def save_wasted_vs_useful_tokens_table(self, df):
        """Save wasted vs useful tokens data table as CSV"""
        table_data = df[['doc_short', 'method', 'total_tokens', 'useful_tokens', 'wasted_tokens', 'efficiency_ratio_pct']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'total_tokens': 'Total Tokens',
            'useful_tokens': 'Useful Tokens',
            'wasted_tokens': 'Wasted Tokens',
            'efficiency_ratio_pct': 'Efficiency Ratio (%)'
        })

        # Round values
        table_data['Total Tokens'] = table_data['Total Tokens'].round(0).astype(int)
        table_data['Useful Tokens'] = table_data['Useful Tokens'].round(0).astype(int)
        table_data['Wasted Tokens'] = table_data['Wasted Tokens'].round(0).astype(int)
        table_data['Efficiency Ratio (%)'] = table_data['Efficiency Ratio (%)'].round(1)

        csv_path = self.output_dir / "wasted_vs_useful_tokens.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_wasted_vs_useful_tokens_explanation(self, df):
        """Save explanation for wasted vs useful tokens visualization"""
        explanation_path = self.output_dir / "wasted_vs_useful_tokens_EXPLANATION.txt"

        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WASTED vs USEFUL TOKENS BREAKDOWN\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This stacked bar chart shows the token usage breakdown for each document and method.\n")
            f.write("Blue bars represent useful tokens (for validated entities).\n")
            f.write("Orange bars represent wasted tokens (for filtered entities).\n")
            f.write("Percentage labels above bars show the efficiency ratio.\n")
            f.write("Token values are shown in thousands (K) for readability.\n\n")

            f.write("CALCULATION FORMULAS\n")
            f.write("-" * 80 + "\n\n")
            f.write("1. Total Tokens = Input Tokens + Output Tokens\n")
            f.write("   - Includes all tokens consumed during Stage 2 extraction\n\n")

            f.write("2. Wasted Tokens = (Filtered Entities / Stage 2 Entities) × Total Tokens\n")
            f.write("   - Represents tokens spent on entities that didn't pass Stage 3 validation\n")
            f.write("   - Filtered Entities = Stage 2 Entities - Stage 3 Entities\n\n")

            f.write("3. Useful Tokens = Total Tokens - Wasted Tokens\n")
            f.write("   - Represents tokens spent on entities that passed validation\n")
            f.write("   - These entities remain in the final knowledge graph\n\n")

            f.write("4. Efficiency Ratio (%) = (Useful Tokens / Total Tokens) × 100\n")
            f.write("   - Shows what percentage of token usage was productive\n")
            f.write("   - Higher values indicate better token efficiency\n\n")

            f.write("KEY STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write("Ontology-Guided:\n")
            f.write(f"  • Total Tokens: {ont_df['total_tokens'].sum():,}\n")
            f.write(f"  • Useful Tokens: {ont_df['useful_tokens'].sum():,.0f}\n")
            f.write(f"  • Wasted Tokens: {ont_df['wasted_tokens'].sum():,.0f}\n")
            f.write(f"  • Average Efficiency: {ont_df['efficiency_ratio_pct'].mean():.1f}%\n\n")

            f.write("Baseline:\n")
            f.write(f"  • Total Tokens: {base_df['total_tokens'].sum():,}\n")
            f.write(f"  • Useful Tokens: {base_df['useful_tokens'].sum():,.0f}\n")
            f.write(f"  • Wasted Tokens: {base_df['wasted_tokens'].sum():,.0f}\n")
            f.write(f"  • Average Efficiency: {base_df['efficiency_ratio_pct'].mean():.1f}%\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("• Efficiency Ratio close to 100% = Most tokens are useful (minimal waste)\n")
            f.write("• Efficiency Ratio far from 100% = Significant wasted tokens on filtered entities\n")
            f.write("• Lower wasted tokens = Better extraction quality and validation alignment\n")
            f.write("• Token efficiency directly correlates with cost efficiency\n\n")

        print(f"✓ Saved: {explanation_path}")

    def plot_efficiency_comparison(self, df):
        """Plot efficiency ratio comparison between methods"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data
        ontology_df = df[df['method'] == 'Ontology-Guided'].sort_values('doc_short')
        baseline_df = df[df['method'] == 'Baseline'].sort_values('doc_short')

        documents = ontology_df['doc_short'].values
        x = np.arange(len(documents))
        width = 0.35

        ont_efficiency = ontology_df['efficiency_ratio_pct'].values
        base_efficiency = baseline_df['efficiency_ratio_pct'].values

        ax.bar(x - width/2, ont_efficiency, width, label='Ontology-Guided', color='#06A77D', alpha=0.8)
        ax.bar(x + width/2, base_efficiency, width, label='Baseline', color='#A23B72', alpha=0.8)

        ax.set_title('Cost Efficiency Ratio: % of Cost Spent on Validated Entities', fontsize=14, fontweight='bold')
        ax.set_xlabel('Document', fontsize=12)
        ax.set_ylabel('Efficiency Ratio (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(documents, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)

        # Add value labels
        for i, (ont_val, base_val) in enumerate(zip(ont_efficiency, base_efficiency)):
            ax.text(i - width/2, ont_val + 1, f'{ont_val:.1f}%', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, base_val + 1, f'{base_val:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / "efficiency_ratio_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def save_efficiency_ratio_table(self, df):
        """Save efficiency ratio comparison data table as CSV"""
        table_data = df[['doc_short', 'method', 'stage2_entities', 'stage3_entities', 'entity_retention_pct', 'efficiency_ratio_pct']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'stage2_entities': 'Stage 2 Entities',
            'stage3_entities': 'Stage 3 Entities',
            'entity_retention_pct': 'Entity Retention (%)',
            'efficiency_ratio_pct': 'Efficiency Ratio (%)'
        })

        # Round values
        table_data['Entity Retention (%)'] = table_data['Entity Retention (%)'].round(1)
        table_data['Efficiency Ratio (%)'] = table_data['Efficiency Ratio (%)'].round(1)

        csv_path = self.output_dir / "efficiency_ratio_comparison.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_efficiency_ratio_explanation(self, df):
        """Save explanation for efficiency ratio comparison visualization"""
        explanation_path = self.output_dir / "efficiency_ratio_comparison_EXPLANATION.txt"

        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EFFICIENCY RATIO COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This grouped bar chart compares the efficiency ratios between methods for each document.\n")
            f.write("Green bars represent Ontology-Guided method.\n")
            f.write("Purple bars represent Baseline method.\n")
            f.write("Percentage labels above bars show the exact efficiency ratio values.\n\n")

            f.write("CALCULATION FORMULA\n")
            f.write("-" * 80 + "\n\n")
            f.write("Efficiency Ratio (%) = (Stage 3 Entities / Stage 2 Entities) × 100\n")
            f.write("                     = Entity Retention Rate (%)\n")
            f.write("                     = (Useful Cost / Total Cost) × 100\n\n")

            f.write("Formula Explanation:\n")
            f.write("  • Numerator: Number of entities that passed Stage 3 validation\n")
            f.write("  • Denominator: Number of entities extracted in Stage 2\n")
            f.write("  • Result: Percentage of extracted entities that were validated\n\n")

            f.write("This ratio represents:\n")
            f.write("  • What % of extracted entities remained after validation\n")
            f.write("  • What % of extraction cost was spent on validated entities\n")
            f.write("  • How well the extraction aligns with validation criteria\n\n")

            f.write("KEY STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write("Ontology-Guided:\n")
            f.write(f"  • Total Stage 2 Entities: {ont_df['stage2_entities'].sum()}\n")
            f.write(f"  • Total Stage 3 Entities: {ont_df['stage3_entities'].sum()}\n")
            f.write(f"  • Total Filtered Entities: {ont_df['filtered_entities'].sum()}\n")
            f.write(f"  • Average Efficiency Ratio: {ont_df['efficiency_ratio_pct'].mean():.1f}%\n")
            f.write(f"  • Min Efficiency: {ont_df['efficiency_ratio_pct'].min():.1f}%\n")
            f.write(f"  • Max Efficiency: {ont_df['efficiency_ratio_pct'].max():.1f}%\n\n")

            f.write("Baseline:\n")
            f.write(f"  • Total Stage 2 Entities: {base_df['stage2_entities'].sum()}\n")
            f.write(f"  • Total Stage 3 Entities: {base_df['stage3_entities'].sum()}\n")
            f.write(f"  • Total Filtered Entities: {base_df['filtered_entities'].sum()}\n")
            f.write(f"  • Average Efficiency Ratio: {base_df['efficiency_ratio_pct'].mean():.1f}%\n")
            f.write(f"  • Min Efficiency: {base_df['efficiency_ratio_pct'].min():.1f}%\n")
            f.write(f"  • Max Efficiency: {base_df['efficiency_ratio_pct'].max():.1f}%\n\n")

            efficiency_diff = ont_df['efficiency_ratio_pct'].mean() - base_df['efficiency_ratio_pct'].mean()

            f.write("COMPARISON\n")
            f.write("-" * 80 + "\n")
            if efficiency_diff > 0:
                f.write(f"Ontology-Guided is {efficiency_diff:.1f}% MORE efficient than Baseline on average.\n")
            else:
                f.write(f"Baseline is {abs(efficiency_diff):.1f}% MORE efficient than Ontology-Guided on average.\n")
            f.write("\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("• 100% Efficiency = All extracted entities passed validation (no waste)\n")
            f.write("• 50% Efficiency = Half of extracted entities were filtered (50% waste)\n")
            f.write("• Higher ratio = Better extraction quality, less rework, lower wasted cost\n")
            f.write("• Lower ratio = More entities filtered, higher wasted cost, misalignment with schema\n\n")

        print(f"✓ Saved: {explanation_path}")

    def save_efficiency_table(self, df):
        """Save comprehensive efficiency table as CSV"""
        # Reorder columns for clarity
        columns_order = [
            'document', 'doc_short', 'method',
            'stage2_entities', 'stage3_entities', 'filtered_entities', 'entity_retention_pct',
            'total_cost', 'useful_cost', 'wasted_cost', 'efficiency_ratio_pct',
            'total_tokens', 'useful_tokens', 'wasted_tokens',
            'cost_per_validated_entity'
        ]

        output_df = df[columns_order].copy()

        # Round numerical columns
        output_df['entity_retention_pct'] = output_df['entity_retention_pct'].round(1)
        output_df['total_cost'] = output_df['total_cost'].round(3)
        output_df['useful_cost'] = output_df['useful_cost'].round(3)
        output_df['wasted_cost'] = output_df['wasted_cost'].round(3)
        output_df['efficiency_ratio_pct'] = output_df['efficiency_ratio_pct'].round(1)
        output_df['cost_per_validated_entity'] = output_df['cost_per_validated_entity'].round(4)

        # Save to CSV
        csv_path = self.output_dir / "cost_efficiency_analysis.csv"
        output_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_summary_table(self, df):
        """Save aggregated summary table"""
        summary_data = []

        for method in ['Ontology-Guided', 'Baseline']:
            method_df = df[df['method'] == method]

            summary_data.append({
                'Method': method,
                'Total Documents': len(method_df),
                'Total Stage 2 Entities': method_df['stage2_entities'].sum(),
                'Total Stage 3 Entities': method_df['stage3_entities'].sum(),
                'Total Filtered Entities': method_df['filtered_entities'].sum(),
                'Average Entity Retention (%)': method_df['entity_retention_pct'].mean(),
                'Total Cost (USD)': method_df['total_cost'].sum(),
                'Total Useful Cost (USD)': method_df['useful_cost'].sum(),
                'Total Wasted Cost (USD)': method_df['wasted_cost'].sum(),
                'Average Efficiency Ratio (%)': method_df['efficiency_ratio_pct'].mean(),
                'Total Tokens': method_df['total_tokens'].sum(),
                'Total Useful Tokens': method_df['useful_tokens'].sum(),
                'Total Wasted Tokens': method_df['wasted_tokens'].sum(),
                'Average Cost per Validated Entity (USD)': method_df['cost_per_validated_entity'].mean()
            })

        summary_df = pd.DataFrame(summary_data)

        # Round numerical columns
        summary_df['Average Entity Retention (%)'] = summary_df['Average Entity Retention (%)'].round(1)
        summary_df['Total Cost (USD)'] = summary_df['Total Cost (USD)'].round(3)
        summary_df['Total Useful Cost (USD)'] = summary_df['Total Useful Cost (USD)'].round(3)
        summary_df['Total Wasted Cost (USD)'] = summary_df['Total Wasted Cost (USD)'].round(3)
        summary_df['Average Efficiency Ratio (%)'] = summary_df['Average Efficiency Ratio (%)'].round(1)
        summary_df['Average Cost per Validated Entity (USD)'] = summary_df['Average Cost per Validated Entity (USD)'].round(4)

        csv_path = self.output_dir / "cost_efficiency_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_explanation(self, df):
        """Save detailed explanation of cost efficiency metrics"""
        explanation_path = self.output_dir / "cost_efficiency_EXPLANATION.txt"

        # Calculate summary statistics
        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        ont_avg_efficiency = ont_df['efficiency_ratio_pct'].mean()
        base_avg_efficiency = base_df['efficiency_ratio_pct'].mean()
        ont_total_wasted = ont_df['wasted_cost'].sum()
        base_total_wasted = base_df['wasted_cost'].sum()
        ont_total_cost = ont_df['total_cost'].sum()
        base_total_cost = base_df['total_cost'].sum()

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COST EFFICIENCY ANALYSIS: WASTED vs USEFUL TOKEN SPENDING\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write("This analysis calculates how much cost and tokens were spent on entities that\n")
            f.write("were later filtered out during Stage 3 validation, versus those that passed\n")
            f.write("validation and remained in the final knowledge graph.\n\n")

            f.write("METHODOLOGY\n")
            f.write("-" * 80 + "\n\n")

            f.write("1. ENTITY FILTERING METRICS\n\n")
            f.write("   Filtered Entities = Stage 2 Entities - Stage 3 Entities\n")
            f.write("   Entity Retention Rate (%) = (Stage 3 Entities / Stage 2 Entities) × 100\n\n")

            f.write("2. COST ALLOCATION\n\n")
            f.write("   We assume cost is proportional to the number of entities extracted:\n\n")
            f.write("   Wasted Cost (USD) = (Filtered Entities / Stage 2 Entities) × Total Cost\n")
            f.write("   Useful Cost (USD) = Total Cost - Wasted Cost\n\n")
            f.write("   Rationale: Each entity extraction consumes tokens. Entities that don't pass\n")
            f.write("   validation represent wasted token spending because they are discarded.\n\n")

            f.write("3. TOKEN ALLOCATION\n\n")
            f.write("   Similar to cost allocation:\n\n")
            f.write("   Total Tokens = Input Tokens + Output Tokens\n")
            f.write("   Wasted Tokens = (Filtered Entities / Stage 2 Entities) × Total Tokens\n")
            f.write("   Useful Tokens = Total Tokens - Wasted Tokens\n\n")

            f.write("4. EFFICIENCY METRICS\n\n")
            f.write("   Efficiency Ratio (%) = (Useful Cost / Total Cost) × 100\n")
            f.write("                        = (Stage 3 Entities / Stage 2 Entities) × 100\n")
            f.write("                        = Entity Retention Rate (%)\n\n")
            f.write("   Cost per Validated Entity (USD) = Total Cost / Stage 3 Entities\n\n")
            f.write("   Higher efficiency ratio = more cost-effective extraction (less waste)\n")
            f.write("   Lower cost per validated entity = better value for validated entities\n\n")

            f.write("=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            f.write("ONTOLOGY-GUIDED EXTRACTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"  • Average Efficiency Ratio: {ont_avg_efficiency:.1f}%\n")
            f.write(f"  • Total Cost: ${ont_total_cost:.3f}\n")
            f.write(f"  • Total Useful Cost: ${ont_df['useful_cost'].sum():.3f}\n")
            f.write(f"  • Total Wasted Cost: ${ont_total_wasted:.3f}\n")
            f.write(f"  • Total Entities Extracted (Stage 2): {ont_df['stage2_entities'].sum()}\n")
            f.write(f"  • Total Entities Validated (Stage 3): {ont_df['stage3_entities'].sum()}\n")
            f.write(f"  • Total Entities Filtered: {ont_df['filtered_entities'].sum()}\n")
            f.write(f"  • Average Cost per Validated Entity: ${ont_df['cost_per_validated_entity'].mean():.4f}\n\n")

            f.write("BASELINE EXTRACTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"  • Average Efficiency Ratio: {base_avg_efficiency:.1f}%\n")
            f.write(f"  • Total Cost: ${base_total_cost:.3f}\n")
            f.write(f"  • Total Useful Cost: ${base_df['useful_cost'].sum():.3f}\n")
            f.write(f"  • Total Wasted Cost: ${base_total_wasted:.3f}\n")
            f.write(f"  • Total Entities Extracted (Stage 2): {base_df['stage2_entities'].sum()}\n")
            f.write(f"  • Total Entities Validated (Stage 3): {base_df['stage3_entities'].sum()}\n")
            f.write(f"  • Total Entities Filtered: {base_df['filtered_entities'].sum()}\n")
            f.write(f"  • Average Cost per Validated Entity: ${base_df['cost_per_validated_entity'].mean():.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("=" * 80 + "\n\n")

            efficiency_diff = ont_avg_efficiency - base_avg_efficiency
            wasted_diff = base_total_wasted - ont_total_wasted

            f.write(f"1. EFFICIENCY COMPARISON\n\n")
            if efficiency_diff > 0:
                f.write(f"   Ontology-guided extraction is {efficiency_diff:.1f}% MORE efficient than baseline.\n")
                f.write(f"   This means {efficiency_diff:.1f}% more of the extraction cost results in validated entities.\n\n")
            else:
                f.write(f"   Baseline extraction is {abs(efficiency_diff):.1f}% MORE efficient than ontology-guided.\n\n")

            f.write(f"2. WASTED COST COMPARISON\n\n")
            if wasted_diff > 0:
                f.write(f"   Baseline wastes ${wasted_diff:.3f} MORE than ontology-guided.\n")
                f.write(f"   This represents money spent on entities that were filtered in validation.\n\n")
            else:
                f.write(f"   Ontology-guided wastes ${abs(wasted_diff):.3f} MORE than baseline.\n\n")

            f.write(f"3. COST PER VALIDATED ENTITY\n\n")
            ont_cost_per = ont_df['cost_per_validated_entity'].mean()
            base_cost_per = base_df['cost_per_validated_entity'].mean()
            if ont_cost_per < base_cost_per:
                f.write(f"   Ontology-guided: ${ont_cost_per:.4f} per validated entity\n")
                f.write(f"   Baseline: ${base_cost_per:.4f} per validated entity\n")
                f.write(f"   Ontology-guided is ${base_cost_per - ont_cost_per:.4f} CHEAPER per validated entity.\n\n")
            else:
                f.write(f"   Ontology-guided: ${ont_cost_per:.4f} per validated entity\n")
                f.write(f"   Baseline: ${base_cost_per:.4f} per validated entity\n")
                f.write(f"   Baseline is ${ont_cost_per - base_cost_per:.4f} CHEAPER per validated entity.\n\n")

            f.write("=" * 80 + "\n")
            f.write("VISUALIZATIONS GENERATED\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. wasted_vs_useful_cost.png\n")
            f.write("   - Stacked bar charts showing cost breakdown for each document\n")
            f.write("   - Green = useful cost (validated entities)\n")
            f.write("   - Red = wasted cost (filtered entities)\n")
            f.write("   - Percentage labels show efficiency ratio\n\n")

            f.write("2. wasted_vs_useful_tokens.png\n")
            f.write("   - Stacked bar charts showing token usage breakdown\n")
            f.write("   - Blue = useful tokens\n")
            f.write("   - Orange = wasted tokens\n\n")

            f.write("3. efficiency_ratio_comparison.png\n")
            f.write("   - Side-by-side comparison of efficiency ratios\n")
            f.write("   - Shows which method achieves higher cost efficiency per document\n\n")

            f.write("=" * 80 + "\n")
            f.write("DATA TABLES GENERATED\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. cost_efficiency_analysis.csv\n")
            f.write("   - Detailed per-document metrics for both methods\n")
            f.write("   - Includes all calculated values: wasted/useful costs, tokens, ratios\n\n")

            f.write("2. cost_efficiency_summary.csv\n")
            f.write("   - Aggregated statistics for each method\n")
            f.write("   - Totals and averages across all documents\n\n")

            f.write("=" * 80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("=" * 80 + "\n\n")
            f.write("HIGH EFFICIENCY RATIO (close to 100%):\n")
            f.write("  ✓ Most extracted entities pass validation\n")
            f.write("  ✓ Low waste - cost is well-spent\n")
            f.write("  ✓ Extraction is precise and aligned with validation criteria\n\n")

            f.write("LOW EFFICIENCY RATIO (far from 100%):\n")
            f.write("  ✗ Many extracted entities are filtered in validation\n")
            f.write("  ✗ High waste - significant cost spent on discarded entities\n")
            f.write("  ✗ Extraction may be over-extracting or misaligned with schema\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF EXPLANATION\n")
            f.write("=" * 80 + "\n")

        print(f"✓ Saved: {explanation_path}")

    def plot_cost_comparison(self, df):
        """Plot total cost comparison between methods"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Prepare data
        df_plot = df.copy()
        df_plot['total_cost_display'] = df_plot['total_cost']

        # Left subplot: Total cost by document
        df_pivot = df_plot.pivot(index='doc_short', columns='method', values='total_cost_display')
        df_pivot.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'], alpha=0.8)

        ax1.set_title('Total Cost by Document', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Document', fontsize=12)
        ax1.set_ylabel('Total Cost (USD)', fontsize=12)
        ax1.legend(title='Method', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels
        for container in ax1.containers:
            ax1.bar_label(container, fmt='$%.2f', fontsize=8)

        # Right subplot: Total cost by method (aggregated)
        method_totals = df_plot.groupby('method')['total_cost'].sum().reset_index()
        ax2.bar(method_totals['method'], method_totals['total_cost'], color=['#2E86AB', '#A23B72'], alpha=0.8)

        ax2.set_title('Total Cost by Method (All Documents)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Total Cost (USD)', fontsize=12)
        ax2.legend(title='Method', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels
        for i, (method, cost) in enumerate(zip(method_totals['method'], method_totals['total_cost'])):
            ax2.text(i, cost + 0.05, f'${cost:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "cost_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def save_cost_comparison_table(self, df):
        """Save cost comparison data table as CSV"""
        table_data = df[['doc_short', 'method', 'total_cost', 'total_tokens']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'total_cost': 'Total Cost (USD)',
            'total_tokens': 'Total Tokens'
        })

        # Round values
        table_data['Total Cost (USD)'] = table_data['Total Cost (USD)'].round(3)

        csv_path = self.output_dir / "cost_comparison.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_cost_comparison_explanation(self, df):
        """Save explanation for cost comparison visualization"""
        explanation_path = self.output_dir / "cost_comparison_EXPLANATION.txt"

        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TOTAL COST COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This figure consists of two visualizations:\n\n")

            f.write("Left Panel - Total Cost by Document:\n")
            f.write("  • Grouped bar chart comparing costs between methods for each document\n")
            f.write("  • Blue bars = Ontology-Guided method\n")
            f.write("  • Purple bars = Baseline method\n")
            f.write("  • Shows per-document cost differences\n\n")

            f.write("Right Panel - Total Cost by Method:\n")
            f.write("  • Bar chart showing aggregated costs across all documents\n")
            f.write("  • Compares total spending for each method\n\n")

            f.write("DATA SOURCE\n")
            f.write("-" * 80 + "\n")
            f.write("Cost data comes from Stage 2 extraction cost tracking files:\n")
            f.write("  • outputs/cost_tracking/*_ontology_guided_cost_report.json\n")
            f.write("  • outputs/cost_tracking/*_baseline_cost_report.json\n\n")

            f.write("Total Cost includes:\n")
            f.write("  • Input token costs (prompt tokens)\n")
            f.write("  • Output token costs (generation tokens)\n")
            f.write("  • Across all segments processed for each document\n\n")

            f.write("KEY STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write("Ontology-Guided:\n")
            f.write(f"  • Total Cost (All Documents): ${ont_df['total_cost'].sum():.3f}\n")
            f.write(f"  • Average Cost per Document: ${ont_df['total_cost'].mean():.3f}\n")
            f.write(f"  • Min Cost: ${ont_df['total_cost'].min():.3f}\n")
            f.write(f"  • Max Cost: ${ont_df['total_cost'].max():.3f}\n\n")

            f.write("Baseline:\n")
            f.write(f"  • Total Cost (All Documents): ${base_df['total_cost'].sum():.3f}\n")
            f.write(f"  • Average Cost per Document: ${base_df['total_cost'].mean():.3f}\n")
            f.write(f"  • Min Cost: ${base_df['total_cost'].min():.3f}\n")
            f.write(f"  • Max Cost: ${base_df['total_cost'].max():.3f}\n\n")

            cost_diff = ont_df['total_cost'].sum() - base_df['total_cost'].sum()

            f.write("COMPARISON\n")
            f.write("-" * 80 + "\n")
            if cost_diff > 0:
                f.write(f"Ontology-Guided costs ${cost_diff:.3f} MORE than Baseline in total.\n")
                pct_diff = (cost_diff / base_df['total_cost'].sum()) * 100
                f.write(f"This represents a {pct_diff:.1f}% increase in total cost.\n")
            elif cost_diff < 0:
                f.write(f"Ontology-Guided costs ${abs(cost_diff):.3f} LESS than Baseline in total.\n")
                pct_diff = (abs(cost_diff) / base_df['total_cost'].sum()) * 100
                f.write(f"This represents a {pct_diff:.1f}% decrease in total cost.\n")
            else:
                f.write("Both methods have identical total costs.\n")
            f.write("\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("• This comparison shows RAW COSTS only, not cost efficiency\n")
            f.write("• Lower total cost doesn't necessarily mean better value\n")
            f.write("• Must consider cost efficiency (wasted vs useful spending) and quality\n")
            f.write("• See efficiency_ratio_comparison and wasted_vs_useful_cost for full picture\n\n")

        print(f"✓ Saved: {explanation_path}")

    def plot_cost_per_entity(self, df):
        """Plot cost per entity comparison"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Prepare data
        df_plot = df.copy()
        df_pivot = df_plot.pivot(index='doc_short', columns='method', values='cost_per_validated_entity')
        df_pivot.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'], alpha=0.8)

        ax.set_title('Cost Efficiency: Cost per Validated Entity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Document', fontsize=12)
        ax.set_ylabel('Cost per Entity ($)', fontsize=12)
        ax.legend(title='Method', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='$%.4f', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / "cost_per_entity.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def save_cost_per_entity_table(self, df):
        """Save cost per entity data table as CSV"""
        table_data = df[['doc_short', 'method', 'total_cost', 'stage3_entities', 'cost_per_validated_entity']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'total_cost': 'Total Cost (USD)',
            'stage3_entities': 'Validated Entities',
            'cost_per_validated_entity': 'Cost per Validated Entity (USD)'
        })

        # Round values
        table_data['Total Cost (USD)'] = table_data['Total Cost (USD)'].round(3)
        table_data['Cost per Validated Entity (USD)'] = table_data['Cost per Validated Entity (USD)'].round(4)

        csv_path = self.output_dir / "cost_per_entity.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

    def save_cost_per_entity_explanation(self, df):
        """Save explanation for cost per entity visualization"""
        explanation_path = self.output_dir / "cost_per_entity_EXPLANATION.txt"

        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COST PER VALIDATED ENTITY\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This grouped bar chart compares the cost per validated entity between methods.\n")
            f.write("Blue bars represent Ontology-Guided method.\n")
            f.write("Purple bars represent Baseline method.\n")
            f.write("Lower values indicate better cost efficiency.\n\n")

            f.write("CALCULATION FORMULA\n")
            f.write("-" * 80 + "\n\n")
            f.write("Cost per Validated Entity (USD) = Total Cost / Stage 3 Entities\n\n")

            f.write("Formula Explanation:\n")
            f.write("  • Numerator: Total cost spent on Stage 2 extraction (USD)\n")
            f.write("  • Denominator: Number of entities that passed Stage 3 validation\n")
            f.write("  • Result: Average cost to obtain one validated entity\n\n")

            f.write("This metric represents:\n")
            f.write("  • The effective cost per useful entity in the final knowledge graph\n")
            f.write("  • Takes into account both extraction cost and validation filtering\n")
            f.write("  • Lower values = better cost-effectiveness\n\n")

            f.write("KEY STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write("Ontology-Guided:\n")
            f.write(f"  • Total Cost: ${ont_df['total_cost'].sum():.3f}\n")
            f.write(f"  • Total Validated Entities: {ont_df['stage3_entities'].sum()}\n")
            f.write(f"  • Average Cost per Validated Entity: ${ont_df['cost_per_validated_entity'].mean():.4f}\n")
            f.write(f"  • Min: ${ont_df['cost_per_validated_entity'].min():.4f}\n")
            f.write(f"  • Max: ${ont_df['cost_per_validated_entity'].max():.4f}\n\n")

            f.write("Baseline:\n")
            f.write(f"  • Total Cost: ${base_df['total_cost'].sum():.3f}\n")
            f.write(f"  • Total Validated Entities: {base_df['stage3_entities'].sum()}\n")
            f.write(f"  • Average Cost per Validated Entity: ${base_df['cost_per_validated_entity'].mean():.4f}\n")
            f.write(f"  • Min: ${base_df['cost_per_validated_entity'].min():.4f}\n")
            f.write(f"  • Max: ${base_df['cost_per_validated_entity'].max():.4f}\n\n")

            cost_diff = ont_df['cost_per_validated_entity'].mean() - base_df['cost_per_validated_entity'].mean()

            f.write("COMPARISON\n")
            f.write("-" * 80 + "\n")
            if cost_diff > 0:
                f.write(f"Ontology-Guided costs ${cost_diff:.4f} MORE per validated entity than Baseline.\n")
                pct_diff = (cost_diff / base_df['cost_per_validated_entity'].mean()) * 100
                f.write(f"This represents a {pct_diff:.1f}% increase in per-entity cost.\n")
            elif cost_diff < 0:
                f.write(f"Ontology-Guided costs ${abs(cost_diff):.4f} LESS per validated entity than Baseline.\n")
                pct_diff = (abs(cost_diff) / base_df['cost_per_validated_entity'].mean()) * 100
                f.write(f"This represents a {pct_diff:.1f}% cost savings per entity.\n")
            else:
                f.write("Both methods have identical cost per validated entity.\n")
            f.write("\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("• Lower cost per entity = Better value for validated entities\n")
            f.write("• This metric accounts for filtering losses (wasted cost)\n")
            f.write("• Complements total cost comparison by showing cost-effectiveness\n")
            f.write("• Should be considered alongside quality metrics and efficiency ratio\n\n")

        print(f"✓ Saved: {explanation_path}")

    def run_analysis(self):
        """Run complete cost efficiency analysis"""
        print("\n" + "=" * 80)
        print("COST EFFICIENCY ANALYSIS: WASTED vs USEFUL SPENDING")
        print("=" * 80)
        print("")

        # Load efficiency data
        print("Loading efficiency data from Stage 2, Stage 3, and cost tracking...")
        df = self.load_efficiency_data()
        print(f"✓ Loaded data for {len(df[df['method'] == 'Ontology-Guided'])} documents × 2 methods")
        print("")

        # Generate visualizations
        print("Generating visualizations...")
        self.plot_wasted_vs_useful_cost(df)
        self.plot_wasted_vs_useful_tokens(df)
        self.plot_efficiency_comparison(df)
        self.plot_cost_comparison(df)
        self.plot_cost_per_entity(df)
        print("")

        # Save CSV tables for each visualization
        print("Generating CSV data tables...")
        self.save_wasted_vs_useful_cost_table(df)
        self.save_wasted_vs_useful_tokens_table(df)
        self.save_efficiency_ratio_table(df)
        self.save_cost_comparison_table(df)
        self.save_cost_per_entity_table(df)
        self.save_efficiency_table(df)
        self.save_summary_table(df)
        print("")

        # Save explanation files for each visualization
        print("Generating explanation files...")
        self.save_wasted_vs_useful_cost_explanation(df)
        self.save_wasted_vs_useful_tokens_explanation(df)
        self.save_efficiency_ratio_explanation(df)
        self.save_cost_comparison_explanation(df)
        self.save_cost_per_entity_explanation(df)
        self.save_explanation(df)
        print("")

        print("=" * 80)
        print("✓ COST EFFICIENCY ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print("")


def main():
    """Main execution"""
    base_dir = Path(__file__).parent.parent.parent
    analyzer = CostEfficiencyAnalyzer(base_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
