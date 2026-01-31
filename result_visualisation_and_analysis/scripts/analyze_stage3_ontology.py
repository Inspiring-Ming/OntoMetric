#!/usr/bin/env python3
"""
Stage 3 Ontology-Guided Validation Analysis: Detailed Validation Metrics
Analyzes validation quality, rule compliance, and semantic accuracy for ontology-guided method
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import csv

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class OntologyValidationAnalyzer:
    """Analyzes detailed validation metrics for ontology-guided method"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.stage3_dir = self.base_dir / "outputs" / "stage3_ontology_guided_validation"
        self.output_dir = self.base_dir / "result_visualisation_and_analysis" / "stage3_ontology"

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

    def load_validation_data(self):
        """Load all ontology-guided validation files"""
        all_data = []

        for file_path in sorted(self.stage3_dir.glob("*_validated.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                doc_name = file_path.stem.replace("_validated", "")

                metadata = data.get('validation_metadata', {})

                all_data.append({
                    'document': doc_name,
                    'doc_short': self.doc_names.get(doc_name, doc_name),
                    'metadata': metadata,
                    'entities': data.get('entities', []),
                    'relationships': data.get('relationships', []),
                    'file_path': str(file_path)
                })

        return all_data

    def analyze_rule_compliance(self, data_list):
        """Analyze validation rule compliance across documents"""
        rule_compliance = {}

        for item in data_list:
            doc_short = item['doc_short']
            metadata = item['metadata']
            rule_scores = metadata.get('rule_scores', {})

            rule_compliance[doc_short] = {}
            for rule_id, rule_data in rule_scores.items():
                rule_compliance[doc_short][rule_id] = {
                    'score': rule_data.get('score', 0),
                    'passed': rule_data.get('score', 0) == 100.0,
                    'violations': rule_data.get('violations', 0),
                    'total_items': rule_data.get('total_items', 0)
                }

        return rule_compliance

    def plot_rule_compliance_heatmap(self, rule_compliance):
        """Plot rule compliance scores as heatmap"""
        # Extract unique rules
        all_rules = set()
        for doc_data in rule_compliance.values():
            all_rules.update(doc_data.keys())
        all_rules = sorted(all_rules)

        # Create matrix
        documents = sorted(rule_compliance.keys())
        scores_matrix = []

        for doc in documents:
            row = []
            for rule in all_rules:
                score = rule_compliance[doc].get(rule, {}).get('score', 0)
                row.append(score)
            scores_matrix.append(row)

        scores_matrix = np.array(scores_matrix)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(scores_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(np.arange(len(all_rules)))
        ax.set_yticks(np.arange(len(documents)))
        ax.set_xticklabels(all_rules)
        ax.set_yticklabels(documents)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(documents)):
            for j in range(len(all_rules)):
                score = scores_matrix[i, j]
                color = 'white' if score < 50 else 'black'
                text = ax.text(j, i, f'{score:.1f}%', ha="center", va="center", color=color, fontweight='bold')

        ax.set_title('Validation Rule Compliance Scores by Document (%)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Validation Rules', fontsize=12, fontweight='bold')
        ax.set_ylabel('Documents', fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Compliance Score (%)', rotation=270, labelpad=20, fontweight='bold')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "rule_compliance_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

        plt.close()

        # Save CSV
        csv_path = self.output_dir / "rule_compliance_heatmap.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Document'] + all_rules)
            for i, doc in enumerate(documents):
                row = [doc] + [f'{scores_matrix[i, j]:.2f}' for j in range(len(all_rules))]
                writer.writerow(row)
        print(f"✅ Saved: {csv_path}")

    def plot_quality_metrics(self, data_list):
        """Plot quality metrics (schema compliance, semantic accuracy, retention)"""
        documents = []
        schema_compliance = []
        semantic_accuracy = []
        retention_rate = []

        for item in sorted(data_list, key=lambda x: x['doc_short']):
            doc_short = item['doc_short']
            metadata = item['metadata']
            quality = metadata.get('quality_metrics', {})

            documents.append(doc_short)
            schema_compliance.append(quality.get('schema_compliance_weighted', 0))
            semantic_accuracy.append(quality.get('semantic_type_accuracy', 0))
            retention_rate.append(quality.get('relationship_retention_rate', 0))

        # Create grouped bar chart
        x = np.arange(len(documents))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 8))

        bars1 = ax.bar(x - width, schema_compliance, width, label='Schema Compliance',
                       color='#3498db', alpha=0.9)
        bars2 = ax.bar(x, semantic_accuracy, width, label='Semantic Accuracy',
                       color='#e74c3c', alpha=0.9)
        bars3 = ax.bar(x + width, retention_rate, width, label='Relationship Retention',
                       color='#2ecc71', alpha=0.9)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Validation Quality Metrics by Document (Ontology-Guided)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(documents, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='lower left')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "validation_quality_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

        plt.close()

        # Save CSV
        csv_path = self.output_dir / "validation_quality_metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Document', 'Schema Compliance (%)', 'Semantic Accuracy (%)',
                           'Relationship Retention (%)'])
            for i, doc in enumerate(documents):
                writer.writerow([doc, f'{schema_compliance[i]:.2f}',
                               f'{semantic_accuracy[i]:.2f}', f'{retention_rate[i]:.2f}'])
        print(f"✅ Saved: {csv_path}")

    def plot_entity_filtering(self, data_list):
        """Plot entity filtering impact (before/after validation)"""
        documents = []
        original_entities = []
        validated_entities = []
        filtered_entities = []

        for item in sorted(data_list, key=lambda x: x['doc_short']):
            doc_short = item['doc_short']
            metadata = item['metadata']

            original = metadata.get('original_entity_count', 0)
            validated = metadata.get('validated_entity_count', 0)
            filtered = original - validated

            documents.append(doc_short)
            original_entities.append(original)
            validated_entities.append(validated)
            filtered_entities.append(filtered)

        # Create stacked bar chart
        x = np.arange(len(documents))
        width = 0.6

        fig, ax = plt.subplots(figsize=(14, 8))

        bars1 = ax.bar(x, validated_entities, width, label='Validated (Kept)',
                       color='#2ecc71', alpha=0.9)
        bars2 = ax.bar(x, filtered_entities, width, bottom=validated_entities,
                       label='Filtered (Removed)', color='#e74c3c', alpha=0.9)

        # Add value labels
        for i, (val, filt, orig) in enumerate(zip(validated_entities, filtered_entities, original_entities)):
            # Validated count
            ax.text(i, val/2, str(val), ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')

            # Filtered count
            if filt > 0:
                ax.text(i, val + filt/2, str(filt), ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')

            # Total on top
            retention = (val / orig * 100) if orig > 0 else 0
            ax.text(i, orig + orig * 0.02, f'{orig}\n({retention:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Number of Entities', fontsize=12, fontweight='bold')
        ax.set_title('Entity Filtering Impact (Ontology-Guided)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(documents, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "entity_filtering_impact.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

        plt.close()

        # Save CSV
        csv_path = self.output_dir / "entity_filtering_impact.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Document', 'Original Entities', 'Validated Entities',
                           'Filtered Entities', 'Retention Rate (%)'])
            for i, doc in enumerate(documents):
                retention = (validated_entities[i] / original_entities[i] * 100) if original_entities[i] > 0 else 0
                writer.writerow([doc, original_entities[i], validated_entities[i],
                               filtered_entities[i], f'{retention:.2f}'])
        print(f"✅ Saved: {csv_path}")

    def plot_validation_cost(self, data_list):
        """Plot semantic validation LLM costs"""
        documents = []
        validation_costs = []
        entity_counts = []

        for item in sorted(data_list, key=lambda x: x['doc_short']):
            doc_short = item['doc_short']
            metadata = item['metadata']
            semantic_val = metadata.get('semantic_validation', {})

            cost = semantic_val.get('llm_cost', 0)
            entities = metadata.get('original_entity_count', 0)

            documents.append(doc_short)
            validation_costs.append(cost)
            entity_counts.append(entities)

        # Create bar chart with dual axis
        fig, ax1 = plt.subplots(figsize=(14, 8))

        x = np.arange(len(documents))
        width = 0.35

        # Cost bars
        bars1 = ax1.bar(x - width/2, validation_costs, width, label='Validation Cost (USD)',
                        color='#3498db', alpha=0.9)

        ax1.set_ylabel('Validation Cost (USD)', fontsize=12, fontweight='bold', color='#3498db')
        ax1.set_xlabel('Documents', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#3498db')
        ax1.set_xticks(x)
        ax1.set_xticklabels(documents, rotation=45, ha='right')

        # Add value labels on bars
        for bar, cost in zip(bars1, validation_costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height * 0.02,
                    f'${cost:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#3498db')

        # Entity count bars (secondary axis)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, entity_counts, width, label='Entities Validated',
                        color='#e74c3c', alpha=0.9)

        ax2.set_ylabel('Number of Entities', fontsize=12, fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        # Add value labels on bars
        for bar, count in zip(bars2, entity_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height * 0.02,
                    str(count), ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#e74c3c')

        # Title and legends
        ax1.set_title('Semantic Validation Cost vs Entities Validated (Ontology-Guided)',
                     fontsize=14, fontweight='bold', pad=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "validation_cost_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

        plt.close()

        # Save CSV
        csv_path = self.output_dir / "validation_cost_analysis.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Document', 'Validation Cost (USD)', 'Entities Validated',
                           'Cost per Entity (USD)'])
            for i, doc in enumerate(documents):
                cost_per_entity = validation_costs[i] / entity_counts[i] if entity_counts[i] > 0 else 0
                writer.writerow([doc, f'{validation_costs[i]:.4f}', entity_counts[i],
                               f'{cost_per_entity:.6f}'])

            # Add totals
            total_cost = sum(validation_costs)
            total_entities = sum(entity_counts)
            avg_cost_per_entity = total_cost / total_entities if total_entities > 0 else 0
            writer.writerow([])
            writer.writerow(['TOTAL', f'{total_cost:.4f}', total_entities,
                           f'{avg_cost_per_entity:.6f}'])
        print(f"✅ Saved: {csv_path}")

    def generate_summary_report(self, data_list):
        """Generate text summary report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("STAGE 3 ONTOLOGY-GUIDED VALIDATION: DETAILED ANALYSIS")
        report_lines.append("=" * 100)
        report_lines.append("")

        # Overall statistics
        total_original = sum(item['metadata'].get('original_entity_count', 0) for item in data_list)
        total_validated = sum(item['metadata'].get('validated_entity_count', 0) for item in data_list)
        total_cost = sum(item['metadata'].get('semantic_validation', {}).get('llm_cost', 0) for item in data_list)

        avg_schema = np.mean([item['metadata'].get('quality_metrics', {}).get('schema_compliance_weighted', 0) for item in data_list])
        avg_semantic = np.mean([item['metadata'].get('quality_metrics', {}).get('semantic_type_accuracy', 0) for item in data_list])
        avg_retention = np.mean([item['metadata'].get('quality_metrics', {}).get('relationship_retention_rate', 0) for item in data_list])

        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 100)
        report_lines.append(f"  Total Documents Analyzed:        {len(data_list)}")
        report_lines.append(f"  Total Entities (Before):         {total_original}")
        report_lines.append(f"  Total Entities (After):          {total_validated}")
        report_lines.append(f"  Overall Retention Rate:          {(total_validated/total_original*100):.2f}%")
        report_lines.append(f"  Total Validation Cost:           ${total_cost:.4f}")
        report_lines.append("")
        report_lines.append(f"  Average Schema Compliance:       {avg_schema:.2f}%")
        report_lines.append(f"  Average Semantic Accuracy:       {avg_semantic:.2f}%")
        report_lines.append(f"  Average Relationship Retention:  {avg_retention:.2f}%")
        report_lines.append("")

        # Per-document details
        report_lines.append("PER-DOCUMENT DETAILS")
        report_lines.append("-" * 100)
        report_lines.append("")

        for item in sorted(data_list, key=lambda x: x['doc_short']):
            doc_short = item['doc_short']
            metadata = item['metadata']
            quality = metadata.get('quality_metrics', {})

            original = metadata.get('original_entity_count', 0)
            validated = metadata.get('validated_entity_count', 0)
            retention = (validated / original * 100) if original > 0 else 0

            cost = metadata.get('semantic_validation', {}).get('llm_cost', 0)

            report_lines.append(f"{doc_short}:")
            report_lines.append(f"  Entities: {original} → {validated} ({retention:.1f}% retention)")
            report_lines.append(f"  Schema Compliance: {quality.get('schema_compliance_weighted', 0):.2f}%")
            report_lines.append(f"  Semantic Accuracy: {quality.get('semantic_type_accuracy', 0):.2f}%")
            report_lines.append(f"  Relationship Retention: {quality.get('relationship_retention_rate', 0):.2f}%")
            report_lines.append(f"  Validation Cost: ${cost:.4f}")

            # Failed rules
            failed_rules = []
            for rule_id, rule_data in metadata.get('rule_scores', {}).items():
                if rule_data.get('score', 100) < 100:
                    failed_rules.append(f"{rule_id} ({rule_data.get('score', 0):.1f}%)")

            if failed_rules:
                report_lines.append(f"  Failed Rules: {', '.join(failed_rules)}")
            else:
                report_lines.append(f"  Failed Rules: None (100% compliance)")

            report_lines.append("")

        report_lines.append("=" * 100)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 100)

        # Save report
        output_path = self.output_dir / "validation_analysis_report.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"✅ Saved: {output_path}")

    def run_analysis(self):
        """Run all analyses"""
        print("\n" + "=" * 100)
        print("STAGE 3 ONTOLOGY-GUIDED VALIDATION ANALYSIS")
        print("=" * 100 + "\n")

        # Load data
        print("📂 Loading validation data...")
        data_list = self.load_validation_data()
        print(f"   Loaded {len(data_list)} documents\n")

        # Analyze rule compliance
        print("📊 Analyzing rule compliance...")
        rule_compliance = self.analyze_rule_compliance(data_list)
        self.plot_rule_compliance_heatmap(rule_compliance)
        print()

        # Plot quality metrics
        print("📊 Plotting quality metrics...")
        self.plot_quality_metrics(data_list)
        print()

        # Plot entity filtering
        print("📊 Analyzing entity filtering impact...")
        self.plot_entity_filtering(data_list)
        print()

        # Plot validation cost
        print("📊 Analyzing validation costs...")
        self.plot_validation_cost(data_list)
        print()

        # Generate summary report
        print("📝 Generating summary report...")
        self.generate_summary_report(data_list)
        print()

        print("=" * 100)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 100)
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  1. rule_compliance_heatmap.png + .csv")
        print("  2. validation_quality_metrics.png + .csv")
        print("  3. entity_filtering_impact.png + .csv")
        print("  4. validation_cost_analysis.png + .csv")
        print("  5. validation_analysis_report.txt")
        print()


def main():
    """Main entry point"""
    import sys

    # Get base directory
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path(__file__).parent.parent.parent

    # Run analysis
    analyzer = OntologyValidationAnalyzer(base_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
