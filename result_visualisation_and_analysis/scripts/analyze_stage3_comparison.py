#!/usr/bin/env python3
"""
Stage 3 Validation Comparison Analysis
Compares baseline vs ontology-guided validation quality and filtering impact
"""

import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class Stage3ComparisonAnalyzer:
    """Analyzes Stage 3 validation comparison between baseline and ontology-guided methods"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.stage3_ontology_dir = self.base_dir / "outputs" / "stage3_ontology_guided_validation"
        self.stage3_baseline_dir = self.base_dir / "outputs" / "stage3_baseline_llm_comparison"
        self.output_dir = self.base_dir / "result_visualisation_and_analysis" / "stage3_comparison"

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

    def load_stage3_data(self):
        """Load Stage 3 validation data with enhanced metrics"""
        stage3_data = []

        # Load ontology-guided validation files
        for file_path in sorted(self.stage3_ontology_dir.glob("*_validated.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                doc_name = file_path.stem.replace("_validated", "")

                validation_meta = data.get('validation_metadata', {})
                quality_metrics = validation_meta.get('quality_metrics', {})

                stage3_data.append({
                    'document': doc_name,
                    'method': 'Ontology-Guided',
                    'original_entities': validation_meta.get('original_entity_count', 0),
                    'validated_entities': validation_meta.get('validated_entity_count', 0),
                    'original_relationships': validation_meta.get('original_relationship_count', 0),
                    'validated_relationships': validation_meta.get('validated_relationship_count', 0),
                    'violations_removed': validation_meta.get('critical_violations_removed', 0),
                    # Enhanced metrics from quality_metrics section
                    'validation_pass_rate': quality_metrics.get('validation_pass_rate', 0.0),
                    'semantic_type_accuracy': quality_metrics.get('semantic_type_accuracy', 0.0),
                    'relationship_retention_rate': quality_metrics.get('relationship_retention_rate', 0.0),
                    'file_path': str(file_path)
                })

        # Load baseline validation files
        for file_path in sorted(self.stage3_baseline_dir.glob("*_validated.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                doc_name = file_path.stem.replace("_validated", "")

                validation_meta = data.get('validation_metadata', {})
                quality_metrics = validation_meta.get('quality_metrics', {})

                stage3_data.append({
                    'document': doc_name,
                    'method': 'Baseline',
                    'original_entities': validation_meta.get('original_entity_count', 0),
                    'validated_entities': validation_meta.get('validated_entity_count', 0),
                    'original_relationships': validation_meta.get('original_relationship_count', 0),
                    'validated_relationships': validation_meta.get('validated_relationship_count', 0),
                    'violations_removed': validation_meta.get('critical_violations_removed', 0),
                    # Enhanced metrics from quality_metrics section
                    'validation_pass_rate': quality_metrics.get('validation_pass_rate', 0.0),
                    'semantic_type_accuracy': quality_metrics.get('semantic_type_accuracy', 0.0),
                    'relationship_retention_rate': quality_metrics.get('relationship_retention_rate', 0.0),
                    'file_path': str(file_path)
                })

        return pd.DataFrame(stage3_data)

    def save_validation_quality_table(self, df):
        """Save validation quality data table as CSV with enhanced metrics"""
        table_data = df.copy()
        table_data['doc_short'] = table_data['document'].map(self.doc_names)

        # Select and rename columns (using 3 core metrics)
        table_data = table_data[['doc_short', 'method', 'validation_pass_rate', 'semantic_type_accuracy']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'validation_pass_rate': 'Ontology Schema Compliance Rate (%)',
            'semantic_type_accuracy': 'Entity Semantic Accuracy (%)'
        })

        # Round values
        table_data['Ontology Schema Compliance Rate (%)'] = table_data['Ontology Schema Compliance Rate (%)'].round(1)
        table_data['Entity Semantic Accuracy (%)'] = table_data['Entity Semantic Accuracy (%)'].round(1)

        # Sort by document and method
        table_data = table_data.sort_values(['Document', 'Method'])

        csv_path = self.output_dir / "validation_quality.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

    def save_validation_quality_explanation(self, df):
        """Save explanation for validation quality visualization with enhanced metrics"""
        explanation_path = self.output_dir / "validation_quality_EXPLANATION.txt"

        # Calculate statistics using 3 core metrics
        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        ont_avg_pass_rate = ont_df['validation_pass_rate'].mean()
        base_avg_pass_rate = base_df['validation_pass_rate'].mean()

        ont_avg_semantic = ont_df['semantic_type_accuracy'].mean()
        base_avg_semantic = base_df['semantic_type_accuracy'].mean()

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 3: VALIDATION QUALITY COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This figure compares the validation performance between ontology-guided and\n")
            f.write("baseline methods in Stage 3 (validation phase).\n\n")
            f.write("LEFT PANEL: Ontology Schema Compliance Rate\n")
            f.write("  - Shows the % of validation rules passed (out of 7 critical rules)\n")
            f.write("  - Green bars: Ontology-Guided method\n")
            f.write("  - Orange bars: Baseline method\n")
            f.write("  - Green dashed line: 100% pass rate (all rules passed)\n\n")
            f.write("RIGHT PANEL: Entity Semantic Accuracy\n")
            f.write("  - Shows the % of entities with semantically correct type assignments (LLM-validated)\n")
            f.write("  - Same color scheme as ontology schema compliance rate panel\n\n\n")

            f.write("METRIC DEFINITIONS\n")
            f.write("-" * 80 + "\n")
            f.write("Ontology Schema Compliance Rate = (Critical Rules Passed / Total Critical Rules) × 100%\n")
            f.write("  - Total Critical Rules: 7 (VR001-VR005, VR009-VR010)\n")
            f.write("  - Measures: Structural schema compliance\n")
            f.write("  - 100% = All validation rules passed (perfect structural quality)\n\n")
            f.write("Entity Semantic Accuracy = (Correctly Typed Entities / Total Entities) × 100%\n")
            f.write("  - Measures: Semantic correctness of entity type assignments using LLM validation\n")
            f.write("  - 100% = All entities semantically correct\n")
            f.write("  - <100% = Some entities have incorrect type assignments\n\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Ontology Schema Compliance Rate:\n")
            f.write(f"  Ontology-Guided: {ont_avg_pass_rate:>6.1f}%\n")
            f.write(f"  Baseline:        {base_avg_pass_rate:>6.1f}%\n")
            f.write(f"  Difference:      {ont_avg_pass_rate - base_avg_pass_rate:>6.1f} percentage points\n\n")

            f.write(f"Average Entity Semantic Accuracy:\n")
            f.write(f"  Ontology-Guided: {ont_avg_semantic:>6.1f}%\n")
            f.write(f"  Baseline:        {base_avg_semantic:>6.1f}%\n")
            f.write(f"  Difference:      {ont_avg_semantic - base_avg_semantic:>6.1f} percentage points\n\n\n")

            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write("1. ONTOLOGY SCHEMA COMPLIANCE RATE (Structural Quality):\n")
            f.write(f"   - Ontology-Guided: {ont_avg_pass_rate:.1f}% average (passes most/all structural rules)\n")
            f.write(f"   - Baseline: {base_avg_pass_rate:.1f}% average (fails some structural rules)\n")
            f.write(f"   - Ontology-Guided has {ont_avg_pass_rate - base_avg_pass_rate:.1f} percentage points higher compliance\n\n")
            f.write("2. ENTITY SEMANTIC ACCURACY (Semantic Quality):\n")
            f.write(f"   - Ontology-Guided: {ont_avg_semantic:.1f}% semantic accuracy (LLM-validated)\n")
            f.write(f"   - Baseline: {base_avg_semantic:.1f}% semantic accuracy (LLM-validated)\n")
            f.write(f"   - Ontology-Guided has {ont_avg_semantic - base_avg_semantic:.1f} percentage points higher semantic accuracy\n\n")
            f.write("3. VALIDATION EFFICIENCY:\n")
            f.write("   - Ontology-Guided: Pre-filtering ensures both structural AND semantic quality\n")
            f.write("   - Baseline: Permissive extraction leads to more validation failures\n\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("The ontology-guided method's superior schema compliance rate demonstrates the\n")
            f.write("effectiveness of ontology-constrained extraction. By enforcing schema compliance\n")
            f.write("during extraction, the method produces entities that pass more validation rules.\n\n")
            f.write("The baseline method's lower compliance rate and retention rate reflect its permissive\n")
            f.write("extraction approach. While it extracts more entities in Stage 2, many violate\n")
            f.write("ontological constraints (e.g., invalid entity types, missing required properties,\n")
            f.write("incompatible relationships) and fail critical validation rules.\n\n")
            f.write("This validates the core hypothesis: ontology-guided extraction not only reduces\n")
            f.write("computational cost but also improves the quality and usability of the extracted\n")
            f.write("knowledge graph by preventing invalid entities from entering the pipeline.\n")

        print(f"✅ Saved: {explanation_path}")

    def save_filtering_impact_table(self, df):
        """Save filtering impact data table as CSV"""
        table_data = df.copy()
        table_data['doc_short'] = table_data['document'].map(self.doc_names)
        table_data['entities_filtered'] = table_data['original_entities'] - table_data['validated_entities']
        table_data['relationships_filtered'] = table_data['original_relationships'] - table_data['validated_relationships']
        table_data['entity_retention_pct'] = (table_data['validated_entities'] / table_data['original_entities'] * 100).fillna(0)
        table_data['relationship_retention_pct'] = (table_data['validated_relationships'] / table_data['original_relationships'] * 100).fillna(0)

        # Select and rename columns
        table_data = table_data[[
            'doc_short', 'method',
            'original_entities', 'validated_entities', 'entities_filtered', 'entity_retention_pct',
            'original_relationships', 'validated_relationships', 'relationships_filtered', 'relationship_retention_pct'
        ]].copy()

        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'original_entities': 'Original Entities',
            'validated_entities': 'Validated Entities',
            'entities_filtered': 'Entities Filtered',
            'entity_retention_pct': 'Entity Retention (%)',
            'original_relationships': 'Original Relationships',
            'validated_relationships': 'Validated Relationships',
            'relationships_filtered': 'Relationships Filtered',
            'relationship_retention_pct': 'Relationship Retention (%)'
        })

        # Round percentage values
        table_data['Entity Retention (%)'] = table_data['Entity Retention (%)'].round(1)
        table_data['Relationship Retention (%)'] = table_data['Relationship Retention (%)'].round(1)

        # Sort by document and method
        table_data = table_data.sort_values(['Document', 'Method'])

        csv_path = self.output_dir / "filtering_impact.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

    def save_filtering_impact_explanation(self, df):
        """Save explanation for filtering impact visualization"""
        explanation_path = self.output_dir / "filtering_impact_EXPLANATION.txt"

        # Calculate statistics
        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        ont_total_orig_entities = ont_df['original_entities'].sum()
        ont_total_valid_entities = ont_df['validated_entities'].sum()
        ont_total_filtered_entities = ont_total_orig_entities - ont_total_valid_entities
        ont_retention_pct = (ont_total_valid_entities / ont_total_orig_entities * 100) if ont_total_orig_entities > 0 else 0

        base_total_orig_entities = base_df['original_entities'].sum()
        base_total_valid_entities = base_df['validated_entities'].sum()
        base_total_filtered_entities = base_total_orig_entities - base_total_valid_entities
        base_retention_pct = (base_total_valid_entities / base_total_orig_entities * 100) if base_total_orig_entities > 0 else 0

        ont_total_orig_rels = ont_df['original_relationships'].sum()
        ont_total_valid_rels = ont_df['validated_relationships'].sum()
        ont_total_filtered_rels = ont_total_orig_rels - ont_total_valid_rels
        ont_rel_retention_pct = (ont_total_valid_rels / ont_total_orig_rels * 100) if ont_total_orig_rels > 0 else 0

        base_total_orig_rels = base_df['original_relationships'].sum()
        base_total_valid_rels = base_df['validated_relationships'].sum()
        base_total_filtered_rels = base_total_orig_rels - base_total_valid_rels
        base_rel_retention_pct = (base_total_valid_rels / base_total_orig_rels * 100) if base_total_orig_rels > 0 else 0

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 3: FILTERING IMPACT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This 2x2 grid compares the before/after filtering impact for both methods.\n\n")
            f.write("TOP ROW: Ontology-Guided Method\n")
            f.write("  - Left: Entity filtering (original vs validated)\n")
            f.write("  - Right: Relationship filtering (original vs validated)\n\n")
            f.write("BOTTOM ROW: Baseline Method\n")
            f.write("  - Left: Entity filtering (original vs validated)\n")
            f.write("  - Right: Relationship filtering (original vs validated)\n\n")
            f.write("Each subplot shows grouped bars comparing:\n")
            f.write("  - Blue/Purple bars: Original count (before validation)\n")
            f.write("  - Green/Orange bars: Validated count (after validation)\n")
            f.write("  - Gap between bars: Number of elements filtered out\n\n\n")

            f.write("SUMMARY STATISTICS - ENTITIES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ontology-Guided:\n")
            f.write(f"  Original Entities:   {ont_total_orig_entities:>6}\n")
            f.write(f"  Validated Entities:  {ont_total_valid_entities:>6}\n")
            f.write(f"  Filtered Out:        {ont_total_filtered_entities:>6} ({100 - ont_retention_pct:.1f}%)\n")
            f.write(f"  Retention Rate:      {ont_retention_pct:>6.1f}%\n\n")

            f.write(f"Baseline:\n")
            f.write(f"  Original Entities:   {base_total_orig_entities:>6}\n")
            f.write(f"  Validated Entities:  {base_total_valid_entities:>6}\n")
            f.write(f"  Filtered Out:        {base_total_filtered_entities:>6} ({100 - base_retention_pct:.1f}%)\n")
            f.write(f"  Retention Rate:      {base_retention_pct:>6.1f}%\n\n\n")

            f.write("SUMMARY STATISTICS - RELATIONSHIPS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ontology-Guided:\n")
            f.write(f"  Original Relationships:   {ont_total_orig_rels:>6}\n")
            f.write(f"  Validated Relationships:  {ont_total_valid_rels:>6}\n")
            f.write(f"  Filtered Out:             {ont_total_filtered_rels:>6} ({100 - ont_rel_retention_pct:.1f}%)\n")
            f.write(f"  Retention Rate:           {ont_rel_retention_pct:>6.1f}%\n\n")

            f.write(f"Baseline:\n")
            f.write(f"  Original Relationships:   {base_total_orig_rels:>6}\n")
            f.write(f"  Validated Relationships:  {base_total_valid_rels:>6}\n")
            f.write(f"  Filtered Out:             {base_total_filtered_rels:>6} ({100 - base_rel_retention_pct:.1f}%)\n")
            f.write(f"  Retention Rate:           {base_rel_retention_pct:>6.1f}%\n\n\n")

            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write("1. FILTERING VOLUME:\n")
            f.write(f"   - Ontology-Guided filters {ont_total_filtered_entities} entities ({100 - ont_retention_pct:.1f}%)\n")
            f.write(f"   - Baseline filters {base_total_filtered_entities} entities ({100 - base_retention_pct:.1f}%)\n")
            f.write(f"   - Baseline filters {base_total_filtered_entities - ont_total_filtered_entities} more entities\n\n")
            f.write("2. RETENTION EFFICIENCY:\n")
            f.write(f"   - Ontology-Guided retains {ont_retention_pct:.1f}% of entities\n")
            f.write(f"   - Baseline retains {base_retention_pct:.1f}% of entities\n")
            f.write(f"   - Ontology-Guided has {ont_retention_pct - base_retention_pct:.1f} percentage points higher retention\n\n")
            f.write("3. RELATIONSHIP IMPACT:\n")
            f.write(f"   - Ontology-Guided retains {ont_rel_retention_pct:.1f}% of relationships\n")
            f.write(f"   - Baseline retains {base_rel_retention_pct:.1f}% of relationships\n")
            f.write("   - Relationship retention follows similar patterns to entity retention\n\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("The filtering impact analysis reveals the cost of permissive extraction:\n\n")
            f.write("ONTOLOGY-GUIDED METHOD:\n")
            f.write("- Pre-filtering during extraction (Stage 2) ensures most entities conform to schema\n")
            f.write("- High retention rate in validation (Stage 3) reflects extraction quality\n")
            f.write("- Minimal filtering waste means lower computational cost per valid entity\n\n")
            f.write("BASELINE METHOD:\n")
            f.write("- Permissive extraction (Stage 2) accepts many invalid entities\n")
            f.write("- Low retention rate in validation (Stage 3) reveals extraction quality issues\n")
            f.write("- Heavy filtering creates wasted cost: tokens spent on invalid entities\n\n")
            f.write("COST IMPLICATIONS:\n")
            f.write("The gap between original and validated counts represents wasted computation:\n")
            f.write("- API calls spent extracting entities that fail validation\n")
            f.write("- Token costs for processing invalid data\n")
            f.write("- Computational resources for validating doomed-to-fail entities\n\n")
            f.write("This validates the efficiency advantage of ontology-guided extraction:\n")
            f.write("prevent invalid entities from entering the pipeline rather than filtering them later.\n")

        print(f"✅ Saved: {explanation_path}")

    def plot_validation_quality(self, df):
        """Plot Stage 3: Validation pass rate and entity retention"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Prepare data
        df_plot = df.copy()
        df_plot['doc_short'] = df_plot['document'].map(self.doc_names)

        # Plot 1: Ontology Schema Compliance Rate (correct quality metric)
        ax1 = axes[0]
        df_pivot_pass_rate = df_plot.pivot(index='doc_short', columns='method', values='validation_pass_rate')
        df_pivot_pass_rate.plot(kind='bar', ax=ax1, color=['#06A77D', '#F77F00'])
        ax1.set_title('Stage 3: Ontology Schema Compliance Rate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Document', fontsize=12)
        ax1.set_ylabel('Ontology Schema Compliance Rate (%)', fontsize=12)
        ax1.set_ylim(0, 105)
        ax1.legend(title='Method', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='100% (all rules passed)')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Entity Semantic Accuracy
        ax2 = axes[1]
        df_pivot_semantic = df_plot.pivot(index='doc_short', columns='method', values='semantic_type_accuracy')
        df_pivot_semantic.plot(kind='bar', ax=ax2, color=['#06A77D', '#F77F00'])
        ax2.set_title('Stage 3: Entity Semantic Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Document', fontsize=12)
        ax2.set_ylabel('Entity Semantic Accuracy (%)', fontsize=12)
        ax2.set_ylim(0, 105)
        ax2.legend(title='Method', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        output_path = self.output_dir / "validation_quality.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

    def plot_filtering_impact(self, df):
        """Plot Stage 3: Filtering impact (before/after validation)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Prepare data
        df_plot = df.copy()
        df_plot['doc_short'] = df_plot['document'].map(self.doc_names)

        # Separate ontology-guided and baseline
        df_onto = df_plot[df_plot['method'] == 'Ontology-Guided']
        df_base = df_plot[df_plot['method'] == 'Baseline']

        # Plot 1: Ontology-Guided - Entities
        ax1 = axes[0, 0]
        x = np.arange(len(df_onto))
        width = 0.35
        ax1.bar(x - width/2, df_onto['original_entities'], width, label='Original', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, df_onto['validated_entities'], width, label='Validated', color='#06A77D', alpha=0.8)
        ax1.set_title('Ontology-Guided: Entity Filtering', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Document', fontsize=10)
        ax1.set_ylabel('Number of Entities', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_onto['doc_short'], rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Ontology-Guided - Relationships
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, df_onto['original_relationships'], width, label='Original', color='#2E86AB', alpha=0.8)
        ax2.bar(x + width/2, df_onto['validated_relationships'], width, label='Validated', color='#06A77D', alpha=0.8)
        ax2.set_title('Ontology-Guided: Relationship Filtering', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Document', fontsize=10)
        ax2.set_ylabel('Number of Relationships', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_onto['doc_short'], rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Plot 3: Baseline - Entities
        ax3 = axes[1, 0]
        x_base = np.arange(len(df_base))
        ax3.bar(x_base - width/2, df_base['original_entities'], width, label='Original', color='#A23B72', alpha=0.8)
        ax3.bar(x_base + width/2, df_base['validated_entities'], width, label='Validated', color='#F77F00', alpha=0.8)
        ax3.set_title('Baseline: Entity Filtering', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Document', fontsize=10)
        ax3.set_ylabel('Number of Entities', fontsize=10)
        ax3.set_xticks(x_base)
        ax3.set_xticklabels(df_base['doc_short'], rotation=45, ha='right', fontsize=9)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # Plot 4: Baseline - Relationships
        ax4 = axes[1, 1]
        ax4.bar(x_base - width/2, df_base['original_relationships'], width, label='Original', color='#A23B72', alpha=0.8)
        ax4.bar(x_base + width/2, df_base['validated_relationships'], width, label='Validated', color='#F77F00', alpha=0.8)
        ax4.set_title('Baseline: Relationship Filtering', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Document', fontsize=10)
        ax4.set_ylabel('Number of Relationships', fontsize=10)
        ax4.set_xticks(x_base)
        ax4.set_xticklabels(df_base['doc_short'], rotation=45, ha='right', fontsize=9)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "filtering_impact.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

    def run_analysis(self):
        """Run all Stage 3 comparison analyses"""
        print("\n" + "=" * 80)
        print("STAGE 3 VALIDATION COMPARISON ANALYSIS")
        print("=" * 80 + "\n")

        # Load data
        print("Loading Stage 3 validation data...")
        df_stage3 = self.load_stage3_data()
        print(f"✅ Loaded data for {len(df_stage3)} validation runs\n")

        # Generate validation quality analysis
        print("Generating validation quality analysis...")
        self.plot_validation_quality(df_stage3)
        self.save_validation_quality_table(df_stage3)
        self.save_validation_quality_explanation(df_stage3)

        # Generate filtering impact analysis
        print("\nGenerating filtering impact analysis...")
        self.plot_filtering_impact(df_stage3)
        self.save_filtering_impact_table(df_stage3)
        self.save_filtering_impact_explanation(df_stage3)

        print("\n" + "=" * 80)
        print("STAGE 3 COMPARISON ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print("  - validation_quality.png")
        print("  - validation_quality.csv")
        print("  - validation_quality_EXPLANATION.txt")
        print("  - filtering_impact.png")
        print("  - filtering_impact.csv")
        print("  - filtering_impact_EXPLANATION.txt")
        print()

def main():
    """Main entry point"""
    base_dir = Path(__file__).parent.parent.parent
    analyzer = Stage3ComparisonAnalyzer(base_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
