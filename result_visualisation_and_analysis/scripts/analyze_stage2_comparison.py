#!/usr/bin/env python3
"""
Stage 2 Extraction Comparison Analysis
Compares baseline vs ontology-guided entity and relationship extraction volumes
"""

import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class Stage2ComparisonAnalyzer:
    """Analyzes Stage 2 extraction comparison between baseline and ontology-guided methods"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.stage2_ontology_dir = self.base_dir / "outputs" / "stage2_ontology_guided_extraction"
        self.stage2_baseline_dir = self.base_dir / "outputs" / "stage2_baseline_llm_extraction"
        self.output_dir = self.base_dir / "result_visualisation_and_analysis" / "stage2_comparison"

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

    def load_stage2_data(self):
        """Load Stage 2 extraction data (ontology-guided and baseline)"""
        stage2_data = []

        # Load ontology-guided files
        for file_path in sorted(self.stage2_ontology_dir.glob("*_ontology_guided.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                doc_name = file_path.stem.replace("_ontology_guided", "")
                stage2_data.append({
                    'document': doc_name,
                    'method': 'Ontology-Guided',
                    'entities': len(data.get('entities', [])),
                    'relationships': len(data.get('relationships', [])),
                    'file_path': str(file_path)
                })

        # Load baseline files
        for file_path in sorted(self.stage2_baseline_dir.glob("*_baseline_llm.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                doc_name = file_path.stem.replace("_baseline_llm", "")

                # Extract from nested structure
                if 'experiments' in data and '2_baseline_llm' in data['experiments']:
                    exp_data = data['experiments']['2_baseline_llm']
                else:
                    exp_data = data

                stage2_data.append({
                    'document': doc_name,
                    'method': 'Baseline',
                    'entities': len(exp_data.get('entities', [])),
                    'relationships': len(exp_data.get('relationships', [])),
                    'file_path': str(file_path)
                })

        return pd.DataFrame(stage2_data)

    def save_extraction_comparison_table(self, df):
        """Save extraction comparison data table as CSV"""
        table_data = df.copy()
        table_data['doc_short'] = table_data['document'].map(self.doc_names)

        # Select and rename columns
        table_data = table_data[['doc_short', 'method', 'entities', 'relationships']].copy()
        table_data = table_data.rename(columns={
            'doc_short': 'Document',
            'method': 'Method',
            'entities': 'Entities Extracted',
            'relationships': 'Relationships Extracted'
        })

        # Sort by document and method
        table_data = table_data.sort_values(['Document', 'Method'])

        csv_path = self.output_dir / "extraction_comparison.csv"
        table_data.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

    def save_extraction_comparison_explanation(self, df):
        """Save explanation for extraction comparison visualization"""
        explanation_path = self.output_dir / "extraction_comparison_EXPLANATION.txt"

        # Calculate statistics
        ont_df = df[df['method'] == 'Ontology-Guided']
        base_df = df[df['method'] == 'Baseline']

        ont_total_entities = ont_df['entities'].sum()
        base_total_entities = base_df['entities'].sum()
        ont_total_rels = ont_df['relationships'].sum()
        base_total_rels = base_df['relationships'].sum()

        ont_avg_entities = ont_df['entities'].mean()
        base_avg_entities = base_df['entities'].mean()
        ont_avg_rels = ont_df['relationships'].mean()
        base_avg_rels = base_df['relationships'].mean()

        entity_ratio = base_total_entities / ont_total_entities if ont_total_entities > 0 else 0
        rel_ratio = base_total_rels / ont_total_rels if ont_total_rels > 0 else 0

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 2: EXTRACTION COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This figure compares the extraction volumes between ontology-guided and\n")
            f.write("baseline methods in Stage 2 (extraction phase).\n\n")
            f.write("LEFT PANEL: Entities Extracted\n")
            f.write("  - Shows the number of entities extracted per document by each method\n")
            f.write("  - Blue bars: Baseline method (permissive extraction)\n")
            f.write("  - Purple bars: Ontology-Guided method (constrained extraction)\n\n")
            f.write("RIGHT PANEL: Relationships Extracted\n")
            f.write("  - Shows the number of relationships extracted per document\n")
            f.write("  - Same color scheme as entities panel\n\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Entities Extracted:\n")
            f.write(f"  Ontology-Guided Total:  {ont_total_entities:>6} entities\n")
            f.write(f"  Baseline Total:         {base_total_entities:>6} entities\n")
            f.write(f"  Ontology-Guided Average:{ont_avg_entities:>7.1f} entities per document\n")
            f.write(f"  Baseline Average:       {base_avg_entities:>7.1f} entities per document\n")
            f.write(f"  Baseline/Ontology Ratio:{entity_ratio:>7.2f}x\n\n")

            f.write(f"Relationships Extracted:\n")
            f.write(f"  Ontology-Guided Total:  {ont_total_rels:>6} relationships\n")
            f.write(f"  Baseline Total:         {base_total_rels:>6} relationships\n")
            f.write(f"  Ontology-Guided Average:{ont_avg_rels:>7.1f} relationships per document\n")
            f.write(f"  Baseline Average:       {base_avg_rels:>7.1f} relationships per document\n")
            f.write(f"  Baseline/Ontology Ratio:{rel_ratio:>7.2f}x\n\n\n")

            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write("1. VOLUME DIFFERENCE:\n")
            f.write(f"   - Baseline extracts ~{entity_ratio:.1f}x more entities than ontology-guided\n")
            f.write(f"   - Baseline extracts ~{rel_ratio:.1f}x more relationships than ontology-guided\n\n")
            f.write("2. METHOD CHARACTERISTICS:\n")
            f.write("   - Baseline: Permissive extraction, accepts more varied entity/relationship types\n")
            f.write("   - Ontology-Guided: Constrained extraction, only accepts ontology-defined types\n\n")
            f.write("3. IMPLICATIONS:\n")
            f.write("   - Higher volume in baseline does NOT mean higher quality\n")
            f.write("   - Stage 3 validation will filter out invalid entities/relationships\n")
            f.write("   - Ontology-guided pre-filtering reduces downstream validation burden\n\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("The baseline method's higher extraction volume reflects its permissive approach,\n")
            f.write("which accepts any entity or relationship type without ontology constraints.\n")
            f.write("However, many of these extracted elements will fail validation in Stage 3 because\n")
            f.write("they violate ontological rules (e.g., invalid entity types, missing required\n")
            f.write("properties, incompatible relationship endpoints).\n\n")
            f.write("The ontology-guided method's lower volume reflects its pre-filtering approach,\n")
            f.write("which only extracts entities and relationships that conform to the ontology schema.\n")
            f.write("This reduces computational cost and improves data quality by preventing invalid\n")
            f.write("extractions from entering the pipeline.\n\n")
            f.write("The true measure of quality comes in Stage 3, where validation quality scores\n")
            f.write("reveal which method produces more valid, usable knowledge graph elements.\n")

        print(f"✅ Saved: {explanation_path}")

    def plot_extraction_comparison(self, df):
        """Plot Stage 2: Entity and Relationship extraction comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Prepare data
        df_plot = df.copy()
        df_plot['doc_short'] = df_plot['document'].map(self.doc_names)

        # Plot 1: Entities Extracted
        ax1 = axes[0]
        df_pivot_entities = df_plot.pivot(index='doc_short', columns='method', values='entities')
        df_pivot_entities.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'])
        ax1.set_title('Stage 2: Entities Extracted', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Document', fontsize=12)
        ax1.set_ylabel('Number of Entities', fontsize=12)
        ax1.legend(title='Method', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Relationships Extracted
        ax2 = axes[1]
        df_pivot_rels = df_plot.pivot(index='doc_short', columns='method', values='relationships')
        df_pivot_rels.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72'])
        ax2.set_title('Stage 2: Relationships Extracted', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Document', fontsize=12)
        ax2.set_ylabel('Number of Relationships', fontsize=12)
        ax2.legend(title='Method', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        output_path = self.output_dir / "extraction_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

    def run_analysis(self):
        """Run all Stage 2 comparison analyses"""
        print("\n" + "=" * 80)
        print("STAGE 2 EXTRACTION COMPARISON ANALYSIS")
        print("=" * 80 + "\n")

        # Load data
        print("Loading Stage 2 extraction data...")
        df_stage2 = self.load_stage2_data()
        print(f"✅ Loaded data for {len(df_stage2)} extraction runs\n")

        # Generate visualizations and tables
        print("Generating extraction comparison analysis...")
        self.plot_extraction_comparison(df_stage2)
        self.save_extraction_comparison_table(df_stage2)
        self.save_extraction_comparison_explanation(df_stage2)

        print("\n" + "=" * 80)
        print("STAGE 2 COMPARISON ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print("  - extraction_comparison.png")
        print("  - extraction_comparison.csv")
        print("  - extraction_comparison_EXPLANATION.txt")
        print()

def main():
    """Main entry point"""
    base_dir = Path(__file__).parent.parent.parent
    analyzer = Stage2ComparisonAnalyzer(base_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
