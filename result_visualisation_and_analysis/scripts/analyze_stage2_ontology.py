#!/usr/bin/env python3
"""
Stage 2 Ontology-Guided Analysis: Entity and Relationship Type Distributions
Analyzes the internal structure of ontology-guided extraction results
"""

import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class OntologyDistributionAnalyzer:
    """Analyzes entity and relationship type distributions in ontology-guided extraction"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.stage2_dir = self.base_dir / "outputs" / "stage2_ontology_guided_extraction"
        self.output_dir = self.base_dir / "result_visualisation_and_analysis" / "stage2_ontology"

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

    def load_ontology_data(self):
        """Load all ontology-guided extraction files"""
        all_data = []

        for file_path in sorted(self.stage2_dir.glob("*_ontology_guided.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                doc_name = file_path.stem.replace("_ontology_guided", "")

                all_data.append({
                    'document': doc_name,
                    'doc_short': self.doc_names.get(doc_name, doc_name),
                    'entities': data.get('entities', []),
                    'relationships': data.get('relationships', []),
                    'file_path': str(file_path)
                })

        return all_data

    def analyze_entity_types(self, data_list):
        """Analyze entity type distribution across all documents"""
        entity_type_counts = {}

        for item in data_list:
            doc_short = item['doc_short']
            entities = item['entities']

            for entity in entities:
                entity_type = entity.get('type', 'Unknown')

                if doc_short not in entity_type_counts:
                    entity_type_counts[doc_short] = Counter()

                entity_type_counts[doc_short][entity_type] += 1

        return entity_type_counts

    def analyze_relationship_types(self, data_list):
        """Analyze relationship type distribution across all documents"""
        rel_type_counts = {}

        for item in data_list:
            doc_short = item['doc_short']
            relationships = item['relationships']

            for rel in relationships:
                rel_type = rel.get('type', 'Unknown')

                if doc_short not in rel_type_counts:
                    rel_type_counts[doc_short] = Counter()

                rel_type_counts[doc_short][rel_type] += 1

        return rel_type_counts

    def plot_entity_type_distribution(self, entity_type_counts):
        """Plot entity type distribution as stacked bar chart"""
        # Collect all unique entity types
        all_types = set()
        for counts in entity_type_counts.values():
            all_types.update(counts.keys())

        all_types = sorted(list(all_types))
        documents = sorted(entity_type_counts.keys())

        # Create data matrix
        data_matrix = []
        for entity_type in all_types:
            type_counts = [entity_type_counts[doc].get(entity_type, 0) for doc in documents]
            data_matrix.append(type_counts)

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(16, 10))

        x = np.arange(len(documents))
        width = 0.6

        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_types)))

        bottom = np.zeros(len(documents))
        bars = []

        for idx, (entity_type, counts) in enumerate(zip(all_types, data_matrix)):
            bar = ax.bar(x, counts, width, bottom=bottom, label=entity_type,
                        color=colors[idx], alpha=0.8)
            bars.append(bar)
            bottom += np.array(counts)

        ax.set_title('Entity Type Distribution by Document (Ontology-Guided)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Document', fontsize=12)
        ax.set_ylabel('Number of Entities', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(documents, rotation=45, ha='right')
        ax.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "entity_type_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

        # Save explanation
        self._save_entity_distribution_explanation(entity_type_counts, all_types)

        # Save CSV
        self._save_entity_distribution_csv(entity_type_counts, all_types, documents)

    def plot_relationship_type_distribution(self, rel_type_counts):
        """Plot relationship type distribution as stacked bar chart"""
        # Collect all unique relationship types
        all_types = set()
        for counts in rel_type_counts.values():
            all_types.update(counts.keys())

        all_types = sorted(list(all_types))
        documents = sorted(rel_type_counts.keys())

        # Create data matrix
        data_matrix = []
        for rel_type in all_types:
            type_counts = [rel_type_counts[doc].get(rel_type, 0) for doc in documents]
            data_matrix.append(type_counts)

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(16, 10))

        x = np.arange(len(documents))
        width = 0.6

        # Color palette
        colors = plt.cm.tab20b(np.linspace(0, 1, len(all_types)))

        bottom = np.zeros(len(documents))
        bars = []

        for idx, (rel_type, counts) in enumerate(zip(all_types, data_matrix)):
            bar = ax.bar(x, counts, width, bottom=bottom, label=rel_type,
                        color=colors[idx], alpha=0.8)
            bars.append(bar)
            bottom += np.array(counts)

        ax.set_title('Relationship Type Distribution by Document (Ontology-Guided)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Document', fontsize=12)
        ax.set_ylabel('Number of Relationships', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(documents, rotation=45, ha='right')
        ax.legend(title='Relationship Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "relationship_type_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

        # Save explanation
        self._save_relationship_distribution_explanation(rel_type_counts, all_types)

        # Save CSV
        self._save_relationship_distribution_csv(rel_type_counts, all_types, documents)

    def plot_entity_type_heatmap(self, entity_type_counts):
        """Create heatmap showing entity type distribution"""
        # Get all types and documents
        all_types = set()
        for counts in entity_type_counts.values():
            all_types.update(counts.keys())

        all_types = sorted(list(all_types))
        documents = sorted(entity_type_counts.keys())

        # Create matrix
        matrix = []
        for doc in documents:
            row = [entity_type_counts[doc].get(etype, 0) for etype in all_types]
            matrix.append(row)

        matrix = np.array(matrix)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(len(all_types)))
        ax.set_yticks(np.arange(len(documents)))
        ax.set_xticklabels(all_types, rotation=90, ha='right', fontsize=9)
        ax.set_yticklabels(documents, fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=15)

        # Add text annotations
        for i in range(len(documents)):
            for j in range(len(all_types)):
                value = matrix[i, j]
                if value > 0:
                    text = ax.text(j, i, int(value),
                                 ha="center", va="center", color="black" if value < matrix.max()/2 else "white",
                                 fontsize=8)

        ax.set_title('Entity Type Heatmap (Ontology-Guided)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Entity Type', fontsize=12)
        ax.set_ylabel('Document', fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / "entity_type_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

        # Save CSV and explanation
        self._save_heatmap_csv(entity_type_counts, all_types, documents)
        self._save_heatmap_explanation(entity_type_counts, all_types)

    def _save_heatmap_csv(self, entity_type_counts, all_types, documents):
        """Save entity type heatmap data as CSV"""
        # Create DataFrame with documents as rows and entity types as columns
        data = []
        for doc in documents:
            row = {'Document': doc}
            for etype in all_types:
                row[etype] = entity_type_counts[doc].get(etype, 0)
            data.append(row)

        df = pd.DataFrame(data)
        csv_path = self.output_dir / "entity_type_heatmap.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

    def _save_heatmap_explanation(self, entity_type_counts, all_types):
        """Save explanation for entity type heatmap"""
        explanation_path = self.output_dir / "entity_type_heatmap_EXPLANATION.txt"

        with open(explanation_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ENTITY TYPE HEATMAP (ONTOLOGY-GUIDED)\n")
            f.write("=" * 80 + "\n\n")

            f.write("FIGURE DESCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write("This heatmap visualizes the distribution of entity types across all documents.\n")
            f.write("- Rows represent documents\n")
            f.write("- Columns represent entity types\n")
            f.write("- Color intensity indicates count (darker = more entities of that type)\n")
            f.write("- Numbers in cells show exact entity counts\n\n")

            f.write("DATA SOURCE\n")
            f.write("-" * 80 + "\n")
            f.write("Entity data from Stage 2 ontology-guided extraction:\n")
            f.write("  • outputs/stage2_ontology_guided_extraction/*_ontology_guided.json\n\n")

            f.write("ENTITY TYPES ANALYZED\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total entity types found: {len(all_types)}\n\n")

            for etype in sorted(all_types):
                total = sum(counts.get(etype, 0) for counts in entity_type_counts.values())
                pct = (total / sum(sum(counts.values()) for counts in entity_type_counts.values())) * 100
                f.write(f"{etype}:\n")
                f.write(f"  • Total count: {total}\n")
                f.write(f"  • Percentage: {pct:.1f}%\n")
                f.write(f"  • Present in {sum(1 for counts in entity_type_counts.values() if counts.get(etype, 0) > 0)} documents\n\n")

            f.write("KEY INSIGHTS\n")
            f.write("-" * 80 + "\n")
            f.write("• Helps identify which entity types are most common across documents\n")
            f.write("• Shows document-specific patterns (which docs focus on which types)\n")
            f.write("• Reveals ontology schema coverage and balance\n")
            f.write("• Can identify missing or underrepresented entity types\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("• Empty cells (count = 0) = That entity type was not found in that document\n")
            f.write("• Darker cells = Higher concentration of that entity type\n")
            f.write("• Rows with many dark cells = Documents with diverse entity types\n")
            f.write("• Columns with many dark cells = Entity types common across documents\n\n")

        print(f"✅ Saved: {explanation_path}")

    def _save_entity_distribution_explanation(self, entity_type_counts, all_types):
        """Save explanation for entity distribution visualization"""
        explanation_path = self.output_dir / "entity_type_distribution_EXPLANATION.txt"

        with open(explanation_path, 'w') as f:
            f.write("ENTITY TYPE DISTRIBUTION (ONTOLOGY-GUIDED)\n")
            f.write("=" * 80 + "\n\n")
            f.write("This visualization shows the breakdown of entity types extracted from each document\n")
            f.write("using the ontology-guided extraction method.\n\n")

            f.write("CHART TYPE: Stacked Bar Chart\n")
            f.write("- Each bar represents one document\n")
            f.write("- Different colors represent different entity types\n")
            f.write("- Height of each color segment = count of that entity type\n\n")

            f.write(f"ENTITY TYPES FOUND: {len(all_types)}\n")
            for etype in sorted(all_types):
                total = sum(counts.get(etype, 0) for counts in entity_type_counts.values())
                f.write(f"  - {etype}: {total} total across all documents\n")

            f.write("\nKEY INSIGHTS:\n")
            f.write("- Shows which entity types are most common in ESG disclosures\n")
            f.write("- Reveals document-specific focus areas\n")
            f.write("- Demonstrates ontology schema coverage across documents\n")

        print(f"✅ Saved: {explanation_path}")

    def _save_relationship_distribution_explanation(self, rel_type_counts, all_types):
        """Save explanation for relationship distribution visualization"""
        explanation_path = self.output_dir / "relationship_type_distribution_EXPLANATION.txt"

        with open(explanation_path, 'w') as f:
            f.write("RELATIONSHIP TYPE DISTRIBUTION (ONTOLOGY-GUIDED)\n")
            f.write("=" * 80 + "\n\n")
            f.write("This visualization shows the breakdown of relationship types extracted from each document\n")
            f.write("using the ontology-guided extraction method.\n\n")

            f.write("CHART TYPE: Stacked Bar Chart\n")
            f.write("- Each bar represents one document\n")
            f.write("- Different colors represent different relationship types\n")
            f.write("- Height of each color segment = count of that relationship type\n\n")

            f.write(f"RELATIONSHIP TYPES FOUND: {len(all_types)}\n")
            for rtype in sorted(all_types):
                total = sum(counts.get(rtype, 0) for counts in rel_type_counts.values())
                f.write(f"  - {rtype}: {total} total across all documents\n")

            f.write("\nKEY INSIGHTS:\n")
            f.write("- Shows which relationship types connect entities in ESG knowledge graphs\n")
            f.write("- Reveals document-specific relationship patterns\n")
            f.write("- Demonstrates semantic connections captured by ontology\n")

        print(f"✅ Saved: {explanation_path}")

    def _save_entity_distribution_csv(self, entity_type_counts, all_types, documents):
        """Save entity distribution as CSV"""
        # Create DataFrame
        data_dict = {'Document': documents}

        for etype in all_types:
            data_dict[etype] = [entity_type_counts[doc].get(etype, 0) for doc in documents]

        df = pd.DataFrame(data_dict)

        # Add totals row
        totals = {'Document': 'TOTAL'}
        for etype in all_types:
            totals[etype] = df[etype].sum()

        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

        csv_path = self.output_dir / "entity_type_distribution.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

    def _save_relationship_distribution_csv(self, rel_type_counts, all_types, documents):
        """Save relationship distribution as CSV"""
        # Create DataFrame
        data_dict = {'Document': documents}

        for rtype in all_types:
            data_dict[rtype] = [rel_type_counts[doc].get(rtype, 0) for doc in documents]

        df = pd.DataFrame(data_dict)

        # Add totals row
        totals = {'Document': 'TOTAL'}
        for rtype in all_types:
            totals[rtype] = df[rtype].sum()

        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

        csv_path = self.output_dir / "relationship_type_distribution.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")

    def run_analysis(self):
        """Run complete distribution analysis"""
        print("\n" + "=" * 80)
        print("ONTOLOGY-GUIDED DISTRIBUTION ANALYSIS")
        print("=" * 80)
        print("")

        # Load data
        print("Loading ontology-guided extraction data...")
        data_list = self.load_ontology_data()
        print(f"✅ Loaded {len(data_list)} documents")
        print("")

        # Analyze entity types
        print("Analyzing entity type distribution...")
        entity_type_counts = self.analyze_entity_types(data_list)
        total_entity_types = len(set().union(*[set(counts.keys()) for counts in entity_type_counts.values()]))
        print(f"✅ Found {total_entity_types} unique entity types")

        # Analyze relationship types
        print("Analyzing relationship type distribution...")
        rel_type_counts = self.analyze_relationship_types(data_list)
        total_rel_types = len(set().union(*[set(counts.keys()) for counts in rel_type_counts.values()]))
        print(f"✅ Found {total_rel_types} unique relationship types")
        print("")

        # Generate visualizations
        print("Generating visualizations...")
        self.plot_entity_type_distribution(entity_type_counts)
        self.plot_relationship_type_distribution(rel_type_counts)
        self.plot_entity_type_heatmap(entity_type_counts)
        print("")

        print("=" * 80)
        print("✅ DISTRIBUTION ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print("")


def main():
    """Main execution"""
    base_dir = Path(__file__).parent.parent.parent
    analyzer = OntologyDistributionAnalyzer(base_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
