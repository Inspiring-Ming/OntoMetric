#!/usr/bin/env python3
"""
Generate ESGMKG Knowledge Graph Structure Figure for Section 3.2
Shows the complete ontology design with entities, properties, and relationships
Verified against: ontology_guided_extraction_prompt.txt and stage3_validation.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 8.5
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = Path(__file__).parent.parent / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_kg_structure_figure():
    """
    Create comprehensive ER-diagram style knowledge graph structure showing:
    - 5 entity types with all properties (verified from extraction prompt)
    - 5 relationship types with cardinality
    - Clear visual hierarchy and bidirectional relationships
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme for entity types (consistent with other figures)
    colors = {
        'Industry': '#ffe6e6',
        'ReportingFramework': '#e6f3ff',
        'Category': '#e6ffe6',
        'Metric': '#fff4e6',
        'Model': '#f3e6ff'
    }

    header_colors = {
        'Industry': '#ffcccc',
        'ReportingFramework': '#cce5ff',
        'Category': '#ccffcc',
        'Metric': '#ffe5cc',
        'Model': '#e5ccff'
    }

    # Entity boxes with ALL properties verified from ontology_guided_extraction_prompt.txt
    # Lines 26-46: Entity property requirements
    entities = [
        {
            'type': 'Industry',
            'pos': (0.5, 7.2),
            'width': 4.2,
            'height': 2.2,
            'properties': [
                ('id', 'PK'),
                ('sector', '"Financials"'),
                ('country', '"Australia"'),
                ('standard_reference', '"SICS® FN-CB"')
            ]
        },
        {
            'type': 'ReportingFramework',
            'pos': (5.3, 7.2),
            'width': 4.8,
            'height': 2.5,
            'properties': [
                ('id', 'PK'),
                ('name', '"SASB"'),
                ('version', '"2018-10"'),
                ('year', '"2018"'),
                ('publisher', '"SASB Foundation"')
            ]
        },
        {
            'type': 'Category',
            'pos': (10.7, 7.2),
            'width': 4.8,
            'height': 2.2,
            'properties': [
                ('id', 'PK'),
                ('section_title', '"Data Security"'),
                ('section_id', '"CB-230a"'),
                ('page_range', '"8-9"')
            ]
        },
        {
            'type': 'Metric',
            'pos': (0.5, 0.5),
            'width': 7.0,
            'height': 4.8,
            'properties': [
                ('id', 'PK'),
                ('measurement_type', '"Quantitative"'),
                ('metric_type', '"CalculatedMetric"'),
                ('unit', '"Number / Percentage"'),
                ('code', '"FN-CB-230a.1"'),
                ('description', '"(1) Number of..."'),
                ('disaggregations', '["Scope 1", ...]')
            ]
        },
        {
            'type': 'Model',
            'pos': (8.5, 0.5),
            'width': 7.0,
            'height': 4.8,
            'properties': [
                ('id', 'PK'),
                ('description', '"Composite metric..."'),
                ('equation', '"f(Breaches, ...)"'),
                ('input_variables', '["Number of...", ...]')
            ]
        }
    ]

    # Draw entity boxes (ER-diagram style)
    entity_centers = {}
    for ent in entities:
        x, y = ent['pos']
        w, h = ent['width'], ent['height']

        # Main box
        main_box = plt.Rectangle((x, y), w, h,
                                 facecolor=colors[ent['type']],
                                 edgecolor='#333',
                                 linewidth=2)
        ax.add_patch(main_box)

        # Header box
        header_height = 0.45
        header_box = plt.Rectangle((x, y + h - header_height), w, header_height,
                                   facecolor=header_colors[ent['type']],
                                   edgecolor='#333',
                                   linewidth=2)
        ax.add_patch(header_box)

        # Entity type name
        ax.text(x + w/2, y + h - header_height/2, ent['type'],
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='black')

        # Properties
        prop_y = y + h - header_height - 0.25
        for prop_name, prop_value in ent['properties']:
            # Property name
            if prop_value == 'PK':
                # Primary key - underlined
                ax.text(x + 0.15, prop_y, f'• {prop_name}',
                       ha='left', va='top', fontsize=8.5,
                       fontweight='bold', style='italic',
                       color='#2c3e50')
                # Underline
                ax.plot([x + 0.15, x + 0.15 + len(prop_name)*0.08],
                       [prop_y - 0.05, prop_y - 0.05],
                       'k-', linewidth=0.8)
                ax.text(x + 0.15 + len(prop_name)*0.08 + 0.1, prop_y,
                       '(PK)',
                       ha='left', va='top', fontsize=7,
                       color='#7f8c8d', style='italic')
            else:
                # Regular property
                ax.text(x + 0.15, prop_y, f'• {prop_name}:',
                       ha='left', va='top', fontsize=8.5,
                       fontweight='bold',
                       color='#2c3e50')
                ax.text(x + 0.15, prop_y - 0.18, f'  {prop_value}',
                       ha='left', va='top', fontsize=7.5,
                       color='#555', family='monospace')
                prop_y -= 0.18

            prop_y -= 0.50

        # Store connection points
        entity_centers[ent['type']] = {
            'center': (x + w/2, y + h/2),
            'top': (x + w/2, y + h),
            'bottom': (x + w/2, y),
            'left': (x, y + h/2),
            'right': (x + w, y + h/2)
        }

    # Relationships with cardinality (verified from ontology_guided_extraction_prompt.txt)
    # Lines 50-55: Cardinality rules
    relationships = [
        {
            'from': 'Industry',
            'from_point': 'right',
            'to': 'ReportingFramework',
            'to_point': 'left',
            'label': 'ReportUsing',
            'label_pos': (4.9, 8.5),
            'from_cardinality': '1',
            'to_cardinality': '1',
            'color': '#34495e'
        },
        {
            'from': 'ReportingFramework',
            'from_point': 'right',
            'to': 'Category',
            'to_point': 'left',
            'label': 'Include',
            'label_pos': (10.3, 8.5),
            'from_cardinality': '1',
            'to_cardinality': 'N',
            'color': '#34495e'
        },
        {
            'from': 'Category',
            'from_point': 'bottom',
            'to': 'Metric',
            'to_point': 'top',
            'label': 'ConsistOf',
            'label_pos': (11.5, 6.3),
            'from_cardinality': '1',
            'to_cardinality': 'N',
            'color': '#34495e'
        },
        {
            'from': 'Metric',
            'from_point': 'right',
            'to': 'Model',
            'to_point': 'left',
            'label': 'IsCalculatedBy',
            'label_pos': (7.8, 2.2),
            'from_cardinality': '1',
            'to_cardinality': '0..1',
            'color': '#c0392b',
            'offset': -0.25
        },
        {
            'from': 'Model',
            'from_point': 'left',
            'to': 'Metric',
            'to_point': 'right',
            'label': 'RequiresInputFrom',
            'label_pos': (7.8, 3.5),
            'from_cardinality': '1',
            'to_cardinality': '1..N',
            'color': '#8e44ad',
            'offset': 0.25
        }
    ]

    # Draw relationships with cardinality
    for rel in relationships:
        x1, y1 = entity_centers[rel['from']][rel['from_point']]
        x2, y2 = entity_centers[rel['to']][rel['to_point']]

        # Apply offset for bidirectional arrows
        offset = rel.get('offset', 0)
        if offset != 0:
            if rel['from_point'] in ['left', 'right']:
                y1 += offset
                y2 += offset
            else:
                x1 += offset
                x2 += offset

        # Draw arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.35,head_length=0.45',
            color=rel['color'],
            linewidth=2.2,
            zorder=1
        )
        ax.add_patch(arrow)

        # Relationship label (diamond style background)
        lx, ly = rel['label_pos']
        ax.text(lx, ly, rel['label'],
               ha='center', va='center', fontsize=9.5, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor=rel['color'], linewidth=2),
               zorder=2)

        # Cardinality labels
        # Source cardinality
        mid_x1 = x1 + (x2 - x1) * 0.15
        mid_y1 = y1 + (y2 - y1) * 0.15
        ax.text(mid_x1, mid_y1 + 0.15, rel['from_cardinality'],
               ha='center', va='bottom', fontsize=8, style='italic',
               color=rel['color'], fontweight='bold')

        # Target cardinality
        mid_x2 = x1 + (x2 - x1) * 0.85
        mid_y2 = y1 + (y2 - y1) * 0.85
        ax.text(mid_x2, mid_y2 + 0.15, rel['to_cardinality'],
               ha='center', va='bottom', fontsize=8, style='italic',
               color=rel['color'], fontweight='bold')

    # Title
    ax.text(8, 9.7, 'ESGMKG Knowledge Graph Structure',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Legend box
    legend_x, legend_y = 11.8, 6.0
    legend_w, legend_h = 3.7, 1.8
    legend_box = plt.Rectangle((legend_x, legend_y), legend_w, legend_h,
                               facecolor='#f9f9f9', edgecolor='#666',
                               linewidth=1.5)
    ax.add_patch(legend_box)

    ax.text(legend_x + legend_w/2, legend_y + legend_h - 0.25, 'Cardinality',
           ha='center', va='top', fontsize=10, fontweight='bold')

    legend_items = [
        '1 = Exactly one',
        'N = One or more',
        '0..1 = Zero or one',
        '1..N = One or more'
    ]
    item_y = legend_y + legend_h - 0.6
    for item in legend_items:
        ax.text(legend_x + 0.2, item_y, f'• {item}',
               ha='left', va='top', fontsize=8)
        item_y -= 0.35

    # Footer note
    footer_text = (
        'All entities share common fields: id (unique identifier), type (entity type), '
        'label (display name), properties (type-specific), source (provenance)'
    )
    ax.text(8, 0.15, footer_text,
           ha='center', va='center', fontsize=7.5, style='italic',
           color='#666',
           bbox=dict(boxstyle='round,pad=0.35', facecolor='#ecf0f1',
                    edgecolor='#95a5a6', linewidth=1.2, alpha=0.9))

    plt.tight_layout()

    # Save
    output_path_png = OUTPUT_DIR / "fig_kg_structure.png"
    output_path_pdf = OUTPUT_DIR / "fig_kg_structure.pdf"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: ESGMKG Knowledge Graph Structure Figure")
    print(f"   PNG: {output_path_png}")
    print(f"   PDF: {output_path_pdf}")


if __name__ == '__main__':
    print("=" * 80)
    print("GENERATING ESGMKG KNOWLEDGE GRAPH STRUCTURE FIGURE")
    print("=" * 80)
    create_kg_structure_figure()
    print("=" * 80)
