#!/usr/bin/env python3
"""
Generate ESGMKG Ontology Structure Diagram for Section 3
Shows entity types with properties - verified against extraction prompts
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = Path(__file__).parent.parent / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_ontology_structure_diagram():
    """
    ESGMKG ontology structure - all properties verified against extraction prompts
    Compact layout: property name + description + example on same line
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    colors = {
        'Industry': '#e74c3c',
        'ReportingFramework': '#3498db',
        'Category': '#2ecc71',
        'Metric': '#f39c12',
        'Model': '#9b59b6'
    }

    # Entity definitions verified from ontology_guided_extraction_prompt.txt
    # Format: (property_name, what_llm_extracts, example_value)
    # Box sizes adjusted to fit content perfectly
    entities = [
        {
            'type': 'Industry',
            'pos': (0.3, 6.2),
            'width': 4.2,
            'height': 1.85,
            'properties': [
                ('sector', 'Industry sector from text', 'Financials, Technology'),
                ('country', 'Geographic location if mentioned', 'Australia, N/A'),
                ('standard_reference', 'Industry classification code', 'SICS® FN-CB')
            ]
        },
        {
            'type': 'ReportingFramework',
            'pos': (4.9, 6.2),
            'width': 4.2,
            'height': 2.25,
            'properties': [
                ('name', 'Framework identifier', 'SASB, TCFD, IFRS S2'),
                ('version', 'Release version if stated', '2023-12, 2018-10'),
                ('year', 'Publication year', '2023, 2018'),
                ('publisher', 'Issuing organization', 'IFRS Foundation, SASB')
            ]
        },
        {
            'type': 'Category',
            'pos': (9.5, 6.2),
            'width': 4.2,
            'height': 1.85,
            'properties': [
                ('section_title', 'Topic/disclosure area name', 'Data Security, Emissions'),
                ('section_id', 'Section code if present', 'CB-230a, TC-SI-130a'),
                ('page_range', 'Page location in document', '6, 8-9, 10-12')
            ]
        },
        {
            'type': 'Metric',
            'pos': (0.3, 0.5),
            'width': 6.7,
            'height': 3.8,
            'properties': [
                ('measurement_type', 'Data format', 'Quantitative, Qualitative'),
                ('metric_type', 'Classification by composition', 'CalculatedMetric, DirectMetric, InputMetric'),
                ('unit', 'Measurement unit from text', 'Number, Percentage (%), tonnes CO2-e, N/A'),
                ('code', 'Standard metric reference', 'FN-CB-230a.1, TC-SI-130a.1, N/A'),
                ('description', 'Full metric definition verbatim', '(1) Number of data breaches, (2) percentage...'),
                ('disaggregations', 'Breakdown sub-categories (optional)', "['Scope 1', 'Scope 2', 'Scope 3'] or []")
            ]
        },
        {
            'type': 'Model',
            'pos': (7.3, 0.5),
            'width': 6.7,
            'height': 2.3,
            'properties': [
                ('description', 'Model purpose and calculation logic', 'Composite metric combining data breaches...'),
                ('equation', 'Mathematical formula or function', 'f(NumberBreaches, PercentagePersonal, ...)'),
                ('input_variables', 'List of required input metric names', "['Number of data breaches', 'Percentage...']")
            ]
        }
    ]

    # Draw entity boxes
    entity_boxes = {}
    for ent in entities:
        x, y = ent['pos']
        w, h = ent['width'], ent['height']
        color = colors[ent['type']]

        # Box
        rect = plt.Rectangle((x, y), w, h,
                            facecolor=color, edgecolor='black',
                            linewidth=2, alpha=0.25)
        ax.add_patch(rect)

        # Header
        ax.text(x + w/2, y + h - 0.12, ent['type'],
               ha='center', va='top', fontsize=11, fontweight='bold')

        # Separator
        ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.35, y + h - 0.35],
               'k-', linewidth=1.5)

        # Properties (compact: all on one line)
        prop_y = y + h - 0.52
        line_height = 0.52

        for prop_name, prop_desc, prop_example in ent['properties']:
            # Property name (bold, black)
            ax.text(x + 0.15, prop_y, f'• {prop_name}:',
                   ha='left', va='top', fontsize=7.5, fontweight='bold')

            # Description (regular, gray) - on same line
            desc_text = f'{prop_desc}  |  e.g., {prop_example}'
            ax.text(x + 0.15, prop_y - 0.18, desc_text,
                   ha='left', va='top', fontsize=6.8, color='dimgray')

            prop_y -= line_height

        # Store connection points
        entity_boxes[ent['type']] = {
            'top': (x + w/2, y + h),
            'bottom': (x + w/2, y),
            'left': (x, y + h/2),
            'right': (x + w, y + h/2),
        }

    # Relationships - verified from prompts
    relationships = [
        ('Industry', 'right', 'ReportingFramework', 'left', 'ReportUsing', (4.6, 7.15)),
        ('ReportingFramework', 'right', 'Category', 'left', 'Include', (9.2, 7.35)),
        ('Category', 'bottom', 'Metric', 'top', 'ConsistOf', (10.0, 5.6)),
        ('Metric', 'right', 'Model', 'left', 'IsCalculatedBy', (7.15, 1.9)),
        ('Model', 'left', 'Metric', 'right', 'RequiresInputFrom', (7.15, 2.5))
    ]

    for source, source_pt, target, target_pt, label, label_pos in relationships:
        x1, y1 = entity_boxes[source][source_pt]
        x2, y2 = entity_boxes[target][target_pt]

        # Offset bidirectional arrows
        if label == 'IsCalculatedBy':
            y1 -= 0.15
            y2 -= 0.15
        elif label == 'RequiresInputFrom':
            y1 += 0.15
            y2 += 0.15

        # Arrow
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        # Label
        lx, ly = label_pos
        ax.text(lx, ly, label,
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='black', linewidth=1.2))

    # Common fields note
    note = 'Note: All entities share common fields — id (unique identifier), type (entity type), label (display name), properties (type-specific), provenance (source tracking)'
    ax.text(7, 0.08, note,
           ha='center', va='center', fontsize=7.2, style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                    edgecolor='gray', linewidth=1, alpha=0.8))

    plt.tight_layout()

    # Save
    output_path_png = OUTPUT_DIR / "fig_ontology_structure.png"
    output_path_pdf = OUTPUT_DIR / "fig_ontology_structure.pdf"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: ESGMKG Ontology Structure Diagram")
    print(f"   PNG: {output_path_png}")
    print(f"   PDF: {output_path_pdf}")


if __name__ == '__main__':
    print("=" * 80)
    print("GENERATING ESGMKG ONTOLOGY STRUCTURE DIAGRAM")
    print("=" * 80)
    create_ontology_structure_diagram()
    print("=" * 80)
