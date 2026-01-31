#!/usr/bin/env python3
"""
Generate SEPARATE knowledge graph visualizations:
1. Ontology-guided results graph (validated entities + relationships)
2. Failed entities graph (failed entities with failure reasons)

Generates 2 files per document (10 total for 5 documents)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import textwrap
import networkx as nx
from collections import defaultdict

# Color scheme for entity types
COLORS = {
    'ReportingFramework': '#4A90E2',  # Blue
    'Industry': '#7B68EE',            # Purple-blue
    'Category': '#50C878',            # Green
    'DirectMetric': '#FFD700',        # Gold
    'CalculatedMetric': '#FFA500',    # Orange
    'InputMetric': '#FFEB3B',         # Yellow
    'Model': '#9C27B0',               # Purple
    'Metric': '#98D8C8',              # Light teal (for generic Metric)
    'Failed': '#E74C3C'               # Red
}

RELATIONSHIP_COLORS = {
    'Include': '#95A5A6',
    'ConsistOf': '#3498DB',
    'RequiresInputFrom': '#E67E22',
    'IsCalculatedBy': '#9B59B6',
    'ReportUsing': '#1ABC9C'
}


def load_json(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_ontology_graph(doc_info, base_dir, output_dir):
    """Generate knowledge graph showing ONLY validated entities and relationships"""
    print(f"\n  → Generating ontology graph for: {doc_info['title']}")

    # Load Stage 3 validated data
    stage3_path = base_dir / doc_info['validated_file']
    if not stage3_path.exists():
        print(f"    ⚠️  Validation file not found: {stage3_path}")
        return

    stage3_data = load_json(stage3_path)

    # Extract validated entities
    entities = stage3_data.get('entities', [])
    relationships = stage3_data.get('relationships', [])

    if not entities:
        print(f"    ⚠️  No entities found")
        return

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    entities_dict = {}
    nodes_by_type = defaultdict(list)

    for entity in entities:
        entity_id = entity.get('id')
        entity_type = entity.get('type')
        label = entity.get('label', 'Unnamed')

        # Determine color and effective type (for InputMetrics)
        effective_type = entity_type
        if entity_type == 'Metric':
            metric_type = entity.get('properties', {}).get('metric_type', 'DirectMetric')
            color = COLORS.get(metric_type, COLORS.get('Metric', '#98D8C8'))
            # Group InputMetrics separately for layout purposes
            if metric_type == 'InputMetric':
                effective_type = 'InputMetric'
        else:
            color = COLORS.get(entity_type, '#CCCCCC')

        G.add_node(entity_id, label=label, type=entity_type, color=color)
        entities_dict[entity_id] = entity
        nodes_by_type[effective_type].append(entity_id)

    # Add edges
    edges_list = []
    for rel in relationships:
        subject_id = rel.get('subject')
        predicate = rel.get('predicate')
        object_id = rel.get('object')

        if subject_id in entities_dict and object_id in entities_dict:
            G.add_edge(subject_id, object_id, label=predicate,
                      color=RELATIONSHIP_COLORS.get(predicate, '#95A5A6'))
            edges_list.append((subject_id, object_id, predicate))

    # Create hierarchical layout
    pos = create_hierarchical_layout(G, nodes_by_type, relationships)

    # Create figure with adjusted size - taller for row-based layout, not as wide
    fig, ax = plt.subplots(figsize=(28, 36))
    ax.set_facecolor('#FAFAFA')  # Light background for better contrast

    # Draw edges with varied styles for clarity - different style per relationship type
    relationship_styles = {
        'ReportUsing': ('solid', 0.25, 3.5),      # Straight, large curve, thick
        'Include': ('dotted', 0.1, 2.5),          # Dotted, small curve, medium
        'ConsistOf': ('solid', 0.15, 3.0),        # Solid, medium curve, thick
        'IsCalculatedBy': ('dashdot', 0.2, 3.0),  # Dash-dot, medium curve, thick
        'RequiresInputFrom': ('dashed', 0.25, 3.5) # Dashed, large curve, thick
    }

    for subject_id, object_id, predicate in edges_list:
        if subject_id in pos and object_id in pos:
            edge_color = RELATIONSHIP_COLORS.get(predicate, '#95A5A6')

            # Get style parameters for this relationship type
            style, curve, width = relationship_styles.get(
                predicate,
                ('solid', 0.15, 2.5)  # default
            )

            # Higher alpha for better visibility
            alpha = 0.9

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(subject_id, object_id)],
                edge_color=edge_color,
                arrows=True,
                arrowsize=35,  # Even larger arrows for better visibility
                arrowstyle='-|>',  # Filled arrow head for clearer endpoint
                width=width,
                alpha=alpha,
                style=style,
                connectionstyle=f'arc3,rad={curve}',  # Varied curve by type
                node_size=8000,  # Tell networkx the node size to avoid arrow overlap
                ax=ax
            )

    # Draw nodes by type with better visibility
    for node_type, node_ids in nodes_by_type.items():
        valid_node_ids = [nid for nid in node_ids if nid in pos]
        if not valid_node_ids:
            continue

        color = COLORS.get(node_type, COLORS.get('Metric', '#98D8C8'))
        # Larger node sizes to ensure all text fits perfectly inside circles
        # Increased to prevent character overflow for longer names
        size = 7500 if 'Metric' in node_type else 8000

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=valid_node_ids,
            node_color=color,
            node_size=size,
            alpha=0.95,
            edgecolors='black',
            linewidths=2.5,
            ax=ax
        )

    # Draw labels with better readability - dark text on light backgrounds, white on dark
    labels = {}
    label_colors = {}

    for node_id in G.nodes():
        if node_id in pos:
            label = G.nodes[node_id]['label']
            # More aggressive wrapping (15 chars) to ensure text fits in circles
            wrapped_label = '\n'.join(textwrap.wrap(label, width=15))
            labels[node_id] = wrapped_label

            # Determine text color based on background brightness
            node_type = G.nodes[node_id]['type']
            effective_type = node_type
            if node_type == 'Metric':
                # Check if it's an InputMetric
                entity = entities_dict.get(node_id)
                if entity:
                    metric_type = entity.get('properties', {}).get('metric_type', 'DirectMetric')
                    if metric_type == 'InputMetric':
                        effective_type = 'InputMetric'

            bg_color = COLORS.get(effective_type, COLORS.get('Metric', '#98D8C8'))

            # Use dark text for light backgrounds, white for dark backgrounds
            # Light backgrounds: Yellow (#FFEB3B), Gold (#FFD700), Light teal (#98D8C8)
            if effective_type in ['InputMetric', 'DirectMetric', 'Metric']:
                label_colors[node_id] = '#1A1A1A'  # Dark text
            else:
                label_colors[node_id] = 'white'  # White text for darker backgrounds

    # Draw labels with color mapping
    for node_id, label_text in labels.items():
        if node_id in pos:
            nx.draw_networkx_labels(
                G, pos,
                {node_id: label_text},
                font_size=9,  # Slightly smaller to fit better
                font_weight='bold',
                font_family='sans-serif',
                font_color=label_colors.get(node_id, 'white'),
                ax=ax
            )

    # Edge labels removed - rely on color-coded legend for relationship types

    # Create improved legend with better formatting
    legend_elements = []

    # Entity types section
    legend_elements.append(mpatches.Patch(color='none', label='━━ Entity Types ━━'))
    for entity_type in sorted(nodes_by_type.keys()):
        color = COLORS.get(entity_type, COLORS.get('Metric', '#98D8C8'))
        count = len(nodes_by_type[entity_type])
        legend_elements.append(mpatches.Patch(color=color, label=f'  {entity_type}: {count}',
                                             edgecolor='black', linewidth=1))

    legend_elements.append(mpatches.Patch(color='none', label=''))

    # Relationships section with line style indicators
    legend_elements.append(mpatches.Patch(color='none', label='━━ Relationships ━━'))

    # Map relationship types to their visual styles for legend
    rel_style_labels = {
        'ReportUsing': '  ReportUsing (━━)',
        'Include': '  Include (····)',
        'ConsistOf': '  ConsistOf (━━)',
        'IsCalculatedBy': '  IsCalculatedBy (━·━)',
        'RequiresInputFrom': '  RequiresInputFrom (- - -)'
    }

    for rel_type, color in RELATIONSHIP_COLORS.items():
        label_text = rel_style_labels.get(rel_type, f'  {rel_type}')
        legend_elements.append(mpatches.Patch(color=color, label=label_text,
                                             edgecolor='black', linewidth=1))

    ax.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(1.01, 1), fontsize=12, framealpha=0.97,
             edgecolor='black', fancybox=True, shadow=True)

    # Add improved statistics box
    validation_meta = stage3_data.get('validation_metadata', {})
    stats_text = f"""╔═══ Document Statistics ═══╗

  Document: {doc_info['title']}

  ✓ Validated Entities: {len(entities)}
  ✓ Relationships: {len(relationships)}
  ✓ Entity Types: {len(nodes_by_type)}

╚════════════════════════════╝"""

    ax.text(0.015, 0.985, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=1', facecolor='#E8F4F8',
                    edgecolor='#2E86AB', linewidth=2.5, alpha=0.95),
           family='monospace', weight='bold')

    ax.set_title(f"Knowledge Graph: {doc_info['title']}\n(Ontology-Guided Extraction - Validated Entities & Relationships)",
                fontsize=20, fontweight='bold', pad=35, color='#1A1A1A')
    ax.axis('off')

    # Auto-adjust plot limits to prevent empty space at top
    if pos:
        y_coords = [y for x, y in pos.values()]
        y_min, y_max = min(y_coords), max(y_coords)
        y_range = y_max - y_min
        # Add 10% padding on each side
        padding = y_range * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

    # Save
    safe_name = doc_info['title'].replace(' ', '_').replace('/', '_')
    output_file = output_dir / f"{safe_name}_ontology_graph.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    ✓ Saved: {output_file.name}")
    print(f"      Entities: {len(entities)}, Relationships: {len(relationships)}")


def generate_failure_graph(doc_info, base_dir, output_dir):
    """Generate separate graph showing ONLY failed entities with reasons"""
    print(f"  → Generating failure graph for: {doc_info['title']}")

    # Load Stage 2 and Stage 3 data
    stage2_path = base_dir / doc_info['stage2_file']
    stage3_path = base_dir / doc_info['validated_file']

    if not stage2_path.exists() or not stage3_path.exists():
        print(f"    ⚠️  Missing files")
        return

    stage2_data = load_json(stage2_path)
    stage3_data = load_json(stage3_path)

    # Get failed entities
    stage2_entity_ids = {e.get('id') for e in stage2_data.get('entities', [])}
    stage3_entity_ids = {e.get('id') for e in stage3_data.get('entities', [])}
    removed_entity_ids = stage2_entity_ids - stage3_entity_ids

    # Get semantic validation mismatches
    validation_meta = stage3_data.get('validation_metadata', {})
    semantic_validation = validation_meta.get('semantic_validation', {})
    mismatches = semantic_validation.get('mismatches', [])

    if not mismatches and not removed_entity_ids:
        print(f"    ℹ️  No validation failures")
        return

    # Create figure with improved grid layout
    num_failures = len(mismatches) + len(removed_entity_ids - {m.get('entity_id') for m in mismatches})
    cols = 2
    rows = max(1, (num_failures + 1) // 2)

    fig, axes = plt.subplots(rows, cols, figsize=(22, 6 * rows))
    fig.patch.set_facecolor('#FAFAFA')

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    idx = 0

    # Stage 2 entities lookup
    stage2_entities = {e.get('id'): e for e in stage2_data.get('entities', [])}

    # Process semantic validation mismatches
    for mismatch in mismatches:
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        entity_id = mismatch.get('entity_id', 'Unknown')
        entity_type = mismatch.get('entity_type', 'Unknown')
        label = mismatch.get('label', 'Unknown')
        reasoning = mismatch.get('llm_reasoning', 'No reasoning provided')

        # Wrap text
        wrapped_label = '\n'.join(textwrap.wrap(label, width=50))
        wrapped_reasoning = '\n'.join(textwrap.wrap(reasoning, width=70))

        # Display with improved formatting
        ax.axis('off')
        ax.set_facecolor('white')

        info_text = f"╔═══ FAILED ENTITY #{idx+1} ═══╗\n\n"
        info_text += f"Entity ID:\n  {entity_id}\n\n"
        info_text += f"Type: {entity_type}\n\n"
        info_text += f"Label:\n  {wrapped_label}\n\n"
        info_text += f"{'━'*65}\n\n"
        info_text += f"Failure Reason:\n{wrapped_reasoning}\n\n"
        info_text += f"╚{'═'*65}╝"

        color = COLORS.get(entity_type, '#CCCCCC')

        ax.text(0.5, 0.5, info_text,
               ha='center', va='center',
               fontsize=10,
               bbox=dict(boxstyle='round,pad=1.8', facecolor=color,
                        alpha=0.25, edgecolor='#C0392B', linewidth=4),
               transform=ax.transAxes,
               family='monospace',
               wrap=True)

        ax.set_title(f"❌ {entity_type} Validation Failure",
                    fontsize=14, fontweight='bold', color='#C0392B',
                    pad=10, bbox=dict(boxstyle='round,pad=0.5',
                                     facecolor='#FFE5E5', edgecolor='#C0392B',
                                     linewidth=2))

        idx += 1

    # Process other removed entities
    processed_ids = {m.get('entity_id') for m in mismatches}
    for removed_id in removed_entity_ids:
        if removed_id not in processed_ids and idx < len(axes_flat):
            ax = axes_flat[idx]
            ax.axis('off')
            ax.set_facecolor('white')

            entity = stage2_entities.get(removed_id, {})
            entity_type = entity.get('type', 'Unknown')
            label = entity.get('label', 'Unknown')

            wrapped_label = '\n'.join(textwrap.wrap(label, width=50))

            info_text = f"╔═══ FAILED ENTITY #{idx+1} ═══╗\n\n"
            info_text += f"Entity ID:\n  {removed_id}\n\n"
            info_text += f"Type: {entity_type}\n\n"
            info_text += f"Label:\n  {wrapped_label}\n\n"
            info_text += f"{'━'*65}\n\n"
            info_text += f"Failure Reason:\nRemoved during validation (CQ rule violation)\n\n"
            info_text += f"╚{'═'*65}╝"

            color = COLORS.get(entity_type, '#CCCCCC')

            ax.text(0.5, 0.5, info_text,
                   ha='center', va='center',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=1.8', facecolor=color,
                            alpha=0.25, edgecolor='#C0392B', linewidth=4),
                   transform=ax.transAxes,
                   family='monospace',
                   wrap=True)

            ax.set_title(f"❌ {entity_type} Validation Failure",
                        fontsize=14, fontweight='bold', color='#C0392B',
                        pad=10, bbox=dict(boxstyle='round,pad=0.5',
                                         facecolor='#FFE5E5', edgecolor='#C0392B',
                                         linewidth=2))

            idx += 1

    # Hide unused subplots
    for i in range(idx, len(axes_flat)):
        axes_flat[i].axis('off')

    # Overall title with better formatting
    original_count = validation_meta.get('original_entity_count', 0)
    validated_count = validation_meta.get('validated_entity_count', 0)
    removed_count = original_count - validated_count
    retention_rate = (validated_count / original_count * 100) if original_count > 0 else 0

    title_text = (f"🔍 Validation Failures Analysis: {doc_info['title']}\n"
                 f"Total Failures: {num_failures} │ "
                 f"Original Entities: {original_count} → Validated: {validated_count} │ "
                 f"Retention: {retention_rate:.1f}%")

    fig.suptitle(title_text,
                fontsize=17, fontweight='bold', color='#C0392B',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE5E5',
                         edgecolor='#C0392B', linewidth=2.5))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    safe_name = doc_info['title'].replace(' ', '_').replace('/', '_')
    output_file = output_dir / f"{safe_name}_failures.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    ✓ Saved: {output_file.name}")
    print(f"      Failed Entities: {num_failures}")


def _place_category_row(row_categories, grouped_metrics, pos, all_categories, y_pos, x_spacing):
    """Helper function to place multiple categories' metrics in the same row"""
    # Calculate total metrics and positions for this row
    all_metrics_in_row = []
    for cat in row_categories:
        if cat in grouped_metrics:
            for metric in grouped_metrics[cat]:
                cat_x = pos[cat][0] if cat in pos else 0
                all_metrics_in_row.append((metric, cat, cat_x))

    # Sort by category x-position to maintain left-to-right order
    all_metrics_in_row.sort(key=lambda x: x[2])

    # Place metrics with proper spacing, centered under their parent categories
    for metric, cat, cat_x in all_metrics_in_row:
        # Get all metrics for this category in this row
        cat_metrics = [m for m, c, _ in all_metrics_in_row if c == cat]
        num_metrics = len(cat_metrics)
        metric_idx = cat_metrics.index(metric)

        # Center metrics under the category
        metric_x_start = cat_x - (num_metrics - 1) * x_spacing / 2
        metric_x = metric_x_start + metric_idx * x_spacing
        pos[metric] = (metric_x, y_pos)


def _place_category_row_filtered(row_categories, grouped_metrics, pos, all_categories, y_pos, x_spacing, placed_metrics):
    """Helper function to place multiple categories' metrics in the same row with buffer between category groups"""
    # Group metrics by category with their parent's x-position
    category_groups = []
    for cat in row_categories:
        if cat in grouped_metrics:
            cat_metrics = [m for m in grouped_metrics[cat] if m not in placed_metrics]
            if cat_metrics:
                cat_x = pos[cat][0] if cat in pos else 0
                category_groups.append((cat, cat_x, cat_metrics))

    # Sort by category x-position
    category_groups.sort(key=lambda x: x[1])

    # Calculate positions with buffer space between different categories
    buffer_between_categories = 4.0  # Extra space between different category groups
    current_x = 0

    # First pass: calculate total width needed
    total_width = 0
    for i, (cat, cat_x, cat_metrics) in enumerate(category_groups):
        num_metrics = len(cat_metrics)
        width = (num_metrics - 1) * x_spacing if num_metrics > 0 else 0
        total_width += width
        if i > 0:  # Add buffer before each category except the first
            total_width += buffer_between_categories

    # Start from the left, centered
    current_x = -total_width / 2

    # Second pass: place metrics
    for i, (cat, cat_x, cat_metrics) in enumerate(category_groups):
        if i > 0:
            current_x += buffer_between_categories  # Add buffer before this category group

        for idx, metric in enumerate(cat_metrics):
            pos[metric] = (current_x + idx * x_spacing, y_pos)
            placed_metrics.add(metric)

        # Move to position after this category's metrics
        current_x += (len(cat_metrics) - 1) * x_spacing + x_spacing


def create_hierarchical_layout(G, nodes_by_type, relationships):
    """Create smart hierarchical layout that minimizes crossing lines"""
    pos = {}

    # Build relationship map to understand connections
    parent_map = {}  # child -> parent
    children_map = {}  # parent -> [children]

    for edge in relationships:
        subject = edge.get('subject', '')
        predicate = edge.get('predicate', '')
        obj = edge.get('object', '')

        # Track Category -> Metric relationships
        if predicate == 'ConsistOf':
            parent_map[obj] = subject
            if subject not in children_map:
                children_map[subject] = []
            children_map[subject].append(obj)

        # Track Model -> InputMetric relationships
        if predicate == 'RequiresInputFrom':
            parent_map[obj] = subject
            if subject not in children_map:
                children_map[subject] = []
            children_map[subject].append(obj)

    y_offset = 0
    y_spacing = 8.5

    # Level 1: ReportingFramework
    frameworks = nodes_by_type.get('ReportingFramework', [])
    if frameworks:
        x_spacing = 15.0  # Large spacing for top-level nodes
        x_start = -(len(frameworks) - 1) * x_spacing / 2
        for idx, node_id in enumerate(frameworks):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing

    # Level 2: Industry
    industries = nodes_by_type.get('Industry', [])
    if industries:
        x_spacing = 16.0  # Extra large spacing - industries can have long labels
        x_start = -(len(industries) - 1) * x_spacing / 2
        for idx, node_id in enumerate(industries):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing

    # Level 3: Categories
    categories = nodes_by_type.get('Category', [])
    if categories:
        x_spacing = 17.0  # Very large spacing - categories often have long labels
        x_start = -(len(categories) - 1) * x_spacing / 2
        for idx, node_id in enumerate(categories):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing

    # Level 4+: Metrics (optimized row packing - combine sparse categories)
    # NOTE: Exclude InputMetrics - they will be placed after Models
    all_metrics = []
    for mtype in ['DirectMetric', 'CalculatedMetric', 'Metric']:
        metrics_of_type = nodes_by_type.get(mtype, [])
        # Filter out InputMetrics (they have their own level after Models)
        all_metrics.extend([m for m in metrics_of_type if m not in nodes_by_type.get('InputMetric', [])])

    if all_metrics:
        # Group metrics by parent category
        grouped_metrics = {}
        orphan_metrics = []

        for metric in all_metrics:
            parent = parent_map.get(metric)
            if parent and parent in categories:
                if parent not in grouped_metrics:
                    grouped_metrics[parent] = []
                grouped_metrics[parent].append(metric)
            else:
                orphan_metrics.append(metric)

        # Separate categories into sparse (≤3 metrics) and dense (>3 metrics)
        # Sparse categories can be packed together to save vertical space
        sparse_categories = []
        dense_categories = []

        for category in categories:
            if category in grouped_metrics:
                num_metrics = len(grouped_metrics[category])
                if num_metrics <= 3:
                    sparse_categories.append(category)
                else:
                    dense_categories.append(category)

        # Track which metrics have been placed to avoid duplicates
        placed_metrics = set()

        # Process dense categories first - each gets its own dedicated row
        for category in dense_categories:
            category_metrics = grouped_metrics[category]
            # Filter out already placed metrics (duplicates in multiple categories)
            category_metrics = [m for m in category_metrics if m not in placed_metrics]

            if not category_metrics:
                continue

            category_x = pos[category][0] if category in pos else 0

            num_metrics = len(category_metrics)
            # Adaptive spacing - increased to prevent overlap with larger node sizes
            if num_metrics >= 5:
                x_spacing_metrics = 16.0  # Extra wide for very dense rows
            elif num_metrics == 4:
                x_spacing_metrics = 15.0  # Wide for dense rows
            elif num_metrics == 3:
                x_spacing_metrics = 14.0  # Medium spacing
            else:
                x_spacing_metrics = 13.0  # Standard spacing for 1-2 metrics

            metric_x_start = category_x - (num_metrics - 1) * x_spacing_metrics / 2

            for idx, metric in enumerate(category_metrics):
                pos[metric] = (metric_x_start + idx * x_spacing_metrics, -y_offset)
                placed_metrics.add(metric)

            y_offset += y_spacing

        # Smart packing for sparse categories - pack multiple in same row if they fit
        # Keep categories in their original order to minimize crossing
        current_row_categories = []
        current_row_total_metrics = 0
        max_metrics_per_row = 6  # Allow more metrics per row with proper spacing
        sparse_spacing = 13.0  # Increased spacing for sparse rows

        for category in sparse_categories:
            category_metrics = grouped_metrics[category]
            # Filter out already placed metrics
            category_metrics_filtered = [m for m in category_metrics if m not in placed_metrics]
            num_metrics = len(category_metrics_filtered)

            if num_metrics == 0:
                continue  # Skip if all metrics already placed

            # Check if adding this category would exceed the row limit
            if current_row_total_metrics + num_metrics > max_metrics_per_row and current_row_categories:
                # Place current row and start new row
                _place_category_row_filtered(current_row_categories, grouped_metrics, pos, categories, -y_offset, sparse_spacing, placed_metrics)
                y_offset += y_spacing
                current_row_categories = []
                current_row_total_metrics = 0

            # Add this category to current row
            current_row_categories.append(category)
            current_row_total_metrics += num_metrics

        # Place any remaining categories in the last row
        if current_row_categories:
            _place_category_row_filtered(current_row_categories, grouped_metrics, pos, categories, -y_offset, sparse_spacing, placed_metrics)
            y_offset += y_spacing

        # Place orphan metrics in a separate row if any exist
        if orphan_metrics:
            x_spacing_orphan = 13.0  # Increased spacing for orphan metrics
            orphan_x_start = -(len(orphan_metrics) - 1) * x_spacing_orphan / 2
            for idx, metric in enumerate(orphan_metrics):
                pos[metric] = (orphan_x_start + idx * x_spacing_orphan, -y_offset)
            y_offset += y_spacing

    # Level 5: Models (all in one row)
    models = nodes_by_type.get('Model', [])
    if models:
        x_spacing = 15.0  # Large spacing - models can have long labels
        x_start = -(len(models) - 1) * x_spacing / 2
        for idx, node_id in enumerate(models):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing

    # Level 6+: InputMetrics (each model gets its own dedicated row below ALL models)
    input_metrics = nodes_by_type.get('InputMetric', [])
    if input_metrics:
        # Group InputMetrics by their parent Model
        grouped_inputs = {}
        orphan_inputs = []

        for input_metric in input_metrics:
            parent = parent_map.get(input_metric)
            if parent and parent in models:
                if parent not in grouped_inputs:
                    grouped_inputs[parent] = []
                grouped_inputs[parent].append(input_metric)
            else:
                orphan_inputs.append(input_metric)

        # Position each model's input metrics in horizontal rows
        # Process in the same order as models to maintain alignment
        for model in models:
            if model in grouped_inputs:
                model_inputs = grouped_inputs[model]
                # Get model position from above
                model_x = pos[model][0] if model in pos else 0

                # Place input metrics in a horizontal row below this model
                num_inputs = len(model_inputs)
                x_spacing_inputs = 13.0  # Increased spacing for input metrics
                input_x_start = model_x - (num_inputs - 1) * x_spacing_inputs / 2

                for idx, input_metric in enumerate(model_inputs):
                    pos[input_metric] = (input_x_start + idx * x_spacing_inputs, -y_offset)

                # Move to next row for next model's inputs
                y_offset += y_spacing

        # Place orphan inputs in a separate row if any exist
        if orphan_inputs:
            x_spacing_orphan = 13.0  # Increased spacing for orphan inputs
            orphan_x_start = -(len(orphan_inputs) - 1) * x_spacing_orphan / 2
            for idx, input_metric in enumerate(orphan_inputs):
                pos[input_metric] = (orphan_x_start + idx * x_spacing_orphan, -y_offset)
            y_offset += y_spacing

    return pos


def main():
    base_dir = Path('.')
    output_dir = Path(__file__).parent

    documents = [
        {
            'title': 'SASB Commercial Banks',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/1. SASB-commercial-banks-standard_en-gb_ontology_guided.json',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1. SASB-commercial-banks-standard_en-gb_validated.json'
        },
        {
            'title': 'SASB Semiconductors',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/1.SASB-semiconductors-standard_en-gb_ontology_guided.json',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1.SASB-semiconductors-standard_en-gb_validated.json'
        },
        {
            'title': 'IFRS S2',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures_ontology_guided.json',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures_validated.json'
        },
        {
            'title': 'Australia AASB S2',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/2.Australia-AASBS2_09-24_ontology_guided.json',
            'validated_file': 'outputs/stage3_ontology_guided_validation/2.Australia-AASBS2_09-24_validated.json'
        },
        {
            'title': 'TCFD Report',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/2.FINAL-2017-TCFD-Report_ontology_guided.json',
            'validated_file': 'outputs/stage3_ontology_guided_validation/2.FINAL-2017-TCFD-Report_validated.json'
        }
    ]

    print("=" * 80)
    print("Knowledge Graph Visualization - Separate Graphs Generator")
    print("=" * 80)
    print(f"Output directory: {output_dir}\n")

    for doc in documents:
        print(f"\nProcessing: {doc['title']}")
        print("-" * 60)

        # Generate ontology graph (validated entities)
        generate_ontology_graph(doc, base_dir, output_dir)

        # Generate failure graph (failed entities with reasons)
        generate_failure_graph(doc, base_dir, output_dir)

    print("\n" + "=" * 80)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Files saved in: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
