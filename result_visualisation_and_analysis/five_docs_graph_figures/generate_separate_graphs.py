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
    'InputMetric': '#FFD700',         # Gold (same as DirectMetric - InputMetrics are direct measurements used as model inputs)
    'Model': '#9C27B0',               # Purple
    'Metric': '#98D8C8',              # Light teal (for generic Metric)
    'Failed': '#E74C3C'               # Red
}

RELATIONSHIP_COLORS = {
    'Include': '#95A5A6',          # Gray
    'ConsistOf': '#2C7BBF',        # Darker blue for better visibility (was #3498DB)
    'RequiresInputFrom': '#D35400',# Darker orange for better visibility (was #E67E22)
    'IsCalculatedBy': '#8E44AD',   # Darker purple for better visibility (was #9B59B6)
    'ReportUsing': '#16A085'       # Slightly darker teal (was #1ABC9C)
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

    # ========================================
    # MERGE DUPLICATE METRICS
    # ========================================
    # Find metrics with same label (case-insensitive) and merge InputMetrics into DirectMetrics
    # This removes duplicate nodes where the same metric appears as both DirectMetric and InputMetric

    # Build label -> entity mapping for metrics
    metrics_by_label = {}
    for entity in entities:
        if entity.get('type') == 'Metric':
            label_normalized = entity.get('label', '').lower().strip()
            if label_normalized not in metrics_by_label:
                metrics_by_label[label_normalized] = []
            metrics_by_label[label_normalized].append(entity)

    # Build merge map: InputMetric ID -> DirectMetric ID (for redirecting relationships)
    merge_map = {}  # old_id -> new_id
    entities_to_remove = set()

    for label, metric_group in metrics_by_label.items():
        if len(metric_group) > 1:
            # Find DirectMetric (preferred) or CalculatedMetric to keep
            direct_metrics = [m for m in metric_group if m.get('properties', {}).get('metric_type') == 'DirectMetric']
            calculated_metrics = [m for m in metric_group if m.get('properties', {}).get('metric_type') == 'CalculatedMetric']
            input_metrics = [m for m in metric_group if m.get('properties', {}).get('metric_type') == 'InputMetric']

            # Determine the primary metric to keep (prefer DirectMetric with Category connection)
            primary = None
            if direct_metrics:
                # Prefer DirectMetric that has ConsistOf relationship (connected to Category)
                for dm in direct_metrics:
                    dm_id = dm.get('id')
                    has_category = any(r.get('object') == dm_id and r.get('predicate') == 'ConsistOf' for r in relationships)
                    if has_category:
                        primary = dm
                        break
                if not primary:
                    primary = direct_metrics[0]
            elif calculated_metrics:
                primary = calculated_metrics[0]
            elif input_metrics:
                primary = input_metrics[0]

            if primary:
                primary_id = primary.get('id')
                # Mark other metrics with same label for removal and redirect their relationships
                for m in metric_group:
                    m_id = m.get('id')
                    if m_id != primary_id:
                        merge_map[m_id] = primary_id
                        entities_to_remove.add(m_id)

    # Filter out merged entities
    entities = [e for e in entities if e.get('id') not in entities_to_remove]

    # Redirect relationships to merged entities
    updated_relationships = []
    for rel in relationships:
        subject_id = rel.get('subject')
        object_id = rel.get('object')

        # Redirect if subject or object was merged
        new_subject = merge_map.get(subject_id, subject_id)
        new_object = merge_map.get(object_id, object_id)

        # Skip if this creates a self-loop or duplicate
        if new_subject != new_object:
            updated_rel = rel.copy()
            updated_rel['subject'] = new_subject
            updated_rel['object'] = new_object
            updated_relationships.append(updated_rel)

    # Remove duplicate relationships after merging
    seen_rels = set()
    relationships = []
    for rel in updated_relationships:
        rel_key = (rel.get('subject'), rel.get('predicate'), rel.get('object'))
        if rel_key not in seen_rels:
            seen_rels.add(rel_key)
            relationships.append(rel)

    if merge_map:
        print(f"    ℹ️  Merged {len(merge_map)} duplicate metrics")

    # ========================================
    # BUILD GRAPH
    # ========================================

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    entities_dict = {}
    nodes_by_type = defaultdict(list)

    for entity in entities:
        entity_id = entity.get('id')
        entity_type = entity.get('type')
        label = entity.get('label', 'Unnamed')

        # Determine color and effective type (separate DirectMetric, CalculatedMetric, InputMetric)
        effective_type = entity_type
        if entity_type == 'Metric':
            metric_type = entity.get('properties', {}).get('metric_type', 'DirectMetric')
            color = COLORS.get(metric_type, COLORS.get('Metric', '#98D8C8'))
            # Group all metric types separately for layout and legend purposes
            if metric_type in ['DirectMetric', 'CalculatedMetric', 'InputMetric']:
                effective_type = metric_type
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
        'ReportUsing': ('solid', 0.13, 2.5),      # Solid, further reduced curve for cleaner appearance
        'Include': ('dotted', 0.06, 2.0),         # Dotted, further reduced curve
        'ConsistOf': ('solid', 0.09, 2.5),        # Solid, further reduced curve
        'IsCalculatedBy': ('dashdot', 0.11, 2.5), # Dash-dot, further reduced curve
        'RequiresInputFrom': ('dashed', 0.13, 2.5) # Dashed, further reduced curve
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
            alpha = 0.95  # Increased for clearer lines

            # Determine appropriate node size for margin calculation
            # Use actual node sizes for proper arrow positioning
            source_type = G.nodes[subject_id]['type']
            target_type = G.nodes[object_id]['type']

            # Get actual node sizes (matching the sizes used in drawing)
            if source_type == 'Category':
                source_size = 16000
            elif 'Metric' in source_type:
                source_size = 13500
            elif source_type == 'Model':
                source_size = 14000
            else:
                source_size = 13000

            if target_type == 'Category':
                target_size = 16000
            elif 'Metric' in target_type:
                target_size = 13500
            elif target_type == 'Model':
                target_size = 14000
            else:
                target_size = 13000

            # Use larger of the two sizes for consistent margins
            node_size_for_margin = max(source_size, target_size)

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(subject_id, object_id)],
                edge_color=edge_color,
                arrows=True,
                arrowsize=40,  # Larger arrows for better visibility of endpoints
                arrowstyle='-|>',  # Filled arrow head for clearer endpoint
                width=width,
                alpha=alpha,
                style=style,
                connectionstyle=f'arc3,rad={curve}',  # Varied curve by type
                node_size=node_size_for_margin,  # Use actual node size for proper margins
                min_source_margin=18,  # Increased margin from source node edge
                min_target_margin=18,  # Increased margin to target node edge
                ax=ax
            )

    # Draw nodes by type with better visibility
    for node_type, node_ids in nodes_by_type.items():
        valid_node_ids = [nid for nid in node_ids if nid in pos]
        if not valid_node_ids:
            continue

        color = COLORS.get(node_type, COLORS.get('Metric', '#98D8C8'))
        # Optimized node sizes based on actual text analysis
        # Categories have longest names (up to 87 chars, 4-6 wrapped lines)
        # Metrics also long (up to 77 chars, 4 wrapped lines)
        if node_type == 'Category':
            size = 16000  # Extra large for categories (longest names, most lines)
        elif 'Metric' in node_type:
            size = 13500  # Very large for metrics (also have long names)
        elif node_type == 'Model':
            size = 14000  # Large for models (up to 57 chars)
        else:
            size = 13000  # Large for frameworks, industries

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
            # Optimized wrapping: 15 chars/line = 1-2 words per line
            # Creates vertical stack that fits circles better with LARGER font
            # 87 char label: 15 chars → 8 lines of short text (very readable!)
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
                font_size=11.5,  # LARGER font! 15 chars/line allows bigger, more readable text
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

    # Custom legend labels for entity types
    legend_labels = {
        'CalculatedMetric': 'Metric(Calculated)'
    }

    # Combine DirectMetric and InputMetric counts (same color)
    direct_count = len(nodes_by_type.get('DirectMetric', []))
    input_count = len(nodes_by_type.get('InputMetric', []))
    combined_direct_count = direct_count + input_count

    for entity_type in sorted(nodes_by_type.keys()):
        # Skip InputMetric - it's combined with DirectMetric
        if entity_type == 'InputMetric':
            continue

        color = COLORS.get(entity_type, COLORS.get('Metric', '#98D8C8'))

        # Use combined count for DirectMetric
        if entity_type == 'DirectMetric':
            count = combined_direct_count
            display_label = 'Metric(Direct)'
        else:
            count = len(nodes_by_type[entity_type])
            display_label = legend_labels.get(entity_type, entity_type)

        legend_elements.append(mpatches.Patch(color=color, label=f'  {display_label}: {count}',
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

    # Legend positioned at upper left, moved further left to minimize extra width
    ax.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(0.90, 1), fontsize=14, framealpha=0.97,
             edgecolor='black', fancybox=True, shadow=True)

    # Add improved statistics box
    validation_meta = stage3_data.get('validation_metadata', {})
    stats_text = f"""╔═══ Document Statistics ═══╗

  Document: {doc_info['title']}

  ✓ Validated Entities: {len(entities)}
  ✓ Relationships: {len(relationships)}
  ✓ Entity Types: {len(nodes_by_type)}

╚════════════════════════════╝"""

    ax.text(0.015, 0.985, stats_text, transform=ax.transAxes, fontsize=13,
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


def _calculate_adaptive_spacing(num_entities, min_spacing=20.0, max_spacing=30.0):
    """
    Calculate adaptive horizontal spacing based on number of entities.
    Fewer entities = MORE generous space (up to max_spacing)
    More entities = tighter but still adequate spacing (down to min_spacing)

    Philosophy: Sparse rows get extra space, dense rows get minimum safe spacing
    Significantly increased spacing for very large node sizes (Categories: 16000, Metrics: 13500)
    """
    if num_entities <= 1:
        return max_spacing + 5.0  # Extra generous for single entity
    elif num_entities == 2:
        return max_spacing  # Very generous for pairs
    elif num_entities == 3:
        return max_spacing - 2.0  # Still generous
    elif num_entities == 4:
        return max_spacing - 4.0  # Comfortable
    elif num_entities == 5:
        return max_spacing - 6.0  # Moderate
    elif num_entities == 6:
        return max_spacing - 8.0  # Getting tighter
    elif num_entities >= 7:
        return min_spacing  # Minimum safe spacing for dense rows
    return min_spacing


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
    buffer_between_categories = 6.0  # Increased extra space between different category groups for clearer connections
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


def _place_input_metric_row(row_models, grouped_inputs, pos, y_pos, x_spacing):
    """Helper function to place multiple models' input metrics in the same row with buffer between model groups"""
    # Group input metrics by model with their parent's x-position
    model_groups = []
    for model in row_models:
        if model in grouped_inputs:
            model_inputs = grouped_inputs[model]
            if model_inputs:
                model_x = pos[model][0] if model in pos else 0
                model_groups.append((model, model_x, model_inputs))

    # Sort by model x-position
    model_groups.sort(key=lambda x: x[1])

    # Calculate positions with buffer space between different model groups
    buffer_between_models = 6.0  # Extra space between different model input groups for clearer connections
    current_x = 0

    # First pass: calculate total width needed
    total_width = 0
    for i, (model, model_x, model_inputs) in enumerate(model_groups):
        num_inputs = len(model_inputs)
        width = (num_inputs - 1) * x_spacing if num_inputs > 0 else 0
        total_width += width
        if i > 0:  # Add buffer before each model group except the first
            total_width += buffer_between_models

    # Start from the left, centered
    current_x = -total_width / 2

    # Second pass: place input metrics
    for i, (model, model_x, model_inputs) in enumerate(model_groups):
        if i > 0:
            current_x += buffer_between_models  # Add buffer before this model group

        for idx, input_metric in enumerate(model_inputs):
            pos[input_metric] = (current_x + idx * x_spacing, y_pos)

        # Move to position after this model's input metrics
        current_x += (len(model_inputs) - 1) * x_spacing + x_spacing


def _optimize_entity_order(entities, children_map):
    """
    Optimize left-to-right ordering of entities to minimize connection line crossing.
    Uses a greedy algorithm that groups entities with similar connection patterns together.

    Strategy:
    1. Sort by number of children (entities with more children get more space)
    2. Group entities with no children at the edges
    3. Place entities with children in the middle for better vertical alignment
    """
    if not entities:
        return entities

    # Calculate metrics for each entity
    entity_metrics = []
    for entity in entities:
        num_children = len(children_map.get(entity, []))
        entity_metrics.append((entity, num_children))

    # Sort strategy:
    # - Entities with children in the middle (better for vertical alignment)
    # - Entities without children at edges (less likely to cause crossing)
    # - Among entities with children, sort by number (more children = more central)

    with_children = [(e, n) for e, n in entity_metrics if n > 0]
    without_children = [(e, n) for e, n in entity_metrics if n == 0]

    # Sort entities with children by count (descending - more children in center)
    with_children.sort(key=lambda x: x[1], reverse=True)

    # Distribute: some without children on left, with children in middle, rest on right
    left_orphans = without_children[:len(without_children)//2]
    right_orphans = without_children[len(without_children)//2:]

    # Build final order
    ordered = []
    ordered.extend([e for e, _ in left_orphans])
    ordered.extend([e for e, _ in with_children])
    ordered.extend([e for e, _ in right_orphans])

    return ordered


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
    y_spacing = 20.0  # Base vertical spacing between rows (further increased for clearer connection lines)
    y_spacing_between_types = 26.0  # Extra spacing between different entity types (very clear separation)

    # Level 1: ReportingFramework
    frameworks = nodes_by_type.get('ReportingFramework', [])
    if frameworks:
        x_spacing = _calculate_adaptive_spacing(len(frameworks), min_spacing=24.0, max_spacing=34.0)
        x_start = -(len(frameworks) - 1) * x_spacing / 2
        for idx, node_id in enumerate(frameworks):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing_between_types  # Extra space before next type

    # Level 2: Industry
    industries = nodes_by_type.get('Industry', [])
    if industries:
        x_spacing = _calculate_adaptive_spacing(len(industries), min_spacing=25.0, max_spacing=35.0)
        x_start = -(len(industries) - 1) * x_spacing / 2
        for idx, node_id in enumerate(industries):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing_between_types  # Extra space before next type

    # Level 3: Categories (need maximum spacing - largest nodes at 16000)
    # Optimize ordering to minimize crossing lines
    categories = nodes_by_type.get('Category', [])
    if categories:
        # Optimize category order to minimize metric connection crossings
        categories = _optimize_entity_order(categories, children_map)

        x_spacing = _calculate_adaptive_spacing(len(categories), min_spacing=32.0, max_spacing=42.0)
        x_start = -(len(categories) - 1) * x_spacing / 2
        for idx, node_id in enumerate(categories):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing_between_types  # Extra space before metrics

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
            # Adaptive spacing - generous for large metric nodes (13500)
            # Dense rows need safe distance with clear connection lines, sparse rows get extra space
            if num_metrics >= 7:
                x_spacing_metrics = 26.0  # Further increased for very dense rows (clearer connection lines)
            elif num_metrics == 6:
                x_spacing_metrics = 27.0  # Further increased for dense rows
            elif num_metrics == 5:
                x_spacing_metrics = 28.0  # Further increased comfortable spacing
            elif num_metrics == 4:
                x_spacing_metrics = 30.0  # Further increased generous spacing
            elif num_metrics == 3:
                x_spacing_metrics = 32.0  # Further increased very generous spacing
            else:
                x_spacing_metrics = 34.0  # Further increased maximum space for 1-2 metrics

            metric_x_start = category_x - (num_metrics - 1) * x_spacing_metrics / 2

            for idx, metric in enumerate(category_metrics):
                pos[metric] = (metric_x_start + idx * x_spacing_metrics, -y_offset)
                placed_metrics.add(metric)

            y_offset += y_spacing

        # Smart packing for sparse categories - pack multiple in same row if they fit
        # Keep categories in their original order to minimize crossing
        current_row_categories = []
        current_row_total_metrics = 0
        max_metrics_per_row = 10  # Allow up to 10 metrics per row with proper spacing
        sparse_spacing = 28.0  # Further increased spacing for sparse rows (large metric nodes 13500)

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
            x_spacing_orphan = _calculate_adaptive_spacing(len(orphan_metrics), min_spacing=23.0, max_spacing=32.0)
            orphan_x_start = -(len(orphan_metrics) - 1) * x_spacing_orphan / 2
            for idx, metric in enumerate(orphan_metrics):
                pos[metric] = (orphan_x_start + idx * x_spacing_orphan, -y_offset)
            y_offset += y_spacing

    # Level 5: Models (all in one row - node size 14000)
    # Optimize ordering to minimize crossing lines
    models = nodes_by_type.get('Model', [])
    if models:
        # Optimize model order to minimize InputMetric connection crossings
        models = _optimize_entity_order(models, children_map)

        x_spacing = _calculate_adaptive_spacing(len(models), min_spacing=30.0, max_spacing=40.0)
        x_start = -(len(models) - 1) * x_spacing / 2
        for idx, node_id in enumerate(models):
            pos[node_id] = (x_start + idx * x_spacing, -y_offset)
        y_offset += y_spacing_between_types  # Extra space before InputMetrics

    # Level 6+: InputMetrics (smart packing - pack sparse rows together)
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

        # Separate models into sparse (≤3 input metrics) and dense (>3 input metrics)
        sparse_models = []
        dense_models = []

        for model in models:
            if model in grouped_inputs:
                num_inputs = len(grouped_inputs[model])
                if num_inputs <= 3:
                    sparse_models.append(model)
                else:
                    dense_models.append(model)

        # Process dense models first - each gets its own dedicated row
        for model in dense_models:
            model_inputs = grouped_inputs[model]
            model_x = pos[model][0] if model in pos else 0

            num_inputs = len(model_inputs)
            # Adaptive horizontal spacing for InputMetrics
            if num_inputs >= 7:
                x_spacing_inputs = 26.0
            elif num_inputs == 6:
                x_spacing_inputs = 27.0
            elif num_inputs == 5:
                x_spacing_inputs = 28.0
            elif num_inputs == 4:
                x_spacing_inputs = 30.0
            else:
                x_spacing_inputs = 32.0

            input_x_start = model_x - (num_inputs - 1) * x_spacing_inputs / 2

            for idx, input_metric in enumerate(model_inputs):
                pos[input_metric] = (input_x_start + idx * x_spacing_inputs, -y_offset)

            y_offset += y_spacing

        # Smart packing for sparse models - pack multiple in same row with clear separation
        current_row_models = []
        current_row_total_inputs = 0
        max_inputs_per_row = 10  # Maximum inputs per row (up to 10 with proper spacing)

        for model in sparse_models:
            model_inputs = grouped_inputs[model]
            num_inputs = len(model_inputs)

            # Check if adding this model's inputs would exceed the row limit
            if current_row_total_inputs + num_inputs > max_inputs_per_row and current_row_models:
                # Place current row and start new row
                _place_input_metric_row(current_row_models, grouped_inputs, pos, -y_offset, 28.0)
                y_offset += y_spacing
                current_row_models = []
                current_row_total_inputs = 0

            # Add this model to current row
            current_row_models.append(model)
            current_row_total_inputs += num_inputs

        # Place any remaining models in the last row
        if current_row_models:
            _place_input_metric_row(current_row_models, grouped_inputs, pos, -y_offset, 28.0)
            y_offset += y_spacing

        # Place orphan inputs in a separate row if any exist
        if orphan_inputs:
            x_spacing_orphan = _calculate_adaptive_spacing(len(orphan_inputs), min_spacing=23.0, max_spacing=32.0)
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

        # Note: Failure analysis available in validation_failures_all_documents.csv
        # (generated by generate_failures_csv.py)

    print("\n" + "=" * 80)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Files saved in: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
