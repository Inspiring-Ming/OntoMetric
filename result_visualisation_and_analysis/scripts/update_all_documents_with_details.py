import json
from pathlib import Path

def load_validated_data(file_path):
    """Load validated JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_entities_by_type(data):
    """Extract and organize entities by type"""
    from collections import defaultdict

    entities_by_type = {
        'Category': [],
        'Industry': [],
        'Metric': [],
        'Model': [],
        'ReportingFramework': []
    }

    # Store all entities with their IDs for lookup
    entities_dict = {}
    for entity in data.get('entities', []):
        entities_dict[entity.get('id')] = entity

    # Extract relationships
    relationships = data.get('relationships', [])
    model_to_inputs = {}  # model_id -> [input_metric_labels]
    metric_to_categories = defaultdict(list)  # metric_id -> [category_labels]
    category_to_metric_count = defaultdict(int)  # category_id -> count

    for rel in relationships:
        if rel.get('predicate') == 'RequiresInputFrom':
            model_id = rel.get('subject')
            input_metric_id = rel.get('object')
            if model_id not in model_to_inputs:
                model_to_inputs[model_id] = []
            # Get the label of the input metric
            if input_metric_id in entities_dict:
                input_label = entities_dict[input_metric_id].get('label')
                model_to_inputs[model_id].append(input_label)

        elif rel.get('predicate') == 'ConsistOf':
            subject_id = rel.get('subject')
            object_id = rel.get('object')
            subject = entities_dict.get(subject_id, {})
            obj = entities_dict.get(object_id, {})

            # Category -> Metric relationship
            if subject.get('type') == 'Category' and obj.get('type') == 'Metric':
                category_label = subject.get('label')
                metric_to_categories[object_id].append(category_label)
                category_to_metric_count[subject_id] += 1

    for entity in data.get('entities', []):
        entity_type = entity.get('type')
        if entity_type in entities_by_type:
            entity_data = {
                'id': entity.get('id'),
                'label': entity.get('label'),
                'properties': entity.get('properties', {})
            }
            # Add input metrics for models
            if entity_type == 'Model':
                entity_data['input_metrics'] = model_to_inputs.get(entity.get('id'), [])
            # Add categories for metrics
            elif entity_type == 'Metric':
                categories = metric_to_categories.get(entity.get('id'), [])
                entity_data['categories'] = categories if categories else ['N/A']
            # Add metric count for categories
            elif entity_type == 'Category':
                entity_data['metric_count'] = category_to_metric_count.get(entity.get('id'), 0)

            entities_by_type[entity_type].append(entity_data)

    return entities_by_type

def format_detailed_entity_tables(entities_by_type):
    """Format all entity types as markdown tables"""
    lines = []

    # Category table - WITH METRIC COUNT
    categories = sorted(entities_by_type['Category'], key=lambda x: x['id'])
    if categories:
        lines.append("#### Categories")
        lines.append("| Label | Section ID | Metrics |")
        lines.append("|-------|------------|---------|")
        for e in categories:
            props = e['properties']
            metric_count = e.get('metric_count', 0)
            lines.append(f"| {e['label']} | {props.get('section_id', 'N/A')} | {metric_count} |")
        lines.append("")

    # Industry table
    industries = sorted(entities_by_type['Industry'], key=lambda x: x['id'])
    if industries:
        lines.append("#### Industries")
        lines.append("| Label | Sector |")
        lines.append("|-------|--------|")
        for e in industries:
            props = e['properties']
            lines.append(f"| {e['label']} | {props.get('sector', 'N/A')} |")
        lines.append("")

    # Metric table - WITH CATEGORY
    metrics = sorted(entities_by_type['Metric'], key=lambda x: x['id'])
    if metrics:
        lines.append("#### Metrics")
        lines.append("| Label | Code | Type | Category |")
        lines.append("|-------|------|------|----------|")
        for e in metrics:
            props = e['properties']
            code = props.get('code', 'N/A')
            metric_type = props.get('metric_type', 'N/A')
            categories = e.get('categories', ['N/A'])
            category_str = ', '.join(categories) if len(categories) <= 2 else f"{categories[0]}, +{len(categories)-1} more"
            lines.append(f"| {e['label']} | {code} | {metric_type} | {category_str} |")
        lines.append("")

    # Model table
    models = sorted(entities_by_type['Model'], key=lambda x: x['id'])
    if models:
        lines.append("#### Models")
        lines.append("| Label | Formula | Input Metrics |")
        lines.append("|-------|---------|---------------|")
        for e in models:
            props = e['properties']
            formula = props.get('equation', 'N/A')
            # Get input metrics from relationships
            input_metrics = e.get('input_metrics', [])
            input_metrics_str = ', '.join(input_metrics) if input_metrics else 'N/A'
            lines.append(f"| {e['label']} | {formula} | {input_metrics_str} |")
        lines.append("")

    # ReportingFramework table
    frameworks = sorted(entities_by_type['ReportingFramework'], key=lambda x: x['id'])
    if frameworks:
        lines.append("#### Reporting Framework")
        lines.append("| Label | Version | Year |")
        lines.append("|-------|---------|------|")
        for e in frameworks:
            props = e['properties']
            lines.append(f"| {e['label']} | {props.get('version', 'N/A')} | {props.get('year', 'N/A')} |")
        lines.append("")

    return "\n".join(lines)

def generate_updated_all_documents_results():
    """Generate updated All_Documents_Results.md with detailed entity tables"""

    base_dir = Path('.')

    # Document mappings
    documents = [
        {
            'title': 'SASB Commercial Banks',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1. SASB-commercial-banks-standard_en-gb_validated.json',
            'stage1_title': 'Commercial Banks Sustainability Accounting Standard FINANCIALS SECTOR',
            'stage1_sections': 10,
            'stage1_segments': 10,
            'stage1_example': 'Sustainability Disclosure Topics & Metrics (pages 6-7)',
            'stage2_entities': 53,
            'stage2_triple': '(metric_SASB-CB_6_001, IsCalculatedBy, model_SASB-CB_6_001): "Data Breaches Composite Metric" calculated by composite model',
            'stage3_validated': 42,
            'stage3_schema': 82.72,
            'stage3_semantic': 79.25,
            'stage3_relationship': 79.25,
            'stage3_example': '"Number of Past Due Small Business Loans" - financial metric, not ESG'
        },
        {
            'title': 'SASB Semiconductors',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1.SASB-semiconductors-standard_en-gb_validated.json',
            'stage1_title': 'Semiconductors Sustainability Accounting Standard TECHNOLOGY & COMMUNICATIONS SECTOR',
            'stage1_sections': 13,
            'stage1_segments': 13,
            'stage1_example': 'Greenhouse Gas Emissions (pages 8-10)',
            'stage2_entities': 69,
            'stage2_triple': '(Category, ConsistOf, Metric): "GHG Emissions" → "Gross global Scope 1 emissions"',
            'stage3_validated': 62,
            'stage3_schema': 82.02,
            'stage3_semantic': 89.86,
            'stage3_relationship': 90.14,
            'stage3_example': 'Non-ESG competitive behavior metrics'
        },
        {
            'title': 'IFRS S2',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures_validated.json',
            'stage1_title': 'June 2023 IFRS S2 IFRS Sustainability Disclosure Standard',
            'stage1_sections': 10,
            'stage1_segments': 10,
            'stage1_example': 'Strategy (pages 8-23)',
            'stage2_entities': 80,
            'stage2_triple': '(ReportingFramework, Include, Category): IFRS S2 → "Strategy"',
            'stage3_validated': 68,
            'stage3_schema': 81.48,
            'stage3_semantic': 85.00,
            'stage3_relationship': 89.23,
            'stage3_example': 'Metrics with missing unit specifications'
        },
        {
            'title': 'Australia AASB S2',
            'validated_file': 'outputs/stage3_ontology_guided_validation/2.Australia-AASBS2_09-24_validated.json',
            'stage1_title': 'Australian Sustainability Reporting Standard AASB S2 September 2024',
            'stage1_sections': 8,
            'stage1_segments': 8,
            'stage1_example': 'APPENDICES (pages 11-57)',
            'stage2_entities': 74,
            'stage2_triple': '(Category, ConsistOf, Metric): "Strategy" → climate risk metrics',
            'stage3_validated': 66,
            'stage3_schema': 83.33,
            'stage3_semantic': 89.19,
            'stage3_relationship': 86.67,
            'stage3_example': 'Metrics lacking valid code/unit'
        },
        {
            'title': 'TCFD Report',
            'validated_file': 'outputs/stage3_ontology_guided_validation/2.FINAL-2017-TCFD-Report_validated.json',
            'stage1_title': 'FINAL-2017-TCFD-Report',
            'stage1_sections': 19,
            'stage1_segments': 19,
            'stage1_example': 'Guidance for All Sectors (pages 19-24)',
            'stage2_entities': 88,
            'stage2_triple': '(Category, ConsistOf, Metric): "Guidance All Sectors" → "Scope 1, 2, and 3 GHG emissions"',
            'stage3_validated': 57,
            'stage3_schema': 80.09,
            'stage3_semantic': 64.77,
            'stage3_relationship': 62.79,
            'stage3_example': 'Narrative guidance paragraphs extracted as Metrics'
        }
    ]

    output = []
    output.append("# ESG Metric Extraction Experiments - All Results\n")
    output.append("---\n")

    # Process each document
    for doc in documents:
        output.append(f"## {doc['title']}\n")

        # Stage 1
        output.append("### Stage 1: PDF Segmentation")
        output.append("| Title page | Number of sections | Number of segments extracted | Example of segment |")
        output.append("|------------|-------------------|------------------------------|-------------------|")
        output.append(f"| {doc['stage1_title']} | {doc['stage1_sections']} | {doc['stage1_segments']} | {doc['stage1_example']} |\n")

        # Stage 2
        output.append("### Stage 2: Ontology-Guided Extraction")
        output.append("| Number of entities extracted | Example of triple |")
        output.append("|------------------------------|-------------------|")
        output.append(f"| {doc['stage2_entities']} | {doc['stage2_triple']} |\n")

        # Stage 3
        output.append("### Stage 3: Two-Phase Validation")
        output.append("| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |")
        output.append("|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|")
        output.append(f"| VR001, VR002, VR003, VR004, VR005, VR006 | {doc['stage2_entities']} | {doc['stage3_validated']} | {doc['stage3_schema']} | {doc['stage3_semantic']} | {doc['stage3_relationship']} | {doc['stage3_example']} |\n")

        # Resulting Ontology - Now with detailed tables
        output.append("### Resulting Ontology\n")

        # Load and extract entities
        validated_path = base_dir / doc['validated_file']
        if validated_path.exists():
            data = load_validated_data(validated_path)
            entities_by_type = extract_entities_by_type(data)

            # Add summary counts
            output.append("**Entity Counts:**")
            output.append("| Category | Industry | Metric | Model | ReportingFramework | Total |")
            output.append("|----------|----------|--------|-------|--------------------|-------|")
            counts = [
                len(entities_by_type['Category']),
                len(entities_by_type['Industry']),
                len(entities_by_type['Metric']),
                len(entities_by_type['Model']),
                len(entities_by_type['ReportingFramework'])
            ]
            total = sum(counts)
            output.append(f"| {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} | {counts[4]} | {total} |\n")

            # Add detailed tables
            output.append(format_detailed_entity_tables(entities_by_type))

        output.append("---\n")

    # Summary Statistics
    output.append("## Summary Statistics\n")
    output.append("### Overall Results")
    output.append("| Document | Total Pages | Segments | Entities Extracted | Entities Validated | Retention Rate (%) |")
    output.append("|----------|------------|----------|-------------------|-------------------|--------------------|")
    output.append("| SASB Commercial Banks | 23 | 10 | 53 | 42 | 79.25 |")
    output.append("| SASB Semiconductors | 27 | 13 | 69 | 62 | 89.86 |")
    output.append("| IFRS S2 | 46 | 10 | 80 | 68 | 85.00 |")
    output.append("| Australia AASB S2 | 58 | 8 | 74 | 66 | 89.19 |")
    output.append("| TCFD Report | 74 | 19 | 88 | 57 | 64.77 |")
    output.append("| **TOTAL** | **228** | **60** | **364** | **295** | **81.04** |\n")

    output.append("### Validation Quality Metrics")
    output.append("| Document | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) |")
    output.append("|----------|----------------------|----------------------|---------------------------|")
    output.append("| SASB Commercial Banks | 82.72 | 79.25 | 79.25 |")
    output.append("| SASB Semiconductors | 82.02 | 89.86 | 90.14 |")
    output.append("| IFRS S2 | 81.48 | 85.00 | 89.23 |")
    output.append("| Australia AASB S2 | 83.33 | 89.19 | 86.67 |")
    output.append("| TCFD Report | 80.09 | 64.77 | 62.79 |")
    output.append("")

    return "\n".join(output)

def main():
    print("Generating updated All_Documents_Results.md with detailed entity tables...")

    content = generate_updated_all_documents_results()

    output_file = Path('./result_visualisation_and_analysis/All_Documents_Results.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Updated: {output_file}")
    print("\nAll documents now include detailed entity breakdown tables!")

if __name__ == '__main__':
    main()
