import json
import os
from pathlib import Path

def load_validated_data(file_path):
    """Load validated JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_entities_by_type(data):
    """Extract and organize entities by type"""
    entities_by_type = {
        'Category': [],
        'Industry': [],
        'Metric': [],
        'Model': [],
        'ReportingFramework': []
    }

    for entity in data.get('entities', []):
        entity_type = entity.get('type')
        if entity_type in entities_by_type:
            entities_by_type[entity_type].append({
                'id': entity.get('id'),
                'label': entity.get('label'),
                'properties': entity.get('properties', {})
            })

    return entities_by_type

def format_entity_table(entity_type, entities):
    """Format entities as markdown table"""
    if not entities:
        return f"No {entity_type} entities found.\n"

    # Sort entities by ID for consistency
    entities = sorted(entities, key=lambda x: x['id'])

    lines = []
    lines.append(f"### {entity_type} ({len(entities)} total)\n")

    if entity_type == 'Category':
        lines.append("| ID | Label | Section ID | Page Range |")
        lines.append("|-----|-------|------------|------------|")
        for e in entities:
            props = e['properties']
            lines.append(f"| {e['id']} | {e['label']} | {props.get('section_id', 'N/A')} | {props.get('page_range', 'N/A')} |")

    elif entity_type == 'Industry':
        lines.append("| ID | Label | Sector | Standard Reference |")
        lines.append("|-----|-------|--------|-------------------|")
        for e in entities:
            props = e['properties']
            lines.append(f"| {e['id']} | {e['label']} | {props.get('sector', 'N/A')} | {props.get('standard_reference', 'N/A')} |")

    elif entity_type == 'Metric':
        lines.append("| ID | Label | Code | Metric Type | Unit |")
        lines.append("|-----|-------|------|-------------|------|")
        for e in entities:
            props = e['properties']
            code = props.get('code', 'N/A')
            metric_type = props.get('metric_type', 'N/A')
            unit = props.get('unit', 'N/A')
            # Truncate long units
            if len(unit) > 50:
                unit = unit[:47] + "..."
            lines.append(f"| {e['id']} | {e['label']} | {code} | {metric_type} | {unit} |")

    elif entity_type == 'Model':
        lines.append("| ID | Label | Description |")
        lines.append("|-----|-------|-------------|")
        for e in entities:
            props = e['properties']
            desc = props.get('description', 'N/A')
            # Truncate long descriptions
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(f"| {e['id']} | {e['label']} | {desc} |")

    elif entity_type == 'ReportingFramework':
        lines.append("| ID | Label | Version | Year | Publisher |")
        lines.append("|-----|-------|---------|------|-----------|")
        for e in entities:
            props = e['properties']
            publisher = props.get('publisher', 'N/A')
            # Truncate long publishers
            if len(publisher) > 60:
                publisher = publisher[:57] + "..."
            lines.append(f"| {e['id']} | {e['label']} | {props.get('version', 'N/A')} | {props.get('year', 'N/A')} | {publisher} |")

    lines.append("")
    return "\n".join(lines)

def generate_document_detailed_results(doc_name, doc_title, validated_file):
    """Generate detailed results for a single document"""
    data = load_validated_data(validated_file)
    entities_by_type = extract_entities_by_type(data)

    output = []
    output.append(f"# {doc_title} - Detailed Entity Breakdown\n")
    output.append("---\n")

    # Summary
    total = sum(len(entities) for entities in entities_by_type.values())
    output.append("## Summary\n")
    output.append(f"**Total Entities After Validation:** {total}\n")
    output.append("")

    # Entity counts
    for entity_type in ['Category', 'Industry', 'Metric', 'Model', 'ReportingFramework']:
        count = len(entities_by_type[entity_type])
        output.append(f"- **{entity_type}:** {count}")
    output.append("\n---\n")

    # Detailed tables for each type
    for entity_type in ['Category', 'Industry', 'Metric', 'Model', 'ReportingFramework']:
        output.append(format_entity_table(entity_type, entities_by_type[entity_type]))
        output.append("")

    return "\n".join(output)

def main():
    # Document mappings
    documents = [
        {
            'name': '1. SASB-commercial-banks-standard_en-gb',
            'title': 'SASB Commercial Banks',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1. SASB-commercial-banks-standard_en-gb_validated.json'
        },
        {
            'name': '1.SASB-semiconductors-standard_en-gb',
            'title': 'SASB Semiconductors',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1.SASB-semiconductors-standard_en-gb_validated.json'
        },
        {
            'name': '1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures',
            'title': 'IFRS S2',
            'validated_file': 'outputs/stage3_ontology_guided_validation/1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures_validated.json'
        },
        {
            'name': '2.Australia-AASBS2_09-24',
            'title': 'Australia AASB S2',
            'validated_file': 'outputs/stage3_ontology_guided_validation/2.Australia-AASBS2_09-24_validated.json'
        },
        {
            'name': '2.FINAL-2017-TCFD-Report',
            'title': 'TCFD Report',
            'validated_file': 'outputs/stage3_ontology_guided_validation/2.FINAL-2017-TCFD-Report_validated.json'
        }
    ]

    base_dir = Path('.')
    output_dir = base_dir / 'result_visualisation_and_analysis' / 'detailed_entity_breakdowns'
    output_dir.mkdir(exist_ok=True)

    for doc in documents:
        validated_path = base_dir / doc['validated_file']

        if not validated_path.exists():
            print(f"Warning: File not found: {validated_path}")
            continue

        print(f"Processing: {doc['title']}")

        # Generate detailed breakdown
        content = generate_document_detailed_results(
            doc['name'],
            doc['title'],
            validated_path
        )

        # Save to file
        output_file = output_dir / f"{doc['title'].replace(' ', '_')}_Detailed_Entities.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✓ Created: {output_file}")

    print("\nAll detailed entity breakdown files created successfully!")

if __name__ == '__main__':
    main()
