#!/usr/bin/env python3
"""
Generate a single CSV file consolidating all validation failures from 5 documents.
Compares Stage 2 ontology-guided extraction with Stage 3 ontology-guided validation.
Replaces the 5 separate failure PNG figures with one consolidated CSV.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def load_json(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_failures(stage2_data, stage3_data, document_name):
    """
    Extract failed entities by comparing Stage 2 (before validation) and Stage 3 (after validation).
    Returns list of failure records with document name, entity details, and detailed LLM reasoning.
    """
    failures = []

    # Get Stage 2 and Stage 3 entities
    stage2_entities = {e['id']: e for e in stage2_data.get('entities', [])}
    stage3_entity_ids = {e['id'] for e in stage3_data.get('entities', [])}

    # Get semantic validation mismatches from Stage 3 (this contains detailed LLM reasoning)
    semantic_validation = stage3_data.get('validation_metadata', {}).get('semantic_validation', {})
    mismatches = semantic_validation.get('mismatches', [])

    # Build mapping of entity ID to detailed LLM reasoning
    llm_reasoning_map = {}
    for mismatch in mismatches:
        entity_id = mismatch.get('entity_id')
        reasoning = mismatch.get('llm_reasoning', 'No reasoning provided')
        llm_reasoning_map[entity_id] = reasoning

    # Find entities in Stage 2 but not in Stage 3
    for entity_id, entity in stage2_entities.items():
        if entity_id not in stage3_entity_ids:
            # Try to get detailed LLM reasoning, fall back to generic message
            failure_reason = llm_reasoning_map.get(
                entity_id,
                'Entity removed during validation (detailed reasoning not available)'
            )

            failures.append({
                'Document': document_name,
                'Entity_ID': entity_id,
                'Entity_Type': entity.get('type', 'Unknown'),
                'Entity_Label': entity.get('label', 'No label'),
                'Failure_Reason': failure_reason
            })

    return failures


def main():
    base_dir = Path('.')

    # Document configurations - using correct Stage 2 ontology-guided and Stage 3 ontology-guided paths
    documents = [
        {
            'name': 'SASB Commercial Banks',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/1. SASB-commercial-banks-standard_en-gb_ontology_guided.json',
            'stage3_file': 'outputs/stage3_ontology_guided_validation/1. SASB-commercial-banks-standard_en-gb_validated.json'
        },
        {
            'name': 'SASB Semiconductors',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/1.SASB-semiconductors-standard_en-gb_ontology_guided.json',
            'stage3_file': 'outputs/stage3_ontology_guided_validation/1.SASB-semiconductors-standard_en-gb_validated.json'
        },
        {
            'name': 'IFRS S2',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures_ontology_guided.json',
            'stage3_file': 'outputs/stage3_ontology_guided_validation/1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures_validated.json'
        },
        {
            'name': 'Australia AASB S2',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/2.Australia-AASBS2_09-24_ontology_guided.json',
            'stage3_file': 'outputs/stage3_ontology_guided_validation/2.Australia-AASBS2_09-24_validated.json'
        },
        {
            'name': 'TCFD Report',
            'stage2_file': 'outputs/stage2_ontology_guided_extraction/2.FINAL-2017-TCFD-Report_ontology_guided.json',
            'stage3_file': 'outputs/stage3_ontology_guided_validation/2.FINAL-2017-TCFD-Report_validated.json'
        }
    ]

    print("="*80)
    print("CONSOLIDATING VALIDATION FAILURES INTO CSV")
    print("="*80)
    print()

    all_failures = []

    # Process each document
    for doc in documents:
        print(f"Processing: {doc['name']}")

        stage2_path = base_dir / doc['stage2_file']
        stage3_path = base_dir / doc['stage3_file']

        if not stage2_path.exists():
            print(f"  ⚠️  Stage 2 file not found: {stage2_path}")
            continue

        if not stage3_path.exists():
            print(f"  ⚠️  Stage 3 file not found: {stage3_path}")
            continue

        # Load data
        stage2_data = load_json(stage2_path)
        stage3_data = load_json(stage3_path)

        # Extract failures
        failures = extract_failures(stage2_data, stage3_data, doc['name'])
        all_failures.extend(failures)

        print(f"  ✓ Found {len(failures)} failed validations")

    print()
    print(f"Total failures across all documents: {len(all_failures)}")
    print()

    # Write to CSV
    output_file = base_dir / 'result_visualisation_and_analysis/figures/validation_failures_all_documents.csv'

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Document', 'Entity_ID', 'Entity_Type', 'Entity_Label', 'Failure_Reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for failure in all_failures:
            writer.writerow(failure)

    print(f"✓ CSV file created: {output_file}")
    print(f"  Total rows: {len(all_failures) + 1} (including header)")
    print()

    # Show summary by document
    print("="*80)
    print("SUMMARY BY DOCUMENT")
    print("="*80)

    doc_counts = defaultdict(int)
    for failure in all_failures:
        doc_counts[failure['Document']] += 1

    for doc_name in [d['name'] for d in documents]:
        count = doc_counts[doc_name]
        print(f"  {doc_name:30s}: {count:3d} failures")

    print("="*80)
    print()


if __name__ == '__main__':
    main()
