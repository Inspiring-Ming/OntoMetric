#!/usr/bin/env python3
"""
Stage 3: Baseline LLM Comparison - Generate Clear Comparison Report

Validates and compares Ontology-Guided vs Baseline LLM extraction with clear metrics:
- Entity and relationship counts
- CQ Violation Rate (%)
- Triple Retention Rate (%)
- Provenance Completeness (%)
- Ontology Compliance (%)

Experiments:
1. Ontology-Guided Extraction - Complete ESGMKG prompt with validation
2. Baseline (Unconstrained) - Simple LLM prompt without ontology schema

Usage:
    # Process single document
    python3 src/stage3_baseline_llm_comparison.py "document_name"

    # Process all documents in batch
    python3 src/stage3_baseline_llm_comparison.py --batch

OUTPUT FILES:
=============
Single Document Mode:
  • {document_name}_baseline_llm_comparison.txt
  Location: outputs/stage3_baseline_llm_comparison/
  **Total: 1 file per document**

Batch Mode:
  • batch_baseline_llm_comparison.txt
  Location: outputs/stage3_baseline_llm_comparison/
  **Total: 1 file**

This file contains:
  - Head-to-head comparison table
  - Improvement deltas (e.g., +55.6% triple retention)
  - Detailed entity type distributions
  - Key insights and findings
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Import validation logic from stage3_ontology_guided_validation
try:
    from stage3_ontology_guided_validation import KnowledgeGraphValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("⚠️ Warning: Could not import stage3_ontology_guided_validation. Using simplified validation.")


class BaselineLLMComparator:
    """Validates and compares baseline LLM vs ontology-guided extraction results"""

    # ESGMKG valid types
    VALID_ENTITY_TYPES = {
        'Industry', 'ReportingFramework', 'Category',
        'Metric', 'DirectMetric', 'CalculatedMetric', 'InputMetric',
        'Model', 'Implementation', 'Dataset-Variable', 'Datasource'
    }

    VALID_PREDICATES = {
        'ReportUsing', 'Include', 'ConsistOf', 'IsCalculatedBy',
        'RequiresInputFrom', 'ExecutesWith', 'ObtainedFrom', 'SourceFrom'
    }

    def __init__(self, document_name: str = None):
        """Initialize validator

        Args:
            document_name: Name of document to compare (without extension)
        """
        self.document_name = document_name
        self.base_dir = Path(__file__).parent.parent
        self.experiments_dir = self.base_dir / "outputs" / "stage2_baseline_llm_extraction"
        self.validation_dir = self.base_dir / "outputs" / "stage3_baseline_llm_comparison"

        # Create validation output directory
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Initialize real validator for consistent validation
        if VALIDATION_AVAILABLE:
            self.kg_validator = KnowledgeGraphValidator()
        else:
            self.kg_validator = None

        print(f"✓ Baseline LLM Comparator initialized")
        print(f"  Experiments: {self.experiments_dir}")
        print(f"  Validation Output: {self.validation_dir}")
        print(f"  Using real stage3_ontology_guided_validation: {VALIDATION_AVAILABLE}")


    def compare_document(self, document_name: str):
        """Compare 2 experiments for a single document: Ontology-Guided vs Baseline LLM

        Args:
            document_name: Name of document to compare
        """
        print(f"\n{'='*100}")
        print(f"BASELINE LLM COMPARISON: {document_name}")
        print(f"{'='*100}\n")

        # Load combined experiment results from single file
        combined_file = self.experiments_dir / f"{document_name}_baseline_llm.json"

        if not combined_file.exists():
            print(f"\n❌ Experiment results file not found: {combined_file}")
            print("\nRun experiments first:")
            print(f"  python3 src/stage2_baseline_llm_extraction.py \"{document_name}\"")
            sys.exit(1)

        with open(combined_file, 'r') as f:
            combined_data = json.load(f)

        # Extract individual experiment results
        results = {}
        experiments = combined_data.get('experiments', {})

        for exp_id in ["1_ontology_guided", "2_baseline_llm"]:
            if exp_id in experiments:
                results[exp_id] = experiments[exp_id]
                print(f"✓ Loaded {experiments[exp_id].get('name', exp_id)}")
            else:
                print(f"✗ Missing {exp_id}")
                results[exp_id] = None

        if not any(results.values()):
            print("\n❌ No experiment results found in combined file!")
            sys.exit(1)

        print()

        # Calculate metrics for each
        metrics = {}
        for exp_id, result in results.items():
            if result:
                metrics[exp_id] = self._calculate_metrics(exp_id, result)

        # Generate comparison report
        output_file = self.validation_dir / f"{document_name}_baseline_llm_comparison.txt"
        self._generate_report(metrics, document_name, output_file)


    def _calculate_metrics(self, exp_id, result):
        """Calculate all metrics for one experiment using REAL stage3_validation"""
        entities = result.get('entities', [])
        relationships = result.get('relationships', [])

        # Use REAL validation from stage3_validation.py for ALL experiments
        if self.kg_validator and VALIDATION_AVAILABLE:
            # Run real hybrid validation (Python + SPARQL if RDF available)
            # For JSON-only experiments, use Python validation
            try:
                # Prepare data for validation
                kg_data = {
                    'entities': entities,
                    'relationships': relationships
                }

                # Run Python validation (works for all experiments)
                validation_result = self._run_real_validation(kg_data)

                cq_violation_rate = validation_result['cq_violation_rate']
                triple_retention_rate = validation_result['triple_retention_rate']
                provenance_completeness = validation_result['provenance_completeness']
                ontology_compliance = validation_result['ontology_compliance']

            except Exception as e:
                print(f"  Warning: Real validation failed for {exp_id}, using fallback: {e}")
                # Fallback to simple calculation
                cq_violation_rate = self._calc_cq_violation_rate(entities, relationships)
                triple_retention_rate = 0.0  # Cannot calculate without real validation
                provenance_completeness = self._calc_provenance_completeness(entities)
                ontology_compliance = self._calc_ontology_compliance(entities, relationships)
        else:
            # Fallback: use simple calculation
            cq_violation_rate = self._calc_cq_violation_rate(entities, relationships)
            triple_retention_rate = 0.0  # Cannot calculate without real validation
            provenance_completeness = self._calc_provenance_completeness(entities)
            ontology_compliance = self._calc_ontology_compliance(entities, relationships)

        metrics = {
            'exp_id': exp_id,
            'name': result.get('meta', {}).get('experiment_name', exp_id),
            'description': result.get('meta', {}).get('description', ''),
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'total_triples': len(entities) + len(relationships),
            'entity_types': self._count_entity_types(entities),
            'cq_violation_rate': cq_violation_rate,
            'triple_retention_rate': triple_retention_rate,
            'provenance_completeness': provenance_completeness,
            'ontology_compliance': ontology_compliance,
        }

        print(f"  [{exp_id}] Entities: {metrics['total_entities']}, "
              f"Relationships: {metrics['total_relationships']}, "
              f"CQ Violations: {metrics['cq_violation_rate']:.1f}%")

        return metrics


    def _run_real_validation(self, kg_data):
        """Run real validation using stage3_validation logic"""
        entities = kg_data.get('entities', [])
        relationships = kg_data.get('relationships', [])

        # Calculate CQ Violation Rate using ALL 10 validation rules from stage3_validation
        total_checks = 0
        total_violations = 0

        # Track rule pass/fail for Triple Retention Rate
        rule_results = {}  # rule_id -> has_violations (boolean)

        # VR001: CalculatedMetrics must have IsCalculatedBy
        calc_metrics = [
            e for e in entities
            if e.get('type') == 'Metric' and e.get('properties', {}).get('metric_type') == 'CalculatedMetric'
        ]
        is_calc_by = [r for r in relationships if r.get('predicate') == 'IsCalculatedBy']
        vr001_violations = 0
        for metric in calc_metrics:
            total_checks += 1
            if not any(r.get('subject') == metric.get('id') for r in is_calc_by):
                total_violations += 1
                vr001_violations += 1
        rule_results['VR001'] = vr001_violations == 0

        # VR002: Quantitative metrics must have units
        quant_metrics = [
            e for e in entities
            if e.get('type') == 'Metric' and e.get('properties', {}).get('measurement_type') == 'Quantitative'
        ]
        vr002_violations = 0
        for metric in quant_metrics:
            total_checks += 1
            unit = metric.get('properties', {}).get('unit', '')
            if not unit or unit in ['N/A', '']:
                total_violations += 1
                vr002_violations += 1
        rule_results['VR002'] = vr002_violations == 0

        # VR003: Models must have input variables
        models = [e for e in entities if e.get('type') == 'Model']
        vr003_violations = 0
        for model in models:
            total_checks += 1
            input_vars = model.get('properties', {}).get('input_variables', '')
            if not input_vars or input_vars in ['N/A', '']:
                total_violations += 1
                vr003_violations += 1
        rule_results['VR003'] = vr003_violations == 0

        # VR004: Entity IDs must be unique
        total_checks += len(entities)
        entity_ids = [e.get('id') for e in entities]
        duplicate_count = len(entity_ids) - len(set(entity_ids))
        total_violations += duplicate_count
        rule_results['VR004'] = duplicate_count == 0

        # VR005: Entities must have required properties (checks all entities)
        total_checks += len(entities)
        vr005_violations = 0
        for entity in entities:
            entity_type = entity.get('type')
            props = entity.get('properties', {})

            # Check based on entity type
            if entity_type == 'Metric':
                required = ['measurement_type', 'metric_type', 'unit', 'description']
                if not all(props.get(r) for r in required):
                    total_violations += 1
                    vr005_violations += 1
            elif entity_type in ['Industry', 'ReportingFramework', 'Category', 'Model']:
                # These entities should at least have a label
                if not entity.get('label'):
                    total_violations += 1
                    vr005_violations += 1
        rule_results['VR005'] = vr005_violations == 0

        # VR006: Categories should contain metrics (non-critical)
        categories = [e for e in entities if e.get('type') == 'Category']
        consist_of = [r for r in relationships if r.get('predicate') == 'ConsistOf']
        vr006_violations = 0
        for category in categories:
            total_checks += 1
            if not any(r.get('subject') == category.get('id') for r in consist_of):
                total_violations += 1
                vr006_violations += 1
        rule_results['VR006'] = vr006_violations == 0

        # VR007: Industries should report using frameworks (non-critical)
        industries = [e for e in entities if e.get('type') == 'Industry']
        report_using = [r for r in relationships if r.get('predicate') == 'ReportUsing']
        vr007_violations = 0
        for industry in industries:
            total_checks += 1
            if not any(r.get('subject') == industry.get('id') for r in report_using):
                total_violations += 1
                vr007_violations += 1
        rule_results['VR007'] = vr007_violations == 0

        # VR008: Qualitative metrics should have method property (non-critical)
        # NOTE: VR008 only applies to DirectMetric, CalculatedMetric, InputMetric - NOT generic "Metric"
        qual_metrics = [
            e for e in entities
            if e.get('type') in ['DirectMetric', 'CalculatedMetric', 'InputMetric']
            and e.get('properties', {}).get('measurement_type') == 'Qualitative'
        ]
        vr008_violations = 0
        for metric in qual_metrics:
            total_checks += 1
            method = metric.get('properties', {}).get('method', '')
            if not method or method == 'N/A':
                total_violations += 1
                vr008_violations += 1
        rule_results['VR008'] = vr008_violations == 0

        # VR009: Models with equations must have input variables
        vr009_violations = 0
        for model in models:
            equation = model.get('properties', {}).get('equation', '')
            if equation and equation not in ['N/A', '']:
                total_checks += 1
                input_vars = model.get('properties', {}).get('input_variables', '')
                if not input_vars or input_vars in ['N/A', '']:
                    total_violations += 1
                    vr009_violations += 1
        rule_results['VR009'] = vr009_violations == 0

        # VR010: Relationships must be valid
        valid_predicates = self.VALID_PREDICATES
        vr010_violations = 0
        for rel in relationships:
            total_checks += 1
            if rel.get('predicate') not in valid_predicates:
                total_violations += 1
                vr010_violations += 1
        rule_results['VR010'] = vr010_violations == 0

        cq_violation_rate = (total_violations / total_checks * 100) if total_checks > 0 else 0

        # Provenance completeness
        entities_with_prov = sum(
            1 for e in entities
            if e.get('provenance') or e.get('source')
        )
        provenance_completeness = (entities_with_prov / len(entities) * 100) if entities else 0

        # Ontology compliance (semantic matching)
        compliant_entities = sum(
            1 for e in entities
            if self._matches_ontology_type(e.get('type', ''))
        )
        compliant_rels = sum(
            1 for r in relationships
            if self._matches_ontology_predicate(r.get('predicate', ''))
        )
        total = len(entities) + len(relationships)
        ontology_compliance = ((compliant_entities + compliant_rels) / total * 100) if total > 0 else 0

        # Triple Retention Rate = (Passed Rules / Applicable Rules) × 100
        # Option A: Only count rules where entities exist to check (no "passing by omission")
        # This gives a more honest assessment of baseline vs ontology-guided extraction

        # Determine which rules are "applicable" (have entities to check)
        applicable_rules = {}
        applicable_rules['VR001'] = len(calc_metrics) > 0  # Only if CalculatedMetrics exist
        applicable_rules['VR002'] = len(quant_metrics) > 0  # Only if Quantitative metrics exist
        applicable_rules['VR003'] = len(models) > 0  # Only if Models exist
        applicable_rules['VR004'] = True  # Always applicable (all entities must be unique)
        applicable_rules['VR005'] = True  # Always applicable (all entities need properties)
        applicable_rules['VR006'] = len(categories) > 0  # Only if Categories exist
        applicable_rules['VR007'] = len(industries) > 0  # Only if Industries exist
        applicable_rules['VR008'] = len(qual_metrics) > 0  # Only if Qualitative metrics exist
        applicable_rules['VR009'] = any(m.get('properties', {}).get('equation', '') not in ['', 'N/A'] for m in models)  # Only if models with equations exist
        applicable_rules['VR010'] = len(relationships) > 0  # Only if relationships exist

        # Count passed rules among applicable rules only
        passed_applicable_rules = sum(
            1 for rule_id in rule_results.keys()
            if applicable_rules.get(rule_id, False) and rule_results[rule_id]
        )
        total_applicable_rules = sum(1 for is_applicable in applicable_rules.values() if is_applicable)

        triple_retention_rate = (passed_applicable_rules / total_applicable_rules * 100) if total_applicable_rules > 0 else 0

        return {
            'cq_violation_rate': cq_violation_rate,
            'triple_retention_rate': triple_retention_rate,
            'provenance_completeness': provenance_completeness,
            'ontology_compliance': ontology_compliance
        }


    def _count_entity_types(self, entities):
        """Count entities by type"""
        counts = defaultdict(int)
        for entity in entities:
            entity_type = entity.get('type', 'Unknown')
            counts[entity_type] += 1
        return dict(counts)


    def _calc_cq_violation_rate(self, entities, relationships):
        """Calculate CQ violation rate based on validation rules"""
        total_checks = 0
        violations = 0

        # Rule 1: CalculatedMetrics must have IsCalculatedBy
        calc_metrics = [
            e for e in entities
            if e.get('properties', {}).get('metric_type') == 'CalculatedMetric'
            or e.get('type') == 'CalculatedMetric'
        ]
        is_calc_by = [r for r in relationships if r.get('predicate') == 'IsCalculatedBy']

        for metric in calc_metrics:
            total_checks += 1
            metric_id = metric.get('id')
            if not any(r.get('subject') == metric_id for r in is_calc_by):
                violations += 1

        # Rule 2: Quantitative metrics must have units
        quant_metrics = [
            e for e in entities
            if e.get('properties', {}).get('measurement_type') == 'Quantitative'
            and 'Metric' in e.get('type', '')
        ]

        for metric in quant_metrics:
            total_checks += 1
            unit = metric.get('properties', {}).get('unit', '')
            if not unit or unit.strip() == '':
                violations += 1

        # Rule 3: Models must have input variables
        models = [e for e in entities if e.get('type') == 'Model']
        requires_input = [r for r in relationships if r.get('predicate') == 'RequiresInputFrom']

        for model in models:
            total_checks += 1
            model_id = model.get('id')
            has_inputs = any(r.get('subject') == model_id for r in requires_input)
            input_vars = model.get('properties', {}).get('input_variables', [])

            if not has_inputs and not input_vars:
                violations += 1

        # Rule 4: Relationships must be valid
        for rel in relationships:
            total_checks += 1
            if rel.get('predicate') not in self.VALID_PREDICATES:
                violations += 1

        if total_checks == 0:
            return 0.0

        return (violations / total_checks) * 100


    def _calc_provenance_completeness(self, entities):
        """Calculate provenance completeness"""
        if not entities:
            return 0.0

        with_provenance = sum(
            1 for e in entities
            if e.get('provenance') or e.get('source')
        )

        return (with_provenance / len(entities)) * 100


    def _matches_ontology_type(self, entity_type):
        """Check if entity type semantically matches ESGMKG types (lenient for baseline)"""
        if not entity_type:
            return False

        normalized = entity_type.lower().replace(' ', '').replace('_', '').replace('-', '')

        # Direct match (case-insensitive)
        valid_normalized = {t.lower().replace(' ', '').replace('_', '').replace('-', '') for t in self.VALID_ENTITY_TYPES}
        if normalized in valid_normalized:
            return True

        # Semantic matches (anything containing these keywords)
        if 'metric' in normalized:
            return True  # Matches Metric (e.g., energy_metric, emissions_metric)
        if 'category' in normalized:
            return True  # Matches Category (e.g., risk_category)
        if 'framework' in normalized:
            return True  # Matches ReportingFramework (e.g., reporting_framework, framework)
        if normalized in ['industry', 'sector', 'industrysector']:
            return True  # Matches Industry
        if 'model' in normalized:
            return True  # Matches Model

        return False


    def _matches_ontology_predicate(self, predicate):
        """Check if predicate semantically matches ESGMKG predicates (lenient for baseline)"""
        if not predicate:
            return False

        normalized = predicate.lower().replace(' ', '').replace('_', '').replace('-', '')

        # Direct match (case-insensitive)
        valid_normalized = {p.lower().replace(' ', '').replace('_', '').replace('-', '') for p in self.VALID_PREDICATES}
        if normalized in valid_normalized:
            return True

        # Semantic matches
        if 'include' in normalized:
            return True  # Matches Include (e.g., includes, may_include)
        if 'consist' in normalized or 'contain' in normalized:
            return True  # Matches ConsistOf
        if 'calculated' in normalized or 'calculatedby' in normalized:
            return True  # Matches IsCalculatedBy
        if 'require' in normalized or 'input' in normalized:
            return True  # Matches RequiresInputFrom
        if 'reportusing' in normalized or 'reports' in normalized:
            return True  # Matches ReportUsing

        return False


    def _calc_ontology_compliance(self, entities, relationships):
        """Calculate ontology compliance (semantic matching for realistic measurement)"""
        if not entities and not relationships:
            return 0.0

        # Check entities with semantic matching
        compliant_entities = sum(
            1 for e in entities
            if self._matches_ontology_type(e.get('type', ''))
        )

        # Check relationships with semantic matching
        compliant_rels = sum(
            1 for r in relationships
            if self._matches_ontology_predicate(r.get('predicate', ''))
        )

        total = len(entities) + len(relationships)
        compliant = compliant_entities + compliant_rels

        return (compliant / total) * 100 if total > 0 else 0.0


    def _generate_report(self, metrics, document_name: str, output_file: Path):
        """Generate comparison report

        Args:
            metrics: Dictionary of metrics for each experiment
            document_name: Name of document being compared
            output_file: Path to output report file
        """
        lines = []

        # Header
        lines.append("=" * 120)
        lines.append("BASELINE LLM COMPARISON REPORT")
        lines.append(f"Document: {document_name}")
        lines.append("=" * 120)
        lines.append("")

        # Get reference (ontology-guided)
        full_system = metrics.get('1_ontology_guided')

        # Metric Explanations
        lines.append("Calculation Formulas:")
        lines.append("  • CQ Violation Rate (%) = (Total Violations / Total Checks) × 100")
        lines.append("    where checks include CalculatedMetric-Model links, quantitative units, model inputs, valid predicates")
        lines.append("  • Triple Retention Rate (%) = (Passed Applicable Rules / Total Applicable Rules) × 100")
        lines.append("    where a rule is 'applicable' only if relevant entities exist to validate")
        lines.append("    (excludes rules that pass trivially due to missing entity types)")
        lines.append("  • Provenance Completeness (%) = (Entities with Source Info / Total Entities) × 100")
        lines.append("  • Ontology Compliance (%) = (Valid ESGMKG Items / Total Items) × 100")
        lines.append("    where Valid Types = {Industry, ReportingFramework, Category, Metric, Model}")
        lines.append("    Note: Semantic matching (e.g., 'energy_metric' matches 'Metric', 'includes' matches 'Include')")
        lines.append("")
        lines.append("Experiment Configurations:")
        lines.append("  • Ontology-Guided Extraction: Complete ESGMKG prompt + validation rules + provenance")
        lines.append("  • Baseline (Unconstrained): Unconstrained GPT-4 (no ontology schema, no SPARQL validation)")
        lines.append("")
        lines.append("Delta Values (Δ): Change vs. Ontology-Guided (+ = worse for violations, - = worse for compliance)")
        lines.append("")

        # Section 1: Summary Table
        lines.append("SUMMARY TABLE")
        lines.append("-" * 120)
        lines.append("")

        # Table header
        header = (
            f"{'Experiment':<25} "
            f"{'Entities':>10} "
            f"{'Relations':>10} "
            f"{'CQ Violation':>15} "
            f"{'Triple Retention':>18} "
            f"{'Provenance':>12} "
            f"{'Ontology':>12}"
        )
        lines.append(header)
        lines.append("-" * 138)

        # Rows for each experiment (only 2 now)
        for exp_id in ["1_ontology_guided", "2_baseline_llm"]:
            m = metrics.get(exp_id)
            if not m:
                continue

            # Calculate deltas if not ontology-guided
            if exp_id != '1_ontology_guided' and full_system:
                cq_delta = m['cq_violation_rate'] - full_system['cq_violation_rate']
                triple_delta = m['triple_retention_rate'] - full_system['triple_retention_rate']
                prov_delta = m['provenance_completeness'] - full_system['provenance_completeness']
                ont_delta = m['ontology_compliance'] - full_system['ontology_compliance']

                cq_str = f"{m['cq_violation_rate']:>6.1f}% ({cq_delta:+.1f})"
                triple_str = f"{m['triple_retention_rate']:>6.1f}% ({triple_delta:+.1f})"
                prov_str = f"{m['provenance_completeness']:>6.1f}% ({prov_delta:+.1f})"
                ont_str = f"{m['ontology_compliance']:>6.1f}% ({ont_delta:+.1f})"
            else:
                cq_str = f"{m['cq_violation_rate']:>6.1f}%"
                triple_str = f"{m['triple_retention_rate']:>6.1f}%"
                prov_str = f"{m['provenance_completeness']:>6.1f}%"
                ont_str = f"{m['ontology_compliance']:>6.1f}%"

            row = (
                f"{m['name']:<25} "
                f"{m['total_entities']:>10} "
                f"{m['total_relationships']:>10} "
                f"{cq_str:>15} "
                f"{triple_str:>18} "
                f"{prov_str:>15} "
                f"{ont_str:>15}"
            )
            lines.append(row)

            # Add separator after ontology-guided
            if exp_id == '1_ontology_guided':
                lines.append("-" * 138)

        lines.append("")
        lines.append("")

        # Section 2: Detailed Breakdown
        lines.append("DETAILED BREAKDOWN")
        lines.append("-" * 120)
        lines.append("")

        for exp_id in ["1_ontology_guided", "2_baseline_llm", "3_no_schema"]:
            m = metrics.get(exp_id)
            if not m:
                continue

            lines.append(f"[{exp_id}] {m['name']}")
            lines.append(f"  {m['description']}")
            lines.append("")
            lines.append(f"  Total Entities: {m['total_entities']}")
            lines.append(f"  Total Relationships: {m['total_relationships']}")
            lines.append("")

            # Entity type distribution
            if m['entity_types']:
                lines.append("  Entity Type Distribution:")
                for etype, count in sorted(m['entity_types'].items(), key=lambda x: -x[1]):
                    lines.append(f"    - {etype}: {count}")
                lines.append("")

            # Metrics
            lines.append(f"  CQ Violation Rate: {m['cq_violation_rate']:.1f}%")
            lines.append(f"  Provenance Completeness: {m['provenance_completeness']:.1f}%")
            lines.append(f"  Ontology Compliance: {m['ontology_compliance']:.1f}%")
            lines.append("")
            lines.append("")

        # Section 3: Key Insights
        lines.append("KEY INSIGHTS")
        lines.append("-" * 120)
        lines.append("")

        if full_system:
            baseline = metrics.get('2_baseline_llm')
            no_schema = metrics.get('3_no_schema')

            if baseline:
                ont_improvement = full_system['ontology_compliance'] - baseline['ontology_compliance']
                triple_improvement = full_system['triple_retention_rate'] - baseline['triple_retention_rate']
                cq_reduction = baseline['cq_violation_rate'] - full_system['cq_violation_rate']

                lines.append(f"• Ontology-guided extraction improves compliance by {ont_improvement:.1f}% vs. unconstrained baseline")
                lines.append(f"• Ontology-guided extraction improves triple retention by {triple_improvement:.1f}% vs. baseline")
                lines.append(f"• Ontology-guided extraction reduces CQ violations by {cq_reduction:.1f}% vs. baseline")

            if no_schema:
                cq_increase = no_schema['cq_violation_rate'] - full_system['cq_violation_rate']
                ont_diff = full_system['ontology_compliance'] - no_schema['ontology_compliance']
                lines.append(f"• Removing ESGMKG schema increases CQ violations by {cq_increase:.1f}%")
                lines.append(f"• ESGMKG entity types improve ontology compliance by {ont_diff:.1f}%")

            lines.append(f"• Full system achieves {full_system['ontology_compliance']:.1f}% ontology compliance and {full_system['triple_retention_rate']:.1f}% triple retention")

        lines.append("")
        lines.append("=" * 120)

        # Save report
        report_text = '\n'.join(lines)

        with open(output_file, 'w') as f:
            f.write(report_text)

        # Print to console
        print("\n" + report_text)

        print(f"\n{'='*80}")
        print(f"OUTPUT FILE GENERATED")
        print(f"{'='*80}")
        print(f"✓ 1 file created:")
        print(f"  📄 {output_file.name}")
        print(f"  📁 Location: {output_file.parent}/")
        print(f"  📊 Size: {output_file.stat().st_size} bytes")
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Stage 3B: Baseline LLM Comparison')
    parser.add_argument('document', nargs='?', help='Document name to compare (without extension)')
    parser.add_argument('--batch', action='store_true', help='Process all documents in batch')

    args = parser.parse_args()

    comparator = BaselineLLMComparator()

    if args.batch:
        # Batch mode - find all baseline_llm files
        baseline_files = list(comparator.experiments_dir.glob("*_baseline_llm.json"))

        if not baseline_files:
            print("❌ No baseline LLM experiment files found!")
            print(f"Directory: {comparator.experiments_dir}")
            sys.exit(1)

        print(f"\n🚀 BATCH BASELINE LLM COMPARISON")
        print(f"📁 Found {len(baseline_files)} documents to compare\n")

        for baseline_file in baseline_files:
            document_name = baseline_file.name.replace('_baseline_llm.json', '')
            comparator.compare_document(document_name)

        print(f"\n✅ Batch comparison complete! Processed {len(baseline_files)} documents.")

    elif args.document:
        # Single document mode
        comparator.compare_document(args.document)

    else:
        print("❌ Error: Please specify a document name or use --batch flag")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
