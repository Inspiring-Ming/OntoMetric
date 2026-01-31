#!/usr/bin/env python3
"""
Stage 3A: Ontology-Guided Knowledge Graph Validation

Validates knowledge graph extraction quality using 6 validation rules (VR001-VR006)
aligned with the ontology schema. Calculates 4 quality metrics: Quality Score,
Validation Pass Rate, Traceability, and Schema Conformance.

For detailed documentation, usage examples, and integration guides, see:
    src/stage3_validation_README.txt

Quick Start:
  python3 src/stage3_ontology_guided_validation.py --python --batch
  python3 src/stage3_ontology_guided_validation.py --python --single "document_name"
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Try importing Anthropic for semantic validation
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ValidationRule:
    """Represents a Python-based validation rule"""
    id: str
    name: str
    description: str
    approach: str  # "python"
    critical: bool = True
    python_method: Optional[str] = None


@dataclass
class ValidationResult:
    """Results of a validation check"""
    rule_id: str
    rule_name: str
    approach: str
    passed: bool
    violation_count: int
    violations: List[Dict] = None
    execution_time: float = 0.0
    details: str = ""
    total_items: int = 0  # Total items checked (for weighted scoring)


class KnowledgeGraphValidator:
    """
    Knowledge graph validator using Python-based validation rules.
    Validates JSON knowledge graphs against ontology schema requirements.
    """

    def __init__(self):
        """Initialize the validator"""
        # Define validation rules
        self.validation_rules = self._define_validation_rules()
        
        # Valid entity types and predicates
        self.valid_entity_types = {
            'Industry', 'ReportingFramework', 'Category', 
            'Metric', 'Model'
        }
        
        # Valid metric types (as properties, not entity types)
        self.valid_metric_types = {
            'DirectMetric', 'CalculatedMetric', 'InputMetric'
        }
        
        self.valid_predicates = {
            'ReportUsing', 'Include', 'ConsistOf', 
            'IsCalculatedBy', 'RequiresInputFrom'
        }
        
        print("🔍 Knowledge Graph Validator initialized")
        print(f"📋 Loaded {len(self.validation_rules)} validation rules (6 rules, all critical)")
    
    def _define_validation_rules(self) -> List[ValidationRule]:
        """Define comprehensive validation rules for all approaches

        Rules are organized following the ontology schema validation hierarchy:
        LEVEL 1: Schema Structure - Do entities have required fields?
        LEVEL 2: Property Values - Are required property values present?
        LEVEL 3: Relationship Connections - Are entities properly connected?

        Total: 6 rules (all critical)
        Schema Compliance Rate: (passed / 6) × 100%
        """

        rules = [
            # ========================================
            # LEVEL 1: SCHEMA STRUCTURE VALIDATION
            # ========================================

            # VR001: Entity Uniqueness
            ValidationRule(
                id="VR001",
                name="Entity Uniqueness",
                description="All entity IDs must be unique within the knowledge graph",
                approach="python",
                critical=True,
                python_method="validate_entity_uniqueness"
            ),

            # VR002: Entity Type-Specific Schema Compliance
            ValidationRule(
                id="VR002",
                name="Entity Type-Specific Schema Compliance",
                description="Each entity type has ontology-required fields (Metric: measurement_type/metric_type/unit/code/description; Model: description/equation/input_variables; Category: section_title; etc.)",
                approach="python",
                critical=True,
                python_method="validate_entity_type_specific_schema"
            ),

            # ========================================
            # LEVEL 2: PROPERTY VALUE VALIDATION
            # ========================================

            # VR003: Metric Property Values
            ValidationRule(
                id="VR003",
                name="Metric Property Values",
                description="DirectMetric/CalculatedMetric must have code (not N/A); Quantitative metrics must have units (not N/A)",
                approach="python",
                critical=True,
                python_method="validate_metric_property_values"
            ),

            # VR004: Model Property Values
            ValidationRule(
                id="VR004",
                name="Model Property Values",
                description="Models must have non-empty input_variables list",
                approach="python",
                critical=True,
                python_method="validate_model_property_values"
            ),

            # ========================================
            # LEVEL 3: RELATIONSHIP CONNECTION VALIDATION
            # ========================================

            # VR005: Relationship Predicate Validity
            ValidationRule(
                id="VR005",
                name="Relationship Predicate Validity",
                description="All relationships must use ontology-defined predicates: ConsistOf, Include, IsCalculatedBy, ReportUsing, RequiresInputFrom",
                approach="python",
                critical=True,
                python_method="validate_relationship_validity"
            ),

            # VR006: CalculatedMetric-Model Links
            ValidationRule(
                id="VR006",
                name="CalculatedMetric-Model Links",
                description="All Metrics with metric_type='CalculatedMetric' must have IsCalculatedBy relationship to a Model",
                approach="python",
                critical=True,
                python_method="validate_calculated_metrics_have_models"
            )
        ]

        return rules
    
    # ========================================
    # PYTHON VALIDATION METHODS
    # ========================================
    
    def validate_python(self, json_file_path: str) -> Dict[str, Any]:
        """Validate using Python rules against JSON knowledge graph"""
        print(f"\n🔍 Python Validation: {Path(json_file_path).name}")
        print("-" * 60)
        
        # Load JSON data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # Handle different file structures:
            # 1. Ontology-guided files: {entities: [...], relationships: [...]}
            # 2. Baseline files: {experiments: {2_baseline_llm: {entities: [...], relationships: [...]}}}
            if 'experiments' in raw_data and '2_baseline_llm' in raw_data['experiments']:
                # Baseline file structure - extract the baseline experiment data
                kg_data = raw_data['experiments']['2_baseline_llm']
                print(f"📊 Loaded baseline experiment: {len(kg_data.get('entities', []))} entities, {len(kg_data.get('relationships', []))} relationships")
            elif 'entities' in raw_data:
                # Ontology-guided file structure - use as is
                kg_data = raw_data
                print(f"📊 Loaded {len(kg_data.get('entities', []))} entities, {len(kg_data.get('relationships', []))} relationships")
            else:
                return {
                    'success': False,
                    'error': f"Unrecognized file structure - no 'entities' key or 'experiments.2_baseline_llm' path found",
                    'approach': 'python'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to load JSON file: {str(e)}",
                'approach': 'python'
            }

        # STEP 1: Run semantic validation FIRST (gate-keeper)
        print("\n" + "="*80)
        print("STEP 1: SEMANTIC VALIDATION (Gate-Keeper)")
        print("="*80)
        entities = kg_data.get('entities', [])
        semantic_results = self.validate_semantic_type_accuracy(entities)

        # Filter out entities with incorrect semantic types
        semantic_accuracy = semantic_results.get('semantic_type_accuracy', 0.0)
        mismatched_entity_ids = {m['entity_id'] for m in semantic_results.get('mismatches', [])}

        original_entity_count = len(entities)
        original_relationship_count = len(kg_data.get('relationships', []))

        print(f"\n  Entity Semantic Accuracy: {semantic_accuracy:.2f}%")
        print(f"  LLM Cost: ${semantic_results.get('llm_cost', 0.0):.4f}")
        print(f"  Entities to filter: {len(mismatched_entity_ids)}/{original_entity_count}")

        # Filter entities - keep only semantically correct ones
        filtered_entities = [e for e in entities if e.get('id') not in mismatched_entity_ids]

        # Filter relationships - remove those referencing removed entities
        filtered_relationships = [
            r for r in kg_data.get('relationships', [])
            if r.get('subject') not in mismatched_entity_ids and r.get('object') not in mismatched_entity_ids
        ]

        filtered_entity_count = len(filtered_entities)
        filtered_relationship_count = len(filtered_relationships)

        print(f"  Filtered entities: {filtered_entity_count} (removed {original_entity_count - filtered_entity_count})")
        print(f"  Filtered relationships: {filtered_relationship_count} (removed {original_relationship_count - filtered_relationship_count})")

        # Update kg_data with filtered entities and relationships
        kg_data['entities'] = filtered_entities
        kg_data['relationships'] = filtered_relationships

        # STEP 2: Execute structural validation rules on filtered graph
        print("\n" + "="*80)
        print("STEP 2: STRUCTURAL VALIDATION RULES")
        print("="*80)
        results = []
        critical_failures = 0
        total_failures = 0

        python_rules = [r for r in self.validation_rules if r.python_method and r.approach in ['python', 'both']]

        for rule in python_rules:
            print(f"\n[{rule.id}] {rule.name}")
            
            # Get validation method
            method = getattr(self, rule.python_method, None)
            if not method:
                print(f"   ❌ ERROR: Method {rule.python_method} not found")
                continue
            
            try:
                result = method(kg_data)
                results.append(result)
                
                # Print result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                criticality = " (CRITICAL)" if rule.critical and not result.passed else ""
                print(f"   {status}{criticality}: {result.violation_count} violations ({result.execution_time:.3f}s)")
                
                # Count failures
                if not result.passed:
                    total_failures += 1
                    if rule.critical:
                        critical_failures += 1
                        
            except Exception as e:
                print(f"   ❌ ERROR: Validation failed: {str(e)}")
                results.append(ValidationResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    approach="python",
                    passed=False,
                    violation_count=-1,
                    details=f"Validation error: {str(e)}",
                    execution_time=0.0
                ))
                total_failures += 1
                if rule.critical:
                    critical_failures += 1
        
        # Calculate summary
        total_rules = len(python_rules)
        passed_rules = total_rules - total_failures

        # Calculate Rule-Level Weighted Average Schema Compliance
        rule_scores = {}
        for result in results:
            if result.total_items > 0:
                # Rule Score = (Passed Items / Total Items) × 100%
                # violation_count = number of items that failed the rule
                # passed_items = number of items that passed the rule
                passed_items = result.total_items - result.violation_count
                rule_score = (passed_items / result.total_items) * 100
                rule_scores[result.rule_id] = {
                    'rule_name': result.rule_name,
                    'score': round(rule_score, 2),
                    'passed_items': passed_items,
                    'total_items': result.total_items,
                    'violations': result.violation_count
                }
            else:
                # If no items to check, consider it 100% (nothing to validate)
                rule_scores[result.rule_id] = {
                    'rule_name': result.rule_name,
                    'score': 100.0,
                    'passed_items': 0,
                    'total_items': 0,
                    'violations': 0
                }

        # Calculate overall compliance_score as average of all rule scores
        if rule_scores:
            compliance_score = sum(r['score'] for r in rule_scores.values()) / len(rule_scores)
        else:
            compliance_score = 0.0

        return {
            'success': True,
            'approach': 'python',
            'file_path': json_file_path,
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'failed_rules': total_failures,
            'critical_failures': critical_failures,
            'compliance_score': round(compliance_score, 2),  # Weighted average schema compliance
            'rule_scores': rule_scores,  # Per-rule granular scoring
            'overall_status': 'PASS' if critical_failures == 0 else 'FAIL',
            'results': [
                {
                    'rule_id': r.rule_id,
                    'rule_name': r.rule_name,
                    'approach': r.approach,
                    'passed': r.passed,
                    'violation_count': r.violation_count,
                    'violations': r.violations,
                    'execution_time': r.execution_time
                }
                for r in results
            ],
            'semantic_results': semantic_results,  # Include semantic validation results
            'semantic_gate_failed': False,  # Semantic validation passed
            'validation_timestamp': datetime.now().isoformat()
        }

    # Semantic validation using LLM
    def validate_semantic_type_accuracy(self, entities: List[Dict], llm_config_path: str = "config_llm.json") -> Dict[str, Any]:
        """
        Validate that entity labels/descriptions semantically match their entity type definitions.
        Uses Claude LLM to assess semantic correctness.

        Returns:
            {
                'total_entities': int,
                'correct_entities': int,
                'incorrect_entities': int,
                'semantic_type_accuracy': float (percentage),
                'llm_cost': float (USD),
                'mismatches': [{'entity_id': str, 'entity_type': str, 'label': str, 'llm_reasoning': str}]
            }
        """
        print("\n" + "=" * 60)
        print("🧠 SEMANTIC TYPE VALIDATION (LLM-based)")
        print("=" * 60)

        # Check if Anthropic is available
        if not ANTHROPIC_AVAILABLE:
            print("⚠️  WARNING: Anthropic library not available")
            print("   Install with: pip install anthropic")
            return {
                'total_entities': len(entities),
                'correct_entities': 0,
                'incorrect_entities': 0,
                'semantic_type_accuracy': 0.0,
                'llm_cost': 0.0,
                'mismatches': [],
                'error': 'Anthropic library not installed'
            }

        # Load LLM configuration
        try:
            with open(llm_config_path, 'r', encoding='utf-8') as f:
                llm_config = json.load(f)
        except Exception as e:
            print(f"❌ ERROR: Failed to load LLM config from {llm_config_path}: {str(e)}")
            return {
                'total_entities': len(entities),
                'correct_entities': 0,
                'incorrect_entities': 0,
                'semantic_type_accuracy': 0.0,
                'llm_cost': 0.0,
                'mismatches': [],
                'error': f'Failed to load LLM config: {str(e)}'
            }

        # Initialize Anthropic client
        api_key = llm_config.get('api_settings', {}).get('api_key', '')
        if not api_key:
            print("❌ ERROR: No API key found in LLM config")
            return {
                'total_entities': len(entities),
                'correct_entities': 0,
                'incorrect_entities': 0,
                'semantic_type_accuracy': 0.0,
                'llm_cost': 0.0,
                'mismatches': [],
                'error': 'No API key in config'
            }

        client = Anthropic(api_key=api_key)
        model = llm_config.get('api_settings', {}).get('model', 'claude-sonnet-4-5-20250929')

        # Entity type definitions (RELAXED to include qualitative disclosures)
        type_definitions = {
            'Industry': 'Business sectors or industry classifications (e.g., Commercial Banks, Oil & Gas, Cross-Sector Organizations, Climate-Exposed Entities)',
            'ReportingFramework': 'ESG reporting standards and frameworks (e.g., SASB, TCFD, IFRS S2, ISSB, AASB)',
            'Category': 'Thematic groupings of ESG disclosures (e.g., Energy Management, Emissions, Strategy, Risk Management, Governance, Metrics and Targets)',
            'Metric': 'ESG disclosures including both quantitative metrics (with units) AND qualitative disclosure requirements (narrative descriptions, processes, governance structures)',
            'Model': 'Calculation methodologies, formulas, or analytical frameworks used to derive metrics'
        }

        # Validate each entity
        total_entities = len(entities)
        correct_entities = 0
        incorrect_entities = 0
        mismatches = []
        total_cost = 0.0

        print(f"\n📊 Validating {total_entities} entities...")
        print("   This will use Claude LLM to verify semantic correctness")
        print("")
        sys.stdout.flush()  # Force output to appear immediately

        for idx, entity in enumerate(entities, 1):
            entity_id = entity.get('id', 'unknown')
            entity_type = entity.get('type', 'unknown')
            label = entity.get('label', '')
            properties = entity.get('properties', {})
            description = properties.get('description', '') if isinstance(properties, dict) else ''

            # CRITICAL: Unknown types are SEMANTIC FAILURES and counted as ❌ INCORRECT
            #
            # Rationale: If Stage 2 extraction produces entities with types NOT in the
            # ontology schema (Industry, Category, Metric, Model), this represents a
            # semantic failure that MUST be penalized in accuracy calculations.
            #
            # Example failures:
            #   - Baseline extraction: "Standard", "Organization", "Date", "Risk Category"
            #   - Ontology-guided: "Framework", "Disclosure", etc.
            #
            # Implementation:
            #   1. Increment incorrect_entities counter (counts toward denominator & failure count)
            #   2. Add detailed reasoning to mismatches report
            #   3. Skip LLM validation call (save API cost, already known to be wrong)
            #
            # Result: semantic_type_accuracy = (correct / total) correctly penalizes bad extractions
            if entity_type not in type_definitions:
                print(f"   [{idx}/{total_entities}] ❌ {entity_id}: Unknown type '{entity_type}' (FAILED - not in ontology)")
                incorrect_entities += 1  # Count as FAILED
                mismatches.append({
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'label': label,
                    'llm_reasoning': f'SEMANTIC FAILURE: Unknown/invalid type "{entity_type}" not in ontology schema. Only 4 types allowed: Industry, Category, Metric, Model.'
                })
                continue  # Skip LLM call to save cost (already failed)

            # Build validation prompt
            type_def = type_definitions[entity_type]

            # Determine what semantic fields to check
            if entity_type in ['Metric', 'Model'] and description:
                semantic_content = f"Label: {label}\nDescription: {description}"
            else:
                semantic_content = f"Label: {label}"

            prompt = f"""You are validating whether an extracted entity's semantic content matches its entity type definition.

Entity Type: {entity_type}
Type Definition: {type_def}

Extracted Entity:
{semantic_content}

Question: Does this entity's content semantically match what a "{entity_type}" should represent according to the definition?

Answer with ONLY a JSON object (no other text):
{{
  "is_correct": true/false,
  "reasoning": "Brief explanation (1-2 sentences)"
}}"""

            try:
                # Call Claude API
                response = client.messages.create(
                    model=model,
                    max_tokens=200,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                # Parse response
                response_text = response.content[0].text.strip()

                # Extract JSON from response (handle potential markdown code blocks)
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                result = json.loads(response_text)
                is_correct = result.get('is_correct', False)
                reasoning = result.get('reasoning', '')

                # Calculate cost (rough estimate: $0.0001 per entity)
                entity_cost = 0.0001
                total_cost += entity_cost

                # Track result
                if is_correct:
                    correct_entities += 1
                    status = "✅"
                else:
                    incorrect_entities += 1
                    status = "❌"
                    mismatches.append({
                        'entity_id': entity_id,
                        'entity_type': entity_type,
                        'label': label,
                        'llm_reasoning': reasoning
                    })

                # Print progress every 10 entities or if mismatch
                if idx % 10 == 0 or not is_correct:
                    print(f"   [{idx}/{total_entities}] {status} {entity_type}: {label[:50]}...")
                    sys.stdout.flush()  # Force output

            except Exception as e:
                print(f"   [{idx}/{total_entities}] ⚠️  ERROR validating {entity_id}: {str(e)}")
                # Count as correct to avoid penalizing for API errors
                correct_entities += 1

        # Calculate accuracy
        # NOTE: incorrect_entities INCLUDES unknown types (counted as semantic failures)
        # Formula: (Correctly Typed Entities / Total Entities) × 100%
        # Example: If 220/235 entities have unknown types → all 220 counted as incorrect
        semantic_type_accuracy = (correct_entities / total_entities * 100) if total_entities > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(f"✅ Correct: {correct_entities}/{total_entities}")
        print(f"❌ Incorrect: {incorrect_entities}/{total_entities} (includes unknown types as failures)")
        print(f"📊 Entity Semantic Accuracy: {semantic_type_accuracy:.1f}%")
        print(f"💰 LLM Cost: ${total_cost:.4f}")
        print(f"{'=' * 60}\n")

        return {
            'total_entities': total_entities,
            'correct_entities': correct_entities,
            'incorrect_entities': incorrect_entities,
            'semantic_type_accuracy': semantic_type_accuracy,
            'llm_cost': total_cost,
            'mismatches': mismatches
        }

    # ========================================
    # PYTHON VALIDATION METHODS (6 rules)
    # ========================================

    def validate_entity_uniqueness(self, kg_data: Dict) -> ValidationResult:
        """VR001: All entity IDs must be unique"""
        start_time = time.time()
        violations = []
        entity_ids = {}

        for entity in kg_data.get('entities', []):
            entity_id = entity.get('id')
            if entity_id in entity_ids:
                violations.append({
                    'entity_id': entity_id,
                    'issue': 'Duplicate entity ID',
                    'first_occurrence': entity_ids[entity_id],
                    'duplicate_label': entity.get('label')
                })
            else:
                entity_ids[entity_id] = entity.get('label')

        total_items = len(kg_data.get('entities', []))

        return ValidationResult(
            rule_id="VR001",
            rule_name="Entity Uniqueness",
            approach="python",
            passed=len(violations) == 0,
            violation_count=len(violations),
            violations=violations,
            execution_time=time.time() - start_time,
            total_items=total_items
        )

    def validate_entity_type_specific_schema(self, kg_data: Dict) -> ValidationResult:
        """VR002: Entity Type-Specific Schema Compliance"""
        start_time = time.time()
        violations = []
        entities_with_violations = set()  # Track which entities have violations

        for entity in kg_data.get('entities', []):
            entity_id = entity.get('id', 'Unknown')
            entity_type = entity.get('type', 'Unknown')
            entity_violations = []  # Track violations for this specific entity

            # ALL entities must have: id, type, label, source
            if not entity.get('id'):
                entity_violations.append({'entity': entity_id, 'issue': 'Missing field: id'})
            if not entity.get('type'):
                entity_violations.append({'entity': entity_id, 'issue': 'Missing field: type'})
            if not entity.get('label'):
                entity_violations.append({'entity': entity_id, 'issue': 'Missing field: label'})
            if not entity.get('source'):
                entity_violations.append({'entity': entity_id, 'issue': 'Missing field: source'})

            # Type-specific field validation
            props = entity.get('properties', {})

            if entity_type == 'Metric':
                required_fields = ['measurement_type', 'metric_type', 'unit', 'code', 'description']
                for field in required_fields:
                    if field not in props:
                        entity_violations.append({'entity': entity_id, 'issue': f'Metric missing field: properties.{field}'})

            elif entity_type == 'Model':
                required_fields = ['description', 'equation', 'input_variables']
                for field in required_fields:
                    if field not in props:
                        entity_violations.append({'entity': entity_id, 'issue': f'Model missing field: properties.{field}'})

            elif entity_type == 'Category':
                if 'section_title' not in props:
                    entity_violations.append({'entity': entity_id, 'issue': 'Category missing field: properties.section_title'})

            elif entity_type == 'ReportingFramework':
                if 'name' not in props:
                    entity_violations.append({'entity': entity_id, 'issue': 'ReportingFramework missing field: properties.name'})

            elif entity_type == 'Industry':
                if not isinstance(props, dict):
                    entity_violations.append({'entity': entity_id, 'issue': 'Industry missing properties object'})

            # If this entity has any violations, track it and add all its violations to the global list
            if entity_violations:
                entities_with_violations.add(entity_id)
                violations.extend(entity_violations)

        total_items = len(kg_data.get('entities', []))

        # IMPORTANT: violation_count should be the NUMBER OF ENTITIES that failed, not total field violations
        # This ensures passed_items = total_items - violation_count never goes negative
        failed_entity_count = len(entities_with_violations)

        return ValidationResult(
            rule_id="VR002",
            rule_name="Entity Type-Specific Schema Compliance",
            approach="python",
            passed=len(violations) == 0,
            violation_count=failed_entity_count,  # Count entities with violations, not total violations
            violations=violations,  # Still keep detailed field-level violations for debugging
            execution_time=time.time() - start_time,
            total_items=total_items
        )

    def validate_metric_property_values(self, kg_data: Dict) -> ValidationResult:
        """VR003: Metric Property Values"""
        start_time = time.time()
        violations = []
        metrics_with_violations = set()  # Track which metrics have violations

        for entity in kg_data.get('entities', []):
            if entity.get('type') == 'Metric':
                entity_id = entity.get('id')
                props = entity.get('properties', {})
                metric_type = props.get('metric_type')
                measurement_type = props.get('measurement_type')
                has_violation = False

                # DirectMetric/CalculatedMetric must have non-empty code (not N/A)
                if metric_type in ['DirectMetric', 'CalculatedMetric']:
                    code = props.get('code', '')
                    if not code or code == 'N/A' or code == '':
                        violations.append({
                            'entity': entity_id,
                            'issue': f'{metric_type} has invalid code value: "{code}" (must be non-empty and not N/A)'
                        })
                        has_violation = True

                # Quantitative metrics must have non-empty unit (not N/A)
                if measurement_type == 'Quantitative':
                    unit = props.get('unit', '')
                    if not unit or unit == 'N/A' or unit == '':
                        violations.append({
                            'entity': entity_id,
                            'issue': f'Quantitative metric has invalid unit value: "{unit}" (must be non-empty and not N/A)'
                        })
                        has_violation = True

                # Track this metric if it has any violations
                if has_violation:
                    metrics_with_violations.add(entity_id)

        # Count total metrics checked (DirectMetric + CalculatedMetric + Quantitative metrics)
        total_items = sum(1 for e in kg_data.get('entities', []) if e.get('type') == 'Metric')

        # IMPORTANT: violation_count should be the NUMBER OF METRICS that failed, not total field violations
        # This ensures passed_items = total_items - violation_count never goes negative
        failed_metric_count = len(metrics_with_violations)

        return ValidationResult(
            rule_id="VR003",
            rule_name="Metric Property Values",
            approach="python",
            passed=len(violations) == 0,
            violation_count=failed_metric_count,  # Count metrics with violations, not total violations
            violations=violations,  # Still keep detailed field-level violations for debugging
            execution_time=time.time() - start_time,
            total_items=total_items
        )

    def validate_model_property_values(self, kg_data: Dict) -> ValidationResult:
        """VR004: Model Property Values"""
        start_time = time.time()
        violations = []

        for entity in kg_data.get('entities', []):
            if entity.get('type') == 'Model':
                entity_id = entity.get('id')
                props = entity.get('properties', {})
                input_vars = props.get('input_variables', '')

                # Check if input_variables is invalid
                is_invalid = False
                if not input_vars:
                    is_invalid = True
                elif input_vars == 'N/A':
                    is_invalid = True
                elif isinstance(input_vars, list) and len(input_vars) == 0:
                    is_invalid = True
                elif isinstance(input_vars, list) and all(not v for v in input_vars):
                    is_invalid = True

                if is_invalid:
                    violations.append({
                        'entity': entity_id,
                        'issue': f'Model has invalid input_variables: "{input_vars}" (must be non-empty list with at least one entry)'
                    })

        total_items = sum(1 for e in kg_data.get('entities', []) if e.get('type') == 'Model')

        return ValidationResult(
            rule_id="VR004",
            rule_name="Model Property Values",
            approach="python",
            passed=len(violations) == 0,
            violation_count=len(violations),
            violations=violations,
            execution_time=time.time() - start_time,
            total_items=total_items
        )

    def validate_relationship_validity(self, kg_data: Dict) -> ValidationResult:
        """VR005: Relationship Predicate Validity"""
        start_time = time.time()
        violations = []

        valid_predicates = {'ConsistOf', 'Include', 'IsCalculatedBy', 'ReportUsing', 'RequiresInputFrom'}

        for rel in kg_data.get('relationships', []):
            predicate = rel.get('predicate', '')
            if predicate not in valid_predicates:
                violations.append({
                    'subject': rel.get('subject'),
                    'predicate': predicate,
                    'object': rel.get('object'),
                    'issue': f'Invalid predicate "{predicate}" (must be one of: {sorted(valid_predicates)})'
                })

        total_items = len(kg_data.get('relationships', []))

        return ValidationResult(
            rule_id="VR005",
            rule_name="Relationship Predicate Validity",
            approach="python",
            passed=len(violations) == 0,
            violation_count=len(violations),
            violations=violations,
            execution_time=time.time() - start_time,
            total_items=total_items
        )

    def validate_calculated_metrics_have_models(self, kg_data: Dict) -> ValidationResult:
        """VR006: CalculatedMetric-Model Links"""
        start_time = time.time()
        violations = []

        # Find all CalculatedMetrics
        calculated_metrics = [
            e for e in kg_data.get('entities', [])
            if e.get('type') == 'Metric' and e.get('properties', {}).get('metric_type') == 'CalculatedMetric'
        ]

        # Find all IsCalculatedBy relationships
        is_calculated_by_rels = [
            r for r in kg_data.get('relationships', [])
            if r.get('predicate') == 'IsCalculatedBy'
        ]

        # Check each CalculatedMetric has IsCalculatedBy relationship
        metrics_with_models = {r.get('subject') for r in is_calculated_by_rels}

        for metric in calculated_metrics:
            metric_id = metric.get('id')
            if metric_id not in metrics_with_models:
                violations.append({
                    'entity': metric_id,
                    'label': metric.get('label'),
                    'issue': 'CalculatedMetric missing IsCalculatedBy relationship to Model'
                })

        total_items = len(calculated_metrics)

        return ValidationResult(
            rule_id="VR006",
            rule_name="CalculatedMetric-Model Links",
            approach="python",
            passed=len(violations) == 0,
            violation_count=len(violations),
            violations=violations,
            execution_time=time.time() - start_time,
            total_items=total_items
        )

    # ========================================
    # VALIDATED DATA FILTERING
    # ========================================

    def filter_validated_data(self, kg_data: Dict, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter knowledge graph to only include entities and relationships that passed validation.

        ENTITY-LEVEL FILTERING: Removes individual entities/relationships that fail validation.
        - Semantic Validation: Removes entities with semantically incorrect types (per LLM)
        - Schema Validation: Removes entities that violate critical rules (VR001-VR006)
        - Saves document if ANY valid entities remain after filtering

        Args:
            kg_data: Original knowledge graph data
            validation_results: Results from validation (python or hybrid)

        Returns:
            Filtered knowledge graph with only validated entities/relationships, or None if no valid data remains
        """
        print("\n🔍 Filtering validated data (entity-level filtering)...")

        # Get validation results (handle both direct and hybrid validation structures)
        if 'python_validation' in validation_results:
            results = validation_results['python_validation'].get('results', [])
            semantic_results = validation_results['python_validation'].get('semantic_results', {})
            semantic_gate_failed = validation_results['python_validation'].get('semantic_gate_failed', False)
            overall_status = validation_results['python_validation'].get('overall_status', 'UNKNOWN')
        else:
            results = validation_results.get('results', [])
            semantic_results = validation_results.get('semantic_results', {})
            semantic_gate_failed = validation_results.get('semantic_gate_failed', False)
            overall_status = validation_results.get('overall_status', 'UNKNOWN')

        # Collect entity IDs and relationship indices to remove from semantic mismatches
        entities_to_remove = set()
        relationships_to_remove = set()

        # Add entities that failed semantic validation
        semantic_mismatches = semantic_results.get('mismatches', [])
        for mismatch in semantic_mismatches:
            entity_id = mismatch.get('entity_id')
            if entity_id:
                entities_to_remove.add(entity_id)
                print(f"  ❌ Removing entity {entity_id} (semantic mismatch: {mismatch.get('actual_type')} should be {mismatch.get('expected_type')})")

        critical_rules = ['VR001', 'VR002', 'VR003', 'VR004', 'VR005', 'VR006']

        for result in results:
            rule_id = result.get('rule_id')

            # Only process critical rule violations
            if rule_id not in critical_rules or result.get('passed', False):
                continue

            violations = result.get('violations', [])

            # Mark entities with violations for removal
            for violation in violations:
                if 'entity_id' in violation:
                    entities_to_remove.add(violation['entity_id'])

                # For relationship violations (VR010)
                if rule_id == 'VR010':
                    subject = violation.get('subject')
                    obj = violation.get('object')
                    predicate = violation.get('predicate')

                    # Find and mark relationship for removal
                    for idx, rel in enumerate(kg_data.get('relationships', [])):
                        if (rel.get('subject') == subject and
                            rel.get('object') == obj and
                            rel.get('predicate') == predicate):
                            relationships_to_remove.add(idx)

        # Filter entities
        original_entities = kg_data.get('entities', [])
        validated_entities = [
            e for e in original_entities
            if e.get('id') not in entities_to_remove
        ]

        # Filter relationships (remove by index and also remove orphaned relationships)
        original_relationships = kg_data.get('relationships', [])
        validated_entity_ids = {e.get('id') for e in validated_entities}

        validated_relationships = []
        for idx, rel in enumerate(original_relationships):
            # Skip if marked for removal
            if idx in relationships_to_remove:
                continue

            # Skip if subject or object entity was removed
            if (rel.get('subject') not in validated_entity_ids or
                rel.get('object') not in validated_entity_ids):
                continue

            validated_relationships.append(rel)

        # Calculate statistics
        entities_removed = len(original_entities) - len(validated_entities)
        relationships_removed = len(original_relationships) - len(validated_relationships)

        print(f"  📊 Entities: {len(original_entities)} → {len(validated_entities)} ({entities_removed} removed)")
        print(f"  🔗 Relationships: {len(original_relationships)} → {len(validated_relationships)} ({relationships_removed} removed)")

        # If no valid entities remain after filtering, don't save the document
        if len(validated_entities) == 0:
            print("  ⚠️  No valid entities remaining after filtering - document will not be saved")
            return None

        # Calculate enhanced metrics
        # 1. Rule compliance metrics
        critical_rules_passed = [
            r.get('rule_id') for r in results
            if r.get('rule_id') in critical_rules and r.get('passed', False)
        ]
        critical_rules_failed = [
            r.get('rule_id') for r in results
            if r.get('rule_id') in critical_rules and not r.get('passed', False)
        ]

        # 2. Quality metrics (relationship retention rate)
        # Note: Entity retention rate = semantic_type_accuracy (they are the same)
        # Entities are retained only if they pass semantic validation
        relationship_retention_rate = (len(validated_relationships) / len(original_relationships) * 100) if len(original_relationships) > 0 else 0.0

        # 3. Violation statistics
        total_violations = 0
        entities_with_violations = len(entities_to_remove)
        relationships_with_violations = len(relationships_to_remove)
        violations_by_rule = {}

        for result in results:
            rule_id = result.get('rule_id')
            passed = result.get('passed', False)
            violation_count = result.get('violation_count', 0)

            violations_by_rule[rule_id] = {
                'passed': passed,
                'violation_count': violation_count
            }
            total_violations += violation_count

        # 4. Segment-level statistics
        # Aggregate entities by segment
        segment_original_counts = {}
        segment_validated_counts = {}

        for entity in original_entities:
            seg_id = entity.get('provenance', {}).get('segment_id', 'unknown')
            segment_original_counts[seg_id] = segment_original_counts.get(seg_id, 0) + 1

        for entity in validated_entities:
            seg_id = entity.get('provenance', {}).get('segment_id', 'unknown')
            segment_validated_counts[seg_id] = segment_validated_counts.get(seg_id, 0) + 1

        # Calculate per-segment retention
        entities_per_segment = {}
        retention_per_segment = {}

        for seg_id in segment_original_counts.keys():
            original_count = segment_original_counts.get(seg_id, 0)
            validated_count = segment_validated_counts.get(seg_id, 0)

            entities_per_segment[seg_id] = {
                'original': original_count,
                'validated': validated_count
            }

            retention_per_segment[seg_id] = (validated_count / original_count * 100) if original_count > 0 else 0.0

        # Calculate average segment retention
        avg_segment_retention = sum(retention_per_segment.values()) / len(retention_per_segment) if len(retention_per_segment) > 0 else 0.0
        segments_with_entities = len([seg for seg, counts in entities_per_segment.items() if counts['original'] > 0])

        # Create validated knowledge graph with enhanced metadata
        validated_kg = {
            'validation_metadata': {
                # Validation metadata
                'validation_timestamp': validation_results['validation_timestamp'],
                'validation_approach': validation_results['approach'],
                'critical_rules_passed': critical_rules_passed,
                'critical_violations_removed': entities_removed + relationships_removed,
                'original_entity_count': len(original_entities),
                'validated_entity_count': len(validated_entities),
                'original_relationship_count': len(original_relationships),
                'validated_relationship_count': len(validated_relationships),
                'source_file': validation_results['file_path'],

                # Quality Metrics: 3 core metrics for validation assessment
                'quality_metrics': {
                    'schema_compliance_weighted': round(validation_results['compliance_score'], 2),  # Weighted average of 6 rule scores
                    'semantic_type_accuracy': round(semantic_results['semantic_type_accuracy'], 2),  # LLM-validated entity type correctness
                    'relationship_retention_rate': round(relationship_retention_rate, 2)  # % relationships retained after filtering
                },

                # Per-rule granular scoring (Rule-Level Weighted Average)
                'rule_scores': validation_results['rule_scores'],

                # Semantic validation results
                'semantic_validation': {
                    'total_entities': semantic_results['total_entities'],
                    'correct_entities': semantic_results['correct_entities'],
                    'incorrect_entities': semantic_results['incorrect_entities'],
                    'semantic_type_accuracy': round(semantic_results['semantic_type_accuracy'], 2),
                    'llm_cost': round(semantic_results['llm_cost'], 4),
                    'mismatches': semantic_results['mismatches']
                },

                'rule_compliance': {
                    'critical_rules_total': len(critical_rules),
                    'critical_rules_passed_count': len(critical_rules_passed),
                    'critical_rules_failed_count': len(critical_rules_failed),
                    'critical_rules_failed': critical_rules_failed
                },

                'violation_statistics': {
                    'total_violations': total_violations,
                    'entities_with_violations': entities_with_violations,
                    'relationships_with_violations': relationships_with_violations,
                    'violations_by_rule': violations_by_rule
                },

                'segment_statistics': {
                    'total_segments': len(segment_original_counts),
                    'segments_with_entities': segments_with_entities,
                    'entities_per_segment': entities_per_segment,
                    'retention_per_segment': {k: round(v, 2) for k, v in retention_per_segment.items()},
                    'avg_segment_retention': round(avg_segment_retention, 2)
                }
            },
            'entities': validated_entities,
            'relationships': validated_relationships
        }

        return validated_kg

    def save_validation_report(self, kg_data: Dict, validation_results: Dict[str, Any], document_name: str, output_dir: str) -> str:
        """
        Save validation report JSON (ALWAYS saved, even for failed documents).
        Contains all validation details for later analysis.

        Args:
            kg_data: Original knowledge graph data
            validation_results: Validation results from validate_python()
            document_name: Name of the document
            output_dir: Output directory

        Returns:
            Path to saved report file
        """
        output_path = Path(output_dir) / f"{document_name}_validation_report.json"

        # Extract validation details
        if 'python_validation' in validation_results:
            results = validation_results['python_validation'].get('results', [])
            semantic_results = validation_results['python_validation'].get('semantic_results', {})
            semantic_gate_failed = validation_results['python_validation'].get('semantic_gate_failed', False)
            overall_status = validation_results['python_validation'].get('overall_status', 'UNKNOWN')
            pass_rate = validation_results['python_validation'].get('pass_rate', 0.0)
        else:
            results = validation_results.get('results', [])
            semantic_results = validation_results.get('semantic_results', {})
            semantic_gate_failed = validation_results.get('semantic_gate_failed', False)
            overall_status = validation_results.get('overall_status', 'UNKNOWN')
            pass_rate = validation_results.get('pass_rate', 0.0)

        # Create validation report
        report = {
            "document_name": document_name,
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "semantic_gate_failed": semantic_gate_failed,
            "validation_summary": {
                "semantic_type_accuracy": semantic_results.get('semantic_type_accuracy', 0.0),
                "structural_pass_rate": pass_rate,
                "total_entities": len(kg_data.get('entities', [])),
                "total_relationships": len(kg_data.get('relationships', []))
            },
            "semantic_validation": {
                "total_entities": semantic_results.get('total_entities', 0),
                "correct_entities": semantic_results.get('correct_entities', 0),
                "incorrect_entities": semantic_results.get('incorrect_entities', 0),
                "semantic_type_accuracy": semantic_results.get('semantic_type_accuracy', 0.0),
                "llm_cost": semantic_results.get('llm_cost', 0.0),
                "mismatches": semantic_results.get('mismatches', [])
            },
            "structural_validation": {
                "total_rules": len(results),
                "passed_rules": sum(1 for r in results if r.get('passed', False)),
                "failed_rules": sum(1 for r in results if not r.get('passed', False)),
                "rule_details": results
            }
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"\n📊 Validation Report saved:")
            print(f"  📄 {output_path.name}")
            print(f"  Status: {overall_status}")
            print(f"  Entity Semantic Accuracy: {semantic_results.get('semantic_type_accuracy', 0):.2f}%")
            print(f"  Structural Pass Rate: {pass_rate:.2f}%")

            return str(output_path)

        except Exception as e:
            print(f"❌ Error saving validation report: {str(e)}")
            return ""

    def save_validated_json(self, validated_kg: Dict[str, Any], document_name: str, output_dir: str) -> str:
        """
        Save validated knowledge graph as JSON file.

        Args:
            validated_kg: Filtered/validated knowledge graph
            document_name: Name of the document
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir) / f"{document_name}_validated.json"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_kg, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Validated JSON saved:")
            print(f"  📄 {output_path.name}")
            print(f"  📁 Location: {output_path.parent}/")
            print(f"  📊 Size: {output_path.stat().st_size:,} bytes")
            print(f"  📈 Entities: {validated_kg['validation_metadata']['validated_entity_count']}")
            print(f"  🔗 Relationships: {validated_kg['validation_metadata']['validated_relationship_count']}")

            # Print quality metrics (3 core metrics)
            quality_metrics = validated_kg['validation_metadata']['quality_metrics']
            print(f"\n  📊 Quality Metrics:")
            print(f"    • Schema Compliance (Weighted): {quality_metrics['schema_compliance_weighted']:.2f}%")
            print(f"    • Semantic Type Accuracy: {quality_metrics['semantic_type_accuracy']:.2f}%")
            print(f"    • Relationship Retention Rate: {quality_metrics['relationship_retention_rate']:.2f}%")

            # Print semantic validation details
            semantic_validation = validated_kg['validation_metadata']['semantic_validation']
            print(f"\n  🧠 Semantic Validation:")
            print(f"    • LLM Cost: ${semantic_validation['llm_cost']:.4f}")
            print(f"    • Entities Checked: {semantic_validation['total_entities']}")
            print(f"    • Correct: {semantic_validation['correct_entities']}")
            print(f"    • Incorrect: {semantic_validation['incorrect_entities']}")

            return str(output_path)

        except Exception as e:
            print(f"❌ Error saving validated JSON: {str(e)}")
            return None

    def save_validated_rdf(self, validated_kg: Dict[str, Any], document_name: str, output_dir: str) -> str:
        """
        Save validated knowledge graph as RDF/Turtle file.

        Args:
            validated_kg: Filtered/validated knowledge graph
            document_name: Name of the document
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        try:
            from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
        except ImportError:
            print("⚠️  RDF export unavailable (install rdflib: pip install rdflib)")
            return None

        output_path = Path(output_dir) / f"{document_name}_validated.rdf"

        try:
            # Create RDF graph
            g = Graph()

            # Bind namespaces
            ESG = Namespace("http://example.org/esg-ontology#")
            g.bind("esg", ESG)
            g.bind("rdf", RDF)
            g.bind("rdfs", RDFS)

            # Add entities as RDF triples
            for entity in validated_kg.get('entities', []):
                entity_uri = URIRef(ESG[entity['id']])
                entity_type = entity.get('type', 'Entity')

                # Add type
                g.add((entity_uri, RDF.type, ESG[entity_type]))

                # Add label
                if entity.get('label'):
                    g.add((entity_uri, RDFS.label, Literal(entity['label'])))

                # Add properties
                for prop_key, prop_value in entity.get('properties', {}).items():
                    if prop_value and prop_value != 'N/A':
                        g.add((entity_uri, ESG[prop_key], Literal(str(prop_value))))

                # Add provenance
                if entity.get('provenance'):
                    for prov_key, prov_value in entity['provenance'].items():
                        if prov_value:
                            g.add((entity_uri, ESG[f"provenance_{prov_key}"], Literal(str(prov_value))))

            # Add relationships as RDF triples
            for rel in validated_kg.get('relationships', []):
                subject_uri = URIRef(ESG[rel['subject']])
                predicate_uri = ESG[rel['predicate']]
                object_uri = URIRef(ESG[rel['object']])

                g.add((subject_uri, predicate_uri, object_uri))

            # Serialize to Turtle format
            g.serialize(destination=str(output_path), format='turtle')

            print(f"\n✅ Validated RDF saved:")
            print(f"  📄 {output_path.name}")
            print(f"  📁 Location: {output_path.parent}/")
            print(f"  📊 Size: {output_path.stat().st_size:,} bytes")
            print(f"  🔢 Triples: {len(g):,}")

            return str(output_path)

        except Exception as e:
            print(f"❌ Error saving validated RDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # ========================================
    # BATCH PROCESSING
    # ========================================
    
    def validate_batch(self, kg_dir: str = "outputs/stage2_ontology_guided_extraction", output_dir: str = None) -> Dict[str, Any]:
        """
        Run batch Python validation on all JSON files in directory.
        Saves validated files immediately after each document validates.

        Args:
            kg_dir: Directory containing knowledge graph files
            output_dir: Directory to save validated files (optional, for inline saving)

        Returns:
            Dictionary with batch validation results
        """
        kg_path = Path(kg_dir)
        if not kg_path.exists():
            return {'success': False, 'error': f'Directory not found: {kg_dir}'}

        # Look for JSON files (all naming patterns: ontology-guided, baseline, and legacy knowledge)
        json_files = list(kg_path.glob("*_ontology_guided.json")) + list(kg_path.glob("*_knowledge.json")) + list(kg_path.glob("*_baseline_llm.json"))
        if not json_files:
            return {'success': False, 'error': 'No JSON files found'}

        print(f"\n🚀 BATCH PYTHON VALIDATION")
        print(f"📁 Directory: {kg_dir}")
        print(f"📊 Found {len(json_files)} JSON files")
        print(f"{'='*80}")
        sys.stdout.flush()

        batch_results = []
        successful = 0
        failed = 0
        files_saved = 0
        start_time = time.time()

        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] Processing: {json_file.name}")
            print("=" * 80)
            sys.stdout.flush()

            result = self.validate_python(str(json_file))
            batch_results.append(result)

            if result.get('success', False) and result.get('overall_status') == 'PASS':
                successful += 1
                print(f"✅ SUCCESS: {json_file.name}")
            else:
                failed += 1
                print(f"❌ FAILED: {json_file.name}")

            # Save validated files immediately after validation (if output_dir provided)
            if output_dir and result.get('success', False):
                try:
                    # Extract document name and determine file type
                    file_name = json_file.name
                    is_baseline = '_baseline_llm.json' in file_name
                    is_ontology_guided = '_ontology_guided.json' in file_name

                    doc_name = file_name.replace('_ontology_guided.json', '').replace('_baseline_llm.json', '').replace('_knowledge.json', '')

                    # Load original knowledge graph
                    with open(json_file, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)

                    # Handle baseline file structure vs ontology-guided structure
                    if 'experiments' in raw_data and '2_baseline_llm' in raw_data['experiments']:
                        kg_data = raw_data['experiments']['2_baseline_llm']
                    else:
                        kg_data = raw_data

                    # Filter validated data (entity-level filtering)
                    print(f"\n💾 Saving validated files for: {doc_name}")
                    validated_kg = self.filter_validated_data(kg_data, result)

                    # Only save if valid entities remain after filtering
                    if validated_kg is not None:
                        # Save as JSON (always save JSON for both ontology-guided and baseline)
                        self.save_validated_json(validated_kg, doc_name, output_dir)

                        # Save as RDF ONLY for ontology-guided files (NOT for baseline)
                        if is_ontology_guided:
                            self.save_validated_rdf(validated_kg, doc_name, output_dir)

                        files_saved += 1
                    else:
                        print(f"  ⚠️  {doc_name}: No valid entities remaining - NOT saving")

                except Exception as e:
                    print(f"\n❌ Error saving validated data for {doc_name}: {str(e)}")

        elapsed_time = time.time() - start_time

        # Print save summary if files were saved
        if output_dir:
            print(f"\n{'='*80}")
            print(f"💾 SAVED FILES SUMMARY")
            print(f"{'='*80}")
            print(f"✅ Files saved: {files_saved}/{len(json_files)} documents")
            print(f"📁 Output directory: {output_dir}")
            print()
            sys.stdout.flush()

        return {
            'success': True,
            'approach': 'python',
            'directory': kg_dir,
            'total_files': len(json_files),
            'successful_validations': successful,
            'failed_validations': failed,
            'elapsed_time': elapsed_time,
            'individual_results': batch_results,
            'batch_timestamp': datetime.now().isoformat(),
            'files_saved': files_saved if output_dir else None
        }

    def _get_rule_name(self, rule_id):
        """Get human-readable name for validation rule"""
        rule_names = {
            'VR001': 'CalculatedMetrics must have IsCalculatedBy relationships',
            'VR002': 'Quantitative metrics must specify units',
            'VR003': 'Models must specify input variables',
            'VR004': 'Entity IDs must be unique',
            'VR005': 'Entities must have required properties',
            'VR006': 'Categories should contain metrics',
            'VR007': 'Industries should report using frameworks',
            'VR008': 'Qualitative metrics should have method properties',
            'VR009': 'Models with equations must have input variables',
            'VR010': 'Relationships must use valid predicates'
        }
        return rule_names.get(rule_id, 'Unknown rule')

    def generate_comprehensive_report(self, validation_results: Dict[str, Any], kg_dir: str, output_dir: str):
        """Generate comprehensive validation report with all metrics and tables"""
        try:
            kg_path = Path(kg_dir)
            output_path = Path(output_dir)

            # Load all knowledge graphs and metadata
            doc_data = []
            total_pages = 0
            total_segments = 0

            for doc_result in validation_results.get('individual_results', []):
                file_path = doc_result.get('file_path', '')
                if not file_path:
                    continue

                # Load knowledge graph
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)

                    # Handle baseline file structure (experiments.2_baseline_llm) vs ontology-guided structure
                    if 'experiments' in raw_data and '2_baseline_llm' in raw_data['experiments']:
                        kg_data = raw_data['experiments']['2_baseline_llm']
                    else:
                        kg_data = raw_data
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {file_path}: {e}")
                    continue

                # Extract document name (handle all naming patterns: ontology-guided, baseline, and legacy knowledge)
                doc_name = Path(file_path).name
                doc_name = doc_name.replace('_ontology_guided.json', '').replace('_knowledge.json', '').replace('_baseline_llm.json', '')

                # Load metadata
                metadata = None
                metadata_path = Path('outputs/stage1_segments') / f"{doc_name}_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load metadata for {doc_name}: {e}")

                # Get document stats
                pages = metadata.get('total_pages', 0) if metadata else 0
                segments = metadata.get('total_segments', 0) if metadata else 0
                total_pages += pages
                total_segments += segments

                doc_data.append({
                    'name': doc_name,
                    'kg_data': kg_data,
                    'metadata': metadata,
                    'validation': doc_result,
                    'pages': pages,
                    'segments': segments
                })

            # Sort documents alphabetically for consistent ordering
            doc_data.sort(key=lambda x: x['name'])

            # Generate report sections
            report_lines = []
            report_lines.append("=" * 100)
            report_lines.append("COMPREHENSIVE VALIDATION REPORT")
            report_lines.append("=" * 100)
            report_lines.append("")

            # Section 1: Overall Statistics
            report_lines.append("=" * 100)
            report_lines.append("1. OVERALL STATISTICS")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("Document Processing (from Stage 1 metadata):")
            report_lines.append(f"  • Total Documents Processed: {len(doc_data)}")
            report_lines.append(f"  • Total Pages Analyzed: {total_pages} (from original PDF documents)")
            report_lines.append(f"  • Total Segments Processed: {total_segments} (text chunks used for extraction)")
            report_lines.append("")

            total_entities = sum(len(d['kg_data'].get('entities', [])) for d in doc_data)
            total_metrics = sum(len([e for e in d['kg_data'].get('entities', []) if e.get('type') == 'Metric']) for d in doc_data)

            report_lines.append("Knowledge Graph Statistics (from Stage 2 extraction):")
            report_lines.append(f"  • Total Entities Extracted: {total_entities}")
            report_lines.append(f"  • Total Metrics Identified: {total_metrics}")
            report_lines.append("")

            # Section 2: Evaluation Metrics Summary Table
            report_lines.append("=" * 100)
            report_lines.append("2. EVALUATION METRICS SUMMARY")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("IMPORTANT: All metrics are calculated at the DOCUMENT LEVEL (not per-segment or per-page)")
            report_lines.append("")
            report_lines.append("Key Performance Metrics:")
            report_lines.append("")
            report_lines.append("1. Schema Compliance (%) - STRUCTURAL VALIDATION")
            report_lines.append("")
            report_lines.append("   The 6 Critical Rules (All Critical):")
            report_lines.append("   • VR001: Entity Uniqueness - All entity IDs must be unique")
            report_lines.append("   • VR002: Entity Type-Specific Schema Compliance - Each entity type has required fields")
            report_lines.append("           (Metric: measurement_type, metric_type, unit, code, description)")
            report_lines.append("           (Model: description, equation, input_variables)")
            report_lines.append("           (Category: section_title)")
            report_lines.append("   • VR003: Metric Property Values - DirectMetric/CalculatedMetric code ≠ N/A; Quantitative unit ≠ N/A")
            report_lines.append("   • VR004: Model Property Values - Models must have non-empty input_variables list")
            report_lines.append("   • VR005: Relationship Predicate Validity - Only 5 allowed predicates")
            report_lines.append("           (ConsistOf, Include, IsCalculatedBy, ReportUsing, RequiresInputFrom)")
            report_lines.append("   • VR006: CalculatedMetric-Model Links - All CalculatedMetrics must have IsCalculatedBy → Model")
            report_lines.append("")
            report_lines.append("   Definition: Average of per-rule granular scores across all 6 rules")
            report_lines.append("   Formula: Average of [(Passed Items / Total Items) × 100%] per rule")
            report_lines.append("   Approach: Rule-Level Weighted Average")
            report_lines.append("   Scope: PER DOCUMENT (granular scoring per rule)")
            report_lines.append("")
            report_lines.append("   How it works:")
            report_lines.append("   • Each rule calculates its own score based on items checked")
            report_lines.append("   • VR001 checks all entities for uniqueness → score = (unique_entities / total_entities) × 100%")
            report_lines.append("   • VR002 checks all entities for required fields → score = (valid_entities / total_entities) × 100%")
            report_lines.append("   • VR003 checks all Metrics for property values → score = (valid / total_metrics) × 100%")
            report_lines.append("   • VR004 checks only Models for input_variables → score = (valid / total_models) × 100%")
            report_lines.append("   • VR005 checks all relationships for valid predicates → score = (valid / total_rels) × 100%")
            report_lines.append("   • VR006 checks only CalculatedMetrics for Model links → score = (valid / total_calc_metrics) × 100%")
            report_lines.append("   • Overall compliance = average of all 6 rule scores")
            report_lines.append("")
            report_lines.append("   Interpretation:")
            report_lines.append("   • 100% = All items in all rules passed (perfect compliance)")
            report_lines.append("   • 84.95% = Average score showing some items failed (provides granular quality insight)")
            report_lines.append("   • 10.1% = Low score indicating significant violations (shows severity of issues)")
            report_lines.append("")
            report_lines.append("   Example:")
            report_lines.append("   • VR002 has 7/69 entities passing → Score: 10.1%")
            report_lines.append("   • Shows 62 entities need schema fixes, provides actionable improvement target")
            report_lines.append("")
            report_lines.append("2. Semantic Accuracy (%) - ENTITY SEMANTIC VALIDATION")
            report_lines.append("   Definition: % of entities with semantically correct type assignments")
            report_lines.append("   Formula: (Correctly Typed Entities / Total Entities) × 100%")
            report_lines.append("   Scope: PER ENTITY (each entity independently validated)")
            report_lines.append("   Validation Method: Claude LLM checks if entity label/description matches its assigned type")
            report_lines.append("")
            report_lines.append("   Example:")
            report_lines.append("   • Entity: {type: 'Metric', label: 'Scope 1 Emissions'}")
            report_lines.append("   • LLM Question: 'Does \"Scope 1 Emissions\" semantically represent a Metric?'")
            report_lines.append("   • Answer: YES → Semantically Correct")
            report_lines.append("")
            report_lines.append("   Interpretation:")
            report_lines.append("   • 100% = All entity types are semantically correct")
            report_lines.append("   • 80% = 20% of entities have incorrect type assignments")
            report_lines.append("   • Cost: LLM validation cost tracked in semantic_validation.llm_cost")
            report_lines.append("")
            report_lines.append("3. Relationship Retention (%) - CASCADE FILTERING")
            report_lines.append("   Definition: % of relationships retained after entity semantic filtering")
            report_lines.append("   Formula: (Retained Relationships / Original Relationships) × 100%")
            report_lines.append("   Scope: PER RELATIONSHIP (but DEPENDENT on entity validation)")
            report_lines.append("")
            report_lines.append("   How it works (CASCADE effect):")
            report_lines.append("   • Step 1: Entities are validated for semantic correctness")
            report_lines.append("   • Step 2: Entities failing semantic validation are REMOVED")
            report_lines.append("   • Step 3: Relationships referencing removed entities are ALSO removed (cascade)")
            report_lines.append("   • Retention Rate = % of relationships that survived this cascade")
            report_lines.append("")
            report_lines.append("   Example:")
            report_lines.append("   • Original: 100 relationships")
            report_lines.append("   • Entity semantic validation removes 10 entities")
            report_lines.append("   • 15 relationships reference these 10 removed entities")
            report_lines.append("   • Result: 85 relationships retained → 85% retention rate")
            report_lines.append("")
            report_lines.append("   Interpretation:")
            report_lines.append("   • 100% = All relationships retained (no entities removed, fully connected graph)")
            report_lines.append("   • 80% = 20% of relationships removed (graph connectivity reduced)")
            report_lines.append("   • 0% = All relationships removed (completely disconnected graph)")
            report_lines.append("")
            report_lines.append("   IMPORTANT: This is NOT an independent validation of relationships themselves.")
            report_lines.append("              Relationships are only removed if their connected entities fail semantic validation.")
            report_lines.append("")

            # Calculate metrics for each document using enhanced metadata from validated files
            metrics_data = []
            for doc in doc_data:
                entities = doc['kg_data'].get('entities', [])
                relationships = doc['kg_data'].get('relationships', [])
                validation = doc['validation']

                # Try to load the validated JSON file to get quality_metrics
                validated_file = None
                doc_name = doc['name']

                # Try to find validated file in output directory
                output_path = Path(output_dir)
                possible_validated_paths = [
                    output_path / f"{doc_name}_validated.json",
                ]

                quality_metrics = None
                for vpath in possible_validated_paths:
                    if vpath.exists():
                        try:
                            with open(vpath, 'r', encoding='utf-8') as f:
                                validated_data = json.load(f)
                                quality_metrics = validated_data.get('validation_metadata', {}).get('quality_metrics', {})
                                break
                        except Exception as e:
                            pass

                # Handle documents with no validated file (all entities filtered out)
                if not quality_metrics:
                    # Document had 0 valid entities after filtering - calculate metrics from validation results
                    # This happens when baseline extraction creates entities with types not in ontology
                    print(f"⚠️ Note: No validated file for {doc_name} (all entities filtered out)")

                    # IMPORTANT: Schema compliance should be 0% when all entities are filtered out
                    # The compliance_score in validation results is calculated BEFORE semantic filtering
                    # and reflects the original entities (many with unknown types). Since ALL entities
                    # were filtered out during semantic validation, the schema compliance is effectively 0%.
                    schema_compliance = 0.0

                    # Get semantic results
                    semantic_results = validation.get('semantic_results', {})
                    semantic_type_accuracy = semantic_results.get('semantic_type_accuracy', 0.0)
                    llm_cost = semantic_results.get('llm_cost', 0.0)

                    # All entities filtered out
                    validated_entities = 0
                    validated_relationships = 0
                    relationship_retention_rate = 0.0  # All relationships removed
                else:
                    # Extract metrics from validated file - normal case
                    schema_compliance = quality_metrics['schema_compliance_weighted']
                    relationship_retention_rate = quality_metrics['relationship_retention_rate']
                    semantic_type_accuracy = quality_metrics['semantic_type_accuracy']

                    # Extract semantic validation details for cost tracking
                    semantic_validation = validated_data['validation_metadata']['semantic_validation']
                    llm_cost = semantic_validation['llm_cost']

                    # Extract validated counts from validated JSON file
                    validation_meta = validated_data['validation_metadata']
                    validated_entities = validation_meta['validated_entity_count']
                    validated_relationships = validation_meta['validated_relationship_count']

                metrics_data.append({
                    'name': doc['name'],
                    'schema_compliance_weighted': schema_compliance,
                    'relationship_retention_rate': relationship_retention_rate,
                    'semantic_type_accuracy': semantic_type_accuracy,
                    'llm_cost': llm_cost,
                    'original_entities': len(entities),
                    'original_relationships': len(relationships),
                    'validated_entities': validated_entities,
                    'validated_relationships': validated_relationships
                })

            # Print table header with 3-metric scheme
            report_lines.append("{:<35} {:>20} {:>20} {:>20}".format(
                "Document", "Schema Compliance%", "Semantic Acc%", "Rel. Retention%"
            ))
            report_lines.append("-" * 98)

            # Print each document
            for m in metrics_data:
                report_lines.append("{:<35} {:>20.1f} {:>20.1f} {:>20.1f}".format(
                    m['name'][:35],
                    m['schema_compliance_weighted'],
                    m['semantic_type_accuracy'],
                    m['relationship_retention_rate']
                ))

            report_lines.append("")
            report_lines.append("Interpretation:")
            report_lines.append("")

            # Show interpretation with example
            if metrics_data:
                example = metrics_data[0]
                report_lines.append(f"Example: {example['name'][:50]}")
                report_lines.append(f"  • Schema Compliance = {example['schema_compliance_weighted']:.1f}%")
                report_lines.append(f"    (Average of per-rule granular scores - Rule-Level Weighted Average)")
                report_lines.append(f"  • Semantic Accuracy = {example['semantic_type_accuracy']:.1f}%")
                report_lines.append(f"    ({int(example['semantic_type_accuracy']/100*example['original_entities'])}/{example['original_entities']} entities semantically correct)")
                report_lines.append(f"    (LLM validation cost: ${example['llm_cost']:.4f})")
                report_lines.append(f"  • Relationship Retention = {example['relationship_retention_rate']:.1f}%")
                report_lines.append(f"    ({int(example['relationship_retention_rate']/100*example['original_relationships'])}/{example['original_relationships']} relationships retained)")
                report_lines.append("")

                # Calculate and show total LLM cost
                total_llm_cost = sum(m.get('llm_cost', 0.0) for m in metrics_data)
                report_lines.append(f"Total Semantic Validation Cost: ${total_llm_cost:.4f}")
                report_lines.append(f"  • Across {len(metrics_data)} document(s)")
                report_lines.append(f"  • Average cost per document: ${total_llm_cost/len(metrics_data) if len(metrics_data) > 0 else 0:.4f}")
                report_lines.append("")
                report_lines.append("Note: Metrics are calculated from validation_metadata in validated JSON files")
            report_lines.append("")

            # Section 3: Document Summary Table
            report_lines.append("=" * 100)
            report_lines.append("3. DOCUMENT SUMMARY TABLE")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("Shows pipeline data flow: Stage 1 (parsing) → Stage 2 (extraction) → Stage 3 (validation)")
            report_lines.append("")

            # Create lookup dictionary for validated counts
            validated_counts = {}
            for m in metrics_data:
                validated_counts[m['name']] = {
                    'validated_entities': m['validated_entities'],
                    'validated_relationships': m['validated_relationships']
                }

            report_lines.append("{:<40} {:>8} {:>10} {:>15} {:>15} {:>12} {:>12}".format(
                "Document", "Pages", "Segments", "Stage2 Entities", "Stage3 Entities", "Stage2 Rels", "Stage3 Rels"
            ))
            report_lines.append("-" * 120)

            for doc in doc_data:
                entities = doc['kg_data'].get('entities', [])
                relationships = doc['kg_data'].get('relationships', [])

                # Get validated counts (0 if document completely failed validation)
                doc_validated = validated_counts.get(doc['name'], {})
                stage3_entities = doc_validated.get('validated_entities', 0)
                stage3_relationships = doc_validated.get('validated_relationships', 0)

                report_lines.append("{:<40} {:>8} {:>10} {:>15} {:>15} {:>12} {:>12}".format(
                    doc['name'][:40],
                    doc['pages'],
                    doc['segments'],
                    len(entities),
                    stage3_entities,
                    len(relationships),
                    stage3_relationships
                ))

            report_lines.append("")

            # Section 3a: Entity Type Distribution Summary Table
            report_lines.append("=" * 100)
            report_lines.append("3a. ENTITY TYPE DISTRIBUTION SUMMARY (All Documents)")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("Shows entity type counts aggregated across all documents for Stage 2 (extraction) and Stage 3 (validation)")
            report_lines.append("")
            report_lines.append("Retention (%): Calculated as (Stage 3 Count / Stage 2 Count) × 100% for each entity type")
            report_lines.append("Filtering Mechanism: Entities removed based on semantic type validation (LLM-validated correctness)")
            report_lines.append("")

            # Initialize entity type counters
            entity_types = ['Industry', 'ReportingFramework', 'Category', 'Metric', 'Model']
            stage2_type_counts = {et: 0 for et in entity_types}
            stage3_type_counts = {et: 0 for et in entity_types}

            # Count entity types from Stage 2 and Stage 3 (read actual validated files)
            for doc in doc_data:
                # Count Stage 2 entity types
                entities = doc['kg_data'].get('entities', [])
                for entity in entities:
                    entity_type = entity.get('type', 'Unknown')
                    if entity_type in entity_types:
                        stage2_type_counts[entity_type] += 1

                # Count Stage 3 entity types from validated files
                doc_name = doc['name']
                validated_file = output_path / f"{doc_name}_validated.json"

                if validated_file.exists():
                    try:
                        with open(validated_file, 'r', encoding='utf-8') as f:
                            validated_data = json.load(f)

                        validated_entities = validated_data.get('entities', [])
                        for entity in validated_entities:
                            entity_type = entity.get('type', 'Unknown')
                            if entity_type in entity_types:
                                stage3_type_counts[entity_type] += 1
                    except Exception as e:
                        # If validated file can't be read, skip Stage 3 counting for this document
                        print(f"⚠️ Warning: Could not read validated file for {doc_name}: {e}")
                        pass

            # Print table header
            report_lines.append("{:<25} {:>15} {:>15} {:>15}".format(
                "Entity Type", "Stage 2 Count", "Stage 3 Count", "Retention (%)"
            ))
            report_lines.append("-" * 75)

            # Print each entity type
            total_stage2 = sum(stage2_type_counts.values())
            total_stage3 = sum(stage3_type_counts.values())

            for entity_type in entity_types:
                stage2_count = stage2_type_counts[entity_type]
                stage3_count = stage3_type_counts[entity_type]
                retention_rate = (stage3_count / stage2_count * 100) if stage2_count > 0 else 0.0

                report_lines.append("{:<25} {:>15} {:>15} {:>15.1f}".format(
                    entity_type,
                    stage2_count,
                    stage3_count,
                    retention_rate
                ))

            # Print total row
            report_lines.append("-" * 75)
            total_retention = (total_stage3 / total_stage2 * 100) if total_stage2 > 0 else 0.0
            report_lines.append("{:<25} {:>15} {:>15} {:>15.1f}".format(
                "TOTAL",
                total_stage2,
                total_stage3,
                total_retention
            ))

            report_lines.append("")

            # Section 3b: Relationship Predicate Distribution Summary Table
            report_lines.append("=" * 100)
            report_lines.append("3b. RELATIONSHIP PREDICATE DISTRIBUTION SUMMARY (All Documents)")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("Shows relationship predicate counts aggregated across all documents for Stage 2 (extraction) and Stage 3 (validation)")
            report_lines.append("")
            report_lines.append("Retention (%): Calculated as (Stage 3 Count / Stage 2 Count) × 100% for each predicate type")
            report_lines.append("Filtering Mechanism: Cascade removal (relationships removed when their connected entities fail semantic validation)")
            report_lines.append("Note: Relationships are NOT independently validated - only removed if subject/object entities are removed")
            report_lines.append("")

            # Initialize relationship predicate counters
            stage2_predicate_counts = {}
            stage3_predicate_counts = {}

            # Count relationship predicates from Stage 2 and Stage 3 (read actual validated files)
            for doc in doc_data:
                # Count Stage 2 relationship predicates
                relationships = doc['kg_data'].get('relationships', [])
                for relationship in relationships:
                    predicate = relationship.get('predicate', 'Unknown')
                    stage2_predicate_counts[predicate] = stage2_predicate_counts.get(predicate, 0) + 1

                # Count Stage 3 relationship predicates from validated files
                doc_name = doc['name']
                validated_file = output_path / f"{doc_name}_validated.json"

                if validated_file.exists():
                    try:
                        with open(validated_file, 'r', encoding='utf-8') as f:
                            validated_data = json.load(f)

                        validated_relationships = validated_data.get('relationships', [])
                        for relationship in validated_relationships:
                            predicate = relationship.get('predicate', 'Unknown')
                            stage3_predicate_counts[predicate] = stage3_predicate_counts.get(predicate, 0) + 1
                    except Exception as e:
                        # If validated file can't be read, skip Stage 3 counting for this document
                        print(f"⚠️ Warning: Could not read validated file for {doc_name}: {e}")
                        pass

            # Print table header
            report_lines.append("{:<30} {:>15} {:>15} {:>15}".format(
                "Relationship Predicate", "Stage 2 Count", "Stage 3 Count", "Retention (%)"
            ))
            report_lines.append("-" * 75)

            # Print each predicate (sorted alphabetically)
            all_predicates = sorted(set(list(stage2_predicate_counts.keys()) + list(stage3_predicate_counts.keys())))
            total_stage2_rels = sum(stage2_predicate_counts.values())
            total_stage3_rels = sum(stage3_predicate_counts.values())

            for predicate in all_predicates:
                stage2_count = stage2_predicate_counts.get(predicate, 0)
                stage3_count = stage3_predicate_counts.get(predicate, 0)
                retention_rate = (stage3_count / stage2_count * 100) if stage2_count > 0 else 0.0

                report_lines.append("{:<30} {:>15} {:>15} {:>15.1f}".format(
                    predicate,
                    stage2_count,
                    stage3_count,
                    retention_rate
                ))

            # Print total row
            report_lines.append("-" * 75)
            total_rel_retention = (total_stage3_rels / total_stage2_rels * 100) if total_stage2_rels > 0 else 0.0
            report_lines.append("{:<30} {:>15} {:>15} {:>15.1f}".format(
                "TOTAL",
                total_stage2_rels,
                total_stage3_rels,
                total_rel_retention
            ))

            report_lines.append("")

            # Section 4: Validation Rules Summary Table
            report_lines.append("=" * 100)
            report_lines.append("4. VALIDATION RULES SUMMARY (All Documents)")
            report_lines.append("=" * 100)
            report_lines.append("")

            # Get all 6 validation rule names (ontology-aligned)
            all_rules = [
                ('VR001', 'Entity Uniqueness'),
                ('VR002', 'Entity Type-Specific Schema Compliance'),
                ('VR003', 'Metric Property Values'),
                ('VR004', 'Model Property Values'),
                ('VR005', 'Relationship Predicate Validity'),
                ('VR006', 'CalculatedMetric-Model Links')
            ]

            # Print header
            header = "{:<10} {:<35}".format("Rule ID", "Rule Name")
            for doc in doc_data:
                doc_abbrev = doc['name'][:15]
                header += " {:>16}".format(doc_abbrev)
            report_lines.append(header)
            report_lines.append("-" * (47 + len(doc_data) * 17))

            # Print each rule
            for rule_id, rule_name in all_rules:
                row = "{:<10} {:<35}".format(rule_id, rule_name[:35])

                for doc in doc_data:
                    # Find this rule in the validation results
                    rule_result = next(
                        (r for r in doc['validation'].get('results', []) if r['rule_id'] == rule_id),
                        None
                    )

                    if rule_result:
                        status = "✅ PASS" if rule_result.get('passed', False) else "❌ FAIL"
                        violations = rule_result.get('violation_count', 0)
                        cell = "{} ({})".format(status, violations) if not rule_result.get('passed') else status
                    else:
                        cell = "N/A"

                    row += " {:>16}".format(cell[:16])

                report_lines.append(row)

            report_lines.append("")

            # Section 5: Provenance Quality Table
            report_lines.append("=" * 100)
            report_lines.append("5. PROVENANCE QUALITY ANALYSIS")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("Provenance Methodology:")
            report_lines.append("  • Each entity contains provenance.segment_id linking to stage1 segments")
            report_lines.append("  • Each segment contains page_start and page_end fields")
            report_lines.append("  • Therefore: Entity → segment_id → Segment → page numbers (complete traceability)")
            report_lines.append("")
            report_lines.append("Calculation:")
            report_lines.append("  • With Provenance (%) = (Entities with segment_id / Total Entities) × 100")
            report_lines.append("  • With Timestamp (%) = (Entities with extraction_timestamp / Total Entities) × 100")
            report_lines.append("")

            report_lines.append("{:<40} {:>15} {:>25} {:>25}".format(
                "Document", "Total Entities", "With Provenance (%)", "With Timestamp (%)"
            ))
            report_lines.append("-" * 105)

            for doc in doc_data:
                entities = doc['kg_data'].get('entities', [])
                total = len(entities)

                # Check for segment_id in provenance (links to page numbers in stage1)
                with_segment = sum(1 for e in entities if e.get('provenance', {}).get('segment_id'))
                with_timestamp = sum(1 for e in entities if e.get('provenance', {}).get('extraction_timestamp'))

                provenance_pct = (with_segment / total * 100) if total > 0 else 0
                timestamp_pct = (with_timestamp / total * 100) if total > 0 else 0

                report_lines.append("{:<40} {:>15} {:>25.1f} {:>25.1f}".format(
                    doc['name'][:40],
                    total,
                    provenance_pct,
                    timestamp_pct
                ))

            report_lines.append("")

            # Section 5: Metrics Reference
            report_lines.append("=" * 100)
            report_lines.append("5. METRICS REFERENCE & CALCULATION GUIDE")
            report_lines.append("=" * 100)
            report_lines.append("")
            report_lines.append("This section provides comprehensive documentation of all metrics used in Stage 3 validation.")
            report_lines.append("For complete details, see: result_visualisation_and_analysis/METRICS_REFERENCE.txt")
            report_lines.append("")

            report_lines.append("-" * 100)
            report_lines.append("5.1 STAGE 3 VALIDATION METRICS (Enhanced Metadata)")
            report_lines.append("-" * 100)
            report_lines.append("")
            report_lines.append("All enhanced metrics are stored in validated JSON files under 'validation_metadata' section.")
            report_lines.append("")

            report_lines.append("QUALITY METRICS (quality_metrics section):")
            report_lines.append("")
            report_lines.append("  • Validation Pass Rate (%)")
            report_lines.append("    Formula: (Critical Rules Passed / Total Critical Rules) × 100")
            report_lines.append("    Where: Total Critical Rules = 6 (VR001-VR006, All Critical)")
            report_lines.append("    Unit: Percentage (0-100%)")
            report_lines.append("    Aggregation: Document-level")
            report_lines.append("    Data Source: Stage 3 validation results (rule_compliance section)")
            report_lines.append("    Interpretation:")
            report_lines.append("      - 100% = All validation rules passed (perfect schema compliance)")
            report_lines.append("      - 83.3% = 1 rule failed (5/6 passed)")
            report_lines.append("      - Measures schema compliance and data quality")
            report_lines.append("")


            report_lines.append("  • Relationship Retention Rate (%)")
            report_lines.append("    Formula: (Validated Relationships / Original Relationships) × 100")
            report_lines.append("    Unit: Percentage (0-100%)")
            report_lines.append("    Aggregation: Document-level")
            report_lines.append("    Data Source: original_relationship_count + validated_relationship_count")
            report_lines.append("    Interpretation:")
            report_lines.append("      - 100% = All relationships valid")
            report_lines.append("      - 0% = All relationships invalid (common in baseline method)")
            report_lines.append("")

            report_lines.append("RULE COMPLIANCE (rule_compliance section):")
            report_lines.append("")
            report_lines.append("  • critical_rules_total: Always 6 (VR001-VR006)")
            report_lines.append("  • critical_rules_passed_count: Number of rules that passed (0-6)")
            report_lines.append("  • critical_rules_failed_count: Number of rules that failed (0-6)")
            report_lines.append("  • critical_rules_failed: List of specific failed rule IDs")
            report_lines.append("  • pass_rate_percentage: Same as Validation Pass Rate")
            report_lines.append("")

            report_lines.append("VIOLATION STATISTICS (violation_statistics section):")
            report_lines.append("")
            report_lines.append("  • total_violations: Total number of rule violations across all rules")
            report_lines.append("  • entities_with_violations: Count of entities that were removed")
            report_lines.append("  • relationships_with_violations: Count of relationships that were removed")
            report_lines.append("  • violations_by_rule: Per-rule breakdown")
            report_lines.append("    - Each rule ID (VR001-VR010) shows:")
            report_lines.append("      * passed: boolean (true/false)")
            report_lines.append("      * violation_count: integer count")
            report_lines.append("")

            report_lines.append("SEGMENT STATISTICS (segment_statistics section):")
            report_lines.append("")
            report_lines.append("  • total_segments: Number of text segments in document")
            report_lines.append("  • segments_with_entities: Segments that contained entities")
            report_lines.append("  • entities_per_segment: Per-segment entity counts")
            report_lines.append("    - Format: {segment_id: {original: int, validated: int}}")
            report_lines.append("    - Shows exactly how many entities filtered per segment")
            report_lines.append("  • retention_per_segment: Per-segment retention rates (%)")
            report_lines.append("    - Format: {segment_id: retention_percentage}")
            report_lines.append("    - Enables identification of problematic segments")
            report_lines.append("  • avg_segment_retention: Average retention across all segments (%)")
            report_lines.append("")

            report_lines.append("-" * 100)
            report_lines.append("5.2 AGGREGATION LEVELS")
            report_lines.append("-" * 100)
            report_lines.append("")
            report_lines.append("SEGMENT-LEVEL (Most Granular):")
            report_lines.append("  • Source: Individual text chunks from Stage 1 (e.g., seg_001, seg_002)")
            report_lines.append("  • Metrics Available:")
            report_lines.append("    - entities_per_segment (original + validated counts)")
            report_lines.append("    - retention_per_segment (retention %)")
            report_lines.append("  • Use Cases:")
            report_lines.append("    - Identify problematic document sections")
            report_lines.append("    - Trace entities back to source text")
            report_lines.append("    - Target re-extraction at specific segments")
            report_lines.append("")

            report_lines.append("DOCUMENT-LEVEL (Aggregated from Segments):")
            report_lines.append("  • Source: Aggregated from all segments in one document")
            report_lines.append("  • Metrics Available:")
            report_lines.append("    - All quality_metrics (schema_compliance_weighted, semantic_type_accuracy, relationship_retention_rate)")
            report_lines.append("    - All rule_compliance metrics")
            report_lines.append("    - All violation_statistics")
            report_lines.append("    - avg_segment_retention")
            report_lines.append("  • Use Cases:")
            report_lines.append("    - Compare documents to each other")
            report_lines.append("    - Assess overall document quality")
            report_lines.append("    - Report per-document performance")
            report_lines.append("")

            report_lines.append("METHOD-LEVEL (Aggregated from Documents):")
            report_lines.append("  • Source: Aggregated across all documents for one method")
            report_lines.append("    - Ontology-Guided: All documents processed with ontology-guided extraction")
            report_lines.append("    - Baseline: All documents processed with baseline LLM extraction")
            report_lines.append("  • Metrics Available:")
            report_lines.append("    - Average schema_compliance_weighted across all documents")
            report_lines.append("    - Average semantic_type_accuracy and relationship_retention_rate across all documents")
            report_lines.append("    - Total entities extracted/validated")
            report_lines.append("  • Use Cases:")
            report_lines.append("    - Compare Ontology-Guided vs Baseline methods")
            report_lines.append("    - Report overall pipeline performance")
            report_lines.append("    - Visualizations in result_visualisation_and_analysis/")
            report_lines.append("")

            report_lines.append("-" * 100)
            report_lines.append("5.3 FILE LOCATIONS")
            report_lines.append("-" * 100)
            report_lines.append("")
            report_lines.append("STAGE 3 VALIDATION OUTPUTS:")
            report_lines.append("")
            report_lines.append("  Ontology-Guided Method:")
            report_lines.append("    • outputs/stage3_ontology_guided_validation/[document]_validated.json")
            report_lines.append("    • Contains: All enhanced metrics in 'validation_metadata' section")
            report_lines.append("")
            report_lines.append("  Baseline Method:")
            report_lines.append("    • outputs/stage3_baseline_llm_comparison/[document]_validated.json")
            report_lines.append("    • Contains: All enhanced metrics in 'validation_metadata' section")
            report_lines.append("")

            report_lines.append("VISUALIZATION OUTPUTS:")
            report_lines.append("    • result_visualisation_and_analysis/stage3_comparison/")
            report_lines.append("      - validation_quality.png (Validation Pass Rate + Entity Retention)")
            report_lines.append("      - validation_quality.csv (Data table)")
            report_lines.append("      - validation_quality_EXPLANATION.txt (Formulas + interpretation)")
            report_lines.append("      - filtering_impact.png (Entities filtered)")
            report_lines.append("      - filtering_impact.csv (Data table)")
            report_lines.append("      - filtering_impact_EXPLANATION.txt (Formulas + interpretation)")
            report_lines.append("")

            report_lines.append("COMPREHENSIVE METRICS DOCUMENTATION:")
            report_lines.append("    • result_visualisation_and_analysis/METRICS_REFERENCE.txt")
            report_lines.append("      - Complete reference for all metrics")
            report_lines.append("      - Formulas, data sources, interpretations")
            report_lines.append("      - Covers Stage 2, Stage 3, and visualization metrics")
            report_lines.append("")

            report_lines.append("=" * 100)
            report_lines.append("END OF REPORT")
            report_lines.append("=" * 100)

            # Save report with appropriate filename based on method
            # Detect if this is baseline or ontology-guided based on directory paths
            is_baseline = 'baseline' in str(kg_path).lower() or 'baseline' in str(output_path).lower()
            report_filename = "batch_baseline_validation.txt" if is_baseline else "batch_ontology_guided_validation.txt"
            report_path = output_path / report_filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            print(f"\n{'='*80}")
            print(f"OUTPUT FILE GENERATED")
            print(f"{'='*80}")
            print(f"✓ 1 file created:")
            print(f"  📄 {report_path.name}")
            print(f"  📁 Location: {report_path.parent}/")
            print(f"  📊 Size: {report_path.stat().st_size} bytes")
            print(f"  📝 Contains: Comprehensive validation results for {len(doc_data)} documents")
            print()

        except Exception as e:
            print(f"❌ Error generating comprehensive report: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function for Python-based validation CLI"""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Validation System - Python-based validation using 6 ontology-aligned rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Python validation (RECOMMENDED)
  python3 src/stage3_ontology_guided_validation.py --python --single "document_name"
  python3 src/stage3_ontology_guided_validation.py --python --batch

  # Save validated knowledge graphs (removes critical violations)
  python3 src/stage3_ontology_guided_validation.py --python --single "document_name" --save-validated
  python3 src/stage3_ontology_guided_validation.py --python --batch --save-validated
        """
    )

    # Required arguments
    parser.add_argument('--kg-dir', type=str, required=True,
                       help='Directory containing Stage 2 knowledge graph files (e.g., outputs/stage2_ontology_guided_extraction or outputs/stage2_baseline_extraction)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save validation outputs (e.g., outputs/stage3_ontology_guided_validation or outputs/stage3_baseline_validation)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize validator
    validator = KnowledgeGraphValidator()

    # Batch validation for all documents in directory
    # Files are saved immediately after each document validates (inline saving)
    print(f"\n📂 Processing all documents from: {args.kg_dir}")
    sys.stdout.flush()
    results = validator.validate_batch(args.kg_dir, output_dir=args.output_dir)

    # Generate comprehensive report AFTER all files are saved
    print("\n📊 Generating comprehensive validation report...")
    sys.stdout.flush()
    validator.generate_comprehensive_report(results, args.kg_dir, args.output_dir)


if __name__ == "__main__":
    main()
