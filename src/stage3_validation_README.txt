================================================================================
STAGE 3: KNOWLEDGE GRAPH VALIDATION
================================================================================

Purpose: Validate and filter knowledge graphs from Stage 2 using two-stage
         validation (semantic + structural) to ensure high-quality output.

Version: 2.0
Last Updated: 2025-11-20

================================================================================
PART 1: WORKFLOW AND DESIGN LOGIC
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 3 VALIDATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT:
  • Stage 2 Knowledge Graphs (JSON files)
    - Ontology-Guided: outputs/stage2_ontology_guided_extraction/*.json
    - Baseline LLM: outputs/stage2_baseline_llm_extraction/*.json

  • Validation Rules (JSON)
    - validation_rules.json (6 critical structural rules: VR001-VR006)

  • ESGMKG Ontology (Stage 3 Validation Schema)
    - 5 allowed entity types: Industry, ReportingFramework, Category, Metric, Model
    - Note: 'Metric' is a unified type (Stage 2's DirectMetric/CalculatedMetric/InputMetric
      become 'Metric' with metric_type property)
    - 5 allowed predicates: ReportUsing, Include, ConsistOf, IsCalculatedBy, RequiresInputFrom
    - Note: Stage 2 may extract 7 predicates, but Stage 3 only validates these 5

OUTPUT:
  • Validated Knowledge Graphs (JSON + RDF)
    - Only documents with >0 valid entities are saved
    - outputs/stage3_ontology_guided_validation/*_validated.json
    - outputs/stage3_ontology_guided_validation/*_validated.rdf

  • Comprehensive Validation Report (TXT)
    - Batch report for all documents in directory
    - outputs/stage3_*/batch_*_validation.txt

================================================================================
1.1 TWO-STAGE VALIDATION ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 OUTPUT (Knowledge Graph)                                          │
│  • 364 entities (ontology-guided) or 729 entities (baseline)              │
│  • Mixed quality, may have wrong types                                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: SEMANTIC VALIDATION (Gate-Keeper)                                │
│  ═══════════════════════════════════════════════════════════════════════   │
│  Purpose: Filter entities with semantically incorrect type assignments     │
│  Method: LLM-based validation (Claude)                                     │
│                                                                             │
│  For each entity:                                                           │
│    1. Check if type in ontology → If NO, mark FAILED                      │
│    2. LLM validates: "Does '{label}' represent a {type}?" → YES/NO        │
│    3. Mark entity as CORRECT or INCORRECT                                 │
│                                                                             │
│  Metrics: semantic_type_accuracy, llm_cost, mismatches                     │
│  Results: Ontology: 80.8% | Baseline: 3.5%                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENTITY FILTERING (Cascade Effect)                                        │
│  • Remove entities that failed semantic validation                         │
│  • Cascade: Remove relationships referencing removed entities              │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: STRUCTURAL VALIDATION (6 Critical Rules)                         │
│  ═══════════════════════════════════════════════════════════════════════   │
│  VR001: Entity Uniqueness - All IDs unique                                │
│  VR002: Type-Specific Schema - Required fields per type                    │
│  VR003: Metric Properties - code/unit not N/A                             │
│  VR004: Model Properties - non-empty input_variables                       │
│  VR005: Relationship Predicates - Only 5 allowed                          │
│  VR006: CalculatedMetric Links - Must have IsCalculatedBy → Model         │
│                                                                             │
│  Overall: schema_compliance = Average of 6 rule scores                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3 CORE QUALITY METRICS                                                    │
│  1. Schema Compliance: Average of 6 rules (0-100%)                        │
│  2. Semantic Accuracy: % correct types (0-100%)                           │
│  3. Relationship Retention: % rels kept (0-100%)                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SAVE DECISION                                                             │
│  IF entities > 0: Save JSON + RDF  |  ELSE: Skip (include in report)     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VALIDATED OUTPUT                                                          │
│  • 295 entities (ontology) or 6 entities (baseline)                       │
│  • All semantically + structurally validated                              │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
1.2 BATCH PROCESSING WORKFLOW
================================================================================

START → Load JSON files → FOR each document:
           │                  │
           │                  ├─→ Validate (semantic + structural)
           │                  ├─→ Save immediately if entities > 0
           │                  └─→ Next document
           │
           └─→ Generate comprehensive report → END

Key: Immediate-save pattern (fault-tolerant, real-time output)

================================================================================
PART 2: IMPLEMENTATION ALGORITHMS (PSEUDO-CODE)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 1: MAIN VALIDATION FLOW                                            │
└─────────────────────────────────────────────────────────────────────────────┘

Step | Function                  | Input                | Output
-----|---------------------------|----------------------|-------------------
1    | main()                    | CLI args             | Exit code
2    | validate_batch()          | kg_dir, output_dir   | Batch results
3    | validate_python()         | KG JSON file         | Validation result
4    | validate_semantic_types() | Entities list        | Semantic results
5    | validate_VR001-VR006()    | Entities, rels       | Rule results
6    | filter_validated_data()   | KG + results         | Filtered KG
7    | save_validated_json()     | Filtered KG          | File path
8    | generate_report()         | Batch results        | Report file

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 2: SEMANTIC VALIDATION ALGORITHM                                    │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION validate_semantic_types(entities):
    correct = 0, incorrect = 0, mismatches = [], cost = 0

    FOR each entity IN entities:
        // CHECK 1: Type in ontology?
        IF entity.type NOT IN ['Industry', 'ReportingFramework', 'Category', 'Metric', 'Model']:
            incorrect++
            mismatches.APPEND(entity_id, reason="Unknown type")
            CONTINUE

        // CHECK 2: LLM validation
        prompt = "Does '{entity.label}' represent a {entity.type}?"
        response = CALL_CLAUDE_API(prompt)
        cost += response.cost

        IF response.answer == "YES":
            correct++
        ELSE:
            incorrect++
            mismatches.APPEND(entity_id, reason=response.reasoning)

    accuracy = (correct / total) * 100
    RETURN {total, correct, incorrect, accuracy, cost, mismatches}

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 3: STRUCTURAL VALIDATION ALGORITHM (Example: VR002)                │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION validate_VR002(entities, relationships):
    // VR002: Entity Type-Specific Schema Compliance

    required_fields = {
        "Metric": [measurement_type, metric_type, unit, code, description],
        "Model": [description, equation, input_variables],
        "Category": [section_title]
    }

    violations = [], valid_count = 0

    FOR each entity IN entities:
        required = required_fields[entity.type]
        missing = []

        FOR field IN required:
            IF field NOT IN entity.properties OR field is EMPTY:
                missing.APPEND(field)

        IF missing.length > 0:
            violations.APPEND({entity_id, missing_fields})
        ELSE:
            valid_count++

    score = (valid_count / total_entities) * 100
    passed = (violations.length == 0)

    RETURN {rule_id: "VR002", passed, score, violations}

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 4: DATA FILTERING ALGORITHM (CASCADE EFFECT)                       │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION filter_validated_data(kg_data, validation_results):
    original_entities = kg_data.entities
    original_rels = kg_data.relationships

    // Get failed entities from semantic validation
    failed_ids = SET([m.entity_id FOR m IN validation_results.mismatches])

    // STEP 1: Filter entities
    validated_entities = [e FOR e IN original_entities IF e.id NOT IN failed_ids]

    IF validated_entities.length == 0:
        RETURN NULL  // Signal: Don't save this document

    // STEP 2: Filter relationships (cascade)
    valid_ids = SET([e.id FOR e IN validated_entities])
    validated_rels = [r FOR r IN original_rels
                      IF r.subject IN valid_ids AND r.object IN valid_ids]

    // STEP 3: Calculate metrics
    rel_retention = (validated_rels.count / original_rels.count) * 100

    quality_metrics = {
        schema_compliance: validation_results.compliance_score,
        semantic_accuracy: validation_results.semantic_results.accuracy,
        relationship_retention: rel_retention
    }

    // STEP 4: Build validated KG
    validated_kg = {
        validation_metadata: {
            quality_metrics: quality_metrics,
            original_entity_count: original_entities.length,
            validated_entity_count: validated_entities.length,
            ...
        },
        entities: validated_entities,
        relationships: validated_rels
    }

    RETURN validated_kg

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 5: BATCH PROCESSING ALGORITHM                                       │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION validate_batch(kg_dir, output_dir):
    json_files = FIND_FILES(kg_dir, "*.json")
    batch_results = [], files_saved = 0

    FOR each json_file IN json_files:
        // Validate document
        result = validate_python(json_file)
        batch_results.APPEND(result)

        // IMMEDIATE SAVE (if output_dir provided)
        IF output_dir AND result.success:
            doc_name = EXTRACT_NAME(json_file)
            kg_data = LOAD_JSON(json_file)

            // Filter data
            validated_kg = filter_validated_data(kg_data, result)

            // Save if entities remain
            IF validated_kg is NOT NULL:
                save_validated_json(validated_kg, doc_name, output_dir)
                IF is_ontology_guided:
                    save_validated_rdf(validated_kg, doc_name, output_dir)
                files_saved++
            ELSE:
                PRINT("No valid entities - NOT saving")

    RETURN {batch_results, files_saved, ...}

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 6: REPORT GENERATION ALGORITHM                                      │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION generate_comprehensive_report(results, kg_dir, output_dir):
    doc_data = [], metrics_data = []

    // Aggregate data from all documents
    FOR each doc_result IN results.individual_results:
        kg_data = LOAD_JSON(doc_result.file_path)
        metadata = LOAD_JSON(stage1_metadata_path)

        doc_data.APPEND({name, kg_data, metadata, validation: doc_result})

    // Calculate metrics per document
    FOR each doc IN doc_data:
        validated_file = f"{output_dir}/{doc.name}_validated.json"

        IF FILE_EXISTS(validated_file):
            // Normal case: read from validated file
            data = LOAD_JSON(validated_file)
            metrics = data.validation_metadata.quality_metrics
        ELSE:
            // Edge case: all entities filtered (0 valid entities)
            PRINT("No validated file (all entities filtered)")

            // IMPORTANT: Schema compliance = 0% when all filtered
            metrics = {
                schema_compliance: 0.0,
                semantic_accuracy: doc.validation.semantic_results.accuracy,
                relationship_retention: 0.0
            }
            validated_entities = 0
            validated_rels = 0

        metrics_data.APPEND({name, metrics, original_counts, validated_counts})

    // Generate report sections
    report = []
    report.APPEND("1. OVERALL STATISTICS")
    report.APPEND(f"Total Docs: {doc_data.length}")
    report.APPEND(f"Total Entities: {SUM(original_entities)}")

    report.APPEND("2. EVALUATION METRICS SUMMARY")
    FOR m IN metrics_data:
        report.APPEND(f"{m.name} | {m.schema_compliance}% | {m.semantic_accuracy}% | {m.rel_retention}%")

    report.APPEND("3. DOCUMENT SUMMARY TABLE")
    FOR doc IN doc_data:
        report.APPEND(f"{doc.name} | Pages: {pages} | Entities: {stage2} → {stage3}")

    WRITE_FILE(f"{output_dir}/batch_validation.txt", report)

================================================================================
KEY DESIGN DECISIONS
================================================================================

1. TWO-STAGE VALIDATION
   Reason: Semantic filtering (gate-keeper) removes wrong-type entities before
           structural validation, preventing misleading high compliance scores

2. IMMEDIATE-SAVE PATTERN
   Reason: Fault-tolerant (files saved right after validation), real-time
           output (user sees progress immediately)

3. ENTITY-LEVEL FILTERING
   Reason: Remove bad entities individually (not whole documents), preserve
           as much valid data as possible

4. CASCADE FILTERING
   Reason: Relationships referencing removed entities must also be removed
           to maintain graph integrity

5. ZERO-ENTITY HANDLING
   Reason: Documents with all entities filtered show 0% compliance (not
           misleading pre-filtering scores) and included in report

6. THREE CORE METRICS
   Reason: Simple, interpretable, covers all validation aspects (semantic,
           structural, graph connectivity)

================================================================================
USAGE
================================================================================

Command:
  python3 src/stage3_validation.py \
    --kg-dir outputs/stage2_ontology_guided_extraction \
    --output-dir outputs/stage3_ontology_guided_validation

Output Files:
  • *_validated.json (per document with >0 entities)
  • *_validated.rdf (ontology-guided only)
  • batch_*_validation.txt (comprehensive report)

For More Information:
  • Validation Rules: validation_rules.json
  • Metrics Reference: result_visualisation_and_analysis/METRICS_REFERENCE.txt
  • Verification: VALIDATION_VERIFICATION.md

================================================================================
END OF STAGE 3 VALIDATION README
================================================================================
