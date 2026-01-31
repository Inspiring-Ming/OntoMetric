================================================================================
STAGE 2: ONTOLOGY-GUIDED KNOWLEDGE EXTRACTION
================================================================================

Purpose: Extract structured knowledge graphs from segmented PDF content using
         LLM-based extraction guided by the ESGMKG ontology schema.

Version: 2.0
Last Updated: 2025-11-20

================================================================================
PART 1: WORKFLOW AND DESIGN LOGIC
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2 EXTRACTION PIPELINE (7 STEPS)                   │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT:
  • Stage 1 Segmented Content (JSON files)
    - outputs/stage1_segments/*_segments.json
    - outputs/stage1_segments/*_metadata.json

  • ESGMKG Ontology Schema (Stage 2 Extraction Schema)
    - 8 allowed entity types: Industry, ReportingFramework, Category,
      DirectMetric, CalculatedMetric, InputMetric, Model, Implementation
    - Note: DirectMetric/CalculatedMetric/InputMetric are normalized to 'Metric'
      (with metric_type property) in Stage 3 validation
    - 7 allowed predicates: ReportUsing, Include, ConsistOf, IsCalculatedBy,
      RequiresInputFrom, ExecutesWith, ObtainedFrom
    - Note: Stage 3 only validates 5 predicates (excludes ExecutesWith, ObtainedFrom)

  • Ontology-Guided Extraction Prompt
    - Prompts/ontology_guided_extraction_prompt.txt
    - Framework-specific customizations (SASB, TCFD, IFRS)

OUTPUT:
  • Knowledge Graphs (per document)
    - outputs/stage2_ontology_guided_extraction/*_ontology_guided.json
    - outputs/stage2_ontology_guided_extraction/*_ontology_guided.rdf
    - outputs/stage2_ontology_guided_extraction/*_extraction_log.txt

  • LLM Cost Tracking Reports
    - outputs/stage2_ontology_guided_extraction/*_cost_tracking.json
    - outputs/stage2_ontology_guided_extraction/*_cost_tracking.txt

================================================================================
1.1 SEVEN-STEP EXTRACTION WORKFLOW
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD SEGMENT                                                     │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Load text segment from Stage 1 output                           │
│  Input: Document name (e.g., "SASB-commercial-banks")                     │
│  Process:                                                                  │
│    1. Read *_segments.json (contains all text chunks)                     │
│    2. Read *_metadata.json (contains document info)                       │
│    3. For each segment: extract segment_id, section_title, content        │
│  Output: List of segment objects with provenance info                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: APPLY ONTOLOGY-GUIDED PROMPT                                    │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Build extraction prompt using ontology schema                    │
│  Method: Template-based prompt engineering                                │
│                                                                             │
│  Process:                                                                  │
│    1. Identify document type (SASB/TCFD/IFRS/Base)                        │
│    2. Load base extraction prompt template                                │
│    3. Customize prompt for document type:                                 │
│       • SASB: Industry-specific metrics, quantitative units              │
│       • TCFD: Climate risks, governance, scenario analysis               │
│       • IFRS: Sustainability disclosures, materiality                    │
│    4. Inject segment data into prompt:                                    │
│       • {document_id}: Document identifier                               │
│       • {section_title}: Section heading                                 │
│       • {text_block}: Actual segment text content                        │
│                                                                             │
│  Output: Customized extraction prompt ready for LLM                        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: LLM EXTRACTION                                                   │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Extract structured entities and relationships using LLM         │
│  Method: API call to Claude/GPT with ontology-guided prompt               │
│                                                                             │
│  Process:                                                                  │
│    1. Call LLM API (Anthropic Claude / OpenAI GPT / Azure OpenAI)        │
│    2. Pass extraction prompt + segment text                               │
│    3. LLM returns JSON response:                                          │
│       {                                                                    │
│         "entities": [                                                      │
│           {                                                                │
│             "id": "metric_001",                                           │
│             "type": "DirectMetric",                                       │
│             "label": "Scope 1 Emissions",                                 │
│             "properties": {                                               │
│               "measurement_type": "Quantitative",                         │
│               "metric_type": "DirectMetric",                              │
│               "unit": "Metric tons CO2-e",                                │
│               "code": "TC-SI-110a.1",                                     │
│               "description": "..."                                        │
│             }                                                              │
│           }                                                                │
│         ],                                                                 │
│         "relationships": [                                                 │
│           {                                                                │
│             "subject": "category_001",                                    │
│             "predicate": "ConsistOf",                                     │
│             "object": "metric_001"                                        │
│           }                                                                │
│         ]                                                                  │
│       }                                                                    │
│    4. Parse JSON response                                                 │
│    5. Track token usage and cost (if cost tracking enabled)              │
│                                                                             │
│  Metrics: Input tokens, output tokens, LLM cost                           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: ADD PROVENANCE                                                   │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Link extracted entities/relationships back to source segment     │
│  Method: Inject provenance metadata into each extracted item              │
│                                                                             │
│  For each entity:                                                          │
│    entity.provenance = {                                                   │
│      "document_id": "SASB-commercial-banks",                              │
│      "segment_id": "seg_003",                                             │
│      "section_title": "Greenhouse Gas Emissions",                         │
│      "extraction_timestamp": "2025-11-20T10:30:45"                        │
│    }                                                                       │
│                                                                             │
│  For each relationship:                                                    │
│    relationship.provenance = {                                             │
│      "document_id": "SASB-commercial-banks",                              │
│      "segment_id": "seg_003",                                             │
│      "section_title": "Greenhouse Gas Emissions",                         │
│      "extraction_timestamp": "2025-11-20T10:30:45"                        │
│    }                                                                       │
│                                                                             │
│  Result: 100% traceability (every entity/relationship → segment → pages)  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: QUALITY SCORING                                                  │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Assess extraction quality for each segment                      │
│  Method: Rule-based validation checks                                     │
│                                                                             │
│  Quality Checks:                                                           │
│    1. Entity Type Validity                                                │
│       • Count entities with valid types (in ontology)                     │
│       • Score = (valid entities / total entities)                         │
│                                                                             │
│    2. Relationship Type Validity                                          │
│       • Count relationships with valid predicates (in ontology)           │
│       • Score = (valid relationships / total relationships)               │
│                                                                             │
│    3. CalculatedMetric-Model Links (Constraint)                           │
│       • Check: IsCalculatedBy subject must be CalculatedMetric            │
│       • Flag violations                                                    │
│                                                                             │
│    4. Category-Metric Section Alignment (Constraint)                      │
│       • Check: ConsistOf links should be from same section                │
│       • Flag cross-section links                                          │
│                                                                             │
│    5. Relationship Reference Integrity                                    │
│       • Check: subject/object IDs must exist in entity list               │
│       • Flag dangling references                                          │
│                                                                             │
│  Overall Quality Score = Average of all factor scores (0-1.0)             │
│  Validation Notes = List of issues found                                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: CONSOLIDATE & FILTERING                                         │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Merge results from all segments, remove duplicates              │
│  Method: Three-pass consolidation algorithm                               │
│                                                                             │
│  PASS 1: Fix Duplicate IDs (ID Renumbering)                               │
│    • Problem: Same ID used across multiple segments                       │
│    • Solution: Renumber duplicates with unique IDs                        │
│    • Example: metric_001 (dup) → metric_001_001, metric_001_002          │
│    • Track ID mappings: old_id → new_id                                   │
│    • Update relationships to use new IDs                                  │
│                                                                             │
│  PASS 2: Merge Entities by Semantic Key (Deduplication)                   │
│    • For Metrics: Merge by 'code' (same metric across segments)          │
│      - Keep entity with more detail (CalculatedMetric > DirectMetric)    │
│      - Merge provenance lists (track all source segments)                │
│    • For Others: Merge by 'label' (exact string match)                   │
│      - Keep first occurrence, merge provenance                            │
│    • Track entity merge mappings: merged_id → kept_id                    │
│    • Update relationships to use kept IDs                                 │
│                                                                             │
│  PASS 3: Deduplicate Relationships (Unique Tuples)                        │
│    • Use tuple (subject, predicate, object) as unique key                │
│    • Keep first occurrence, discard duplicates                            │
│                                                                             │
│  Result: Consolidated knowledge graph (unique entities + relationships)   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: GENERATE OUTPUT FILES                                            │
│  ══════════════════════════════════════════════════════════════════════    │
│  Purpose: Save extraction results in multiple formats                     │
│                                                                             │
│  OUTPUT 1: JSON Knowledge Graph (*_ontology_guided.json)                  │
│    {                                                                       │
│      "entities": [...],         // List of all entities with provenance   │
│      "relationships": [...],    // List of all relationships              │
│      "metadata": {                                                         │
│        "document_name": "...",                                            │
│        "total_entities": 53,                                              │
│        "total_relationships": 53,                                         │
│        "entity_breakdown": {...},                                         │
│        "relationship_breakdown": {...},                                   │
│        "extraction_timestamp": "..."                                      │
│      }                                                                     │
│    }                                                                       │
│                                                                             │
│  OUTPUT 2: RDF Knowledge Graph (*_ontology_guided.rdf)                    │
│    @prefix esg: <http://example.org/esg-ontology#> .                     │
│    esg:metric_001 a esg:DirectMetric ;                                   │
│      rdfs:label "Scope 1 Emissions" ;                                    │
│      esg:unit "Metric tons CO2-e" .                                      │
│    esg:category_001 esg:ConsistOf esg:metric_001 .                       │
│                                                                             │
│  OUTPUT 3: Extraction Log (*_extraction_log.txt)                          │
│    • Per-segment extraction statistics                                    │
│    • Quality scores and validation notes                                  │
│    • Entity/relationship counts                                           │
│                                                                             │
│  OUTPUT 4: Cost Tracking Reports (*_cost_tracking.json/txt)               │
│    • Per-segment token usage (input/output tokens)                        │
│    • Model used (e.g., claude-sonnet-4.5)                                │
│    • Cost calculation ($0.0053 per document)                              │
│    • Page numbers for each segment                                        │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
1.2 BATCH PROCESSING WORKFLOW
================================================================================

START → Process document 1 → (Steps 1-7) → Save outputs
          │
          ├─→ Process document 2 → (Steps 1-7) → Save outputs
          │
          └─→ Process document N → (Steps 1-7) → Save outputs → END

Key: Each document processed independently, outputs saved immediately

================================================================================
PART 2: IMPLEMENTATION ALGORITHMS (PSEUDO-CODE)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 1: MAIN EXTRACTION FLOW                                            │
└─────────────────────────────────────────────────────────────────────────────┘

Step | Function                        | Input                | Output
-----|----------------------------------|----------------------|-------------------
1    | main()                          | CLI args             | Exit code
2    | process_document()              | Document name        | Extraction summary
3    | _extract_from_segment()         | Segment + metadata   | ExtractionResult
4    | _build_extraction_prompt()      | Segment + ontology   | Prompt string
5    | _call_llm_for_extraction()      | Prompt               | Entities + Rels
6    | _validate_extraction()          | Entities + Rels      | Quality score
7    | _consolidate_results()          | All entities/rels    | Consolidated KG
8    | _save_extraction_results()      | Consolidated KG      | File paths

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 2: ONTOLOGY-GUIDED KNOWLEDGE EXTRACTION ALGORITHM                  │
│  (Aligned with User's LaTeX Algorithm 2)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

ALGORITHM: Ontology-Guided Knowledge Extraction (Stage 2)

INPUT:
  • S: Segments (from Stage 1)
  • O: Ontology schema (ESGMKG)
  • L: LLM service (Claude/GPT)

OUTPUT:
  • G = (E, R): Knowledge graph
    - E: Set of entities
    - R: Set of relationships

PROCEDURE:

1. INITIALIZE
   E ← ∅  // Empty entity set
   R ← ∅  // Empty relationship set

2. FOR each segment s IN S DO:

   2.1 BUILD ONTOLOGY-GUIDED PROMPT
       prompt ← BuildPrompt(s, O)
       // Inject: document_id, section_title, text_block
       // Add: ontology schema constraints (entity types, predicates)
       // Customize: by document type (SASB/TCFD/IFRS)

   2.2 CALL LLM FOR EXTRACTION
       response ← L(prompt)
       // API call with ontology-guided prompt
       // Returns JSON: {entities: [...], relationships: [...]}

   2.3 PARSE JSON RESPONSE
       extraction ← ParseJSON(response)
       // Extract entities and relationships from LLM response
       // Handle JSON cleaning (remove markdown code blocks)

   2.4 VALIDATE AND ADD ENTITIES
       FOR each entity e IN extraction.entities DO:
           // Add provenance metadata
           e.provenance ← {
               document_id: s.document_id,
               segment_id: s.segment_id,
               section_title: s.section_title,
               extraction_timestamp: NOW()
           }
           
           // Validate entity against ontology
           IF ValidateEntity(e, O) THEN:
               E ← E ∪ {e}  // Add to entity set
           ELSE:
               LogWarning("Invalid entity type: " + e.type)

   2.5 VALIDATE AND ADD RELATIONSHIPS
       FOR each relation r IN extraction.relationships DO:
           // Add provenance metadata
           r.provenance ← {
               document_id: s.document_id,
               segment_id: s.segment_id,
               section_title: s.section_title,
               extraction_timestamp: NOW()
           }
           
           // Validate relationship against ontology
           IF ValidateRelationship(r, O, E) THEN:
               R ← R ∪ {r}  // Add to relationship set
           ELSE:
               LogWarning("Invalid predicate: " + r.predicate)

   2.6 CHECK CONSTRAINTS (Quality Scoring)
       violations ← CheckConstraints(E, R, O)
       // Check CalculatedMetric-Model links
       // Check Category-Metric section alignment
       // Check relationship reference integrity
       
       IF |violations| > 0 THEN:
           LogViolations(violations)
       
       quality_score ← CalculateQualityScore(E, R, violations)

3. CONSOLIDATE RESULTS
   E_consolidated, R_consolidated ← Consolidate(E, R)
   // Remove duplicate IDs (renumber)
   // Merge duplicate entities (by code/label)
   // Deduplicate relationships

4. RETURN KNOWLEDGE GRAPH
   G ← (E_consolidated, R_consolidated)
   RETURN G

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 3: PROMPT BUILDING ALGORITHM (STEP 2)                              │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION BuildPrompt(segment, ontology):
    // Identify document type
    doc_type ← IdentifyDocumentType(segment.document_title)
    // Options: "SASB", "TCFD", "IFRS", "base"
    
    // Load base prompt template
    base_template ← LoadTemplate("ontology_guided_extraction_prompt.txt")
    
    // Customize for document type
    IF doc_type == "SASB":
        prompt_template ← CustomizeForSASB(base_template)
        // Add: Industry-specific metrics, quantitative units focus
    ELSE IF doc_type == "TCFD":
        prompt_template ← CustomizeForTCFD(base_template)
        // Add: Climate risks, governance, scenario analysis focus
    ELSE IF doc_type == "IFRS":
        prompt_template ← CustomizeForIFRS(base_template)
        // Add: Sustainability disclosures, materiality focus
    ELSE:
        prompt_template ← base_template
    
    // Inject segment data into template
    prompt ← prompt_template.format(
        document_id: segment.document_id,
        section_title: segment.section_title,
        text_block: segment.content
    )
    
    RETURN prompt

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 4: LLM EXTRACTION ALGORITHM (STEP 3)                               │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION CallLLM(prompt, segment_id, page_start, page_end):
    // Get LLM service configuration
    service ← config.llm_service  // "anthropic", "openai", "azure_openai"
    model ← config.model          // "claude-sonnet-4.5", "gpt-4", etc.
    
    TRY:
        // Call appropriate LLM API
        IF service == "anthropic":
            response ← CallAnthropicAPI(prompt, model)
            
            // Track token usage for cost tracking
            IF cost_tracking_enabled:
                input_tokens ← response.usage.input_tokens
                output_tokens ← response.usage.output_tokens
                RecordUsage(segment_id, model, input_tokens, output_tokens,
                           page_start, page_end)
            
        ELSE IF service == "openai":
            response ← CallOpenAIAPI(prompt, model)
            
        ELSE IF service == "azure_openai":
            response ← CallAzureOpenAIAPI(prompt, model)
        
        // Extract text content from response
        content ← response.content.text
        
        // Clean JSON response (remove markdown code blocks)
        IF content.startswith("```json"):
            content ← content[7:]  // Remove "```json"
        IF content.endswith("```"):
            content ← content[:-3]  // Remove "```"
        
        // Parse JSON
        extracted_data ← JSON.parse(content)
        
        RETURN extracted_data
        
    CATCH APIError:
        LogError("LLM API call failed")
        RETURN NULL

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 5: PROVENANCE ADDITION ALGORITHM (STEP 4)                          │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION AddProvenance(entities, relationships, segment):
    provenance_metadata ← {
        document_id: segment.document_id,
        segment_id: segment.segment_id,
        section_title: segment.section_title,
        extraction_timestamp: NOW()
    }
    
    // Add provenance to all entities
    FOR each entity IN entities:
        entity.provenance ← provenance_metadata
    
    // Add provenance to all relationships
    FOR each relationship IN relationships:
        relationship.provenance ← provenance_metadata
    
    // Result: 100% traceability
    // Each entity/relationship → segment_id → segment → page_start/page_end
    
    RETURN entities, relationships

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 6: QUALITY SCORING ALGORITHM (STEP 5)                              │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION ValidateExtraction(entities, relationships, segment):
    validation_notes ← []
    quality_factors ← []
    
    // Build entity lookup map
    entity_map ← {e.id: e FOR e IN entities}
    
    // CHECK 1: Entity Type Validity
    valid_entity_types ← COUNT(e FOR e IN entities WHERE e.type IN ontology.entity_types)
    IF entities.length > 0:
        entity_type_score ← valid_entity_types / entities.length
        quality_factors.APPEND(entity_type_score)
        IF entity_type_score < 1.0:
            validation_notes.APPEND("Some entities have invalid types")
    
    // CHECK 2: Relationship Type Validity
    valid_rel_types ← COUNT(r FOR r IN relationships WHERE r.predicate IN ontology.predicates)
    IF relationships.length > 0:
        rel_type_score ← valid_rel_types / relationships.length
        quality_factors.APPEND(rel_type_score)
        IF rel_type_score < 1.0:
            validation_notes.APPEND("Some relationships have invalid predicates")
    
    // CHECK 3: CalculatedMetric-Model Links (Constraint)
    FOR each rel IN relationships WHERE rel.predicate == "IsCalculatedBy":
        metric ← entity_map[rel.subject]
        IF metric.properties.metric_type != "CalculatedMetric":
            validation_notes.APPEND(
                f"ERROR: '{metric.label}' is {metric_type} but has IsCalculatedBy link"
            )
    
    // CHECK 4: Relationship Reference Integrity
    entity_ids ← SET([e.id FOR e IN entities])
    dangling_refs ← 0
    FOR each rel IN relationships:
        IF rel.subject NOT IN entity_ids OR rel.object NOT IN entity_ids:
            dangling_refs++
            validation_notes.APPEND(f"Dangling reference: {rel}")
    
    // Calculate overall quality score
    IF quality_factors.length > 0:
        quality_score ← AVERAGE(quality_factors)
    ELSE:
        quality_score ← 0.0
    
    RETURN quality_score, validation_notes

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 7: CONSOLIDATION ALGORITHM (STEP 6)                                │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION ConsolidateResults(all_entities, all_relationships):
    // PASS 1: Fix Duplicate IDs
    unique_entities ← {}
    id_counter ← {}
    id_mapping ← {}  // Track old_id → new_id
    
    FOR each entity IN all_entities:
        IF entity.id IN unique_entities:
            // Duplicate ID found - generate new unique ID
            base_id ← ExtractBaseID(entity.id)
            
            // Find next available number
            WHILE f"{base_id}_{id_counter[base_id]:03d}" IN unique_entities:
                id_counter[base_id]++
            
            new_id ← f"{base_id}_{id_counter[base_id]:03d}"
            id_mapping[entity.id] ← new_id  // Track mapping
            entity.id ← new_id
            id_counter[base_id]++
        
        unique_entities[entity.id] ← entity
    
    // Update relationships with new IDs
    FOR each rel IN all_relationships:
        IF rel.subject IN id_mapping:
            rel.subject ← id_mapping[rel.subject]
        IF rel.object IN id_mapping:
            rel.object ← id_mapping[rel.object]
    
    // PASS 2: Merge Entities by Semantic Key
    consolidated_entities ← {}
    entity_merge_mapping ← {}  // Track merged_id → kept_id
    
    FOR each entity IN unique_entities.values():
        // For Metrics: use 'code' as unique key
        IF entity.type == "Metric":
            code ← entity.properties.code
            IF code AND code != "N/A":
                key ← ("Metric", code)
                
                IF key IN consolidated_entities:
                    // Duplicate found - merge
                    existing ← consolidated_entities[key]
                    
                    // Keep entity with more detail
                    IF entity.type == "CalculatedMetric" AND existing.type == "DirectMetric":
                        entity_merge_mapping[existing.id] ← entity.id
                        consolidated_entities[key] ← entity
                    ELSE:
                        entity_merge_mapping[entity.id] ← existing.id
                    
                    // Merge provenance lists
                    MergeProvenance(consolidated_entities[key], entity)
                ELSE:
                    consolidated_entities[key] ← entity
        
        // For Others: use 'label' as unique key
        ELSE:
            key ← (entity.type, entity.label)
            
            IF key IN consolidated_entities:
                // Duplicate found - keep first, merge provenance
                existing ← consolidated_entities[key]
                entity_merge_mapping[entity.id] ← existing.id
                MergeProvenance(existing, entity)
            ELSE:
                consolidated_entities[key] ← entity
    
    // Update relationships with merged entity IDs
    FOR each rel IN all_relationships:
        IF rel.subject IN entity_merge_mapping:
            rel.subject ← entity_merge_mapping[rel.subject]
        IF rel.object IN entity_merge_mapping:
            rel.object ← entity_merge_mapping[rel.object]
    
    // PASS 3: Deduplicate Relationships
    unique_relationships ← {}
    
    FOR each rel IN all_relationships:
        key ← (rel.subject, rel.predicate, rel.object)  // Unique tuple
        IF key NOT IN unique_relationships:
            unique_relationships[key] ← rel
    
    RETURN consolidated_entities.values(), unique_relationships.values()

┌─────────────────────────────────────────────────────────────────────────────┐
│  TABLE 8: OUTPUT GENERATION ALGORITHM (STEP 7)                            │
└─────────────────────────────────────────────────────────────────────────────┘

FUNCTION SaveExtractionResults(document_name, entities, relationships, metadata):
    output_files ← []
    
    // OUTPUT 1: JSON Knowledge Graph
    json_file ← f"{output_dir}/{document_name}_ontology_guided.json"
    json_data ← {
        "entities": [entity.to_dict() FOR entity IN entities],
        "relationships": [rel.to_dict() FOR rel IN relationships],
        "metadata": {
            "document_name": document_name,
            "total_entities": entities.length,
            "total_relationships": relationships.length,
            "entity_breakdown": GetEntityBreakdown(entities),
            "relationship_breakdown": GetRelationshipBreakdown(relationships),
            "extraction_timestamp": NOW()
        }
    }
    WRITE_JSON(json_file, json_data)
    output_files.APPEND(json_file)
    
    // OUTPUT 2: RDF Knowledge Graph
    rdf_file ← f"{output_dir}/{document_name}_ontology_guided.rdf"
    rdf_content ← ConvertToRDF(entities, relationships, metadata)
    WRITE_FILE(rdf_file, rdf_content)
    output_files.APPEND(rdf_file)
    
    // OUTPUT 3: Extraction Log
    log_file ← f"{output_dir}/{document_name}_extraction_log.txt"
    log_content ← GenerateExtractionLog(extraction_results, metadata)
    WRITE_FILE(log_file, log_content)
    output_files.APPEND(log_file)
    
    // OUTPUT 4: Cost Tracking Reports (if enabled)
    IF cost_tracking_enabled:
        cost_json ← cost_tracker.save_detailed_report(document_name)
        cost_txt ← cost_tracker.save_human_readable_report(document_name)
        output_files.APPEND(cost_json, cost_txt)
    
    RETURN output_files

================================================================================
KEY DESIGN DECISIONS
================================================================================

1. ONTOLOGY-GUIDED PROMPTS
   Reason: Constrain LLM output to valid entity types and predicates,
           improve extraction accuracy by providing schema examples

2. FRAMEWORK-SPECIFIC CUSTOMIZATION
   Reason: Different frameworks (SASB/TCFD/IFRS) have different terminology
           and focus areas, customized prompts improve extraction quality

3. 100% PROVENANCE TRACKING
   Reason: Every entity/relationship must be traceable back to source segment
           and page numbers for verification and debugging

4. THREE-PASS CONSOLIDATION
   Reason: Entities can appear in multiple segments with different IDs or
           slightly different labels, need systematic deduplication

5. QUALITY SCORING PER SEGMENT
   Reason: Track extraction quality at granular level, identify problematic
           segments for re-extraction or manual review

6. IMMEDIATE COST TRACKING
   Reason: Track LLM API costs in real-time (per segment, per page) for
           budget management and optimization

7. MULTIPLE OUTPUT FORMATS
   Reason: JSON for downstream processing, RDF for semantic web tools,
           logs for debugging, cost reports for analysis

================================================================================
USAGE
================================================================================

Single Document:
  python3 src/stage2_ontology_guided_extraction.py \
    --document "SASB-commercial-banks-standard_en-gb"

Batch Processing:
  python3 src/stage2_ontology_guided_extraction.py \
    --batch

Output Files (per document):
  • {document_name}_ontology_guided.json (entities + relationships)
  • {document_name}_ontology_guided.rdf (RDF graph)
  • {document_name}_extraction_log.txt (extraction statistics)
  • {document_name}_cost_tracking.json (detailed cost data)
  • {document_name}_cost_tracking.txt (human-readable cost report)

Configuration:
  • config_llm.json (LLM service settings: API key, model, temperature, etc.)
  • Prompts/ontology_guided_extraction_prompt.txt (base extraction prompt)

For More Information:
  • Next Stage: src/stage3_validation_README.txt
  • Ontology Schema: validation_rules.json
  • Metrics Reference: result_visualisation_and_analysis/METRICS_REFERENCE.txt

================================================================================
END OF STAGE 2 EXTRACTION README
================================================================================
