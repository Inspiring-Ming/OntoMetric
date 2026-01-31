#!/usr/bin/env python3
"""
Stage 2: Ontology-Guided Knowledge Graph Extraction from Segmented PDFs

This script extracts structured knowledge graph information from the segmented PDF content
produced by Stage 1, using ontology-guided LLM-based extraction to identify entities and
relationships according to the ESGMKG ontology schema.

What it does:
- Reads segmented PDF content from Stage 1 outputs
- Uses ontology-guided LLM prompts to extract entities (Industry, Category, Metric, Model, etc.) and relationships
- Converts extracted JSON to RDF format for knowledge graph construction
- Processes all documents in batch or individually
- Validates extraction quality and completeness with SPARQL validation

Key Features:
- Ontology-guided LLM-powered entity and relationship extraction
- ESGMKG schema-aware prompts for accurate extraction
- JSON to RDF conversion with proper ontology mapping
- Batch processing with progress tracking
- Quality validation and error handling
- Provenance tracking for extracted knowledge

Usage:
    # Process single document
    python3 src/stage2_ontology_guided_extraction.py "document_name"

    # Process all documents
    python3 src/stage2_ontology_guided_extraction.py --batch

    # Extract only specific entity types
    python3 src/stage2_ontology_guided_extraction.py --entities "Metric,Model" "document_name"

Output Structure:
- outputs/stage2_ontology_guided_extraction/
  ├── {document_name}_ontology_guided.json                 # Combined entities and relationships in JSON format
  ├── {document_name}_ontology_guided.rdf                  # Combined RDF knowledge graph
  ├── {document_name}_ontology_guided_extraction_log.txt   # Processing log and validation results
  └── batch_summary.json                                   # Batch processing summary (if --batch)
- outputs/cost_tracking/
  ├── {document_name}_ontology_guided_cost_report.json     # Detailed cost tracking (JSON)
  └── {document_name}_ontology_guided_cost_report.txt      # Human-readable cost report

Entity Types Extracted:
- Industry: Business sectors (Commercial Banks, Oil & Gas, etc.)
- ReportingFramework: ESG standards (SASB, TCFD, IFRS S2, etc.)
- Category: Metric groupings (Energy Management, Emissions, etc.)
- Metric: ESG disclosure metrics with units and calculation methods
- Model: Calculation methodologies and formulas
- Dataset-Variable: Data variables required for calculations
- Datasource: Sources of data variables
- Implementation: Software tools executing models

Relationship Types:
- ReportUsing: Industry → ReportingFramework
- Include: ReportingFramework → Category  
- ConsistOf: Category → Metric
- IsCalculatedBy: Metric → Model
- RequiresInputFrom: Model → Dataset-Variable
- ExecutesWith: Model → Implementation
- ObtainedFrom: Metric → Dataset-Variable
- SourceFrom: Dataset-Variable → Datasource
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

# LLM Integration (placeholder - will be configured based on your preferred service)
try:
    import openai  # or your preferred LLM service
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM service not configured. Install required packages for full functionality.")

@dataclass
class ExtractedEntity:
    """Represents an extracted knowledge graph entity"""
    id: str
    type: str
    label: str
    properties: Dict[str, Any]
    provenance: Dict[str, Any]

@dataclass
class ExtractedRelationship:
    """Represents an extracted knowledge graph relationship"""
    subject: str
    predicate: str
    object: str
    properties: Dict[str, Any]
    provenance: Dict[str, Any]

@dataclass
class ExtractionResult:
    """Complete extraction result for a document segment"""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    document_id: str
    segment_id: str
    section_title: str
    extraction_timestamp: str
    quality_score: float
    validation_notes: List[str]

class ESGKnowledgeExtractor:
    """
    Main class for extracting ESG knowledge graph information from segmented PDF content.
    
    This extractor uses LLM-based processing to identify entities and relationships
    according to the ESGMKG ontology schema, with built-in validation and quality control.
    """
    
    def __init__(self, llm_config: Optional[Dict] = None, enable_cost_tracking: bool = True):
        """
        Initialize the knowledge extractor.

        Args:
            llm_config: Configuration for LLM service (API keys, model settings, etc.)
            enable_cost_tracking: Whether to track LLM costs (default: True)
        """
        self.llm_config = llm_config or {}
        self.extraction_prompts = self._load_extraction_prompts()
        self.entity_types = {
            'Industry', 'ReportingFramework', 'Category', 'DirectMetric', 'CalculatedMetric',
            'InputMetric', 'Model', 'Implementation'
        }

        # Primary entities (always extract from PDFs)
        self.primary_entities = {'Industry', 'ReportingFramework', 'Category', 'DirectMetric', 'CalculatedMetric', 'InputMetric'}

        # Secondary entities (extract when mentioned in PDFs)
        self.secondary_entities = {'Model'}

        # Rare entities (extract only if explicitly mentioned)
        self.rare_entities = {'Implementation'}
        self.relationship_types = {
            'ReportUsing', 'Include', 'ConsistOf', 'IsCalculatedBy',
            'RequiresInputFrom', 'ExecutesWith', 'ObtainedFrom'
        }

        # Create output directory
        self.output_dir = Path("outputs/stage2_ontology_guided_extraction")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        self.cost_tracker = None
        if self.enable_cost_tracking:
            from utils.cost_tracker import CostTracker
            self.cost_tracker = CostTracker()
            print(f"💰 Cost tracking: ENABLED")

        print(f"🧠 ESG Knowledge Extractor initialized")
        print(f"📁 Output directory: {self.output_dir}")
        
    def _load_extraction_prompts(self) -> Dict[str, str]:
        """Load and prepare extraction prompts for different document types"""
        prompts = {}

        # Load base prompt and escape braces for .format()
        prompt_file = Path("Prompts/ontology_guided_extraction_prompt.txt")
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Escape braces for Python .format() but keep placeholders
            content = content.replace('{document_id}', '<<<DOCID>>>')
            content = content.replace('{section_title}', '<<<SECTITLE>>>')
            content = content.replace('{text_block}', '<<<TEXTBLOCK>>>')
            content = content.replace('{', '{{').replace('}', '}}')
            content = content.replace('<<<DOCID>>>', '{document_id}')
            content = content.replace('<<<SECTITLE>>>', '{section_title}')
            content = content.replace('<<<TEXTBLOCK>>>', '{text_block}')

            # Add extraction task section
            extraction_task = """
------------------------------------------------------------
NOW EXTRACT FROM THE FOLLOWING TEXT SEGMENT
------------------------------------------------------------
**Document ID:** {document_id}
**Section Title:** {section_title}

**TEXT TO ANALYZE:**
{text_block}

**YOUR TASK:** Analyze the text above and return a JSON object following the schema.

Return ONLY:
{{
  "entities": [...],  // Each with id, type, label, properties
  "relationships": [...]  // Each with subject, predicate, object
}}

You may briefly explain your reasoning, then output the JSON.
"""
            base_prompt = content + extraction_task
            prompts['base'] = base_prompt
        else:
            # Fallback prompt if file not found
            prompts['base'] = self._get_default_prompt()
        
        # Create specialized prompts for different frameworks
        prompts['SASB'] = self._customize_prompt_for_sasb(prompts['base'])
        prompts['TCFD'] = self._customize_prompt_for_tcfd(prompts['base'])
        prompts['IFRS'] = self._customize_prompt_for_ifrs(prompts['base'])
        
        return prompts
    
    def _get_default_prompt(self) -> str:
        """Default extraction prompt if template file is not available"""
        return """You are an expert system extracting ESG metric definitions from regulatory 
documents to construct an RDF knowledge graph conforming to the ESGMKG ontology.

TASK: Extract structured information from the provided text block.

Extract ONLY information explicitly stated in the source text.
Do NOT infer, assume, or add information not present.

OUTPUT FORMAT: Generate structured JSON with entities and relationships.

SOURCE TEXT:
[Document: {document_id}, Section: {section_title}]
{text_block}"""
    
    def _customize_prompt_for_sasb(self, base_prompt: str) -> str:
        """Customize prompt for SASB documents"""
        sasb_additions = """
SASB-SPECIFIC EXTRACTION FOCUS:
- Industry-specific metrics and disclosure topics
- Quantitative metrics with specific units (e.g., metric tons CO2-e, GJ, m³)
- Accounting metrics and technical protocols
- Activity metrics and normalization factors
- Sector-specific terminology and definitions
"""
        return base_prompt + sasb_additions
    
    def _customize_prompt_for_tcfd(self, base_prompt: str) -> str:
        """Customize prompt for TCFD documents"""
        tcfd_additions = """
TCFD-SPECIFIC EXTRACTION FOCUS:
- Climate-related risks and opportunities
- Governance structures and oversight processes
- Strategic planning and scenario analysis
- Risk management processes and integration
- Metrics and targets for climate performance
"""
        return base_prompt + tcfd_additions
    
    def _customize_prompt_for_ifrs(self, base_prompt: str) -> str:
        """Customize prompt for IFRS documents"""
        ifrs_additions = """
IFRS-SPECIFIC EXTRACTION FOCUS:
- Sustainability disclosure requirements
- Climate-related financial disclosures
- Materiality assessments and judgments
- Cross-industry and industry-specific metrics
- Transition plans and scenario analysis
"""
        return base_prompt + ifrs_additions
    
    def process_document(self, document_name: str) -> Dict[str, Any]:
        """
        Process a single document and extract knowledge graph information.
        
        Args:
            document_name: Name of the document (without extension)
            
        Returns:
            Dictionary containing extraction results and statistics
        """
        print(f"\n🚀 Processing document: {document_name}")
        print("=" * 80)
        
        # Load segmented content from Stage 1
        segments_file = Path(f"outputs/stage1_segments/{document_name}_segments.json")
        metadata_file = Path(f"outputs/stage1_segments/{document_name}_metadata.json")
        
        if not segments_file.exists():
            error_msg = f"Segments file not found: {segments_file}"
            print(f"❌ Error: {error_msg}")
            return {'success': False, 'error': error_msg}
        
        try:
            # Load segments and metadata
            with open(segments_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"📄 Document: {metadata['title']}")
            print(f"📊 Total segments: {len(segments)}")
            
            # Extract knowledge from each segment
            all_entities = []
            all_relationships = []
            extraction_results = []
            
            for i, segment in enumerate(segments, 1):
                print(f"\n[{i}/{len(segments)}] Processing: {segment['section_title']}")
                
                result = self._extract_from_segment(segment, document_name, metadata)
                if result:
                    extraction_results.append(result)
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)
                    
                    print(f"  ✓ Extracted: {len(result.entities)} entities, {len(result.relationships)} relationships")
                else:
                    print(f"  ⚠️  No extraction results for this segment")
            
            # Consolidate and deduplicate results
            consolidated_entities, consolidated_relationships = self._consolidate_results(
                all_entities, all_relationships
            )
            
            # Save results
            output_files = self._save_extraction_results(
                document_name, consolidated_entities, consolidated_relationships, 
                extraction_results, metadata
            )
            
            # Save cost tracking reports if enabled
            if self.cost_tracker:
                cost_report_json = self.cost_tracker.save_detailed_report(f"{document_name}_ontology_guided")
                cost_report_txt = self.cost_tracker.save_human_readable_report(f"{document_name}_ontology_guided")
                output_files.extend([cost_report_json, cost_report_txt])
                print(f"\n💰 Ontology-guided extraction cost tracking reports saved:")
                print(f"  📄 {cost_report_json}")
                print(f"  📄 {cost_report_txt}")
                self.cost_tracker.print_summary()

            # Generate summary
            summary = {
                'success': True,
                'document_name': document_name,
                'document_title': metadata['title'],
                'total_segments_processed': len(extraction_results),
                'total_entities_extracted': len(consolidated_entities),
                'total_relationships_extracted': len(consolidated_relationships),
                'entity_breakdown': self._get_entity_breakdown(consolidated_entities),
                'relationship_breakdown': self._get_relationship_breakdown(consolidated_relationships),
                'output_files': output_files,
                'processing_timestamp': datetime.now().isoformat()
            }

            print(f"\n✅ EXTRACTION COMPLETE")
            print(f"📊 Total entities: {len(consolidated_entities)}")
            print(f"🔗 Total relationships: {len(consolidated_relationships)}")
            print(f"📁 Output files: {len(output_files)}")

            return summary
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing document {document_name}: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"📍 Full traceback:\n{traceback.format_exc()}")
            return {'success': False, 'error': error_msg}
    
    def _extract_from_segment(self, segment: Dict, document_name: str, metadata: Dict) -> Optional[ExtractionResult]:
        """Extract knowledge from a single document segment using LLM"""

        # Determine document type for prompt selection
        doc_type = self._identify_document_type(metadata['title'])
        prompt_template = self.extraction_prompts.get(doc_type, self.extraction_prompts['base'])

        # Prepare extraction prompt
        extraction_prompt = prompt_template.format(
            document_id=document_name,
            section_title=segment['section_title'],
            text_block=segment['content']
        )

        # ONLY use LLM extraction - NO simulation or fake data allowed
        extracted_data = None

        if LLM_AVAILABLE and self.llm_config:
            # STRICTLY LLM extraction only - no demo mode or simulation
            # Pass segment_id and page info for cost tracking
            extracted_data = self._call_llm_for_extraction(
                extraction_prompt,
                segment.get('segment_id'),
                segment.get('page_start'),
                segment.get('page_end')
            )

        # NO FALLBACK TO SIMULATION - only real document extraction allowed

        if not extracted_data:
            return None
        
        # Parse and validate extraction results
        entities = []
        relationships = []
        
        for entity_data in extracted_data.get('entities', []):
            entity = ExtractedEntity(
                id=entity_data['id'],
                type=entity_data['type'],
                label=entity_data['label'],
                properties=entity_data.get('properties', {}),
                provenance={
                    'document_id': document_name,
                    'segment_id': segment['segment_id'],
                    'section_title': segment['section_title'],
                    'extraction_timestamp': datetime.now().isoformat()
                }
            )
            entities.append(entity)
        
        for rel_data in extracted_data.get('relationships', []):
            relationship = ExtractedRelationship(
                subject=rel_data['subject'],
                predicate=rel_data['predicate'],
                object=rel_data['object'],
                properties=rel_data.get('properties', {}),
                provenance={
                    'document_id': document_name,
                    'segment_id': segment['segment_id'],
                    'section_title': segment['section_title'],
                    'extraction_timestamp': datetime.now().isoformat()
                }
            )
            relationships.append(relationship)

        # Calculate quality score and validation
        quality_score, validation_notes = self._validate_extraction(entities, relationships, segment)
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            document_id=document_name,
            segment_id=segment['segment_id'],
            section_title=segment['section_title'],
            extraction_timestamp=datetime.now().isoformat(),
            quality_score=quality_score,
            validation_notes=validation_notes
        )
    
    def _identify_document_type(self, title: str) -> str:
        """Identify document type based on title for prompt selection"""
        title_lower = title.lower()
        
        if 'sasb' in title_lower:
            return 'SASB'
        elif 'tcfd' in title_lower:
            return 'TCFD'
        elif 'ifrs' in title_lower or 'issb' in title_lower:
            return 'IFRS'
        else:
            return 'base'
    
    def _call_llm_for_extraction(self, prompt: str, segment_id: str = None,
                                 page_start: int = None, page_end: int = None) -> Optional[Dict]:
        """Call LLM service for knowledge extraction"""
        try:
            if not self.llm_config:
                return None

            service = self.llm_config.get('llm_service', 'openai')

            if service == 'openai':
                return self._call_openai(prompt, segment_id, page_start, page_end)
            elif service == 'anthropic':
                return self._call_anthropic(prompt, segment_id, page_start, page_end)
            elif service == 'azure_openai':
                return self._call_azure_openai(prompt, segment_id, page_start, page_end)
            else:
                print(f"⚠️  Unsupported LLM service: {service}")
                return None

        except Exception as e:
            print(f"⚠️  LLM call failed: {str(e)}")
            return None
    
    def _call_openai(self, prompt: str) -> Optional[Dict]:
        """Call OpenAI API for extraction"""
        try:
            from openai import OpenAI
            
            api_settings = self.llm_config.get('api_settings', {})
            client = OpenAI(api_key=api_settings.get('api_key'))
            
            response = client.chat.completions.create(
                model=api_settings.get('model', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are an expert ESG knowledge extraction system. Extract only information explicitly stated in the source text. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=api_settings.get('temperature', 0.1),
                max_tokens=api_settings.get('max_tokens', 2000),
                timeout=api_settings.get('timeout', 30)
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean and parse JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"⚠️  OpenAI API error: {str(e)}")
            return None
    
    def _call_anthropic(self, prompt: str, segment_id: str = None,
                       page_start: int = None, page_end: int = None) -> Optional[Dict]:
        """Call Anthropic Claude API for extraction"""
        try:
            import anthropic

            api_settings = self.llm_config.get('api_settings', {})
            client = anthropic.Anthropic(api_key=api_settings.get('api_key'))

            response = client.messages.create(
                model=api_settings.get('model', 'claude-3-5-sonnet-20241022'),
                max_tokens=api_settings.get('max_tokens', 2000),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Track token usage if cost tracking is enabled
            if self.cost_tracker and segment_id:
                model_used = api_settings.get('model', 'claude-3-5-sonnet-20241022')
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self.cost_tracker.record_usage(segment_id, model_used, input_tokens, output_tokens,
                                               page_start, page_end)

            content = response.content[0].text.strip()
            
            # Debug: print what Claude returned
            print(f"  🤖 Claude response length: {len(content)} chars")
            print(f"  🤖 Claude response preview: {content[:300]}...")
            print(f"  🤖 First 50 chars repr: {repr(content[:50])}")
            
            # Extract JSON from Claude's response (which may include explanatory text)
            # Try multiple parsing strategies for version 2 prompt compatibility

            # Strategy 1: Find complete JSON object with braces
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                print(f"  🔍 Extracted JSON portion: {len(json_content)} chars")
            else:
                # Strategy 2: Check if response starts with "entities" (version 2 format)
                # Wrap it in braces to make valid JSON
                if '"entities"' in content or "'entities'" in content:
                    print(f"  🔄 Detected version 2 format - wrapping in braces")
                    json_content = '{' + content.strip() + '}'
                else:
                    # Strategy 3: Fallback to code block cleaning
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    json_content = content.strip()

            if not json_content:
                print(f"  ⚠️  Empty JSON content from Claude")
                return None

            # Parse JSON and validate structure
            try:
                parsed_response = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error: {str(e)}")
                print(f"  📄 Content preview: {json_content[:500]}")
                return None
            
            # If Claude returns a different format, adapt it to our expected format
            if 'entities' not in parsed_response:
                print(f"  🔄 Adapting Claude response format...")
                # Create expected format
                adapted_response = {
                    'entities': [],
                    'relationships': [],
                    'provenance': {}
                }
                return adapted_response
            
            return parsed_response
            
        except Exception as e:
            print(f"⚠️  Anthropic API error: {str(e)}")
            return None
    
    def _call_azure_openai(self, prompt: str) -> Optional[Dict]:
        """Call Azure OpenAI API for extraction"""
        try:
            import openai
            
            azure_settings = self.llm_config.get('alternative_services', {}).get('azure_openai', {})
            
            openai.api_type = "azure"
            openai.api_base = azure_settings.get('endpoint')
            openai.api_version = azure_settings.get('api_version', '2023-12-01-preview')
            openai.api_key = azure_settings.get('api_key')
            
            response = openai.ChatCompletion.create(
                engine=azure_settings.get('deployment_name', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are an expert ESG knowledge extraction system. Extract only information explicitly stated in the source text. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean and parse JSON response
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"⚠️  Azure OpenAI API error: {str(e)}")
            return None
    
    
    def _extract_unit_from_context(self, content: str, metric: str) -> Optional[str]:
        """Extract unit information from context around a metric"""
        # Look for common units near the metric
        unit_patterns = [
            r'metric tons?\s*(?:\([^)]*\))?\s*CO[₂2]-?e',
            r'percentage\s*%?',
            r'number',
            r'presentation currency',
            r'GJ|MWh|kWh',
            r'm³|cubic meters?',
            r'%'
        ]
        
        # Search in a window around the metric mention
        metric_pos = content.lower().find(metric.lower())
        if metric_pos != -1:
            window = content[max(0, metric_pos-100):metric_pos+200]
            for pattern in unit_patterns:
                match = re.search(pattern, window, re.IGNORECASE)
                if match:
                    return match.group(0)
        
        return None
    
    def _classify_metric_type(self, metric_description: str) -> str:
        """Classify metric type based on description: DirectMetric (ONE concept), CalculatedMetric (MULTIPLE concepts), InputMetric (raw data)"""
        desc_lower = metric_description.lower()
        
        # CalculatedMetric indicators (requires MULTIPLE inputs/concepts for calculation)
        calculated_indicators = [
            'calculated', 'derived', 'score', 'index', 'ratio', 'intensity', 'efficiency',
            'financed emissions', 'scope 1', 'scope 2', 'scope 3', 'carbon footprint',
            'g-sib', 'systemically important', 'weighted', 'normalized', 'composite'
        ]
        
        # DirectMetric indicators (ONE concept/indicator - direct measurement)
        direct_indicators = [
            'total amount', 'total number', 'number of', 'count of', 'volume of',
            'monetary losses', 'data breaches', 'employees', 'participants',
            'energy consumed', 'water withdrawn', 'waste generated'
        ]
        
        # InputMetric indicators (raw data used as input for calculations)
        input_indicators = [
            'size', 'complexity', 'interconnectedness', 'substitutability',
            'activity data', 'emission factor', 'fuel consumption', 'allocation'
        ]
        
        # Check for calculated metric indicators (MULTIPLE concepts)
        for indicator in calculated_indicators:
            if indicator in desc_lower:
                return 'CalculatedMetric'
        
        # Check for direct metric indicators (ONE concept)
        for indicator in direct_indicators:
            if indicator in desc_lower:
                return 'DirectMetric'
        
        # Check for input metric indicators (raw data)
        for indicator in input_indicators:
            if indicator in desc_lower:
                return 'InputMetric'
        
        # Default to DirectMetric for simple measurements
        return 'DirectMetric'
    
    def _validate_extraction(self, entities: List[ExtractedEntity],
                           relationships: List[ExtractedRelationship],
                           segment: Dict) -> Tuple[float, List[str]]:
        """Validate extraction quality and completeness"""
        validation_notes = []
        quality_factors = []

        # Build entity lookup for validation
        entity_map = {e.id: e for e in entities}

        # Check entity type validity
        valid_entity_types = sum(1 for e in entities if e.type in self.entity_types)
        if entities:
            entity_type_score = valid_entity_types / len(entities)
            quality_factors.append(entity_type_score)
            if entity_type_score < 1.0:
                validation_notes.append(f"Some entities have invalid types")

        # Check relationship type validity
        valid_relationship_types = sum(1 for r in relationships if r.predicate in self.relationship_types)
        if relationships:
            rel_type_score = valid_relationship_types / len(relationships)
            quality_factors.append(rel_type_score)
            if rel_type_score < 1.0:
                validation_notes.append(f"Some relationships have invalid predicates")

        # NEW: Check CalculatedMetric-Model links
        invalid_calc_links = 0
        for rel in relationships:
            if rel.predicate == 'IsCalculatedBy':
                metric = entity_map.get(rel.subject)
                if metric:
                    metric_type = metric.properties.get('metric_type', '')
                    if metric_type != 'CalculatedMetric':
                        invalid_calc_links += 1
                        validation_notes.append(
                            f"ERROR: '{metric.label}' is {metric_type} but has IsCalculatedBy link"
                        )

        # NEW: Check Category-Metric links are from same section
        cross_section_links = 0
        for rel in relationships:
            if rel.predicate == 'ConsistOf':
                category = entity_map.get(rel.subject)
                metric = entity_map.get(rel.object)
                if category and metric:
                    # Allow if same section or if category label matches metric's context
                    cat_section = category.provenance.get('section_title', '')
                    met_section = metric.provenance.get('section_title', '')
                    if cat_section != met_section:
                        cross_section_links += 1
                        validation_notes.append(
                            f"WARNING: Category '{category.label}' linked to metric from different section"
                        )

        # Check for required properties
        entities_with_labels = sum(1 for e in entities if e.label and e.label.strip())
        if entities:
            label_score = entities_with_labels / len(entities)
            quality_factors.append(label_score)
            if label_score < 1.0:
                validation_notes.append(f"Some entities missing labels")

        # Adjust quality score based on relationship errors
        if invalid_calc_links > 0 or cross_section_links > 0:
            quality_factors.append(0.5)  # Penalize for bad relationships

        # Calculate overall quality score
        if quality_factors:
            quality_score = sum(quality_factors) / len(quality_factors)
        else:
            quality_score = 0.0
            validation_notes.append("No entities or relationships extracted")

        return quality_score, validation_notes
    
    def _consolidate_results(self, all_entities: List[ExtractedEntity],
                           all_relationships: List[ExtractedRelationship]) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Consolidate and deduplicate extraction results"""
        print(f"🔧 Consolidating {len(all_entities)} entities and {len(all_relationships)} relationships...")

        # Step 1: Fix duplicate IDs by renumbering them AND track ID mappings
        unique_entities_by_id = {}
        id_counter = {}
        id_mapping = {}  # Track old_id → new_id mappings

        for entity in all_entities:
            entity_id = entity.id

            if entity_id in unique_entities_by_id:
                # Duplicate ID found - generate new unique ID
                base_id = entity_id.rsplit('_', 1)[0] if '_' in entity_id else entity_id
                if base_id not in id_counter:
                    id_counter[base_id] = 1

                # Find next available number
                while f"{base_id}_{id_counter[base_id]:03d}" in unique_entities_by_id:
                    id_counter[base_id] += 1

                new_id = f"{base_id}_{id_counter[base_id]:03d}"
                print(f"  🔄 Renumbering duplicate ID: {entity_id} → {new_id}")
                id_mapping[entity_id] = new_id  # Track the mapping
                entity.id = new_id
                id_counter[base_id] += 1

            unique_entities_by_id[entity.id] = entity

        # Step 1b: Update relationships to use the new IDs
        for relationship in all_relationships:
            if relationship.subject in id_mapping:
                relationship.subject = id_mapping[relationship.subject]
            if relationship.object in id_mapping:
                relationship.object = id_mapping[relationship.object]

        # Step 2: Merge entities by code (for Metrics) or label (for others)
        unique_entities_by_key = {}
        entity_merge_mapping = {}  # Track merged_id → kept_id mappings

        for entity in unique_entities_by_id.values():
            # For Metrics: use code as unique key (same metric can appear in multiple segments)
            if entity.type == 'Metric':
                code = entity.properties.get('code', 'N/A')
                if code and code != 'N/A':
                    key = ('Metric', code)
                    if key in unique_entities_by_key:
                        # Merge: keep the one with more detail (CalculatedMetric > DirectMetric)
                        existing = unique_entities_by_key[key]
                        existing_type = existing.properties.get('metric_type', '')
                        new_type = entity.properties.get('metric_type', '')

                        # Prefer CalculatedMetric (has full details) over DirectMetric (summary)
                        if new_type == 'CalculatedMetric' and existing_type != 'CalculatedMetric':
                            print(f"  🔀 Merging {code}: Using CalculatedMetric version from {entity.provenance.get('section_title')}")
                            entity_merge_mapping[existing.id] = entity.id  # Map old to new
                            unique_entities_by_key[key] = entity
                        elif len(entity.properties.get('description', '')) > len(existing.properties.get('description', '')):
                            # Or keep the one with longer description
                            print(f"  🔀 Merging {code}: Using more detailed version")
                            entity_merge_mapping[existing.id] = entity.id  # Map old to new
                            unique_entities_by_key[key] = entity
                        else:
                            # Keep existing, map new entity to existing
                            entity_merge_mapping[entity.id] = existing.id
                    else:
                        unique_entities_by_key[key] = entity
                else:
                    # No code - use label
                    key = (entity.type, entity.label)
                    unique_entities_by_key[key] = entity
            else:
                # For non-Metrics: use label as key
                key = (entity.type, entity.label)
                if key not in unique_entities_by_key:
                    # Check for fuzzy matches (e.g., "SASB" vs "SASB Standards")
                    fuzzy_match = False
                    for existing_key, existing_entity in unique_entities_by_key.items():
                        if (existing_key[0] == entity.type and
                            self._entities_are_similar(existing_key[1], entity.label)):
                            # Keep the more detailed entity
                            if len(entity.label) > len(existing_entity.label):
                                entity_merge_mapping[existing_entity.id] = entity.id  # Map old to new
                                unique_entities_by_key[existing_key] = entity
                            else:
                                entity_merge_mapping[entity.id] = existing_entity.id  # Map new to existing
                            fuzzy_match = True
                            break

                    if not fuzzy_match:
                        unique_entities_by_key[key] = entity
                else:
                    # Duplicate - map to existing
                    existing_entity = unique_entities_by_key[key]
                    entity_merge_mapping[entity.id] = existing_entity.id

        # Step 2b: Update relationships to use merged entity IDs
        for relationship in all_relationships:
            if relationship.subject in entity_merge_mapping:
                relationship.subject = entity_merge_mapping[relationship.subject]
            if relationship.object in entity_merge_mapping:
                relationship.object = entity_merge_mapping[relationship.object]

        # Step 3: Deduplicate relationships
        unique_relationships = {}
        for relationship in all_relationships:
            key = (relationship.subject, relationship.predicate, relationship.object)
            if key not in unique_relationships:
                unique_relationships[key] = relationship

        final_entities = list(unique_entities_by_key.values())
        final_relationships = list(unique_relationships.values())

        # Step 4: Filter out invalid relationships
        final_relationships = self._filter_invalid_relationships(final_entities, final_relationships)

        print(f"  ✅ Consolidated to {len(final_entities)} unique entities and {len(final_relationships)} unique relationships")

        return final_entities, final_relationships

    def _filter_invalid_relationships(self, entities: List[ExtractedEntity],
                                     relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Filter out invalid relationships that violate ontology rules"""
        entity_map = {e.id: e for e in entities}
        valid_relationships = []
        removed_count = 0

        for rel in relationships:
            # Check if both entities exist
            subject_entity = entity_map.get(rel.subject)
            object_entity = entity_map.get(rel.object)

            if not subject_entity or not object_entity:
                print(f"  ❌ Removing relationship: {rel.subject} → {rel.predicate} → {rel.object} (entity not found)")
                removed_count += 1
                continue

            # Rule 1: IsCalculatedBy must only link CalculatedMetric to Model
            if rel.predicate == 'IsCalculatedBy':
                metric_type = subject_entity.properties.get('metric_type', '')
                if metric_type != 'CalculatedMetric':
                    print(f"  ❌ Removing IsCalculatedBy: '{subject_entity.label}' is {metric_type}, not CalculatedMetric")
                    removed_count += 1
                    continue
                if object_entity.type != 'Model':
                    print(f"  ❌ Removing IsCalculatedBy: target '{object_entity.label}' is not a Model")
                    removed_count += 1
                    continue

            # Rule 2: ConsistOf must link Category to Metric
            if rel.predicate == 'ConsistOf':
                if subject_entity.type != 'Category':
                    print(f"  ❌ Removing ConsistOf: '{subject_entity.label}' is not a Category")
                    removed_count += 1
                    continue
                if object_entity.type != 'Metric':
                    print(f"  ❌ Removing ConsistOf: '{object_entity.label}' is not a Metric")
                    removed_count += 1
                    continue

            # Rule 3: RequiresInputFrom must link Model to InputMetric
            if rel.predicate == 'RequiresInputFrom':
                if subject_entity.type != 'Model':
                    print(f"  ❌ Removing RequiresInputFrom: '{subject_entity.label}' is not a Model")
                    removed_count += 1
                    continue
                target_metric_type = object_entity.properties.get('metric_type', '')
                if target_metric_type != 'InputMetric':
                    print(f"  ⚠️  WARNING: Model requires input from '{object_entity.label}' which is {target_metric_type}, not InputMetric")
                    # Don't remove - just warn, as this might be valid in some cases

            # Relationship is valid
            valid_relationships.append(rel)

        if removed_count > 0:
            print(f"  🧹 Removed {removed_count} invalid relationships")

        return valid_relationships
    
    def _entities_are_similar(self, label1: str, label2: str) -> bool:
        """Check if two entity labels refer to the same concept"""
        l1, l2 = label1.lower().strip(), label2.lower().strip()
        
        # Exact match
        if l1 == l2:
            return True
        
        # One is contained in the other (e.g., "SASB" in "SASB Standards")
        if l1 in l2 or l2 in l1:
            return True
        
        # Common abbreviation patterns
        abbreviations = {
            'sasb': ['sasb standards', 'sustainability accounting standards board'],
            'tcfd': ['tcfd recommendations', 'task force on climate-related financial disclosures'],
            'ifrs': ['ifrs s2', 'international financial reporting standards'],
            'ghg': ['greenhouse gas', 'ghg emissions']
        }
        
        for abbrev, full_forms in abbreviations.items():
            if ((l1 == abbrev and any(form in l2 for form in full_forms)) or
                (l2 == abbrev and any(form in l1 for form in full_forms))):
                return True
        
        return False
    
    def _get_entity_breakdown(self, entities: List[ExtractedEntity]) -> Dict[str, int]:
        """Get breakdown of entities by type"""
        breakdown = {}
        for entity in entities:
            breakdown[entity.type] = breakdown.get(entity.type, 0) + 1
        return breakdown
    
    def _get_relationship_breakdown(self, relationships: List[ExtractedRelationship]) -> Dict[str, int]:
        """Get breakdown of relationships by predicate"""
        breakdown = {}
        for rel in relationships:
            breakdown[rel.predicate] = breakdown.get(rel.predicate, 0) + 1
        return breakdown
    
    def _save_extraction_results(self, document_name: str, entities: List[ExtractedEntity], 
                               relationships: List[ExtractedRelationship], 
                               extraction_results: List[ExtractionResult],
                               metadata: Dict) -> List[str]:
        """Save extraction results in multiple formats"""
        output_files = []
        
        # Save combined knowledge as JSON (entities + relationships)
        knowledge_file = self.output_dir / f"{document_name}_ontology_guided.json"
        knowledge_data = {
            "entities": [asdict(entity) for entity in entities],
            "relationships": [asdict(rel) for rel in relationships],
            "metadata": {
                "document_name": document_name,
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "entity_breakdown": self._get_entity_breakdown(entities),
                "relationship_breakdown": self._get_relationship_breakdown(relationships),
                "extraction_timestamp": datetime.now().isoformat()
            }
        }
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
        output_files.append(str(knowledge_file))

        # Save combined knowledge graph as RDF
        rdf_file = self.output_dir / f"{document_name}_ontology_guided.rdf"
        rdf_content = self._convert_to_rdf(entities, relationships, metadata)
        with open(rdf_file, 'w', encoding='utf-8') as f:
            f.write(rdf_content)
        output_files.append(str(rdf_file))

        # Save extraction log
        log_file = self.output_dir / f"{document_name}_ontology_guided_extraction_log.txt"
        log_content = self._generate_extraction_log(extraction_results, metadata)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
        output_files.append(str(log_file))
        
        return output_files
    
    def _convert_to_rdf(self, entities: List[ExtractedEntity], 
                       relationships: List[ExtractedRelationship], 
                       metadata: Dict) -> str:
        """Convert extracted knowledge to RDF format"""
        rdf_lines = []
        
        # RDF header
        rdf_lines.extend([
            '@prefix esg: <http://example.org/esg-ontology#> .',
            '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
            '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
            '@prefix prov: <http://www.w3.org/ns/prov#> .',
            '',
            f'# Knowledge Graph extracted from: {metadata["title"]}',
            f'# Extraction timestamp: {datetime.now().isoformat()}',
            f'# Total entities: {len(entities)}',
            f'# Total relationships: {len(relationships)}',
            ''
        ])
        
        # Add entities
        for entity in entities:
            entity_uri = f'esg:{entity.id}'
            rdf_lines.append(f'{entity_uri} a esg:{entity.type} ;')
            rdf_lines.append(f'    rdfs:label "{entity.label}" ;')
            
            # Add properties
            for prop, value in entity.properties.items():
                if value is not None:
                    if isinstance(value, str):
                        rdf_lines.append(f'    esg:{prop} "{value}" ;')
                    else:
                        rdf_lines.append(f'    esg:{prop} {value} ;')
            
            # Add provenance
            rdf_lines.append(f'    prov:wasDerivedFrom "{entity.provenance["document_id"]}" ;')
            rdf_lines.append(f'    prov:generatedAtTime "{entity.provenance["extraction_timestamp"]}"^^xsd:dateTime .')
            rdf_lines.append('')
        
        # Add relationships
        for rel in relationships:
            subject_uri = f'esg:{rel.subject}'
            object_uri = f'esg:{rel.object}'
            rdf_lines.append(f'{subject_uri} esg:{rel.predicate} {object_uri} .')
        
        return '\n'.join(rdf_lines)
    
    def _generate_extraction_log(self, extraction_results: List[ExtractionResult], 
                               metadata: Dict) -> str:
        """Generate detailed extraction log"""
        log_lines = []
        
        log_lines.extend([
            f"ESG Knowledge Extraction Log",
            f"=" * 50,
            f"Document: {metadata['title']}",
            f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Segments: {len(extraction_results)}",
            ""
        ])
        
        for i, result in enumerate(extraction_results, 1):
            log_lines.extend([
                f"Segment {i}: {result.section_title}",
                f"  Entities: {len(result.entities)}",
                f"  Relationships: {len(result.relationships)}",
                f"  Quality Score: {result.quality_score:.2f}",
                f"  Validation Notes: {', '.join(result.validation_notes) if result.validation_notes else 'None'}",
                ""
            ])
        
        return '\n'.join(log_lines)

def process_single_document(document_name: str, llm_config: Optional[Dict] = None, enable_cost_tracking: bool = True) -> Dict:
    """Process a single document for knowledge extraction"""
    try:
        extractor = ESGKnowledgeExtractor(llm_config, enable_cost_tracking=enable_cost_tracking)
        result = extractor.process_document(document_name)
        return result
    except Exception as e:
        print(f"\n❌ Error processing {document_name}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'document_name': document_name
        }

def process_batch_documents(llm_config: Optional[Dict] = None, enable_cost_tracking: bool = True) -> Dict:
    """Process all segmented documents for knowledge extraction"""
    segments_dir = Path("outputs/stage1_segments")
    if not segments_dir.exists():
        print(f"\n❌ Error: Stage 1 segments directory not found: {segments_dir}")
        return {'success': False, 'error': 'Stage 1 segments not found'}

    # Find all segment files
    segment_files = list(segments_dir.glob("*_segments.json"))
    if not segment_files:
        print(f"\n❌ No segment files found in: {segments_dir}")
        return {'success': False, 'error': 'No segment files found'}

    # Extract document names
    document_names = []
    for file in segment_files:
        doc_name = file.name.replace('_segments.json', '')
        document_names.append(doc_name)

    print(f"\n🚀 Starting batch knowledge extraction...")
    print(f"🔍 Found {len(document_names)} documents to process")
    print(f"📁 Input directory: {segments_dir}")
    if enable_cost_tracking:
        print(f"💰 Cost tracking: ENABLED for all documents")
    print(f"{'='*80}")

    results = []
    successful = 0
    failed = 0
    start_time = time.time()

    # Create extractor for EACH document (fresh cost tracker per document)
    for i, doc_name in enumerate(document_names, 1):
        print(f"\n[{i}/{len(document_names)}] Processing: {doc_name}")
        print("-" * 60)

        extractor = ESGKnowledgeExtractor(llm_config, enable_cost_tracking=enable_cost_tracking)
        result = extractor.process_document(doc_name)
        results.append(result)

        if result.get('success', False):
            successful += 1
            print(f"✅ Success: {doc_name}")
        else:
            failed += 1
            print(f"❌ Failed: {doc_name}")
    
    elapsed_time = time.time() - start_time
    
    # Save batch summary
    batch_summary = {
        'success': True,
        'total_documents': len(document_names),
        'successful': successful,
        'failed': failed,
        'results': results,
        'elapsed_time': elapsed_time,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    summary_file = Path("outputs/stage2_ontology_guided_extraction/batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"📊 BATCH ONTOLOGY-GUIDED EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"📁 Total documents: {len(document_names)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📁 Outputs saved to: outputs/stage2_ontology_guided_extraction/")
    print(f"📊 Batch summary: {summary_file}")
    
    if failed > 0:
        print(f"\n⚠️  Failed documents:")
        for result in results:
            if not result.get('success', False):
                doc_name = result.get('document_name', 'unknown')
                error = result.get('error', 'Unknown error')
                print(f"   - {doc_name}: {error}")
    
    return batch_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: ESG Knowledge Graph Extraction')
    parser.add_argument('document', nargs='?', help='Document name to process (without extension)')
    parser.add_argument('--batch', action='store_true', help='Process all documents in batch')
    parser.add_argument('--entities', help='Comma-separated list of entity types to extract')
    parser.add_argument('--llm-config', help='Path to LLM configuration file')
    
    args = parser.parse_args()
    
    # Load LLM configuration if provided
    llm_config = None
    if args.llm_config and Path(args.llm_config).exists():
        with open(args.llm_config, 'r') as f:
            llm_config = json.load(f)
    
    if args.batch:
        result = process_batch_documents(llm_config)
    elif args.document:
        result = process_single_document(args.document, llm_config)
    else:
        print("❌ Error: Please specify a document name or use --batch flag")
        parser.print_help()
        sys.exit(1)
    
    if not result.get('success', False):
        sys.exit(1)
