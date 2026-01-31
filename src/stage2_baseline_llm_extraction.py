#!/usr/bin/env python3
"""
Stage 2B: Baseline LLM Extraction - Compare Ontology-Guided vs Unconstrained Extraction

Runs 2 extraction experiments to compare effectiveness:
1. Ontology-Guided Extraction - Complete ESGMKG prompt with all guidance
2. Baseline (Unconstrained) - Simple LLM prompt without ontology schema

Usage:
    # Process single document
    python3 src/stage2_baseline_llm_extraction.py "document_name"

    # Process all documents in batch
    python3 src/stage2_baseline_llm_extraction.py --batch

Output Structure:
- outputs/stage2_baseline_llm_extraction/
  └── {document_name}_baseline_llm.json    # Combined results for both experiments

  In batch mode, also creates:
  └── batch_summary.json                    # Overall batch processing summary

Each document file contains:
  - Experiment 1: Ontology-Guided Extraction (entities, relationships, metadata)
  - Experiment 2: Baseline LLM Extraction (entities, relationships, metadata)
  - Comparison summary (entity counts, relationship counts, timestamp)
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# LLM Integration
try:
    import anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  Anthropic not available. Install: pip install anthropic")
    sys.exit(1)


class BaselineLLMExtractor:
    """Runs 2 baseline LLM extraction experiments for comparison"""

    # The 2 experiments
    EXPERIMENTS = {
        "1_ontology_guided": {
            "name": "Ontology-Guided Extraction",
            "description": "Complete ESGMKG prompt with all guidance",
            "prompt_file": "ontology_guided_extraction_prompt.txt"
        },
        "2_baseline_llm": {
            "name": "Baseline (Unconstrained)",
            "description": "Unconstrained LLM (no ontology schema, no SPARQL validation)",
            "prompt_file": "baseline_llm.txt"
        }
    }

    def __init__(self, document_name: str = None, enable_cost_tracking: bool = True):
        """Initialize runner

        Args:
            document_name: Name of document to process (without extension)
            enable_cost_tracking: Whether to track LLM costs (default: True)
        """
        self.document_name = document_name
        self.base_dir = Path(__file__).parent.parent
        self.segments_dir = self.base_dir / "outputs" / "stage1_segments"
        self.output_base_dir = self.base_dir / "outputs" / "stage2_baseline_llm_extraction"
        self.prompts_dir = self.base_dir / "Prompts"

        # Create base output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Load LLM config
        config_path = self.base_dir / "config_llm.json"
        with open(config_path, 'r') as f:
            self.llm_config = json.load(f)

        # Get API key from config or environment
        api_key = self.llm_config.get('api_settings', {}).get('api_key') or os.environ.get('ANTHROPIC_API_KEY')

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)

        # Initialize cost tracking
        self.enable_cost_tracking = enable_cost_tracking
        self.cost_tracker = None
        if self.enable_cost_tracking:
            sys.path.insert(0, str(self.base_dir / "src"))
            from utils.cost_tracker import CostTracker
            self.cost_tracker = CostTracker(output_dir=str(self.base_dir / "outputs" / "cost_tracking"))
            print(f"💰 Cost tracking: ENABLED")

        print(f"✓ Baseline LLM Extractor initialized")
        print(f"  Experiments: {len(self.EXPERIMENTS)}")
        print(f"  Output base: {self.output_base_dir}")


    def process_document(self, document_name: str) -> dict:
        """Process a single document with all experiments

        Args:
            document_name: Name of document to process (without extension)

        Returns:
            Dictionary containing processing results
        """
        print(f"\n{'='*80}")
        print(f"BASELINE LLM COMPARISON: {document_name}")
        print(f"{'='*80}\n")

        # Load segments
        segments = self._load_segments(document_name)
        print(f"✓ Loaded {len(segments)} segments\n")

        # Run each experiment and collect results
        experiment_results = {}
        for exp_id, exp_config in self.EXPERIMENTS.items():
            print(f"\n[{exp_id}] {exp_config['name']}")
            print(f"  {exp_config['description']}")
            print("-" * 80)

            start_time = time.time()

            try:
                result = self._run_experiment(exp_id, exp_config, segments, document_name)
                experiment_results[exp_id] = result

                elapsed = time.time() - start_time
                print(f"✓ Completed in {elapsed:.1f}s")
                print(f"  Entities: {len(result.get('entities', []))}")
                print(f"  Relationships: {len(result.get('relationships', []))}")

            except Exception as e:
                print(f"✗ Error: {str(e)}")
                experiment_results[exp_id] = None

        # Save cost tracking reports if enabled (only for experiment 2 - baseline)
        if self.cost_tracker:
            cost_report_json = self.cost_tracker.save_detailed_report(f"{document_name}_baseline")
            cost_report_txt = self.cost_tracker.save_human_readable_report(f"{document_name}_baseline")
            print(f"\n💰 Baseline extraction cost tracking reports saved:")
            print(f"  📄 {cost_report_json}")
            print(f"  📄 {cost_report_txt}")
            self.cost_tracker.print_summary()

        # Save combined results to single file
        output_file = self._save_combined_results(experiment_results, document_name)

        print(f"\n{'='*80}")
        print(f"✓ All experiments completed for {document_name}!")
        print(f"\nOutput file created:")
        print(f"  • {output_file.name}")
        print(f"{'='*80}\n")

        return {
            'success': True,
            'document_name': document_name,
            'total_experiments': len(self.EXPERIMENTS),
            'successful_experiments': sum(1 for r in experiment_results.values() if r is not None),
            'failed_experiments': sum(1 for r in experiment_results.values() if r is None),
            'output_file': str(output_file),
            'results': experiment_results
        }


    def _load_segments(self, document_name: str):
        """Load segmented document

        Args:
            document_name: Name of document to load

        Returns:
            List of document segments
        """
        segments_file = self.segments_dir / f"{document_name}_segments.json"

        if not segments_file.exists():
            raise FileNotFoundError(f"Segments not found: {segments_file}")

        with open(segments_file, 'r') as f:
            return json.load(f)


    def _use_validated_output(self, exp_id, exp_config, document_name: str):
        """Copy pre-validated Ontology-Guided output (Experiment 1 only)

        Args:
            exp_id: Experiment ID
            exp_config: Experiment configuration
            document_name: Name of document being processed

        Returns:
            Dictionary containing entities and relationships
        """
        print(f"  Using pre-validated Ontology-Guided output...", end='')

        # Load the validated stage2_ontology_guided_extraction output
        knowledge_file = self.base_dir / "outputs" / "stage2_ontology_guided_extraction" / f"{document_name}_ontology_guided.json"

        if not knowledge_file.exists():
            print(f" [file not found]", end='')
            return {"entities": [], "relationships": []}

        with open(knowledge_file, 'r') as f:
            kg_data = json.load(f)

        # Extract entities and relationships
        entities = kg_data.get('entities', [])
        relationships = kg_data.get('relationships', [])

        print(f" {len(entities)} entities, {len(relationships)} relationships")

        # Return result (will be saved to combined file later)
        return {
            "name": exp_config['name'],
            "description": exp_config['description'],
            "note": "Copied from stage2_ontology_guided_extraction output (validated ontology-guided system)",
            "entities": entities,
            "relationships": relationships,
            "entity_count": len(entities),
            "relationship_count": len(relationships)
        }


    def _run_experiment(self, exp_id, exp_config, segments, document_name: str):
        """Run single experiment

        Args:
            exp_id: Experiment ID
            exp_config: Experiment configuration
            segments: Document segments
            document_name: Name of document being processed

        Returns:
            Dictionary containing experiment results
        """

        # Special case: Experiment 1 (Ontology-Guided) - use pre-validated output
        if exp_id == "1_ontology_guided":
            return self._use_validated_output(exp_id, exp_config, document_name)

        # Load prompt
        prompt_template = self._load_prompt(exp_config)

        # Extract from all segments
        all_entities = []
        all_relationships = []

        for i, segment in enumerate(segments, 1):
            segment_id = segment.get('segment_id', f'seg_{i:03d}')
            section_title = segment.get('section_title', 'Unknown Section')
            content = segment.get('content', '')
            page_start = segment.get('page_start')
            page_end = segment.get('page_end')

            # Skip short segments
            if len(content.strip()) < 100:
                continue

            print(f"  Processing {segment_id}...", end='')

            # Prepare prompt
            prompt = prompt_template.replace("{document_name}", document_name)
            prompt = prompt.replace("{section_title}", section_title)
            prompt = prompt.replace("{content}", content[:6000])  # Limit length

            # Call LLM with segment_id and page info for cost tracking
            response = self._call_llm(prompt, f"baseline_{segment_id}", page_start, page_end)

            # Parse response
            segment_data = self._parse_response(response)

            # Collect entities and relationships
            entities = segment_data.get('entities', [])
            relationships = segment_data.get('relationships', [])

            all_entities.extend(entities)
            all_relationships.extend(relationships)

            print(f" {len(entities)} entities, {len(relationships)} relationships")

            # Small delay to avoid rate limits
            time.sleep(0.5)

        # Return result (will be saved to combined file later)
        return {
            "name": exp_config['name'],
            "description": exp_config['description'],
            "entities": all_entities,
            "relationships": all_relationships,
            "entity_count": len(all_entities),
            "relationship_count": len(all_relationships)
        }


    def _load_prompt(self, exp_config):
        """Load and prepare prompt for experiment"""
        prompt_file = self.prompts_dir / exp_config['prompt_file']

        with open(prompt_file, 'r') as f:
            prompt = f.read()

        return prompt


    def _call_llm(self, prompt, segment_id: str = None, page_start: int = None, page_end: int = None):
        """Call Anthropic API"""
        response = self.client.messages.create(
            model=self.llm_config['api_settings']['model'],
            max_tokens=self.llm_config['api_settings']['max_tokens'],
            temperature=self.llm_config['api_settings']['temperature'],
            messages=[{"role": "user", "content": prompt}]
        )

        # Track token usage if cost tracking is enabled
        if self.cost_tracker and segment_id:
            model_used = self.llm_config['api_settings']['model']
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self.cost_tracker.record_usage(segment_id, model_used, input_tokens, output_tokens,
                                           page_start, page_end)

        return response.content[0].text


    def _parse_response(self, response):
        """Parse LLM JSON response"""
        import re

        # Try direct JSON parse
        try:
            return json.loads(response)
        except:
            pass

        # Try extracting JSON from markdown
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Try finding JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        # Fallback: empty structure
        print(" [parse failed]", end='')
        return {"entities": [], "relationships": []}


    def _save_combined_results(self, results, document_name: str) -> Path:
        """Save all experiment results to a single combined file

        Args:
            results: Dictionary of experiment results
            document_name: Name of document

        Returns:
            Path to the combined output file
        """
        combined_data = {
            "document": document_name,
            "timestamp": datetime.now().isoformat(),
            "experiments": {}
        }

        # Add each experiment's full data
        for exp_id, result in results.items():
            if result:
                combined_data["experiments"][exp_id] = result
            else:
                combined_data["experiments"][exp_id] = {
                    "name": self.EXPERIMENTS[exp_id]['name'],
                    "status": "failed"
                }

        # Save to single file
        output_file = self.output_base_dir / f"{document_name}_baseline_llm.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        return output_file


def process_single_document(document_name: str) -> dict:
    """Process a single document for baseline LLM comparison

    Args:
        document_name: Name of document to process (without extension)

    Returns:
        Dictionary containing processing results
    """
    try:
        extractor = BaselineLLMExtractor()
        result = extractor.process_document(document_name)
        return result
    except Exception as e:
        print(f"\n❌ Error processing {document_name}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'document_name': document_name
        }


def process_batch_documents() -> dict:
    """Process all segmented documents for baseline LLM comparison

    Returns:
        Dictionary containing batch processing results
    """
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

    print(f"\n🚀 Starting batch baseline LLM comparison...")
    print(f"🔍 Found {len(document_names)} documents to process")
    print(f"📁 Input directory: {segments_dir}")
    print(f"{'='*80}")

    results = []
    successful = 0
    failed = 0
    start_time = time.time()

    extractor = BaselineLLMExtractor()

    for i, doc_name in enumerate(document_names, 1):
        print(f"\n[{i}/{len(document_names)}] Processing: {doc_name}")
        print("-" * 60)

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

    summary_file = Path("outputs/stage2_baseline_llm_extraction/batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"📊 BATCH BASELINE LLM COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"📁 Total documents: {len(document_names)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📁 Outputs saved to: outputs/stage2_baseline_llm_extraction/")
    print(f"📊 Batch summary: {summary_file}")

    if failed > 0:
        print(f"\n⚠️  Failed documents:")
        for result in results:
            if not result.get('success', False):
                doc_name = result.get('document_name', 'unknown')
                error = result.get('error', 'Unknown error')
                print(f"   - {doc_name}: {error}")

    return batch_summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Stage 2B: Baseline LLM Extraction Comparison')
    parser.add_argument('document', nargs='?', help='Document name to process (without extension)')
    parser.add_argument('--batch', action='store_true', help='Process all documents in batch')

    args = parser.parse_args()

    if args.batch:
        result = process_batch_documents()
    elif args.document:
        result = process_single_document(args.document)
    else:
        print("❌ Error: Please specify a document name or use --batch flag")
        parser.print_help()
        sys.exit(1)

    if not result.get('success', False):
        sys.exit(1)


if __name__ == "__main__":
    main()
