#!/usr/bin/env python3
"""
Cost Tracking Utility for Ontometric Pipeline

Tracks LLM API usage and calculates costs based on current Anthropic Claude pricing.
Supports all Claude models with accurate pricing.

Pricing as of January 2025:
- Claude 3.5 Sonnet: $3.00/MTok input, $15.00/MTok output
- Claude 3.5 Haiku: $1.00/MTok input, $5.00/MTok output
- Claude 3 Opus: $15.00/MTok input, $75.00/MTok output
- Claude 3 Sonnet: $3.00/MTok input, $15.00/MTok output
- Claude 3 Haiku: $0.25/MTok input, $1.25/MTok output

Note: MTok = Million Tokens (1,000,000 tokens)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class TokenUsage:
    """Records token usage for a single LLM call"""
    timestamp: str
    segment_id: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    page_start: int = None
    page_end: int = None
    page_span: int = None


class CostTracker:
    """
    Tracks LLM API costs across the entire pipeline.

    Features:
    - Per-segment cost tracking
    - Per-document cost aggregation
    - Total pipeline cost calculation
    - Model pricing database
    - Cost reports (JSON and human-readable)
    """

    # Pricing per million tokens (MTok) in USD
    # Updated January 2025 - verify at https://www.anthropic.com/pricing
    PRICING = {
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
        'claude-3-5-sonnet-20240620': {'input': 3.00, 'output': 15.00},
        'claude-3-5-haiku-20241022': {'input': 1.00, 'output': 5.00},
        'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
        'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
        'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
    }

    def __init__(self, output_dir: str = "outputs/cost_tracking"):
        """Initialize cost tracker with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.usage_records: List[TokenUsage] = []

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """
        Calculate cost for a single LLM call.

        Args:
            model: Model identifier (e.g., 'claude-3-5-sonnet-20241022')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with input_cost, output_cost, and total_cost in USD
        """
        # Get pricing for model (default to Claude 3.5 Sonnet if unknown)
        pricing = self.PRICING.get(model, self.PRICING['claude-3-5-sonnet-20241022'])

        # Calculate costs (pricing is per million tokens)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost

        return {
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'total_cost_usd': total_cost
        }

    def record_usage(self, segment_id: str, model: str, input_tokens: int, output_tokens: int,
                     page_start: int = None, page_end: int = None) -> TokenUsage:
        """
        Record token usage for a segment.

        Args:
            segment_id: Segment identifier
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
            page_start: Starting page number (optional)
            page_end: Ending page number (optional)

        Returns:
            TokenUsage record
        """
        costs = self.calculate_cost(model, input_tokens, output_tokens)

        # Calculate page span if both page_start and page_end are provided
        page_span = None
        if page_start is not None and page_end is not None:
            page_span = page_end - page_start + 1

        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            segment_id=segment_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=costs['input_cost_usd'],
            output_cost_usd=costs['output_cost_usd'],
            total_cost_usd=costs['total_cost_usd'],
            page_start=page_start,
            page_end=page_end,
            page_span=page_span
        )

        self.usage_records.append(usage)
        return usage

    def get_document_summary(self) -> Dict:
        """
        Calculate summary statistics for all recorded usage.

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.usage_records:
            return {
                'total_segments': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_cost_usd': 0.0,
                'input_cost_usd': 0.0,
                'output_cost_usd': 0.0,
                'models_used': []
            }

        total_input = sum(r.input_tokens for r in self.usage_records)
        total_output = sum(r.output_tokens for r in self.usage_records)
        total_cost = sum(r.total_cost_usd for r in self.usage_records)
        input_cost = sum(r.input_cost_usd for r in self.usage_records)
        output_cost = sum(r.output_cost_usd for r in self.usage_records)

        models_used = list(set(r.model for r in self.usage_records))

        avg_cost = total_cost / len(self.usage_records) if self.usage_records else 0.0

        return {
            'total_segments': len(self.usage_records),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_input + total_output,
            'total_cost_usd': total_cost,
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'models_used': models_used,
            'average_cost_per_segment_usd': avg_cost
        }

    def save_detailed_report(self, document_name: str) -> str:
        """
        Save detailed cost report (JSON format) for a document.

        Args:
            document_name: Name of the document

        Returns:
            Path to saved report
        """
        report_path = self.output_dir / f"{document_name}_cost_report.json"

        report = {
            'document_name': document_name,
            'report_timestamp': datetime.now().isoformat(),
            'summary': self.get_document_summary(),
            'per_segment_usage': [asdict(r) for r in self.usage_records],
            'pricing_reference': {
                'note': 'Pricing per million tokens (MTok) in USD',
                'models': self.PRICING
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(report_path)

    def save_human_readable_report(self, document_name: str) -> str:
        """
        Save human-readable cost report (text format).

        Args:
            document_name: Name of the document

        Returns:
            Path to saved report
        """
        report_path = self.output_dir / f"{document_name}_cost_report.txt"

        summary = self.get_document_summary()

        lines = []
        lines.append("=" * 80)
        lines.append("LLM COST REPORT")
        lines.append("=" * 80)
        lines.append(f"Document: {document_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Segments Processed: {summary['total_segments']}")
        lines.append(f"Models Used: {', '.join(summary['models_used'])}")
        lines.append("")

        lines.append("Token Usage:")
        lines.append(f"  Input Tokens:  {summary['total_input_tokens']:>12,}")
        lines.append(f"  Output Tokens: {summary['total_output_tokens']:>12,}")
        lines.append(f"  Total Tokens:  {summary['total_tokens']:>12,}")
        lines.append("")

        lines.append("Cost Breakdown:")
        lines.append(f"  Input Cost:  ${summary['input_cost_usd']:>10.4f} USD")
        lines.append(f"  Output Cost: ${summary['output_cost_usd']:>10.4f} USD")
        lines.append(f"  {'─' * 24}")
        lines.append(f"  TOTAL COST:  ${summary['total_cost_usd']:>10.4f} USD")
        lines.append("")

        lines.append(f"Average Cost per Segment: ${summary['average_cost_per_segment_usd']:.4f} USD")
        lines.append("")

        # Per-segment details
        if self.usage_records:
            lines.append("PER-SEGMENT BREAKDOWN")
            lines.append("-" * 80)
            lines.append(f"{'Segment ID':<15} {'Pages':<10} {'Input Tok':>12} {'Output Tok':>12} {'Cost (USD)':>12}")
            lines.append("-" * 80)

            for record in self.usage_records:
                # Format page range
                if record.page_start is not None and record.page_end is not None:
                    if record.page_start == record.page_end:
                        page_str = f"p.{record.page_start}"
                    else:
                        page_str = f"p.{record.page_start}-{record.page_end}"
                else:
                    page_str = "N/A"

                lines.append(f"{record.segment_id:<15} {page_str:<10} {record.input_tokens:>12,} "
                           f"{record.output_tokens:>12,} ${record.total_cost_usd:>11.4f}")

            lines.append("")

        # Pricing reference
        lines.append("PRICING REFERENCE (per million tokens)")
        lines.append("-" * 80)
        for model, prices in self.PRICING.items():
            lines.append(f"{model}")
            lines.append(f"  Input:  ${prices['input']:.2f}/MTok")
            lines.append(f"  Output: ${prices['output']:.2f}/MTok")
            lines.append("")

        lines.append("=" * 80)
        lines.append("Note: Verify latest pricing at https://www.anthropic.com/pricing")
        lines.append("=" * 80)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return str(report_path)

    def print_summary(self):
        """Print cost summary to console"""
        summary = self.get_document_summary()

        print(f"\n{'='*60}")
        print(f"💰 LLM COST SUMMARY")
        print(f"{'='*60}")
        print(f"Segments: {summary['total_segments']}")
        print(f"Tokens: {summary['total_tokens']:,} ({summary['total_input_tokens']:,} in + {summary['total_output_tokens']:,} out)")
        print(f"Cost: ${summary['total_cost_usd']:.4f} USD (${summary['input_cost_usd']:.4f} in + ${summary['output_cost_usd']:.4f} out)")
        print(f"Avg/segment: ${summary['average_cost_per_segment_usd']:.4f} USD")
        print(f"{'='*60}\n")


def load_cost_report(report_path: str) -> Dict:
    """
    Load a previously saved cost report.

    Args:
        report_path: Path to JSON cost report

    Returns:
        Cost report dictionary
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_batch_costs(cost_report_dir: str = "outputs/cost_tracking") -> Dict:
    """
    Aggregate costs across all documents in batch processing.

    Args:
        cost_report_dir: Directory containing cost reports

    Returns:
        Aggregated cost summary
    """
    cost_dir = Path(cost_report_dir)
    if not cost_dir.exists():
        return {'error': 'Cost tracking directory not found'}

    # Load all JSON reports
    report_files = list(cost_dir.glob("*_cost_report.json"))

    if not report_files:
        return {'error': 'No cost reports found'}

    total_segments = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    documents = []

    for report_file in report_files:
        report = load_cost_report(str(report_file))
        summary = report.get('summary', {})

        total_segments += summary.get('total_segments', 0)
        total_input_tokens += summary.get('total_input_tokens', 0)
        total_output_tokens += summary.get('total_output_tokens', 0)
        total_cost += summary.get('total_cost_usd', 0.0)

        documents.append({
            'name': report.get('document_name', 'unknown'),
            'segments': summary.get('total_segments', 0),
            'cost_usd': summary.get('total_cost_usd', 0.0)
        })

    return {
        'total_documents': len(report_files),
        'total_segments': total_segments,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'total_cost_usd': total_cost,
        'average_cost_per_document_usd': total_cost / len(report_files) if report_files else 0.0,
        'average_cost_per_segment_usd': total_cost / total_segments if total_segments else 0.0,
        'documents': sorted(documents, key=lambda x: x['cost_usd'], reverse=True)
    }


if __name__ == "__main__":
    # Example usage
    tracker = CostTracker()

    # Simulate some usage
    tracker.record_usage("seg_001", "claude-3-5-sonnet-20241022", 1000, 500)
    tracker.record_usage("seg_002", "claude-3-5-sonnet-20241022", 1200, 600)

    # Print summary
    tracker.print_summary()

    # Save reports
    tracker.save_detailed_report("example_document")
    tracker.save_human_readable_report("example_document")

    print("✅ Cost tracking example completed")
