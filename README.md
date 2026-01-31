# ESGMKG: Ontology-Guided ESG Metric Knowledge Graph Extraction Pipeline

> Automated extraction of ESG metrics and relationships from regulatory documents using ontology-guided LLM extraction with validation

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Output Files Reference](#output-files-reference)
- [Key Metrics Explained](#key-metrics-explained)
- [Validation Approach](#validation-approach)
- [Baseline Comparison](#baseline-comparison)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements an ontology-guided knowledge graph extraction pipeline for ESG (Environmental, Social, Governance) metrics from regulatory documents. It uses Claude API with carefully designed prompts based on the ESGMKG (ESG Metric Knowledge Graph) ontology to extract structured knowledge with high precision and semantic consistency.

### Key Features

- **Ontology-Guided Extraction**: Uses ESGMKG schema to guide LLM extraction
- **Multi-Stage Pipeline**: Segmentation, Extraction, and Validation
- **Validation System**: Semantic correctness checking with 10 validation rules
- **Provenance Tracking**: Full traceability from entities back to source pages
- **Baseline Comparison**: Demonstrates 40-50% improvement over unconstrained LLM
- **Batch Processing**: Process multiple documents automatically

### Supported Document Types

- SASB Industry Standards
- TCFD Climate-Related Disclosures
- IFRS Sustainability Standards (S2)
- Australian Sustainability Reporting Standards
- Other ESG regulatory frameworks

---

## Project Structure

```
Ontometric-Pipeline/
├── README.md                          # This file
├── config_llm.json                    # LLM API configuration
│
├── data/
│   └── input_pdfs/                    # Input PDF documents
│       ├── 1.SASB-semiconductors-standard_en-gb.pdf
│       ├── 1.SASB-commercial-banks-standard_en-gb.pdf
│       ├── 2.FINAL-2017-TCFD-Report.pdf
│       └── ...
│
├── Prompts/
│   ├── ontology_guided_extraction_prompt.txt    # Ontology-guided extraction prompt
│   └── baseline_llm.txt                         # Baseline unconstrained prompt
│
├── src/
│   ├── stage1_segmentation.py                   # PDF segmentation
│   ├── stage2_ontology_guided_extraction.py     # Ontology-guided KG extraction
│   ├── stage2_baseline_llm_extraction.py        # Baseline comparison experiments
│   ├── stage3_ontology_guided_validation.py     # Main validation system
│   └── stage3_baseline_llm_comparison.py        # Baseline vs ontology comparison
│
└── outputs/
    ├── stage1_segments/                         # Segmented PDF content (JSON)
    ├── stage2_ontology_guided_extraction/       # Extracted knowledge graphs
    ├── stage2_baseline_llm_extraction/          # Baseline comparison data
    ├── stage3_ontology_guided_validation/       # Validation reports
    └── stage3_baseline_llm_comparison/          # Comparison analysis
```

---

## Pipeline Stages

### Stage 1: Document Segmentation

**Script**: `src/stage1_segmentation.py`
**Input**: PDF documents from `data/input_pdfs/`
**Output**: Segmented JSON files in `outputs/stage1_segments/`

**What it does**:
- Extracts text from PDF pages
- Segments documents into logical sections (e.g., metric definitions, tables)
- Preserves page numbers and section titles for provenance
- Outputs structured JSON for downstream processing

**Output Files** (per document):
- `{document_name}_segments.json` - All segments with metadata

---

### Stage 2A: Ontology-Guided Extraction

**Script**: `src/stage2_ontology_guided_extraction.py`
**Input**: Segments from Stage 1
**Output**: Knowledge graphs in `outputs/stage2_ontology_guided_extraction/`

**What it does**:
- Uses ontology-guided prompts to extract entities and relationships
- Follows ESGMKG schema (Industry, Category, Metric, Model, etc.)
- Adds provenance metadata to all entities
- Converts to RDF format for semantic validation
- Merges duplicate entities across segments

**Output Files** (per document):
- `{document_name}_ontology_guided.json` - Extracted entities and relationships (JSON)
- `{document_name}_ontology_guided.rdf` - Knowledge graph in RDF format
- `{document_name}_ontology_guided_extraction_log.txt` - Processing log
- `batch_summary.json` - Summary when processing multiple documents (batch mode only)

**Entity Types Extracted**:
- `Industry` - Business sectors (e.g., "Semiconductors")
- `ReportingFramework` - Standards (e.g., "SASB", "TCFD")
- `Category` - ESG categories (e.g., "GHG Emissions")
- `Metric` - Specific metrics (e.g., "Scope 1 GHG Emissions")
- `Model` - Calculation models for metrics
- `DirectMetric`, `CalculatedMetric`, `InputMetric` - Metric subtypes

**Relationship Types**:
- `ReportUsing` - Industry reports using framework
- `Include` - Framework includes category
- `ConsistOf` - Category consists of metrics
- `IsCalculatedBy` - Metric calculated by model
- `RequiresInputFrom` - Model requires input metrics

---

### Stage 2B: Baseline LLM Extraction (Comparison)

**Script**: `src/stage2_baseline_llm_extraction.py`
**Input**: Segments from Stage 1 + Ontology-guided results from Stage 2A
**Output**: Comparison experiments in `outputs/stage2_baseline_llm_extraction/`

**What it does**:
- Runs TWO experiments on the same document:
  1. **Ontology-Guided**: Reuses validated output from Stage 2A (does not re-run)
  2. **Baseline (Unconstrained)**: Runs unconstrained LLM without ontology guidance
- Supports both single document and batch processing
- Stores combined results for comparison in Stage 3B

**Output Files** (per document):
- `{document_name}_baseline_llm.json` - Combined file with both experiments:
  - Experiment 1: Ontology-guided results (copied from Stage 2A)
  - Experiment 2: Baseline unconstrained results
  - Metadata and entity/relationship counts

---

### Stage 3A: Ontology-Guided Validation

**Script**: `src/stage3_ontology_guided_validation.py`
**Input**: Knowledge graphs (JSON files) from Stage 2A
**Output**: Validation report in `outputs/stage3_ontology_guided_validation/`

**What it does**:
- Validates JSON knowledge graph data quality (Python validation)
- Runs 10 validation rules (VR001-VR010)
- Calculates 4 quality metrics per document
- Supports single document and batch validation modes
- Generates human-readable .txt validation reports

**Primary Validation Approach**: Python validation (`--python`) validates JSON files and is the recommended approach for standard workflows. Advanced users can optionally use SPARQL/Hybrid validation if they need RDF semantic queries.

**Output Files**:
- **Single Document Mode**: `{document_name}_ontology_guided_validation.txt` - Validation report for one document
- **Batch Mode**: `batch_ontology_guided_validation.txt` - Comprehensive report for all documents

**Validation Rules**:
- **VR001**: CalculatedMetrics must have IsCalculatedBy relationships
- **VR002**: Quantitative metrics must specify units
- **VR003**: Models must specify input variables
- **VR004**: Entity IDs must be unique
- **VR005**: Entities must have required properties
- **VR006**: Categories should contain metrics
- **VR007**: Industries should report using frameworks
- **VR008**: Qualitative metrics should have method properties
- **VR009**: Models with equations must have input variables
- **VR010**: Relationships must use valid predicates

**Metrics Calculated**:
1. **CQ Violation Rate (%)** - Data quality issues
2. **Triple Retention Rate (%)** - Percentage of validation rules passed
3. **Provenance Completeness (%)** - Percentage of entities with source tracking
4. **Ontology Compliance (%)** - Percentage of items matching ESGMKG schema

---

### Stage 3B: Baseline LLM Comparison

**Script**: `src/stage3_baseline_llm_comparison.py`
**Input**: Experiments from Stage 2B
**Output**: Comparison report in `outputs/stage3_baseline_llm_comparison/`

**What it does**:
- Compares ontology-guided vs baseline (unconstrained) extraction
- Validates both using the same 10 validation rules
- Calculates improvement deltas (e.g., +55.6% triple retention)
- Supports single document and batch comparison modes
- Generates comparative analysis report

**Output Files**:
- **Single Document Mode**: `{document_name}_baseline_llm_comparison.txt` - Comparison for one document
- **Batch Mode**: `batch_baseline_llm_comparison.txt` - Comparison across all documents

**Comparison Metrics**:
- Entity/Relationship counts
- CQ Violation Rate improvement
- Triple Retention Rate improvement (coverage-based)
- Provenance Completeness difference
- Ontology Compliance improvement

---

## Installation

### Prerequisites

- Python 3.8+
- Anthropic API key (Claude)
- PDF documents to process

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/anonymous-submission/Ontometric-Pipeline.git
cd Ontometric-Pipeline
```

2. **Install dependencies**:
```bash
pip install anthropic PyPDF2 rdflib
```

3. **Configure API key**:

Edit `config_llm.json`:
```json
{
  "api_settings": {
    "api_key": "your-anthropic-api-key-here",
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 16000,
    "temperature": 0
  }
}
```

Or set environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

4. **Add PDF documents**:
```bash
# Place your PDFs in:
data/input_pdfs/
```

---

## Quick Start

### Process a Single Document

```bash
# Step 1: Segment the PDF
python3 src/stage1_segmentation.py "1.SASB-semiconductors-standard_en-gb"

# Step 2: Extract knowledge graph
python3 src/stage2_ontology_guided_extraction.py "1.SASB-semiconductors-standard_en-gb"

# Step 3: Validate extraction quality
python3 src/stage3_ontology_guided_validation.py --python --single "1.SASB-semiconductors-standard_en-gb"
```

### Batch Process All Documents

```bash
# Process all PDFs in data/input_pdfs/
python3 src/stage1_segmentation.py --batch
python3 src/stage2_ontology_guided_extraction.py --batch
python3 src/stage3_ontology_guided_validation.py --python --batch
```

### Run Baseline Comparison (Single Document)

```bash
# Step 1: Run baseline extraction (reuses ontology-guided results from Stage 2A)
python3 src/stage2_baseline_llm_extraction.py "1.SASB-semiconductors-standard_en-gb"

# Step 2: Compare ontology-guided vs unconstrained baseline
python3 src/stage3_baseline_llm_comparison.py "1.SASB-semiconductors-standard_en-gb"
```

### Run Baseline Comparison (Batch)

```bash
# Process all documents with baseline comparison
python3 src/stage2_baseline_llm_extraction.py --batch
python3 src/stage3_baseline_llm_comparison.py --batch
```

---

## Detailed Usage

### Stage 1: Segmentation Options

```bash
# Process single document
python3 src/stage1_segmentation.py "document_name"

# Process all documents
python3 src/stage1_segmentation.py --batch

# Custom output directory
python3 src/stage1_segmentation.py --output-dir custom_segments/ "document_name"
```

### Stage 2A: Ontology-Guided Extraction Options

```bash
# Process single document
python3 src/stage2_ontology_guided_extraction.py "document_name"

# Process all documents
python3 src/stage2_ontology_guided_extraction.py --batch

# Extract only specific entity types
python3 src/stage2_ontology_guided_extraction.py --entities "Metric,Model" "document_name"
```

### Stage 2B: Baseline LLM Extraction Options

```bash
# Process single document (requires Stage 2A output)
python3 src/stage2_baseline_llm_extraction.py "document_name"

# Process all documents in batch
python3 src/stage2_baseline_llm_extraction.py --batch

# This will:
# 1. Reuse validated ontology-guided output from Stage 2A (does not re-run)
# 2. Run unconstrained baseline extraction on the same document
# 3. Save combined results to {document_name}_baseline_llm.json
```

### Stage 3A: Validation Options

**Recommended (Standard Workflow):**
```bash
# Single document validation
python3 src/stage3_ontology_guided_validation.py --python --single "document_name"

# Batch validation - all documents
python3 src/stage3_ontology_guided_validation.py --python --batch

# Custom directories
python3 src/stage3_ontology_guided_validation.py --python --batch \
  --kg-dir custom_knowledge_graphs/ \
  --output-dir custom_validation_reports/
```

**Advanced Options (Only if you need RDF semantic validation):**
```bash
# SPARQL validation (requires RDF files)
python3 src/stage3_ontology_guided_validation.py --sparql --batch

# Hybrid validation (Python + SPARQL combined)
python3 src/stage3_ontology_guided_validation.py --hybrid --batch

# Quick validation (fast SPARQL for CI/CD)
python3 src/stage3_ontology_guided_validation.py --quick --batch

# Show all options
python3 src/stage3_ontology_guided_validation.py --help
```

**Note**: Python validation (`--python`) is the primary approach. It validates all 10 rules and calculates all 4 metrics from JSON files. SPARQL/Hybrid are advanced options only needed for RDF semantic queries.

### Stage 3B: Baseline LLM Comparison Options

```bash
# Compare single document
python3 src/stage3_baseline_llm_comparison.py "document_name"

# Compare all documents in batch
python3 src/stage3_baseline_llm_comparison.py --batch

# This will:
# 1. Load combined experiments from stage2_baseline_llm_extraction/
# 2. Run the same validation on both
# 3. Calculate improvement deltas
# 4. Generate comparison report
```

---

## Output Files Reference

### Complete Output File Inventory

#### Stage 1 Outputs (`outputs/stage1_segments/`)
- `{document_name}_segments.json` - Segmented document content

#### Stage 2A Outputs (`outputs/stage2_ontology_guided_extraction/`)
- `{document_name}_ontology_guided.json` - Extracted knowledge graph (JSON)
- `{document_name}_ontology_guided.rdf` - Knowledge graph (RDF/Turtle)
- `{document_name}_ontology_guided_extraction_log.txt` - Processing log
- `batch_summary.json` - Batch processing summary (batch mode only)

#### Stage 2B Outputs (`outputs/stage2_baseline_llm_extraction/`)
**Per Document:**
- `{document_name}_baseline_llm.json` - Combined file with both experiments:
  - Experiment 1: Ontology-guided results (copied from Stage 2A)
  - Experiment 2: Baseline unconstrained results

**Batch Mode:**
- `batch_summary.json` - Overall batch summary

#### Stage 3A Outputs (`outputs/stage3_ontology_guided_validation/`)
**Single Document Mode:**
- `{document_name}_ontology_guided_validation.txt` - Validation report for one document

**Batch Mode:**
- `batch_ontology_guided_validation.txt` - Comprehensive validation report for all documents

#### Stage 3B Outputs (`outputs/stage3_baseline_llm_comparison/`)
**Single Document Mode:**
- `{document_name}_baseline_llm_comparison.txt` - Comparison analysis for one document

**Batch Mode:**
- `batch_baseline_llm_comparison.txt` - Comparison across all documents

---

## Key Metrics Explained

### 1. CQ Violation Rate (%)

**Formula**: `(Total Violations / Total Checks) x 100`

**What it measures**: Percentage of validation checks that failed

**Lower is better**: 0% = perfect, 100% = all checks failed

**Example**: If 1 out of 190 checks fail, the violation rate is 0.5%

---

### 2. Triple Retention Rate (%)

**Formula**: `(Passed Applicable Rules / Total Applicable Rules) x 100`

**What it measures**: Percentage of validation rules that passed completely

**Higher is better**: 100% = all rules passed, 0% = all rules failed

**Coverage-based calculation**: Only counts rules where relevant entities exist to validate (prevents "passing by omission")

**Example**:
- Ontology-Guided: 8/9 applicable rules pass = 88.9%
- Baseline: 1/3 applicable rules pass = 33.3%

---

### 3. Provenance Completeness (%)

**Formula**: `(Entities with Source Info / Total Entities) x 100`

**What it measures**: Percentage of entities that have traceability to source documents

**Higher is better**: 100% = full traceability

**Provenance includes**:
- `document_id` - Source document
- `segment_id` - Source segment
- `section_title` - Section name
- `extraction_timestamp` - When extracted

---

### 4. Ontology Compliance (%)

**Formula**: `(Valid ESGMKG Items / Total Items) x 100`

**What it measures**: Percentage of entities and relationships that match ESGMKG schema

**Higher is better**: 100% = perfect schema adherence

**Uses semantic matching**: Accepts variations like "energy_metric" matching "Metric"

---

## Validation Approach

### Primary Validation Mode (Recommended)

**Python Validation** (`--python`) - Recommended for most users
- Validates JSON knowledge graph files
- Runs all 10 validation rules (VR001-VR010)
- Calculates all 4 quality metrics
- Ultra-fast execution (~0.001s per rule)
- This is the standard validation approach
- Works directly with Stage 2A JSON output

**Usage**:
```bash
python3 src/stage3_ontology_guided_validation.py --python --single "document_name"
python3 src/stage3_ontology_guided_validation.py --python --batch
```

### Advanced Validation Modes (Optional)

**SPARQL Validation** (`--sparql`)
- Validates RDF knowledge graphs using SPARQL queries
- Requires RDF files (*.rdf) from Stage 2A
- Only needed for RDF-specific semantic validation
- Most users do not need this

**Hybrid Validation** (`--hybrid`)
- Combines Python + SPARQL validation
- Requires both JSON and RDF files
- Averages results from both approaches
- Only useful if you need both data quality and RDF semantic validation

**Note**: For standard workflows, use `--python`. It validates all rules and calculates all metrics. SPARQL/Hybrid are advanced options only needed for RDF semantic queries.

### Validation Rules Summary

| Rule | Description | Critical? |
|------|-------------|-----------|
| VR001 | CalculatedMetrics must link to Models | Critical |
| VR002 | Quantitative metrics must have Units | Critical |
| VR003 | Models must have Input Variables | Critical |
| VR004 | Unique Entity IDs | Critical |
| VR005 | Required Properties | Critical |
| VR006 | Categories must link to Metrics | Quality |
| VR007 | Industries must link to Frameworks | Quality |
| VR008 | Qualitative metrics should have Methods | Quality |
| VR009 | Equations must have Variables | Critical |
| VR010 | Valid Predicates | Critical |

---

## Baseline Comparison

### Purpose

Demonstrates the value of ontology-guided extraction by comparing it to a simple unconstrained LLM approach.

### Typical Results

| Metric | Ontology-Guided | Baseline | Improvement |
|--------|----------------|----------|-------------|
| CQ Violation Rate | 0.5% | 39.0% | -38.5% |
| Triple Retention | 88.9% | 33.3% | +55.6% |
| Provenance | 100.0% | 0.0% | +100.0% |
| Ontology Compliance | 100.0% | 55.9% | +44.1% |

### Key Findings

- Ontology-guided extraction reduces CQ violations by 38.5%
- Ontology-guided extraction improves triple retention by 55.6%
- Ontology-guided extraction provides 100% provenance tracking
- Ontology-guided extraction achieves 100% ontology compliance

---

## Common Workflows

### Workflow 1: Process New Document

```bash
# 1. Add PDF to data/input_pdfs/
cp "new_document.pdf" "data/input_pdfs/"

# 2. Run full pipeline
python3 src/stage1_segmentation.py "new_document"
python3 src/stage2_ontology_guided_extraction.py "new_document"
python3 src/stage3_ontology_guided_validation.py --hybrid --single "new_document"

# 3. Check results
cat "outputs/stage3_ontology_guided_validation/COMPREHENSIVE_REPORT.txt"
```

### Workflow 2: Batch Process All Documents

```bash
# Run full pipeline on all documents
python3 src/stage1_segmentation.py --batch
python3 src/stage2_ontology_guided_extraction.py --batch
python3 src/stage3_ontology_guided_validation.py --hybrid --batch

# Review comprehensive report
cat "outputs/stage3_ontology_guided_validation/COMPREHENSIVE_REPORT.txt"
```

### Workflow 3: Generate Baseline Comparison (for research/publication)

```bash
# 1. Ensure ontology-guided extraction is done
python3 src/stage2_ontology_guided_extraction.py "1.SASB-semiconductors-standard_en-gb"

# 2. Run baseline comparison
python3 src/stage2_baseline_llm_extraction.py

# 3. Generate comparison report
python3 src/stage3_baseline_llm_comparison.py

# 4. View comparison
cat "outputs/stage3_baseline_llm_comparison/baseline_llm_comparison_report.txt"
```

### Workflow 4: CI/CD Pipeline Integration

```bash
#!/bin/bash
# ci_validation.sh

# Quick validation for CI/CD
python3 src/stage3_ontology_guided_validation.py --quick --batch

if [ $? -eq 0 ]; then
  echo "Knowledge graphs validated successfully"
  exit 0
else
  echo "Validation failed - check reports for details"
  exit 1
fi
```

---

## Troubleshooting

### Issue: "API key not found"

**Solution**: Set your Anthropic API key in `config_llm.json` or as environment variable:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Issue: "Stage2 knowledge file not found"

**Solution**: Run Stage 2A extraction first:
```bash
python3 src/stage2_ontology_guided_extraction.py "document_name"
```

### Issue: "Cannot import stage3_ontology_guided_validation"

**Solution**: Ensure you are running from the project root directory:
```bash
cd Ontometric-Pipeline
python3 src/stage3_baseline_llm_comparison.py
```

### Issue: High CQ Violation Rate

**Possible causes**:
- Document has complex nested structures
- Extraction prompt may need tuning for document type
- Document quality issues (scanned PDFs, poor formatting)

**Solution**: Check the detailed breakdown in validation report to identify which rules are failing.

### Issue: Low Ontology Compliance

**Possible causes**:
- Prompt not following ESGMKG schema
- Document uses non-standard terminology

**Solution**: Review entity type distribution in validation report and adjust extraction prompt if needed.

---

## Project Metadata

- **Version**: 1.0
- **Last Updated**: October 2025
- **Python**: 3.8+
- **LLM**: Claude (Anthropic)
- **Ontology**: ESGMKG (ESG Metric Knowledge Graph)

---

## License

This project is for research and educational purposes.

---

## Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review validation reports in `outputs/stage3_ontology_guided_validation/`
3. Check extraction logs in `outputs/stage2_ontology_guided_extraction/`
