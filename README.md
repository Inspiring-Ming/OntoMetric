# OntoMetric: An Ontology-Driven LLM-Assisted Framework for Automated ESG Metric Knowledge Graph Generation

> An ontology-guided framework for constructing ESG metric knowledge graphs from regulatory documents

## Overview

OntoMetric is a three-stage pipeline for extracting structured ESG (Environmental, Social, Governance) metric knowledge graphs from regulatory documents. It embeds the ESGMKG ontology into extraction and validation, achieving 65–90% semantic accuracy and over 80% schema compliance across five major ESG regulatory documents.

## Key Features

- **Ontology-Guided Extraction**: ESGMKG schema constrains LLM output to valid entity types and relationships
- **Multi-Stage Pipeline**: Document Segmentation, Knowledge Extraction, and Validation
- **Provenance Tracking**: Full traceability from extracted entities to source document pages
- **Baseline Comparison**: Quantitative evaluation against unconstrained LLM extraction

## Repository Structure

```
OntoMetric/
├── src/
│   ├── stage1_segmentation.py          # Document segmentation
│   ├── stage2_ontology_guided_extraction.py   # Ontology-guided extraction
│   ├── stage2_baseline_llm_extraction.py      # Baseline comparison
│   └── stage3_validation.py             # Validation system
├── Prompts/
│   ├── ontology_guided_extraction_prompt.txt  # ESGMKG-guided prompt
│   └── baseline_llm.txt                 # Unconstrained baseline prompt
├── data/input_pdfs/                     # Input regulatory documents
├── outputs/                             # Extraction results and validation reports
└── result_visualisation_and_analysis/   # Figures and analysis scripts
```

## Quick Start

```bash
# Install dependencies
pip install anthropic PyPDF2 rdflib

# Configure API key
export ANTHROPIC_API_KEY="your-api-key"

# Run pipeline on a document
python3 src/stage1_segmentation.py "document_name"
python3 src/stage2_ontology_guided_extraction.py "document_name"
python3 src/stage3_validation.py --python --single "document_name"

# Run baseline comparison
python3 src/stage2_baseline_llm_extraction.py "document_name"
```

## Pipeline Stages

### Stage 1: Structure-Aware Segmentation
Partitions regulatory documents into semantically coherent segments using TOC-aligned boundaries.

### Stage 2: Ontology-Constrained Extraction
Extracts entities and relationships following the ESGMKG schema:
- **Entity Types**: Industry, ReportingFramework, Category, Metric (Direct/Calculated/Input), Model
- **Relationships**: ReportUsing, Include, ConsistOf, IsCalculatedBy, RequiresInputFrom

### Stage 3: Validation
Validates extracted knowledge graphs using 10 semantic rules and calculates 4 quality metrics:
- CQ Violation Rate
- Triple Retention Rate
- Provenance Completeness
- Ontology Compliance

## Results Summary

| Metric | Ontology-Guided | Baseline | Improvement |
|--------|----------------|----------|-------------|
| CQ Violation Rate | 0.5% | 39.0% | -38.5% |
| Triple Retention | 88.9% | 33.3% | +55.6% |
| Provenance Completeness | 100% | 0% | +100% |
| Ontology Compliance | 100% | 55.9% | +44.1% |

## Supported Documents

- SASB Industry Standards
- TCFD Climate-Related Disclosures
- IFRS S2 Sustainability Standards
- Australian AASB S2 Standards

## Citation

```bibtex
@misc{ontometric2026,
  title={OntoMetric: An Ontology-Driven LLM-Assisted Framework for Automated ESG Metric Knowledge Graph Generation},
  author={Anonymous},
  note={Under Review},
  year={2026}
}
```

## License

This project is for research and educational purposes.
