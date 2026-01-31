# Results Visualization and Analysis

This directory contains experimental results, visualizations, and analysis for the OntoMetric pipeline.

## Contents

```
result_visualisation_and_analysis/
├── README.md                    # This file
├── Results.md                   # Complete experimental results for all documents
├── ANALYSIS_FINDINGS.md         # Category-Metric relationship analysis
│
├── figures/                     # Knowledge graph visualizations
│   ├── SASB_Commercial_Banks_ontology_graph.png
│   ├── SASB_Semiconductors_ontology_graph.png
│   ├── IFRS_S2_ontology_graph.png
│   ├── Australia_AASB_S2_ontology_graph.png
│   ├── TCFD_Report_ontology_graph.png
│   └── quality_metrics_comparison.png
│
├── paper_figures/               # Publication-ready figures
│   ├── fig1_method_comparison.png
│   ├── fig2_document_performance.png
│   ├── fig3_validation_quality.png
│   └── fig_ontology_structure.png
│
├── detailed_entity_breakdowns/  # Per-document entity details
│   ├── SASB_Commercial_Banks_Detailed_Entities.md
│   ├── SASB_Semiconductors_Detailed_Entities.md
│   ├── IFRS_S2_Detailed_Entities.md
│   ├── Australia_AASB_S2_Detailed_Entities.md
│   └── TCFD_Report_Detailed_Entities.md
│
├── cost_analysis/               # LLM cost tracking and efficiency
├── stage2_ontology/             # Stage 2 extraction analysis
├── stage2_comparison/           # Ontology-guided vs baseline comparison
├── stage3_ontology/             # Stage 3 validation analysis
├── stage3_comparison/           # Validation quality comparison
│
├── scripts/                     # Analysis and visualization scripts
└── run_all_analyses.py          # Master script to regenerate all figures
```

## Quick Start

Regenerate all visualizations:

```bash
python3 result_visualisation_and_analysis/run_all_analyses.py
```

## Key Results

| Document | Entities | Validated | Retention |
|----------|----------|-----------|-----------|
| SASB Commercial Banks | 53 | 42 | 79.25% |
| SASB Semiconductors | 69 | 62 | 89.86% |
| IFRS S2 | 80 | 68 | 85.00% |
| Australia AASB S2 | 74 | 66 | 89.19% |
| TCFD Report | 88 | 57 | 64.77% |
| **Total** | **364** | **295** | **81.04%** |

## Comparison: Ontology-Guided vs Baseline

| Metric | Ontology-Guided | Baseline | Improvement |
|--------|----------------|----------|-------------|
| CQ Violation Rate | 0.5% | 39.0% | -38.5% |
| Triple Retention | 88.9% | 33.3% | +55.6% |
| Provenance | 100% | 0% | +100% |
| Ontology Compliance | 100% | 55.9% | +44.1% |
