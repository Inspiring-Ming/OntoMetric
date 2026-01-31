================================================================================
ONTOMETRIC PIPELINE - RESULTS VISUALIZATION AND ANALYSIS
================================================================================
Generated: 2025-11-18
Purpose: Comprehensive analysis and visualization of pipeline results

================================================================================
DIRECTORY STRUCTURE
================================================================================

result_visualisation_and_analysis/
├── generate_comprehensive_analysis.py    # Main analysis script
├── README.txt                            # This file
│
├── stage2_analysis/                      # Stage 2 extraction analysis
│   ├── stage2_extraction_comparison.png  # Entities/relationships extracted
│   ├── stage2_extraction_comparison_EXPLANATION.txt
│   ├── stage2_extraction_summary.csv     # Stage 2 summary table
│   ├── stage3_validation_summary.csv     # Stage 3 summary table
│   ├── cost_summary.csv                  # Cost summary table
│   └── combined_summary.csv              # All metrics combined
│
├── stage3_analysis/                      # Stage 3 validation analysis
│   ├── stage3_validation_quality.png     # Quality scores and retention rates
│   └── stage3_filtering_impact.png       # Before/after validation filtering
│
├── stage2_vs_stage3_comparison/          # Cross-stage comparison
│   ├── overall_summary.png               # 6-panel summary of all metrics
│   └── comprehensive_analysis_report.txt # Detailed analysis report
│
└── cost_analysis/                        # Cost efficiency analysis
    ├── cost_comparison.png               # Cost and token usage by document
    └── cost_per_entity.png               # Cost efficiency metrics

================================================================================
QUICK START
================================================================================

To regenerate all visualizations and reports:

    cd /path/to/Ontometric_pipeline
    python3 result_visualisation_and_analysis/generate_comprehensive_analysis.py

This will:
- Load data from Stage 2, Stage 3, and cost tracking outputs
- Generate 6 comprehensive visualization charts
- Create 4 summary CSV tables
- Produce a detailed text report with key findings

================================================================================
KEY FINDINGS SUMMARY
================================================================================

EXTRACTION (Stage 2)
--------------------
Ontology-Guided:
  • 364 entities, 350 relationships extracted
  • Average: 72.8 entities/doc, 70.0 relationships/doc

Baseline:
  • 729 entities, 940 relationships extracted
  • Average: 145.8 entities/doc, 188.0 relationships/doc
  • Extracted 100.3% MORE entities than ontology-guided

VALIDATION (Stage 3)
--------------------
Ontology-Guided:
  • 100.0% entity retention rate (0 entities filtered)
  • 100.0% average quality score
  • 0 validation violations

Baseline:
  • 63.5% entity retention rate (266 entities filtered)
  • 63.0% average quality score
  • 1,206 validation violations removed

COST EFFICIENCY
---------------
Ontology-Guided:
  • Total cost: $4.53
  • 413,831 input tokens, 218,905 output tokens
  • 60 segments processed
  • $0.905 per document

Baseline:
  • Total cost: $4.48
  • 181,125 input tokens, 262,470 output tokens
  • 174 segments processed
  • $0.896 per document

KEY INSIGHT: Baseline costs 1% less but produces 37% lower quality results

================================================================================
OUTPUT FILES BY DIRECTORY
================================================================================

1. STAGE2_ANALYSIS/ (Stage 2 Extraction Results)
   ------------------------------------------------
   • stage2_extraction_comparison.png (192KB)
     - Side-by-side comparison of entities and relationships extracted
     - Shows baseline extracts ~2x more than ontology-guided

   • stage2_extraction_comparison_EXPLANATION.txt
     - Detailed explanation of the visualization

   • stage2_extraction_summary.csv
     - Pivot table: entities/relationships by document and method

   • stage3_validation_summary.csv
     - Validation metrics: retention rates, quality scores

   • cost_summary.csv
     - Cost data: tokens, segments, total cost

   • combined_summary.csv
     - All metrics in one table for easy comparison

2. STAGE3_ANALYSIS/ (Stage 3 Validation Results)
   ------------------------------------------------
   • stage3_validation_quality.png (189KB)
     - Quality scores: ontology-guided 100%, baseline 63%
     - Entity retention rates comparison

   • stage3_filtering_impact.png (333KB)
     - 4-panel view: before/after validation for both methods
     - Shows ontology-guided has zero filtering

3. STAGE2_VS_STAGE3_COMPARISON/ (Cross-Stage Analysis)
   ------------------------------------------------------
   • overall_summary.png (304KB)
     - 6-panel dashboard with all key metrics:
       * Total entities extracted (Stage 2)
       * Total relationships extracted (Stage 2)
       * Average quality score (Stage 3)
       * Total validated entities (Stage 3)
       * Total violations removed (Stage 3)
       * Total extraction cost

   • comprehensive_analysis_report.txt
     - Text report with detailed statistics and key findings

4. COST_ANALYSIS/ (Cost Efficiency)
   -----------------------------------
   • cost_comparison.png (244KB)
     - Total cost and token usage by document
     - Similar costs between methods (~$0.90/doc)

   • cost_per_entity.png (211KB)
     - Cost efficiency metric: dollars per entity
     - Shows cost-effectiveness of each approach

================================================================================
DATA SOURCES
================================================================================

The analysis script reads data from:

1. Stage 2 Ontology-Guided:
   outputs/stage2_ontology_guided_extraction/*_ontology_guided.json

2. Stage 2 Baseline:
   outputs/stage2_baseline_llm_extraction/*_baseline_llm.json

3. Stage 3 Ontology-Guided Validation:
   outputs/stage3_ontology_guided_validation/*_validated.json

4. Stage 3 Baseline Validation:
   outputs/stage3_baseline_llm_comparison/*_validated.json

5. Cost Tracking:
   outputs/cost_tracking/*_cost_report.json

================================================================================
DOCUMENTS ANALYZED
================================================================================

1. SASB Semiconductors Standard (1.SASB-semiconductors-standard_en-gb)
2. SASB Commercial Banks Standard (1. SASB-commercial-banks-standard_en-gb)
3. IFRS S2 Climate Disclosures (1.issb(sasb)-general-a-ifrs-s2-climate-related-disclosures)
4. TCFD Final Report 2017 (2.FINAL-2017-TCFD-Report)
5. Australia AASB S2 (2.Australia-AASBS2_09-24)

Total: 5 ESG disclosure documents

================================================================================
METHODOLOGY COMPARISON
================================================================================

ONTOLOGY-GUIDED EXTRACTION
--------------------------
Approach:
  • Uses ESGMKG ontology schema for structured extraction
  • SPARQL validation during extraction
  • Strict type constraints and relationship validation
  • Provenance tracking built-in

Advantages:
  ✓ Perfect quality (100% validation pass rate)
  ✓ Zero post-extraction filtering needed
  ✓ Schema-compliant entities and relationships
  ✓ Full traceability to source documents

Tradeoffs:
  ✗ Extracts fewer entities (more conservative)
  ✗ Slightly higher cost per entity

BASELINE (UNCONSTRAINED) EXTRACTION
------------------------------------
Approach:
  • No ontology guidance during extraction
  • LLM generates entities/relationships freely
  • Validation applied post-extraction

Advantages:
  ✓ Extracts more entities (2x volume)
  ✓ Slightly lower total cost
  ✓ Faster per-segment processing

Tradeoffs:
  ✗ 37% lower quality score
  ✗ 36.5% of entities filtered in validation
  ✗ 100% of relationships invalid (VR010 violations)
  ✗ No provenance tracking

================================================================================
VALIDATION RULES APPLIED (Stage 3)
================================================================================

Critical Rules (must pass):
  VR001: Entity ID uniqueness
  VR002: Relationship endpoint validity
  VR003: Entity type validity
  VR005: Relationship type validity
  VR009: Entity label presence
  VR010: Valid ESGMKG predicates

Results:
  • Ontology-guided: 0 violations across all rules
  • Baseline: 1,206 violations (mainly VR010)

================================================================================
REQUIREMENTS
================================================================================

Python Dependencies:
  • pandas
  • matplotlib
  • numpy

To install:
  pip3 install pandas matplotlib numpy

================================================================================
USAGE EXAMPLES
================================================================================

# Regenerate all visualizations
python3 result_visualisation_and_analysis/generate_comprehensive_analysis.py

# View the comprehensive report
cat result_visualisation_and_analysis/stage2_vs_stage3_comparison/comprehensive_analysis_report.txt

# Open summary tables
open result_visualisation_and_analysis/stage2_analysis/combined_summary.csv

# View all figures
open result_visualisation_and_analysis/stage2_analysis/*.png
open result_visualisation_and_analysis/stage3_analysis/*.png
open result_visualisation_and_analysis/stage2_vs_stage3_comparison/*.png
open result_visualisation_and_analysis/cost_analysis/*.png

================================================================================
FILE FORMATS
================================================================================

PNG Files:
  • 300 DPI resolution for publication quality
  • 12x8 or 16x6 inch dimensions
  • Professional color scheme

CSV Files:
  • UTF-8 encoding
  • Comma-separated
  • Header row included
  • Compatible with Excel, Google Sheets, R, Python

TXT Files:
  • Plain text format
  • 80-character line width
  • Hierarchical sections with separators

================================================================================
DIRECTORY ORGANIZATION LOGIC
================================================================================

stage2_analysis/
  - Focus: Extraction results from Stage 2
  - Contains: Extraction comparisons, all summary tables
  - Why: Central repository for raw extraction metrics

stage3_analysis/
  - Focus: Validation results from Stage 3
  - Contains: Quality scores, filtering impact visualizations
  - Why: Shows validation process effectiveness

stage2_vs_stage3_comparison/
  - Focus: Cross-stage analysis and overall summary
  - Contains: Comprehensive report, overall dashboard
  - Why: Big picture view of entire pipeline

cost_analysis/
  - Focus: Cost efficiency metrics
  - Contains: Cost comparisons, efficiency metrics
  - Why: Economic analysis separate from quality metrics

================================================================================
CUSTOMIZATION
================================================================================

To modify visualizations:

1. Edit generate_comprehensive_analysis.py
2. Adjust color schemes in plot functions:
   - Ontology-guided colors: #2E86AB, #06A77D
   - Baseline colors: #A23B72, #F77F00

3. Change output directories:
   - Update directory paths in __init__() method

4. Add new metrics:
   - Extend load_*_data() functions
   - Create new plot_*() function
   - Add to run_full_analysis()

================================================================================
TROUBLESHOOTING
================================================================================

Issue: Script fails with "Module not found"
Solution: Install required packages (see REQUIREMENTS)

Issue: No cost data shown ($0.00)
Solution: Ensure cost tracking JSON files have 'summary' field with correct structure

Issue: Missing documents in analysis
Solution: Check that all 5 documents have corresponding files in Stage 2 and Stage 3 directories

Issue: Charts look different
Solution: Matplotlib version may affect styling. Script uses 'seaborn-v0_8-whitegrid' style

================================================================================
CONTACT & ATTRIBUTION
================================================================================

This analysis framework was created for the Ontometric Pipeline project.
For questions or issues, please refer to the main project documentation.

================================================================================
VERSION HISTORY
================================================================================

v2.0 (2025-11-18)
- Reorganized into 4 logical subdirectories
- Added explanation files for visualizations
- Improved directory structure clarity

v1.0 (2025-11-18)
- Initial release
- 6 comprehensive visualizations
- 4 summary tables
- Detailed text report
- Support for 5 ESG documents
- Cost tracking integration

================================================================================
END OF README
================================================================================
