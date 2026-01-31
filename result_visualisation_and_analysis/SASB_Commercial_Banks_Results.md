# SASB Commercial Banks - Experimental Results

## Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Commercial Banks Sustainability Accounting Standard FINANCIALS SECTOR | 10 | 10 | Sustainability Disclosure Topics & Metrics (pages 6-7) |

## Stage 2: Ontology-Guided Extraction
| Number of entities extracted | Example of triple |
|------------------------------|-------------------|
| 53 | (Metric, IsCalculatedBy, Model): "Number of data breaches" |

## Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 53 | 42 | 82.72 | 79.25 | 79.25 | "Number of Past Due Small Business Loans" - financial metric, not ESG |

## Resulting Ontology
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 8 | 1 | 27 | 5 | 1 | 42 |
