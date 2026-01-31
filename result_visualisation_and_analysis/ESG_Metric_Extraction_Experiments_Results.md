# ESG Metric Extraction Experiments - Results

---

## SASB Commercial Banks

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Commercial Banks Sustainability Accounting Standard FINANCIALS SECTOR | 10 | 10 | Sustainability Disclosure Topics & Metrics (pages 6-7) |

### Stage 2: Ontology-Guided Extraction
| Number of entities before merging | Number of entities after merging | Example of triple |
|--------------------------------------------|--------------------------------------------|-------------------|
| 53 | 53 | (Metric, IsCalculatedBy, Model): "Number of data breaches" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 53 | 42 | 82.72 | 79.25 | 79.25 | "Number of Past Due Small Business Loans" - financial metric, not ESG |

### Resulting Ontology
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 8 | 1 | 27 | 5 | 1 | 42 |

---

## SASB Semiconductors

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Semiconductors Sustainability Accounting Standard TECHNOLOGY & COMMUNICATIONS SECTOR | 13 | 13 | Greenhouse Gas Emissions (pages 8-10) |

### Stage 2: Ontology-Guided Extraction
| Number of entities before merging | Number of entities after merging | Example of triple |
|--------------------------------------------|--------------------------------------------|-------------------|
| 69 | 69 | (Category, ConsistOf, Metric): "GHG Emissions" → "Gross global Scope 1 emissions" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 69 | 62 | 82.02 | 89.86 | 90.14 | Non-ESG competitive behavior metrics |

### Resulting Ontology
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 11 | 1 | 44 | 12 | 1 | 69 |

---

## IFRS S2

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| June 2023 IFRS S2 IFRS Sustainability Disclosure Standard | 10 | 10 | Strategy (pages 8-23) |

### Stage 2: Ontology-Guided Extraction
| Number of entities before merging | Number of entities after merging | Example of triple |
|--------------------------------------------|--------------------------------------------|-------------------|
| 80 | 80 | (ReportingFramework, Include, Category): IFRS S2 → "Strategy" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 80 | 68 | 81.48 | 85.00 | 89.23 | Metrics with missing unit specifications |

### Resulting Ontology
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 7 | 5 | 63 | 4 | 1 | 80 |

---

## Australia AASB S2

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Australian Sustainability Reporting Standard AASB S2 September 2024 | 8 | 8 | APPENDICES (pages 11-57) |

### Stage 2: Ontology-Guided Extraction
| Number of entities before merging | Number of entities after merging | Example of triple |
|--------------------------------------------|--------------------------------------------|-------------------|
| 74 | 74 | (Category, ConsistOf, Metric): "Strategy" → climate risk metrics |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 74 | 66 | 83.33 | 89.19 | 86.67 | Metrics lacking valid code/unit |

### Resulting Ontology
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 8 | 4 | 56 | 5 | 1 | 74 |

---

## TCFD Report

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| FINAL-2017-TCFD-Report | 19 | 19 | Guidance for All Sectors (pages 19-24) |

### Stage 2: Ontology-Guided Extraction
| Number of entities before merging | Number of entities after merging | Example of triple |
|--------------------------------------------|--------------------------------------------|-------------------|
| 88 | 88 | (Category, ConsistOf, Metric): "Guidance All Sectors" → "Scope 1, 2, and 3 GHG emissions" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 88 | 57 | 80.09 | 64.77 | 62.79 | Narrative guidance paragraphs extracted as Metrics |

### Resulting Ontology
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 10 | 9 | 65 | 3 | 1 | 88 |

---

## Summary Statistics

### Overall Results
| Document | Total Pages | Segments | Entities Extracted | Entities Validated | Retention Rate (%) |
|----------|------------|----------|-------------------|-------------------|--------------------|
| SASB Commercial Banks | 23 | 10 | 53 | 42 | 79.25 |
| SASB Semiconductors | 27 | 13 | 69 | 62 | 89.86 |
| IFRS S2 | 46 | 10 | 80 | 68 | 85.00 |
| Australia AASB S2 | 58 | 8 | 74 | 66 | 89.19 |
| TCFD Report | 74 | 19 | 88 | 57 | 64.77 |
| **TOTAL** | **228** | **60** | **364** | **295** | **81.04** |

### Validation Quality Metrics
| Document | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) |
|----------|----------------------|----------------------|---------------------------|
| SASB Commercial Banks | 82.72 | 79.25 | 79.25 |
| SASB Semiconductors | 82.02 | 89.86 | 90.14 |
| IFRS S2 | 81.48 | 85.00 | 89.23 |
| Australia AASB S2 | 83.33 | 89.19 | 86.67 |
| TCFD Report | 80.09 | 64.77 | 62.79 |
