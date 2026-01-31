# ESG Metric Extraction Experiments - All Results

---

## SASB Commercial Banks

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Commercial Banks Sustainability Accounting Standard FINANCIALS SECTOR | 10 | 10 | Sustainability Disclosure Topics & Metrics (pages 6-7) |

### Stage 2: Ontology-Guided Extraction
| Number of entities extracted | Example of triple |
|------------------------------|-------------------|
| 53 | (metric_SASB-CB_6_001, IsCalculatedBy, model_SASB-CB_6_001): "Data Breaches Composite Metric" calculated by composite model |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 53 | 42 | 82.72 | 79.25 | 79.25 | "Number of Past Due Small Business Loans" - financial metric, not ESG |

### Resulting Ontology

**Entity Counts:**
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 8 | 1 | 27 | 5 | 1 | 42 |

#### Categories
| Label | Section ID | Metrics |
|-------|------------|---------|
| Incorporation of Environmental, Social, and Governance Factors in Credit Analysis | CB-410a | 2 |
| Data Security | CB-230a | 2 |
| Financial Inclusion & Capacity Building | CB-240a | 4 |
| Incorporation of Environmental, Social, and Governance (ESG) Factors in Credit Analysis | CB-410a | 1 |
| Financed Emissions | CB-410b | 4 |
| Activity Metrics | FN-CB-000 | 0 |
| Business Ethics | FN-CB-510a | 2 |
| Systemic Risk Management | FN-CB-550a | 2 |

#### Industries
| Label | Sector |
|-------|--------|
| Commercial Banks | Financials |

#### Metrics
| Label | Code | Type | Category |
|-------|------|------|----------|
| Whistleblower Policies and Procedures | FN-CB-510a.2 | DirectMetric | Business Ethics |
| G-SIB Overall Score | FN-CB-550a.1 | CalculatedMetric | Systemic Risk Management |
| G-SIB Size Score | N/A | InputMetric | N/A |
| G-SIB Cross-Jurisdictional Activity Score | N/A | InputMetric | N/A |
| Number of Loans Outstanding | N/A | InputMetric | N/A |
| Amount of Loans Outstanding | N/A | InputMetric | N/A |
| ESG Factors in Credit Analysis | N/A | DirectMetric | Incorporation of Environmental, Social, and Governance Factors in Credit Analysis |
| Absolute Gross Financed Emissions | FN-CB-410b.1 | DirectMetric | Financed Emissions |
| Gross Exposure by Industry and Asset Class | FN-CB-410b.2 | DirectMetric | Financed Emissions |
| Percentage of Gross Exposure in Financed Emissions | FN-CB-410b.3 | DirectMetric | Financed Emissions |
| Financed Emissions Methodology Description | FN-CB-410b.4 | DirectMetric | Financed Emissions |
| Monetary Losses from Legal Proceedings | FN-CB-510a.1 | DirectMetric | Business Ethics |
| Integration of Stress Test Results into Business Strategy | FN-CB-550a.2 | DirectMetric | Systemic Risk Management |
| Data Breaches Composite Metric | FN-CB-230a.1 | CalculatedMetric | Data Security |
| Percentage of Personal Data Breaches | N/A | InputMetric | N/A |
| Data Security Risk Approach | FN-CB-230a.2 | DirectMetric | Data Security |
| Small Business and Community Development Loans | FN-CB-240a.1 | CalculatedMetric | Financial Inclusion & Capacity Building |
| Number of Small Business Loans | N/A | InputMetric | N/A |
| Amount of Small Business Loans Outstanding | N/A | InputMetric | N/A |
| Past Due and Nonaccrual Small Business Loans | FN-CB-240a.2 | CalculatedMetric | Financial Inclusion & Capacity Building |
| No-Cost Retail Checking Accounts | FN-CB-240a.3 | DirectMetric | Financial Inclusion & Capacity Building |
| Financial Literacy Initiative Participants | FN-CB-240a.4 | DirectMetric | Financial Inclusion & Capacity Building |
| ESG Factors in Credit Analysis | FN-CB-410a.2 | DirectMetric | Incorporation of Environmental, Social, and Governance (ESG) Factors in Credit Analysis, Incorporation of Environmental, Social, and Governance Factors in Credit Analysis |
| Size Category Score | N/A | InputMetric | N/A |
| Number of Data Breaches | N/A | InputMetric | N/A |
| Percentage Personal Data Breaches | N/A | InputMetric | N/A |
| Number of Account Holders Affected | N/A | InputMetric | N/A |

#### Models
| Label | Formula | Input Metrics |
|-------|---------|---------------|
| Small Business and Community Development Loans Model | f(NumberOfLoans, AmountOfLoans) | Number of Loans Outstanding, Amount of Loans Outstanding |
| Data Breaches Composite Model | f(NumberBreaches, PercentagePersonal, AccountHoldersAffected) | Percentage of Personal Data Breaches, Number of Data Breaches, Percentage Personal Data Breaches, Number of Account Holders Affected |
| Small Business Loans Composite Model | f(NumberLoans, AmountOutstanding) | Number of Small Business Loans, Amount of Small Business Loans Outstanding |
| Past Due Loans Composite Model | f(NumberPastDue, AmountPastDue) | N/A |
| G-SIB Score Calculation Model | N/A | Size Category Score, G-SIB Size Score, G-SIB Cross-Jurisdictional Activity Score |

#### Reporting Framework
| Label | Version | Year |
|-------|---------|------|
| SASB | 2023-12 | 2023 |

---

## SASB Semiconductors

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Semiconductors Sustainability Accounting Standard TECHNOLOGY & COMMUNICATIONS SECTOR | 13 | 13 | Greenhouse Gas Emissions (pages 8-10) |

### Stage 2: Ontology-Guided Extraction
| Number of entities extracted | Example of triple |
|------------------------------|-------------------|
| 69 | (Category, ConsistOf, Metric): "GHG Emissions" → "Gross global Scope 1 emissions" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 69 | 62 | 82.02 | 89.86 | 90.14 | Non-ESG competitive behavior metrics |

### Resulting Ontology

**Entity Counts:**
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 10 | 1 | 38 | 12 | 1 | 62 |

#### Categories
| Label | Section ID | Metrics |
|-------|------------|---------|
| Greenhouse Gas Emissions | TC-SC-110a | 4 |
| Energy Management in Manufacturing | TC-SC-130a | 2 |
| Water Management | TC-SC-140a | 5 |
| Waste Management | TC-SC-150a | 1 |
| Workforce Health & Safety | TC-SC-320a | 2 |
| Recruiting & Managing a Global & Skilled Workforce | TC-SC-330a | 1 |
| Intellectual Property Protection & Competitive Behaviour | TC-SC-520a | 2 |
| Materials Sourcing | TC-SC-440a | 1 |
| Product Lifecycle Management | TC-SC-410a | 2 |
| Activity Metrics | TC-SC-000 | 1 |

#### Industries
| Label | Sector |
|-------|--------|
| Semiconductors | Technology & Communications |

#### Metrics
| Label | Code | Type | Category |
|-------|------|------|----------|
| Total Energy Consumed | N/A | InputMetric | N/A |
| Grid Electricity Consumed | N/A | InputMetric | N/A |
| Renewable Energy Consumed | N/A | InputMetric | N/A |
| Water Risk Analysis | N/A | DirectMetric | Water Management |
| Water Withdrawn in High Stress Locations (Percentage) | N/A | CalculatedMetric | Water Management |
| Water Withdrawn in High Stress Locations | N/A | InputMetric | N/A |
| Total Water Withdrawn | N/A | InputMetric | N/A |
| Water Consumed in High Stress Locations (Percentage) | N/A | CalculatedMetric | Water Management |
| Water Consumed in High Stress Locations | N/A | InputMetric | N/A |
| Total Water Consumed | N/A | InputMetric | N/A |
| Total Weight of Hazardous Waste Generated | N/A | InputMetric | N/A |
| Weight of Hazardous Waste Recycled | N/A | InputMetric | N/A |
| Percentage of Employees Requiring Work Visa | TC-SC-330a.1 | CalculatedMetric | Recruiting & Managing a Global & Skilled Workforce |
| Number of Employees Requiring Work Visa | N/A | InputMetric | N/A |
| Total Number of Employees | N/A | InputMetric | N/A |
| Percentage of Products Containing IEC 62474 Declarable Substances | TC-SC-410a.1 | CalculatedMetric | Greenhouse Gas Emissions, Product Lifecycle Management |
| Revenue from Products Containing Declarable Substances | N/A | InputMetric | N/A |
| Processor Energy Efficiency | TC-SC-410a.2 | CalculatedMetric | Greenhouse Gas Emissions, Product Lifecycle Management |
| Server Performance per Watt | N/A | InputMetric | N/A |
| Critical Materials Risk Management | TC-SC-440a.1 | DirectMetric | Energy Management in Manufacturing, Materials Sourcing |
| Description of Anti-competitive Fines and Settlements | TC-SC-520a.1 (Note) | DirectMetric | Intellectual Property Protection & Competitive Behaviour |
| Gross Global Scope 1 Emissions and PFC Emissions | TC-SC-110a.1 | CalculatedMetric | Greenhouse Gas Emissions |
| Gross Global Scope 1 Emissions | N/A | InputMetric | N/A |
| Emissions from Perfluorinated Compounds | N/A | InputMetric | N/A |
| GHG Emissions Management Strategy | TC-SC-110a.2 | DirectMetric | Greenhouse Gas Emissions |
| Total Energy Consumed with Grid and Renewable Breakdown | TC-SC-130a.1 | CalculatedMetric | Energy Management in Manufacturing |
| Percentage Grid Electricity | N/A | InputMetric | N/A |
| Percentage Renewable | N/A | InputMetric | N/A |
| Water Withdrawn and Consumed with Baseline Stress | TC-SC-140a.1 | CalculatedMetric | Water Management |
| Hazardous Waste and Recycling Rate | TC-SC-150a.1 | CalculatedMetric | Waste Management |
| Amount of Hazardous Waste | N/A | InputMetric | N/A |
| Percentage of Hazardous Waste Recycled | N/A | InputMetric | N/A |
| Workforce Health Hazards Assessment | TC-SC-320a.1 | DirectMetric | Workforce Health & Safety |
| Monetary Losses from Health and Safety Violations | TC-SC-320a.2 | DirectMetric | Workforce Health & Safety |
| Monetary losses from anti-competitive behaviour | TC-SC-520a.1 | DirectMetric | Water Management, Intellectual Property Protection & Competitive Behaviour |
| Percentage of production from owned facilities | TC-SC-000.B | DirectMetric | Activity Metrics |
| Gross global Scope 1 emissions | N/A | InputMetric | N/A |
| Emissions from perfluorinated compounds | N/A | InputMetric | N/A |

#### Models
| Label | Formula | Input Metrics |
|-------|---------|---------------|
| Energy Consumption Analysis Model | Grid % = (Grid Electricity / Total Energy) × 100; Renewable % = (Renewable Energy / Total Energy) × 100 | Total Energy Consumed, Grid Electricity Consumed, Renewable Energy Consumed |
| Water Stress Assessment Model | f(TotalWithdrawn, TotalConsumed, StressRegionPct) | N/A |
| Water Withdrawn Stress Percentage Model | (Water withdrawn in high stress locations / Total water withdrawn) × 100 | Water Withdrawn in High Stress Locations, Total Water Withdrawn |
| Water Consumed Stress Percentage Model | (Water consumed in high stress locations / Total water consumed) × 100 | Water Consumed in High Stress Locations, Total Water Consumed |
| Hazardous Waste Recycling Model | (Weight of hazardous waste recycled / Total weight of hazardous waste) × 100 | Total Weight of Hazardous Waste Generated, Weight of Hazardous Waste Recycled |
| Work Visa Percentage Calculation Model | (Number of employees requiring work visa / Total number of employees) × 100 | Number of Employees Requiring Work Visa, Total Number of Employees |
| IEC 62474 Declarable Substances Percentage Model | (Revenue from products with declarable substances / Total revenue from products) × 100 | Revenue from Products Containing Declarable Substances |
| Processor Energy Efficiency Benchmarking Model | f(ServerBenchmark, DesktopBenchmark, LaptopBenchmark) where each benchmark = PerformanceScore / PowerConsumption | Server Performance per Watt |
| GHG Emissions Composite Model | f(Scope1Emissions, PFCEmissions) | Gross Global Scope 1 Emissions, Emissions from Perfluorinated Compounds, Gross global Scope 1 emissions, Emissions from perfluorinated compounds |
| Energy Management Composite Model | f(TotalEnergy, GridElectricityPct, RenewablePct) | Percentage Grid Electricity, Percentage Renewable |
| Water Management Composite Model | f(WaterWithdrawn, WaterConsumed) | N/A |
| Waste Management Composite Model | f(HazardousWaste, RecycledPct) | Amount of Hazardous Waste, Percentage of Hazardous Waste Recycled |

#### Reporting Framework
| Label | Version | Year |
|-------|---------|------|
| SASB | 2023-12 | 2023 |

---

## IFRS S2

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| June 2023 IFRS S2 IFRS Sustainability Disclosure Standard | 10 | 10 | Strategy (pages 8-23) |

### Stage 2: Ontology-Guided Extraction
| Number of entities extracted | Example of triple |
|------------------------------|-------------------|
| 80 | (ReportingFramework, Include, Category): IFRS S2 → "Strategy" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 80 | 68 | 81.48 | 85.0 | 89.23 | Metrics with missing unit specifications |

### Resulting Ontology

**Entity Counts:**
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 6 | 3 | 54 | 4 | 1 | 68 |

#### Categories
| Label | Section ID | Metrics |
|-------|------------|---------|
| Strategy: Climate Risks and Opportunities | Paragraphs 11-14 | 7 |
| Governance | IFRS-S2-Governance | 0 |
| Climate Resilience Assessment | Appendix B, B1-B18 | 9 |
| Climate-Related Financial Disclosures | Para-15-16 | 8 |
| Metrics and Targets | para_27-37 | 15 |
| Risk Management | Para 24-26 | 0 |

#### Industries
| Label | Sector |
|-------|--------|
| Cross-Industry (General Requirements) | Cross-Industry |
| General Cross-Industry | Cross-Industry |
| General (Cross-Industry) | Cross-Industry |

#### Metrics
| Label | Code | Type | Category |
|-------|------|------|----------|
| Absolute Gross GHG Emissions | Para 29(a)(i) | DirectMetric | Metrics and Targets |
| Financed Emissions (Asset Management) | Para B61 | CalculatedMetric | Metrics and Targets |
| Assets Under Management (AUM) | Para B61(b) | InputMetric | N/A |
| Allocation Methodology for Financed Emissions | Para B61(d) | InputMetric | N/A |
| Financed Emissions (Commercial Banking) | Para B62 | CalculatedMetric | Metrics and Targets |
| Climate-related Transition Risks - Asset Vulnerability | Para 29(b) | DirectMetric | Metrics and Targets |
| Climate-related Physical Risks - Asset Vulnerability | Para 29(c) | DirectMetric | Metrics and Targets |
| Internal Carbon Price | Para 29(f)(ii) | DirectMetric | Metrics and Targets |
| Climate-related Executive Remuneration | Para 29(g)(ii) | DirectMetric | Metrics and Targets |
| Anticipated Financial Effects of Climate-related Risks and Opportunities | N/A | DirectMetric | N/A |
| Current and Anticipated Effects on Business Model and Value Chain | para_13 | DirectMetric | N/A |
| Climate Resilience Assessment | para_22 | DirectMetric | N/A |
| Qualitative Information on Financial Effects | N/A | DirectMetric | Climate Resilience Assessment |
| Quantitative Information on Combined Financial Effects | N/A | DirectMetric | Climate Resilience Assessment |
| Climate Resilience Assessment | N/A | DirectMetric | Climate Resilience Assessment |
| Climate-related Scenario Analysis Methodology | N/A | DirectMetric | Climate Resilience Assessment |
| Absolute Gross Greenhouse Gas Emissions | para_29(a)(i) | CalculatedMetric | Metrics and Targets |
| Scope 1 Greenhouse Gas Emissions | para_29(a)(i)(1) | InputMetric | N/A |
| Scope 2 Greenhouse Gas Emissions | para_29(a)(i)(2) | InputMetric | N/A |
| Scope 3 Greenhouse Gas Emissions | para_29(a)(i)(3) | InputMetric | N/A |
| Assets Vulnerable to Climate-Related Transition Risks | para_29(b) | DirectMetric | Metrics and Targets |
| Assets Vulnerable to Climate-Related Physical Risks | para_29(c) | DirectMetric | Metrics and Targets |
| Assets Aligned with Climate-Related Opportunities | para_29(d) | DirectMetric | Metrics and Targets |
| Capital Deployment for Climate-Related Risks and Opportunities | para_29(e) | DirectMetric | Metrics and Targets |
| Internal Carbon Prices | para_29(f) | DirectMetric | Metrics and Targets |
| Climate-Related Considerations in Executive Remuneration | para_29(g) | DirectMetric | Metrics and Targets |
| Climate-Related Targets | para_33-37 | DirectMetric | Metrics and Targets |
| Governance Body Responsibilities | IFRS-S2-6a | DirectMetric | N/A |
| Governance Skills and Competencies | IFRS-S2-6a-ii | DirectMetric | N/A |
| Governance Information Flow | IFRS-S2-6a-iii | DirectMetric | Strategy: Climate Risks and Opportunities |
| Management Role in Climate Governance | IFRS-S2-6b | DirectMetric | Strategy: Climate Risks and Opportunities |
| Climate-Related Risks and Opportunities Description | N/A | DirectMetric | Strategy: Climate Risks and Opportunities |
| Climate-related scenario analysis approach | IFRS S2 Para 22, B1-B18 | CalculatedMetric | Climate Resilience Assessment |
| Exposure to climate-related risks and opportunities | IFRS S2 B4-B5 | InputMetric | Climate Resilience Assessment |
| Skills, capabilities and resources available | IFRS S2 B6-B7 | InputMetric | Climate Resilience Assessment |
| Selection of inputs to climate-related scenario analysis | IFRS S2 B11-B13 | DirectMetric | Climate Resilience Assessment |
| Analytical choices in climate-related scenario analysis | IFRS S2 B14-B15 | DirectMetric | Climate Resilience Assessment |
| Effects on Business Model and Value Chain | IFRS-S2-13 | DirectMetric | Strategy: Climate Risks and Opportunities |
| Climate Transition Plan | IFRS-S2-14 | DirectMetric | Strategy: Climate Risks and Opportunities |
| Financial Effects of Climate Risks and Opportunities | IFRS-S2-15-21 | DirectMetric | Strategy: Climate Risks and Opportunities |
| Climate Resilience Assessment | IFRS-S2-22 | DirectMetric | Strategy: Climate Risks and Opportunities |
| Effects on Business Model and Value Chain | Para 9(b) | DirectMetric | N/A |
| Climate Resilience Assessment | Para 9(e) | DirectMetric | N/A |
| Description of Climate-related Risks and Opportunities | Para 10(a) | DirectMetric | N/A |
| Classification of Physical vs Transition Risks | Para 10(b) | DirectMetric | N/A |
| Definition of Time Horizons | Para 10(d) | DirectMetric | N/A |
| Direct Mitigation and Adaptation Efforts | Para-14(a)(ii) | DirectMetric | Climate-Related Financial Disclosures |
| Indirect Mitigation and Adaptation Efforts | Para-14(a)(iii) | DirectMetric | Climate-Related Financial Disclosures |
| Climate-Related Transition Plan | Para-14(a)(iv) | DirectMetric | Climate-Related Financial Disclosures |
| Plans to Achieve Climate Targets | Para-14(a)(v) | DirectMetric | Climate-Related Financial Disclosures |
| Resourcing of Climate Activities | Para-14(b) | DirectMetric | Climate-Related Financial Disclosures |
| Progress of Climate Plans | Para-14(c) | DirectMetric | Climate-Related Financial Disclosures |
| Current Financial Effects of Climate Risks and Opportunities | Para-15(a) | DirectMetric | Climate-Related Financial Disclosures |
| Effects on Financial Position and Performance | Para-16(a) | DirectMetric | Climate-Related Financial Disclosures |

#### Models
| Label | Formula | Input Metrics |
|-------|---------|---------------|
| Financed Emissions Calculation Model (Asset Management) | f(AUM, Allocation_Method, Investee_Emissions) | Assets Under Management (AUM), Allocation Methodology for Financed Emissions |
| Financed Emissions Calculation Model (Commercial Banking) | f(Gross_Exposure, Industry_Classification, Asset_Class, Allocation_Method, Counterparty_Emissions) | N/A |
| GHG Emissions Aggregation Model | Total GHG Emissions = Scope 1 + Scope 2 + Scope 3 | Scope 1 Greenhouse Gas Emissions, Scope 2 Greenhouse Gas Emissions, Scope 3 Greenhouse Gas Emissions |
| Climate Resilience Assessment Model | f(Exposure Assessment, Available Skills and Resources) | Exposure to climate-related risks and opportunities, Skills, capabilities and resources available |

#### Reporting Framework
| Label | Version | Year |
|-------|---------|------|
| IFRS S2 Climate-Related Disclosures | 2023 | 2023 |

---

## Australia AASB S2

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| Australian Sustainability Reporting Standard AASB S2 September 2024 | 8 | 8 | APPENDICES (pages 11-57) |

### Stage 2: Ontology-Guided Extraction
| Number of entities extracted | Example of triple |
|------------------------------|-------------------|
| 74 | (Category, ConsistOf, Metric): "Strategy" → climate risk metrics |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 74 | 66 | 83.33 | 89.19 | 86.67 | Metrics lacking valid code/unit |

### Resulting Ontology

**Entity Counts:**
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 7 | 1 | 52 | 5 | 1 | 66 |

#### Categories
| Label | Section ID | Metrics |
|-------|------------|---------|
| Governance | 5 | 2 |
| Metrics and Targets | 27 | 8 |
| Strategy | 8 | 4 |
| Climate-related metrics | Paragraph 29 | 11 |
| Risk Management - Scope 3 and Financed Emissions | B52-B65 | 0 |
| Asset Management Financed Emissions | B61 | 4 |
| Insurance Financed Emissions | B63 | 3 |

#### Industries
| Label | Sector |
|-------|--------|
| Financial Services | Financials |

#### Metrics
| Label | Code | Type | Category |
|-------|------|------|----------|
| Current Financial Effects of Climate-related Risks and Opportunities | AASB-S2-15(a) | DirectMetric | Strategy |
| Climate Resilience Assessment | AASB-S2-22 | DirectMetric | Strategy |
| Risk Management Processes | AASB-S2-25 | DirectMetric | N/A |
| Absolute Gross Greenhouse Gas Emissions | AASB-S2-29(a)(i) | CalculatedMetric | Metrics and Targets |
| Scope 1 Greenhouse Gas Emissions | N/A | InputMetric | N/A |
| Scope 2 Greenhouse Gas Emissions | N/A | InputMetric | N/A |
| Scope 3 Greenhouse Gas Emissions | N/A | InputMetric | N/A |
| Assets Vulnerable to Climate-related Transition Risks | AASB-S2-29(b) | DirectMetric | Metrics and Targets |
| Assets Vulnerable to Climate-related Physical Risks | AASB-S2-29(c) | DirectMetric | Metrics and Targets |
| Assets Aligned with Climate-related Opportunities | AASB-S2-29(d) | DirectMetric | Metrics and Targets |
| Capital Deployment for Climate | AASB-S2-29(e) | DirectMetric | Metrics and Targets |
| Internal Carbon Price | AASB-S2-29(f) | DirectMetric | Metrics and Targets |
| Executive Remuneration Linked to Climate Considerations | AASB-S2-29(g)(ii) | CalculatedMetric | Metrics and Targets |
| Climate-linked Remuneration Amount | N/A | InputMetric | N/A |
| Climate-related Targets | AASB-S2-33 | DirectMetric | Metrics and Targets |
| Scope 1 greenhouse gas emissions | 29(a)(i)(1) | DirectMetric | Climate-related metrics |
| Scope 2 greenhouse gas emissions | 29(a)(i)(2) | DirectMetric | Climate-related metrics |
| Scope 3 greenhouse gas emissions | 29(a)(i)(3) | DirectMetric | Climate-related metrics |
| Absolute gross greenhouse gas emissions | 29(a)(i) | CalculatedMetric | Climate-related metrics |
| Assets or business activities vulnerable to climate-related transition risks | 29(b) | DirectMetric | Climate-related metrics |
| Assets or business activities vulnerable to climate-related physical risks | 29(c) | DirectMetric | Climate-related metrics |
| Assets or business activities aligned with climate-related opportunities | 29(d) | DirectMetric | Climate-related metrics |
| Capital deployment towards climate-related risks and opportunities | 29(e) | DirectMetric | Climate-related metrics |
| Internal carbon price | 29(f)(ii) | DirectMetric | Climate-related metrics |
| Percentage of executive remuneration linked to climate-related considerations | 29(g)(ii) | DirectMetric | Climate-related metrics |
| Financed emissions (absolute gross) | B61(a), B62(a), B63(a) | CalculatedMetric | Climate-related metrics |
| Entity financial exposure (AUM or gross exposure) | B61(b), B62(b), B63(b) | InputMetric | N/A |
| Counterparty greenhouse gas emissions (Scope 1, 2, 3) | B58–B63 | InputMetric | N/A |
| Governance Body Oversight Disclosures | AASB S2.6(a) | DirectMetric | Governance |
| Management Role in Climate Governance | AASB S2.6(b) | DirectMetric | Governance |
| Business Model and Value Chain Effects | AASB S2.13 | DirectMetric | Strategy |
| Strategy and Decision-making Disclosures | AASB S2.14 | DirectMetric | Strategy |
| Absolute Gross Financed Emissions - Asset Management | B61(a) | CalculatedMetric | Asset Management Financed Emissions |
| Scope 1 Financed Emissions - Asset Management | N/A | InputMetric | N/A |
| Scope 2 Financed Emissions - Asset Management | N/A | InputMetric | N/A |
| Scope 3 Financed Emissions - Asset Management | N/A | InputMetric | N/A |
| Total AUM Included in Financed Emissions | B61(b) | DirectMetric | Asset Management Financed Emissions |
| Percentage of Total AUM Included | B61(c) | DirectMetric | Asset Management Financed Emissions |
| Financed Emissions Calculation Methodology | B61(d) | DirectMetric | Asset Management Financed Emissions |
| Absolute Gross Financed Emissions - Commercial Banking | B62(a) | CalculatedMetric | N/A |
| Scope 1 Financed Emissions - Commercial Banking | N/A | InputMetric | N/A |
| Scope 2 Financed Emissions - Commercial Banking | N/A | InputMetric | N/A |
| Scope 3 Financed Emissions - Commercial Banking | N/A | InputMetric | N/A |
| Gross Exposure by Industry by Asset Class | B62(b) | DirectMetric | N/A |
| Percentage of Gross Exposure Included - Commercial Banking | B62(c) | DirectMetric | N/A |
| Financed Emissions Methodology - Commercial Banking | B62(d) | DirectMetric | N/A |
| Absolute Gross Financed Emissions - Insurance | B63(a) | CalculatedMetric | Insurance Financed Emissions |
| Scope 1 Financed Emissions - Insurance | N/A | InputMetric | N/A |
| Scope 2 Financed Emissions - Insurance | N/A | InputMetric | N/A |
| Scope 3 Financed Emissions - Insurance | N/A | InputMetric | N/A |
| Gross Exposure by Industry by Asset Class - Insurance | B63(b) | DirectMetric | Insurance Financed Emissions |
| Percentage of Gross Exposure Included - Insurance | B63(c) | DirectMetric | Insurance Financed Emissions |

#### Models
| Label | Formula | Input Metrics |
|-------|---------|---------------|
| GHG Emissions Aggregation Model | Total GHG Emissions = Scope 1 + Scope 2 + Scope 3 | Scope 1 Greenhouse Gas Emissions, Scope 2 Greenhouse Gas Emissions, Scope 3 Greenhouse Gas Emissions, Scope 1 greenhouse gas emissions, Scope 2 greenhouse gas emissions, Scope 3 greenhouse gas emissions |
| Climate-linked Executive Remuneration Model | Climate-linked Remuneration % = (Climate-linked Remuneration / Total Executive Remuneration) × 100 | Climate-linked Remuneration Amount |
| Asset Management Financed Emissions Calculation Model | Total Financed Emissions = Scope 1 + Scope 2 + Scope 3 | Scope 1 Financed Emissions - Asset Management, Scope 2 Financed Emissions - Asset Management, Scope 3 Financed Emissions - Asset Management, Entity financial exposure (AUM or gross exposure), Counterparty greenhouse gas emissions (Scope 1, 2, 3) |
| Commercial Banking Financed Emissions Calculation Model | Total Financed Emissions = Σ(Scope 1 + Scope 2 + Scope 3) by Industry by Asset Class | Scope 1 Financed Emissions - Commercial Banking, Scope 2 Financed Emissions - Commercial Banking, Scope 3 Financed Emissions - Commercial Banking |
| Insurance Financed Emissions Calculation Model | Total Financed Emissions = Σ(Scope 1 + Scope 2 + Scope 3) by Industry by Asset Class | Scope 1 Financed Emissions - Insurance, Scope 2 Financed Emissions - Insurance, Scope 3 Financed Emissions - Insurance |

#### Reporting Framework
| Label | Version | Year |
|-------|---------|------|
| AASB S2 Climate-related Disclosures | 2024 | 2024 |

---

## TCFD Report

### Stage 1: PDF Segmentation
| Title page | Number of sections | Number of segments extracted | Example of segment |
|------------|-------------------|------------------------------|-------------------|
| FINAL-2017-TCFD-Report | 19 | 19 | Guidance for All Sectors (pages 19-24) |

### Stage 2: Ontology-Guided Extraction
| Number of entities extracted | Example of triple |
|------------------------------|-------------------|
| 88 | (Category, ConsistOf, Metric): "Guidance All Sectors" → "Scope 1, 2, and 3 GHG emissions" |

### Stage 3: Two-Phase Validation
| Validation rules used | Entities before validation | Entities after validation | Schema Compliance (%) | Semantic Accuracy (%) | Relationship Retention (%) | Example of invalid entity removed |
|-------------------------------|---------------------------|--------------------------|----------------------|----------------------|---------------------------|----------------------------------|
| VR001, VR002, VR003, VR004, VR005, VR006 | 88 | 57 | 80.09 | 64.77 | 62.79 | Narrative guidance paragraphs extracted as Metrics |

### Resulting Ontology

**Entity Counts:**
| Category | Industry | Metric | Model | ReportingFramework | Total |
|----------|----------|--------|-------|--------------------|-------|
| 9 | 8 | 36 | 3 | 1 | 57 |

#### Categories
| Label | Section ID | Metrics |
|-------|------------|---------|
| Climate-Related Risks and Opportunities | Section B | 2 |
| Principles for Effective Disclosures | C | 0 |
| Scenario Analysis and Climate-Related Issues | D | 5 |
| Climate-Related Opportunities | 2 | 1 |
| Climate-Related Risks, Opportunities, and Financial Impacts | B | 2 |
| Governance | TCFD-Governance | 4 |
| Metrics and Targets | TCFD-MetricsTargets | 5 |
| Risk Management | TCFD-RiskManagement | 6 |
| Strategy | TCFD-Strategy | 6 |

#### Industries
| Label | Sector |
|-------|--------|
| Financial Markets | Financials |
| Organizations with Climate-Related Exposures | Cross-sector |
| Cross-Sector Organizations | All Sectors |
| Asset Managers and Asset Owners | Financials |
| Financial Sector Organizations | Financials |
| Climate-Exposed Organizations | Cross-sector |
| Cross-Sector (TCFD Applicable) | All Sectors |
| All Sectors | Cross-Sector |

#### Metrics
| Label | Code | Type | Category |
|-------|------|------|----------|
| Climate-Related Risks Disclosure | N/A | DirectMetric | Climate-Related Risks and Opportunities |
| Climate-Related Opportunities Disclosure | N/A | DirectMetric | Climate-Related Risks and Opportunities |
| Board Oversight of Climate-related Risks and Opportunities | Governance-a | DirectMetric | Governance |
| Management's Role in Climate-related Risk Assessment | Governance-b | DirectMetric | Governance |
| Climate-related Risks and Opportunities Identified | Strategy-a | DirectMetric | Strategy |
| Impact of Climate-related Risks on Business and Strategy | Strategy-b | DirectMetric | Strategy |
| Resilience of Strategy under Climate Scenarios | Strategy-c | DirectMetric | Strategy |
| Processes for Identifying Climate-related Risks | RiskManagement-a | DirectMetric | Risk Management |
| Processes for Managing Climate-related Risks | RiskManagement-b | DirectMetric | Risk Management |
| Integration of Climate Risk Processes into Overall Risk Management | RiskManagement-c | DirectMetric | Risk Management |
| Greenhouse Gas Emissions (Scope 1, 2, and 3) | MetricsTargets-b | DirectMetric | Metrics and Targets |
| Board Oversight of Climate-Related Issues | TCFD-Gov-a | DirectMetric | Governance |
| Management's Role in Climate-Related Issues | TCFD-Gov-b | DirectMetric | Governance |
| Climate-related Risks and Opportunities Description | TCFD-Strategy-a | DirectMetric | Strategy |
| Time Horizons Definition | TCFD-Strategy-a1 | DirectMetric | Strategy |
| Scenario Analysis Description | TCFD-Strategy-c | DirectMetric | Strategy |
| Processes for Identifying and Assessing Climate-related Risks | TCFD-RiskMgmt-a | DirectMetric | Risk Management |
| Processes for Managing Climate-related Risks | TCFD-RiskMgmt-b | DirectMetric | Risk Management |
| Climate-Related Risk Management Integration | TCFD-RM-c | DirectMetric | Risk Management |
| Key Climate-Related Metrics | TCFD-MT-a | DirectMetric | Metrics and Targets |
| Total GHG Emissions (Scope 1, 2, and 3) | TCFD-MT-a.1 | CalculatedMetric | Metrics and Targets |
| Scope 1 GHG Emissions | N/A | InputMetric | N/A |
| Scope 2 GHG Emissions | N/A | InputMetric | N/A |
| Scope 3 GHG Emissions | N/A | InputMetric | N/A |
| GHG Efficiency Ratios | N/A | DirectMetric | Metrics and Targets |
| Key Performance Indicators for Target Assessment | TCFD-MT-b.2 | DirectMetric | Metrics and Targets |
| Scenario Analysis Assessment | TCFD-SA-1 | DirectMetric | Scenario Analysis and Climate-Related Issues |
| Scenarios Used Disclosure | TCFD-Figure8-Item1 | DirectMetric | Scenario Analysis and Climate-Related Issues |
| Critical Input Parameters and Assumptions | TCFD-Figure8-Item2 | CalculatedMetric | Scenario Analysis and Climate-Related Issues |
| Technology Responses and Timing Assumptions | N/A | InputMetric | N/A |
| Scenario Time Frames Disclosure | TCFD-Figure8-Item3 | DirectMetric | Scenario Analysis and Climate-Related Issues |
| Organizational Resiliency Assessment | TCFD-Figure8-Item4 | CalculatedMetric | Scenario Analysis and Climate-Related Issues |
| Climate-Related Risk Governance and Responsibility Structure | N/A | DirectMetric | N/A |
| Reputation Risk | N/A | DirectMetric | Climate-Related Risks, Opportunities, and Financial Impacts |
| Resilience Opportunity | N/A | DirectMetric | Climate-Related Opportunities |
| Revenue Impact from Climate Risks | N/A | DirectMetric | Climate-Related Risks, Opportunities, and Financial Impacts |

#### Models
| Label | Formula | Input Metrics |
|-------|---------|---------------|
| GHG Emissions Aggregation Model | Total GHG Emissions = Scope 1 + Scope 2 + Scope 3 (if appropriate) | Scope 1 GHG Emissions, Scope 2 GHG Emissions, Scope 3 GHG Emissions |
| Scenario Input Parameters Analysis Model | f(TechnologyResponses, RegionalDifferences, KeySensitivities) | Technology Responses and Timing Assumptions |
| Strategic Resiliency Assessment Model | f(StrategicPerformance, ValueChainImplications, CapitalAllocation, R&DFocus, FinancialImplications) | N/A |

#### Reporting Framework
| Label | Version | Year |
|-------|---------|------|
| TCFD | Final Report | 2017 |

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
