# Datasheet: [Dataset Name]

> **Attribution**: This template is adapted from Gebru et al. (2021), "Datasheets for Datasets" ([arXiv:1803.09010](https://arxiv.org/abs/1803.09010)), and aligns with the MLCommons Data and Documentation (DAD) framework. It provides structured documentation for dataset creators and users to assess fitness for purpose, identify limitations, and mitigate risks.

---

## Motivation

### For What Purpose Was the Dataset Created?
[Describe the specific task or research question the dataset was designed to address. Be specific: "predicting loan default risk for applicants with thin credit files" rather than "machine learning."]

### Who Created the Dataset and On Behalf of Which Entity?
| Field | Description |
|-------|-------------|
| **Creator(s)** | [Names or team] |
| **Affiliation** | [Organization] |
| **Contact** | [heather@fireworks.ai] |
| **Funding Source** | [Grant, internal budget, etc.] |
| **Creation Date** | [YYYY-MM-DD] |

### Who Funded the Creation of the Dataset?
[Describe funding sources and any potential conflicts of interest.]

---

## Composition

### What Do the Instances That Comprise the Dataset Represent?
[Each instance is a... (e.g., single hospital admission, one consumer's credit history over 24 months, one image with associated labels)]

### How Many Instances Are There In Total?
| Split | Count | Notes |
|-------|-------|-------|
| **Training** | [N] | |
| **Validation** | [N] | |
| **Test** | [N] | |
| **Total** | [N] | |

### Does the Dataset Contain All Possible Instances or Is It a Sample?
- [ ] Complete population
- [ ] Sample (describe sampling method: random stratified, convenience, etc.)

### What Data Does Each Instance Consist Of?
[Raw data, features, labels, metadata. List feature names and types.]

| Feature Name | Type | Description | Sensitive? |
|--------------|------|-------------|------------|
| | | | |

### Is There a Label or Target Associated With Each Instance?
- [ ] Yes (describe labels, class distribution, and labeling methodology)
- [ ] No (describe what predictions or analyses are expected)

### Is Any Information Missing From Individual Instances?
[Missing data patterns, missingness mechanisms (MCAR, MAR, MNAR), and handling during collection.]

### Are Relationships Between Individual Instances Made Explicit?
[Temporal sequences, hierarchical structures, linked records (e.g., multiple admissions per patient).]

### Are There Recommended Data Splits?
[Fixed splits, cross-validation strategy, temporal splits for time-series data.]

### Are There Any Errors, Sources of Noise, or Redundancies?
[Known labeling errors, OCR errors, transcription noise, duplicate records.]

### Is the Dataset Self-Contained, or Does It Link to External Resources?
- [ ] Self-contained
- [ ] Links to external data (describe dependencies and stability)

### Does the Dataset Contain Data That Might Be Considered Confidential?
- [ ] No
- [ ] Yes (describe: medical records, financial data, legal proceedings, etc.)

### Does the Dataset Contain Data That Might Be Considered Sensitive?
| Attribute | Present? | Description |
|-----------|----------|-------------|
| Race / ethnicity | [ ] | |
| Gender / sex | [ ] | |
| Age | [ ] | |
| Religion | [ ] | |
| Sexual orientation | [ ] | |
| Political opinions | [ ] | |
| Health / medical | [ ] | |
| Genetic / biometric | [ ] | |
| Financial | [ ] | |
| Criminal history | [ ] | |
| Geographic location | [ ] | |
| Other: | [ ] | |

---

## Collection Process

### How Was the Data Associated With Each Instance Acquired?
[Direct measurement, surveys, web scraping, API extraction, crowdsourcing, synthetic generation, etc.]

### What Mechanisms or Procedures Were Used to Collect the Data?
[Instrumentation, survey instruments, scraping tools, data-sharing agreements.]

### If the Dataset Is a Sample From a Larger Set, What Was the Sampling Strategy?
[Random, stratified, convenience, snowball, etc. Describe inclusion/exclusion criteria.]

### Who Was Involved in the Data Collection Process?
[Research assistants, contractors, volunteers, automated systems. Compensation and training.]

### Over What Timeframe Was the Data Collected?
| Start Date | End Date | Frequency |
|------------|----------|-----------|
| | | [One-time / daily / monthly / continuous] |

### How Was the Data Collection Funded?
[Same as creation funding, or distinct source.]

### Was Ethical Review Conducted?
- [ ] IRB / Ethics board approval obtained (reference number: ___)
- [ ] Institutional review waived (explain)
- [ ] No formal review (explain rationale)

### Were Individuals Notified That Their Data Was Being Collected?
- [ ] Yes (describe mechanism: informed consent, terms of service, public notice)
- [ ] No (explain why and assess ethical implications)

### Did Individuals Consent to the Collection and Use of Their Data?
- [ ] Explicit informed consent
- [ ] Implied consent (e.g., terms of service, public posting)
- [ ] No consent (explain justification)

### If Consent Was Obtained, Were the Consenting Individuals Provided With a Mechanism to Revoke Their Consent?
- [ ] Yes (describe mechanism)
- [ ] No
- [ ] N/A

### Has an Analysis of the Potential Impact of the Dataset and Its Use on Data Subjects Been Conducted?
[Privacy risk assessment, harm analysis, vulnerability to re-identification.]

---

## Preprocessing, Cleaning, and Labeling

### Was Any Preprocessing/Cleaning/Labeling of the Data Done?
- [ ] Raw data (no preprocessing)
- [ ] Preprocessing applied (describe below)

| Step | Description | Tool / Method | Parameters |
|------|-------------|---------------|------------|
| Missing value imputation | | | |
| Outlier removal | | | |
| Normalization / scaling | | | |
| Encoding | | | |
| Deduplication | | | |
| PII scrubbing / anonymization | | | |
| Text cleaning | | | |
| Image preprocessing | | | |
| Audio preprocessing | | | |
| Other: | | | |

### Was the "Raw" Data Saved In Addition to the Preprocessed/Cleaned/Labeled Data?
- [ ] Yes (location: ___)
- [ ] No

### Is the Software Used to Preprocess/Clean/Label the Data Available?
- [ ] Yes (link to code / pipeline)
- [ ] No
- [ ] Partially (describe)

---

## Uses

### Has the Dataset Been Used For Any Tasks Already?
[Published research, production systems, benchmarks, competitions.]

### Is There a Repository That Links To Any or All Papers or Systems That Use the Dataset?
[Google Scholar alerts, Papers With Code, Kaggle, etc.]

### What (Other) Tasks Could the Dataset Be Used For?
[Secondary uses, transfer learning, augmentation, synthetic data generation.]

### Is There Anything About the Composition of the Dataset or the Way It Was Collected and Preprocessed/Cleaned/Labeled That Might Impact Future Uses?
[Temporal cutoff, geographic bias, demographic skew, labeling errors that propagate.]

### Are There Tasks for Which the Dataset Should Not Be Used?
[High-stakes uses where known limitations create unacceptable risk; uses that violate creator intent or data subject rights.]

---

## Distribution

### Will the Dataset Be Distributed to Third Parties Outside of the Entity That Created It?
- [ ] Yes (describe terms: open access, license, commercial, research-only)
- [ ] No (internal use only)
- [ ] Conditional (describe)

### How Will the Dataset Be Distributed?
[Download link, API, data marketplace, physical media.]

### When Will the Dataset Be Distributed?
[Already available, embargoed, rolling release, etc.]

### Will the Dataset Be Distributed Under a Copyright or Other Intellectual Property (IP) License, and/or Under Applicable Terms of Use (ToU)?
| License | Link |
|---------|------|
| [e.g., CC BY 4.0, CDLA, Proprietary] | |

### Have Any Third Parties Imposed IP-Based or Other Restrictions On the Data?
[Scraped data subject to platform ToS; licensed data with usage restrictions; patented features.]

### Do Any Export Controls or Other Regulatory Restrictions Apply To the Dataset or To Individual Instances?
[ITAR, EAR, HIPAA, GDPR cross-border transfer, etc.]

---

## Maintenance

### Who Is Supporting/Hosting/Maintaining the Dataset?
| Role | Contact |
|------|---------|
| Host | |
| Maintainer | |
| Curator | |

### How Can the Owner/Curator/Manager of the Dataset Be Contacted?
[Email, issue tracker, mailing list.]

### Is There an Erratum?
[Known errors, corrections, updates. Link to changelog.]

### Will the Dataset Be Updated?
- [ ] Yes (frequency: ___; versioning strategy: ___)
- [ ] No
- [ ] End-of-life date: ___

### If Others Want to Extend/Augment/Build On/Contribute to the Dataset, Is There a Mechanism For Them To Do So?
[Contribution guidelines, data improvement bounties, community curation.]

---

## Known Biases and Limitations

### Demographic Representation
| Group | Dataset Proportion | Target Population Proportion | Gap |
|-------|-------------------|------------------------------|-----|
| Gender | | | |
| Age bracket | | | |
| Race / ethnicity | | | |
| Geography | | | |
| Socioeconomic | | | |
| Language | | | |

### Temporal Limitations
[Data collected during specific period; may not reflect post-pandemic, post-policy-change, or seasonal conditions.]

### Selection Bias
[Who is over/underrepresented due to collection method? E.g., web forum users skew young and male; clinical trial participants skew healthier.]

### Measurement Bias
[Instruments or labeling processes that systematically mismeasure certain groups.]

### Labeling Limitations
[Inter-annotator agreement, label noise, incomplete labels, proxy labels that don't capture the true construct.]

### Generalization Limits
[Domain shift, geographic transfer, temporal decay, population mismatch between training and deployment contexts.]

---

## Ethical Considerations

### Potential for Harmful Use
[Surveillance, discrimination, misinformation generation, deepfake creation, automated weapon targeting.]

### Potential for Dual Use
[Legitimate research tool that could be repurposed for harm.]

### Environmental Impact
[Carbon footprint of collection, storage, and intended model training.]

### Labor and Compensation
[Data annotator wages, working conditions, crowdsourcing platform fairness.]

### Community Engagement
[Were affected communities consulted during dataset design? Was there a mechanism for feedback or redress?]

---

## Version History

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 1.0.0 | | Initial release | |

---

*For questions or corrections, contact [heather@fireworks.ai](mailto:heather@fireworks.ai) or open an issue at [https://github.com/heathriel/ai-safety-resources](https://github.com/heathriel/ai-safety-resources)*
