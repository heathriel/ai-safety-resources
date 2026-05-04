# Model Card: [Model Name]

> **Attribution**: This template is adapted from the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) paper by Mitchell et al. (2019) and the [MLCommons Model Card Toolkit](https://github.com/mlcommons/model-card-toolkit). It aligns with NIST AI RMF transparency requirements and the EU AI Act's high-risk system documentation obligations.

---

## Model Details

| Field | Description |
|-------|-------------|
| **Organization** | [Name] |
| **Model Date** | [YYYY-MM-DD] |
| **Model Version** | [e.g., 1.2.0] |
| **Model Type** | [e.g., text classification, code generation, multimodal vision-language] |
| **Architecture** | [e.g., Transformer decoder-only, 8B parameters] |
| **License** | [e.g., Apache-2.0, Proprietary] |
| **Contact** | [heather@fireworks.ai] |
| **Intended Use Cases** | [List primary use cases] |
| **Out-of-Scope Uses** | [Explicitly state prohibited or high-risk uses] |

---

## Intended Use

### Primary Use Cases
- [e.g., Customer support triage, document summarization, code completion]

### Out-of-Scope Uses
- [e.g., Medical diagnosis without human oversight, criminal risk assessment, autonomous weapon systems]
- [e.g., Decisions affecting individual rights without human review]

---

## Factors

### Relevant Factors
| Factor | Description | Evaluation Approach |
|--------|-------------|---------------------|
| Demographics | Age, gender, race, geographic region | Stratified evaluation across subgroups |
| Domain | Medical, legal, financial, general | Per-domain benchmark testing |
| Language | English, Spanish, code (Python, JavaScript) | Multilingual evaluation suite |
| Modality | Text, image, audio | Per-modality accuracy metrics |

### Evaluation Factors
| Factor | Groups / Values | Why Relevant |
|--------|-----------------|--------------|
| [Gender] | [Male, Female, Non-binary] | [Potential for differential performance in pronoun resolution] |
| [Age group] | [<18, 18-65, >65] | [Different vocabulary and syntax patterns] |

---

## Metrics

### Performance Metrics
| Metric | Value | Threshold | Test Set |
|--------|-------|-----------|----------|
| Accuracy | [0.XX] | [≥0.XX] | [Benchmark name] |
| F1 Score | [0.XX] | [≥0.XX] | [Benchmark name] |
| Perplexity | [X.XX] | [≤X.XX] | [Perplexity benchmark] |
| BLEU / ROUGE | [0.XX] | [≥0.XX] | [Generation benchmark] |
| Latency (p99) | [Xms] | [≤Xms] | [Production traffic] |

### Fairness Metrics
| Metric | Overall | Subgroup A | Subgroup B | Threshold |
|--------|---------|------------|------------|-----------|
| Demographic Parity | [0.XX] | [0.XX] | [0.XX] | [≤0.05 diff] |
| Equalized Odds | [0.XX] | [0.XX] | [0.XX] | [≤0.05 diff] |
| Disparate Impact | [0.XX] | [0.XX] | [0.XX] | [≥0.80] |

---

## Evaluation Data

### Training Data
| Property | Description |
|----------|-------------|
| **Source** | [e.g., Common Crawl, internal documents, synthetic data] |
| **Size** | [e.g., 1.2T tokens, 50M documents] |
| **Date Range** | [e.g., 2020-01 to 2023-12] |
| **Preprocessing** | [e.g., PII scrubbing, deduplication, quality filtering] |
| **Known Limitations** | [e.g., Western-centric, English-heavy, temporal cutoff] |
| **Data Sheet** | [Link to full data documentation] |

### Evaluation Data
| Dataset | Purpose | Size | Split | Date |
|---------|---------|------|-------|------|
| [Benchmark A] | [General capability] | [N samples] | [Test] | [2024-01] |
| [Benchmark B] | [Safety / red teaming] | [N prompts] | [Test] | [2024-03] |
| [Internal eval] | [Domain-specific] | [N samples] | [Holdout] | [2024-06] |

---

## Ethical Considerations

| Consideration | Assessment | Mitigation |
|---------------|------------|------------|
| **Bias / Fairness** | [e.g., Potential gender bias in occupation associations] | [Reweighing during training, post-hoc threshold tuning] |
| **Privacy** | [e.g., Risk of memorizing PII from training data] | [Differential privacy (ε=X), PII scrubbing, extraction testing] |
| **Transparency** | [e.g., Black-box neural network] | [Published evaluation results, model card, API documentation] |
| **Accountability** | [e.g., Difficult to attribute harmful outputs] | [Audit logging, version pinning, human-in-the-loop for high-stakes] |
| **Environmental** | [e.g., Training consumed X MWh] | [Carbon offset purchase, efficiency optimizations for inference] |

---

## Caveats and Recommendations

- **Known Failure Modes**: [e.g., Hallucinates factual claims, struggles with arithmetic, biased on rare names]
- **Recommended Human Oversight**: [e.g., Required for medical, legal, financial decisions]
- **Retraining Cadence**: [e.g., Evaluate drift monthly, retrain quarterly]
- **Incident Reporting**: [e.g., Report harmful outputs to heather@fireworks.ai]

---

## Version History

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 1.0.0 | [Date] | Initial release | [Name] |
| 1.1.0 | [Date] | Added fairness metrics | [Name] |

---

*For questions or corrections, contact [heather@fireworks.ai] or open an issue at [https://github.com/heathriel/ai-safety-resources](https://github.com/heathriel/ai-safety-resources)*