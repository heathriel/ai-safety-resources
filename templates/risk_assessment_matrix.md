# AI Risk Assessment Matrix

A structured template for identifying, scoring, and mitigating risks throughout the AI system lifecycle. Aligned with NIST AI Risk Management Framework (AI RMF) and ISO/IEC 23894:2023.

> **Attribution**: Adapted from the NIST AI Risk Management Framework (AI RMF 1.0) and ISO/IEC 23894:2023 (AI Risk Management).

---

## Risk Scoring Criteria

### Likelihood (1–5)
| Score | Description |
|-------|-------------|
| 1 | Rare — No known occurrences in similar systems; requires multiple unlikely conditions |
| 2 | Unlikely — Occasional occurrences in industry; requires specific conditions |
| 3 | Possible — Known to occur in similar systems under normal operation |
| 4 | Likely — Occurs frequently in similar systems; expected without mitigation |
| 5 | Almost Certain — Inevitable without active mitigation measures |

### Impact (1–5)
| Score | Description |
|-------|-------------|
| 1 | Negligible — No measurable harm; internal process inconvenience only |
| 2 | Minor — Limited reputational or financial impact; recoverable within days |
| 3 | Moderate — Significant financial loss, regulatory scrutiny, or user harm affecting individuals |
| 4 | Major — Severe harm to individuals (physical, psychological, economic); major regulatory breach; lasting reputational damage |
| 5 | Catastrophic — Loss of life; systemic societal harm; criminal liability; organization-threatening |

### Risk Score = Likelihood × Impact
| Score | Priority | Action |
|-------|----------|--------|
| 1–4 | Low | Monitor; accept or document |
| 5–9 | Medium | Mitigate within defined timeline; assign owner |
| 10–19 | High | Immediate mitigation required; escalate to leadership |
| 20–25 | Critical | Halt deployment/operation until resolved; emergency response |

---

## Risk Register Template

### Project Information
| Field | Value |
|-------|-------|
| System Name | |
| Version / Date | |
| Risk Assessment Owner | |
| Assessment Date | |
| Next Review Date | |

### Risk Entries

#### Risk ID: [R-001]
| Field | Description |
|-------|-------------|
| **Risk Category** | [e.g., Bias/Fairness, Privacy, Security, Reliability, Transparency, Environmental] |
| **Risk Statement** | [What could go wrong? Be specific.] |
| **Threat Source** | [e.g., Adversarial users, training data bias, system design flaw, regulatory change] |
| **Vulnerability** | [e.g., No output filtering, unrepresentative training data, black-box model] |
| **Affected Stakeholders** | [e.g., End users, marginalized groups, organization, regulators] |
| **Likelihood** | [1–5] |
| **Impact** | [1–5] |
| **Risk Score** | [L × I] |
| **Priority** | [Low / Medium / High / Critical] |
| **Mitigation Strategy** | [Specific action to reduce likelihood or impact] |
| **Residual Likelihood** | [1–5 after mitigation] |
| **Residual Impact** | [1–5 after mitigation] |
| **Residual Risk Score** | [New L × I] |
| **Mitigation Owner** | [Name / Team] |
| **Target Date** | [YYYY-MM-DD] |
| **Status** | [Open / In Progress / Mitigated / Accepted / Transferred] |
| **Monitoring Method** | [How will you detect if this risk materializes?] |

---

## Common AI Risk Categories

### Bias and Fairness
| Risk | Example |
|------|---------|
| Training data underrepresents protected group | Medical AI trained primarily on light-skinned patients |
| Proxy discrimination | ZIP code proxies for race in lending decisions |
| Temporal drift | Consumer behavior changes post-pandemic; model becomes biased |
| Annotation bias | Labelers systematically mislabel certain dialects as "toxic" |

### Privacy and Data Protection
| Risk | Example |
|------|---------|
| Training data memorization | LLM outputs exact PII from training corpus |
| Membership inference | Attacker determines if a specific individual was in training data |
| Model inversion | Reconstruction of training images from model weights |
| Unauthorized data sharing | Third-party API retains and resells user queries |

### Security
| Risk | Example |
|------|---------|
| Prompt injection | RAG system tricked into revealing internal documents |
| Model extraction | API queries used to train a competing model |
| Supply chain compromise | Compromised pre-trained weights or LoRA adapter |
| Adversarial evasion | Slight image perturbation bypasses safety classifier |

### Reliability and Safety
| Risk | Example |
|------|---------|
| Hallucination | Legal AI cites non-existent case law |
| Distribution shift | Autonomous vehicle model fails in unseen weather conditions |
| Cascading failure | Automated trading algorithm triggers market flash crash |
| Uncalibrated confidence | Model reports 99% confidence on wrong answer |

### Transparency and Accountability
| Risk | Example |
|------|---------|
| Black-box decision | Loan denied; applicant cannot understand or challenge reason |
| Audit gap | No logging of model versions, inputs, or outputs for regulatory review |
| Responsibility diffusion | Vendor blames customer training; customer blames vendor base model |
| Documentation debt | Model deployed without model card, data sheet, or evaluation results |

### Environmental and Societal
| Risk | Example |
|------|---------|
| Carbon footprint | Large model training emits tons of CO2 without offset |
| Job displacement | Automation deployed without transition support for affected workers |
| Amplification of misinformation | Recommendation algorithm promotes conspiracy content |
| Concentration of power | Capabilities monopolized by few organizations |

---

## NIST AI RMF Mapping

| AI RMF Function | Risk Category Alignment | Key Activities |
|-----------------|------------------------|--------------|
| **Govern** | All | Establish risk tolerance, policies, accountability structures |
| **Map** | Transparency, Bias | Identify context, stakeholders, and known risks |
| **Measure** | All | Quantify bias, accuracy, robustness, privacy metrics |
| **Manage** | Security, Reliability | Implement mitigations, monitor residual risks |

---

*For questions or corrections, contact [heather@fireworks.ai] or open an issue at [https://github.com/heathriel/ai-safety-resources](https://github.com/heathriel/ai-safety-resources)*
