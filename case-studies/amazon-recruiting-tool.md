# Case Study: Amazon AI Recruiting Tool

## Overview

| Attribute | Details |
|-----------|---------|
| **Organization** | Amazon |
| **Year(s)** | 2014–2017 (discontinued 2017) |
| **Domain** | Human Resources / Hiring |
| **System Type** | ML-based resume screening |
| **Impact** | Systematic gender discrimination in technical role recommendations |

## What Happened

Amazon developed an experimental AI system to review job applicants' resumes and assign star ratings (1–5) to identify top candidates. The system was trained on 10 years of Amazon's historical hiring data.

Because the tech industry and Amazon's own workforce were male-dominated, the training data reflected this imbalance. The model learned to penalize resumes containing the word "women's" (as in "women's chess club captain") and favored resumes with traditionally male-coded language.

Amazon's engineers attempted to make the model gender-neutral but could not guarantee it wouldn't find other proxies for gender (e.g., attending women's colleges, certain extracurricular activities).

## Root Causes

1. **Training Data Bias**: Historical hiring decisions encoded systemic gender bias in tech.
2. **Proxy Discrimination**: Even after explicit gender features were removed, the model inferred gender from correlated features (schools, activities, writing style).
3. **Feedback Loop**: The system's recommendations would have reinforced existing hiring patterns if deployed, amplifying bias over time.
4. **Inadequate Fairness Auditing**: No pre-deployment disparate impact analysis across gender subgroups.

## Principles Violated

| Principle | How It Was Violated |
|-----------|---------------------|
| **Fairness / Equitability** | System systematically downranked women applicants for technical roles. |
| **Transparency / Explainability** | Candidates were never informed an AI screened their resumes; no recourse mechanism existed. |
| **Accountability** | No clear owner for fairness outcomes; model developed by a team without HR ethics oversight. |

## Lessons Learned

1. **Historical data is not neutral**. Training on past decisions encodes past biases. Pre-training data audits for demographic representation are essential.
2. **Removing protected attributes is insufficient**. Models infer protected classes from proxy features. Use causal fairness methods or adversarial debiasing.
3. **Fairness metrics must be monitored continuously**. Disparate impact, demographic parity, and equalized odds should be evaluated before deployment and during operation.
4. **Human-in-the-loop is not a panacea**. If humans defer to AI recommendations (automation bias), the loop doesn't prevent harm. Design for meaningful human oversight, not rubber-stamp approval.

## Mitigation Strategies

| Strategy | Application |
|----------|-------------|
| Reweighing / resampling | Balance gender representation in training data |
| Adversarial debiasing | Train to prevent gender inference from non-protected features |
| Counterfactual fairness | Ensure decisions don't change if gender is flipped |
| Human review quotas | Randomly sample and manually review AI rejections by demographic group |
| Audit logging | Record all AI recommendations and human overrides for compliance review |

## References

- [Reuters: Amazon scraps secret AI recruiting tool that showed bias against women](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G/)
- [MIT Technology Review: Amazon's gender-biased algorithm is not alone](https://www.technologyreview.com/2018/10/16/139127/amazons-gender-biased-algorithm-is-not-alone/)
- [Barocas & Selbst: Big Data's Disparate Impact](https://www.californialawreview.org/print/big-datas-disparate-impact/) — foundational legal analysis of proxy discrimination
