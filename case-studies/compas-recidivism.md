# Case Study: COMPAS Recidivism Algorithm

## Overview

| Attribute | Details |
|-----------|---------|
| **Organization** | Northpointe (now Equivant) |
| **Year(s)** | 2016 (exposed); deployed since 1998 |
| **Domain** | Criminal Justice / Risk Assessment |
| **System Type** | Proprietary recidivism risk scoring algorithm |
| **Impact** | Racial bias in risk scores influenced bail, sentencing, and parole decisions across multiple U.S. jurisdictions |

## What Happened

ProPublica published a 2016 investigation analyzing the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) recidivism risk tool, used in courts across the United States to assess defendants' likelihood of reoffending.

The investigation found that COMPAS was **twice as likely to falsely flag Black defendants as high risk** (false positive rate ~44.9% for Black vs. ~23.5% for white defendants). Conversely, white defendants who did reoffend were more likely to be labeled low risk (false negative rate ~47.7% for white vs. ~28.0% for Black defendants).

The COMPAS algorithm is proprietary — neither defendants nor their attorneys can inspect how scores are calculated. Northpointe disputed ProPublica's analysis, arguing that the algorithm satisfied "calibration" (equal prediction accuracy across groups), which is mathematically incompatible with equal false positive rates when base rates differ.

This sparked a fundamental debate in algorithmic fairness: **different mathematical definitions of "fairness" are mutually exclusive**, and choosing one implicitly prioritizes who bears the burden of errors.

## Root Causes

1. **Black-Box Proprietary Algorithm**: Neither the public nor independent researchers could audit the model's logic, features, or training process.
2. **Feature Bias**: Risk factors included questions about family criminal history, neighborhood, and employment — all correlated with race due to structural inequities.
3. **Mathematical Fairness Tradeoffs**: Satisfying calibration (equal prediction accuracy across groups) inherently produces unequal false positive/negative rates when base rates differ. The developer chose one definition of fairness without disclosing or justifying the choice.
4. **Lack of Human Override**: Judges often treated COMPAS scores as deterministic rather than advisory, with insufficient training on the tool's limitations.

## Principles Violated

| Principle | How It Was Violated |
|-----------|---------------------|
| **Fairness** | Disparate false positive rates meant Black defendants faced higher risk of unjust pretrial detention and harsher sentencing. |
| **Transparency / Explainability** | Proprietary algorithm with no public documentation of features, weights, or training data. Defendants could not challenge specific score components. |
| **Accountability** | No clear party responsible for auditing or correcting bias; vendor and courts deflected responsibility to each other. |
| **Human Oversight** | Judges lacked training on the algorithm's limitations and treated scores as objective truth. |

## Lessons Learned

1. **"Fairness" is not a single metric**. Calibration, demographic parity, equalized odds, and individual fairness can conflict. Developers must disclose which definition they used and why — and acknowledge who bears the cost of errors under that definition.
2. **Proprietary algorithms in high-stakes domains require algorithmic auditing rights**. Courts and defendants need access to independent audits, even if the full source code remains proprietary.
3. **Features encode structural inequity**. Even "race-blind" features (employment history, neighborhood, family background) can proxy for race. Feature selection must include sociological review, not just statistical optimization.
4. **Risk scores need context, not just numbers**. Judges need training on base rates, confidence intervals, and the distinction between group-level predictions and individual assessments.

## The Fairness Tradeoff (Technical Deep-Dive)

When base rates differ across groups (here, historical re-arrest rates), three common fairness criteria become mathematically incompatible:

| Criterion | COMPAS Status | Impact |
|-----------|---------------|--------|
| **Calibration** | Satisfied | Equal prediction accuracy across groups |
| **Equalized Odds** | Violated | Unequal false positive and false negative rates |
| **Demographic Parity** | Violated | Different proportion of high-risk labels across groups |

Northpointe optimized for calibration. ProPublica highlighted the unequal false positive rates. Both are correct — they measure different, incompatible fairness properties.

**Implication**: There is no mathematically neutral choice. The developer's fairness optimization is a value judgment about who should bear the burden of algorithmic error.

## Mitigation Strategies

| Strategy | Application |
|----------|-------------|
| Algorithmic auditing rights | Mandate independent third-party audits for all high-stakes risk assessment tools |
| Feature transparency | Publish all input features and their correlations with protected attributes |
| Multi-metric fairness reporting | Report calibration, equalized odds, demographic parity, and individual fairness; explain tradeoffs |
| Human-in-the-loop training | Train judges on base rates, confidence intervals, and the advisory nature of risk scores |
| Regular bias audits | Continuous monitoring of false positive/negative rates by race, gender, and age |
| Contestability | Allow defendants to request feature review and provide counter-evidence |

## References

- [ProPublica: Machine Bias — Risk Assessments in Criminal Sentencing](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [Northpointe/Equivant: Response to ProPublica](https://www.equivant.com/response-to-propublica/)
- [Chouldechova: Fair Prediction with Disparate Impact (2017)](https://arxiv.org/abs/1610.07524) — mathematical proof of fairness incompatibility
- [Kleinberg et al.: Inherent Trade-Offs in the Fair Determination of Risk Scores (2017)](https://arxiv.org/abs/1609.05807)
- [Dressel & Farid: The accuracy, fairness, and limits of predicting recidivism (2018)](https://www.science.org/doi/10.1126/sciadv.aao5580)
