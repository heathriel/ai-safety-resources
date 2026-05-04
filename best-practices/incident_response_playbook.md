# AI Incident Response Playbook

A structured guide for detecting, containing, investigating, and recovering from AI system incidents. Designed for cross-functional teams including engineering, legal, communications, and executive leadership.

> **Attribution**: Adapted from NIST AI RMF Govern and Manage functions, NIST SP 800-61 (Computer Security Incident Handling Guide), and the AI Incident Database (AIID) taxonomy.

---

## Incident Definition

An **AI incident** is any event where an AI system causes, contributes to, or fails to prevent:
- Harm to individuals (physical, psychological, economic, or dignitary)
- Violations of law, regulation, or organizational policy
- Significant degradation of system performance or trustworthiness
- Exposure of sensitive data or intellectual property

---

## Response Team Roles

| Role | Responsibility | Typical Owner |
|------|---------------|---------------|
| **Incident Commander** | Coordinates response, makes go/no-go decisions, communicates to leadership | VP Engineering or CTO |
| **Technical Lead** | Leads containment, forensics, and remediation | ML/AI Platform Lead |
| **Legal / Compliance** | Assesses regulatory obligations, manages disclosure timelines, preserves evidence | General Counsel or DPO |
| **Communications** | Manages internal and external messaging, stakeholder notifications | Comms Director or PR |
| **Ethics / Safety Officer** | Evaluates harm scope, advises on ethical obligations, liaises with affected communities | Chief Ethics Officer or external advisor |
| **Customer Success** | Notifies affected customers, coordinates remediation offers, tracks sentiment | VP Customer Success |

---

## Severity Levels

| Level | Criteria | Response Time | Escalation |
|-------|----------|---------------|------------|
| **SEV-1 Critical** | Active harm to life/safety; ongoing data breach; regulatory emergency | Immediate (< 15 min) | CEO + Board notified within 1 hour |
| **SEV-2 High** | Significant bias affecting protected groups; major hallucination in high-stakes domain; potential legal liability | < 1 hour | C-suite notified within 4 hours |
| **SEV-3 Medium** | Isolated fairness degradation; model drift crossing thresholds; customer-visible errors | < 4 hours | VP-level notified within 24 hours |
| **SEV-4 Low** | Documentation gaps; minor monitoring alerts; near-misses requiring preventive action | < 24 hours | Team lead notified; tracked for patterns |

---

## Response Phases

### Phase 1: Detection & Triage (0–30 minutes)

**Goal:** Determine if an incident exists, assign severity, and activate the response team.

#### Detection Triggers
| Source | Trigger | Auto-Action |
|--------|---------|-------------|
| Monitoring dashboards | Disparate impact > threshold; accuracy drop > X% | Page on-call engineer |
| Automated safety filters | Toxicity rate spike; PII leakage detected | Block outputs + alert |
| Customer reports | Complaint about biased output; incorrect medical/legal advice | Log ticket + flag for safety review |
| Media / social media | Viral harmful output screenshot | Alert comms + legal immediately |
| Internal testing | Red team discovers new jailbreak; extraction attack succeeds | Create incident ticket |
| Regulatory inquiry | Government agency requests information | Immediate legal + exec alert |

#### Triage Checklist
- [ ] Log incident timestamp, detector, and initial description
- [ ] Assign preliminary severity (SEV-1 through SEV-4)
- [ ] Page Incident Commander for SEV-1/SEV-2
- [ ] Create war room (Slack channel, Zoom bridge, or physical room)
- [ ] Preserve evidence: snapshot model version, inputs, outputs, system logs
- [ ] Identify if incident is **ongoing** (active harm) or **retrospective** (past event)

---

### Phase 2: Containment (30 minutes–2 hours)

**Goal:** Stop ongoing harm and prevent expansion.

#### Immediate Actions by Severity

**SEV-1 (Critical):**
- [ ] **Kill switch**: Disable model endpoint or route traffic to safe fallback
- [ ] Block identified attack pattern or toxic prompt signature at edge
- [ ] If data breach: activate data breach response protocol; preserve access logs
- [ ] Notify legal for potential regulatory breach (GDPR 72-hour clock starts)
- [ ] Draft holding statement for stakeholders

**SEV-2 (High):**
- [ ] Throttle or shadow-ban affected model version
- [ ] Route affected user segment to human review or previous stable model
- [ ] Deploy emergency output filter or prompt hardening patch
- [ ] Notify affected customers if error rate or harm is user-facing

**SEV-3 (Medium):**
- [ ] Flag for next scheduled deployment rollback
- [ ] Increase monitoring granularity for affected model/feature
- [ ] Queue for post-mortem in next sprint

**SEV-4 (Low):**
- [ ] Create ticket in backlog
- [ ] Review at next team standup

---

### Phase 3: Investigation (2 hours–48 hours)

**Goal:** Determine root cause, scope of impact, and affected parties.

#### Technical Investigation
| Question | Method | Owner |
|----------|--------|-------|
| Which model version? | Check deployment logs, version pins, canary IDs | Technical Lead |
| What input triggered it? | Query access logs for prompt patterns; reproduce | ML Engineer |
| Was it a data issue? | Inspect training data for contamination, drift, or poisoned samples | Data Engineer |
| Was it a model issue? | Evaluate model on held-out safety test set; check for regression | ML Engineer |
| Was it a system issue? | Review feature store, upstream service, or prompt template changes | Platform Engineer |
| Was it adversarial? | Analyze for prompt injection patterns, membership inference, or extraction attempts | Security Engineer |

#### Impact Assessment
| Dimension | Assessment |
|-----------|------------|
| **Affected users** | Count unique users exposed to harmful outputs; identify protected groups disproportionately affected |
| **Data exposure** | Determine if PII, trade secrets, or confidential training data was leaked |
| **Financial impact** | Estimate direct costs (refunds, legal fees) and indirect costs (churn, reputation) |
| **Regulatory exposure** | Assess GDPR, CCPA, BIPA, EU AI Act, sector-specific (HIPAA, FCRA) obligations |
| **Media / public exposure** | Track social media mentions, press inquiries, viral spread |

#### Evidence Preservation
- [ ] Export system logs, model versions, and feature stores to immutable storage
- [ ] Document chain of custody for all evidence
- [ ] Freeze access to incident-related systems for non-response-team members
- [ ] Legal hold: prevent deletion of emails, Slack messages, or meeting recordings

---

### Phase 4: Communication (Parallel with all phases)

**Goal:** Inform the right stakeholders at the right time with the right level of detail.

#### Internal Communication Timeline
| Audience | Timing | Channel | Content |
|----------|--------|---------|---------|
| Response team | Immediate | War room channel | Incident summary, severity, roles |
| Executive team | SEV-1: <1 hr; SEV-2: <4 hr; SEV-3: <24 hr | Email or meeting | Business impact, regulatory exposure, public risk |
| All employees | Before external disclosure | All-hands or memo | What happened, what we're doing, what not to say externally |
| Board | SEV-1: <4 hr; SEV-2: <24 hr | Direct call or board briefing | Strategic implications, liability, governance gaps |

#### External Communication Timeline
| Audience | Timing | Channel | Content |
|----------|--------|---------|---------|
| Affected customers | Before public disclosure if possible | Direct email or in-app notice | What happened, impact on them, remediation steps |
| Regulators | Per regulatory deadlines (GDPR: 72 hr; others vary) | Formal filing or call | Factual summary, scope, containment, planned remediation |
| Media / public | Controlled; never before customer notification | Blog post, press release | What happened, what we know, what we're doing, what we don't know yet |
| Researchers / community | After containment, before full resolution | Security advisory or blog | Technical details for peer learning; responsible disclosure if applicable |

#### Communication Templates

**Holding Statement (first hour):**
> "We are investigating reports of [brief description]. We take this seriously and have activated our incident response team. We will share more information as our investigation proceeds. For urgent concerns, contact [incident@company.com]."

**Customer Notification:**
> "On [date], our AI system [brief description of what happened]. This affected [scope]. We have [containment action]. We are [remediation action]. If you experienced [specific harm], please [contact / compensation process]."

---

### Phase 5: Remediation (24 hours–2 weeks)

**Goal:** Fix the root cause, restore service, and prevent recurrence.

#### Remediation Options
| Root Cause | Remediation | Verification |
|------------|-------------|--------------|
| Training data bias | Retrain with balanced data; apply reweighing or adversarial debiasing | Re-run fairness metrics; A/B test with holdout |
| Prompt injection vulnerability | Deploy input filtering, prompt hardening, output moderation | Red team retests with same + novel attacks |
| Hallucination in high-stakes domain | Switch to RAG with citations; add human review gate; reduce temperature | Evaluate on factual accuracy benchmark |
| Model drift | Retrain on recent data; implement automated retraining pipeline | Monitor accuracy/fairness metrics post-deployment |
| Security breach (extraction, inversion) | Rotate API keys; add rate limiting; deploy query anomaly detection | Penetration test by external firm |
| Supply chain compromise | Rebuild from clean base image; verify checksums; scan dependencies | SBOM audit; dependency vulnerability scan |

#### Rollback vs. Forward-Fix Decision Matrix
| Factor | Rollback | Forward-Fix |
|--------|----------|-------------|
| Time to safe state | Hours | Days |
| Data loss risk | None | Risk of reintroducing in new deployment |
| Confidence in fix | Low | High |
| Customer disruption | Immediate (revert to older model) | Delayed (stay on buggy version until patch) |
| Regulatory pressure | High (need immediate containment) | Lower (if harm is contained) |

---

### Phase 6: Post-Incident Review (1–2 weeks after closure)

**Goal:** Learn from the incident, improve systems, and update playbooks.

#### Post-Mortem Template

**Metadata**
| Field | Value |
|-------|-------|
| Incident ID | |
| Date / Time (start) | |
| Date / Time (resolved) | |
| Duration | |
| Severity | |
| Detection source | |
| Response team members | |

**Summary**
- What happened? (2–3 sentences)
- What was the impact? (users, data, financial, regulatory)
- How was it detected?
- How was it contained?

**Timeline**
| Time | Event | Owner |
|------|-------|-------|
| T+0 | Detection / alert | |
| T+15 min | Triage complete; SEV-X assigned | |
| T+1 hr | Containment action deployed | |
| ... | ... | |

**Root Cause Analysis (5 Whys)**
1. Why did the incident occur? →
2. Why did [answer 1] happen? →
3. Why did [answer 2] happen? →
4. Why did [answer 3] happen? →
5. Why did [answer 4] happen? →

**What Went Well**
- [ ]
- [ ]

**What Went Poorly**
- [ ]
- [ ]

**Action Items**
| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| | | | |

**Playbook Updates**
- [ ] Monitoring alert thresholds adjusted
- [ ] New detection rule added
- [ ] Runbook updated with lessons
- [ ] Training material updated for team

---

## Regulatory Notification Reference

| Regulation | Trigger | Deadline | To Whom |
|------------|---------|----------|---------|
| **GDPR (EU)** | Personal data breach | 72 hours to supervisory authority | Lead DPA |
| **GDPR (EU)** | High risk to rights/freedom | Without delay to data subjects | Affected individuals |
| **CCPA/CPRA (California)** | Unauthorized access to personal info | Without unreasonable delay | California AG + affected consumers |
| **BIPA (Illinois)** | Biometric data breach | Without unreasonable delay | Affected individuals |
| **EU AI Act (High-Risk)** | Serious incident or malfunction | Without delay to market surveillance authority | National AI regulator |
| **HIPAA (US Healthcare)** | PHI breach > 500 individuals | 60 days to HHS; 60 days to individuals | HHS OCR + affected individuals |
| **Sector-specific** | Varies by industry | Varies | Varies |

---

## Appendix: Incident Severity Decision Tree

```
Is there active harm to life or safety?
├── YES → SEV-1 (kill switch, exec alert, legal, 72-hr regulatory clock)
└── NO → Is there significant harm to protected groups or potential legal liability?
    ├── YES → SEV-2 (throttle, legal review, customer notification)
    └── NO → Is there user-facing degradation or drift crossing thresholds?
        ├── YES → SEV-3 (rollback plan, increased monitoring)
        └── NO → SEV-4 (backlog, preventive action)
```

---

*For questions or corrections, contact [heather@fireworks.ai](mailto:heather@fireworks.ai) or open an issue at [https://github.com/heathriel/ai-safety-resources](https://github.com/heathriel/ai-safety-resources)*
