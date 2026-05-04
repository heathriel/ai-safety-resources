# Case Study: Air Canada Chatbot

## Overview

| Attribute | Details |
|-----------|---------|
| **Organization** | Air Canada |
| **Year(s)** | 2022–2024 |
| **Domain** | Customer Service / Travel |
| **System Type** | LLM-based customer support chatbot |
| **Impact** | Fabricated bereavement fare policy; airline held legally liable for chatbot hallucination |

## What Happened

In 2022, Air Canada deployed a customer service chatbot on its website to handle common inquiries about bookings, policies, and fares. A passenger named Jake Moffatt asked the chatbot about Air Canada's bereavement fare policy after his grandmother died.

The chatbot incorrectly stated that Air Canada offered retroactive bereavement fares — passengers could book a full-fare flight and apply for a partial refund after travel by submitting documentation. This was false. Air Canada's actual policy required bereavement fares to be booked *before* travel.

Moffatt relied on the chatbot's information, booked full-fare tickets, and later submitted a refund request. Air Canada denied the refund, claiming the chatbot had provided incorrect information. Moffatt took the case to small claims court.

In February 2024, a Canadian civil tribunal ruled against Air Canada, holding the airline liable for the chatbot's misinformation. The tribunal stated that Air Canada was responsible for all information on its website, regardless of whether it came from a chatbot, static page, or human agent.

## Root Causes

1. **Hallucination / Confabulation**: The LLM generated a plausible-sounding but false policy that did not exist.
2. **No Source Grounding**: The chatbot was not constrained to retrieve information from an authoritative policy database (RAG was not used or failed).
3. **No Human Escalation for High-Stakes Queries**: Bereavement fares involve significant money and emotional distress — no automatic handoff to a human agent occurred.
4. **Corporate Liability Gap**: Air Canada initially argued it was not responsible for the chatbot's errors, treating it as a third-party entity rather than company communication.

## Principles Violated

| Principle | How It Was Violated |
|-----------|---------------------|
| **Reliability / Accuracy** | Generated false policy information that directly caused financial harm. |
| **Accountability** | Air Canada attempted to disclaim responsibility for its own chatbot's outputs. |
| **Transparency** | Users were not warned that chatbot information might be inaccurate or that they should verify with official policy documents. |

## Lessons Learned

1. **LLMs are not knowledge bases**. They generate plausible text, not verified facts. For policy, pricing, medical, or legal information, LLMs must be grounded in authoritative sources (RAG, structured databases) with citation links.
2. **Your chatbot is your employee**. Courts and customers will hold your organization liable for what your AI says. You cannot outsource accountability to "the algorithm."
3. **High-stakes queries need guardrails**. Financial, legal, medical, or emotionally sensitive inquiries should trigger human review or authoritative source verification before a response is given.
4. **Confidence calibration matters**. If an LLM is unsure, it should say so — or escalate — rather than confabitate a confident-sounding answer.

## Mitigation Strategies

| Strategy | Application |
|----------|-------------|
| RAG with citations | Ground responses in official policy documents; provide links to source text |
| Structured output for known policies | Use deterministic lookup for standard policies; reserve LLM for clarification of edge cases |
| Confidence thresholds | Low-confidence responses trigger human handoff rather than being served |
| Disclaimers | Clear warnings that chatbot information is not official policy and users should verify |
| Audit logging | Record all chatbot interactions for liability and quality assurance review |

## Legal Precedent

This case established an important precedent: **companies are liable for AI-generated misinformation on their platforms**, even when the AI makes an honest (statistical) error. The tribunal explicitly rejected Air Canada's argument that the chatbot was a "separate legal entity."

## References

- [BBC: Air Canada chatbot gave a customer false information. The airline must pay up](https://www.bbc.com/news/world-us-canada-68357046)
- [The Guardian: Air Canada must honor refund policy invented by airline's chatbot](https://www.theguardian.com/world/2024/feb/19/air-canada-chatbot-refund-policy-ruling)
- [Civil Resolution Tribunal Decision (Moffatt v. Air Canada)](https://tribunalsdecisions.gov.bc.ca/crt/2032428) — full legal ruling
