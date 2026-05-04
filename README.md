# AI Safety Resources for Government Agencies

Welcome to the AI Safety Resources repository, a curated collection of materials aimed at promoting the responsible and secure deployment of artificial intelligence (AI) within government agencies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Quickstart Guide](#quickstart-guide)
- [License & Contact](#license--contact)

---

## Project Overview

This repository provides a comprehensive set of resources, including academic papers, reference materials, evaluation scripts, and best practice guidelines, to assist government agencies in implementing AI technologies safely and ethically.

---

## Directory Structure

- **`best-practices/`**  
  Guidelines and checklists for ethical AI deployment, model governance, and LLM security.
  - Ethical AI Checklist
  - Responsible AI Toolkit
  - Model Governance Guidelines
  - LLM Security Guide — threat taxonomy and defense-in-depth for large language models

- **`evaluation-scripts/`**  
  Scripts and notebooks for evaluating AI models, including bias detection and adversarial testing.
  - Bias Detection Demo (runnable with synthetic dataset)
  - Prompt Adversary Test
  - Data Preprocessing Script

- **`templates/`**  
  Reusable templates for AI governance documentation.
  - Model Card Template — aligned with MLCommons, NIST AI RMF, and EU AI Act
  - Risk Assessment Matrix — structured risk register with NIST AI RMF mapping

- **`case-studies/`**  
  Real-world AI incident analyses for training and risk assessment.
  - Amazon Recruiting Tool (bias), Microsoft Tay (jailbreaking), Air Canada Chatbot (hallucination)
  - Clearview AI (privacy), COMPAS Recidivism (fairness tradeoffs), GPT-3 Medical (crisis escalation)

- **`paper/`**  
  Academic publications related to AI safety.

- **`references/`**  
  External resources including whitepapers, frameworks, courses, and newsletters.
  - Updated with EU AI Act 2024, ISO/IEC 42001, and Anthropic RSP

---

## Quickstart Guide

To get started with this repository:

1. **Clone the repository**  
   Run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/heathriel/ai-safety-resources.git
   ```

2. **Navigate to the repository directory**  
   Move into the repository directory:
   ```bash
   cd ai-safety-resources
   ```

3. **Install necessary dependencies**  
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Explore the resources**  
   - Review the real-world incident analyses in `case-studies/` for training and risk planning.
   - Use templates in `templates/` to document your models and assess risks.
   - Execute the evaluation scripts in `evaluation-scripts/` for bias detection and adversarial testing.
   - Follow the guidelines in `best-practices/` for ethical deployment and LLM security.
   - Consult reference materials in `references/` for frameworks, whitepapers, and courses.
   - Review academic publications in `paper/`.

---

## License & Contact

This project is licensed under the MIT License. For more information, see the [MIT LICENSE](./LICENSE) file.

For inquiries or contributions, please contact:

**Heather Renze**  
Email: [heather@fireworks.ai](mailto:heather@fireworks.ai)
