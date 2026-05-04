# Evaluation Scripts

This directory contains runnable scripts and Jupyter notebooks for testing AI systems against common safety and robustness concerns.

> **Prerequisite**: Install dependencies before running any script.
> ```bash
> pip install -r requirements.txt
> ```

---

## Scripts Overview

| Script | Purpose | Runtime | Key Dependencies |
|--------|---------|---------|------------------|
| [`bias_detection_demo.ipynb`](bias_detection_demo.ipynb) | Detect, visualize, and mitigate dataset bias using synthetic data | ~2 min | pandas, scikit-learn, matplotlib, aif360 (optional) |
| [`prompt_adversary_test.ipynb`](prompt_adversary_test.ipynb) | Test adversarial prompt robustness against an LLM | ~1 min | openai, os |
| [`data_preprocessing.py`](data_preprocessing.py) | Standard preprocessing pipeline (cleaning, encoding, SMOTE) | ~30 sec | pandas, scikit-learn, imbalanced-learn |

---

## Bias Detection Demo

**What it does:**
- Generates a **synthetic hiring dataset** with intentional gender bias (no external data files needed)
- Trains a logistic regression classifier
- Measures **disparate impact** and **statistical parity difference** (via AIF360 if installed, or manual fallback)
- Visualizes outcome and experience distributions by gender
- Applies **reweighing** bias mitigation and compares before/after metrics

**How to run:**
```bash
jupyter notebook evaluation-scripts/bias_detection_demo.ipynb
```
Then execute cells sequentially. All cells are self-contained.

**Expected output:**
- Dataset shape and gender distribution printed
- Baseline accuracy ~0.78
- Disparate Impact before mitigation: ~0.7–0.8 (indicates bias)
- Disparate Impact after reweighing: ~1.0 (parity achieved)
- Side-by-side histograms and a before/after comparison table

**If AIF360 is not installed:** The notebook falls back to manual metrics and prints instructions for installing AIF360:
```bash
pip install aif360
```

---

## Prompt Adversary Test

**What it does:**
- Sends **adversarial prompts** to an LLM via the Fireworks AI API
- Tests direct jailbreak attempts, instruction overrides, and role manipulation
- Evaluates whether the model maintains safety guardrails under adversarial input

**How to run:**
1. Set your Fireworks API key as an environment variable:
   ```bash
   export FIREWORKS_API_KEY=your-key-here
   ```
2. Open the notebook:
   ```bash
   jupyter notebook evaluation-scripts/prompt_adversary_test.ipynb
   ```
3. Execute cells. The final cell loops through adversarial prompts and prints responses.

**Expected output:**
- For benign prompts (translation): normal completion
- For jailbreak attempts: ideally a refusal or safe response
- For adversarial role override: ideally a refusal

**Notes:**
- Uses `llama-v3p1-8b-instruct` via Fireworks AI's OpenAI-compatible endpoint
- Replace the model ID with your own deployment if testing a fine-tuned or private model
- **Never run this against production systems without explicit authorization**

---

## Data Preprocessing Script

**What it does:**
- Loads a CSV dataset
- Handles missing values, encodes categoricals, caps outliers
- Normalizes features and applies SMOTE for class imbalance
- Outputs a cleaned CSV for downstream modeling

**How to run:**
```bash
python evaluation-scripts/data_preprocessing.py
```

**Expected output:**
- `processed_data.csv` in the working directory
- Console summary of preprocessing steps applied

**Notes:**
- Modify the `input_file` path at the top of the script to point to your dataset
- The script assumes a target column named `target`; adjust if yours differs

---

## Adding New Evaluation Scripts

When contributing new scripts:
1. Include a header cell/docstring explaining **what** the script tests and **why** it matters
2. List all dependencies and how to install them
3. Provide sample input/output or expected behavior
4. Add a row to the table above in this README
5. Update `requirements.txt` if new packages are needed

---

## Safety Notes

- **Never test adversarial prompts on production APIs without authorization**
- **Never use evaluation results to attack or bypass third-party systems**
- **Always disclose findings responsibly** to system owners before public disclosure
- Bias detection should be run on **synthetic or anonymized data** when possible to avoid exposing PII during analysis

---

*For questions or issues, contact [heather@fireworks.ai](mailto:heather@fireworks.ai) or open an issue at [https://github.com/heathriel/ai-safety-resources](https://github.com/heathriel/ai-safety-resources)*
