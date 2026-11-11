# AI Testing & QA Evaluation Suite

A comprehensive, reproducible **AI/ML quality assurance portfolio project** that plans, executes, and documents quality checks across LLM output validation, annotation/dataset quality control, RLHF preference-data evaluation, **RAG groundedness & citation QA**, and **prompt robustness & regression QA**. The suite runs with the Python standard library and produces release-style quality-gate reports with evidence-backed findings.

**Author:** [Yazan Aljabari](https://www.linkedin.com/in/yazan-aljabari-878202139/)  
**License:** [MIT](LICENSE)  
**Branch:** `main`

## Overview

This repository demonstrates practical QA engineering for AI systems: validating model behavior, auditing human-labeled data, evaluating preference signals, testing retrieval-grounded answers for faithfulness/attribution, and checking prompt-variant robustness. It combines automated checks, statistical metrics, test planning artifacts, bug reports, and CI-friendly quality gating.

### Included AI testing topics

| Area | Focus | Key outputs |
|---|---|---|
| LLM output validation | Format, factual/content constraints, safety refusals, prompt-injection resistance, PII leakage, length/repetition/emptiness | Pass/fail per case, failures by check, validator self-check (precision/recall) |
| Dataset / annotation QC | Schema/label validity, duplicates, completeness, class balance, inter-annotator agreement, gold-set accuracy, adjudication needs | Cohen/Fleiss kappa, annotator diagnostics, issue findings |
| RLHF / human-feedback evaluation | Majority-vote outcomes, win rates with confidence intervals, agreement, position/verbosity bias detection | Model win-rate summary, annotator bias flags, adjudication cases |
| **RAG groundedness & citation QA** *(added topic)* | Context relevance, answer faithfulness/support, unsupported claims, citation presence/validity/attribution | Case-level relevance/faithfulness metrics, citation/grounding failures, evaluator self-check |
| **Prompt robustness & regression QA** *(added topic)* | Expectation compliance across prompt variants/paraphrases, consistency/stability, regression and sensitivity detection | Variant pass/fail, consistency metrics, regression/inconsistency findings, evaluator self-check |

## Problem statement

AI systems can fail in ways traditional software tests miss: hallucinated or ungrounded answers, unsafe or privacy-leaking outputs, inconsistent behavior under prompt paraphrases, biased human feedback, and low-quality training/evaluation labels. Without structured, evidence-based QA, these defects propagate into product behavior, model evaluation, and downstream training.

## Objectives

- Provide a reproducible QA framework for evaluating AI outputs and data quality end-to-end.
- Detect correctness, safety, privacy, grounding, and robustness defects with explainable checks and auditable reports.
- Measure annotation and preference-data reliability using agreement/statistical diagnostics.
- **Extend coverage with two additional AI testing topics:** RAG groundedness/citation QA and prompt robustness/regression QA.
- Support CI/release decisions through a clear quality gate and machine-readable artifacts.

## Features

- **Five-layer AI QA coverage:** output validation, dataset/annotation QC, RLHF evaluation, RAG groundedness/citation QA, and prompt robustness/regression QA.
- **Automated pipeline & quality gate:** `run_qa.py` generates `reports/qa_report.md` and `reports/qa_report.json` (exit code control via `--fail-on-gate`).
- **Explainable findings:** severity-ranked issues with examples, metrics, evaluator self-checks, and recommended actions.
- **Synthetic, reproducible sample data:** deterministic generators and included datasets with planted defects for demonstration (no fabricated real-world claims).
- **Test assets:** test plan, test cases, bug reports, annotation guidelines, and pytest suite.
- **CI workflow:** GitHub Actions runs tests and produces QA reports.

## Technologies

- **Language:** Python 3.9+ (core pipeline uses the standard library)
- **Testing:** pytest
- **Data formats:** JSONL, JSON, Markdown, CSV
- **CI:** GitHub Actions
- **Methods/metrics:** rule-based oracles, Cohen’s kappa, Fleiss’ kappa, Wilson confidence intervals, lexical relevance/faithfulness heuristics, consistency analysis

## Project structure

```text
.
├── data/                         # Sample datasets (annotations, model outputs, preferences, RAG, robustness)
├── docs/                         # Test plan, test cases, annotation guidelines, bug reports
├── reports/                      # Generated QA reports (Markdown/JSON)
├── scripts/                      # Sample data generation utilities
├── src/qa_suite/                 # QA modules and report/gate logic
├── tests/                        # Unit and end-to-end tests
├── .github/workflows/ci.yml      # CI pipeline
├── run_qa.py                     # CLI entry point and quality gate
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Dependencies (pytest for testing)
├── LICENSE                       # MIT license
└── README.md
```

## Installation

### Prerequisites
- Python **3.9+** (3.10+ recommended)
- `pip`

### Setup
```bash
git clone https://github.com/yazanus94-dot/AI-Testing-QA.git
cd AI-Testing-QA

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt  # installs pytest (runtime pipeline uses stdlib)
```

## Usage

```bash
# (Optional) regenerate synthetic sample data with planted defects
python scripts/generate_sample_data.py

# Run full QA suite and generate reports
python run_qa.py

# Enforce quality gate in CI/CD (exit code 1 on FAIL)
python run_qa.py --fail-on-gate

# Run tests
pytest -q
```

Outputs:
- `reports/qa_report.md` — human-readable QA report and findings
- `reports/qa_report.json` — machine-readable results, summaries, and issues

## Methodology

### 1. LLM output validation
- Constraint-driven checks per record (JSON/schema expectations, required/forbidden content, refusals, length limits, PII patterns, repetition/emptiness).
- **Validator self-check:** compares automated verdicts to labeled expectations to report precision/recall and known blind spots.

### 2. Dataset / annotation quality control
- Schema/label validity, empty/duplicate detection, completeness, and class distribution analysis.
- Agreement metrics (pairwise Cohen’s kappa, Fleiss’ kappa) and gold-set accuracy to identify unreliable annotators/items and adjudication needs.

### 3. RLHF / human-feedback evaluation
- Majority-vote aggregation, model win rates with **95% Wilson confidence intervals**, inter-annotator agreement.
- Bias diagnostics: position bias (A/B preference skew) and verbosity/length bias.

### 4. RAG groundedness & citation QA *(added topic)*
- **Context relevance:** lexical overlap between question and retrieved contexts (heuristic proxy).
- **Faithfulness/support:** answer-to-context support and detection of weakly supported/unsupported claim sentences.
- **Citation checks:** required citations present, indices valid, and cited contexts relevant to the question.
- Produces case metrics, issue findings (grounding/citation/relevance failures), and evaluator self-check against labeled verdicts.

### 5. Prompt robustness & regression QA *(added topic)*
- Evaluates behavior across prompt variants/paraphrases/perturbations against explicit expectations (content, refusal, exact/JSON/length constraints).
- **Consistency/stability:** normalized answer agreement and majority-share consistency across variants.
- Detects regressions (variant expectation failures) and high sensitivity/inconsistency; includes evaluator self-check.

### Quality gate
- **FAIL:** any critical or major issue.
- **PASS WITH WARNINGS:** only minor issues.
- **PASS:** no issues.

## Results

The included sample run is generated from **synthetic data with deliberately planted defects** (for demonstration of detection capability) and should be interpreted as a QA demonstration, not a claim about a production model.

Included generated artifacts (`reports/qa_report.md`, `reports/qa_report.json`) show, for the bundled sample data:
- **Overall quality gate:** **FAIL** (sample contains planted defects by design).
- **LLM output validation, dataset/annotation QC, and RLHF evaluation:** expected planted issues detected (e.g., schema/label/duplicate/missing-annotation problems, biased/invalid preference signals, output check failures) with documented metrics and validator self-check limitations.
- **RAG groundedness & citation QA:** detects grounding, citation, and/or relevance failures on planted cases (e.g., missing/invalid citations, low-support/irrelevant contexts).
- **Prompt robustness & regression QA:** detects variant regressions and/or inconsistency across prompt paraphrases/perturbations.

> **Transparency:** Results reflect the included synthetic sample datasets and rule/heuristic evaluators. They demonstrate QA behavior and reporting, not real-user performance.

## Testing

```bash
# All tests
pytest -q

# Example targeted tests
pytest -q tests/test_output_validation.py tests/test_dataset_qc.py tests/test_rlhf_eval.py tests/test_rag_eval.py tests/test_robustness_eval.py tests/test_end_to_end.py
```

- Unit tests cover metrics, validators, dataset QC, RLHF evaluation, RAG groundedness/citation checks, and prompt robustness checks.
- End-to-end tests execute the pipeline and verify gate/report behavior.
- CI runs automated tests via GitHub Actions.

## Limitations

- **Rule/heuristic-based evaluation:** lexical overlap and pattern checks cannot fully capture semantic correctness, nuanced faithfulness, or complex safety judgments (false negatives/positives possible).
- **Synthetic data:** included datasets and planted defects are illustrative; outcomes do not represent real production systems.
- **Statistical sensitivity:** agreement and confidence intervals depend on sample size and annotation design.
- **RAG evaluation scope:** retrieval quality and faithfulness are approximated without embedding models or human/LLM judges; citation checks assume provided context indices.
- **Robustness scope:** consistency is measured on normalized outputs and defined expectations; broader adversarial/behavioral coverage requires expanded suites.

## Future improvements

- Add semantic/LLM-as-judge or embedding-based evaluators for faithfulness, relevance, and nuanced correctness.
- Support ordinal/weighted agreement (e.g., Krippendorff’s alpha), per-class metrics, and calibration analyses.
- Expand RAG tests with retrieval-stage metrics, claim-level attribution, and adversarial context cases.
- Broaden robustness suites (multilingual paraphrases, perturbation fuzzing, jailbreak/red-team regressions, metamorphic tests).
- Add dashboards/trend tracking over `qa_report.json` history and configurable policy thresholds per environment.
- Integrate human adjudication workflows and active-learning loops for ambiguous/low-agreement items.

## References

- Landis, J. R., & Koch, G. G. (1977). *The measurement of observer agreement for categorical data.* Biometrics.
- Cohen, J. (1960). *A coefficient of agreement for nominal scales.* Educational and Psychological Measurement.
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* JASA.
- Project artifacts in this repository: `docs/test_plan.md`, `docs/test_cases.csv`, `docs/bug_reports.md`, `docs/annotation_guidelines.md`.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

**Copyright (c) 2026 Yazan Aljabari** — [LinkedIn](https://www.linkedin.com/in/yazan-aljabari-878202139/)
