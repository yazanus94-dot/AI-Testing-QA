# Test Plan: AI Output & Annotation Quality

| | |
|---|---|
| **Product under test** | LLM-based assistant + the human-labeled datasets used to evaluate / fine-tune it |
| **Author** | Yazan |
| **Version** | 1.0 |
| **Status** | Executed against sample data (see `reports/qa_report.md`) |

## 1. Objectives
1. Verify that model outputs meet functional, format, factual, and safety requirements.
2. Verify that annotation data (labels and human preferences) is clean, consistent, and trustworthy *before* it is used for training or evaluation.
3. Detect annotator-level quality problems (low agreement, low gold accuracy, position / verbosity bias).
4. Provide a repeatable, automated quality gate with a clear PASS / FAIL result and an audit trail.

## 2. Scope
| In scope | Out of scope |
|---|---|
| Output format (JSON), factual spot-checks, refusal behavior, prompt-injection resistance, PII leakage, length limits, empty / repetitive output | Model training, latency / load testing |
| Annotation schema, label validity, duplicates, completeness, class balance | Subjective creative quality scoring |
| Inter-annotator agreement (Cohen / Fleiss kappa), gold-set accuracy | UI testing of the annotation tool |
| RLHF pairwise preferences: win rate, agreement, position and length bias | Reward model training |

## 3. Test strategy
| Layer | Technique | Where |
|---|---|---|
| Data validation | Schema, enum, null, duplicate, completeness checks | `dataset_qc.py` |
| Statistical QC | Cohen's / Fleiss' kappa, gold-set accuracy, class balance | `dataset_qc.py`, `metrics.py` |
| Output validation | Rule-based oracles per prompt (constraints are test data, not code) | `output_validation.py` |
| Human-feedback QC | Majority vote, win rate + Wilson CI, bias detection | `rlhf_eval.py` |
| Tool verification | Unit tests incl. known-value kappa, edge cases, end-to-end | `tests/` |
| Oracle verification | Validator is scored against human-labeled verdicts (precision / recall) | report section 2 |
| Manual / exploratory | Human review of ambiguous items, adjudication queue | `docs/annotation_guidelines.md` |

## 4. Entry and exit criteria
**Entry:** datasets exported in JSONL; label schema and guidelines frozen; gold items identified.

**Exit (release gate):**
| Criterion | Threshold |
|---|---|
| Critical or major data issues | 0 |
| Fleiss' kappa (labels) | >= 0.60 |
| Annotator gold accuracy | >= 85% each |
| Preference-data kappa | >= 0.40 |
| Annotator position bias (% choosing A) | 35% to 65% |
| Annotator verbosity bias (% choosing longer) | <= 70% |
| Safety refusal tests | 100% pass |
| PII leakage tests | 100% pass |

## 5. Severity definitions
| Severity | Definition | Example |
|---|---|---|
| Critical | Data unusable or safety / privacy breach with no workaround | Unparseable file, missing required fields |
| Major | Corrupts metrics or model behavior if shipped | Invalid labels, duplicate texts (leakage), biased annotator |
| Minor | Reduces coverage or efficiency, workaround exists | Missing single annotation, items needing adjudication |
| Info | Observation only | Class distribution note |

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Rule-based oracles miss semantic errors (e.g., wrong date) | Report the validator's recall; route to human review / LLM-as-judge (see out_021) |
| Small sample sizes make kappa noisy | Report n for every metric; use confidence intervals for win rates |
| Annotator bias contaminates preference data | Randomize A/B order, calibration pairs, per-annotator bias metrics |
| Confounded signals (e.g., length vs position) | Evaluate each bias separately and review flags together before acting |

## 7. Environment and tooling
Python 3.9+, standard library only (no external dependencies for runtime), pytest for the test suite, GitHub Actions for CI.

## 8. Execution and reporting
1. `python run_qa.py` runs all suites and writes `reports/qa_report.md` and `.json`.
2. Defects are logged using `docs/bug_reports.md` format.
3. Fixed data / outputs are re-run (regression) until the gate is PASS.
4. Edge cases found are added to `docs/test_cases.csv` and the unit tests.
