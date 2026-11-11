"""Markdown report builder with a release-style quality gate."""

from datetime import date

from .common import Issue, count_by_severity

RECOMMENDATIONS = {
    "invalid_label": "Enforce a closed label set in the annotation tool (dropdown, not free text) and add schema validation on export.",
    "empty_text": "Filter empty / whitespace-only items in the ingestion pipeline before they reach annotators.",
    "duplicate_text": "Deduplicate on normalized text before labeling and before train/test splitting to avoid leakage.",
    "missing_annotation": "Re-queue items that are missing an annotator's label until coverage is complete.",
    "low_overall_agreement": "Hold a calibration session, tighten the guidelines for ambiguous classes, and re-measure kappa.",
    "low_annotator_agreement": "Review the outlier annotator's work against the guidelines; retrain or reassign before using their labels.",
    "low_gold_accuracy": "Pause the low-accuracy annotator, give feedback on missed gold items, and re-test on a fresh gold set.",
    "no_majority": "Send no-majority items to an expert adjudicator for the final label.",
    "position_bias": "Randomize A/B order per annotator and add a short refresher on ignoring position.",
    "length_bias": "Add 'do not reward length alone' to the preference rubric and include calibration pairs where the shorter answer is better.",
    "invalid_preference_row": "Validate preference exports (choice in A/B/tie) before aggregation.",
    "low_preference_agreement": "Clarify the preference rubric (helpfulness, accuracy, safety order of precedence) and recalibrate.",
    "no_consensus_pairs": "Collect an additional vote or expert adjudication for pairs without a majority.",
    "low_retrieval_relevance": "Improve retrieval/query formulation, chunking, and metadata filters; re-evaluate with relevance judgments.",
    "citation_failures": "Require grounded answers to cite valid supporting contexts and block responses with missing/invalid citations.",
    "grounding_failures": "Constrain generation to provided contexts, add refusal/abstention for unsupported questions, and review faithfulness failures.",
    "high_inconsistency": "Stabilize prompts/system instructions, normalize inputs, and add regression tests for paraphrases/perturbations.",
    "robustness_regressions": "Fix variant-specific failures and add them to a prompt-regression suite before release.",
}


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def _examples(items, n=5):
    items = [str(i) for i in items]
    if not items:
        return ""
    return ", ".join(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")


def _issue_table(issues):
    if not issues:
        return "_No issues found._"
    order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    rows = [(i.severity.upper(), i.check, i.message, _examples(i.items)) for i in sorted(issues, key=lambda x: order[x.severity])]
    return _table(["Severity", "Check", "Finding", "Examples"], rows)


def gate_status(issues: list[Issue]) -> str:
    c = count_by_severity(issues)
    if c["critical"] or c["major"]:
        return "FAIL"
    return "PASS WITH WARNINGS" if c["minor"] else "PASS"


def build_report(ds: dict, ov: dict, rl: dict, rg: dict = None, rb: dict = None) -> str:
    rg = rg or {"summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": None, "avg_context_relevance": None, "avg_faithfulness": None, "failures_by_check": {}, "evaluator_self_check": {"labeled_records": 0, "precision": None, "recall": None, "false_negatives": [], "false_positives": []}}, "results": [], "issues": []}
    rb = rb or {"summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": None, "avg_majority_answer_share": None, "inconsistent_cases": 0, "failures_by_check": {}, "evaluator_self_check": {"labeled_records": 0, "precision": None, "recall": None, "false_negatives": [], "false_positives": []}}, "results": [], "issues": []}
    all_issues = ds["issues"] + rl["issues"] + rg.get("issues", []) + rb.get("issues", [])
    gate = gate_status(all_issues)
    sev = count_by_severity(all_issues)
    d, o, r = ds["summary"], ov["summary"], rl["summary"]
    g, b = rg["summary"], rb["summary"]
    sc = o.get("validator_self_check", {})
    gsc = g.get("evaluator_self_check", {})
    bsc = b.get("evaluator_self_check", {})
    L = [
        "# AI QA Evaluation Report", f"_Generated {date.today().isoformat()}_", "",
        f"## Quality gate: **{gate}**", "",
        _table(["Area", "Headline result"], [
            ("Dataset annotation QC", f"{d.get('n_items', 0)} items / {d.get('n_rows', 0)} annotations, Fleiss' kappa {d.get('fleiss_kappa')} ({d.get('fleiss_interpretation')}), {len(ds['issues'])} issue type(s)"),
            ("LLM output validation", f"{o.get('passed', 0)}/{o.get('total', 0)} outputs passed ({(o.get('pass_rate') or 0):.0%}); {o.get('failed', 0)} failed"),
            ("RLHF preference data", f"{r.get('n_pairs', 0)} pairs, Fleiss' kappa {r.get('fleiss_kappa')} ({r.get('fleiss_interpretation')}), {len(rl['issues'])} issue type(s)"),
            ("RAG groundedness & citation QA", f"{g.get('passed', 0)}/{g.get('total', 0)} cases passed ({(g.get('pass_rate') or 0):.0%}); avg relevance {g.get('avg_context_relevance')}, avg faithfulness {g.get('avg_faithfulness')}, {len(rg.get('issues', []))} issue type(s)"),
            ("Prompt robustness & regression QA", f"{b.get('passed', 0)}/{b.get('total', 0)} cases passed ({(b.get('pass_rate') or 0):.0%}); avg consistency {b.get('avg_majority_answer_share')}, inconsistent {b.get('inconsistent_cases')}, {len(rb.get('issues', []))} issue type(s)"),
        ]), "",
        f"Issues by severity: critical **{sev['critical']}**, major **{sev['major']}**, minor **{sev['minor']}**. Gate rule: any critical/major issue = FAIL; minor only = PASS WITH WARNINGS.", "",
        "---", "## 1. Dataset annotation QC", "",
        "**Label distribution:** " + (", ".join(f"{k} {v}" for k, v in sorted(d.get("label_distribution", {}).items())) or "n/a"), "",
        "**Pairwise Cohen's kappa**", "",
        _table(["Pair", "Kappa", "Shared items"], [(p, v["kappa"], v["n"]) for p, v in d.get("pairwise_kappa", {}).items()]) if d.get("pairwise_kappa") else "_No pairwise kappa computed._", "",
        "**Accuracy vs gold-standard items**", "",
        _table(["Annotator", "Correct", "Total", "Accuracy", "Mean kappa vs others"], [(a, gld["correct"], gld["total"], f"{gld['accuracy']:.0%}", d.get("annotator_mean_kappa", {}).get(a, "n/a")) for a, gld in d.get("gold_accuracy", {}).items()]) if d.get("gold_accuracy") else "_No gold-set labels provided._", "",
        "**Findings**", "", _issue_table(ds["issues"]), "",
        "---", "## 2. LLM output validation", "",
        f"Pass rate: **{(o.get('pass_rate') or 0):.0%}** ({o.get('passed', 0)}/{o.get('total', 0)}). Failures by check: " + (", ".join(f"{k}: {v}" for k, v in sorted((o.get('failures_by_check') or {}).items())) or "none"), "",
        _table(["Output", "Task", "Failed check", "Detail"], [(x["id"], x.get("task_type"), f["check"], f["detail"]) for x in ov.get("results", []) for f in x.get("failures", [])]) if any(x.get("failures") for x in ov.get("results", [])) else "_No output validation failures._", "",
        "**Validator self-check**", "",
        _table(["Labeled records", "Precision", "Recall", "False negatives", "False positives"], [(sc.get("labeled_records"), sc.get("precision"), sc.get("recall"), ", ".join(sc.get("false_negatives", [])) or "none", ", ".join(sc.get("false_positives", [])) or "none")]), "",
        "---", "## 3. RLHF / human-feedback evaluation", "",
        _table(["Model", "Wins", "Decided pairs", "Win rate", "95% CI"], [(m, w["wins"], w["decided_pairs"], w["win_rate"], f"{w['ci95'][0]:.2f} to {w['ci95'][1]:.2f}") for m, w in r.get("win_rates", {}).items()]) if r.get("win_rates") else "_No preference data._",
        f"\nTies: {r.get('tie_pairs', 0)}, pairs without majority: {r.get('no_consensus_pairs', 0)}.", "",
        _table(["Annotator", "Rows", "% chose A", "% chose longer", "Agrees w/ majority", "Flags"], [(a, s["n"], s["pct_choose_A"], s["pct_choose_longer"], s["agreement_with_majority"], ", ".join(s.get("flags", [])) or "none") for a, s in r.get("annotators_detail", {}).items()]) if r.get("annotators_detail") else "_No annotator details._", "",
        "**Findings**", "", _issue_table(rl["issues"]), "",
        "---", "## 4. RAG groundedness & citation QA", "",
        f"Pass rate: **{(g.get('pass_rate') or 0):.0%}** ({g.get('passed', 0)}/{g.get('total', 0)}). Avg relevance: **{g.get('avg_context_relevance')}**, avg faithfulness: **{g.get('avg_faithfulness')}**.", "",
        _table(["Case", "Verdict", "Relevance", "Faithfulness", "Failed checks"], [(x["id"], x.get("verdict"), (x.get("metrics") or {}).get("context_relevance"), (x.get("metrics") or {}).get("faithfulness"), ", ".join(f["check"] for f in x.get("failures", [])) or "none") for x in rg.get("results", [])]) if rg.get("results") else "_No RAG evaluation data._", "",
        "**Evaluator self-check**", "",
        _table(["Labeled records", "Precision", "Recall", "False negatives", "False positives"], [(gsc.get("labeled_records"), gsc.get("precision"), gsc.get("recall"), ", ".join(gsc.get("false_negatives", [])) or "none", ", ".join(gsc.get("false_positives", [])) or "none")]), "",
        "**Findings**", "", _issue_table(rg.get("issues", [])), "",
        "---", "## 5. Prompt robustness & regression QA", "",
        f"Pass rate: **{(b.get('pass_rate') or 0):.0%}** ({b.get('passed', 0)}/{b.get('total', 0)}). Avg consistency: **{b.get('avg_majority_answer_share')}**, inconsistent: **{b.get('inconsistent_cases')}**.", "",
        _table(["Case", "Verdict", "Variants", "Consistency", "Failed checks"], [(x["id"], x.get("verdict"), (x.get("metrics") or {}).get("n_variants"), (x.get("metrics") or {}).get("majority_answer_share"), ", ".join(f["check"] for f in x.get("failures", [])) or "none") for x in rb.get("results", [])]) if rb.get("results") else "_No robustness evaluation data._", "",
        "**Evaluator self-check**", "",
        _table(["Labeled records", "Precision", "Recall", "False negatives", "False positives"], [(bsc.get("labeled_records"), bsc.get("precision"), bsc.get("recall"), ", ".join(bsc.get("false_negatives", [])) or "none", ", ".join(bsc.get("false_positives", [])) or "none")]), "",
        "**Findings**", "", _issue_table(rb.get("issues", [])), "",
        "---", "## 6. Recommended actions", "",
    ]
    seen = []
    for i in all_issues:
        if i.check in RECOMMENDATIONS and i.check not in seen:
            seen.append(i.check)
            L.append(f"- **{i.check}**: {RECOMMENDATIONS[i.check]}")
    return "\n".join(L) + "\n"
