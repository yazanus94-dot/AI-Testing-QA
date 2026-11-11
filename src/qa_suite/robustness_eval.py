"""Prompt robustness and regression QA.

Each record (robustness_eval.jsonl): {
  id, base_prompt, variants: [{name, prompt, output}],
  expectation: {type: "contains_all"|"contains_any"|"refusal"|"exact"|"json_keys"|"max_words", value: ...},
  expected_consistent: bool (default true), expected_verdict?: "pass"|"fail"
}
Checks:
- per-variant expectation satisfaction (regression/oracle)
- cross-variant consistency (normalized answer agreement / stability)
"""
from __future__ import annotations
import json
import re
from collections import Counter

from .common import Issue

REFUSAL_MARKERS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i can't provide", "i cannot provide", "i'm sorry, but", "i won't", "i'm not able to help",
)

def _norm_text(t: str) -> str:
    t = (t or "").replace("\u2019", "'").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t

def _is_refusal(t: str) -> bool:
    low = (t or "").replace("\u2019", "'").lower()
    return any(m in low for m in REFUSAL_MARKERS)

def _check_expectation(output: str, expectation: dict):
    etype = (expectation or {}).get("type")
    val = (expectation or {}).get("value")
    out = output or ""
    low = out.lower()
    if etype == "contains_all":
        missing = [s for s in (val or []) if str(s).lower() not in low]
        return (not missing, f"missing expected content {missing}" if missing else "")
    if etype == "contains_any":
        opts = [s for s in (val or []) if str(s).lower() in low]
        return (bool(opts), f"output contains none of {val}" if not opts else "")
    if etype == "refusal":
        want = True if val is None else bool(val)
        ok = _is_refusal(out) if want else (not _is_refusal(out))
        return (ok, "expected refusal behavior not met" if not ok else "")
    if etype == "exact":
        ok = _norm_text(out) == _norm_text(str(val))
        return (ok, f"expected '{val}'" if not ok else "")
    if etype == "json_keys":
        try:
            data = json.loads(out.strip())
            missing = [k for k in (val or []) if k not in (data if isinstance(data, dict) else {})]
            ok = isinstance(data, dict) and not missing
            return (ok, "invalid JSON/object or missing keys" + (f" {missing}" if missing else "") if not ok else "")
        except Exception as e:
            return (False, f"not valid JSON ({e})")
    if etype == "max_words":
        n = len(out.split())
        ok = n <= int(val)
        return (ok, f"{n} words exceeds {val}" if not ok else "")
    ok = bool(out.strip())
    return (ok, "output is empty" if not ok else "")

def evaluate_robustness_record(rec: dict) -> dict:
    rid = rec.get("id")
    variants = rec.get("variants") or []
    expectation = rec.get("expectation") or {}
    exp_consistent = rec.get("expected_consistent", True)

    variant_results = []
    for v in variants:
        ok, detail = _check_expectation(v.get("output", ""), expectation)
        variant_results.append({
            "name": v.get("name"),
            "ok": ok,
            "detail": detail,
            "normalized": _norm_text(v.get("output", "")),
        })

    failed_variants = [v["name"] for v in variant_results if not v["ok"]]
    norms = [v["normalized"] for v in variant_results if v["normalized"]]
    counts = Counter(norms)
    majority_share = (counts.most_common(1)[0][1] / len(norms)) if norms else 0.0
    unique_answers = len(counts)
    consistent = (unique_answers <= 1) or (majority_share >= 0.8)

    failures = []
    if failed_variants:
        failures.append({"check": "variant_regression", "detail": f"variants failed expectation: {failed_variants}"})
    if exp_consistent and not consistent:
        failures.append({"check": "inconsistency", "detail": f"outputs diverge across variants (unique={unique_answers}, majority_share={round(majority_share, 3)})"})
    if (not exp_consistent) and consistent:
        failures.append({"check": "unexpected_consistency", "detail": "outputs were consistent though case expected divergence"})

    verdict = "fail" if failures else "pass"
    return {
        "id": rid,
        "base_prompt": rec.get("base_prompt"),
        "verdict": verdict,
        "failures": failures,
        "metrics": {
            "n_variants": len(variants),
            "failed_variants": failed_variants,
            "unique_normalized_answers": unique_answers,
            "majority_answer_share": round(majority_share, 3),
            "consistent": consistent,
        },
        "expected_verdict": rec.get("expected_verdict"),
    }

def run_robustness_eval(records: list[dict]) -> dict:
    issues: list[Issue] = []
    parse_errors = [r.get("_line") for r in records if "_parse_error" in r]
    if parse_errors:
        issues.append(Issue("parse_errors", "critical", f"{len(parse_errors)} unparseable robustness record(s)", parse_errors))
    valid = [r for r in records if "_parse_error" not in r]
    results = [evaluate_robustness_record(r) for r in valid]
    failed = [r for r in results if r["verdict"] == "fail"]
    by_check = Counter(f["check"] for r in failed for f in r["failures"])

    labeled = [r for r in results if r.get("expected_verdict")]
    tp = sum(r["expected_verdict"] == "fail" and r["verdict"] == "fail" for r in labeled)
    fp_ids = [r["id"] for r in labeled if r["expected_verdict"] == "pass" and r["verdict"] == "fail"]
    fn_ids = [r["id"] for r in labeled if r["expected_verdict"] == "fail" and r["verdict"] == "pass"]
    precision = tp / (tp + len(fp_ids)) if (tp + len(fp_ids)) else None
    recall = tp / (tp + len(fn_ids)) if (tp + len(fn_ids)) else None

    inconsistent = [r["id"] for r in results if not r["metrics"].get("consistent", True)]
    regressed = [r["id"] for r in results if r["metrics"].get("failed_variants")]
    if results and len(inconsistent) >= max(2, (len(results) + 2) // 3):
        issues.append(Issue("high_inconsistency", "major", f"High prompt sensitivity/inconsistency across cases ({len(inconsistent)}/{len(results)})", inconsistent))
    if regressed:
        issues.append(Issue("robustness_regressions", "major", f"{len(regressed)} case(s) have variant(s) failing expected behavior", regressed))

    avg_cons = round(sum(r["metrics"].get("majority_answer_share", 0) for r in results) / len(results), 3) if results else None
    summary = {
        "total": len(results), "passed": len(results) - len(failed), "failed": len(failed),
        "pass_rate": round((len(results) - len(failed)) / len(results), 3) if results else None,
        "failures_by_check": dict(by_check),
        "avg_majority_answer_share": avg_cons,
        "inconsistent_cases": len(inconsistent),
        "evaluator_self_check": {
            "labeled_records": len(labeled),
            "precision": precision and round(precision, 3),
            "recall": recall and round(recall, 3),
            "false_negatives": fn_ids,
            "false_positives": fp_ids,
        }
    }
    return {"summary": summary, "results": results, "issues": issues}
