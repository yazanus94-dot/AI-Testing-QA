"""RAG / grounded-answer QA: context relevance, answer faithfulness, and citation coverage.

Each record (rag_eval.jsonl): {
  id, question, contexts: [str], answer, citations: [int] (1-based context refs),
  constraints: {must_be_grounded, min_context_relevance, min_faithfulness, require_citation},
  expected_verdict?: "pass"|"fail"
}
Heuristic/lexical checks (no external models):
- context_relevance: overlap between question terms and retrieved contexts
- answer_faithfulness: answer claims supported by provided contexts (token overlap proxy)
- citation_coverage: cited contexts exist and support the answer; flags missing/invalid citations
- unsupported_claims: sentences in answer with low support from contexts
"""
from __future__ import annotations
import re
from collections import Counter

from .common import Issue

STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were","be","by","it","this","that","as",
    "at","from","what","how","why","when","which","who","do","does","did","can","could","should","would","about","into",
    "than","then","its","their","there","your","you","we","our"
}

def _tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP and len(t) > 1]

def _overlap(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a:
        return 0.0
    return len(a & b) / len(a)

def _sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]

def evaluate_rag_record(rec: dict) -> dict:
    rid = rec.get("id")
    q = rec.get("question") or ""
    contexts = rec.get("contexts") or []
    answer = rec.get("answer") or ""
    citations = rec.get("citations") or []
    c = rec.get("constraints") or {}
    failures = []

    qtok = _tokens(q)
    ctx_text = " \n ".join(str(x) for x in contexts)
    ctxtok = _tokens(ctx_text)

    # context relevance
    rel_scores = [round(_overlap(qtok, _tokens(str(ctx))), 3) for ctx in contexts] if contexts else [0.0]
    relevance = round(sum(rel_scores) / len(rel_scores), 3) if rel_scores else 0.0
    min_rel = c.get("min_context_relevance", 0.15)
    if relevance < min_rel:
        failures.append({"check": "context_relevance", "detail": f"mean question-context relevance {relevance} below {min_rel} (per-context {rel_scores})"})

    # answer faithfulness (supported by contexts)
    atok = _tokens(answer)
    if c.get("must_be_grounded", True):
        if not contexts:
            failures.append({"check": "faithfulness", "detail": "no contexts provided for a grounded answer"})
            faith = 0.0
        else:
            faith = round(_overlap(atok, ctxtok), 3) if atok else 0.0
            min_faith = c.get("min_faithfulness", 0.35)
            if faith < min_faith:
                failures.append({"check": "faithfulness", "detail": f"answer-context support {faith} below {min_faith}"})
    else:
        faith = round(_overlap(atok, ctxtok), 3) if atok and ctxtok else None

    # unsupported claim sentences
    unsupported = []
    if contexts and answer.strip():
        for s in _sentences(answer):
            stok = _tokens(s)
            if len(stok) < 3:
                continue
            sup = _overlap(stok, ctxtok)
            if sup < 0.2:
                unsupported.append({"sentence": s[:120], "support": round(sup, 3)})
    if unsupported and c.get("must_be_grounded", True):
        failures.append({"check": "unsupported_claims", "detail": f"{len(unsupported)} sentence(s) weakly supported by contexts"})

    # citation coverage/validity
    if c.get("require_citation", False):
        if not citations:
            failures.append({"check": "citation", "detail": "answer requires citation but none provided"})
        else:
            invalid = [ci for ci in citations if not isinstance(ci, int) or ci < 1 or ci > len(contexts)]
            if invalid:
                failures.append({"check": "citation", "detail": f"invalid citation indices {invalid} for {len(contexts)} context(s)"})
            if not invalid and contexts and qtok:
                cited_rel = []
                for ci in citations:
                    if 1 <= ci <= len(contexts):
                        cited_rel.append(_overlap(qtok, _tokens(str(contexts[ci - 1]))))
                if cited_rel and (sum(cited_rel) / len(cited_rel)) < 0.1:
                    failures.append({"check": "citation", "detail": "cited contexts have low relevance to the question"})

    verdict = "fail" if failures else "pass"
    return {
        "id": rid,
        "question": q,
        "verdict": verdict,
        "failures": failures,
        "metrics": {
            "context_relevance": relevance,
            "context_relevance_per_doc": rel_scores,
            "faithfulness": faith,
            "unsupported_sentences": unsupported,
            "citations": citations,
        },
        "expected_verdict": rec.get("expected_verdict"),
    }


def run_rag_eval(records: list[dict]) -> dict:
    issues: list[Issue] = []
    parse_errors = [r.get("_line") for r in records if "_parse_error" in r]
    if parse_errors:
        issues.append(Issue("parse_errors", "critical", f"{len(parse_errors)} unparseable RAG record(s)", parse_errors))
    valid = [r for r in records if "_parse_error" not in r]
    results = [evaluate_rag_record(r) for r in valid]
    failed = [r for r in results if r["verdict"] == "fail"]
    by_check = Counter(f["check"] for r in failed for f in r["failures"])

    labeled = [r for r in results if r.get("expected_verdict")]
    tp = sum(r["expected_verdict"] == "fail" and r["verdict"] == "fail" for r in labeled)
    fp_ids = [r["id"] for r in labeled if r["expected_verdict"] == "pass" and r["verdict"] == "fail"]
    fn_ids = [r["id"] for r in labeled if r["expected_verdict"] == "fail" and r["verdict"] == "pass"]
    precision = tp / (tp + len(fp_ids)) if (tp + len(fp_ids)) else None
    recall = tp / (tp + len(fn_ids)) if (tp + len(fn_ids)) else None

    avg_rel = round(sum((r["metrics"].get("context_relevance") or 0) for r in results) / len(results), 3) if results else None
    faith_vals = [r["metrics"]["faithfulness"] for r in results if r["metrics"].get("faithfulness") is not None]
    avg_faith = round(sum(faith_vals) / len(faith_vals), 3) if faith_vals else None

    low_rel = [r["id"] for r in results if (r["metrics"].get("context_relevance") or 0) < 0.15]
    if results and len(low_rel) >= max(2, (len(results) + 1) // 2):
        issues.append(Issue("low_retrieval_relevance", "major", f"Many cases have low question-context relevance ({len(low_rel)}/{len(results)})", low_rel))
    citation_fails = [r["id"] for r in failed if any(f["check"] == "citation" for f in r["failures"])]
    if citation_fails:
        issues.append(Issue("citation_failures", "major", f"{len(citation_fails)} case(s) fail citation/attribution checks", citation_fails))
    faith_fails = [r["id"] for r in failed if any(f["check"] in ("faithfulness", "unsupported_claims") for f in r["failures"])]
    if faith_fails:
        issues.append(Issue("grounding_failures", "major", f"{len(faith_fails)} case(s) show weak grounding/unsupported claims", faith_fails))

    summary = {
        "total": len(results), "passed": len(results) - len(failed), "failed": len(failed),
        "pass_rate": round((len(results) - len(failed)) / len(results), 3) if results else None,
        "failures_by_check": dict(by_check),
        "avg_context_relevance": avg_rel,
        "avg_faithfulness": avg_faith,
        "evaluator_self_check": {
            "labeled_records": len(labeled),
            "precision": precision and round(precision, 3),
            "recall": recall and round(recall, 3),
            "false_negatives": fn_ids,
            "false_positives": fp_ids,
        }
    }
    return {"summary": summary, "results": results, "issues": issues}
