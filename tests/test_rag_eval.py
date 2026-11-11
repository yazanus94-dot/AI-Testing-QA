from qa_suite.rag_eval import evaluate_rag_record, run_rag_eval


def test_grounded_answer_with_valid_citation_passes():
    rec = {
        "id": "r1",
        "question": "What is the return window for electronics?",
        "contexts": ["Electronics can be returned within 30 days of delivery."],
        "answer": "Electronics can be returned within 30 days of delivery.",
        "citations": [1],
        "constraints": {"must_be_grounded": True, "require_citation": True, "min_context_relevance": 0.1, "min_faithfulness": 0.2},
    }
    out = evaluate_rag_record(rec)
    assert out["verdict"] == "pass"


def test_ungrounded_or_bad_citation_fails():
    rec = {
        "id": "r2",
        "question": "What is the warranty period for laptops?",
        "contexts": ["Shipping takes 3 to 5 business days."],
        "answer": "Laptops have a two-year warranty covering accidental damage.",
        "citations": [2],
        "constraints": {"must_be_grounded": True, "require_citation": True, "min_context_relevance": 0.1, "min_faithfulness": 0.2},
    }
    out = evaluate_rag_record(rec)
    checks = {f["check"] for f in out["failures"]}
    assert out["verdict"] == "fail"
    assert "citation" in checks


def test_run_rag_eval_summary():
    rows = [
        {"id": "r1", "question": "When does the plan renew?", "contexts": ["The standard plan renews monthly on the billing date."], "answer": "The standard plan renews monthly on the billing date.", "citations": [1], "constraints": {"must_be_grounded": True, "require_citation": True, "min_context_relevance": 0.1, "min_faithfulness": 0.2}, "expected_verdict": "pass"},
        {"id": "r2", "question": "Can I return a gift card?", "contexts": ["Gift cards are non-refundable."], "answer": "Gift cards are non-refundable.", "citations": [], "constraints": {"must_be_grounded": True, "require_citation": True, "min_context_relevance": 0.1, "min_faithfulness": 0.2}, "expected_verdict": "fail"},
    ]
    out = run_rag_eval(rows)
    assert out["summary"]["total"] == 2
    assert out["summary"]["failed"] == 1
