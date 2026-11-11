from qa_suite.dataset_qc import run_dataset_qc


def row(item, ann, label, text=None, **kw):
    return {"item_id": item, "annotator": ann, "label": label, "text": text if text is not None else f"text for {item}", **kw}


def checks(result):
    return {i.check for i in result["issues"]}


def clean_rows():
    rows = []
    for n, lab in enumerate(["positive", "negative", "neutral"] * 4):
        for ann in ("a", "b", "c"):
            rows.append(row(f"i{n}", ann, lab, text=f"unique sentence number {n}"))
    return rows


def test_clean_dataset_has_no_issues():
    res = run_dataset_qc(clean_rows())
    assert res["issues"] == []
    assert res["summary"]["fleiss_kappa"] == 1.0


def test_detects_invalid_label_and_empty_text():
    rows = clean_rows() + [row("x1", "a", "postive"), row("x2", "a", "neutral", text="   ")]
    found = checks(run_dataset_qc(rows))
    assert {"invalid_label", "empty_text"} <= found


def test_detects_duplicate_text_ignoring_case_and_punctuation():
    rows = clean_rows() + [row("d1", "a", "neutral", text="Hello, World!"), row("d2", "a", "neutral", text="hello world")]
    assert "duplicate_text" in checks(run_dataset_qc(rows))


def test_detects_duplicate_annotation_and_missing_annotation():
    rows = clean_rows() + [row("i0", "a", "positive", text="unique sentence number 0")]
    rows = [r for r in rows if not (r["item_id"] == "i1" and r["annotator"] == "c")]
    found = checks(run_dataset_qc(rows))
    assert {"duplicate_annotation", "missing_annotation"} <= found


def test_detects_low_gold_accuracy():
    rows = [row(f"g{n}", "a", "positive", text=f"gold {n}", gold_label="negative") for n in range(5)]
    assert "low_gold_accuracy" in checks(run_dataset_qc(rows))


def test_detects_unparseable_and_incomplete_rows():
    rows = clean_rows() + [{"_parse_error": "boom", "_line": 99}, {"item_id": "z", "annotator": "a"}]
    found = checks(run_dataset_qc(rows))
    assert {"parse_errors", "schema_missing_fields"} <= found


def test_detects_no_majority():
    rows = [row("t", "a", "positive"), row("t", "b", "negative"), row("t", "c", "neutral")]
    assert "no_majority" in checks(run_dataset_qc(rows))
