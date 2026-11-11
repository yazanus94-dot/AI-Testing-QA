from qa_suite.output_validation import run_output_validation, validate_output


def rec(output, **constraints):
    return {"id": "t", "task_type": "x", "prompt": "p", "output": output, "constraints": constraints}


def failed(record):
    return {f["check"] for f in validate_output(record)["failures"]}


def test_valid_json_passes():
    assert failed(rec('{"a": 1}', required_keys=["a"], types={"a": "int"})) == set()


def test_json_with_prose_wrapper_fails():
    assert failed(rec('Here: {"a": 1}', required_keys=["a"])) == {"json_format"}


def test_json_wrong_type_and_bool_is_not_int():
    assert failed(rec('{"a": "1"}', required_keys=["a"], types={"a": "int"})) == {"json_format"}
    assert failed(rec('{"a": true}', required_keys=["a"], types={"a": "int"})) == {"json_format"}


def test_content_checks():
    assert failed(rec("Paris is the capital", must_include=["paris"])) == set()
    assert failed(rec("Lyon", must_include=["Paris"])) == {"content"}
    assert failed(rec("Paris", must_not_include=["paris"])) == {"content"}


def test_refusal_detection_handles_curly_apostrophe():
    assert failed(rec("I can\u2019t help with that.", should_refuse=True)) == set()
    assert failed(rec("Sure, here you go.", should_refuse=True)) == {"safety_refusal"}


def test_pii_empty_repetition_length():
    assert failed(rec("mail me at a.b@example.com")) == {"pii"}
    assert failed(rec("call +1 415-555-0199")) == {"pii"}
    assert "non_empty" in failed(rec(""))
    assert failed(rec("Yes. Yes. Yes.")) == {"repetition"}
    assert failed(rec("one two three", max_words=2)) == {"length"}


def test_summary_and_self_check_metrics():
    good = {**rec("ok"), "expected_verdict": "pass"}
    bad = {**rec("", ), "expected_verdict": "fail"}
    missed = {**rec("looks fine but wrong"), "expected_verdict": "fail"}
    out = run_output_validation([good, bad, missed])["summary"]
    assert out["failed"] == 1
    assert out["validator_self_check"]["false_negatives"] == ["t"]
    assert out["validator_self_check"]["recall"] == 0.5
