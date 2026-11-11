from qa_suite.rlhf_eval import run_rlhf_eval

LONG = "this is a much longer response with many many extra words in it for testing"
SHORT = "short answer"


def make(pair, ann, choice, a=SHORT, b=LONG, src=("m1", "m2")):
    return {"pair_id": pair, "prompt": "p", "response_a": a, "response_b": b,
            "source_a": src[0], "source_b": src[1], "annotator": ann, "choice": choice}


def flags(res, ann):
    return res["summary"]["annotators_detail"][ann]["flags"]


def test_detects_position_bias():
    rows = [make(f"p{i}", "x", "A") for i in range(10)]
    assert "position_bias" in flags(run_rlhf_eval(rows), "x")


def test_detects_length_bias():
    rows = []
    for i in range(10):
        a, b = (SHORT, LONG) if i % 2 else (LONG, SHORT)
        rows.append(make(f"p{i}", "x", "B" if i % 2 else "A", a=a, b=b))  # always picks the longer one
    res = run_rlhf_eval(rows)
    assert "length_bias" in flags(res, "x")
    assert "position_bias" not in flags(res, "x")


def test_balanced_annotator_not_flagged():
    rows = []
    for i in range(10):
        a, b = (SHORT, LONG) if i % 2 else (LONG, SHORT)
        rows.append(make(f"p{i}", "x", "A" if i % 4 < 2 else "B", a=a, b=b))
    assert flags(run_rlhf_eval(rows), "x") == []


def test_invalid_choice_reported_and_excluded():
    res = run_rlhf_eval([make("p1", "x", "C"), make("p2", "x", "A")])
    assert "invalid_preference_row" in {i.check for i in res["issues"]}
    assert res["summary"]["n_pairs"] == 1


def test_win_rate_uses_majority_vote():
    rows = []
    for ann in ("a", "b", "c"):
        rows.append(make("p1", ann, "A"))                      # m1 wins
        rows.append(make("p2", ann, "B", src=("m1", "m2")))    # m2 wins
        rows.append(make("p3", ann, "A"))                      # m1 wins
    wr = run_rlhf_eval(rows)["summary"]["win_rates"]
    assert wr["m1"]["wins"] == 2 and wr["m2"]["wins"] == 1


def test_no_consensus_pair_reported():
    rows = [make("p1", "a", "A"), make("p1", "b", "B")]
    assert "no_consensus_pairs" in {i.check for i in run_rlhf_eval(rows)["issues"]}
