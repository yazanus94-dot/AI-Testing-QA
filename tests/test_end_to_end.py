import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_qa  # noqa: E402


def test_pipeline_on_sample_data_catches_planted_defects(tmp_path):
    code = run_qa.main(["--data-dir", str(ROOT / "data"), "--report-dir", str(tmp_path), "--fail-on-gate"])
    assert code == 1
    report = json.loads((tmp_path / "qa_report.json").read_text())
    assert report["gate"] == "FAIL"
    checks = {i["check"] for i in report["dataset_qc"]["issues"]}
    assert {"invalid_label", "empty_text", "duplicate_text", "missing_annotation"} <= checks
    rag_checks = {i["check"] for i in report["rag_groundedness"]["issues"]}
    assert rag_checks
    rob_checks = {i["check"] for i in report["prompt_robustness"]["issues"]}
    assert rob_checks


def test_pipeline_does_not_fail_gate_without_flag(tmp_path):
    assert run_qa.main(["--data-dir", str(ROOT / "data"), "--report-dir", str(tmp_path)]) == 0
