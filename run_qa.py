#!/usr/bin/env python3
"""Run the full AI QA suite and write reports/qa_report.md + reports/qa_report.json.

Usage:
    python run_qa.py                       # run on ./data, write ./reports
    python run_qa.py --fail-on-gate        # exit 1 if the quality gate is FAIL (for CI/CD)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qa_suite.common import count_by_severity, load_jsonl  # noqa: E402
from qa_suite.dataset_qc import run_dataset_qc  # noqa: E402
from qa_suite.output_validation import run_output_validation  # noqa: E402
from qa_suite.rag_eval import run_rag_eval  # noqa: E402
from qa_suite.report import build_report, gate_status  # noqa: E402
from qa_suite.rlhf_eval import run_rlhf_eval  # noqa: E402
from qa_suite.robustness_eval import run_robustness_eval  # noqa: E402


def _load(path: Path):
    return load_jsonl(path) if path.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--fail-on-gate", action="store_true", help="exit with code 1 when the gate is FAIL")
    args = ap.parse_args(argv)

    data, out = Path(args.data_dir), Path(args.report_dir)
    ds = run_dataset_qc(_load(data / "annotations.jsonl"))
    ov = run_output_validation(_load(data / "model_outputs.jsonl"))
    rl = run_rlhf_eval(_load(data / "preferences.jsonl"))
    rg = run_rag_eval(_load(data / "rag_eval.jsonl"))
    rb = run_robustness_eval(_load(data / "robustness_eval.jsonl"))

    issues = ds["issues"] + rl["issues"] + rg.get("issues", []) + rb.get("issues", [])

    out.mkdir(exist_ok=True, parents=True)
    (out / "qa_report.md").write_text(build_report(ds, ov, rl, rg, rb), encoding="utf-8")
    (out / "qa_report.json").write_text(json.dumps({
        "gate": gate_status(issues),
        "issue_counts": count_by_severity(issues),
        "dataset_qc": {"summary": ds["summary"], "issues": [i.to_dict() for i in ds["issues"]]},
        "output_validation": ov,
        "rlhf": {"summary": rl["summary"], "issues": [i.to_dict() for i in rl["issues"]]},
        "rag_groundedness": {"summary": rg["summary"], "issues": [i.to_dict() for i in rg.get("issues", [])], "results": rg.get("results", [])},
        "prompt_robustness": {"summary": rb["summary"], "issues": [i.to_dict() for i in rb.get("issues", [])], "results": rb.get("results", [])},
    }, indent=2), encoding="utf-8")

    print(f"Quality gate: {gate_status(issues)}  |  issues: {count_by_severity(issues)}")
    print(f"Report written to {out / 'qa_report.md'}")
    return 1 if (args.fail_on_gate and gate_status(issues) == "FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
