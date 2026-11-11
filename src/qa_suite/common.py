"""Shared helpers: JSONL loading and the Issue record used by every QA module."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEVERITY_ORDER = ("critical", "major", "minor", "info")


@dataclass
class Issue:
    check: str
    severity: str  # critical | major | minor | info
    message: str
    items: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_jsonl(path) -> list[dict]:
    """Load a JSONL file. Unparseable lines are kept as {'_parse_error', '_line'} so QA can report them."""
    rows: list[dict] = []
    with open(Path(path), encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_parse_error": str(exc), "_line": n})
    return rows


def count_by_severity(issues: list[Issue]) -> dict[str, int]:
    return {s: sum(1 for i in issues if i.severity == s) for s in SEVERITY_ORDER}
