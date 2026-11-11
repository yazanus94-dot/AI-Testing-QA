"""Quality control for labeled / annotated datasets (one row = one annotation)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

from .common import Issue
from .metrics import cohens_kappa, fleiss_kappa, interpret_kappa

ALLOWED_LABELS = ("positive", "negative", "neutral")
REQUIRED_FIELDS = ("item_id", "text", "annotator", "label")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def run_dataset_qc(rows, allowed_labels=ALLOWED_LABELS, min_kappa=0.60,
                   min_gold_accuracy=0.85, min_class_share=0.15) -> dict:
    issues: list[Issue] = []

    # 1. Parsing + schema ----------------------------------------------------
    parse_errors = [r["_line"] for r in rows if "_parse_error" in r]
    if parse_errors:
        issues.append(Issue("parse_errors", "critical", f"{len(parse_errors)} unparseable line(s)", parse_errors))
    records = [r for r in rows if "_parse_error" not in r]
    incomplete = [str(r.get("item_id", f"row#{i}")) for i, r in enumerate(records)
                  if any(r.get(f) is None for f in REQUIRED_FIELDS)]
    if incomplete:
        issues.append(Issue("schema_missing_fields", "critical",
                            f"{len(incomplete)} row(s) missing required fields {REQUIRED_FIELDS}", incomplete))
    records = [r for r in records if all(r.get(f) is not None for f in REQUIRED_FIELDS)]

    # 2. Label validity, empty text, duplicate annotations ------------------
    invalid = [f'{r["item_id"]}/{r["annotator"]}={r["label"]!r}' for r in records if r["label"] not in allowed_labels]
    if invalid:
        issues.append(Issue("invalid_label", "major",
                            f"{len(invalid)} annotation(s) use a label outside {list(allowed_labels)}", invalid))
    empty = sorted({r["item_id"] for r in records if not str(r["text"]).strip()})
    if empty:
        issues.append(Issue("empty_text", "major", f"{len(empty)} item(s) have empty text", empty))
    seen = Counter((r["item_id"], r["annotator"]) for r in records)
    dup_ann = [f"{i}/{a}" for (i, a), c in seen.items() if c > 1]
    if dup_ann:
        issues.append(Issue("duplicate_annotation", "major",
                            f"{len(dup_ann)} annotator/item pair(s) labeled more than once", dup_ann))

    # 3. Duplicate texts (leakage / over-weighting) --------------------------
    text_ids = defaultdict(set)
    for r in records:
        t = _norm(str(r["text"]))
        if t:
            text_ids[t].add(r["item_id"])
    dup_text = ["=".join(sorted(ids)) for ids in text_ids.values() if len(ids) > 1]
    if dup_text:
        issues.append(Issue("duplicate_text", "major",
                            f"{len(dup_text)} text(s) appear under multiple item_ids", dup_text))

    # 4. Completeness --------------------------------------------------------
    annotators = sorted({r["annotator"] for r in records})
    present = defaultdict(set)
    for r in records:
        present[r["item_id"]].add(r["annotator"])
    missing = [f"{i}:{'/'.join(sorted(set(annotators) - a))}" for i, a in sorted(present.items())
               if set(annotators) - a]
    if missing:
        issues.append(Issue("missing_annotation", "minor",
                            f"{len(missing)} item(s) not labeled by every annotator", missing))

    # 5. Analysis on valid rows ---------------------------------------------
    valid = [r for r in records if r["label"] in allowed_labels]
    by_item = defaultdict(dict)
    for r in valid:
        by_item[r["item_id"]][r["annotator"]] = r["label"]

    dist = Counter(r["label"] for r in valid)
    total = sum(dist.values()) or 1
    shares = {k: round(v / total, 3) for k, v in dist.items()}
    skewed = [f"{k}={v:.0%}" for k, v in shares.items() if v < min_class_share]
    if skewed:
        issues.append(Issue("class_imbalance", "minor", f"Class share below {min_class_share:.0%}", skewed))

    no_majority = []
    for item, votes in sorted(by_item.items()):
        top = Counter(votes.values()).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            no_majority.append(item)
    if no_majority:
        issues.append(Issue("no_majority", "minor",
                            f"{len(no_majority)} item(s) have no majority label and need adjudication", no_majority))

    pair_kappas = {}
    for a, b in combinations(annotators, 2):
        shared = [i for i, v in by_item.items() if a in v and b in v]
        if len(shared) >= 2:
            k = cohens_kappa([by_item[i][a] for i in shared], [by_item[i][b] for i in shared])
            pair_kappas[f"{a}~{b}"] = {"kappa": round(k, 3), "n": len(shared)}

    complete = [list(v.values()) for v in by_item.values() if len(v) == len(annotators)]
    fleiss = round(fleiss_kappa(complete), 3) if len(complete) >= 2 and len(annotators) >= 2 else None
    if fleiss is not None and fleiss < min_kappa:
        issues.append(Issue("low_overall_agreement", "major",
                            f"Fleiss' kappa {fleiss} ({interpret_kappa(fleiss)}) is below target {min_kappa}"))

    mean_kappa = {}
    for a in annotators:
        ks = [v["kappa"] for p, v in pair_kappas.items() if a in p.split("~")]
        if ks:
            mean_kappa[a] = round(sum(ks) / len(ks), 3)
    low_ann = [f"{a} (mean kappa {k})" for a, k in mean_kappa.items() if k < min_kappa]
    if low_ann:
        issues.append(Issue("low_annotator_agreement", "major",
                            f"Annotator(s) below mean pairwise kappa {min_kappa}", low_ann))

    gold = defaultdict(lambda: [0, 0])
    for r in valid:
        if r.get("gold_label"):
            gold[r["annotator"]][1] += 1
            gold[r["annotator"]][0] += r["label"] == r["gold_label"]
    gold_acc = {a: {"correct": c, "total": t, "accuracy": round(c / t, 3)} for a, (c, t) in sorted(gold.items())}
    weak = [f"{a} ({v['accuracy']:.0%})" for a, v in gold_acc.items() if v["accuracy"] < min_gold_accuracy]
    if weak:
        issues.append(Issue("low_gold_accuracy", "major",
                            f"Annotator(s) below {min_gold_accuracy:.0%} accuracy on gold set", weak))

    summary = {
        "n_rows": len(rows), "n_items": len(present), "annotators": annotators,
        "label_distribution": dict(dist), "class_shares": shares,
        "fleiss_kappa": fleiss, "fleiss_interpretation": interpret_kappa(fleiss) if fleiss is not None else None,
        "pairwise_kappa": pair_kappas, "annotator_mean_kappa": mean_kappa,
        "gold_accuracy": gold_acc, "items_needing_adjudication": len(no_majority),
        "thresholds": {"min_kappa": min_kappa, "min_gold_accuracy": min_gold_accuracy, "min_class_share": min_class_share},
    }
    return {"summary": summary, "issues": issues}
