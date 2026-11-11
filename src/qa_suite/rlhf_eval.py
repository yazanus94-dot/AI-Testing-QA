"""Evaluation of RLHF-style pairwise preference data (human feedback on two model responses)."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from .common import Issue
from .metrics import cohens_kappa, fleiss_kappa, interpret_kappa, wilson_interval

REQUIRED = ("pair_id", "prompt", "response_a", "response_b", "source_a", "source_b", "annotator", "choice")
CHOICES = ("A", "B", "tie")


def _words(t: str) -> int:
    return len(t.split())


def run_rlhf_eval(rows, min_kappa=0.40, position_band=(0.35, 0.65), max_longer_share=0.70) -> dict:
    issues: list[Issue] = []
    bad = [f'{r.get("pair_id", "?")}/{r.get("annotator", "?")}={r.get("choice")!r}' for r in rows
           if any(r.get(f) is None for f in REQUIRED) or r.get("choice") not in CHOICES]
    if bad:
        issues.append(Issue("invalid_preference_row", "major",
                            f"{len(bad)} row(s) with missing fields or choice outside {list(CHOICES)}", bad))
    valid = [r for r in rows if not any(r.get(f) is None for f in REQUIRED) and r["choice"] in CHOICES]

    pairs, meta = defaultdict(dict), {}
    for r in valid:
        pairs[r["pair_id"]][r["annotator"]] = r["choice"]
        meta[r["pair_id"]] = r
    annotators = sorted({r["annotator"] for r in valid})

    consensus = {}
    for pid, votes in pairs.items():
        top = Counter(votes.values()).most_common()
        consensus[pid] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None

    wins, ties, no_consensus = Counter(), 0, 0
    for pid, c in consensus.items():
        if c is None:
            no_consensus += 1
        elif c == "tie":
            ties += 1
        else:
            wins[meta[pid]["source_a" if c == "A" else "source_b"]] += 1
    decided = sum(wins.values())
    win_rates = {}
    for src in sorted({r["source_a"] for r in valid} | {r["source_b"] for r in valid}):
        lo, hi = wilson_interval(wins[src], decided)
        win_rates[src] = {"wins": wins[src], "decided_pairs": decided,
                          "win_rate": round(wins[src] / decided, 3) if decided else None,
                          "ci95": [round(lo, 3), round(hi, 3)]}

    ann_stats = {}
    for a in annotators:
        mine = [r for r in valid if r["annotator"] == a]
        non_tie = [r for r in mine if r["choice"] != "tie"]
        pct_a = sum(r["choice"] == "A" for r in non_tie) / len(non_tie) if non_tie else None
        diff = [r for r in non_tie if _words(r["response_a"]) != _words(r["response_b"])]
        longer = sum((r["choice"] == "A") == (_words(r["response_a"]) > _words(r["response_b"])) for r in diff)
        pct_longer = longer / len(diff) if diff else None
        with_c = [r for r in mine if consensus.get(r["pair_id"]) is not None]
        agree = sum(r["choice"] == consensus[r["pair_id"]] for r in with_c) / len(with_c) if with_c else None
        flags = []
        if pct_a is not None and not (position_band[0] <= pct_a <= position_band[1]):
            flags.append("position_bias")
        if pct_longer is not None and pct_longer > max_longer_share:
            flags.append("length_bias")
        ann_stats[a] = {"n": len(mine),
                        "pct_choose_A": None if pct_a is None else round(pct_a, 3),
                        "pct_choose_longer": None if pct_longer is None else round(pct_longer, 3),
                        "agreement_with_majority": None if agree is None else round(agree, 3),
                        "flags": flags}
    for a, st in ann_stats.items():
        for f in st["flags"]:
            msg = ("chooses response A %.0f%% of the time (expected ~50%%)" % (st["pct_choose_A"] * 100)
                   if f == "position_bias" else
                   "picks the longer response %.0f%% of the time (verbosity bias)" % (st["pct_choose_longer"] * 100))
            issues.append(Issue(f, "major", f"{a} {msg}", [a]))

    pair_kappas = {}
    for a, b in combinations(annotators, 2):
        shared = [p for p, v in pairs.items() if a in v and b in v]
        if len(shared) >= 2:
            pair_kappas[f"{a}~{b}"] = {"kappa": round(cohens_kappa([pairs[p][a] for p in shared],
                                                                    [pairs[p][b] for p in shared]), 3), "n": len(shared)}
    complete = [list(v.values()) for v in pairs.values() if len(v) == len(annotators)]
    fleiss = round(fleiss_kappa(complete), 3) if len(complete) >= 2 and len(annotators) >= 2 else None
    if fleiss is not None and fleiss < min_kappa:
        issues.append(Issue("low_preference_agreement", "major",
                            f"Fleiss' kappa {fleiss} ({interpret_kappa(fleiss)}) is below target {min_kappa}"))
    if no_consensus:
        issues.append(Issue("no_consensus_pairs", "minor",
                            f"{no_consensus} pair(s) have no majority preference; send to expert adjudication",
                            [p for p, c in consensus.items() if c is None]))

    return {"summary": {"n_rows": len(rows), "n_pairs": len(pairs), "annotators": annotators,
                        "win_rates": win_rates, "tie_pairs": ties, "no_consensus_pairs": no_consensus,
                        "fleiss_kappa": fleiss,
                        "fleiss_interpretation": interpret_kappa(fleiss) if fleiss is not None else None,
                        "pairwise_kappa": pair_kappas, "annotators_detail": ann_stats,
                        "thresholds": {"min_kappa": min_kappa, "position_band": list(position_band),
                                       "max_longer_share": max_longer_share}},
            "issues": issues}
