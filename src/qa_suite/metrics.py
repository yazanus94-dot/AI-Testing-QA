"""Agreement and statistics helpers (pure Python, no dependencies)."""
from __future__ import annotations

import math
from collections import Counter


def cohens_kappa(r1: list, r2: list) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(r1) != len(r2) or not r1:
        raise ValueError("Rating lists must be non-empty and equal length")
    n = len(r1)
    po = sum(a == b for a, b in zip(r1, r2)) / n
    c1, c2 = Counter(r1), Counter(r2)
    pe = sum((c1[c] / n) * (c2[c] / n) for c in set(c1) | set(c2))
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list]) -> float:
    """Fleiss' kappa. `ratings` = one list per item, each with the same number of rater labels."""
    n_items = len(ratings)
    if n_items < 2:
        raise ValueError("Need at least 2 items")
    n_raters = len(ratings[0])
    if n_raters < 2 or any(len(r) != n_raters for r in ratings):
        raise ValueError("Every item needs the same number (>=2) of ratings")
    cats = sorted({c for r in ratings for c in r}, key=str)
    p_cat = {c: 0.0 for c in cats}
    p_items = []
    for r in ratings:
        counts = Counter(r)
        for c in cats:
            p_cat[c] += counts[c]
        p_items.append((sum(v * v for v in counts.values()) - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(p_items) / n_items
    pe = sum((v / (n_items * n_raters)) ** 2 for v in p_cat.values())
    if pe == 1:
        return 1.0
    return (p_bar - pe) / (1 - pe)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) interpretation."""
    if k < 0:
        return "poor"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
