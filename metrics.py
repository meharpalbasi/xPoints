"""Ranking and error metrics for scoring archived predictions against results.

Stdlib only, deliberately: the scorecard must run even when the ML stack is
the thing that broke. Every function takes plain lists.
"""
import math
import random


def average_ranks(values):
    """1-based DESCENDING ranks with ties given their average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def spearman(pred, actual):
    """Rank correlation with average ranks for ties (tie-safe)."""
    return pearson(average_ranks(pred), average_ranks(actual))


def bootstrap_spearman_ci(pred, actual, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for Spearman. Seeded for reproducibility."""
    n = len(pred)
    if n < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = spearman([pred[i] for i in idx], [actual[i] for i in idx])
        if not math.isnan(s):
            stats.append(s)
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return (lo, hi)


def mae(pred, actual):
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(pred)


def rmse(pred, actual):
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(pred))


def _top_k_random_ties(values, k, rng):
    """Indices of the top k by value, breaking ties at random."""
    keyed = sorted(range(len(values)), key=lambda i: (-values[i], rng.random()))
    return set(keyed[:k])


def precision_at_k(pred, actual, k, draws=200, seed=0):
    """Share of the predicted top-k that are in the actual top-k.

    Ties are broken at random and averaged over `draws`, because a stable
    argsort would silently leak whatever order the file happened to be in.
    """
    n = len(pred)
    if n == 0 or k <= 0:
        return float("nan")
    k = min(k, n)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(draws):
        p_top = _top_k_random_ties(pred, k, rng)
        a_top = _top_k_random_ties(actual, k, rng)
        total += len(p_top & a_top) / k
    return total / draws


def captain_regret(pred, actual, draws=200, seed=0):
    """Best actual score minus the actual score of the predicted top player.

    The captaincy question is a max problem, not a mean problem. Ties in the
    prediction are broken at random and averaged.
    """
    if not pred:
        return float("nan")
    best = max(actual)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(draws):
        i = next(iter(_top_k_random_ties(pred, 1, rng)))
        total += best - actual[i]
    return total / draws


def gameweeks_to_detect(effect, sd, power=0.8, alpha=0.05):
    """Gameweeks needed for a two-sided test of a mean per-gameweek
    improvement `effect` given per-gameweek sd, one-sample normal approx:
    n = ((z_{1-a/2} + z_power) * sd / effect)^2.
    """
    if effect <= 0 or sd <= 0:
        return float("nan")
    z_alpha = {0.05: 1.960, 0.01: 2.576}[alpha]
    z_power = {0.8: 0.842, 0.9: 1.282}[power]
    return math.ceil(((z_alpha + z_power) * sd / effect) ** 2)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
