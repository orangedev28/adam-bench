"""Statistics used in analyze.py; needs only the Python standard library.
Uses scipy if available, otherwise the built-in implementations (checked against scipy)."""
import math
import statistics as st

try:
    from scipy import stats as _sp
except Exception:  # pragma: no cover
    _sp = None


# ------------------------------------------------- special functions (fallback)
def _betacf(a, b, x, itmax=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def gammaincc(a, x):
    """Regularized upper incomplete gamma function Q(a, x)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:  # series
        ap, s, d = a, 1 / a, 1 / a
        for _ in range(1000):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 3e-14:
                break
        return 1 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d  # continued fraction
    for i in range(1, 1000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < 3e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def t_sf2(t, df):
    """Two-sided p-value of the Student t distribution."""
    if _sp is not None:
        return float(2 * _sp.t.sf(abs(t), df))
    return betainc(df / 2, 0.5, df / (df + t * t))


def t_ppf975(df):
    """Critical value t_{0.975, df} (95% confidence interval)."""
    if _sp is not None:
        return float(_sp.t.ppf(0.975, df))
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_sf(x, df):
    if _sp is not None:
        return float(_sp.chi2.sf(x, df))
    return gammaincc(df / 2, x / 2)


# --------------------------------------------------------------- utilities
def clean(v):
    return [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]


def mean_ci(v):
    """(mean, std, half-width of the 95% CI, n)."""
    v = clean(v)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = st.mean(v)
    if n == 1:
        return m, 0.0, float("nan"), 1
    s = st.stdev(v)
    return m, s, t_ppf975(n - 1) * s / math.sqrt(n), n


def paired_t(a, b):
    """Paired t-test (a - b). a, b: dict seed -> value. Returns (mean difference, p, n)."""
    seeds = sorted(set(a) & set(b))
    d = clean([a[s] - b[s] for s in seeds if a[s] is not None and b[s] is not None])
    n = len(d)
    if n < 2:
        return (d[0] if d else float("nan")), float("nan"), n
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        return m, (1.0 if m == 0 else 0.0), n
    t = m / (sd / math.sqrt(n))
    return m, t_sf2(t, n - 1), n


def holm(pvals):
    """Holm-Bonferroni correction. pvals: dict key -> p. Returns dict key -> adjusted p."""
    items = sorted(((p, k) for k, p in pvals.items() if p == p), key=lambda t: t[0])
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    for k, p in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def ranks_desc(values):
    """Ranks (1 = best = largest value); ties receive the average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


# q_{0.05}(k, inf) of the studentized range distribution; Nemenyi uses q/sqrt(2) (Demsar 2006).
_Q05 = {2: 2.772, 3: 3.314, 4: 3.633, 5: 3.858, 6: 4.030, 7: 4.170, 8: 4.286, 9: 4.387,
        10: 4.474, 11: 4.552, 12: 4.622, 13: 4.685, 14: 4.743, 15: 4.796, 16: 4.845,
        17: 4.891, 18: 4.934, 19: 4.974, 20: 5.012}


def friedman_nemenyi(blocks):
    """blocks: list of lists of values (each block lists the k methods in the same order).
    Returns (avg_rank, chi2_F, p_Friedman, Nemenyi CD at alpha=0.05)."""
    N, k = len(blocks), len(blocks[0])
    R = [0.0] * k
    for b in blocks:
        for i, r in enumerate(ranks_desc(b)):
            R[i] += r
    avg = [r / N for r in R]
    chi2 = 12 * N / (k * (k + 1)) * (sum(a * a for a in avg) - k * (k + 1) ** 2 / 4)
    p = chi2_sf(chi2, k - 1)
    cd = _Q05[k] / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * N)) if k in _Q05 else float("nan")
    return avg, chi2, p, cd
