"""THEORY-4 checks for MV2's learning-rate-regularity project: mastery stopping meets the ceiling.

When practice stops at mastery, each student's window depends on how fast they learn. How does that change what
the straight-line iAFM reports about learning-speed differences on a curve with a ceiling? Worked out before any
phase 1, 2 or 3 fit has run. Standard library only. `python theory4.py --out-dir out` writes out/theory4.md and
out/theory4.csv.

Copyright 2026 Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
Made with AI agents working in the name of MV2 of Munim, Inc.

Notation (one skill): log-odds base + A (1 - exp(-c t)) at opportunity t = 0, 1, ...; x = c t.
- Continuous least squares (LS) slope of A (1 - exp(-c t)) over t in [0, T] is A c S(c T), S(X) = 12 I(X) / X^3,
  I(X) = X/2 - 1 + exp(-X) (1 + X/2).
- eta(X) = X S'(X) / S(X) is the fixed-window ceiling term: THEORY-2b's exponent is e = 1 + eta.
- If the window scales with speed as T ~ c^(-lam), the exponent is e = 1 + eta(c T) (1 - lam).
- Any stopping rule that looks only at answers already given enters the expected log-likelihood of a fitted
  model only through the survival weights pi_t = P(opportunity t is observed). The pseudo-true straight line is
  a pi-weighted fit to the true curve (`wls_slope` on log-odds, `wlogit_fit` on probabilities).
"""
import argparse
import math
import os

Z75 = 0.6744897501960817  # 75th percentile of the standard normal
A_SAT = 2.0               # phase 1 truth T2: total gain in log-odds
TUTOR_L0, TUTOR_T, TUTOR_G, TUTOR_S, TUTOR_THRESHOLD = 0.4, 0.15, 0.2, 0.1, 0.95  # phase 1's BKT mastery rule


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _log1pexp(z):
    if z > 0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


# ---------------------------------------------------------------------------------------------------------------
# Continuous-time least squares on the ceiling curve 1 - exp(-x)
# ---------------------------------------------------------------------------------------------------------------

def I_int(X):
    """I(X) = integral over [0, X] of (x - X/2)(1 - exp(-x)) dx = X/2 - 1 + exp(-X)(1 + X/2)."""
    if X < 0.02:
        return X ** 3 / 12.0 - X ** 4 / 24.0 + X ** 5 / 80.0 - X ** 6 / 360.0 + X ** 7 / 2016.0
    return X / 2.0 - 1.0 + math.exp(-X) * (1.0 + X / 2.0)


def I_prime(X):
    """dI/dX = 1/2 - exp(-X)(1 + X)/2."""
    if X < 0.02:
        return X ** 2 / 4.0 - X ** 3 / 6.0 + X ** 4 / 16.0 - X ** 5 / 60.0 + X ** 6 / 288.0
    return 0.5 - math.exp(-X) * (1.0 + X) / 2.0


def S_ls(X):
    """LS slope of 1 - exp(-x) over x in [0, X], per unit x. S(0) = 1."""
    if X <= 0:
        return 1.0
    return 12.0 * I_int(X) / X ** 3


def eta(X):
    """Elasticity X S'(X) / S(X) = X I'(X) / I(X) - 3. eta(0) = 0; eta = -1 at the reversal point."""
    if X <= 0:
        return 0.0
    return X * I_prime(X) / I_int(X) - 3.0


def exponent_scaled(X, lam):
    """Ceiling exponent d log(slope) / d log(c) when the window T(c) ~ c^(-lam) and X = c T."""
    return 1.0 + eta(X) * (1.0 - lam)


def slope_continuous(c, T, A=1.0):
    """Continuous LS slope (per opportunity) of A (1 - exp(-c t)) over t in [0, T]."""
    return A * c * S_ls(c * T)


def x_star(Lstar, theta, A):
    """Where the curve theta + A (1 - exp(-x)) reaches the absolute level Lstar (needs theta < Lstar < theta + A)."""
    return -math.log(1.0 - (Lstar - theta) / A)


def theta_channel(xs, A):
    """d log(slope) / d theta when practice stops at an absolute level and students differ in intercept theta
    (common gain A): -eta(x*) / (x* A exp(-x*)). Positive: higher prior knowledge reads as faster learning."""
    return -eta(xs) / (xs * A * math.exp(-xs))


def head_start_channel(D):
    """d log(slope) / d x0 for a head start x0 on a common curve when the window is [x0, x*], D = x* - x0:
    -1 - eta(D) / D. Negative: a head start reads as slower learning."""
    return -1.0 - eta(D) / D


# ---------------------------------------------------------------------------------------------------------------
# Discrete opportunities: survival weights of stopping rules
# ---------------------------------------------------------------------------------------------------------------

def survival_run(ps, n=3):
    """pi[t] = P(opportunity t is observed) when practice stops right after the n-th correct answer in a row
    (ASSISTments skill builders; phase 1's `three_in_a_row`). ps[t] = P(correct at t) on the true curve."""
    state = [0.0] * n
    state[0] = 1.0
    pi = []
    for p in ps:
        pi.append(sum(state))
        new = [0.0] * n
        for r, m in enumerate(state):
            if m == 0.0:
                continue
            new[0] += m * (1.0 - p)
            if r + 1 < n:
                new[r + 1] += m * p
        state = new
    return pi


def bkt_update(p_known, correct, T=TUTOR_T, G=TUTOR_G, S=TUTOR_S):
    """Standard BKT: posterior after one response, then the learning transition (as in sim.py)."""
    if correct:
        num = p_known * (1.0 - S)
        den = num + (1.0 - p_known) * G
    else:
        num = p_known * S
        den = num + (1.0 - p_known) * (1.0 - G)
    post = num / den
    return post + (1.0 - post) * T


def survival_bkt(ps, L0=TUTOR_L0, threshold=TUTOR_THRESHOLD):
    """pi[t] when practice stops once the tutor's BKT estimate reaches the threshold (phase 1's `bkt_mastery`).
    Exact over all answer histories (keys are posteriors rounded to 1e-12); use for short caps only."""
    states = {L0: 1.0}
    pi = []
    for p in ps:
        pi.append(sum(states.values()))
        new = {}
        for pk, m in states.items():
            for y, py in ((1, p), (0, 1.0 - p)):
                if py <= 0.0:
                    continue
                q = bkt_update(pk, y)
                if q >= threshold:
                    continue
                key = round(q, 12)
                new[key] = new.get(key, 0.0) + m * py
        states = new
    return pi


def survival_level(Ls, Lstar):
    """Deterministic mastery: opportunity t is observed if t = 0 or the curve was still below Lstar at t - 1."""
    return [1.0 if (t == 0 or Ls[t - 1] < Lstar) else 0.0 for t in range(len(Ls))]


# ---------------------------------------------------------------------------------------------------------------
# Pseudo-true straight lines
# ---------------------------------------------------------------------------------------------------------------

def wls_slope(ts, ys, ws):
    """Weighted least-squares slope of ys on ts. None if the weighted design has no spread."""
    W = sum(ws)
    if W <= 0:
        return None
    tb = sum(w * t for t, w in zip(ts, ws)) / W
    yb = sum(w * y for y, w in zip(ys, ws)) / W
    sxx = sum(w * (t - tb) ** 2 for t, w in zip(ts, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (t - tb) * (y - yb) for t, y, w in zip(ts, ys, ws))
    return sxy / sxx


def _wll(a, b, pts):
    tot = 0.0
    for t, p, w in pts:
        z = a + b * t
        tot += w * (-p * _log1pexp(-z) - (1.0 - p) * _log1pexp(z))
    return tot


def wlogit_fit(ts, ps, ws, iters=200):
    """Pseudo-true logistic line: maximise sum_t w_t [p_t log q_t + (1 - p_t) log(1 - q_t)], q_t = sigmoid(a + b t).
    This is the large-sample limit of a per-student logistic fit when stopping depends only on past answers.
    Returns (a, b) or None."""
    pts = [(t, p, w) for t, p, w in zip(ts, ps, ws) if w > 1e-15]
    if len(set(t for t, _, _ in pts)) < 2:
        return None
    W = sum(w for _, _, w in pts)
    pbar = min(max(sum(w * p for _, p, w in pts) / W, 1e-6), 1.0 - 1e-6)
    a, b = math.log(pbar / (1.0 - pbar)), 0.0
    cur = _wll(a, b, pts)
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for t, p, w in pts:
            q = sigmoid(a + b * t)
            r = w * (p - q)
            v = w * q * (1.0 - q)
            ga += r
            gb += r * t
            haa += v
            hab += v * t
            hbb += v * t * t
        det = haa * hbb - hab * hab
        if det <= 0:
            return None
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        step = 1.0
        while True:
            na, nb = a + step * da, b + step * db
            new = _wll(na, nb, pts)
            if new >= cur - 1e-15 or step < 1e-8:
                break
            step /= 2.0
        a, b, cur = na, nb, new
        if abs(step * da) < 1e-13 and abs(step * db) < 1e-13:
            break
    return a, b


# ---------------------------------------------------------------------------------------------------------------
# Rules and summaries
# ---------------------------------------------------------------------------------------------------------------

RULES = {
    "first5": ("fixed", 5),
    "first10": ("fixed", 10),
    "fixed15": ("fixed", 15),
    "fixed25": ("fixed", 25),
    "level80": ("levelrel", 0.8, 200),   # stop once 80% of the student's own gain is reached (idealised mastery)
    "run3_15": ("run", 3, 15),           # phase 1's three_in_a_row (15 opportunities generated)
    "run3_25": ("run", 3, 25),           # phase 2's t < 25 window on three-in-a-row data
    "run3_200": ("run", 3, 200),         # three in a row, practically uncapped
    "bkt15": ("bkt", 15),                # phase 1's bkt_mastery
}

_CACHE = {}


def weights_for(rule_name, base, A, ct):
    key = (rule_name, round(base, 12), round(A, 12), round(ct, 14))
    if key in _CACHE:
        return _CACHE[key]
    rule = RULES[rule_name]
    kind, cap = rule[0], rule[-1]
    ts = list(range(cap))
    Ls = [base + A * (1.0 - math.exp(-ct * t)) for t in ts]
    ps = [sigmoid(L) for L in Ls]
    if kind == "fixed":
        ws = [1.0] * cap
    elif kind == "levelrel":
        ws = survival_level(Ls, base + rule[1] * A)
    elif kind == "run":
        ws = survival_run(ps, rule[1])
    elif kind == "bkt":
        ws = survival_bkt(ps)
    else:
        raise ValueError(rule_name)
    out = (ts, Ls, ps, ws)
    _CACHE[key] = out
    return out


def slope(rule_name, base, A, ct, method):
    ts, Ls, ps, ws = weights_for(rule_name, base, A, ct)
    if method == "ls":
        return wls_slope(ts, Ls, ws)
    fit = wlogit_fit(ts, ps, ws)
    return None if fit is None else fit[1]


def expected_length(rule_name, base, A, ct):
    return sum(weights_for(rule_name, base, A, ct)[3])


def _pos(*xs):
    return all(x is not None and x > 0 for x in xs)


def effective_exponent(rule_name, base, A, c, method, s=0.5):
    """log(R_est) / log(R_true) for one skill: R = slope at speed exp(+z75 s) over slope at exp(-z75 s)."""
    hi = slope(rule_name, base, A, c * math.exp(Z75 * s), method)
    lo = slope(rule_name, base, A, c * math.exp(-Z75 * s), method)
    if not _pos(hi, lo):
        return None, None
    R = hi / lo
    return R, math.log(R) / (2.0 * Z75 * s)


def window_scaling(rule_name, base, A, c, h=0.25):
    """lam = -d log(expected window length) / d log(c), secant over c exp(-h) .. c exp(+h)."""
    tp = expected_length(rule_name, base, A, c * math.exp(h))
    tm = expected_length(rule_name, base, A, c * math.exp(-h))
    return -(math.log(tp) - math.log(tm)) / (2.0 * h)


def theta_sensitivity(rule_name, base, A, c, method, h=0.25):
    """d log(slope) / d(intercept), secant over base -h .. base +h."""
    bp = slope(rule_name, base + h, A, c, method)
    bm = slope(rule_name, base - h, A, c, method)
    if not _pos(bp, bm):
        return None
    return (math.log(bp) - math.log(bm)) / (2.0 * h)


# Phase 1's T2 population (sim.py): theta ~ N(0, 1), beta ~ N(-0.5, 0.7), c ~ U(0.05, 0.15), A = 2.
GH3 = ((-math.sqrt(3.0), 1.0 / 6.0), (0.0, 2.0 / 3.0), (math.sqrt(3.0), 1.0 / 6.0))
PHASE1_C = (0.05, 0.10, 0.15)


def phase1_pooled(rule_name, method, s=0.5):
    """First-order phase-1-like numbers for truth T2: pooled R_est and exponent over skills and intercepts, and the
    slope spread produced by prior knowledge alone (level 0: every student has the same speed)."""
    sd_base = math.sqrt(1.0 + 0.7 ** 2)
    num = den = 0.0
    for c in PHASE1_C:
        for zn, wn in GH3:
            base = -0.5 + sd_base * zn
            hi = slope(rule_name, base, A_SAT, c * math.exp(Z75 * s), method)
            lo = slope(rule_name, base, A_SAT, c * math.exp(-Z75 * s), method)
            if hi is None or lo is None:
                return None
            num += wn * hi
            den += wn * lo
    R = num / den if den > 0 and num > 0 else None
    e = math.log(R) / (2.0 * Z75 * s) if R else None
    # level 0: student slope = mean over skills (c, beta) of the pseudo-true slope at intercept theta + beta
    ms = []
    for zt, _ in GH3:
        m = 0.0
        for c in PHASE1_C:
            for zb, wb in GH3:
                b = slope(rule_name, zt - 0.5 + 0.7 * zb, A_SAT, c, method)
                if b is None:
                    return None
                m += wb * b / len(PHASE1_C)
        ms.append(m)
    mean_m = sum(w * m for (_, w), m in zip(GH3, ms))
    sd_m = math.sqrt(sum(w * (m - mean_m) ** 2 for (_, w), m in zip(GH3, ms)))
    cov = sum(w * z * (m - mean_m) for (z, w), m in zip(GH3, ms))
    return {"R": R, "e": e, "mean_slope": mean_m, "theta_sd_slope": sd_m, "theta_cov_sign": (cov > 0) - (cov < 0)}


def _f(x, nd=3):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    md = ["# THEORY-4 tables (generated by theory4.py)", "",
          "Made with AI agents working in the name of MV2 of Munim, Inc. CC BY 4.0.", "",
          "## 1. Continuous least squares: e = 1 + eta(X) (1 - lam)", "",
          "| X = c T | S(X) | eta(X) | e, lam = 0 (fixed) | lam = 0.25 | lam = 0.5 | lam = 0.75 | lam = 1 (level) |",
          "|---|---|---|---|---|---|---|---|"]
    for X in (0.25, 0.5, 1.0, 1.25, 2.0, 2.69, 3.0, 4.0, 6.0):
        md.append("| %.2f | %.4f | %.4f | %s |" % (X, S_ls(X), eta(X), " | ".join(
            "%.3f" % exponent_scaled(X, lam) for lam in (0.0, 0.25, 0.5, 0.75, 1.0))))
    md += ["", "## 2. Prior-knowledge channels under stopping (continuous LS, A = 2)", "",
           "| x* or D | theta channel d log b / d theta (absolute level) | head-start channel d log b / d x0 |",
           "|---|---|---|"]
    for X in (0.5, 1.0, 1.25, 1.609, 2.0, 3.0):
        md.append("| %.3f | %.4f | %.4f |" % (X, theta_channel(X, 2.0), head_start_channel(X)))
    rows = []
    md += ["", "## 3. One skill, A = 2, true speed ratio 1.96 (s = 0.5)", "",
           "e = log R_est / log 1.96. lam = window scaling (1 = window shrinks in proportion to speed, 0 = fixed).",
           "dth = d log(slope) / d(intercept), logistic pseudo-true fit.", "",
           "| rule | c | intercept | E[length] | lam | R (ls) | e (ls) | R (logit) | e (logit) | dth (logit) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for rule in ("fixed15", "fixed25", "level80", "run3_25", "run3_200", "bkt15"):
        for c in (0.05, 0.10, 0.15, 0.25, 0.40):
            for base in (-1.5, -0.5, 0.5):
                Rl, el = effective_exponent(rule, base, A_SAT, c, "ls")
                Rg, eg = effective_exponent(rule, base, A_SAT, c, "logit")
                lam = window_scaling(rule, base, A_SAT, c)
                dth = theta_sensitivity(rule, base, A_SAT, c, "logit")
                T = expected_length(rule, base, A_SAT, c)
                rows.append((rule, c, base, T, lam, Rl, el, Rg, eg, dth))
                md.append("| %s | %.2f | %.1f | %.2f | %.3f | %s | %s | %s | %s | %s |" % (
                    rule, c, base, T, lam, _f(Rl), _f(el), _f(Rg), _f(eg), _f(dth)))
    md += ["", "## 4. Phase-1-like first-order values (truth T2, skills c = 0.05/0.10/0.15, intercepts pooled)", "",
           "R and e at level 0.5 (true ratio 1.96). theta SD = spread of the student slope from prior knowledge "
           "alone at level 0; sign = sign of its covariance with prior knowledge.", "",
           "| rule | method | R | e | mean slope | theta SD (level 0) | sign |", "|---|---|---|---|---|---|---|"]
    for rule in ("first5", "first10", "fixed15", "run3_15", "bkt15"):
        for method in ("ls", "logit"):
            r = phase1_pooled(rule, method)
            if r is None:
                md.append("| %s | %s | n/a | n/a | n/a | n/a | n/a |" % (rule, method))
                continue
            md.append("| %s | %s | %s | %s | %s | %s | %+d |" % (
                rule, method, _f(r["R"]), _f(r["e"]), _f(r["mean_slope"], 4), _f(r["theta_sd_slope"], 4),
                r["theta_cov_sign"]))
    text = "\n".join(md) + "\n"
    with open(os.path.join(args.out_dir, "theory4.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out_dir, "theory4.csv"), "w", encoding="utf-8") as fh:
        fh.write("rule,c,intercept,expected_length,lam,R_ls,e_ls,R_logit,e_logit,dtheta_logit\n")
        for row in rows:
            fh.write(",".join("" if v is None else (v if isinstance(v, str) else repr(round(v, 6))) for v in row) + "\n")
    print(text)


if __name__ == "__main__":
    main()
