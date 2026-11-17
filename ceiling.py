# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity, THEORY-2b: how a learning ceiling shrinks the iAFM's learning-rate spread.
# Made with AI agents working in the name of MV2 of Munim, Inc. Standard library only.
#
# Setting (first order, see THEORY-2b.md): a student's log-odds on a skill rise as A * (1 - exp(-k * r * t)),
# t = 0, 1, ..., W-1 opportunities, where k is the skill's rate constant and r = exp(s * z) is the student's
# speed (z standard normal). A straight line in t (the iAFM's shape) fitted by least squares over the window
# has slope A * b(k * r, W). The student-to-student spread of those slopes, relative to their mean, is
# CV ~= s * e(k, W), where e = d log b / d log k is the "ceiling exponent". Amplitude differences
# (A_i varying instead of r_i) give e = 1 in every window.
import argparse
import json
import math
import os
import sys

Z75 = 0.6744897501960817  # 75th percentile of the standard normal
PHASE1_KS = (0.05, 0.10, 0.15)  # low, middle and high T2 rate constants of phase 1 (uniform 0.05..0.15)


def ls_slope(values):
    """Least-squares slope of values on t = 0..n-1."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least two points")
    tbar = (n - 1) / 2.0
    s = sum((t - tbar) ** 2 for t in range(n))
    return sum((t - tbar) * v for t, v in enumerate(values)) / s


def curve_slope(k, W):
    """b(k, W): least-squares slope of 1 - exp(-k t) over t = 0..W-1 (per unit of the ceiling A)."""
    return ls_slope([1.0 - math.exp(-k * t) for t in range(W)])


def curve_slope_dk(k, W):
    """d b / d k = least-squares slope of t * exp(-k t)."""
    return ls_slope([t * math.exp(-k * t) for t in range(W)])


def elasticity(k, W):
    """Ceiling exponent e(k, W) = d log b / d log k. 1 for a straight line, 0 at the reversal point."""
    if k <= 0:
        raise ValueError("k must be positive")
    return k * curve_slope_dk(k, W) / curve_slope(k, W)


def pooled_elasticity(ks, W):
    """Exponent of a student slope pooled over skills with rate constants ks: mean(k b') / mean(b)."""
    num = sum(k * curve_slope_dk(k, W) for k in ks) / len(ks)
    den = sum(curve_slope(k, W) for k in ks) / len(ks)
    return num / den


def normal_grid(n=1201, lo=-6.0, hi=6.0):
    """Points and weights (summing to 1) for an expectation over a standard normal z."""
    step = (hi - lo) / (n - 1)
    zs = [lo + i * step for i in range(n)]
    ws = [math.exp(-0.5 * z * z) for z in zs]
    tot = sum(ws)
    return zs, [w / tot for w in ws]


def student_moments(ks, W, level, grid=None):
    """Mean, SD and CV over students (z ~ N(0,1), r = exp(level z)) of the pooled slope mean_k b(k r, W).
    Exact for the deterministic curve; ignores binomial noise, logistic weighting and random-effect shrinkage."""
    zs, ws = grid or normal_grid()
    vals = []
    for z in zs:
        r = math.exp(level * z)
        vals.append(sum(curve_slope(k * r, W) for k in ks) / len(ks))
    mean = sum(w * v for w, v in zip(ws, vals))
    var = sum(w * (v - mean) ** 2 for w, v in zip(ws, vals))
    sd = math.sqrt(max(var, 0.0))
    return {"mean": mean, "sd": sd, "cv": sd / mean if mean > 0 else None}


def r_est_from_cv(cv):
    """The PNAS-style 75th/25th percentile speed ratio from CV = SD / mean (None if unbounded)."""
    if cv is None:
        return None
    lo = 1.0 - Z75 * cv
    if lo <= 0:
        return None
    return (1.0 + Z75 * cv) / lo


def cv_from_r(ratio):
    """Inverse of r_est_from_cv."""
    return (ratio - 1.0) / ((ratio + 1.0) * Z75)


def true_ratio(level):
    """75th/25th percentile ratio of true speed r = exp(level z)."""
    return math.exp(2.0 * Z75 * level)


def implied_true_ratio(observed_ratio, e):
    """True speed ratio that would show up as observed_ratio if the slope CV is s * e (first order)."""
    if e <= 0:
        return None
    s = cv_from_r(observed_ratio) / e
    return math.exp(2.0 * Z75 * s)


def zero_crossing(W, lo=1e-3, hi=5.0, iters=200):
    """Rate constant k at which b(k, W) peaks, i.e. e(k, W) = 0 (None if there is no sign change)."""
    flo, fhi = curve_slope_dk(lo, W), curve_slope_dk(hi, W)
    if flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = curve_slope_dk(mid, W)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def continuous_root(iters=200):
    """Continuous-time limit: e = 0 where 4 - u = exp(-u) (u^2 + 3u + 4), u = k * (window span); u ~ 2.69."""
    def f(u):
        return 4.0 - u - math.exp(-u) * (u * u + 3.0 * u + 4.0)
    lo, hi = 1.0, 4.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_tables():
    windows = (5, 10, 15, 25)
    ks = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    tab = {"exponent": [], "pooled": [], "phase1_T2": [], "implied": [], "reversal": {}}
    for W in windows:
        tab["exponent"].append({"W": W, "e": {str(k): elasticity(k, W) for k in ks},
                                "b": {str(k): curve_slope(k, W) for k in ks}})
        tab["pooled"].append({"W": W, "e_pooled_phase1": pooled_elasticity(PHASE1_KS, W)})
        tab["reversal"][str(W)] = zero_crossing(W)
    tab["reversal"]["continuous_u"] = continuous_root()
    grid = normal_grid()
    for level in (0.15, 0.3, 0.5):
        m5 = student_moments(PHASE1_KS, 5, level, grid)
        m10 = student_moments(PHASE1_KS, 10, level, grid)
        m15 = student_moments(PHASE1_KS, 15, level, grid)
        tab["phase1_T2"].append({
            "level": level, "true_ratio": true_ratio(level),
            "first_order_cv15": level * pooled_elasticity(PHASE1_KS, 15),
            "cv5": m5["cv"], "cv10": m10["cv"], "cv15": m15["cv"],
            "r_est5": r_est_from_cv(m5["cv"]), "r_est15": r_est_from_cv(m15["cv"]),
            "wdi_5_15": m5["sd"] / m15["sd"], "mr_5_15": m5["mean"] / m15["mean"],
            "ewd_5_15": m5["cv"] / m15["cv"]})
    for e in (1.0, 0.75, 0.5, 0.387, 0.25, 0.19, 0.1):
        tab["implied"].append({"e": e, "observed": 1.14, "implied_true_ratio": implied_true_ratio(1.14, e)})
    return tab


def fmt(x, nd=3):
    return "-" if x is None else ("%." + str(nd) + "f") % x


def write_md(tab, path):
    ks = list(tab["exponent"][0]["e"].keys())
    lines = ["# THEORY-2b tables (generated by ceiling.py)", "",
             "Made with AI agents working in MV2's name. First-order, deterministic-curve values; see THEORY-2b.md.",
             "", "## Ceiling exponent e(k, W)", "", "| W | " + " | ".join("k=" + k for k in ks) + " |",
             "|---|" + "---|" * len(ks)]
    for row in tab["exponent"]:
        lines.append("| %d | %s |" % (row["W"], " | ".join(fmt(row["e"][k]) for k in ks)))
    lines += ["", "## Pooled over phase 1's skills (k = 0.05, 0.10, 0.15)", "", "| W | e |", "|---|---|"]
    for row in tab["pooled"]:
        lines.append("| %d | %s |" % (row["W"], fmt(row["e_pooled_phase1"])))
    lines += ["", "## Reversal point (e = 0)", ""]
    for key, val in tab["reversal"].items():
        lines.append("- %s: %s" % ("continuous limit, u = k * span" if key == "continuous_u" else "W = " + key + ", k",
                                    fmt(val, 4)))
    lines += ["", "## Phase 1 truth T2, exact over students (no noise, no shrinkage)", "",
              "| level | true ratio | CV first5 | CV first10 | CV full15 | R_est first5 | R_est full15 | WDI | MR | EWD |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab["phase1_T2"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["level"], fmt(r["true_ratio"]), fmt(r["cv5"]), fmt(r["cv10"]), fmt(r["cv15"]), fmt(r["r_est5"]),
            fmt(r["r_est15"]), fmt(r["wdi_5_15"]), fmt(r["mr_5_15"]), fmt(r["ewd_5_15"])))
    lines += ["", "## True speed ratio that would look like the PNAS 1.14", "", "| e | implied true ratio |",
              "|---|---|"]
    for r in tab["implied"]:
        lines.append("| %s | %s |" % (fmt(r["e"]), fmt(r["implied_true_ratio"])))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    tab = build_tables()
    with open(os.path.join(args.out_dir, "theory2b.json"), "w") as fh:
        json.dump(tab, fh, indent=1, sort_keys=True)
    write_md(tab, os.path.join(args.out_dir, "theory2b.md"))
    with open(os.path.join(args.out_dir, "theory2b.md")) as fh:
        print(fh.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
