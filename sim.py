# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity, phase 1: a preregistered simulation (see PLAN-1.md).
# Made with AI agents working in the name of MV2 of Munim, Inc.
#
# Question: when students truly differ in how fast they learn, how much of that difference does the
# individualized Additive Factors Model (iAFM; the model behind Koedinger et al. 2023, PNAS, "An astonishing
# regularity in student learning rate") report, and does the answer depend on (a) the true shape of learning
# and (b) the rule that decides when practice stops?
#
# Python makes the data (standard library only); R + lme4 fits the iAFM (fit.R), one fit per subprocess.
import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

Z75 = 0.6744897501960817  # 75th percentile of the standard normal

# Population values, fixed in PLAN-1.md.
SD_THETA = 1.0      # student initial knowledge (logits)
BETA_MEAN = -0.5    # KC easiness (logits)
BETA_SD = 0.7
G_MEAN = 0.10       # T1: mean KC learning rate (logits per opportunity), as reported in the PNAS paper
G_SD = 0.03
G_MIN = 0.02
A_SAT = 2.0         # T2: total gain in logits once learning saturates
C_LO, C_HI = 0.05, 0.15   # T2: KC rate constants
T_LO, T_HI = 0.08, 0.20   # T3: KC learn probabilities (BKT)
T_MED = 0.14
BKT_GUESS = 0.2
BKT_SLIP = 0.1
# The simulated tutor's own (population-level) BKT parameters for the mastery stopping rule.
TUTOR_L0, TUTOR_T, TUTOR_G, TUTOR_S, TUTOR_THRESHOLD = 0.4, 0.15, 0.2, 0.1, 0.95
MAX_OPP = 15

SETTINGS = (
    ("T1", 0.0), ("T1", 0.02), ("T1", 0.05),
    ("T2", 0.0), ("T2", 0.15), ("T2", 0.3), ("T2", 0.5),
    ("T3", 0.0), ("T3", 0.25), ("T3", 0.5),
)
VARIANTS = ("full15", "first10", "first5", "three_in_a_row", "bkt_mastery")
MODELS = ("linear", "log")
REALISTIC = ("full15", "three_in_a_row", "bkt_mastery")


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p):
    return math.log(p / (1.0 - p))


def bkt_update(p_known, correct, T, G, S):
    """Standard BKT: posterior after observing one response, then the learning transition."""
    if correct:
        num = p_known * (1.0 - S)
        den = num + (1.0 - p_known) * G
    else:
        num = p_known * S
        den = num + (1.0 - p_known) * (1.0 - G)
    post = num / den
    return post + (1.0 - post) * T


def cut_three_in_a_row(ys):
    """ASSISTments-style skill builder: stop after three correct answers in a row."""
    out, run = [], 0
    for y in ys:
        out.append(y)
        run = run + 1 if y else 0
        if run >= 3:
            break
    return out


def cut_bkt_mastery(ys, L0=TUTOR_L0, T=TUTOR_T, G=TUTOR_G, S=TUTOR_S, threshold=TUTOR_THRESHOLD):
    """Cognitive-Tutor-style mastery: stop once the tutor's BKT estimate reaches the threshold."""
    out, p = [], L0
    for y in ys:
        out.append(y)
        p = bkt_update(p, y, T, G, S)
        if p >= threshold:
            break
    return out


def apply_variant(ys, variant):
    if variant == "full15":
        return list(ys)
    if variant == "first10":
        return list(ys[:10])
    if variant == "first5":
        return list(ys[:5])
    if variant == "three_in_a_row":
        return cut_three_in_a_row(ys)
    if variant == "bkt_mastery":
        return cut_bkt_mastery(ys)
    raise ValueError(variant)


def respond_sequence(truth, level, theta, z, beta, g, c, tk, u, v):
    """Responses of one student on one KC over len(u) opportunities, from pre-drawn uniforms u (responses)
    and v (BKT initial state and transitions; len(u) + 1 values)."""
    ys = []
    n = len(u)
    if truth == "T1":      # the iAFM itself: logit-linear in opportunities, student slope offset level * z
        for t in range(n):
            ys.append(1 if u[t] < sigmoid(theta + beta + (g + level * z) * t) else 0)
    elif truth == "T2":    # saturating: logit rises by A_SAT * (1 - exp(-c * r * t)), speed r = exp(level * z)
        r = math.exp(level * z)
        for t in range(n):
            ys.append(1 if u[t] < sigmoid(theta + beta + A_SAT * (1.0 - math.exp(-c * r * t))) else 0)
    elif truth == "T3":    # BKT with a student-specific learn probability
        p0 = sigmoid(theta + beta)
        tr = sigmoid(logit(tk) + level * z)
        known = v[0] < p0
        for t in range(n):
            ys.append(1 if u[t] < ((1.0 - BKT_SLIP) if known else BKT_GUESS) else 0)
            if not known and v[t + 1] < tr:
                known = True
    else:
        raise ValueError(truth)
    return ys


def generate(truth, level, n_students, n_kcs, n_opp, seed):
    """Full fixed-length data. Every truth and level uses the same random stream for the same seed, so
    students, KCs and uniforms are shared across settings (common random numbers)."""
    rng = random.Random(seed)
    theta = [rng.gauss(0.0, SD_THETA) for _ in range(n_students)]
    z = [rng.gauss(0.0, 1.0) for _ in range(n_students)]
    beta = [rng.gauss(BETA_MEAN, BETA_SD) for _ in range(n_kcs)]
    g = [max(G_MIN, rng.gauss(G_MEAN, G_SD)) for _ in range(n_kcs)]
    c = [rng.uniform(C_LO, C_HI) for _ in range(n_kcs)]
    tk = [rng.uniform(T_LO, T_HI) for _ in range(n_kcs)]
    seqs = {}
    for i in range(n_students):
        for k in range(n_kcs):
            u = [rng.random() for _ in range(n_opp)]
            v = [rng.random() for _ in range(n_opp + 1)]
            seqs[(i, k)] = respond_sequence(truth, level, theta[i], z[i], beta[k], g[k], c[k], tk[k], u, v)
    return seqs


def true_ratio(truth, level):
    """Learning speed of a 75th-percentile student over a 25th-percentile student (typical KC)."""
    if level == 0:
        return 1.0
    if truth == "T1":
        lo = G_MEAN - Z75 * level
        return (G_MEAN + Z75 * level) / lo if lo > 0 else None
    if truth == "T2":
        return math.exp(2.0 * Z75 * level)
    if truth == "T3":
        return sigmoid(logit(T_MED) + Z75 * level) / sigmoid(logit(T_MED) - Z75 * level)
    raise ValueError(truth)


def r_est(gbar, sd_delta):
    """The same ratio as the iAFM reports it: (mean slope + z75 * student slope SD) / (mean - z75 * SD).
    None when the lower slope is not positive (unbounded)."""
    if gbar is None or sd_delta is None:
        return None
    lo = gbar - Z75 * sd_delta
    if lo <= 0:
        return None
    return (gbar + Z75 * sd_delta) / lo


def write_csv(seqs, variant, path):
    rows = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["student", "kc", "t", "y"])
        for (i, k) in sorted(seqs):
            for t, y in enumerate(apply_variant(seqs[(i, k)], variant)):
                w.writerow([i, k, t, y])
                rows += 1
    return rows


def _num(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in ("", "NA", "NaN", "nan"):
        return None
    if x in ("TRUE", "FALSE"):
        return x == "TRUE"
    try:
        return float(x)
    except ValueError:
        return None


def parse_fit(path):
    with open(path, newline="") as fh:
        row = next(csv.DictReader(fh))
    return {k: _num(v) for k, v in row.items()}


def run_fit(rscript, fit_r, csv_path, model, timeout=1800):
    out_path = csv_path + "." + model + ".out.csv"
    t0 = time.time()
    try:
        proc = subprocess.run([rscript, fit_r, csv_path, out_path, model], capture_output=True, text=True,
                              timeout=timeout)
        res = parse_fit(out_path) if os.path.exists(out_path) else {"error": 1.0}
        if proc.returncode != 0 and not res.get("error"):
            res["error"] = 1.0
        if proc.returncode != 0:
            res["stderr_tail"] = proc.stderr[-300:]
    except subprocess.TimeoutExpired:
        res = {"error": 1.0, "stderr_tail": "timeout"}
    res["seconds"] = round(time.time() - t0, 2)
    if os.path.exists(out_path):
        os.remove(out_path)
    return res


def cell_key(truth, level, variant, model):
    return "%s|%s|%s|%s" % (truth, level, variant, model)


def summarize(fits):
    """fits: list of dicts with truth, level, variant, model, sd_delta, sd_theta, gbar, singular, error."""
    groups = {}
    for f in fits:
        groups.setdefault(cell_key(f["truth"], f["level"], f["variant"], f["model"]), []).append(f)
    cells = {}
    for key, fs in sorted(groups.items()):
        ok = [f for f in fs if not f.get("error") and f.get("sd_delta") is not None and f.get("gbar") is not None]
        sds = [f["sd_delta"] for f in ok]
        ratios = [r_est(f["gbar"], f["sd_delta"]) for f in ok]
        ratios_sorted = sorted(float("inf") if r is None else r for r in ratios)
        truth, level = fs[0]["truth"], fs[0]["level"]
        thetas = [f["sd_theta"] for f in ok if f.get("sd_theta") is not None]
        cell = {
            "truth": truth, "level": level, "variant": fs[0]["variant"], "model": fs[0]["model"],
            "n_fits": len(fs), "n_ok": len(ok),
            "n_singular": sum(1 for f in ok if f.get("singular") is True),
            "n_unbounded": sum(1 for r in ratios if r is None),
            "mean_sd_delta": statistics.mean(sds) if sds else None,
            "se_sd_delta": (statistics.stdev(sds) / math.sqrt(len(sds))) if len(sds) > 1 else None,
            "mean_sd_theta": statistics.mean(thetas) if thetas else None,
            "mean_gbar": statistics.mean([f["gbar"] for f in ok]) if ok else None,
            "median_r_est": statistics.median(ratios_sorted) if ratios_sorted else None,
            "true_ratio": true_ratio(truth, level),
        }
        cells[key] = cell
    return cells


MIN_FITS = 5


def _cell(cells, truth, level, variant, model="linear"):
    c = cells.get(cell_key(truth, level, variant, model))
    if c is None or c["n_ok"] < MIN_FITS or c["mean_sd_delta"] is None:
        return None
    return c


def _ratio_of_means(a, b):
    if a is None or b is None or not b["mean_sd_delta"]:
        return None
    return a["mean_sd_delta"] / b["mean_sd_delta"]


def verdicts(cells):
    out = {}
    few = "inconclusive (fewer than %d usable fits)" % MIN_FITS

    c = _cell(cells, "T1", 0.05, "full15")
    if c is None:
        out["P0"] = {"verdict": few}
    else:
        v = c["mean_sd_delta"]
        out["P0"] = {"observed": v, "verdict": "held" if 0.04 <= v <= 0.0625 else "did not hold"}

    base = _cell(cells, "T1", 0.05, "full15")
    r1 = {}
    for variant in ("three_in_a_row", "bkt_mastery", "first10"):
        r1[variant] = _ratio_of_means(_cell(cells, "T1", 0.05, variant), base)
    for pid, names in (("P1", ("three_in_a_row", "bkt_mastery")), ("P1b", ("first10",))):
        vals = [r1[n] for n in names]
        if any(x is None for x in vals):
            out[pid] = {"ratios": {n: r1[n] for n in names}, "verdict": few}
        else:
            ok = all(0.8 <= x <= 1.25 for x in vals)
            out[pid] = {"ratios": {n: r1[n] for n in names}, "verdict": "held" if ok else "did not hold"}

    r2 = _ratio_of_means(_cell(cells, "T2", 0.5, "first5"), _cell(cells, "T2", 0.5, "full15"))
    out["P2"] = {"ratio": r2, "verdict": few if r2 is None else ("held" if r2 >= 1.5 else "did not hold")}

    att = {}
    for truth in ("T2", "T3"):
        c = _cell(cells, truth, 0.5, "full15")
        if c is None or c["median_r_est"] is None:
            att[truth] = None
        else:
            att[truth] = (c["median_r_est"] - 1.0) / (c["true_ratio"] - 1.0)
    if any(a is None for a in att.values()):
        out["P3"] = {"attenuation": att, "verdict": few}
    else:
        out["P3"] = {"attenuation": att, "verdict": "held" if all(a <= 0.6 for a in att.values()) else "did not hold"}

    for pid, truth, want_low in (("P4", "T2", True), ("P5", "T3", False)):
        c = _cell(cells, truth, 0.0, "full15")
        if c is None:
            out[pid] = {"verdict": few}
            continue
        r = c["median_r_est"]
        low = r <= 1.15
        out[pid] = {"median_r_est": r, "verdict": "held" if low == want_low else "did not hold"}

    hits, checked = [], 0
    for key, c in cells.items():
        if c["model"] != "linear" or c["truth"] not in ("T2", "T3") or c["variant"] not in REALISTIC:
            continue
        if c["true_ratio"] is None or c["true_ratio"] < 1.75 or _cell(cells, c["truth"], c["level"], c["variant"]) is None:
            continue
        checked += 1
        if c["median_r_est"] <= 1.25:
            hits.append(key)
    if checked == 0:
        out["P6"] = {"verdict": few}
    else:
        out["P6"] = {"cells_checked": checked, "cells_at_or_below_1.25": hits,
                     "verdict": "held" if not hits else "did not hold"}
    return out


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isinf(x):
            return "unbounded"
        return ("%." + str(nd) + "f") % x
    return str(x)


def write_md(cells, verd, meta, path):
    lines = ["# learning-rate-regularity phase 1: simulation output", "",
             "Preregistered in PLAN-1.md. Made with AI agents working in MV2's name. Generated by sim.py.", "",
             "Run: " + json.dumps(meta, sort_keys=True), "", "## Verdicts", "",
             "| id | result | verdict |", "|---|---|---|"]
    for pid in sorted(verd):
        v = dict(verd[pid])
        verdict = v.pop("verdict")
        lines.append("| %s | %s | %s |" % (pid, json.dumps(v, sort_keys=True, default=str), verdict))
    lines += ["", "## Cells", "",
              "| truth | level | variant | model | fits ok | singular | mean SD(delta) | SE | mean slope | median R_est | true ratio |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        c = cells[key]
        lines.append("| %s | %s | %s | %s | %d/%d | %d | %s | %s | %s | %s | %s |" % (
            c["truth"], c["level"], c["variant"], c["model"], c["n_ok"], c["n_fits"], c["n_singular"],
            fmt(c["mean_sd_delta"]), fmt(c["se_sd_delta"]), fmt(c["mean_gbar"]), fmt(c["median_r_est"], 3),
            fmt(c["true_ratio"], 3)))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


FIT_FIELDS = ("rep", "truth", "level", "variant", "model", "rows", "sd_theta", "sd_delta", "corr", "gbar",
              "singular", "n_warn", "error", "seconds")


def save_fits(fits, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for f in fits:
            w.writerow(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--students", type=int, default=150)
    ap.add_argument("--kcs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=600.0)
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--fit-r", default="fit.R")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    fits, reps_done = [], 0
    tmp = tempfile.mkdtemp(prefix="lrr-")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rep in range(args.reps):
            if (time.time() - start) / 60.0 > args.max_minutes:
                print("time budget reached before rep", rep, flush=True)
                break
            jobs = []
            for truth, level in SETTINGS:
                seqs = generate(truth, level, args.students, args.kcs, MAX_OPP, args.seed + rep)
                for variant in VARIANTS:
                    path = os.path.join(tmp, "r%d-%s-%s-%s.csv" % (rep, truth, level, variant))
                    rows = write_csv(seqs, variant, path)
                    for model in MODELS:
                        meta = {"rep": rep, "truth": truth, "level": level, "variant": variant, "model": model,
                                "rows": rows}
                        jobs.append((meta, path, pool.submit(run_fit, args.rscript, args.fit_r, path, model)))
            for meta, path, fut in jobs:
                res = fut.result()
                row = dict(meta)
                row.update(res)
                fits.append(row)
            for meta, path, fut in jobs:
                if os.path.exists(path):
                    os.remove(path)
            reps_done = rep + 1
            save_fits(fits, os.path.join(args.out_dir, "fits.csv"))
            errs = sum(1 for f in fits if f.get("error"))
            print("rep %d done: %d fits, %d errors, %.1f min" % (rep, len(fits), errs, (time.time() - start) / 60.0),
                  flush=True)
    cells = summarize(fits)
    verd = verdicts(cells)
    meta = {"reps_done": reps_done, "reps_planned": args.reps, "students": args.students, "kcs": args.kcs,
            "max_opp": MAX_OPP, "seed": args.seed, "fits": len(fits),
            "errors": sum(1 for f in fits if f.get("error")), "minutes": round((time.time() - start) / 60.0, 1)}
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump({"meta": meta, "verdicts": verd, "cells": cells}, fh, indent=1, sort_keys=True, default=str)
    write_md(cells, verd, meta, os.path.join(args.out_dir, "results.md"))
    for pid in sorted(verd):
        print(pid, verd[pid]["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
