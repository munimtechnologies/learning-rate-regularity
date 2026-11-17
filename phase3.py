# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity, phase 3: do students differ in HOW SOON they learn? (preregistered in PLAN-3.md)
# Made with AI agents working in the name of MV2 of Munim, Inc.
#
# Each KC gets a saturating learning curve, logit P = theta_i + beta_k + A_k (1 - exp(-c_k t)). A student who is
# r = exp(v) times faster reaches the same ceiling sooner: A_k (1 - exp(-c_k r t)). To first order in v this adds
# v * h_k(t), with h_k(t) = A_k c_k t exp(-c_k t): a hump, so the gap between fast and slow learners opens and then
# closes. A student whose curve rises higher instead adds w * a_k(t), with a_k(t) = A_k (1 - exp(-c_k t)): a gap
# that opens and stays. Both are ordinary random slopes, so lme4 fits them with the same number of parameters and
# the data decide. SD(v) gives a speed ratio: 75th / 25th percentile student = exp(1.349 SD(v)).
#
# Python prepares the data, simulates the identification check and computes the verdicts (standard library
# only); R + lme4 does the fits (fit3.R), one subprocess per fit. Data loading and selection are phase 2's
# (realdata.py), so phase 3 uses the same students, KCs and bootstrap resamples.
import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request

import realdata as rd

C_GRID = (0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9)
HET_MODELS = ("lin", "amp", "ts", "both")
PAIR_MODELS = ("amp", "ts")
SPEED_K = 2 * rd.Z75
SEED = rd.SEED
N_BOOT = 10
MIN_BOOT = 8
SHARE = 0.9
R_WOW = 1.5
R_BOOT_MIN = 1.2
N_SIMS = 10
SIM_MIN_CORRECT = 8
SIM_SIGMA = 0.5
SIM_STUDENTS = 400
SIM_KCS = 8
SIM_SEED = 20261119


def covariates(t, amp, c):
    """g = 1 - exp(-c t); a = A g (amplitude direction); h = A c t exp(-c t) (time-scale direction)."""
    e = math.exp(-c * t)
    g = 1.0 - e
    return g, amp * g, amp * c * t * e


def speed_ratio(sd_v):
    """75th / 25th percentile learning speed when log-speed is normal with SD sd_v."""
    return None if sd_v is None else math.exp(SPEED_K * sd_v)


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def simulate(truth, sigma, n_students, n_kcs, seed, max_t=rd.MAX_T, run=3):
    """Known-truth data for the identification check. truth 'ts': student speed exp(sigma z) on every KC's curve;
    'amp': student curve height exp(sigma z). Practice stops after `run` correct in a row (as in ASSISTments skill
    builders) or at max_t. The random stream is the same for both truths, so sigma = 0 gives identical data."""
    rng = random.Random(seed)
    kcs = [(rng.gauss(-0.3, 0.5), rng.uniform(1.5, 3.0), rng.uniform(0.1, 0.4)) for _ in range(n_kcs)]
    recs = []
    for i in range(n_students):
        theta, z = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        speed = math.exp(sigma * z) if truth == "ts" else 1.0
        size = math.exp(sigma * z) if truth == "amp" else 1.0
        s = "s%04d" % i
        for k, (beta, amp, c) in enumerate(kcs):
            streak = 0
            for t in range(max_t):
                p = logistic(theta + beta + size * amp * (1.0 - math.exp(-c * speed * t)))
                y = 1 if rng.random() < p else 0
                recs.append((s, "k%02d" % k, t, y))
                streak = streak + 1 if y else 0
                if streak >= run:
                    break
    return recs


def load_dataset(d, data_dir):
    """Phase 2's loader. If the shared copy cannot be read (e.g. another job is still downloading it), fetch a
    private copy."""
    try:
        return rd.load(d, rd.find_or_fetch(d, data_dir))
    except Exception as exc:
        print(d, "shared copy unreadable (%r); downloading a private copy" % (exc,), flush=True)
        path = os.path.join(tempfile.mkdtemp(prefix="lrr3-data-"), rd.URLS[d].rsplit("/", 1)[1])
        urllib.request.urlretrieve(rd.URLS[d], path)
        return rd.load(d, path)


def write_kc_csv(by_student, students, kc, path):
    rows = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["student", "t", "y"])
        for j, s in enumerate(students):
            for k, t, y in by_student.get(s, ()):
                if k == kc:
                    w.writerow([j, t, y])
                    rows += 1
    return rows


def write_het_csv(by_student, sample, kc_index, curves, path):
    """One student-heterogeneity fit's data. A bootstrap sample may repeat a student; each draw is a new id. KCs
    without a fitted curve are left out."""
    rows = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["student", "kc", "t", "y", "g", "a", "h"])
        for j, s in enumerate(sample):
            for k, t, y in by_student.get(s, ()):
                cv = curves.get(k)
                if cv is None:
                    continue
                g, a, h = covariates(t, cv["A"], cv["c"])
                w.writerow([j, kc_index[k], t, y, "%.6g" % g, "%.6g" % a, "%.6g" % h])
                rows += 1
    return rows


def run_r(rscript, fit_r, mode, csv_path, spec, timeout=7200):
    out_path = csv_path + "." + (mode if mode == "curve" else "het-" + spec) + ".out.csv"
    t0 = time.time()
    rows = []
    try:
        proc = subprocess.run([rscript, fit_r, mode, csv_path, out_path, spec], capture_output=True, text=True,
                              timeout=timeout)
        if os.path.exists(out_path):
            with open(out_path, newline="") as fh:
                rows = [{k: rd._num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        if proc.returncode != 0 or not rows:
            rows = rows or [{}]
            for r in rows:
                r["error"] = 1.0
                r["stderr_tail"] = proc.stderr[-300:]
    except subprocess.TimeoutExpired:
        rows = [{"error": 1.0, "stderr_tail": "timeout"}]
    secs = round(time.time() - t0, 2)
    for r in rows:
        r["seconds"] = secs
    if os.path.exists(out_path):
        os.remove(out_path)
    return rows


def best_curve(rows, grid=C_GRID):
    """The grid value of c with the highest likelihood; 'edge' flags a best value at either end of the grid."""
    ok = [r for r in rows if not r.get("error") and r.get("loglik") is not None and r.get("A") is not None
          and r.get("c") is not None]
    if not ok:
        return None
    b = max(ok, key=lambda r: (r["loglik"], -r["c"]))
    edge = b["c"] <= min(grid) * (1 + 1e-9) or b["c"] >= max(grid) * (1 - 1e-9)
    return {"c": b["c"], "A": b["A"], "beta": b.get("beta"), "loglik": b["loglik"], "n_ok": len(ok), "edge": edge}


def start_curves(pool, args, by_student, students, kcs, tmp, tag):
    spec = ",".join(repr(c) for c in C_GRID)
    tasks = []
    for i, k in enumerate(kcs):
        path = os.path.join(tmp, "%s-kc%02d.csv" % (tag, i))
        n = write_kc_csv(by_student, students, k, path)
        tasks.append((i, k, n, path, pool.submit(run_r, args.rscript, args.fit_r, "curve", path, spec)))
    return tasks


def finish_curves(tasks):
    curves, rows = {}, []
    for i, k, n, path, fut in tasks:
        res = fut.result()
        rows += [dict(r, kc_index=i, kc=k, n=n) for r in res]
        best = best_curve(res)
        if best is not None:
            curves[k] = dict(best, kc_index=i, n=n)
        if os.path.exists(path):
            os.remove(path)
    return curves, rows


def collect(jobs, table, rows_out, extra):
    for key, n, path, fut in jobs:
        res = fut.result()[0]
        res["rows"] = n
        table[key] = res
        rows_out.append(dict(res, **extra(key)))
    for path in {j[2] for j in jobs}:
        if os.path.exists(path):
            os.remove(path)


def usable(f):
    return f is not None and not f.get("error") and f.get("loglik") is not None


def combine(flags, why):
    if any(f is False for f in flags):
        return "did not hold"
    if not flags or any(f is None for f in flags):
        return "inconclusive (%s)" % why
    return "held"


def gate(sims, n_sims=N_SIMS):
    """T1: the comparison recovers the truth in simulated data with the same pipeline."""
    out, flags = {}, []
    for truth, other in (("ts", "amp"), ("amp", "ts")):
        correct = done = 0
        for j in range(n_sims):
            a, b = sims.get((truth, j, truth)), sims.get((truth, j, other))
            if a is None and b is None:
                continue
            done += 1
            if usable(a) and usable(b) and a["loglik"] > b["loglik"]:
                correct += 1
        out[truth] = {"correct": correct, "done": done, "planned": n_sims}
        if correct >= SIM_MIN_CORRECT:
            flags.append(True)
        elif done >= n_sims:
            flags.append(False)
        else:
            flags.append(None)
    out["verdict"] = combine(flags, "simulations did not finish")
    return out


def calibration(sims, n_sims=N_SIMS):
    out = {}
    for truth, col in (("ts", "sd_h"), ("amp", "sd_a")):
        vals = [sims[(truth, j, truth)][col] for j in range(n_sims)
                if usable(sims.get((truth, j, truth))) and sims[(truth, j, truth)].get(col) is not None]
        med = rd.pct(vals, 0.5)
        out[truth] = {"n": len(vals), "true_sd": SIM_SIGMA, "median_sd": med,
                      "median_over_true": None if med is None else med / SIM_SIGMA}
    return out


def t2(fits, datasets, n_boot):
    out, flags = {}, []
    for d in datasets:
        lin, amp, ts = (fits.get((d, 0, m)) for m in ("lin", "amp", "ts"))
        pairs = [(fits.get((d, b, "ts")), fits.get((d, b, "amp"))) for b in range(1, n_boot + 1)]
        pairs = [(x, y) for x, y in pairs if usable(x) and usable(y)]
        share = (sum(1 for x, y in pairs if x["loglik"] > y["loglik"]) / len(pairs)) if pairs else None
        out[d] = {"loglik_lin": (lin or {}).get("loglik"), "loglik_amp": (amp or {}).get("loglik"),
                  "loglik_ts": (ts or {}).get("loglik"), "boot_pairs": len(pairs), "share_ts_better": share}
        if not (usable(lin) and usable(amp) and usable(ts)):
            flags.append(None)
        elif not (ts["loglik"] > amp["loglik"] and ts["loglik"] > lin["loglik"]):
            flags.append(False)
        elif len(pairs) < MIN_BOOT:
            flags.append(None)
        else:
            flags.append(share >= SHARE)
    out["verdict"] = combine(flags, "a fit failed or fewer than %d usable bootstrap pairs" % MIN_BOOT)
    return out


def t3(fits, datasets, n_boot):
    out, flags = {}, []
    for d in datasets:
        ts = fits.get((d, 0, "ts"))
        r_full = speed_ratio(ts.get("sd_h")) if usable(ts) else None
        boot = [fits.get((d, b, "ts")) for b in range(1, n_boot + 1)]
        rs = [speed_ratio(f["sd_h"]) for f in boot if usable(f) and f.get("sd_h") is not None]
        share = (sum(1 for r in rs if r >= R_BOOT_MIN) / len(rs)) if rs else None
        out[d] = {"speed_ratio": r_full, "boot_ok": len(rs), "share_boot_at_least_1.2": share,
                  "boot_90": [rd.pct(rs, 0.05), rd.pct(rs, 0.95)] if rs else None}
        if r_full is None:
            flags.append(None)
        elif r_full < R_WOW:
            flags.append(False)
        elif len(rs) < MIN_BOOT:
            flags.append(None)
        else:
            flags.append(share >= SHARE)
    out["verdict"] = combine(flags, "a fit failed or fewer than %d usable bootstrap fits" % MIN_BOOT)
    return out


def t4(fits, datasets):
    out, flags = {}, []
    for d in datasets:
        amp, both = fits.get((d, 0, "amp")), fits.get((d, 0, "both"))
        if usable(amp) and usable(both) and amp.get("aic") is not None and both.get("aic") is not None:
            out[d] = {"aic_amp": amp["aic"], "aic_both": both["aic"]}
            flags.append(both["aic"] < amp["aic"])
        else:
            out[d] = None
            flags.append(None)
    out["verdict"] = combine(flags, "a fit failed")
    return out


def verdicts(fits, sims, datasets, n_boot, n_sims=N_SIMS):
    out = {"T1": gate(sims, n_sims), "T2": t2(fits, datasets, n_boot), "T3": t3(fits, datasets, n_boot),
           "T4": t4(fits, datasets)}
    if out["T1"]["verdict"] != "held":
        for k in ("T2", "T3", "T4"):
            out[k]["verdict"] = "no verdict (T1 %s); rule alone: %s" % (out["T1"]["verdict"], out[k]["verdict"])
    return out


def fmt(x, nd=4):
    return rd.fmt(x, nd)


CURVE_FIELDS = ("dataset", "kc_index", "kc", "n", "c", "loglik", "beta", "A", "sd_theta", "n_warn", "error",
                "seconds", "stderr_tail")
FIT_FIELDS = ("dataset", "boot", "model", "rows", "sd_theta", "sd_t", "sd_a", "sd_h", "corr1", "loglik", "aic",
              "df", "singular", "n_warn", "error", "seconds", "stderr_tail")
SIM_FIELDS = ("truth", "sim", "sigma_true", "model", "rows", "sd_theta", "sd_a", "sd_h", "corr1", "loglik", "aic",
              "singular", "n_warn", "error", "seconds", "stderr_tail")


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(verd, cal, curves, fits, datasets, meta, path):
    lines = ["# learning-rate-regularity phase 3: output", "",
             "Preregistered in PLAN-3.md. Made with AI agents working in MV2's name. Generated by phase3.py.", "",
             "Run: " + json.dumps(meta, sort_keys=True, default=str), "", "## Verdicts", "",
             "| id | result | verdict |", "|---|---|---|"]
    for qid in sorted(verd):
        v = dict(verd[qid])
        verdict = v.pop("verdict")
        lines.append("| %s | %s | %s |" % (qid, json.dumps(v, sort_keys=True, default=str), verdict))
    lines += ["", "Simulation calibration (median fitted SD / true SD 0.5): " + json.dumps(cal, sort_keys=True), "",
              "## Full-sample fits", "",
              "| dataset | model | rows | SD(theta) | SD(t) | SD(a) | SD(h) | speed ratio from SD(h) | corr | loglik | "
              "AIC | singular |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in datasets:
        for m in HET_MODELS:
            f = fits.get((d, 0, m)) or {}
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                d, m, fmt(f.get("rows")), fmt(f.get("sd_theta")), fmt(f.get("sd_t")), fmt(f.get("sd_a")),
                fmt(f.get("sd_h")), fmt(speed_ratio(f.get("sd_h")) if usable(f) and f.get("sd_h") is not None
                                        else None, 3),
                fmt(f.get("corr1"), 3), fmt(f.get("loglik"), 1), fmt(f.get("aic"), 1), fmt(f.get("singular"))))
    lines += ["", "## Per-KC curves (time constant c from the grid, rise A in log-odds)", "",
              "| dataset | KC | rows | c | A | at grid edge |", "|---|---|---|---|---|---|"]
    for d in datasets:
        for k, cv in sorted((curves.get(d) or {}).items(), key=lambda kv: kv[1]["kc_index"]):
            lines.append("| %s | %d | %s | %s | %s | %s |" % (d, cv["kc_index"], fmt(cv["n"]), fmt(cv["c"], 3),
                                                             fmt(cv["A"], 3), cv["edge"]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def save(args, state, desc, datasets, start, stage, boots_done):
    verd = verdicts(state["fits"], state["sims"], datasets, boots_done, args.sims)
    cal = calibration(state["sims"], args.sims)
    meta = {"stage": stage, "datasets": desc, "boots_done": boots_done, "boots_planned": args.boot,
            "sims_planned": args.sims, "sim_sigma": SIM_SIGMA, "c_grid": list(C_GRID), "seed": SEED,
            "minutes": round((time.time() - start) / 60.0, 1)}
    write_csv(os.path.join(args.out_dir, "curves3.csv"), CURVE_FIELDS, state["curve_rows"])
    write_csv(os.path.join(args.out_dir, "fits3.csv"), FIT_FIELDS, state["fit_rows"])
    write_csv(os.path.join(args.out_dir, "sims3.csv"), SIM_FIELDS, state["sim_rows"])
    curves_out = {d: {str(cv["kc_index"]): {x: cv[x] for x in ("c", "A", "beta", "edge", "n")}
                      for cv in state["curves"].get(d, {}).values()} for d in datasets}
    with open(os.path.join(args.out_dir, "results3.json"), "w") as fh:
        json.dump({"meta": meta, "verdicts": verd, "calibration": cal, "curves": curves_out}, fh, indent=1,
                  sort_keys=True, default=str)
    write_md(verd, cal, state["curves"], state["fits"], datasets, meta, os.path.join(args.out_dir, "results3.md"))
    print("saved after", stage, "|", " ".join("%s=%s" % (k, verd[k]["verdict"]) for k in sorted(verd)), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="kdd_alg0506,assist09")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--sims", type=int, default=N_SIMS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=540.0)
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--fit-r", default="fit3.R")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    tmp = tempfile.mkdtemp(prefix="lrr3-")
    names = [d for d in args.datasets.split(",") if d]
    prepared, desc = {}, {}
    for d in names:
        try:
            recs = load_dataset(d, args.data_dir)
        except Exception as exc:  # a missing dataset leaves its predictions inconclusive, never "held"
            print(d, "could not be loaded:", repr(exc), flush=True)
            desc[d] = {"error": repr(exc)[:300]}
            continue
        sel, kcs, students = rd.select(recs, rd.N_KCS, rd.MAX_STUDENTS, rd.MAX_T, SEED)
        del recs
        if not sel:
            desc[d] = {"error": "no usable rows after selection"}
            continue
        prepared[d] = (rd.build_index(sel), students, kcs)
        desc[d] = rd.describe(sel, kcs, students)
        print(d, json.dumps(desc[d], sort_keys=True), flush=True)
    state = {"curve_rows": [], "fit_rows": [], "sim_rows": [], "curves": {}, "fits": {}, "sims": {}}
    fit_extra = lambda key: {"dataset": key[0], "boot": key[1], "model": key[2]}  # noqa: E731
    sim_extra = lambda key: {"truth": key[0], "sim": key[1], "model": key[2], "sigma_true": SIM_SIGMA}  # noqa: E731
    boots_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # 1. per-KC curves on the real data
        tasks = {d: start_curves(pool, args, p[0], p[1], p[2], tmp, d) for d, p in prepared.items()}
        for d, t in tasks.items():
            curves, rows = finish_curves(t)
            state["curves"][d] = curves
            state["curve_rows"] += [dict(r, dataset=d) for r in rows]
        save(args, state, desc, names, start, "curves", 0)
        # 2. student-heterogeneity models, full sample
        jobs = []
        for d, (by_student, students, kcs) in prepared.items():
            path = os.path.join(tmp, "full-%s.csv" % d)
            n = write_het_csv(by_student, students, {k: i for i, k in enumerate(kcs)}, state["curves"][d], path)
            jobs += [((d, 0, m), n, path, pool.submit(run_r, args.rscript, args.fit_r, "het", path, m))
                     for m in HET_MODELS]
        collect(jobs, state["fits"], state["fit_rows"], fit_extra)
        save(args, state, desc, names, start, "full-sample fits", 0)
        # 3. identification check (T1): the same pipeline on simulated data with known truth
        sim_data, sim_tasks = {}, {}
        for ti, truth in enumerate(("ts", "amp")):
            for j in range(args.sims):
                recs = simulate(truth, SIM_SIGMA, SIM_STUDENTS, SIM_KCS, SIM_SEED + 1000 * ti + j)
                by_student = rd.build_index(recs)
                students = sorted(by_student)
                kcs = sorted({r[1] for r in recs})
                sim_data[(truth, j)] = (by_student, students, kcs)
                sim_tasks[(truth, j)] = start_curves(pool, args, by_student, students, kcs, tmp,
                                                     "sim-%s-%02d" % (truth, j))
        jobs = []
        for key, t in sim_tasks.items():
            curves, rows = finish_curves(t)
            state["curve_rows"] += [dict(r, dataset="sim-%s-%02d" % key) for r in rows]
            by_student, students, kcs = sim_data[key]
            if not curves:
                for m in PAIR_MODELS:
                    state["sims"][(key[0], key[1], m)] = {"error": 1.0, "stderr_tail": "no curve could be fitted"}
                continue
            path = os.path.join(tmp, "sim-%s-%02d.csv" % key)
            n = write_het_csv(by_student, students, {k: i for i, k in enumerate(kcs)}, curves, path)
            jobs += [((key[0], key[1], m), n, path, pool.submit(run_r, args.rscript, args.fit_r, "het", path, m))
                     for m in PAIR_MODELS]
        collect(jobs, state["sims"], state["sim_rows"], sim_extra)
        del sim_data
        save(args, state, desc, names, start, "simulations", 0)
        # 4. bootstrap resamples of students (same seeds as phase 2), curves held fixed
        for b in range(1, args.boot + 1):
            if (time.time() - start) / 60.0 > args.max_minutes:
                print("time budget reached before bootstrap", b, flush=True)
                break
            jobs = []
            for di, d in enumerate(names):
                if d not in prepared:
                    continue
                by_student, students, kcs = prepared[d]
                rng = random.Random(SEED * 1000 + b * 10 + di)
                sample = [rng.choice(students) for _ in students]
                path = os.path.join(tmp, "b%d-%s.csv" % (b, d))
                n = write_het_csv(by_student, sample, {k: i for i, k in enumerate(kcs)}, state["curves"][d], path)
                jobs += [((d, b, m), n, path, pool.submit(run_r, args.rscript, args.fit_r, "het", path, m))
                         for m in PAIR_MODELS]
            collect(jobs, state["fits"], state["fit_rows"], fit_extra)
            boots_done = b
            save(args, state, desc, names, start, "bootstrap %d" % b, boots_done)
    save(args, state, desc, names, start, "done", boots_done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
