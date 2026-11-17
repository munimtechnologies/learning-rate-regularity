# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity, phase 2: real tutoring logs (preregistered in PLAN-2.md).
# Made with AI agents working in the name of MV2 of Munim, Inc.
#
# Question: in real tutoring data, the iAFM's student learning-rate spread grows when only each student's first
# few attempts per skill are used (Lee et al., arXiv:2605.01690). If that window dependence comes from the wrong
# learning-curve SHAPE (a straight line in log-odds per opportunity), a better-shaped curve (log(1 + t)) should
# show less of it. Two open logs with mastery-based practice: KDD Cup 2010 Algebra I 2005-2006 (Cognitive Tutor,
# BKT mastery) and ASSISTments 2009-2010 skill builders (three correct in a row).
#
# Python prepares the data (standard library only); R + lme4 fits the iAFM (fit2.R), one fit per subprocess.
import argparse
import concurrent.futures
import csv
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

Z75 = 0.6744897501960817
WINDOWS = (("first5", 5), ("first10", 10), ("first25", 25))
SHORT, LONG = "first5", "first25"
MODELS = ("linear", "log")
MAX_T = 25
N_KCS = 12
MAX_STUDENTS = 800
N_BOOT = 20
MIN_BOOT = 15
SEED = 20261118
Q1_MIN = 1.3
Q2_SHARE = 0.9
URLS = {
    "kdd_alg0506": "http://base.ustc.edu.cn/data/KDD_Cup_2010/algebra_2005_2006.zip",
    "assist09": "http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip",
}
FILE_HINTS = {"kdd_alg0506": "algebra_2005_2006", "assist09": "skill_builder"}


def parse_kdd(lines):
    """KDD Cup 2010 step file (tab-separated, no quoting). Yields (student, kc, t, y) for steps with exactly one
    KC in the first KC model column; t = that KC's opportunity count - 1; y = Correct First Attempt."""
    it = iter(lines)
    header = next(it).rstrip("\r\n").split("\t")
    kc_col = next(i for i, h in enumerate(header) if h.startswith("KC("))
    opp_col = header.index("Opportunity" + header[kc_col][2:])
    s_col = header.index("Anon Student Id")
    y_col = header.index("Correct First Attempt")
    need = max(kc_col, opp_col, s_col, y_col)
    for line in it:
        row = line.rstrip("\r\n").split("\t")
        if len(row) <= need:
            continue
        kc, opp, y = row[kc_col].strip(), row[opp_col].strip(), row[y_col].strip()
        if not kc or "~~" in kc or not opp or "~~" in opp or y not in ("0", "1"):
            continue
        try:
            t = int(opp) - 1
        except ValueError:
            continue
        if t < 0:
            continue
        yield sys.intern(row[s_col].strip()), sys.intern(kc), t, int(y)


def _clean(name):
    return name.replace(chr(0xFEFF), "").replace(chr(0xEF) + chr(0xBB) + chr(0xBF), "").strip()


def parse_assist(fh):
    """ASSISTments 2009-2010 skill-builder CSV. Main problems only (original == 1), single-skill problems only
    (a multi-skill problem repeats its order_id once per skill), ordered by order_id; t counts each student's
    earlier single-skill main problems on the same skill."""
    reader = csv.reader(fh)
    header = [_clean(h) for h in next(reader)]
    col = {h: i for i, h in enumerate(header)}
    o_col, u_col, s_col, y_col = col["order_id"], col["user_id"], col["skill_id"], col["correct"]
    orig_col = col.get("original")
    need = max(o_col, u_col, s_col, y_col, -1 if orig_col is None else orig_col)
    rows, counts = [], {}
    for r in reader:
        if len(r) <= need:
            continue
        if orig_col is not None and r[orig_col].strip() != "1":
            continue
        skill, y = r[s_col].strip(), r[y_col].strip()
        if not skill or y not in ("0", "1"):
            continue
        try:
            order = int(r[o_col])
        except ValueError:
            continue
        rows.append((order, sys.intern(r[u_col].strip()), sys.intern(skill), int(y)))
        counts[order] = counts.get(order, 0) + 1
    rows = sorted(x for x in rows if counts[x[0]] == 1)
    seen = {}
    for order, user, skill, y in rows:
        t = seen.get((user, skill), 0)
        seen[(user, skill)] = t + 1
        yield user, skill, t, y


def _member(zf, suffix):
    names = [n for n in zf.namelist() if n.endswith(suffix) and "__MACOSX" not in n]
    if not names:
        raise SystemExit("no *%s in zip" % suffix)
    return sorted(names)[0]


def load(dataset, zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        if dataset == "kdd_alg0506":
            with zf.open(_member(zf, "_train.txt")) as raw:
                return list(parse_kdd(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")))
        with zf.open(_member(zf, ".csv")) as raw:
            return list(parse_assist(io.TextIOWrapper(raw, encoding="latin-1", newline="")))


def find_or_fetch(dataset, data_dir):
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if FILE_HINTS[dataset] in f and f.endswith(".zip"):
                return os.path.join(root, f)
    target_dir = data_dir if os.access(data_dir, os.W_OK) else tempfile.mkdtemp(prefix="lrr2-data-")
    path = os.path.join(target_dir, URLS[dataset].rsplit("/", 1)[1])
    print("downloading", URLS[dataset], flush=True)
    urllib.request.urlretrieve(URLS[dataset], path)
    return path


def select(recs, n_kcs, max_students, max_t, seed):
    """Keep t < max_t, the n_kcs KCs practised by the most students, and at most max_students students
    (a seeded random sample)."""
    recs = [r for r in recs if r[2] < max_t]
    students_by_kc = {}
    for s, k, t, y in recs:
        students_by_kc.setdefault(k, set()).add(s)
    kcs = sorted(students_by_kc, key=lambda k: (-len(students_by_kc[k]), k))[:n_kcs]
    keep_k = set(kcs)
    recs = [r for r in recs if r[1] in keep_k]
    students = sorted({r[0] for r in recs})
    if len(students) > max_students:
        students = sorted(random.Random(seed).sample(students, max_students))
    keep_s = set(students)
    return [r for r in recs if r[0] in keep_s], kcs, students


def build_index(recs):
    by_student = {}
    for s, k, t, y in recs:
        by_student.setdefault(s, []).append((k, t, y))
    for s in by_student:
        by_student[s].sort()
    return by_student


def describe(recs, kcs, students):
    seqs = {}
    for s, k, t, y in recs:
        seqs.setdefault((s, k), []).append(t)
    lens = [len(v) for v in seqs.values()]
    return {"rows": len(recs), "students": len(students), "kcs": len(kcs), "sequences": len(seqs),
            "accuracy": round(sum(r[3] for r in recs) / len(recs), 4) if recs else None,
            "mean_obs_per_sequence": round(sum(lens) / len(lens), 2) if lens else None,
            "share_sequences_reaching_t5": round(sum(1 for v in seqs.values() if max(v) >= 5) / len(seqs), 4)
            if seqs else None,
            "share_sequences_reaching_t10": round(sum(1 for v in seqs.values() if max(v) >= 10) / len(seqs), 4)
            if seqs else None}


def write_window_csv(by_student, sample, kc_index, window, path):
    """One fit's data: sampled students (a bootstrap sample may repeat a student; each draw is a new student
    id), their observations with t < window."""
    rows = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["student", "kc", "t", "y"])
        for j, s in enumerate(sample):
            for k, t, y in by_student[s]:
                if t < window:
                    w.writerow([j, kc_index[k], t, y])
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


def run_fit(rscript, fit_r, csv_path, model, timeout=3600):
    out_path = csv_path + "." + model + ".out.csv"
    t0 = time.time()
    try:
        proc = subprocess.run([rscript, fit_r, csv_path, out_path, model], capture_output=True, text=True,
                              timeout=timeout)
        if os.path.exists(out_path):
            with open(out_path, newline="") as fh:
                res = {k: _num(v) for k, v in next(csv.DictReader(fh)).items()}
        else:
            res = {"error": 1.0}
        if proc.returncode != 0:
            res["error"] = 1.0
            res["stderr_tail"] = proc.stderr[-300:]
    except subprocess.TimeoutExpired:
        res = {"error": 1.0, "stderr_tail": "timeout"}
    res["seconds"] = round(time.time() - t0, 2)
    if os.path.exists(out_path):
        os.remove(out_path)
    return res


def r_est(gbar, sd_delta):
    if gbar is None or sd_delta is None:
        return None
    lo = gbar - Z75 * sd_delta
    if lo <= 0:
        return None
    return (gbar + Z75 * sd_delta) / lo


def usable(f):
    return f is not None and not f.get("error") and f.get("sd_delta") is not None and f.get("gbar") is not None


def wdi(results, dataset, model, b):
    """Window-dependence index: student-slope SD on the first 5 opportunities over that on the first 25."""
    a, c = results.get((dataset, SHORT, model, b)), results.get((dataset, LONG, model, b))
    if not (usable(a) and usable(c)) or not c["sd_delta"]:
        return None
    return a["sd_delta"] / c["sd_delta"]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def combine(flags):
    if any(f is False for f in flags):
        return "did not hold"
    if any(f is None for f in flags):
        return "inconclusive (a fit failed or fewer than %d usable bootstrap pairs)" % MIN_BOOT
    return "held"


def verdicts(results, datasets, n_boot):
    out = {}
    q1 = {d: wdi(results, d, "linear", 0) for d in datasets}
    out["Q1"] = {"wdi_linear": q1, "verdict": combine([None if v is None else v >= Q1_MIN for v in q1.values()])}

    q2, flags = {}, []
    for d in datasets:
        lin, lg = wdi(results, d, "linear", 0), wdi(results, d, "log", 0)
        pairs = [(wdi(results, d, "linear", b), wdi(results, d, "log", b)) for b in range(1, n_boot + 1)]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        share = (sum(1 for a, g in pairs if g < a) / len(pairs)) if pairs else None
        q2[d] = {"wdi_linear": lin, "wdi_log": lg, "boot_pairs": len(pairs), "share_log_lower": share}
        if lin is None or lg is None or len(pairs) < MIN_BOOT:
            flags.append(None)
        else:
            flags.append(lg < lin and share >= Q2_SHARE)
    out["Q2"] = dict(q2, verdict=combine(flags))

    q3, flags = {}, []
    for d in datasets:
        a, g = results.get((d, LONG, "linear", 0)), results.get((d, LONG, "log", 0))
        if not (usable(a) and usable(g)) or a.get("aic") is None or g.get("aic") is None:
            q3[d] = None
            flags.append(None)
        else:
            q3[d] = {"aic_linear": a["aic"], "aic_log": g["aic"]}
            flags.append(g["aic"] < a["aic"])
    out["Q3"] = dict(q3, verdict=combine(flags))
    return out


def cells(results, datasets, n_boot):
    table = []
    for d in datasets:
        for wname, _ in WINDOWS:
            for m in MODELS:
                p = results.get((d, wname, m, 0)) or {}
                boots = [results.get((d, wname, m, b)) for b in range(1, n_boot + 1)]
                boots = [f for f in boots if usable(f)]
                sds = [f["sd_delta"] for f in boots]
                rs = [r_est(f["gbar"], f["sd_delta"]) for f in boots]
                rs = [float("inf") if r is None else r for r in rs]
                table.append({
                    "dataset": d, "window": wname, "model": m, "rows": p.get("rows"),
                    "ok": usable(p), "singular": p.get("singular"), "sd_theta": p.get("sd_theta"),
                    "sd_delta": p.get("sd_delta"), "corr": p.get("corr"), "gbar": p.get("gbar"),
                    "r_est": r_est(p.get("gbar"), p.get("sd_delta")) if usable(p) else None,
                    "aic": p.get("aic"), "boot_ok": len(boots),
                    "sd_delta_90": [pct(sds, 0.05), pct(sds, 0.95)] if sds else None,
                    "r_est_90": [pct(rs, 0.05), pct(rs, 0.95)] if rs else None})
    return table


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "unbounded" if math.isinf(x) else ("%." + str(nd) + "f") % x
    if isinstance(x, list):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    return str(x)


def write_md(table, verd, meta, path):
    lines = ["# learning-rate-regularity phase 2: real-data output", "",
             "Preregistered in PLAN-2.md. Made with AI agents working in MV2's name. Generated by realdata.py.", "",
             "Run: " + json.dumps(meta, sort_keys=True, default=str), "", "## Verdicts", "",
             "| id | result | verdict |", "|---|---|---|"]
    for qid in sorted(verd):
        v = dict(verd[qid])
        verdict = v.pop("verdict")
        lines.append("| %s | %s | %s |" % (qid, json.dumps(v, sort_keys=True, default=str), verdict))
    lines += ["", "## Cells (point estimate on all selected students; 90% bootstrap intervals over students)", "",
              "| dataset | window | model | rows | SD(theta) | SD(delta) | 90% | mean slope | R_est | 90% | AIC | "
              "singular | boot ok |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in table:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c["dataset"], c["window"], c["model"], fmt(c["rows"]), fmt(c["sd_theta"]), fmt(c["sd_delta"]),
            fmt(c["sd_delta_90"]), fmt(c["gbar"]), fmt(c["r_est"], 3), fmt(c["r_est_90"], 3), fmt(c["aic"], 1),
            fmt(c["singular"]), c["boot_ok"]))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


FIT_FIELDS = ("dataset", "window", "model", "boot", "rows", "sd_theta", "sd_delta", "corr", "gbar", "aic", "bic",
              "loglik", "singular", "n_warn", "error", "seconds", "stderr_tail")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="kdd_alg0506,assist09")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--kcs", type=int, default=N_KCS)
    ap.add_argument("--max-students", type=int, default=MAX_STUDENTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=480.0)
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--fit-r", default="fit2.R")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    names = [d for d in args.datasets.split(",") if d]
    prepared, desc = {}, {}
    for d in names:
        try:
            recs = load(d, find_or_fetch(d, args.data_dir))
        except Exception as exc:  # a missing dataset leaves its predictions inconclusive, never "held"
            print(d, "could not be loaded:", repr(exc), flush=True)
            desc[d] = {"error": repr(exc)[:300]}
            continue
        print(d, "parsed rows:", len(recs), flush=True)
        sel, kcs, students = select(recs, args.kcs, args.max_students, MAX_T, args.seed)
        del recs
        if not sel:
            print(d, "no usable rows after selection", flush=True)
            desc[d] = {"error": "no usable rows after selection"}
            continue
        prepared[d] = (build_index(sel), students, {k: i for i, k in enumerate(kcs)})
        desc[d] = describe(sel, kcs, students)
        print(d, json.dumps(desc[d], sort_keys=True), flush=True)
    results, boots_done = {}, -1
    tmp = tempfile.mkdtemp(prefix="lrr2-")
    fit_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for b in range(args.boot + 1):
            if b > 0 and (time.time() - start) / 60.0 > args.max_minutes:
                print("time budget reached before bootstrap", b, flush=True)
                break
            jobs = []
            for di, d in enumerate(names):
                if d not in prepared:
                    continue
                by_student, students, kc_index = prepared[d]
                rng = random.Random(args.seed * 1000 + b * 10 + di)
                sample = list(students) if b == 0 else [rng.choice(students) for _ in students]
                for wname, wlen in WINDOWS:
                    path = os.path.join(tmp, "b%d-%s-%s.csv" % (b, d, wname))
                    rows = write_window_csv(by_student, sample, kc_index, wlen, path)
                    for m in MODELS:
                        jobs.append(((d, wname, m, b), rows, path,
                                     pool.submit(run_fit, args.rscript, args.fit_r, path, m)))
            for key, rows, path, fut in jobs:
                res = fut.result()
                res["rows"] = rows
                results[key] = res
                fit_rows.append(dict(res, dataset=key[0], window=key[1], model=key[2], boot=key[3]))
            for key, rows, path, fut in jobs:
                if os.path.exists(path):
                    os.remove(path)
            boots_done = b
            with open(os.path.join(args.out_dir, "fits2.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIT_FIELDS, extrasaction="ignore")
                w.writeheader()
                for f in fit_rows:
                    w.writerow(f)
            print("bootstrap %d done: %d fits, %d errors, %.1f min" % (
                b, len(fit_rows), sum(1 for f in fit_rows if f.get("error")), (time.time() - start) / 60.0),
                flush=True)
    n_boot = max(boots_done, 0)
    verd = verdicts(results, names, n_boot)
    table = cells(results, names, n_boot)
    meta = {"datasets": desc, "boots_done": n_boot, "boots_planned": args.boot, "kcs": args.kcs,
            "max_students": args.max_students, "max_t": MAX_T, "seed": args.seed, "fits": len(fit_rows),
            "errors": sum(1 for f in fit_rows if f.get("error")), "minutes": round((time.time() - start) / 60.0, 1)}
    with open(os.path.join(args.out_dir, "results2.json"), "w") as fh:
        json.dump({"meta": meta, "verdicts": verd, "cells": table}, fh, indent=1, sort_keys=True, default=str)
    write_md(table, verd, meta, os.path.join(args.out_dir, "results2.md"))
    for qid in sorted(verd):
        print(qid, verd[qid]["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
