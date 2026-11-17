# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity, phase 2b: the rescaling test (preregistered in PLAN-2b.md).
# Made with AI agents working in the name of MV2 of Munim, Inc.
#
# Reads the per-fit tables that phases 1 and 2 already write (fits.csv from sim.py, fits2.csv from realdata.py)
# and splits the window dependence of the student learning-rate spread into two parts:
#   WDI = SD(slope, short window) / SD(slope, long window)        (phase 2's window-dependence index)
#   MR  = mean slope (short) / mean slope (long)                  (how much the curve bends: pure rescaling)
#   EWD = WDI / MR = CV(short) / CV(long)                         (excess window dependence, CV = SD / mean)
# If a concave curve only rescales every student's slope by the same factor, EWD = 1 and the PNAS-style speed
# ratio R_est (a function of CV alone) does not depend on the window. EWD > 1 means relative differences in
# learning speed look larger in short windows than in long ones. Standard library only; no new fits.
import argparse
import csv
import json
import math
import os
import sys

Z75 = 0.6744897501960817
SHORT, LONG = "first5", "first25"          # phase 2 windows
SIM_SHORT = "first5"                       # phase 1 short window
SIM_VARIANTS = ("full15", "first10", "three_in_a_row", "bkt_mastery")
DATASETS = ("kdd_alg0506", "assist09")
MODELS = ("linear", "log")
EWD_MIN = 1.2
SHARE = 0.9
MIN_BOOT = 15
C1_MIN = 1.1
TEXT_FIELDS = ("dataset", "window", "model", "truth", "variant", "stderr_tail")


def num(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in ("", "NA", "NaN", "nan", "None"):
        return None
    if x in ("TRUE", "True"):
        return True
    if x in ("FALSE", "False"):
        return False
    try:
        return float(x)
    except ValueError:
        return None


def load_rows(path):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({k: (v.strip() if k in TEXT_FIELDS and v is not None else num(v)) for k, v in r.items()})
    return out


def ok(r):
    return bool(r) and not r.get("error") and r.get("sd_delta") is not None and r.get("gbar") is not None


def ratios(short, long):
    """WDI, MR and EWD for one pair of fits; None for a part that is undefined (failed fit, zero long-window
    SD, or a non-positive mean slope)."""
    if not (ok(short) and ok(long)):
        return None
    sd5, sd25, g5, g25 = short["sd_delta"], long["sd_delta"], short["gbar"], long["gbar"]
    wdi = sd5 / sd25 if sd25 > 0 else None
    mr = g5 / g25 if (g5 > 0 and g25 > 0) else None
    ewd = wdi / mr if (wdi is not None and mr is not None) else None
    return {"wdi": wdi, "mr": mr, "ewd": ewd}


def r_est(gbar, sd):
    if gbar is None or sd is None:
        return None
    lo = gbar - Z75 * sd
    if lo <= 0:
        return None
    return (gbar + Z75 * sd) / lo


def combine(flags):
    if any(f is False for f in flags):
        return "did not hold"
    if any(f is None for f in flags):
        return "inconclusive (a fit failed, a mean slope was not positive, or fewer than %d usable bootstraps)" \
            % MIN_BOOT
    return "held"


def phase2(rows, datasets=DATASETS):
    idx = {}
    for r in rows:
        if r.get("boot") is None:
            continue
        idx[(r.get("dataset"), r.get("window"), r.get("model"), int(r["boot"]))] = r
    boots = sorted({k[3] for k in idx if k[3] > 0})
    summary = {}
    for d in datasets:
        for m in MODELS:
            short, long = idx.get((d, SHORT, m, 0)), idx.get((d, LONG, m, 0))
            full = ratios(short, long)
            bs = [ratios(idx.get((d, SHORT, m, b)), idx.get((d, LONG, m, b))) for b in boots]
            bs = [x for x in bs if x is not None]
            ewds = [x["ewd"] for x in bs if x["ewd"] is not None]
            mrs = [x["mr"] for x in bs if x["mr"] is not None]
            summary[(d, m)] = {
                "full": full,
                "boot_ewd_n": len(ewds),
                "boot_share_ewd_gt1": (sum(1 for e in ewds if e > 1) / len(ewds)) if ewds else None,
                "boot_ewd_90": [pct(ewds, 0.05), pct(ewds, 0.95)] if ewds else None,
                "boot_mr_n": len(mrs),
                "boot_share_mr_gt1": (sum(1 for e in mrs if e > 1) / len(mrs)) if mrs else None,
                "r_est_short": r_est(short.get("gbar"), short.get("sd_delta")) if ok(short) else None,
                "r_est_long": r_est(long.get("gbar"), long.get("sd_delta")) if ok(long) else None,
            }
    verd = {}
    for qid, model, part, rule in (("S1", "linear", "mr", None), ("S2", "linear", "ewd", EWD_MIN),
                                   ("S3", "log", "ewd", EWD_MIN)):
        flags, detail = [], {}
        for d in datasets:
            s = summary[(d, model)]
            full = s["full"]
            point = full[part] if full else None
            n, share = (s["boot_mr_n"], s["boot_share_mr_gt1"]) if part == "mr" else \
                (s["boot_ewd_n"], s["boot_share_ewd_gt1"])
            detail[d] = {"full_sample": point, "boot_n": n, "boot_share_gt1": share}
            if point is None or n < MIN_BOOT:
                flags.append(None)
            else:
                cut = 1.0 if rule is None else rule
                point_ok = point > cut if rule is None else point >= cut
                flags.append(point_ok and share >= SHARE)
        verd[qid] = dict(detail, verdict=combine(flags))
    return summary, verd


def phase1(rows):
    """Mean SD(slope) and mean slope over replications for each truth, level, variant and model; then EWD of
    first5 against each other variant. Descriptive, except C1."""
    groups = {}
    for r in rows:
        if not ok(r) or r.get("level") is None:
            continue
        key = (r.get("truth"), round(r["level"], 4), r.get("variant"), r.get("model"))
        groups.setdefault(key, []).append(r)
    means = {k: (sum(f["sd_delta"] for f in v) / len(v), sum(f["gbar"] for f in v) / len(v), len(v))
             for k, v in groups.items()}
    table = []
    for (truth, level, variant, model) in sorted(k for k in means if k[2] == SIM_SHORT):
        sd5, g5, n5 = means[(truth, level, variant, model)]
        for other in SIM_VARIANTS:
            o = means.get((truth, level, other, model))
            if o is None:
                continue
            res = ratios({"sd_delta": sd5, "gbar": g5}, {"sd_delta": o[0], "gbar": o[1]})
            table.append({"truth": truth, "level": level, "model": model, "long": other, "reps": min(n5, o[2]),
                          "wdi": res["wdi"], "mr": res["mr"], "ewd": res["ewd"]})
    c1 = next((t for t in table if t["truth"] == "T2" and t["level"] == 0.5 and t["model"] == "linear"
               and t["long"] == "full15"), None)
    if c1 is None or c1["ewd"] is None:
        v = "inconclusive (no usable T2 level 0.5 cells)"
    else:
        v = "held" if c1["ewd"] > C1_MIN else "did not hold"
    return table, {"C1": {"ewd_T2_0.5_linear_first5_vs_full15": None if c1 is None else c1["ewd"], "verdict": v}}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("%." + str(nd) + "f") % x
    if isinstance(x, list):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    return str(x)


def write_md(summary, verd, sim_table, sim_verd, path):
    lines = ["# learning-rate-regularity phase 2b: the rescaling test (output)", "",
             "Preregistered in PLAN-2b.md. Made with AI agents working in MV2's name. Generated by analyze2b.py.",
             "", "## Verdicts", "", "| id | result | verdict |", "|---|---|---|"]
    for qid in sorted(verd):
        v = dict(verd[qid])
        verdict = v.pop("verdict")
        lines.append("| %s | %s | %s |" % (qid, json.dumps(v, sort_keys=True), verdict))
    for qid in sorted(sim_verd):
        v = dict(sim_verd[qid])
        verdict = v.pop("verdict")
        lines.append("| %s | %s | %s |" % (qid, json.dumps(v, sort_keys=True), verdict))
    lines += ["", "## Real data (first5 against first25, all selected students)", "",
              "| dataset | model | WDI | MR | EWD | EWD 90% (boot) | boot n | R_est first5 | R_est first25 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (d, m), s in sorted(summary.items()):
        f = s["full"] or {}
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            d, m, fmt(f.get("wdi")), fmt(f.get("mr")), fmt(f.get("ewd")), fmt(s["boot_ewd_90"]), s["boot_ewd_n"],
            fmt(s["r_est_short"]), fmt(s["r_est_long"])))
    if sim_table:
        lines += ["", "## Simulation calibration (phase 1; first5 against each longer design, means over reps)", "",
                  "| truth | level | model | long design | reps | WDI | MR | EWD |", "|---|---|---|---|---|---|---|---|"]
        for t in sim_table:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t["truth"], fmt(t["level"], 2), t["model"], t["long"], t["reps"], fmt(t["wdi"]), fmt(t["mr"]),
                fmt(t["ewd"])))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits2", required=True, help="fits2.csv from realdata.py (phase 2)")
    ap.add_argument("--fits1", default=None, help="fits.csv from sim.py (phase 1), optional")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    summary, verd = phase2(load_rows(args.fits2))
    sim_table, sim_verd = ([], {})
    if args.fits1 and os.path.exists(args.fits1):
        sim_table, sim_verd = phase1(load_rows(args.fits1))
    out = {"verdicts": verd, "sim_verdicts": sim_verd, "sim_table": sim_table,
           "real": [dict(v, dataset=k[0], model=k[1]) for k, v in sorted(summary.items())]}
    with open(os.path.join(args.out_dir, "results2b.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    write_md(summary, verd, sim_table, sim_verd, os.path.join(args.out_dir, "results2b.md"))
    for qid in sorted(verd):
        print(qid, verd[qid]["verdict"])
    for qid in sorted(sim_verd):
        print(qid, sim_verd[qid]["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
