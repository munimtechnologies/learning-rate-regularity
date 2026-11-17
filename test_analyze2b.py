# SPDX-License-Identifier: Apache-2.0
# Unit tests for analyze2b.py (phase 2b, the rescaling test). Made with AI agents working in MV2's name.
import csv
import os
import tempfile
import unittest

import analyze2b as a


def fit(dataset, window, model, boot, sd, g, error=0.0):
    return {"dataset": dataset, "window": window, "model": model, "boot": float(boot), "sd_delta": sd, "gbar": g,
            "error": error}


def real_rows(n_boot, lin=(0.3, 0.2, 0.1, 0.1), log=(0.1, 0.1, 0.1, 0.1), datasets=a.DATASETS):
    """lin/log = (sd first5, gbar first5, sd first25, gbar first25)."""
    rows = []
    for d in datasets:
        for b in range(n_boot + 1):
            for m, (s5, g5, s25, g25) in (("linear", lin), ("log", log)):
                rows.append(fit(d, "first5", m, b, s5, g5))
                rows.append(fit(d, "first10", m, b, 0.5, 0.5))
                rows.append(fit(d, "first25", m, b, s25, g25))
    return rows


class Ratios(unittest.TestCase):
    def test_pure_rescaling_gives_ewd_one(self):
        r = a.ratios({"sd_delta": 0.2, "gbar": 0.2}, {"sd_delta": 0.1, "gbar": 0.1})
        self.assertAlmostEqual(r["wdi"], 2.0)
        self.assertAlmostEqual(r["mr"], 2.0)
        self.assertAlmostEqual(r["ewd"], 1.0)

    def test_excess(self):
        r = a.ratios({"sd_delta": 0.3, "gbar": 0.2}, {"sd_delta": 0.1, "gbar": 0.1})
        self.assertAlmostEqual(r["ewd"], 1.5)

    def test_nonpositive_mean_slope(self):
        r = a.ratios({"sd_delta": 0.3, "gbar": 0.2}, {"sd_delta": 0.1, "gbar": -0.01})
        self.assertAlmostEqual(r["wdi"], 3.0)
        self.assertIsNone(r["mr"])
        self.assertIsNone(r["ewd"])

    def test_zero_long_sd(self):
        r = a.ratios({"sd_delta": 0.3, "gbar": 0.2}, {"sd_delta": 0.0, "gbar": 0.1})
        self.assertIsNone(r["wdi"])
        self.assertIsNone(r["ewd"])

    def test_failed_fit(self):
        self.assertIsNone(a.ratios({"sd_delta": 0.3, "gbar": 0.2, "error": 1.0}, {"sd_delta": 0.1, "gbar": 0.1}))
        self.assertIsNone(a.ratios(None, {"sd_delta": 0.1, "gbar": 0.1}))

    def test_r_est_depends_only_on_cv(self):
        self.assertAlmostEqual(a.r_est(0.1, 0.02), a.r_est(0.5, 0.1))
        self.assertIsNone(a.r_est(0.1, 0.2))


class Phase2(unittest.TestCase):
    def test_verdicts(self):
        summary, verd = a.phase2(real_rows(16))
        self.assertEqual(verd["S1"]["verdict"], "held")
        self.assertEqual(verd["S2"]["verdict"], "held")
        self.assertEqual(verd["S3"]["verdict"], "did not hold")
        self.assertAlmostEqual(summary[("assist09", "linear")]["full"]["ewd"], 1.5)
        self.assertEqual(summary[("assist09", "linear")]["boot_ewd_n"], 16)

    def test_threshold(self):
        _, verd = a.phase2(real_rows(16, lin=(0.26, 0.2, 0.1, 0.1)))   # EWD 1.3
        self.assertEqual(verd["S2"]["verdict"], "held")
        _, verd = a.phase2(real_rows(16, lin=(0.22, 0.2, 0.1, 0.1)))   # EWD 1.1: above 1, below 1.2
        self.assertEqual(verd["S2"]["verdict"], "did not hold")

    def test_too_few_boots_inconclusive(self):
        _, verd = a.phase2(real_rows(14))
        self.assertTrue(verd["S2"]["verdict"].startswith("inconclusive"))

    def test_missing_dataset_inconclusive(self):
        _, verd = a.phase2(real_rows(16, datasets=("assist09",)))
        self.assertTrue(verd["S1"]["verdict"].startswith("inconclusive"))

    def test_failed_full_fit_inconclusive(self):
        rows = real_rows(16)
        for r in rows:
            if r["dataset"] == "kdd_alg0506" and r["boot"] == 0.0 and r["window"] == "first25":
                r["error"] = 1.0
        _, verd = a.phase2(rows)
        self.assertTrue(verd["S2"]["verdict"].startswith("inconclusive"))

    def test_one_dataset_failing_beats_inconclusive(self):
        rows = real_rows(16)
        for r in rows:
            if r["dataset"] == "assist09" and r["model"] == "linear" and r["window"] == "first5":
                r["sd_delta"] = 0.2  # EWD 1.0 in assist09
        rows = [r for r in rows if not (r["dataset"] == "kdd_alg0506" and r["boot"] > 3)]
        _, verd = a.phase2(rows)
        self.assertEqual(verd["S2"]["verdict"], "did not hold")


class Phase1(unittest.TestCase):
    def rows(self, sd_full):
        out = []
        for rep in range(3):
            for m in ("linear", "log"):
                out.append({"rep": float(rep), "truth": "T2", "level": 0.5, "variant": "first5", "model": m,
                            "sd_delta": 0.2, "gbar": 0.2, "error": 0.0})
                out.append({"rep": float(rep), "truth": "T2", "level": 0.5, "variant": "full15", "model": m,
                            "sd_delta": sd_full, "gbar": 0.1, "error": 0.0})
        out.append({"rep": 0.0, "truth": "T2", "level": 0.5, "variant": "full15", "model": "linear",
                    "sd_delta": None, "gbar": None, "error": 1.0})
        return out

    def test_c1_held(self):
        table, verd = a.phase1(self.rows(0.05))
        self.assertEqual(verd["C1"]["verdict"], "held")
        row = next(t for t in table if t["model"] == "linear" and t["long"] == "full15")
        self.assertAlmostEqual(row["ewd"], 2.0)
        self.assertEqual(row["reps"], 3)

    def test_c1_not_held(self):
        _, verd = a.phase1(self.rows(0.1))
        self.assertEqual(verd["C1"]["verdict"], "did not hold")

    def test_c1_missing(self):
        _, verd = a.phase1([])
        self.assertTrue(verd["C1"]["verdict"].startswith("inconclusive"))


class EndToEnd(unittest.TestCase):
    def test_csv_round_trip_and_outputs(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "fits2.csv")
        fields = ("dataset", "window", "model", "boot", "rows", "sd_delta", "gbar", "error", "stderr_tail")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in real_rows(16):
                w.writerow(dict(r, rows=100, stderr_tail=""))
        rows = a.load_rows(path)
        self.assertEqual(rows[0]["dataset"], "kdd_alg0506")
        self.assertIsInstance(rows[0]["boot"], float)
        out = os.path.join(tmp, "out")
        self.assertEqual(a.main(["--fits2", path, "--out-dir", out]), 0)
        self.assertTrue(os.path.exists(os.path.join(out, "results2b.json")))
        with open(os.path.join(out, "results2b.md")) as fh:
            self.assertIn("| S2 |", fh.read())


if __name__ == "__main__":
    unittest.main()
