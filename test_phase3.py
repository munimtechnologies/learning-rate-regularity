# SPDX-License-Identifier: Apache-2.0
# Unit tests for phase3.py (standard library only; no R, no download). Made with AI agents working in MV2's name.
import math
import os
import tempfile
import unittest

import phase3 as p3


class Curves(unittest.TestCase):
    def test_covariates_values(self):
        g, a, h = p3.covariates(5, 2.0, 0.2)
        self.assertAlmostEqual(g, 0.6321205588, places=9)
        self.assertAlmostEqual(a, 1.2642411177, places=9)
        self.assertAlmostEqual(h, 0.7357588823, places=9)
        self.assertEqual(p3.covariates(0, 2.0, 0.2), (0.0, 0.0, 0.0))

    def test_h_is_the_time_scale_derivative(self):
        # d/dv of A (1 - exp(-c e^v t)) at v = 0 is h; d/dw of e^w A (1 - exp(-c t)) at w = 0 is a.
        eps = 1e-5
        for t, amp, c in ((3, 2.0, 0.1), (7, 1.5, 0.3), (20, 2.5, 0.05)):
            g, a, h = p3.covariates(t, amp, c)
            ts = lambda v: amp * (1 - math.exp(-c * math.exp(v) * t))  # noqa: E731
            am = lambda w: math.exp(w) * amp * (1 - math.exp(-c * t))  # noqa: E731
            self.assertAlmostEqual((ts(eps) - ts(-eps)) / (2 * eps), h, places=6)
            self.assertAlmostEqual((am(eps) - am(-eps)) / (2 * eps), a, places=6)

    def test_h_is_a_hump(self):
        hs = [p3.covariates(t, 2.0, 0.2)[2] for t in range(25)]
        peak = max(range(25), key=lambda t: hs[t])
        self.assertEqual(peak, 5)  # t = 1 / c
        self.assertLess(hs[24], hs[5] / 2)

    def test_speed_ratio(self):
        self.assertEqual(p3.speed_ratio(0.0), 1.0)
        self.assertIsNone(p3.speed_ratio(None))
        self.assertAlmostEqual(p3.speed_ratio(math.log(1.5) / p3.SPEED_K), 1.5, places=9)
        self.assertAlmostEqual(p3.speed_ratio(0.5), 1.9630, places=3)

    def test_best_curve(self):
        rows = [{"c": 0.03, "loglik": -100.0, "A": 1.0, "beta": 0.0, "error": 0.0},
                {"c": 0.12, "loglik": -90.0, "A": 2.0, "beta": -0.1, "error": 0.0},
                {"c": 0.9, "loglik": None, "A": None, "error": 1.0}]
        b = p3.best_curve(rows)
        self.assertEqual((b["c"], b["A"], b["n_ok"], b["edge"]), (0.12, 2.0, 2, False))
        self.assertTrue(p3.best_curve(rows[:1])["edge"])
        self.assertIsNone(p3.best_curve(rows[2:]))
        self.assertIsNone(p3.best_curve([]))


class Simulation(unittest.TestCase):
    def test_deterministic_and_sigma_zero(self):
        a = p3.simulate("ts", 0.5, 30, 3, 7)
        self.assertEqual(a, p3.simulate("ts", 0.5, 30, 3, 7))
        self.assertEqual(p3.simulate("ts", 0.0, 30, 3, 7), p3.simulate("amp", 0.0, 30, 3, 7))
        self.assertNotEqual(a, p3.simulate("amp", 0.5, 30, 3, 7))

    def test_stopping_rule(self):
        recs = p3.simulate("ts", 0.5, 60, 4, 11, max_t=25, run=3)
        seqs = {}
        for s, k, t, y in recs:
            seqs.setdefault((s, k), []).append((t, y))
        self.assertEqual(len(seqs), 240)
        for seq in seqs.values():
            ts = [t for t, _ in seq]
            self.assertEqual(ts, list(range(len(seq))))
            streak, first = 0, None
            for i, (_, y) in enumerate(seq):
                streak = streak + 1 if y else 0
                if streak >= 3 and first is None:
                    first = i
            if first is None:
                self.assertEqual(len(seq), 25)
            else:
                self.assertEqual(first, len(seq) - 1)


BY_STUDENT = {"S1": [("A", 0, 1), ("A", 5, 0), ("C", 0, 1)], "S2": [("C", 0, 0)]}


class Files(unittest.TestCase):
    def _read(self, path):
        with open(path) as fh:
            return fh.read().splitlines()

    def test_write_kc_csv(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            self.assertEqual(p3.write_kc_csv(BY_STUDENT, ["S1", "S2"], "C", path), 2)
            self.assertEqual(self._read(path), ["student,t,y", "0,0,1", "1,0,0"])
        finally:
            os.remove(path)

    def test_write_het_csv(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            n = p3.write_het_csv(BY_STUDENT, ["S1", "S1", "S2"], {"A": 0, "C": 1}, {"A": {"A": 2.0, "c": 0.2}}, path)
            self.assertEqual(n, 4)
            self.assertEqual(self._read(path), ["student,kc,t,y,g,a,h", "0,0,0,1,0,0,0",
                                                "0,0,5,0,0.632121,1.26424,0.735759", "1,0,0,1,0,0,0",
                                                "1,0,5,0,0.632121,1.26424,0.735759"])
        finally:
            os.remove(path)


def f(ll, sd_h=None, aic=None, sd_a=None):
    return {"loglik": ll, "sd_h": sd_h, "sd_a": sd_a, "aic": aic, "error": 0.0}


def build_fits(n_boot, ts_ll=-990.0, sd_h=0.5, both_aic=2006.0):
    fits = {}
    for d in ("d1", "d2"):
        fits[(d, 0, "lin")] = f(-1000.0, aic=2010.0)
        fits[(d, 0, "amp")] = f(-1000.0, aic=2010.0)
        fits[(d, 0, "ts")] = f(ts_ll, sd_h=sd_h, aic=2 * -ts_ll + 20)
        fits[(d, 0, "both")] = f(-997.0, aic=both_aic)
        for b in range(1, n_boot + 1):
            fits[(d, b, "amp")] = f(-1000.0)
            fits[(d, b, "ts")] = f(ts_ll, sd_h=sd_h)
    return fits


def build_sims(n=10, wrong_amp=0):
    sims = {}
    for j in range(n):
        sims[("ts", j, "ts")] = f(-100.0, sd_h=0.45)
        sims[("ts", j, "amp")] = f(-105.0)
        bad = j < wrong_amp
        sims[("amp", j, "amp")] = f(-105.0 if bad else -100.0, sd_a=0.4)
        sims[("amp", j, "ts")] = f(-100.0 if bad else -105.0)
    return sims


class Verdicts(unittest.TestCase):
    def test_all_held(self):
        v = p3.verdicts(build_fits(10), build_sims(), ["d1", "d2"], 10)
        for k in ("T1", "T2", "T3", "T4"):
            self.assertEqual(v[k]["verdict"], "held", k)
        self.assertEqual(v["T1"]["ts"]["correct"], 10)
        self.assertEqual(v["T2"]["d1"]["share_ts_better"], 1.0)
        self.assertAlmostEqual(v["T3"]["d2"]["speed_ratio"], math.exp(p3.SPEED_K * 0.5))

    def test_failures(self):
        v = p3.verdicts(build_fits(0, ts_ll=-1010.0, sd_h=0.2, both_aic=2012.0), build_sims(), ["d1", "d2"], 0)
        self.assertEqual(v["T1"]["verdict"], "held")
        self.assertEqual(v["T2"]["verdict"], "did not hold")
        self.assertEqual(v["T3"]["verdict"], "did not hold")  # ratio 1.31 < 1.5
        self.assertEqual(v["T4"]["verdict"], "did not hold")

    def test_inconclusive_without_boots(self):
        v = p3.verdicts(build_fits(5), build_sims(), ["d1", "d2"], 5)
        self.assertTrue(v["T2"]["verdict"].startswith("inconclusive"))
        self.assertTrue(v["T3"]["verdict"].startswith("inconclusive"))
        self.assertEqual(v["T4"]["verdict"], "held")
        v = p3.verdicts(build_fits(10), build_sims(), ["d1", "d2", "d3"], 10)
        self.assertTrue(v["T2"]["verdict"].startswith("inconclusive"))

    def test_gate(self):
        g = p3.gate(build_sims(wrong_amp=3))
        self.assertEqual((g["amp"]["correct"], g["amp"]["done"]), (7, 10))
        self.assertEqual(g["verdict"], "did not hold")
        self.assertTrue(p3.gate(build_sims(n=5)).get("verdict").startswith("inconclusive"))
        v = p3.verdicts(build_fits(10), build_sims(wrong_amp=3), ["d1", "d2"], 10)
        self.assertTrue(v["T2"]["verdict"].startswith("no verdict (T1 did not hold); rule alone: held"))

    def test_calibration(self):
        cal = p3.calibration(build_sims())
        self.assertAlmostEqual(cal["ts"]["median_over_true"], 0.9)
        self.assertAlmostEqual(cal["amp"]["median_over_true"], 0.8)
        self.assertEqual(cal["ts"]["n"], 10)

    def test_markdown(self):
        fits = build_fits(10)
        curves = {"d1": {"A": {"c": 0.12, "A": 2.0, "beta": 0.0, "edge": False, "n": 50, "kc_index": 0}}}
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            v = p3.verdicts(fits, build_sims(), ["d1", "d2"], 10)
            p3.write_md(v, p3.calibration(build_sims()), curves, fits, ["d1", "d2"], {"stage": "done"}, path)
            with open(path) as fh:
                text = fh.read()
            self.assertIn("| T3 |", text)
            self.assertIn("| d1 | ts | - | - | - | - | 0.5000 | 1.963 |", text)
            self.assertIn("| d1 | 0 | 50 | 0.120 | 2.000 | False |", text)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
