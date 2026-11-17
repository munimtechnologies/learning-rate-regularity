# SPDX-License-Identifier: Apache-2.0
# Unit tests for sim.py (standard library only; no R needed). Made with AI agents working in MV2's name.
import math
import os
import tempfile
import unittest

import sim


class Basics(unittest.TestCase):
    def test_sigmoid_logit(self):
        self.assertAlmostEqual(sim.sigmoid(0.0), 0.5)
        self.assertAlmostEqual(sim.sigmoid(-50.0) + sim.sigmoid(50.0), 1.0)
        self.assertAlmostEqual(sim.logit(sim.sigmoid(1.3)), 1.3, places=9)

    def test_bkt_update_by_hand(self):
        # pL 0.4, correct: 0.36 / 0.48 = 0.75, then + 0.25 * 0.15 = 0.7875
        self.assertAlmostEqual(sim.bkt_update(0.4, 1, 0.15, 0.2, 0.1), 0.7875, places=9)
        # pL 0.4, wrong: 0.04 / 0.52 = 0.0769231, then + 0.9230769 * 0.15 = 0.2153846
        self.assertAlmostEqual(sim.bkt_update(0.4, 0, 0.15, 0.2, 0.1), 0.2153846, places=6)


class Stopping(unittest.TestCase):
    def test_three_in_a_row(self):
        self.assertEqual(sim.cut_three_in_a_row([1, 1, 1, 1, 0]), [1, 1, 1])
        self.assertEqual(sim.cut_three_in_a_row([0, 1, 1, 0, 1, 1, 1, 0]), [0, 1, 1, 0, 1, 1, 1])
        self.assertEqual(sim.cut_three_in_a_row([1, 0, 1, 0]), [1, 0, 1, 0])
        self.assertEqual(sim.cut_three_in_a_row([]), [])

    def test_bkt_mastery_by_hand(self):
        # 0.4 -> 0.7875 -> 0.9519 (>= 0.95): stops after two correct answers
        self.assertEqual(sim.cut_bkt_mastery([1, 1, 1, 0]), [1, 1])
        # 0.4 -> 0.2154 -> 0.6197 -> 0.8980 -> 0.9791: stops after the fifth answer
        self.assertEqual(sim.cut_bkt_mastery([0, 1, 1, 1, 1, 0, 0]), [0, 1, 1, 1, 1])
        self.assertEqual(sim.cut_bkt_mastery([0, 0, 0]), [0, 0, 0])

    def test_variants_are_prefixes(self):
        seqs = sim.generate("T3", 0.5, 6, 3, sim.MAX_OPP, 7)
        for ys in seqs.values():
            self.assertEqual(len(ys), sim.MAX_OPP)
            for variant in sim.VARIANTS:
                cut = sim.apply_variant(ys, variant)
                self.assertEqual(cut, ys[:len(cut)])
                self.assertGreaterEqual(len(cut), 1)
            self.assertEqual(len(sim.apply_variant(ys, "first5")), 5)
            self.assertEqual(len(sim.apply_variant(ys, "first10")), 10)
        with self.assertRaises(ValueError):
            sim.apply_variant([1], "nope")


class Generation(unittest.TestCase):
    def test_deterministic_and_binary(self):
        a = sim.generate("T2", 0.3, 5, 4, 15, 11)
        b = sim.generate("T2", 0.3, 5, 4, 15, 11)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertTrue(all(y in (0, 1) for ys in a.values() for y in ys))

    def test_respond_sequence_extremes(self):
        u = [0.5] * 4
        v = [0.0] * 5
        # T1 with huge positive logit: all correct; huge negative: all wrong
        self.assertEqual(sim.respond_sequence("T1", 0.0, 30.0, 0.0, 0.0, 0.1, 0.1, 0.1, u, v), [1, 1, 1, 1])
        self.assertEqual(sim.respond_sequence("T1", 0.0, -30.0, 0.0, 0.0, 0.1, 0.1, 0.1, u, v), [0, 0, 0, 0])
        # T3: v[0] = 0 < p0 so the skill is known from the start; u = 0.5 < 0.9 so all correct
        self.assertEqual(sim.respond_sequence("T3", 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, u, v), [1, 1, 1, 1])
        # T3: unknown and never learned (v all 1.0): correct only when u < guess 0.2
        u2 = [0.1, 0.5, 0.1, 0.9]
        v2 = [1.0] * 5
        self.assertEqual(sim.respond_sequence("T3", 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, u2, v2), [1, 0, 1, 0])
        # T2 at t = 0 has no gain yet: sigmoid(0) = 0.5, so u = 0.4 is correct and u = 0.6 is wrong
        self.assertEqual(sim.respond_sequence("T2", 0.5, 0.0, 1.0, 0.0, 0.1, 0.1, 0.1, [0.4], [0.0, 0.0]), [1])
        self.assertEqual(sim.respond_sequence("T2", 0.5, 0.0, 1.0, 0.0, 0.1, 0.1, 0.1, [0.6], [0.0, 0.0]), [0])
        with self.assertRaises(ValueError):
            sim.respond_sequence("T9", 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, u, v)

    def test_write_csv(self):
        seqs = {(0, 0): [0, 1, 1, 1, 0], (0, 1): [1, 1, 1, 0, 0]}
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            rows = sim.write_csv(seqs, "three_in_a_row", path)
            self.assertEqual(rows, 4 + 3)
            with open(path) as fh:
                lines = fh.read().splitlines()
            self.assertEqual(lines[0], "student,kc,t,y")
            self.assertEqual(lines[1], "0,0,0,0")
            self.assertEqual(lines[-1], "0,1,2,1")
        finally:
            os.remove(path)


class Ratios(unittest.TestCase):
    def test_true_ratio(self):
        for truth in ("T1", "T2", "T3"):
            self.assertEqual(sim.true_ratio(truth, 0.0), 1.0)
        self.assertAlmostEqual(sim.true_ratio("T1", 0.05), 2.0177, places=3)
        self.assertAlmostEqual(sim.true_ratio("T2", 0.5), 1.9630, places=3)
        r3 = sim.true_ratio("T3", 0.5)
        self.assertTrue(1.75 < r3 < 1.82)

    def test_r_est(self):
        self.assertAlmostEqual(sim.r_est(0.1, 0.02), 1.3119, places=3)
        self.assertEqual(sim.r_est(0.1, 0.0), 1.0)
        self.assertIsNone(sim.r_est(0.1, 0.2))
        self.assertIsNone(sim.r_est(None, 0.1))

    def test_parse_numbers(self):
        self.assertIsNone(sim._num("NA"))
        self.assertIs(sim._num("TRUE"), True)
        self.assertEqual(sim._num("0.25"), 0.25)


def fake_fits(truth, level, variant, sd, gbar, n=6, model="linear"):
    return [{"rep": i, "truth": truth, "level": level, "variant": variant, "model": model, "sd_delta": sd,
             "sd_theta": 1.0, "gbar": gbar, "singular": False, "error": 0.0} for i in range(n)]


class Verdicts(unittest.TestCase):
    def build(self, overrides=None):
        spec = {
            ("T1", 0.05, "full15"): (0.05, 0.1),
            ("T1", 0.05, "three_in_a_row"): (0.05, 0.1),
            ("T1", 0.05, "bkt_mastery"): (0.055, 0.1),
            ("T1", 0.05, "first10"): (0.045, 0.1),
            ("T2", 0.5, "first5"): (0.06, 0.2),
            ("T2", 0.5, "full15"): (0.03, 0.12),
            ("T2", 0.5, "three_in_a_row"): (0.03, 0.12),
            ("T2", 0.5, "bkt_mastery"): (0.03, 0.12),
            ("T3", 0.5, "full15"): (0.02, 0.12),
            ("T2", 0.0, "full15"): (0.005, 0.12),
            ("T3", 0.0, "full15"): (0.03, 0.12),
        }
        spec.update(overrides or {})
        fits = []
        for (truth, level, variant), (sd, gbar) in spec.items():
            fits += fake_fits(truth, level, variant, sd, gbar)
        return sim.verdicts(sim.summarize(fits))

    def test_all_held(self):
        v = self.build()
        for pid in ("P0", "P1", "P1b", "P2", "P3", "P4", "P5"):
            self.assertEqual(v[pid]["verdict"], "held", pid)
        # T2 0.5 cells: r_est(0.12, 0.03) = 0.1402 / 0.0998 = 1.405 > 1.25; T3 0.5: r_est(0.12, 0.02) = 1.252
        self.assertEqual(v["P6"]["verdict"], "held")
        self.assertEqual(v["P6"]["cells_checked"], 4)

    def test_failures(self):
        v = self.build({("T1", 0.05, "bkt_mastery"): (0.03, 0.1),   # ratio 0.6
                        ("T2", 0.5, "first5"): (0.04, 0.2),         # ratio 1.33 < 1.5
                        ("T3", 0.5, "full15"): (0.01, 0.12),        # r_est 1.119 -> attenuation ~0.15, P6 hit
                        ("T3", 0.0, "full15"): (0.001, 0.12)})      # r_est ~1.01 <= 1.15
        self.assertEqual(v["P1"]["verdict"], "did not hold")
        self.assertEqual(v["P1b"]["verdict"], "held")
        self.assertEqual(v["P2"]["verdict"], "did not hold")
        self.assertEqual(v["P3"]["verdict"], "held")
        self.assertEqual(v["P5"]["verdict"], "did not hold")
        self.assertEqual(v["P6"]["verdict"], "did not hold")
        self.assertEqual(v["P6"]["cells_at_or_below_1.25"], ["T3|0.5|full15|linear"])

    def test_too_few_fits(self):
        fits = fake_fits("T1", 0.05, "full15", 0.05, 0.1, n=3)
        v = sim.verdicts(sim.summarize(fits))
        self.assertTrue(v["P0"]["verdict"].startswith("inconclusive"))
        self.assertTrue(v["P6"]["verdict"].startswith("inconclusive"))

    def test_errors_and_unbounded(self):
        fits = fake_fits("T2", 0.5, "full15", 0.5, 0.1)
        fits.append({"rep": 9, "truth": "T2", "level": 0.5, "variant": "full15", "model": "linear",
                     "sd_delta": None, "gbar": None, "error": 1.0})
        cells = sim.summarize(fits)
        c = cells["T2|0.5|full15|linear"]
        self.assertEqual((c["n_fits"], c["n_ok"], c["n_unbounded"]), (7, 6, 6))
        self.assertTrue(math.isinf(c["median_r_est"]))


class Markdown(unittest.TestCase):
    def test_write_md(self):
        fits = fake_fits("T1", 0.05, "full15", 0.05, 0.1)
        cells = sim.summarize(fits)
        verd = sim.verdicts(cells)
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            sim.write_md(cells, verd, {"reps_done": 6}, path)
            with open(path) as fh:
                text = fh.read()
            self.assertIn("| P0 |", text)
            self.assertIn("| T1 | 0.05 | full15 | linear | 6/6 | 0 | 0.0500 |", text)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
