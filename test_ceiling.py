# SPDX-License-Identifier: Apache-2.0
# Unit tests for ceiling.py (THEORY-2b). Expected values were worked out by hand in THEORY-2b.md.
# Made with AI agents working in MV2's name.
import math
import os
import tempfile
import unittest

import ceiling as c


class Basics(unittest.TestCase):
    def test_ls_slope_of_a_line(self):
        self.assertAlmostEqual(c.ls_slope([3.0 + 0.5 * t for t in range(7)]), 0.5)

    def test_straight_line_limit(self):
        for W in (5, 15, 25):
            self.assertAlmostEqual(c.elasticity(1e-7, W), 1.0, places=4)

    def test_derivative_matches_finite_difference(self):
        for k, W in ((0.1, 5), (0.15, 15), (0.3, 10)):
            h = 1e-6
            num = (c.curve_slope(k + h, W) - c.curve_slope(k - h, W)) / (2 * h)
            self.assertAlmostEqual(c.curve_slope_dk(k, W), num, places=6)


class HandValues(unittest.TestCase):
    """Values computed by hand (THEORY-2b.md, section 3)."""
    def test_window5(self):
        self.assertAlmostEqual(c.elasticity(0.05, 5), 0.9028, delta=0.002)
        self.assertAlmostEqual(c.elasticity(0.10, 5), 0.8113, delta=0.002)
        self.assertAlmostEqual(c.elasticity(0.15, 5), 0.7254, delta=0.002)

    def test_window15(self):
        self.assertAlmostEqual(c.elasticity(0.05, 15), 0.6777, delta=0.002)
        self.assertAlmostEqual(c.elasticity(0.10, 15), 0.4096, delta=0.002)
        self.assertAlmostEqual(c.elasticity(0.15, 15), 0.1919, delta=0.002)

    def test_slopes(self):
        self.assertAlmostEqual(c.curve_slope(0.10, 5), 0.08234, delta=0.0002)
        self.assertAlmostEqual(c.curve_slope(0.10, 15), 0.05248, delta=0.0002)

    def test_pooled_and_first_order_ewd(self):
        e5 = c.pooled_elasticity(c.PHASE1_KS, 5)
        e15 = c.pooled_elasticity(c.PHASE1_KS, 15)
        self.assertAlmostEqual(e5, 0.7883, delta=0.002)
        self.assertAlmostEqual(e15, 0.3869, delta=0.002)
        self.assertAlmostEqual(e5 / e15, 2.037, delta=0.01)

    def test_exponent_falls_with_window(self):
        for k in (0.05, 0.1, 0.15):
            es = [c.elasticity(k, W) for W in (5, 10, 15, 25)]
            self.assertTrue(all(a > b for a, b in zip(es, es[1:])), es)


class Reversal(unittest.TestCase):
    def test_sign_change_window15(self):
        self.assertGreater(c.elasticity(0.20, 15), 0.0)
        self.assertLess(c.elasticity(0.25, 15), 0.0)
        k0 = c.zero_crossing(15)
        self.assertTrue(0.20 < k0 < 0.25, k0)
        self.assertAlmostEqual(c.elasticity(k0, 15), 0.0, places=6)

    def test_no_reversal_in_three_points(self):
        self.assertIsNone(c.zero_crossing(3))

    def test_continuous_root(self):
        u = c.continuous_root()
        self.assertAlmostEqual(u, 2.688, delta=0.01)
        self.assertAlmostEqual(4.0 - u, math.exp(-u) * (u * u + 3 * u + 4), places=6)


class Spread(unittest.TestCase):
    def test_amplitude_differences_are_pure_rescaling(self):
        amps = [0.5, 1.0, 1.5, 2.0]

        def cv(W):
            slopes = [c.ls_slope([a * math.log1p(t) for t in range(W)]) for a in amps]
            m = sum(slopes) / len(slopes)
            sd = math.sqrt(sum((s - m) ** 2 for s in slopes) / len(slopes))
            return sd / m
        self.assertAlmostEqual(cv(5), cv(25), places=10)

    def test_small_level_matches_first_order(self):
        level = 0.01
        m = c.student_moments(c.PHASE1_KS, 15, level)
        self.assertAlmostEqual(m["cv"] / level, c.pooled_elasticity(c.PHASE1_KS, 15), delta=0.008)

    def test_no_spread_at_level_zero(self):
        m = c.student_moments(c.PHASE1_KS, 15, 0.0)
        self.assertAlmostEqual(m["sd"], 0.0, places=12)


class Ratios(unittest.TestCase):
    def test_cv_round_trip(self):
        for r in (1.05, 1.14, 2.0):
            self.assertAlmostEqual(c.r_est_from_cv(c.cv_from_r(r)), r, places=10)

    def test_unbounded(self):
        self.assertIsNone(c.r_est_from_cv(2.0))

    def test_implied_true_ratio(self):
        self.assertAlmostEqual(c.implied_true_ratio(1.14, 1.0), 1.14, delta=0.002)
        self.assertAlmostEqual(c.implied_true_ratio(1.14, 0.387), 1.402, delta=0.005)
        self.assertIsNone(c.implied_true_ratio(1.14, 0.0))

    def test_true_ratio(self):
        self.assertAlmostEqual(c.true_ratio(0.5), 1.963, delta=0.002)


class Main(unittest.TestCase):
    def test_writes_tables(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(c.main(["--out-dir", d]), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "theory2b.json")))
            with open(os.path.join(d, "theory2b.md")) as fh:
                self.assertIn("Ceiling exponent", fh.read())


if __name__ == "__main__":
    unittest.main()
