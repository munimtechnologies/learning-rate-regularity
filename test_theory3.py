"""Tests for theory3.py (Apache-2.0; made with AI agents working in the name of MV2 of Munim, Inc.).

The numeric targets were worked out by hand in THEORY-3.md before this code ever ran, so these tests check the
note as well as the code.
"""
import math
import unittest

import theory3 as T

XS = (0.0, 0.1, 0.4, 0.9, 1.7, 2.5)
V = T.SQRT3 * 0.5  # 0.866..., the outer 3-point Gauss-Hermite node at s = 0.5


class Derivatives(unittest.TestCase):
    def test_hump_is_first_derivative(self):
        e = 1e-5
        for x in XS:
            fd = (T.gap_ts(e, x, 2.0) - T.gap_ts(-e, x, 2.0)) / (2 * e)
            self.assertAlmostEqual(fd, T.hump(x, 2.0), delta=1e-7)

    def test_second_derivative(self):
        e = 1e-3
        for x in XS:
            fd = (T.gap_ts(e, x) + T.gap_ts(-e, x)) / (e * e)
            self.assertAlmostEqual(fd, T.hump(x) * (1 - x), delta=1e-5)

    def test_third_derivative(self):
        e = 1e-2
        for x in XS:
            fd = (T.gap_ts(2 * e, x) - 2 * T.gap_ts(e, x) + 2 * T.gap_ts(-e, x) - T.gap_ts(-2 * e, x)) / (2 * e ** 3)
            self.assertAlmostEqual(fd, T.hump(x) * (1 - 3 * x + x * x), delta=3e-4)


class Projection(unittest.TestCase):
    def test_linear_gap_projects_exactly(self):
        pts = T.sim_like_design(6)
        self.assertAlmostEqual(T.project(pts, lambda A, x: 0.37 * T.hump(x, A) + 1.3), 0.37, delta=1e-12)

    def test_slope_zero_at_zero(self):
        self.assertAlmostEqual(T.slope(T.design(0.25, 6), 0.0), 0.0, delta=1e-15)

    def test_hand_slopes(self):
        d25, d10 = T.design(0.25, 6), T.design(0.10, 6)
        self.assertAlmostEqual(T.slope(d25, V), 0.72816, delta=0.003)
        self.assertAlmostEqual(T.slope(d25, -V), -0.79541, delta=0.003)
        self.assertAlmostEqual(T.slope(d10, V), 1.00175, delta=0.003)
        self.assertAlmostEqual(T.slope(d10, -V), -0.66836, delta=0.003)


class HeadStart(unittest.TestCase):
    def test_head_start_is_intercept_plus_amplitude(self):
        for A in (1.5, 3.0):
            for c in (0.05, 0.3, 0.9):
                for tau in (0.5, 2.0, 7.0):
                    q = 1 - math.exp(-c * tau)
                    for t in range(0, 25, 4):
                        self.assertAlmostEqual(T.gap_head(tau, c, t, A), A * q - q * T.amp(c * t, A), delta=1e-12)

    def test_iafm_reads_head_start_as_slower(self):
        self.assertAlmostEqual(T.iafm_head_start_slope(0.25, 6), -0.1410, delta=0.002)
        self.assertAlmostEqual(T.iafm_head_start_slope(0.10, 6), -0.0785, delta=0.002)


class Identification(unittest.TestCase):
    def test_hand_correlations(self):
        for c, r_amp, r_lin in ((0.10, 0.99562, 0.98622), (0.25, 0.94742, 0.87581), (0.50, 0.64213, 0.35905)):
            pts = T.design(c, 6)
            self.assertAlmostEqual(T.corr(pts, T._h, T._a), r_amp, delta=0.003)
            self.assertAlmostEqual(T.corr(pts, T._h, T._lin), r_lin, delta=0.003)


class Bias(unittest.TestCase):
    def test_hand_kappa(self):
        for c, k in ((0.10, -0.0479), (0.25, -0.5217), (0.50, -0.5054)):
            self.assertAlmostEqual(T.moments(T.design(c, 6))["kappa"], k, delta=0.005)

    def test_second_order_matches_integral_at_small_s(self):
        s = 0.05
        for c in (0.25, 0.5):
            pts = T.design(c, 6)
            self.assertAlmostEqual(T.sd_ratio(pts, s), 1 + T.moments(pts)["kappa"] * s * s, delta=5e-5)

    def test_hand_gauss_hermite(self):
        self.assertAlmostEqual(T.gh3_sd_ratio(T.design(0.25, 6), 0.5), 0.8802, delta=0.005)
        self.assertAlmostEqual(T.gh3_sd_ratio(T.design(0.10, 6), 0.5), 0.9770, delta=0.005)

    def test_full_integral_in_band(self):
        r25 = T.sd_ratio(T.design(0.25, 6), 0.5)
        r10 = T.sd_ratio(T.design(0.10, 6), 0.5)
        self.assertTrue(0.83 <= r25 <= 0.93, r25)
        self.assertTrue(0.94 <= r10 <= 1.02, r10)

    def test_report_runs(self):
        text = T.report()
        self.assertIn("corr(ts, amp)", text)
        self.assertEqual(text.count("\n| c = "), 4)


if __name__ == "__main__":
    unittest.main()
