"""Tests for theory4.py (THEORY-4 of MV2's learning-rate-regularity project). Standard library only.

Every hand number in THEORY-4.md is checked here with a stated tolerance.
Copyright 2026 Munim, Inc. Apache-2.0. Made with AI agents working in the name of MV2 of Munim, Inc.
"""
import math
import unittest

import theory4 as T


def simpson(f, a, b, n=2000):
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3.0


class ContinuousLS(unittest.TestCase):
    def test_I_matches_numeric_integral(self):
        for X in (0.01, 0.5, 1.25, 3.0, 6.0):
            num = simpson(lambda x: (x - X / 2.0) * (1.0 - math.exp(-x)), 0.0, X)
            self.assertAlmostEqual(T.I_int(X), num, delta=1e-10 + 1e-8 * abs(num))

    def test_series_branch_is_continuous(self):
        # the small-X series and the closed form agree just around the switch at X = 0.02
        lo, hi = T.I_int(0.0199999), T.I_int(0.0200001)
        self.assertAlmostEqual(lo / hi, 1.0, delta=1e-4)
        self.assertAlmostEqual(T.eta(0.0199999), T.eta(0.0200001), delta=1e-4)
        self.assertAlmostEqual(T.eta(0.001), -0.0005, delta=1e-6)

    def test_S_hand_values(self):
        # S(0.5): I = 0.25 - 1 + 0.6065306597 * 1.25 = 0.0081633246; 12 I / 0.125
        self.assertAlmostEqual(T.S_ls(0.5), 0.783679, delta=2e-6)
        # S(1.25): I = 0.625 - 1 + 0.2865047969 * 1.625 = 0.0905702950; 12 I / 1.953125
        self.assertAlmostEqual(T.S_ls(1.25), 0.556464, delta=2e-6)
        # S(3): I = 0.5 + 0.0497870684 * 2.5 = 0.624467671; 12 I / 27
        self.assertAlmostEqual(T.S_ls(3.0), 0.277541, delta=2e-6)

    def test_eta_hand_values_and_derivative(self):
        self.assertAlmostEqual(T.eta(0.5), -0.237522, delta=3e-6)
        self.assertAlmostEqual(T.eta(1.25), -0.547732, delta=3e-6)
        self.assertAlmostEqual(T.eta(3.0), -1.076317, delta=3e-6)
        h = 1e-5
        for X in (0.5, 1.25, 3.0):
            num = X * (math.log(T.S_ls(X + h)) - math.log(T.S_ls(X - h))) / (2 * h)
            self.assertAlmostEqual(T.eta(X), num, delta=1e-6)

    def test_reversal_point_matches_theory_2b(self):
        # THEORY-2b: e = 1 + eta = 0 at u ~ 2.69, the root of 4 - u = exp(-u)(u^2 + 3u + 4)
        self.assertGreater(T.eta(2.6), -1.0)
        self.assertLess(T.eta(2.8), -1.0)

    def test_window_scaling_formula(self):
        # b(c) = c S(c T(c)) with T = K c^(-lam); at c = 0.25, K = 2.5, lam = 0.5 the window ends at X = 1.25
        def e_num(lam, K, c=0.25, h=1e-4):
            b = lambda cc: T.slope_continuous(cc, K * cc ** (-lam))
            return (math.log(b(c * math.exp(h))) - math.log(b(c * math.exp(-h)))) / (2 * h)
        self.assertAlmostEqual(e_num(0.5, 2.5), 0.726134, delta=2e-6)
        self.assertAlmostEqual(T.exponent_scaled(1.25, 0.5), 0.726134, delta=2e-6)
        # lam = 1 (window shrinks in proportion to speed): no attenuation at all, for any window end
        for K in (0.5, 1.25, 3.0, 6.0):
            self.assertAlmostEqual(e_num(1.0, K), 1.0, delta=1e-7)
        # lam = 0 (fixed window) is THEORY-2b's exponent 1 + eta(c W)
        self.assertAlmostEqual(e_num(0.0, 5.0), 1.0 + T.eta(1.25), delta=2e-6)

    def test_theta_channel(self):
        # absolute level reached at x* = 1.25 by the typical student (A = 2, theta = 0)
        A, xs = 2.0, 1.25
        Lstar = A * (1.0 - math.exp(-xs))
        self.assertAlmostEqual(T.x_star(Lstar, 0.0, A), xs, delta=1e-12)
        h = 1e-5
        b = lambda th: T.S_ls(T.x_star(Lstar, th, A))
        num = (math.log(b(h)) - math.log(b(-h))) / (2 * h)
        self.assertAlmostEqual(T.theta_channel(xs, A), num, delta=1e-6)
        # hand: 0.547732 / (1.25 * 2 * 0.2865048) = 0.764711
        self.assertAlmostEqual(T.theta_channel(1.25, 2.0), 0.764711, delta=2e-5)
        # hand: 1.076317 / (3 * 2 * 0.0497871) = 3.60307
        self.assertAlmostEqual(T.theta_channel(3.0, 2.0), 3.60307, delta=1e-4)

    def test_head_start_channel(self):
        # window [x0, x*]: slope in x units is exp(-x0) S(x* - x0)
        xs, h = 3.0, 1e-5
        b = lambda x0: math.exp(-x0) * T.S_ls(xs - x0)
        num = (math.log(b(1.75 + h)) - math.log(b(1.75 - h))) / (2 * h)
        self.assertAlmostEqual(T.head_start_channel(1.25), num, delta=1e-6)
        self.assertAlmostEqual(T.head_start_channel(1.25), -0.561814, delta=3e-6)
        self.assertAlmostEqual(T.head_start_channel(3.0), -0.641228, delta=3e-6)


class Survival(unittest.TestCase):
    def test_three_in_a_row_hand_values(self):
        pi = T.survival_run([0.5] * 8, 3)
        want = [1.0, 1.0, 1.0, 0.875, 0.8125, 0.75, 0.6875, 0.6328125]
        for a, b in zip(pi, want):
            self.assertAlmostEqual(a, b, delta=1e-12)

    def test_three_in_a_row_expected_length(self):
        # E[length] = (1 - p^3) / ((1 - p) p^3): 14 at p = 0.5, 4.765625 at p = 0.8
        self.assertAlmostEqual(sum(T.survival_run([0.5] * 1000, 3)), 14.0, delta=1e-9)
        self.assertAlmostEqual(sum(T.survival_run([0.8] * 1000, 3)), 4.765625, delta=1e-9)

    def test_bkt_rule(self):
        # 0.4 -> 0.7875 -> 0.9519 (>= 0.95): always-correct students stop after two answers (as in test_sim.py)
        self.assertEqual(T.survival_bkt([1.0] * 5), [1.0, 1.0, 0.0, 0.0, 0.0])
        # never-correct students never reach mastery
        self.assertEqual(T.survival_bkt([0.0] * 5), [1.0] * 5)
        pi = T.survival_bkt([0.6] * 10)
        self.assertTrue(all(pi[i + 1] <= pi[i] + 1e-15 for i in range(9)))
        self.assertAlmostEqual(pi[2], 1.0 - 0.36, delta=1e-12)  # two correct answers end practice

    def test_level_rule(self):
        Ls = [0.0, 0.5, 0.9, 1.2, 1.4]
        self.assertEqual(T.survival_level(Ls, 1.0), [1.0, 1.0, 1.0, 1.0, 0.0])


class Fits(unittest.TestCase):
    def test_wls_exact_line(self):
        ts = list(range(10))
        ys = [0.3 + 0.07 * t for t in ts]
        ws = T.survival_run([0.7] * 10, 3)
        self.assertAlmostEqual(T.wls_slope(ts, ys, ws), 0.07, delta=1e-12)

    def test_wlogit_recovers_a_true_logistic_line_under_any_stopping(self):
        # ignorability: if the straight line is the true model, survival weights do not move the fit
        ts = list(range(15))
        ps = [T.sigmoid(-0.5 + 0.2 * t) for t in ts]
        for ws in (T.survival_run(ps, 3), [1.0] * 15, T.survival_bkt(ps)):
            a, b = T.wlogit_fit(ts, ps, ws)
            self.assertAlmostEqual(a, -0.5, delta=1e-8)
            self.assertAlmostEqual(b, 0.2, delta=1e-8)

    def test_level_rule_is_self_similar_and_fixed_window_reverses(self):
        # stop at 95% of the gain (x* = ln 20); speeds 1 and 2 at c = 0.02: slope ratio ~ 2 (continuum: exactly 2)
        T.RULES["level95_test"] = ("levelrel", 0.95, 1000)
        b1 = T.slope("level95_test", 0.0, 1.0, 0.02, "ls")
        b2 = T.slope("level95_test", 0.0, 1.0, 0.04, "ls")
        self.assertAlmostEqual(b2 / b1, 2.0, delta=0.06)
        # the same speeds in one fixed window of 150 opportunities: the faster student looks slower
        # (continuum 2 S(5.96) / S(2.98) ~ 0.81)
        T.RULES["fixed150_test"] = ("fixed", 150)
        f1 = T.slope("fixed150_test", 0.0, 1.0, 0.02, "ls")
        f2 = T.slope("fixed150_test", 0.0, 1.0, 0.04, "ls")
        self.assertLess(f2 / f1, 0.9)
        self.assertGreater(f2 / f1, 0.7)

    def test_summaries_run(self):
        R, e = T.effective_exponent("run3_25", -0.5, 2.0, 0.1, "logit")
        self.assertTrue(R is not None and R > 0)
        lam = T.window_scaling("fixed15", -0.5, 2.0, 0.1)
        self.assertAlmostEqual(lam, 0.0, delta=1e-12)
        lam = T.window_scaling("run3_200", -0.5, 2.0, 0.1)
        self.assertTrue(0.0 < lam < 1.0)


if __name__ == "__main__":
    unittest.main()
