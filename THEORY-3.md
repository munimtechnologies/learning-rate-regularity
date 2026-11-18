# THEORY-3: what phase 3 can see, and how its time-scale estimate is biased

Written 2026-11-18 by MV2 of Munim, Inc., before any phase 1, 2 or 3 fit has run. Made with AI agents working in
his name. Code (`theory3.py`, `test_theory3.py`) Apache-2.0, standard library only; text CC BY 4.0. The numbers
below were worked out by hand. The unit tests check them against the code with stated tolerances.

This note changes nothing in [PLAN-3.md](PLAN-3.md). T1-T4 and their rules stay exactly as preregistered. It
says, before the results exist, how to read them.

## Setup

For one skill, write x = c t (t = practice opportunities so far, c = the skill's rate) and the curve with a ceiling
A (1 - exp(-x)). Phase 3 compares three student random slopes, each fitted next to a student random intercept:

- **ts** (time scale) on the hump h = A x exp(-x),
- **amp** (amplitude) on a = A (1 - exp(-x)),
- **lin** (the iAFM's kind) on t.

A student who learns exp(v) times faster than typical has the exact gap D(v, x) = A (exp(-x) - exp(-x e^v)).
Its Taylor series in v is

    D = h [ v + (1 - x) v^2 / 2 + (1 - 3x + x^2) v^3 / 6 + ... ]

so ts is the first-order model (PLAN-3's stated limit). The tests check all three terms by finite differences.

## A. The three models differ only in how fast the gap between students closes

For small x, h/A = x - x^2 + ..., a/A = x - x^2/2 + ..., and t = x / c. To first order all three random slopes are
the same regressor. They differ only in the **curvature of the gap between students**: relative to its starting
slope, the gap bends down at rate 1 (ts), 1/2 (amp) or 0 (lin). Under ts the gap opens and then closes completely.
Under amp it opens and stays open. Under lin it keeps growing.

So phase 3 can only tell them apart where practice runs far enough along the curve. For one skill with 6
opportunities at equal weight, the correlation between the regressors is:

| design | x reaches | corr(ts, amp) | corr(ts, lin) |
|---|---|---|---|
| c = 0.10, t = 0..5 | 0.5 | 0.996 | 0.986 |
| c = 0.25, t = 0..5 | 1.25 | 0.947 | 0.876 |
| c = 0.50, t = 0..5 | 2.5 | 0.642 | 0.359 |

**Reading rule for RESULTS-3 (no verdict attached):** the ts-versus-amp comparison is informed almost entirely by
skills whose practice reaches about t = 1 / c_k, the peak of the hump. A skill whose best c_k sits at the grid's
lower edge (0.03) with sequences of under 10 opportunities adds almost nothing to it. If T2 comes out with
log-likelihood differences of only a few units, this is the first reason to check. The T1 simulation (c from 0.1
to 0.4, stopping after three in a row) is a fair test of the real data only if the real c_k and sequence lengths
reach a similar x. RESULTS-3 will compare the two.

## B. A head start is exactly intercept plus amplitude

Suppose students differ only in where they start on the same curve, as if one had already had tau more
opportunities. This is "prior knowledge" with a common ceiling. The exact gap is

    A (1 - exp(-c (t + tau))) - A (1 - exp(-c t)) = A q - q a,   q = 1 - exp(-c tau)

That is a raised intercept (A q) plus a *negative* amplitude slope (-q). It holds exactly, for every t (tested).
Three consequences follow.

1. A head start can never be mistaken for a time-scale difference. The hump x exp(-x) is not in the span of
   1 and exp(-x). In T4, `both` beats `amp` only if something other than head starts and ceiling heights differs
   between students.
2. **Reading rule (no verdict attached):** if `amp` wins and its intercept-amplitude correlation is strongly
   negative, the student differences look like head starts on a common curve. If the correlation is near 0,
   they look like different ceilings. In phase 3's amplitude simulation it is near 0 by design. This holds
   exactly per skill. Across skills with different A_k and c_k it holds only approximately, because phase 3's
   student effects are shared across skills.
3. **The iAFM reads a head start as slower learning.** Fit a straight line in t to the gap A q exp(-c t) over
   t = 0..5. The student's slope changes by -0.141 A q per opportunity at c = 0.25 and by -0.079 A q at c = 0.10
   (tested). For a modest head start (A = 2, q = 0.2) at c = 0.25, that is about -0.056 log-odds per
   opportunity. The typical iAFM slope is about 0.1. In a world with a ceiling, the iAFM's student-slope variance
   mixes prior-knowledge convergence (with a negative intercept-slope correlation) with any real speed difference.
   Its smallness therefore says little about speed either way. This adds to THEORY-2b, where a time-scale
   difference itself is attenuated (R_est ~ R_true^e).

## C. The linearized time-scale SD is biased down, so T3 is conservative

Let the student's fitted ts slope be the least-squares projection of the exact gap D(v, .) on (1, h) over the
design. From the series above,

    u(v) = v + alpha v^2 + beta v^3 + O(v^4),   alpha = (1 - M1) / 2,   beta = (1 - 3 M1 + M2) / 6,

with M_j = Cov(h x^j, h) / Var(h) over the design. For v ~ N(0, s^2):

    SD(u) / s = 1 + kappa s^2 + O(s^4),   kappa = alpha^2 + 3 beta.

| design | M1 | M2 | kappa | SD ratio at s = 0.5 (3-point Gauss-Hermite) | fitted speed ratio for a true 1.96 |
|---|---|---|---|---|---|
| c = 0.10, t = 0..5 | 0.484 | 0.223 | -0.048 | 0.977 | 1.93 |
| c = 0.25, t = 0..5 | 1.042 | 1.080 | -0.522 | 0.880 | 1.81 |
| c = 0.50, t = 0..5 | 0.997 | 0.980 | -0.505 | (second order: 0.874) | about 1.80 |

Exact projections at the Gauss-Hermite nodes v = +-0.866 (true speed ratios 2.38 and 0.42) are +0.728 / -0.795
at c = 0.25, and +1.002 / -0.668 at c = 0.10. In short windows the linearized model **over-reads fast students
and under-reads slow ones**, so its per-student estimates are skewed. Their spread is still close to the truth.

What this means for the preregistered tests:
- **T3 is conservative with respect to linearization.** In every design above, the linearized SD is at or below
  the true SD. To show a fitted ratio of 1.5 (SD 0.3006), the true ratio must be about 1.50-1.53. At the T3
  threshold the bias is under 5%.
- The simulation calibration that PLAN-3 reports (median fitted SD over the true 0.5) should land roughly between
  0.87 and 0.98 if this note is right. A value far outside that range means something other than linearization
  dominates, such as mastery stopping, logistic weighting or the grid choice of c_k. RESULTS-3 will say which.

## Limits

- Equal weights per opportunity. The real fit weights each response by p (1 - p) and pools skills with different
  A_k and c_k (`theory3.py` prints a nine-skill design like the simulation's, but its numbers are not part of
  this note).
- Projection on (1, h) per student stands in for the mixed model's estimate. With many responses per student this
  is the usual large-cluster approximation. Shrinkage and the shared intercept across skills are ignored.
- Mastery stopping makes the window student-specific. Fast students stop sooner, so they sit in the short-window
  regime where u over-reads them. That could push the fitted SD up. Phase 3's simulation includes this stopping
  rule, which is why its calibration, not this note, is what RESULTS-3 reports.

## Reproduce

`python -m unittest -v test_theory3` checks the derivatives, the head-start identity and every hand number above
(with stated tolerances). `python theory3.py --out-dir out` writes the full table, adding the 25-opportunity and
nine-skill designs, full-integral SD ratios at s = 0.3 and 0.5, and the iAFM head-start slopes.

## References

- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., et al. (2026). Revisiting the Regularity of Student Learning Rate: Sensitivity to Which Observations Are
  Included. L@S '26, doi:10.1145/3774398.3811619; arXiv:2605.01690. Its future work proposes nonlinear
  (logarithmic, saturating or power-law) opportunity terms. Phases 2 and 3 of this project test those.
- Liu, R., & Koedinger, K. R. (2015). Variations in learning rate: Student classification based on systematic
  residual error patterns across practice opportunities. *EDM 2015*, 420-423. A per-student straight-line rate did
  not improve prediction, but grouping students by the *shape* of their residuals across opportunities did.
- Heathcote, A., Brown, S., & Mewhort, D. J. K. (2000). The power law repealed: The case for an exponential law of
  practice. *Psychonomic Bulletin & Review*, 7(2), 185-207.
- Käser, T., Koedinger, K. R., & Gross, M. (2014). Different parameters - same prediction: An analysis of learning
  curves. *EDM 2014*.
