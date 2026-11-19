# THEORY-4: mastery stopping meets the ceiling

Written 2026-11-19 by MV2 of Munim, Inc., before any phase 1, 2 or 3 fit has run. Made with AI agents working in
his name. Code (`theory4.py`, `test_theory4.py`) Apache-2.0, standard library only; text CC BY 4.0. The numbers
below were worked out by hand. The unit tests check every one of them against the code with stated tolerances.

This note changes no preregistered prediction in PLAN-1, PLAN-2, PLAN-2b or PLAN-3. It fills the gap that THEORY-2b
and THEORY-3 both left open in their limits: they assume every student is seen for the same window, but both real
datasets in phase 2 stop practice at mastery. ASSISTments skill builders stop after three correct answers in a
row. The Cognitive Tutor stops when its knowledge-tracing estimate passes 0.95.

## The question

THEORY-2b showed that on a learning curve with a ceiling, the iAFM's straight line attenuates true differences in
learning speed. The reported 75th/25th-percentile ratio behaves like R_true^e, and the "ceiling exponent" e falls
from 1 towards 0, and below, as the window covers more of the curve. That is how a "regular" 1.14 could hide a
twofold difference. Lee et al. (2026) name mastery-based exit as one possible reason the iAFM's learning-rate spread
depends on the window, but give no derivation. Goutte, Durand and Léger (2018) and Pelánek's "mastery attrition
bias" concern aggregate AFM learning curves, not the student-level rate spread.

**Does mastery stopping strengthen or undo the ceiling mechanism?** If faster students stop sooner, each student's
window depends on their own speed. Nobody has worked out what that does to the iAFM's learning-rate spread.

## 1. Any stopping rule enters only through survival weights

Let practice stop by any rule that looks only at answers already given: a fixed cap, three in a row, or a BKT
threshold. Suppose that, given the student's true curve p_t, answers are independent. This holds for phase 1's
truths T1 and T2, but **not** for its all-or-none BKT truth T3, where answers share a latent state. Then whether
opportunity t is observed depends on y_0..y_{t-1} only, and y_t is still Bernoulli(p_t). The expected
log-likelihood of *any* fitted model is therefore

    sum_t pi_t [ p_t log q_t + (1 - p_t) log(1 - q_t) ],   pi_t = P(opportunity t is observed).

Two things follow.
- If the fitted model contains the truth, its maximum is at the truth whatever the weights are. This is the
  ignorability that PLAN-1 relies on (Rubin 1976). `test_wlogit_recovers_a_true_logistic_line_under_any_stopping`
  checks it for three rules.
- If the model is wrong, as a straight line on a ceiling curve is, its large-sample estimate is a
  **pi-weighted fit to the true curve**. The stopping rule changes what the straight line estimates only through
  pi_t. For three in a row at a constant p = 0.5, pi_t = 1, 1, 1, 0.875, 0.8125, 0.75, 0.6875, 0.6328 (tested), and
  the expected length is 14.

## 2. A window that scales with speed cancels the ceiling

Use THEORY-2b's first-order setting: the least-squares slope of the log-odds curve A (1 - exp(-c t)) over a
window t in [0, T]. It is exactly A c S(cT), where

    S(X) = 12 I(X) / X^3,   I(X) = X/2 - 1 + exp(-X) (1 + X/2).

Let eta(X) = X S'(X) / S(X), which is negative. THEORY-2b's fixed-window exponent is 1 + eta(cW). Now let the
window depend on the student's speed, T(c) proportional to c^(-lambda). Then, exactly,

    e = d log(slope) / d log(c) = 1 + eta(X) (1 - lambda),   X = c T.

| X = c T | S(X) | eta(X) | e if fixed (lambda = 0) | e if lambda = 0.5 | e if lambda = 1 |
|---|---|---|---|---|---|
| 0.5 | 0.783679 | -0.237522 | 0.762 | 0.881 | 1 |
| 1.25 | 0.556464 | -0.547732 | 0.452 | 0.726 | 1 |
| 3 | 0.277541 | -1.076317 | -0.076 | 0.462 | 1 |

The fixed-window exponent changes sign between X = 2.6 and 2.8 (tested), in agreement with THEORY-2b's reversal
point at 2.69.

**The result.** Suppose practice stops when the student's own curve reaches a fixed fraction of its gain, an
idealised deterministic mastery rule. Then the window is T = x*/c, lambda = 1, and **e = 1 for every curve shape**.
The proof is one line. The student's data are the same function of c t over the same range of c t, so the straight
line's slope is exactly proportional to c. The ceiling attenuation disappears and so does the rank reversal. The
straight line becomes a fair measure of speed, the time-to-mastery reading of Carroll's (1963) model of school
learning. In a discrete check (stopping at 95% of the gain), doubling speed doubles the fitted slope within the test's 3% tolerance
(`test_level_rule_is_self_similar_and_fixed_window_reverses`). In one fixed window of 150 opportunities, the same
doubling gives the faster student the *shallower* slope, with a ratio of about 0.81.

So THEORY-2b's mechanism, a ceiling that makes real speed differences look regular, **needs windows that do not
shrink in proportion to speed**. Real mastery rules lie in between:

- **Three in a row** can never stop before three opportunities. That floor caps how short a fast student's window
  can get. A slow student near p = 0.5 often stops by luck: the expected wait for three in a row at p = 0.5 is 14
  opportunities, wherever on the curve the student is. Both effects push lambda below 1. In a continuous hazard
  approximation (not used for any number here), lambda is about 1 - H(x)/(x h(x)). Here h is the stopping hazard
  at the point where the student stops, and H/x is its average up to that point. lambda is close to 1 only if the
  hazard rises steeply along the curve.
- **Fixed caps on mastery data**, as in Lee et al.'s first-5 and first-10 refits and phase 2's t < 25, cut the
  windows of slow students only. This brings back fixed-window attenuation for them.

`theory4.py` computes lambda and e exactly for three in a row (caps 15, 25 and 200), phase 1's BKT mastery rule
(cap 15), fixed windows and the idealised rule. It does this both for least squares on the log-odds and for the
logistic pseudo-true fit. The job `lrr-theory4-v1` publishes these tables.

## 3. Under mastery stopping, prior knowledge reads as learning rate

With a fixed window, a student who simply starts higher (a raised intercept theta, same gain A) gets exactly the
same least-squares slope: the curve is shifted, not reshaped. With an absolute mastery level, that student reaches
it **earlier on the curve**. Their window covers only the steep part, so the straight line reads them as faster:

    d log(slope) / d theta = -eta(x*) / (x* A exp(-x*)) > 0.

At A = 2 this is +0.7647 per logit if mastery comes at x* = 1.25, and +3.603 per logit if it comes at x* = 3, close
to the ceiling (both tested). These are first-order local values. Near the ceiling the channel is steep because a
small change in theta moves the stopping point a long way. A student whose ceiling lies below the mastery level
never stops at all: they keep practising until the cap, which is the familiar wheel-spinning pattern.

A *head start on a common curve* works the other way. THEORY-3 B showed it reads as slower learning in a fixed
window. Under mastery stopping the window is [x0, x*], and

    d log(slope) / d x0 = -1 - eta(D) / D < 0,   D = x* - x0,

which is -0.5618 at D = 1.25 and -0.6412 at D = 3 (tested). So under mastery stopping, the **sign of the iAFM's
intercept-slope correlation** separates the two kinds of prior knowledge. Raised curves give a positive sign; head
starts on a shared curve give a negative one. If the iAFM were the true model, stopping would be ignorable and
neither effect would exist.

## What this means for reading the results (no verdict changes)

- **Phase 1, truth T2 at level 0** (no true speed differences at all): with a ceiling, `three_in_a_row` and
  `bkt_mastery` should show a student-slope SD above `full15`'s and a **positive** intercept-slope correlation,
  from prior knowledge alone. With a fixed window the least-squares channel is exactly zero. Logistic weighting
  can add an effect even there; section 4 of the job's tables gives its first-order size and sign for every rule. A zero or negative
  correlation in those cells would mean this note's channel is wrong or swamped. Under T1 (the iAFM is true),
  PLAN-1's P1 tests that no such effect appears.
- **Phase 1, truth T2 at level 0.5:** how much closer `three_in_a_row` and `bkt_mastery` bring R_est to the true
  1.96 than `full15` does is a measure of their lambda. RESULTS-1 will set the fitted values next to the job's
  first-order values.
- **Phase 2, real mastery data:** a positive intercept-slope correlation in the linear iAFM is what a ceiling plus
  stopping predicts. A negative one points to head starts on a shared curve (THEORY-3 B). THEORY-2b's table (d),
  which gives the true ratio behind a reported 1.14, must be read with e = 1 + eta (1 - lambda), not with the
  fixed-window e. If real lambda is near 1, a reported 1.14 means roughly 1.14. The "hidden twofold difference"
  reading then fails, and RESULTS-2b will say so.

## Limits

- First order: no binomial noise, no random-effect shrinkage, and one student-skill pair at a time. The iAFM
  shares one student slope across skills, which this note averages. Hand numbers use continuous least squares on
  the log-odds; the job adds the discrete logistic fits.
- Section 1 assumes answers are independent given the curve. It does not cover BKT-like truths with a latent
  state.
- One family of curves, the exponential approach to a ceiling, except the lambda = 1 result, which holds for any
  curve of the form f(c t).
- The idealised mastery rule is deterministic. Real rules are not, and that difference is the whole of section 2's
  "in between".

## Reproduce

`python -m unittest -v test_theory4` checks the formulas against numerical integration and finite differences, the
survival weights against hand values, ignorability for a true straight line, the self-similarity of level stopping
and every hand number above. `python theory4.py --out-dir out` writes out/theory4.md and out/theory4.csv.

## References

- Carroll, J. B. (1963). A model of school learning. *Teachers College Record*, 64, 723-733.
- Goutte, C., Durand, G., & Léger, S. (2018). On the learning curve attrition bias in additive factor modeling.
  *AIED 2018*, LNCS 10948, 109-113. doi:10.1007/978-3-319-93846-2_21.
- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., Lichand, G., Barnard, C., Klotz, L., Thille, C., Kim, Y., & Domingue, B. W. (2026). Revisiting the
  Regularity of Student Learning Rate: Sensitivity to Which Observations Are Included. L@S '26; arXiv:2605.01690.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
