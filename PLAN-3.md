# PLAN-3: do students differ in how SOON they learn, rather than in how much each step teaches them?

Preregistered 2026-11-17 by MV2 of Munim, Inc., before any phase 1, 2 or 3 fit has run and before the data below
have been opened. Made with AI agents working in his name. Code Apache-2.0 (`phase3.py`, `fit3.R`,
`test_phase3.py`; data loading reused from `realdata.py`), text CC BY 4.0.

## The question

Koedinger et al. (2023, *PNAS*) found that students differ a lot in prior knowledge but very little in learning
rate: in the individualized Additive Factors Model (iAFM), the 75th-percentile student gains only about 1.14 times
as much per practice opportunity as the 25th. Lee et al. (arXiv:2605.01690, L@S 2026) showed that this spread
changes sharply with how much of each student's practice is used. [THEORY-2b.md](THEORY-2b.md) shows why a
learning ceiling can do that. If students differ in **time scale**, the faster of two students reaches the same
ceiling sooner. The iAFM's straight line then reports a speed ratio of about R_true^e, and the "ceiling exponent"
e can be far below 1. On the same data, a twofold real difference can look like 1.14.

THEORY-2b is indirect. It predicts how the iAFM *misbehaves*. Phase 3 asks the question directly. Fit a learning
curve with a ceiling to real tutoring logs. Then ask which kind of student difference the data prefer:
- a different **time scale** (fast learners get to the ceiling sooner),
- a different **amplitude** (some students' curves rise higher), or
- a different **per-opportunity rate** on raw practice count (the iAFM's kind of difference).

The same research group once found the opposite of "regularity" in a different model family. Yudelson,
Koedinger and Gordon (AIED 2013) individualized Bayesian Knowledge Tracing and found that giving each student their
own speed of learning helped prediction more than giving each their own prior knowledge. BKT's learning curve has
a ceiling. Phase 3 asks whether that result and the PNAS regularity are two views of one fact: students differ in
time scale, and a model without a ceiling cannot see it.

## Method (fixed now)

**Data and selection:** exactly phase 2's ([PLAN-2.md](PLAN-2.md)). There are two open logs with mastery-based
practice: KDD Cup 2010 Algebra I 2005-2006 and ASSISTments 2009-2010 skill builders. For each: t < 25, the 12 KCs
practised by the most students, and at most 800 students (seed 20261118). Same parsing, same students, same KCs.
Only aggregate results are published.

**Step 1, one curve per KC.** logit P(correct) = theta_i + beta_k + A_k (1 - exp(-c_k t)), with a random intercept
per student. c_k is chosen from the grid 0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9 by maximum likelihood
(lme4 `glmer`, one KC at a time). A best value at either end of the grid is flagged.

**Step 2, three kinds of student difference, all fitted on all 12 KCs at once.** Fixed part:
`y ~ 0 + kc + kc:g`, with g = 1 - exp(-c_k t) and c_k from step 1. Student part:
- **ts** (time scale): `(1 + h | student)`, with h = A_k c_k t exp(-c_k t). A student r = exp(v) times faster has
  the curve A_k (1 - exp(-c_k r t)). Its first-order change in v is v * h. h is a hump that peaks at t = 1 / c_k:
  the gap between fast and slow learners opens, then closes at the ceiling.
- **amp** (amplitude): `(1 + a | student)`, with a = A_k g. A curve exp(w) times higher adds w * a, a gap that
  opens and stays.
- **lin** (iAFM-style rate): `(1 + t | student)`.
- **both**: `(1 + a + h | student)`.

ts, amp and lin have the same number of parameters (24 fixed effects and 3 for the student part), so their
log-likelihoods compare directly. In ts, SD(v) is the SD of log learning speed. The implied 75th/25th-percentile
**speed ratio** is exp(1.349 SD(v)), the quantity that PNAS reports as about 1.14.

**Step 3, identification check with known truth (simulation).** There are 10 data sets in which students truly
differ in time scale (SD of log speed 0.5, a true speed ratio of 1.96) and 10 in which they truly differ in
amplitude (SD of log height 0.5). Each has 400 students and 8 KCs, with beta_k ~ N(-0.3, 0.5), A_k ~ U(1.5, 3.0),
c_k ~ U(0.1, 0.4) and theta ~ N(0, 1). Practice stops after three correct in a row or at t = 25. Each is put
through exactly steps 1 and 2 (ts and amp).

**Step 4, bootstrap.** There are 10 resamples of students with phase 2's seeds. The step-1 curves are held fixed.
ts and amp are refitted on each.

## Predictions (fixed now)

| id | prediction | rule |
|---|---|---|
| T1 | **Gate:** the pipeline can tell the two kinds of difference apart | at least 8 of 10 time-scale simulations have loglik(ts) > loglik(amp), AND at least 8 of 10 amplitude simulations have loglik(amp) > loglik(ts) |
| T2 | Real students differ in time scale more than in amplitude or raw rate | in both datasets, full sample: loglik(ts) > loglik(amp) and loglik(ts) > loglik(lin); and ts beats amp in at least 90% of usable bootstrap resamples (at least 8 usable) |
| T3 | The time-scale difference is large | in both datasets, the speed ratio from ts is at least 1.5 on the full sample, and at least 1.2 in at least 90% of usable bootstrap fits (at least 8 usable) |
| T4 | Time scale adds to amplitude rather than standing in for it | AIC(both) < AIC(amp) in both datasets, full sample |

If T1 does not hold, T2-T4 are reported with their numbers but carry **no verdict**. The method could not tell
the difference in data built to have it. A failed fit or a missing dataset makes a prediction inconclusive, never
"held". A clear failure in either dataset makes it "did not hold".

**Reported without verdicts:** each KC's c and A and the grid-edge flags; every model's SDs, correlation,
log-likelihood, AIC and singular flag; and the simulation calibration (the median fitted SD over the true 0.5).
The phase 2 iAFM speed ratio on the same students (results2.md, first25, linear) is reported next to the ts ratio
once both exist.

**The wow, if it works:** T1 + T2 + T3. In two open tutoring datasets, students who look "astonishingly regular"
in learning rate would be shown to differ by 1.5 times or more in how fast they reach the same mastery. The
difference is invisible to a straight-line model because the faster learners hit the ceiling first. That would
reframe one of learning science's most-cited recent claims, with a concrete fix for how learning rate should be
measured.

**An honest negative is also informative:** if T1 holds and T2 fails, a ceiling does not explain the window
dependence, and the iAFM's regularity survives a direct test of the most natural alternative.

## Limits, stated in advance

- The time-scale and amplitude models are first-order (linearized) versions of the nonlinear ones. They are
  exact as SD(v) goes to 0. For a speed ratio near 2, the fitted SD may be biased, and the simulation calibration
  reports by how much. T3 uses the uncalibrated value.
- c_k is a grid value chosen with a random intercept only. It is held fixed in step 2 and in the bootstrap, so the
  intervals understate the uncertainty in the curves.
- The curve shape 1 - exp(-c t) is one shape with a ceiling. Others (such as a logistic curve or BKT's all-or-none
  learning) are not tested.
- Mastery stopping means that fast students leave early. Under a correct model, the likelihood still ignores this
  (Rubin 1976). Under a wrong model it may not. The simulation uses the same kind of stopping so that this is part
  of the check.
- This is a test on two datasets, not the 27 used by PNAS (most of those need a DataShop login).

## Procedure

One sealed CPU job (`lrr-phase3-ts-v1`) waits until this file, `phase3.py`, `fit3.R`, `test_phase3.py` and
`realdata.py` are on this repository's main branch and runs exactly those public copies. Its setup log prints
their SHA-256. It installs R and lme4, runs the unit tests, then runs `python phase3.py`. It saves after every
stage, so a time-out still leaves the finished stages. Results go in RESULTS-3.md, whatever they show.

## References

- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., Lichand, G., Barnard, C., Klotz, L., Thille, C., Kim, Y., & Domingue, B. W. (2026). Revisiting the
  Regularity of Student Learning Rate: Sensitivity to Which Observations Are Included. arXiv:2605.01690 (L@S '26).
- Yudelson, M. V., Koedinger, K. R., & Gordon, G. J. (2013). Individualized Bayesian Knowledge Tracing Models. In
  *Artificial Intelligence in Education (AIED 2013)*, LNCS 7926, 171-180. doi:10.1007/978-3-642-39112-5_18.
- Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4.
  *Journal of Statistical Software*, 67(1), 1-48.
- Stamper, J., Niculescu-Mizil, A., Ritter, S., Gordon, G. J., & Koedinger, K. R. (2010). Algebra I 2005-2006.
  Development data set from KDD Cup 2010 Educational Data Mining Challenge.
- Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). Addressing the assessment challenge with an online
  system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, 19(3), 243-266.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
