# PLAN-1: could real differences in learning speed hide behind the "astonishing regularity"?

Preregistered 2026-11-17 by MV2 of Munim, Inc., before any result is computed. Made with AI agents working in
his name. Code Apache-2.0 (`sim.py`, `fit.R`, `test_sim.py`), text CC BY 4.0.

## The question

Koedinger, Carvalho, Liu and McLaughlin (2023, *PNAS* 120(13), e2221311120, "An astonishing regularity in
student learning rate") fit the individualized Additive Factors Model (iAFM) to 27 tutoring datasets and found
that students differ widely in what they know at the start but are "astonishingly similar" in how fast they learn.
The typical gain is about 0.1 log-odds per practice opportunity, and the 25th- and 75th-percentile learners need
about 8 and 7 opportunities to reach 80% correct. If that is true, it matters for every tutor, school and
"fast learner / slow learner" policy: differences in outcomes would come from prior knowledge and the amount of
practice, not from learning ability.

Replications agree (Simpson, Norberg & Fancsali, EDM 2024, on MATHia; arXiv 2604.03246, "Personalized AI
Practice Replicates Learning Rate Regularity at Scale"). But Lee, Lichand, Barnard, Klotz, Thille, Kim and
Domingue (arXiv 2605.01690, L@S 2026) refit the same data keeping only each student's first few attempts per
skill, and the spread of learning rates grew by a median of 118% while the spread of prior knowledge barely moved.
They discuss mastery-based exit, concave learning curves and identifiability as possible reasons without
deciding among them, and leave open how large learning-rate differences really are.

**The point this project builds on:** in a correctly specified model, neither the fixed cut nor mastery
stopping can bias the estimate. Cutting every sequence at a fixed length is a design choice that does not depend on the answers.
Stopping when a student reaches mastery depends only on answers already seen, so it is ignorable for
likelihood-based fitting (Rubin 1976). If the estimate still moves with the window, the model's *shape* must be
wrong. That happens if learning is not a straight line in log-odds per opportunity but saturates, or is
all-or-none as in Bayesian Knowledge Tracing. When the shape is wrong, the iAFM's "learning-rate spread" is not a
property of students. It is a property of the window and of the stopping rule.

So this project asks a quantitative question: **if students truly differ in learning speed by a factor of 1.2,
1.5 or 2, what spread does the iAFM report?** Phase 1 (this plan) answers that in simulation, with known
truth. Phase 2 will use the calibration to say what the published regularity implies about real students, and
fit better-shaped models to open tutoring data.

**What would be remarkable (the "wow"):** the iAFM reports PNAS-level regularity (75th/25th percentile speed
ratio of 1.25 or less) while the simulated students truly differ about two-fold in learning speed. That
would mean the regularity cannot rule out large real differences. The opposite result is also worth knowing: if
two-fold differences always show up clearly, the published regularity bounds how much real students can
differ, and we will say how tightly.

## Simulation (`sim.py`, parameters fixed here)

150 students x 8 knowledge components (KCs) x up to 15 opportunities, 10 replications (seed 20261117 + rep).
Every truth and level uses the same random stream within a replication, so the settings are directly comparable.

- Students: initial knowledge theta ~ N(0, 1); a speed score z ~ N(0, 1), independent of theta.
- KCs: easiness beta ~ N(-0.5, 0.7).
- **T1 (the iAFM is true):** logit P(correct) = theta + beta + (g_k + s z) t, with g_k ~ N(0.10, 0.03) (at least
  0.02). s in {0, 0.02, 0.05}. True speed ratio (g + 0.674 s)/(g - 0.674 s) at g = 0.10: 1, 1.31, 2.02.
- **T2 (saturating learning):** logit P = theta + beta + 2.0 (1 - exp(-c_k r t)), with c_k ~ U(0.05, 0.15) and
  r = exp(s z). s in {0, 0.15, 0.3, 0.5}. True ratio exp(1.349 s): 1, 1.22, 1.50, 1.96.
- **T3 (all-or-none learning, BKT):** P(known at start) = sigmoid(theta + beta); learn probability
  sigmoid(logit(T_k) + s z) with T_k ~ U(0.08, 0.20); guess 0.2, slip 0.1. s in {0, 0.25, 0.5}. True ratio
  of learn probabilities at the median T_k (0.14): 1, 1.37, 1.78.
- **Practice designs**, all cut from the same 15-opportunity sequences: `full15` (fixed length); `first10` and
  `first5` (fixed truncation, as in Lee et al.); `three_in_a_row` (stop after 3 correct in a row, as in
  ASSISTments skill builders); `bkt_mastery` (stop when the tutor's own BKT estimate, with L0 0.4, T 0.15,
  G 0.2, S 0.1, reaches 0.95, as in Cognitive Tutor / MATHia).
- **Model (`fit.R`):** lme4 `glmer(y ~ 0 + kc + kc:t + (1 + t | student), binomial)`, Laplace, bobyqa: KC
  intercepts and slopes fixed, student intercept and slope random and correlated (the iAFM). A second fit with
  log(1 + t) instead of t is reported without verdicts.
- **What the iAFM reports:** R_est = (mean KC slope + 0.674 SD(delta)) / (mean KC slope - 0.674 SD(delta)),
  the speed of a 75th-percentile learner over a 25th-percentile learner. The PNAS paper's 8 vs 7
  opportunities is roughly 1.14. Per cell we report the mean SD(delta) over replications and the median R_est.

## Predictions (linear iAFM; a cell needs at least 5 usable fits, else "inconclusive")

| id | prediction | rule |
|---|---|---|
| P0 | Recovery: under T1 with s = 0.05, `full15` recovers the student slope SD | mean SD(delta) in [0.040, 0.0625] |
| P1 | Ignorability: under T1 (s = 0.05), mastery stopping does not bias the spread | mean SD(delta) of `three_in_a_row` and of `bkt_mastery` each 0.8-1.25x `full15` |
| P1b | Under T1 (s = 0.05), fixed truncation to 10 does not bias it either | `first10` 0.8-1.25x `full15` |
| P2 | Under saturating learning (T2, s = 0.5), the spread depends on the window, reproducing Lee et al.'s pattern | mean SD(delta) of `first5` at least 1.5x `full15` |
| P3 | Under T2 and T3 with s = 0.5, the iAFM reports well under half of the true speed difference | (R_est - 1)/(true - 1) at most 0.6 in `full15`, for both |
| P4 | Saturation alone does not invent speed differences (T2, s = 0) | median R_est at most 1.15 (`full15`) |
| P5 | All-or-none learning does invent them from prior-knowledge differences (T3, s = 0) | median R_est above 1.15 (`full15`) |
| P6 | Two-fold real differences never look PNAS-regular: no T2/T3 cell with true ratio at least 1.75 in `full15`, `three_in_a_row` or `bkt_mastery` has median R_est at most 1.25 | held if none does |

Why these: P0-P1b check that the machinery works and test the ignorability argument above. P2 tests whether a
wrong curve shape alone reproduces the published sensitivity. P3-P6 measure what the iAFM can and cannot see. P6
is the "wow" test: if it does not hold, a two-fold real difference can look like the published regularity.

## Reported without verdicts

Every cell (10 settings x 5 designs x 2 models): usable and singular fits, mean SD(delta) with its standard
error, mean SD(theta), mean KC slope, median R_est and the true ratio. All fits are kept in `fits.csv`.

## Limits, stated in advance

- A simulation shows what the method *can* hide, not what real students are like. Phase 2 must bring real data.
- One set of population values (fixed above, chosen to give the PNAS paper's average slope of about 0.1 under
  T1 and plausible accuracies). Other values could change the size of the effects.
- 10 replications per cell. Differences smaller than a few standard errors are not interpreted.
- Speed and prior knowledge are independent here. In real data they may be correlated.
- The speed ratio is defined per truth (slopes in T1, rate constants in T2, learn probabilities in T3), so
  "two-fold" means something slightly different in each. The ratios are reported, not hidden.

## Procedure

One sealed CPU job (`lrr-phase1-sim-v1`) waits until this file, `sim.py`, `fit.R` and `test_sim.py` are on this
repository's main branch and runs exactly those public copies (its setup log prints their SHA-256). It installs R and lme4 from
Debian packages, runs the unit tests, then `python sim.py`. Results go in RESULTS-1.md whatever they show.

## References

- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., Lichand, G., Barnard, C., Klotz, L., Thille, C., Kim, Y., & Domingue, B. W. (2026). Revisiting the
  Regularity of Student Learning Rate: Sensitivity to Which Observations Are Included. arXiv:2605.01690 (L@S '26).
- Simpson, Norberg & Fancsali (2024). Replicating an "Astonishing Regularity in Student Learning Rate". EDM 2024
  short papers.
- Personalized AI Practice Replicates Learning Rate Regularity at Scale. arXiv:2604.03246 (2026).
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
