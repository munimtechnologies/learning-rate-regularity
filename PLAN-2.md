# PLAN-2: in real tutoring logs, does a better-shaped learning curve remove the window dependence?

Preregistered 2026-11-17 by MV2 of Munim, Inc., before any of the data below has been opened or fitted. Made
with AI agents working in his name. Code Apache-2.0 (`realdata.py`, `fit2.R`, `test_realdata.py`), text CC BY 4.0.
Phase 1 ([PLAN-1.md](PLAN-1.md)) is a simulation with known truth. This phase uses real students. It does not
depend on phase 1's results.

## The question

Koedinger et al. (2023, *PNAS*) report that students differ little in learning rate under the individualized
Additive Factors Model (iAFM). Lee et al. (arXiv:2605.01690, L@S 2026) refit the same 27 datasets with less of
each student's practice per skill. The learning-rate spread then "inflates by a median of 118%", while the
spread of prior knowledge stays about the same. They leave the reason open. Their future-work list includes
nonlinear opportunity terms.

PLAN-1 explains the logic. Under a correctly shaped model, neither a fixed cut nor mastery stopping can bias
the estimate, because stopping depends only on answers already seen (Rubin 1976). So the window dependence
points to a curve with the wrong shape. **The direct test on real data:** fit the same iAFM with a concave curve
(log(1 + t) in place of t) and see whether the window dependence shrinks.

- If it shrinks clearly, the regularity debate is partly about curve shape. That leads to a concrete fix for
  how learning rates should be estimated and reported.
- If it does not shrink, shape is not the explanation. That leaves heterogeneity the iAFM cannot represent,
  such as all-or-none learning (PLAN-1's T3), or selection the model does not capture. It is also worth knowing.

## Data (open, downloaded by the job; only aggregate results are published)

- **kdd_alg0506**: KDD Cup 2010 development data, Algebra I 2005-2006, Carnegie Learning Cognitive Tutor
  (practice ends at BKT mastery). This is the training file from the copy at `base.ustc.edu.cn/data/KDD_Cup_2010/`.
  It keeps steps with exactly one KC in the first KC model column, uses t = that KC's opportunity count - 1,
  and takes y = correct on first attempt.
- **assist09**: ASSISTments 2009-2010 skill builder data, the "corrected" release (practice ends at three correct
  in a row). This is the copy at `base.ustc.edu.cn/data/ASSISTment/`. It keeps main problems only
  (original = 1) and single-skill problems only. A problem whose order_id appears more than once is dropped.
  Rows are ordered by order_id, and t counts each student's earlier such problems on the same skill.
- **Selection, both datasets:** keep t < 25, keep the 12 KCs practised by the most students, and take at most
  800 students (a seeded random sample, seed 20261118). The same students and KCs are used in every window.

## Fits

For each dataset, there are three windows: **first5** (t < 5), **first10** (t < 10) and **first25** (t < 25,
the "long" window). There are two models (`fit2.R`, lme4 `glmer`, binomial, Laplace, bobyqa):
- **linear**: `y ~ 0 + kc + kc:t + (1 + t | student)`, the iAFM.
- **log**: the same model with log(1 + t) in place of t.

There are 20 bootstrap resamples of students (seeded), plus the full sample. Each resample has 12 fits, for
252 fits in all.

- **Window-dependence index:** WDI = SD(student slope) on first5 / SD(student slope) on first25. It is a ratio
  within one model, so the unit of the slope cancels.
- **Reported speed ratio:** R_est = (mean KC slope + 0.674 SD) / (mean KC slope - 0.674 SD). PNAS-level
  regularity is about 1.14.

## Predictions (fixed now)

| id | prediction | rule |
|---|---|---|
| Q1 | Lee et al.'s window dependence appears in these two open datasets | linear-iAFM WDI at least 1.3 in both datasets (full sample) |
| Q2 | A concave curve removes part of it | in both datasets, WDI(log) < WDI(linear) on the full sample, and in at least 90% of the bootstrap resamples where both are estimable (at least 15 such resamples needed) |
| Q3 | The concave curve fits better | AIC(log) < AIC(linear) on first25 in both datasets (full sample) |

A dataset whose fits fail makes a prediction "inconclusive". A clear failure in either dataset makes it
"did not hold".

**Reported without verdicts:** for every dataset, window and model, the rows, SD(prior knowledge), SD(student
slope) with its 90% bootstrap interval, the slope-intercept correlation, the mean KC slope, R_est with its 90%
interval, AIC, the singular-fit flag, and dataset summaries (rows, students, sequences, accuracy, and how many
sequences reach t = 5 and t = 10). All fits are in `fits2.csv`.

**What would be remarkable:** if Q2 holds strongly (WDI near 1 under the log curve), a simple change of curve
shape would resolve a published disagreement about one of learning science's most-cited regularities. The
log-curve R_est would then be the better estimate of how much students really differ in learning speed. If Q2
fails while Q3 holds, the better-fitting curve still does not stabilise the estimate. That would be evidence
that student differences are not differences in a smooth rate at all.

## Limits, stated in advance

- Two datasets, 12 KCs each, at most 800 students: a test of the mechanism, not a re-analysis of all 27 PNAS
  datasets (most of those need a DataShop login).
- log(1 + t) is one concave shape among many. A null result for it does not rule out every shape.
- KDD opportunities count multi-KC steps that are dropped here, so t can skip values. ASSISTments t counts
  only the problems that are kept. The two datasets define t slightly differently.
- Bootstrap intervals resample students only, with KCs fixed.
- Mastery stopping means few sequences reach t = 25, so first25 is "all practice up to 25", not 25 observations.

## Procedure

One sealed CPU job (`lrr-phase2-real-v1`) waits until this file, `realdata.py`, `fit2.R` and `test_realdata.py`
are on this repository's main branch and runs exactly those public copies. Its setup log prints their SHA-256.
It installs R and lme4, runs the unit tests, downloads the two zip files, then runs `python realdata.py`.
Results go in RESULTS-2.md, whatever they show.

## References

- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., Lichand, G., Barnard, C., Klotz, L., Thille, C., Kim, Y., & Domingue, B. W. (2026). Revisiting the
  Regularity of Student Learning Rate: Sensitivity to Which Observations Are Included. arXiv:2605.01690 (L@S '26).
- Stamper, J., Niculescu-Mizil, A., Ritter, S., Gordon, G. J., & Koedinger, K. R. (2010). Algebra I 2005-2006.
  Development data set from KDD Cup 2010 Educational Data Mining Challenge.
- Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). Addressing the assessment challenge with an online
  system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, 19(3), 243-266. (ASSISTments.)
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
