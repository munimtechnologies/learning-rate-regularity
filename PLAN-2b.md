# PLAN-2b: is the window dependence just a change of units? The rescaling test

Preregistered 2026-11-17 by MV2 of Munim, Inc., before any phase 1 or phase 2 fit has finished. Made with AI agents
working in his name. Code Apache-2.0 (`analyze2b.py`, `test_analyze2b.py`), text CC BY 4.0.

This is a secondary analysis of the fits that phase 1 ([PLAN-1.md](PLAN-1.md)) and phase 2 ([PLAN-2.md](PLAN-2.md))
already write (`fits.csv`, `fits2.csv`). It needs no new data and no new model fits. Its code and thresholds are
fixed here, while both jobs are still waiting in the queue.

## Why: a units argument, and what Lee et al.'s own numbers say about it

Suppose each student's log-odds of a correct answer follows a concave curve, θ_i + β_k + b_i·g(t), with
g(t) = log(1 + t). Suppose also that every student is seen on the same opportunities. Fit a straight line in t
over the first k opportunities, as the iAFM does. Each student's fitted slope is then b_i·c_k, where c_k is the
least-squares slope of g on t = 0..k-1. For log(1 + t):

| window | c_k |
|---|---|
| first 5 | 0.391 |
| first 10 | 0.230 |
| first 25 | 0.105 |

Every student's slope is multiplied by the same c_k. So the spread of slopes (SD) and the mean slope change by the
same factor. Their ratio, the coefficient of variation CV = SD / mean, stays the same in every window. The
*PNAS*-style speed ratio, R_est = (mean + 0.674 SD) / (mean - 0.674 SD), depends only on CV, so it does not move
either. Under this kind of curve-shape error, the absolute spread can "inflate" by a factor of 3.7 from first 25
to first 5 (0.391 / 0.105). Yet the claim about how much faster one student learns than another would not change
at all.

**The diagnostic.** Split phase 2's window-dependence index into two parts:

- WDI = SD(slope, first5) / SD(slope, first25): the inflation Lee et al. report.
- MR = mean slope (first5) / mean slope (first25): how much the curve bends, i.e. pure rescaling.
- **EWD = WDI / MR = CV(first5) / CV(first25)**: the *excess* window dependence. Pure rescaling gives EWD = 1.

**Lee et al.'s own numbers already point past pure rescaling.** In their truncation analysis (arXiv:2605.01690,
20 datasets, each student-skill sequence capped at 10 attempts), the median change is +118% in the learning-rate SD
and +18% in the fixed-effect slope. As a ratio of medians, that is 2.18 / 1.18 ≈ 1.85. This is not the median of
per-dataset ratios, which the paper does not give, and their windows (≤ 10 against all practice) differ from ours.
Taken at face value, though, it says students' *relative* differences in learning rate look about 1.8 times larger
when less of their practice is used. A shape error that only rescales slopes cannot do that.

**Two ways it can happen.** Speed can differ in two ways, and the diagnostic tells them apart:

- **Amplitude differences** (b_i·g(t): faster students gain more per step on the same curve). A concave curve
  rescales these, so EWD = 1.
- **Time-scale differences** (g(r_i·t): faster students move along the same curve sooner, towards the same
  ceiling). Early on, slopes are proportional to r_i. Later, fast students have reached the ceiling and slow
  students catch up, so the slopes converge. EWD > 1 follows from shape alone, with no selection needed. Phase 1's
  saturating truth T2 is exactly this case, so its fits calibrate the effect (C1 below).

If real data show EWD > 1 even under the log curve, students differ on a time scale, not just in amplitude. The
long-window regularity would then partly reflect a shared ceiling that everyone reaches eventually. Mastery
stopping and attrition can also push EWD away from 1. The simulation includes both, so C1 and the descriptive
calibration table show how large those effects are with known truth.

## Predictions (fixed now)

From phase 2's `fits2.csv` (two datasets, kdd_alg0506 and assist09, first5 against first25, full sample plus the
20 student bootstraps). "Usable" means both fits succeeded and every quantity needed is defined: mean slopes > 0
and the first25 SD > 0.

| id | prediction | rule |
|---|---|---|
| S1 | (manipulation check) real learning curves bend: the linear iAFM's mean slope is steeper in the first 5 opportunities | linear MR > 1 on the full sample, and MR > 1 in ≥ 90% of usable bootstraps (≥ 15 needed), in both datasets |
| S2 | the window dependence is more than rescaling: relative speed differences look larger in short windows | linear EWD ≥ 1.2 on the full sample, and EWD > 1 in ≥ 90% of usable bootstraps (≥ 15 needed), in both datasets |
| S3 | the concave (log) curve does not remove that excess: the signature of time-scale differences | log-model EWD ≥ 1.2 on the full sample, and EWD > 1 in ≥ 90% of usable bootstraps (≥ 15 needed), in both datasets |

From phase 1's `fits.csv` (truth known), a check on the theory above:

| id | prediction | rule |
|---|---|---|
| C1 | a saturating curve with time-scale speed differences produces excess window dependence by itself | for truth T2 at level 0.5, linear model, EWD(first5 against full15) > 1.1, using means over replications of SD(slope) and mean slope |

Verdicts use the same rules as phase 2. A failed fit or an undefined ratio makes a prediction "inconclusive". A
clear failure in either dataset makes it "did not hold". Reported without verdicts: WDI, MR and EWD for every
dataset and model, 90% bootstrap intervals for EWD, R_est in each window, and the same three numbers for every
phase 1 truth, level, model and longer design, each against first5.

## What each outcome would mean

| S2 | S3 | reading |
|---|---|---|
| did not hold (EWD ≈ 1) | - | In these data, the inflation Lee et al. found is a change of units. The relative regularity Koedinger et al. reported (the speed ratio) does not depend on the window, so their claim survives. |
| held | did not hold | The log curve absorbs the excess. Fitting the right curve shape is enough, and the log-model R_est is the better estimate. |
| held | held | Relative differences in learning speed shrink as more practice is included, even under a concave curve. That is what students who differ in time scale and approach a shared ceiling would produce. C1 shows whether pure shape can produce an excess this large. |

**The "wow" if it works:** S2 and S3 holding, with C1 showing that a shared ceiling alone produces an excess of
that size, would mean the "astonishing regularity" is at least partly a ceiling effect of long practice
sequences. Students would differ in how *soon* they learn rather than in a per-step rate. That reframes one of
learning science's most-cited recent results. S2 failing would instead clear the regularity of Lee et al.'s
critique in these data. Either answer settles a live published disagreement.

## Limits, stated in advance

- Two datasets, 12 skills each, at most 800 students, so this tests a mechanism; it does not re-analyse all 27
  PNAS datasets.
- The rescaling argument is exact only when every student is seen on the same opportunities. Mastery stopping
  breaks that, in either direction. The phase 1 table shows by how much under known truths.
- The iAFM's student slope is a random effect pooled over skills with skill-specific mean slopes, not a
  per-student least-squares line. The argument is a first-order guide, not an identity for glmer.
- EWD uses the mean of the skill-specific fixed slopes as "mean slope", as phase 2 already reports it.
- The Lee et al. figure above (≈ 1.85) is a ratio of their reported medians, used only as motivation. It is not a
  result of this project.

## Procedure

The code is `analyze2b.py` (standard library only), with unit tests in `test_analyze2b.py`. Both are committed with
this file. Once both jobs finish, their aggregate fit tables (`fits.csv`, `fits2.csv`: one row per model fit, with
no student-level data) are added to this repository under `results/`. Then
`python analyze2b.py --fits2 results/phase2/fits2.csv --fits1 results/phase1/fits.csv --out-dir out` writes
`results2b.md` and `results2b.json`. RESULTS-2b.md will report them, whatever they show.

## References

- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., Lichand, G., Barnard, C., Klotz, L., Thille, C., Kim, Y., & Domingue, B. W. (2026). Revisiting the
  Regularity of Student Learning Rate: Sensitivity to Which Observations Are Included. arXiv:2605.01690 (L@S '26).
