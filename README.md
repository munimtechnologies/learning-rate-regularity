# learning-rate-regularity

**Do students really learn at nearly the same rate, or does the way we measure learning rate hide real
differences?** An open, preregistered research project by MV2 of Munim, Inc. The work is made with AI agents
working in his name. Code Apache-2.0; text and results CC BY 4.0.

## Background

In 2023, Koedinger and colleagues reported in *PNAS* an "astonishing regularity" in student learning rate.
Across 27 tutoring datasets, students differed a lot in prior knowledge but very little in how much they gained
from each practice opportunity. A 2026 reanalysis (Lee et al., arXiv:2605.01690) showed that the estimated
spread of learning rates changes a lot depending on how much of each student's practice is included.

This project starts from a simple point. If the model's learning curve had the right shape, neither cutting
practice at a fixed length nor stopping at mastery could bias the estimate. Window dependence therefore suggests
the learning curve has the wrong shape. If so, how large could real differences in learning speed be and still
look "regular"?

## Phases

1. **Phase 1, simulation (preregistered in [PLAN-1.md](PLAN-1.md)).** Students with known learning-speed
   differences (1x to about 2x) and three true shapes of learning: the model's own straight line in log-odds,
   saturating learning, and all-or-none BKT learning. Five practice designs: fixed length, two fixed
   truncations, three-in-a-row mastery and BKT mastery. Each data set is fitted with the same iAFM as in the
   *PNAS* paper (R, lme4), and the reported speed spread is compared with the truth. Eight predictions are fixed
   in advance.
2. **Phase 2, real data (preregistered in [PLAN-2.md](PLAN-2.md)).** Two open tutoring logs with mastery-based
   practice: KDD Cup 2010 Algebra I 2005-2006 (Cognitive Tutor) and ASSISTments 2009-2010 skill builders.
   Each is fitted with the iAFM and with a concave version (log(1 + t)) on the first 5, 10 and 25
   opportunities, with 20 bootstrap resamples of students. The test: does the better-shaped curve shrink the
   window dependence that Lee et al. found? Three predictions are fixed in advance.
3. **Phase 2b, the rescaling test (preregistered in [PLAN-2b.md](PLAN-2b.md)).** If a curve-shape error only
   rescaled every student's slope, the spread and the mean slope would change together and the speed ratio would
   not depend on the window. Phase 2b splits the window dependence into rescaling and an excess (the change in
   SD / mean). It also asks whether students differ in *how soon* they learn on a shared curve with a ceiling,
   rather than in a per-step rate. Four predictions are fixed in advance. It uses only the fits that phases 1
   and 2 already produce.
4. **Theory for phase 2b ([THEORY-2b.md](THEORY-2b.md)).** A worked derivation, done before any result came in.
   When students differ in time scale on a curve with a ceiling, the iAFM's speed ratio behaves like
   R_true^e. The "ceiling exponent" e falls from 1 towards 0 as the practice window covers more of the curve.
   Past a reversal point (about 95% of the way to the ceiling within the window), faster learners get
   *shallower* fitted slopes. A reported ratio of 1.14 is compatible with a twofold true speed difference when e
   is about 0.19. Amplitude differences, by contrast, give a ratio that does not depend on the window. The note
   also gives first-order values that phase 1's simulation should approach.
5. **Phase 3, the direct test (preregistered in [PLAN-3.md](PLAN-3.md)).** Each skill in the phase 2 data gets a
   learning curve with a ceiling, A(1 - exp(-c t)). Three kinds of student difference are then fitted, each with
   the same number of parameters:
   - **time scale:** a random slope on the hump A c t exp(-c t), where faster learners pull ahead and everyone
     meets at the ceiling;
   - **amplitude:** a random slope on A(1 - exp(-c t)), where some curves rise higher;
   - **raw rate:** a random slope on t, the iAFM's kind of difference.

   The time-scale SD gives a 75th/25th-percentile speed ratio directly. A 20-dataset known-truth check must first
   show that the method can tell time scale from amplitude. Four predictions are fixed in advance. The wow would
   be students who look "regular" differing 1.5-fold or more in how soon they reach mastery.

## Files

- `PLAN-1.md`, `sim.py`, `fit.R`, `test_sim.py`: phase 1 (simulation).
- `PLAN-2.md`, `realdata.py`, `fit2.R`, `test_realdata.py`: phase 2 (real data; aggregate results only).
- `PLAN-2b.md`, `analyze2b.py`, `test_analyze2b.py`: phase 2b (secondary analysis of the phase 1 and 2 fits).
- `THEORY-2b.md`, `ceiling.py`, `test_ceiling.py`: the ceiling exponent and the reversal point (standard library
  only; `python ceiling.py --out-dir out` writes every table).
- `PLAN-3.md`, `phase3.py`, `fit3.R`, `test_phase3.py`, `setup3.sh`: phase 3 (time scale vs amplitude; uses
  `realdata.py` for loading, so the students and skills are the same as in phase 2).
- `RESULTS-1.md`, `RESULTS-2.md`, `RESULTS-2b.md` and `RESULTS-3.md` will hold the results, whatever they show.

Run it yourself: install R and lme4, then `python sim.py --out-dir out`,
`python realdata.py --data-dir data --out-dir out` or `python phase3.py --data-dir data --out-dir out` (the
last two download the two open zip files). Phase 2b needs only Python:
`python analyze2b.py --fits2 fits2.csv --fits1 fits.csv --out-dir out`.

## Contact

MV2, Munim, Inc., mv2@munimtech.com. Issues and corrections are welcome.
