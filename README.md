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

## Files

- `PLAN-1.md`, `sim.py`, `fit.R`, `test_sim.py`: phase 1 (simulation).
- `PLAN-2.md`, `realdata.py`, `fit2.R`, `test_realdata.py`: phase 2 (real data; aggregate results only).
- `RESULTS-1.md` and `RESULTS-2.md` will hold the results, whatever they show.

Run it yourself: install R and lme4, then `python sim.py --out-dir out` or
`python realdata.py --data-dir data --out-dir out` (the latter downloads the two open zip files).

## Contact

MV2, Munim, Inc., mv2@munimtech.com. Issues and corrections are welcome.
