# THEORY-2b: how a learning ceiling can manufacture a "regular" learning rate

Written 2026-11-17 by MV2 of Munim, Inc., while phases 1 and 2 were still waiting in the compute queue and before
any fit had finished. Made with AI agents working in his name. Text CC BY 4.0; code (`ceiling.py`,
`test_ceiling.py`) Apache-2.0. The numbers below were worked out by hand and are checked by `test_ceiling.py`.
`python ceiling.py --out-dir out` regenerates every table.

## 1. The question

Koedinger et al. (2023, *PNAS*) fitted the individualized Additive Factors Model (iAFM) to 27 tutoring datasets.
Students differed a lot in prior knowledge but very little in learning rate: a 75th-percentile student learned
only about 1.14 times as fast per opportunity as a 25th-percentile one. Lee et al. (arXiv:2605.01690, L@S '26)
showed that the spread of learning rates grows a lot (median +118%) when sequences are cut at 10 attempts. They
named concave curves and mastery stopping as possible causes, "without adjudicating", and gave the argument in
words only. PLAN-2b (already public) asks whether that growth is only a change of units.

This note works out on paper what a learning ceiling does to the iAFM's learning-rate spread. Students who differ
in *how soon* they learn (time scale) and students who differ in *how much* they gain per step (amplitude) behave
very differently, and the iAFM cannot tell them apart.

## 2. Set-up

On a skill with rate constant k, student i's log-odds of a correct answer at opportunity t = 0, 1, ..., W-1 is

  θ_i + β_k + A·(1 − exp(−k·r_i·t)),  with speed r_i = exp(s·z_i), z_i ~ N(0, 1).

This is phase 1's truth T2 (A = 2, k uniform on 0.05..0.15, s = "level"). The iAFM fits a straight line in t. To
first order (ignoring binomial noise, logistic weighting and random-effect shrinkage), student i's fitted slope on
the skill is A·b(k·r_i, W), where b(k, W) is the least-squares slope of 1 − exp(−k·t) over t = 0..W-1:

  b(k, W) = −Σ (t − t̄)·exp(−k·t) / S,  S = Σ (t − t̄)²,  t̄ = (W − 1)/2.

**The ceiling exponent.** Let e(k, W) = d log b / d log k = k·Σ (t − t̄)·t·exp(−k·t) / (−Σ (t − t̄)·exp(−k·t)).
For small s, the student SD of slopes is A·s·k·b′(k) and the mean is A·b(k), so

  CV = SD / mean ≈ s · e(k, W).

When there are several skills, the iAFM's student slope is shared across them, so e becomes
mean(k·b′) / mean(b) over the skills.

## 3. Results

**(a) Amplitude differences are pure rescaling.** If students differ in A (or in any factor multiplying a fixed
curve g(t)), every student's slope is multiplied by the same window constant. The CV is the same in every window
(EWD = 1), whatever the curve shape. This is PLAN-2b's null hypothesis, now stated exactly (tested in
`test_amplitude_differences_are_pure_rescaling`).

**(b) Time-scale differences are attenuated by a ceiling exponent that falls as the window grows.** e = 1 for a
straight line (k → 0) and decreases as the window covers more of the curve. Values computed by hand:

| W (opportunities) | k = 0.05 | k = 0.10 | k = 0.15 | phase 1 skills pooled |
|---|---|---|---|---|
| 5 | 0.903 | 0.811 | 0.725 | 0.788 |
| 15 | 0.678 | 0.410 | 0.192 | 0.387 |

The 75th/25th-percentile ratio the iAFM reports therefore behaves like R_est ≈ R_true^e. Short windows show
most of the true difference. Long windows, in which fast students reach the ceiling and slow students catch up,
show much less.

**(c) Past a point, the ranking reverses.** b(k, W) is not monotone in k. Once a skill is learned fast enough
compared with the window, a faster student's straight-line slope gets *shallower*: the student's curve rises in
the first step or two and is flat for the rest of the window, while a slower student's rise is spread across it.
e = 0 at the peak and negative beyond it:

- W = 15: e(0.20) > 0 and e(0.25) < 0. The reversal is at k ≈ 0.21, where the curve has risen about 95% of the
  way to its ceiling by the last opportunity.
- In continuous time, e = 0 where 4 − u = exp(−u)(u² + 3u + 4), with u = k × (window span). That gives u ≈ 2.69:
  the curve has gone 93% of the way to its ceiling across the window.

Near the reversal point, the iAFM's learning-rate variance goes to zero *whatever the true speed differences are*.
In phase 1's T2 at level 0.5, roughly 9% of student-skill pairs (k·r > 0.21) lie past the reversal point in the
full 15-opportunity design. This back-of-envelope figure (normal tail areas at k = 0.05, 0.10, 0.15, Simpson's
rule) is not computed by the code.

**(d) What a reported 1.14 could hide.** If real learning works this way, the true speed ratio that would show up
as the PNAS figure of 1.14 is:

| ceiling exponent e | true 75th/25th speed ratio |
|---|---|
| 1 (no ceiling in the window) | 1.14 |
| 0.75 | 1.19 |
| 0.5 | 1.30 |
| 0.387 (phase 1 skills, 15 opportunities) | 1.40 |
| 0.25 | 1.69 |
| 0.19 (k = 0.15, 15 opportunities) | 1.99 |
| 0.1 | 3.70 |

So a "regular" 1.14 is compatible with a twofold difference in learning speed, provided the iAFM's windows reach
well into the ceiling. Whether real tutoring data look like this is exactly what PLAN-2b's S2 and S3 test.

## 4. First-order expectations for phase 1 (written before any phase 1 result)

These are not new verdict rules. PLAN-1's and PLAN-2b's rules stand unchanged. They are what the theory predicts
for truth T2 at level 0.5, linear iAFM, pooled over k = 0.05, 0.10, 0.15, ignoring noise, logistic weighting and
shrinkage:

| quantity | first-order value | preregistered rule |
|---|---|---|
| EWD = CV(first5) / CV(full15) (C1) | 0.788 / 0.387 ≈ 2.04 | C1: > 1.1 |
| MR = mean slope first5 / full15 | 0.0801 / 0.0492 ≈ 1.63 | (reported) |
| WDI = SD first5 / SD full15 (P2) | 2.04 × 1.63 ≈ 3.3 | P2: ≥ 1.5 |
| R_est at full15, against true 1.96 | ≈ 1.30 | P6 flags R_est ≤ 1.25 |
| P3 attenuation (R_est − 1) / (R_true − 1) | ≈ 0.31 | P3: ≤ 0.6 |

Level 0.5 is not small, so the code also gives the exact deterministic-curve moments over students
(`student_moments`; the "exact over students" table). Shrinkage in glmer pulls short-window SDs down more than
long-window ones, so the simulated WDI and EWD should come out *below* these values. A result far above them would
point to an error in the theory or the simulation. The R_est of 1.30 is close to P6's 1.25 line, so P6 may go
either way for T2.

## 5. Why this is new

Lee et al. (arXiv:2605.01690, v2, read 2026-11-17) explain the truncation effect in words: "a linear
approximation of a concave curve produces a steeper slope when fit through early opportunities". They do not
derive it, do not separate time scale from amplitude, never mention a ceiling, and do not discuss rank reversal.
Koedinger et al. (2023) and the EDM 2024 MATHia replication take the iAFM's linear form as given. A web search on
2026-11-17 found no derivation of how a ceiling affects iAFM learning-rate estimates. This is a small analytic
result, but it turns a qualitative worry into numbers that can be tested: the ceiling exponent, the reversal
point and the implied true ratio.

## 6. Limits

- First order: no binomial noise, logistic weighting or random-effect shrinkage. The phase 1 fits (glmer on
  simulated binary data) are the real check.
- One family of curves (exponential approach to a ceiling, in log-odds). Other saturating shapes give other
  exponents but the same qualitative picture.
- It assumes every student is seen on the same opportunities. Mastery stopping changes the windows, student by
  student.
- It shows what *could* produce the regularity. It does not show that real data do. That is PLAN-2b's job, and
  phase 3's direct time-scale fit.

## 7. Next

Once the phase 1 and 2 fit tables are public, PLAN-2b's preregistered analysis runs unchanged. Phase 3 then fits
the time-scale model directly to the real tutoring logs (logit = θ_i + β_k + A_k(1 − exp(−c_k·r_i·t))), compares
it with the amplitude model by AIC, and uses this note's exponent to translate the iAFM's ratio into a speed
ratio.

## References

- Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023). An astonishing regularity in student
  learning rate. *PNAS*, 120(13), e2221311120.
- Lee, H., Lichand, G., Barnard, C., Klotz, L., Thille, C., Kim, Y., & Domingue, B. W. (2026). Revisiting the
  Regularity of Student Learning Rate: Sensitivity to Which Observations Are Included. arXiv:2605.01690.
