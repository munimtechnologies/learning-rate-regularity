# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity phase 2: fit the individualized Additive Factors Model (iAFM) to one real-data window.
# Made with AI agents working in the name of MV2 of Munim, Inc.
# Usage: Rscript fit2.R in.csv out.csv linear|log
#   linear: y ~ 0 + kc + kc:t + (1 + t | student)          (iAFM as in Liu & Koedinger 2017; Koedinger et al. 2023)
#   log:    the same with log(1 + t) in place of t          (a better-shaped, concave learning curve)
# Same model as phase 1's fit.R, plus AIC, BIC and log-likelihood for comparing the two shapes on the same data.
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]
outp <- args[2]
model <- args[3]
suppressPackageStartupMessages(library(lme4))
d <- read.csv(inp)
d$kc <- factor(d$kc)
d$student <- factor(d$student)
if (model == "log") d$x <- log1p(d$t) else d$x <- d$t
warns <- character(0)
res <- tryCatch({
  fit <- withCallingHandlers(
    glmer(y ~ 0 + kc + kc:x + (1 + x | student), data = d, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  vc <- as.data.frame(VarCorr(fit))
  s <- vc[vc$grp == "student", ]
  sd_theta <- s$sdcor[s$var1 == "(Intercept)" & is.na(s$var2)]
  sd_delta <- s$sdcor[s$var1 == "x" & is.na(s$var2)]
  corr <- s$sdcor[!is.na(s$var2)]
  if (length(corr) == 0) corr <- NA
  fe <- fixef(fit)
  slopes <- fe[grepl(":x$", names(fe))]
  data.frame(sd_theta = sd_theta[1], sd_delta = sd_delta[1], corr = corr[1], gbar = mean(slopes),
             aic = AIC(fit), bic = BIC(fit), loglik = as.numeric(logLik(fit)),
             singular = isSingular(fit), n_warn = length(warns), nobs = nrow(d), error = 0)
}, error = function(e) {
  data.frame(sd_theta = NA, sd_delta = NA, corr = NA, gbar = NA, aic = NA, bic = NA, loglik = NA,
             singular = NA, n_warn = length(warns), nobs = nrow(d), error = 1)
})
write.csv(res, outp, row.names = FALSE)
