# SPDX-License-Identifier: Apache-2.0
# learning-rate-regularity phase 3 (PLAN-3.md): R + lme4 fits. Made with AI agents working in the name of MV2 of
# Munim, Inc.
# Usage:
#   Rscript fit3.R curve in.csv out.csv c1,c2,...      one KC (columns student,t,y). For each c on the grid:
#       y ~ 1 + x + (1 | student), x = 1 - exp(-c t). Writes one row per c (c, loglik, beta, A, sd_theta, ...).
#   Rscript fit3.R het in.csv out.csv lin|amp|ts|both  all KCs (columns student,kc,t,y,g,a,h). Fixed part
#       y ~ 0 + kc + kc:g (each KC's saturating curve, its time constant fixed by the curve step). Student part:
#       lin  (1 + t | student)       students differ in a per-opportunity rate (the iAFM's kind of difference)
#       amp  (1 + a | student)       students differ in how high their curve rises   (a = A_k g)
#       ts   (1 + h | student)       students differ in how soon they get there      (h = A_k c_k t exp(-c_k t))
#       both (1 + a + h | student)
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
inp <- args[2]
outp <- args[3]
spec <- args[4]
suppressPackageStartupMessages(library(lme4))
d <- read.csv(inp)
d$student <- factor(d$student)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
quiet_fit <- function(f, data) {
  warns <- character(0)
  fit <- withCallingHandlers(
    glmer(f, data = data, family = binomial, control = ctrl),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  list(fit = fit, n_warn = length(warns))
}
if (mode == "curve") {
  cs <- as.numeric(strsplit(spec, ",")[[1]])
  rows <- lapply(cs, function(cc) {
    dd <- d
    dd$x <- 1 - exp(-cc * dd$t)
    tryCatch({
      r <- quiet_fit(y ~ 1 + x + (1 | student), dd)
      fe <- fixef(r$fit)
      vc <- as.data.frame(VarCorr(r$fit))
      data.frame(c = cc, loglik = as.numeric(logLik(r$fit)), beta = unname(fe[1]), A = unname(fe[2]),
                 sd_theta = vc$sdcor[1], n_warn = r$n_warn, nobs = nrow(dd), error = 0)
    }, error = function(e) {
      data.frame(c = cc, loglik = NA, beta = NA, A = NA, sd_theta = NA, n_warn = NA, nobs = nrow(dd), error = 1)
    })
  })
  write.csv(do.call(rbind, rows), outp, row.names = FALSE)
} else {
  d$kc <- factor(d$kc)
  f <- switch(spec,
              lin = y ~ 0 + kc + kc:g + (1 + t | student),
              amp = y ~ 0 + kc + kc:g + (1 + a | student),
              ts = y ~ 0 + kc + kc:g + (1 + h | student),
              both = y ~ 0 + kc + kc:g + (1 + a + h | student))
  res <- tryCatch({
    if (is.null(f)) stop("unknown model")
    r <- quiet_fit(f, d)
    vc <- as.data.frame(VarCorr(r$fit))
    s <- vc[vc$grp == "student" & is.na(vc$var2), ]
    get_sd <- function(v) {
      x <- s$sdcor[s$var1 == v]
      if (length(x) == 0) NA else x[1]
    }
    cr <- vc[vc$grp == "student" & !is.na(vc$var2), ]
    ll <- logLik(r$fit)
    data.frame(sd_theta = get_sd("(Intercept)"), sd_t = get_sd("t"), sd_a = get_sd("a"), sd_h = get_sd("h"),
               corr1 = if (nrow(cr) > 0) cr$sdcor[1] else NA, loglik = as.numeric(ll), aic = AIC(r$fit),
               df = attr(ll, "df"), singular = isSingular(r$fit), n_warn = r$n_warn, nobs = nrow(d), error = 0)
  }, error = function(e) {
    data.frame(sd_theta = NA, sd_t = NA, sd_a = NA, sd_h = NA, corr1 = NA, loglik = NA, aic = NA, df = NA,
               singular = NA, n_warn = NA, nobs = nrow(d), error = 1)
  })
  write.csv(res, outp, row.names = FALSE)
}
