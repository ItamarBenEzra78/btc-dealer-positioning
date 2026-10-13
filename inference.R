# inference.R — Markov regime-switching inference in R (MSwM).
#
# Consumes data/market_daily.csv (produced by data_layer.py) and fits a
# two-state Markov-switching model on percent log-returns: do distinct
# low/high-vol regimes exist, with their own means, volatilities and sticky
# transitions?
#
# GARCH(1,1) is run on the Python side (stats_spine.garch_persistence, `arch`);
# if rugarch is installed it is also fitted here and its conditional volatility
# is written alongside the regime probabilities.
#
# Outputs:
#   data/r_regime.csv           per-day smoothed P(high-vol regime) [+ cond_vol]
#   data/r_regime_summary.json  regime means / sds / stay probabilities, read by
#                               the dashboard so its caption is never hard-coded
#
# Usage: Rscript inference.R [path/to/market.csv]

suppressMessages(library(MSwM))
HAVE_GARCH <- requireNamespace("rugarch", quietly = TRUE)
if (HAVE_GARCH) suppressMessages(library(rugarch))
HAVE_JSON <- requireNamespace("jsonlite", quietly = TRUE)

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else "data/market_daily.csv"
set.seed(42)                              # msmFit's EM start is randomised

df <- read.csv(path)
df$logret <- c(NA, diff(log(df$close)))
ret <- na.omit(df$logret) * 100          # percent returns, better conditioning
cat(sprintf("Loaded %d daily returns (%s -> %s)\n\n",
            length(ret), df$date[1], df$date[nrow(df)]))

# ---------------------------------------------------------------------------
# 1. GARCH(1,1): optional here, always available in Python
# ---------------------------------------------------------------------------
cond_vol <- rep(NA_real_, length(ret))
if (HAVE_GARCH) {
  cat("== [1] GARCH(1,1) — volatility clustering (rugarch) ==\n")
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std")
  fit <- ugarchfit(spec, data = ret, solver = "hybrid")
  cf <- coef(fit)
  persistence <- as.numeric(cf["alpha1"] + cf["beta1"])
  cat(sprintf("  alpha1=%.4f  beta1=%.4f  persistence=%.4f  %s\n",
              cf["alpha1"], cf["beta1"], persistence,
              ifelse(persistence > 0.9, "-> strong clustering", "-> weak")))
  cond_vol <- as.numeric(sigma(fit))
} else {
  cat("== [1] GARCH(1,1) — skipped here: rugarch not installed ==\n")
  cat("  The Python side fits GARCH(1,1) with `arch` (see stats_spine.py [G]).\n")
}

# ---------------------------------------------------------------------------
# 2. Markov regime-switching: do two regimes exist?
# ---------------------------------------------------------------------------
cat("\n== [2] Markov regime-switching (2 states) ==\n")
mod  <- lm(ret ~ 1)
msm  <- msmFit(mod, k = 2, sw = c(TRUE, TRUE), control = list(parallel = FALSE))
means <- msm@Coef[, 1]
sds   <- msm@std
trans <- msm@transMat
hi <- which.max(sds)                     # high-vol regime index
lo <- setdiff(1:2, hi)
cat(sprintf("  regime LOW-vol : mean=%+.3f%%  sd=%.3f%%\n", means[lo], sds[lo]))
cat(sprintf("  regime HIGH-vol: mean=%+.3f%%  sd=%.3f%%\n", means[hi], sds[hi]))
cat(sprintf("  stay(LOW)=%.3f   stay(HIGH)=%.3f  (>0.9 => sticky regimes)\n",
            trans[lo, lo], trans[hi, hi]))
cat(sprintf("  interpretation: low-vol regime %s; high-vol regime %s\n",
            ifelse(means[lo] > 0, "drifts up", "flat/down"),
            ifelse(means[hi] < means[lo], "carries the downside", "mixed")))

# Smoothed probability of the high-vol regime, aligned back to dates.
# The first return has no history, so its smoothed probability is the model's
# initial-state prior (reported as 1.0 by MSwM), not an inference from data.
p_high <- as.numeric(msm@Fit@smoProb[, hi])
n <- length(p_high)
pad <- function(v, n) { if (length(v) >= n) tail(v, n) else c(rep(NA, n - length(v)), v) }
out <- data.frame(
  date     = pad(df$date, n),
  cond_vol = pad(cond_vol, n),
  p_high_vol_regime = round(p_high, 4)
)
write.csv(out, "data/r_regime.csv", row.names = FALSE)
cat(sprintf("\nSaved per-day regime prob -> data/r_regime.csv (%d rows; row 1 is the prior)\n", n))

summary <- list(
  n_returns   = length(ret),
  date_start  = df$date[1],
  date_end    = df$date[nrow(df)],
  low_mean_pct  = round(unname(means[lo]), 3),
  low_sd_pct    = round(unname(sds[lo]), 3),
  low_stay      = round(trans[lo, lo], 3),
  high_mean_pct = round(unname(means[hi]), 3),
  high_sd_pct   = round(unname(sds[hi]), 3),
  high_stay     = round(trans[hi, hi], 3)
)
if (HAVE_JSON) {
  jsonlite::write_json(summary, "data/r_regime_summary.json", auto_unbox = TRUE, pretty = TRUE)
  cat("Saved regime parameters -> data/r_regime_summary.json\n")
} else {
  cat("jsonlite not installed: data/r_regime_summary.json not written\n")
}
