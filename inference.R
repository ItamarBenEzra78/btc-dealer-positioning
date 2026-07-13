# inference.R — Heavy statistical inference in R (the layer where R beats Python).
#
# Consumes data/market_daily.csv (produced by data_layer.py) and runs two
# formal models that directly test the article's regime thesis:
#
#   1. GARCH(1,1)  — is volatility clustering real & persistent? (rugarch)
#   2. Markov regime-switching — do distinct low/high-vol regimes exist,
#      with their own means & volatilities and sticky transitions? (MSwM)
#
# Outputs: console summary + data/r_regime.csv (per-day smoothed P(high-vol regime)
# and GARCH conditional volatility) for the dashboard.

suppressMessages(library(MSwM))
HAVE_GARCH <- requireNamespace("rugarch", quietly = TRUE)
if (HAVE_GARCH) suppressMessages(library(rugarch))

df <- read.csv("data/market_daily.csv")
df$logret <- c(NA, diff(log(df$close)))
ret <- na.omit(df$logret) * 100          # percent returns, better conditioning
cat(sprintf("Loaded %d daily returns (%s -> %s)\n\n",
            length(ret), df$date[1], df$date[nrow(df)]))

# ---------------------------------------------------------------------------
# 1. GARCH(1,1): volatility clustering  (optional — needs rugarch)
# ---------------------------------------------------------------------------
cond_vol <- rep(NA_real_, length(ret))
if (HAVE_GARCH) {
  cat("== [1] GARCH(1,1) — volatility clustering ==\n")
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std")
  fit <- ugarchfit(spec, data = ret, solver = "hybrid")
  cf <- coef(fit)
  persistence <- as.numeric(cf["alpha1"] + cf["beta1"])
  cat(sprintf("  alpha1=%.4f  beta1=%.4f  persistence=%.4f  %s\n",
              cf["alpha1"], cf["beta1"], persistence,
              ifelse(persistence > 0.9, "-> strong clustering (thesis supported)", "-> weak")))
  cond_vol <- as.numeric(sigma(fit))
} else {
  cat("== [1] GARCH(1,1) — SKIPPED (rugarch unavailable on R 4.6) ==\n")
  cat("  Volatility clustering already confirmed in Python (Levene p=0.009).\n")
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
            ifelse(means[lo] > 0, "drifts up (sticky/bullish)", "flat/down"),
            ifelse(means[hi] < means[lo], "carries the downside (risk-off)", "mixed")))

# smoothed probability of the high-vol regime, aligned back to dates
p_high <- as.numeric(msm@Fit@smoProb[, hi])
n <- length(p_high)
pad <- function(v, n) { if (length(v) >= n) tail(v, n) else c(rep(NA, n - length(v)), v) }
out <- data.frame(
  date     = pad(df$date, n),
  cond_vol = pad(cond_vol, n),
  p_high_vol_regime = round(p_high, 4)
)
write.csv(out, "data/r_regime.csv", row.names = FALSE)
cat(sprintf("\nSaved per-day conditional vol + regime prob -> data/r_regime.csv (%d rows)\n", n))
cat("\nNote: estudy2 is installed for later BTC-vs-SPX event studies (macro layer).\n")
