library(brms)
library(ggplot2)
library(ggdist)
library(posterior)
library(data.table)

# stage2_fit_full.rds is a brmsfit_multiple object (from brm_multiple() across
# 20 mice-imputed copies), not a plain brmsfit -- but it inherits from brmsfit,
# and as_draws_df() returns the already-pooled posterior draws across all M
# imputations x 4 chains (20 x 4 x 1000 post-warmup = 80000 draws): brm_multiple()
# combines the M chain sets into a single draws object per Rubin's-rules logic
# internally, so no separate manual pooling step is needed here.
fit <- readRDS("models/stage2_fit_full.rds")
dir.create("tables_figures/stage2_report", showWarnings = FALSE, recursive = TRUE)

# --- Table + Figure: odds ratios (exp of fixed-effect coefficients), 95% CrI ---
# Same pattern as report_stage1_inference.R: the full posterior draws (not just
# the summary) let the plot show the full posterior density as a "shadow" behind
# each point + interval, via ggdist's half-eye geom.
#
# Stage 2 drops the 7 was_missing_* indicators that Stage 1 carried (this model
# is fit on mice-imputed data, so there's no missingness left to flag) -- these
# are the same 7 substantive predictors modeled the same way in Stage 1.
fixef_terms <- c(
  "time_spent_alone_z", "social_event_attendance_z", "going_outside_z",
  "friends_circle_size_z", "post_frequency_z", "stage_fear_bin",
  "drained_after_socializing_bin"
)
draws_df <- as_draws_df(fit)
stopifnot(nrow(draws_df) == 80000)

posterior_or_long <- rbindlist(lapply(fixef_terms, function(term) {
  data.table(term = term, odds_ratio = exp(draws_df[[paste0("b_", term)]]))
}))

# Point estimate uses the median, not the mean, of the per-draw odds ratios. The
# median commutes with the exp() transform (median(exp(x)) == exp(median(x))), so
# it agrees with exp(fixef(fit)) on the log-odds scale; the mean does not, and for
# a wide or right-skewed posterior on the log-odds scale, mean(exp(draws)) is pulled
# into the right tail by Jensen's inequality, which can misleadingly overstate the
# effect relative to the median-based odds ratio.
odds_ratios <- posterior_or_long[, .(
  odds_ratio = median(odds_ratio),
  or_lower_95 = quantile(odds_ratio, 0.025),
  or_upper_95 = quantile(odds_ratio, 0.975)
), by = term]

# stage_fear_bin / drained_after_socializing_bin run substantially larger here
# (OR ~30/~16) than in Stage 1 (OR ~6.4/~12.7) for the same coefficients, and this
# is not just the usual "individual collinear coefficients are unstable" story --
# the sum below is also ~5.8x larger in Stage 2 than Stage 1 (see
# stage_fear_drained_sum.csv), which a pure collinearity split would not produce.
# The real mechanism, verified directly against models/stage2_mice_imp.rds: mice's
# imputation model includes personality_introvert as a predictor (standard,
# spec-endorsed MI practice -- see design spec "Missing-data strategy", not
# leakage). Because stage_fear_bin and drained_after_socializing_bin are
# themselves a near-duplicate pair (r = 0.995, see design spec
# "Multicollinearity") AND very strongly outcome-correlated, mice reconstructs
# their originally-missing values almost deterministically from the true label:
# averaged across the 20 imputations, stage_fear_bin is imputed as 1 ("Yes") for
# 97.7% of rows with true personality_introvert = 1 vs. only 1.2% of rows with
# true personality_introvert = 0 (drained_after_socializing_bin: 98.7% vs. 3.1%).
# Stage 1's mode imputation, by contrast, sets every missing row to the same
# constant majority value (0/"No") regardless of the true label, and only
# partially recovers the lost signal via the separate was_missing_* indicators
# (OR 3.3/6.3 there) -- so Stage 1 has no equivalent outcome-driven boost. This is
# legitimate MI behavior per the design spec, but borders on circularity (the
# outcome directly informs the imputed value of these two dominant predictors for
# ~6-10% of rows), so both individual coefficients' unusually large magnitude
# should be read with that in mind -- the derived sum below is the more robust
# quantity to lead with for this pair, per the design spec.
odds_ratios[
  term %in% c("stage_fear_bin", "drained_after_socializing_bin"),
  note := "near-duplicate pair, outcome-informed imputation - see sum in stage_fear_drained_sum.csv"
]
fwrite(odds_ratios, "tables_figures/stage2_report/odds_ratios_summary.csv")
cat("\n--- Stage 2 odds-ratio table (all 7 predictors, 95% CrI) ---\n")
print(odds_ratios)

term_order <- odds_ratios[order(odds_ratio), term]
posterior_or_long[, term := factor(term, levels = term_order)]

ggplot(posterior_or_long, aes(y = term, x = odds_ratio)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  stat_halfeye(
    .width = 0.95, point_interval = "median_qi",
    fill = "#4C72B0", slab_alpha = 0.55,
    colour = "#2A4D77", linewidth = 1.1, point_size = 3
  ) +
  scale_x_log10() +
  labs(
    x = "Odds ratio (log10 scale)", y = NULL,
    title = "Stage 2 odds ratios by predictor",
    subtitle = "Posterior distribution, median & 95% CrI (dashed line: no effect, OR = 1)"
  ) +
  theme_minimal(base_size = 12)
ggsave(
  "tables_figures/stage2_report/odds_ratios_forest_plot.png",
  width = 10, height = 7, bg = "white"
)

# --- Derived quantity: stage_fear_bin + drained_after_socializing_bin posterior sum ---
# Same computation as report_stage1_inference.R's identically-named block: per-draw
# sum (not a sum of point estimates), so the CrI properly reflects the joint
# posterior rather than treating the two coefficients as independent. Per the
# design spec "Multicollinearity" section, this sum -- not the individual
# coefficients above -- is what "Reporting insights" should lead with for this
# pair, and it is the more important comparison point against Stage 1's own sum
# (tables_figures/stage1_report/stage_fear_drained_sum.csv) given the
# outcome-informed-imputation caveat documented above.
combined_effect <- draws_df$b_stage_fear_bin + draws_df$b_drained_after_socializing_bin
stage_fear_drained_sum <- data.table(
  term = "stage_fear_bin + drained_after_socializing_bin (sum)",
  estimate = mean(combined_effect),
  q2.5 = quantile(combined_effect, 0.025),
  q97.5 = quantile(combined_effect, 0.975)
)
fwrite(stage_fear_drained_sum, "tables_figures/stage2_report/stage_fear_drained_sum.csv")
cat("\n--- stage_fear_bin + drained_after_socializing_bin posterior sum ---\n")
print(stage_fear_drained_sum)
