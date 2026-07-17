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

fwrite(odds_ratios, "tables_figures/stage2_report/odds_ratios_summary.csv")
cat("\n--- Stage 2 odds-ratio table (all 7 predictors, 95% CrI) ---\n")
print(odds_ratios)

# Observation only (not a required derived-quantity block for this report):
# stage_fear_bin and drained_after_socializing_bin are still the same
# near-duplicate pair as in Stage 1 (r = 0.995, see design spec
# "Multicollinearity") -- both individual CrIs below still exclude OR = 1, so as
# in Stage 1 this is confirmatory rather than a rescue situation.

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
