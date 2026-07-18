library(brms)
library(ggplot2)
library(posterior)
library(data.table)

fit <- readRDS("models/stage1_fit_full.rds")
dir.create("tables_figures/stage1_report", showWarnings = FALSE, recursive = TRUE)

# --- Table + Figure: odds ratios (exp of fixed-effect coefficients), 50%/95% CrI ---
# Two-width interval plot (thick bar = 50% CrI, thin line = 95% CrI, dot = median) --
# the standard bayesplot::mcmc_intervals look. Chosen over a full posterior-density
# plot (e.g. a halfeye) because with 14 terms on one axis, density shadows mostly
# overlap and add clutter; the two-width interval keeps the "how sure are we"
# information (a wide 95% but tight 50% signals a long tail, not just noise) without it.

fixef_terms <- c(
  "time_spent_alone_z", "social_event_attendance_z", "going_outside_z",
  "friends_circle_size_z", "post_frequency_z", "stage_fear_bin",
  "drained_after_socializing_bin", "was_missing_time_spent_alone",
  "was_missing_social_event_attendance", "was_missing_going_outside",
  "was_missing_friends_circle_size", "was_missing_post_frequency",
  "was_missing_stage_fear", "was_missing_drained_after_socializing"
)
draws_df <- as_draws_df(fit)

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
  or_lower_50 = quantile(odds_ratio, 0.25),
  or_upper_50 = quantile(odds_ratio, 0.75),
  or_lower_95 = quantile(odds_ratio, 0.025),
  or_upper_95 = quantile(odds_ratio, 0.975)
), by = term]

fwrite(odds_ratios, "tables_figures/stage1_report/odds_ratios_summary.csv")
cat("\n--- Stage 1 odds-ratio table (all 14 predictors, 50%/95% CrI) ---\n")
print(odds_ratios)

term_order <- odds_ratios[order(odds_ratio), term]
odds_ratios[, `:=`(
  term = factor(term, levels = term_order),
  is_missing_term = grepl("^was_missing_", term)
)]

ggplot(odds_ratios, aes(y = term, x = odds_ratio, colour = is_missing_term)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  geom_linerange(aes(xmin = or_lower_95, xmax = or_upper_95), linewidth = 0.7) +
  geom_linerange(aes(xmin = or_lower_50, xmax = or_upper_50), linewidth = 2.6) +
  geom_point(size = 2.4, colour = "white") +
  geom_point(size = 1.3) +
  scale_colour_manual(
    values = c(`FALSE` = "#2A4D77", `TRUE` = "#B23A48"),
    labels = c(`FALSE` = "raw predictor", `TRUE` = "was_missing_* indicator"),
    name = NULL
  ) +
  scale_x_log10() +
  labs(
    x = "Odds ratio (log10 scale)", y = NULL,
    title = "Stage 1 odds ratios by predictor",
    subtitle = "Median, thick bar = 50% CrI, thin line = 95% CrI (dashed: no effect, OR = 1)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")
ggsave(
  "tables_figures/stage1_report/odds_ratios_forest_plot.png",
  width = 10, height = 7, bg = "white"
)

# --- Derived quantity: stage_fear_bin + drained_after_socializing_bin posterior sum ---
# stage_fear_bin and drained_after_socializing_bin are themselves near-duplicate
# predictors (r = 0.995, see design spec "Multicollinearity"). Their individual
# coefficients can still be well-identified when the *sum* is; here, per the Task 7
# fit, both individual CrIs already exclude 0 (1.52 [1.29, 1.74], 1.61 [1.38, 1.84]),
# so this block is confirmatory rather than a rescue -- but the sum, computed
# per-draw (not from point estimates), is still the more robust reportable quantity
# for "how strongly do fear/drain jointly predict introversion", since it does not
# depend on how the shared signal happens to split between the two collinear terms.
combined_effect <- draws_df$b_stage_fear_bin + draws_df$b_drained_after_socializing_bin
stage_fear_drained_sum <- data.table(
  term = "stage_fear_bin + drained_after_socializing_bin (sum)",
  estimate = mean(combined_effect),
  q2.5 = quantile(combined_effect, 0.025),
  q97.5 = quantile(combined_effect, 0.975)
)
fwrite(stage_fear_drained_sum, "tables_figures/stage1_report/stage_fear_drained_sum.csv")
cat("\n--- stage_fear_bin + drained_after_socializing_bin posterior sum ---\n")
print(stage_fear_drained_sum)

# --- was_missing_* MNAR screen: which missingness indicators have a 95% CrI excluding 0? ---
missing_terms <- grep("^was_missing_", fixef_terms, value = TRUE)
missing_summary <- odds_ratios[term %in% missing_terms]
missing_summary[, crosses_or_1 := or_lower_95 <= 1 & or_upper_95 >= 1]
cat("\n--- was_missing_* MNAR screen (95% CrI on the odds-ratio scale) ---\n")
print(missing_summary[order(term)])

confirmed_mnar <- missing_summary[crosses_or_1 == FALSE, term]

cat("\n--- Confirmed MNAR effects (95% CrI excludes OR = 1) ---\n")
if (length(confirmed_mnar) > 0) {
  print(confirmed_mnar)
} else {
  cat("None -- no was_missing_* term's 95% CrI excludes OR = 1.\n")
}
