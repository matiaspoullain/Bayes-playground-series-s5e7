library(brms)
library(ggplot2)
library(ggdist)
library(posterior)
library(data.table)

# stage2_fit_full.rds / stage2_fit_full_outcome_free.rds are both
# brmsfit_multiple objects (from brm_multiple() across 20 mice-imputed copies),
# not plain brmsfit -- but each inherits from brmsfit, and as_draws_df() returns
# the already-pooled posterior draws across all M imputations x 4 chains (20 x 4
# x 1000 post-warmup = 80000 draws): brm_multiple() combines the M chain sets
# into a single draws object per Rubin's-rules logic internally, so no separate
# manual pooling step is needed here, for either model.
#
# Model A (stage2_fit_full.rds): trained on the full training set, mice-imputed
# WITH personality_introvert as a predictor (outcome-informed, legitimate since
# training data's outcome is always known). Model B
# (stage2_fit_full_outcome_free.rds): trained on the full training set,
# mice-imputed WITHOUT personality_introvert as a predictor at all
# (outcome-free, matching what's achievable on test.csv). See design spec's
# "Two-model split".
fit_A <- readRDS("models/stage2_fit_full.rds")
fit_B <- readRDS("models/stage2_fit_full_outcome_free.rds")
dir.create("tables_figures/stage2_report", showWarnings = FALSE, recursive = TRUE)

# --- Table + Figure: odds ratios (exp of fixed-effect coefficients), 95% CrI ---
# Same pattern as report_stage1_inference.R: the full posterior draws (not just
# the summary) let the plot show the full posterior density as a "shadow" behind
# each point + interval, via ggdist's half-eye geom.
#
# Both Stage 2 models drop the 7 was_missing_* indicators that Stage 1 carried
# (both are fit on mice-imputed data, so there's no missingness left to flag) --
# these are the same 7 substantive predictors modeled the same way in Stage 1.
fixef_terms <- c(
  "time_spent_alone_z", "social_event_attendance_z", "going_outside_z",
  "friends_circle_size_z", "post_frequency_z", "stage_fear_bin",
  "drained_after_socializing_bin"
)

#' Build a long data.table of per-draw odds ratios for a set of fixed-effect terms
#'
#' @param draws_df a `posterior::as_draws_df()` result (one row per pooled draw)
#' @param terms character vector of fixed-effect term names, without the "b_"
#'   prefix `brms` uses internally
#' @return long data.table with columns `term`, `odds_ratio` (one row per
#'   posterior draw per term)
odds_ratio_draws_long <- function(draws_df, terms) {
  rbindlist(lapply(terms, function(term) {
    data.table(term = term, odds_ratio = exp(draws_df[[paste0("b_", term)]]))
  }))
}

#' Summarize per-draw odds-ratio draws into a median + 95% CrI per term
#'
#' Point estimate uses the median, not the mean, of the per-draw odds ratios.
#' The median commutes with the exp() transform (median(exp(x)) ==
#' exp(median(x))), so it agrees with exp(fixef(fit)) on the log-odds scale;
#' the mean does not, and for a wide or right-skewed posterior on the log-odds
#' scale, mean(exp(draws)) is pulled into the right tail by Jensen's
#' inequality, which can misleadingly overstate the effect relative to the
#' median-based odds ratio.
#'
#' @param posterior_or_long output of `odds_ratio_draws_long()`
#' @return one row per term: `term`, `odds_ratio`, `or_lower_95`, `or_upper_95`
summarize_odds_ratios <- function(posterior_or_long) {
  posterior_or_long[, .(
    odds_ratio = median(odds_ratio),
    or_lower_95 = quantile(odds_ratio, 0.025),
    or_upper_95 = quantile(odds_ratio, 0.975)
  ), by = term]
}

draws_df_A <- as_draws_df(fit_A)
stopifnot(nrow(draws_df_A) == 80000)
posterior_or_long_A <- odds_ratio_draws_long(draws_df_A, fixef_terms)
odds_ratios_A <- summarize_odds_ratios(posterior_or_long_A)

draws_df_B <- as_draws_df(fit_B)
stopifnot(nrow(draws_df_B) == 80000)
posterior_or_long_B <- odds_ratio_draws_long(draws_df_B, fixef_terms)
odds_ratios_B <- summarize_odds_ratios(posterior_or_long_B)

# stage_fear_bin / drained_after_socializing_bin run substantially larger for
# Model A (OR ~30/~16) than in Stage 1 (OR ~6.4/~12.7) for the same
# coefficients, and this is not just the usual "individual collinear
# coefficients are unstable" story -- the sum below is also ~5.8x larger in
# Model A than Stage 1 (see stage_fear_drained_sum.csv), which a pure
# collinearity split would not produce. The real mechanism, verified directly
# against models/stage2_mice_imp.rds: mice's imputation model for Model A
# includes personality_introvert as a predictor (standard, spec-endorsed MI
# practice -- see design spec "Missing-data strategy", not leakage). Because
# stage_fear_bin and drained_after_socializing_bin are themselves a
# near-duplicate pair (r = 0.995, see design spec "Multicollinearity") AND very
# strongly outcome-correlated, mice reconstructs their originally-missing
# values almost deterministically from the true label: averaged across the 20
# imputations, stage_fear_bin is imputed as 1 ("Yes") for 97.7% of rows with
# true personality_introvert = 1 vs. only 1.2% of rows with true
# personality_introvert = 0 (drained_after_socializing_bin: 98.7% vs. 3.1%).
# Stage 1's mode imputation, by contrast, sets every missing row to the same
# constant majority value (0/"No") regardless of the true label, and only
# partially recovers the lost signal via the separate was_missing_* indicators
# (OR 3.3/6.3 there) -- so Stage 1 has no equivalent outcome-driven boost. This
# is legitimate MI behavior for Model A per the design spec, but borders on
# circularity (the outcome directly informs the imputed value of these two
# dominant predictors for ~6-10% of rows), so both individual coefficients'
# unusually large magnitude for Model A should be read with that in mind -- the
# derived sum below is the more robust quantity to lead with for this pair.
#
# Model B's mice imputation EXCLUDES personality_introvert entirely (see
# fit_stage2_full_outcome_free.R / design spec "Two-model split"), so the
# outcome-driven-reconstruction mechanism above cannot operate for Model B's
# imputed cells. The naive expectation going in was that Model B's
# stage_fear_bin/drained_after_socializing_bin ORs would therefore sit much
# closer to Stage 1's (~6.4/~12.7) than to Model A's (~30/~16) -- but the
# actual fitted numbers (checked below) show Model B's ORs are only modestly
# smaller than Model A's (~19/~14), not close to Stage 1's. This is a genuine
# finding, not an error: only ~10.2%/6.2% of stage_fear_bin/
# drained_after_socializing_bin values are missing at all (verified against
# prepare_stage2_data() output), so both models are fit on the same ~90-94%
# of rows where these two predictors are genuinely observed, and outcome-free
# vs. outcome-informed imputation only changes the missing minority.
# Naive 2x2 tables on the OBSERVED (non-imputed) subset alone already show a
# near-deterministic relationship (raw odds ratio ~600 for each predictor
# against personality_introvert) -- so most of both models' huge coefficient
# magnitude is intrinsic to the genuinely-observed data, not an artifact of
# either imputation scheme. Model A's outcome-informed imputation of the
# missing minority pushes the OR further still, but that's a secondary effect
# on top of an already-large base rate, not the dominant driver Stage 1's much
# smaller OR would suggest. (Stage 1's much lower OR is a separate story: its
# mode imputation sets every missing row to a constant, diluting the observed
# relationship, whereas both Stage 2 models impute plausible varying values
# for the missing minority and leave the observed majority untouched.)
odds_ratios_A[
  term %in% c("stage_fear_bin", "drained_after_socializing_bin"),
  note := "near-duplicate pair, outcome-informed imputation - see sum in stage_fear_drained_sum.csv"
]
fwrite(odds_ratios_A, "tables_figures/stage2_report/odds_ratios_summary.csv")
fwrite(odds_ratios_B, "tables_figures/stage2_report/odds_ratios_summary_modelB.csv")

odds_ratios_comparison <- rbindlist(list(
  odds_ratios_A[, .(model = "A", term, odds_ratio, or_lower_95, or_upper_95)],
  odds_ratios_B[, .(model = "B", term, odds_ratio, or_lower_95, or_upper_95)]
))
fwrite(odds_ratios_comparison, "tables_figures/stage2_report/odds_ratios_comparison.csv")

cat("\n--- Model A odds-ratio table (all 7 predictors, 95% CrI) ---\n")
print(odds_ratios_A)
cat("\n--- Model B odds-ratio table (all 7 predictors, 95% CrI) ---\n")
print(odds_ratios_B)

# Explicit numeric check of the "Model B should not show Model A's
# outcome-driven inflation" expectation above, printed rather than just
# assumed. Result (see comment block above for the full explanation): Model
# B's ORs for this pair are only modestly smaller than Model A's, NOT close to
# Stage 1's much lower ORs -- a genuine surprise relative to the naive
# expectation, flagged explicitly rather than papered over.
inflated_terms <- c("stage_fear_bin", "drained_after_socializing_bin")
cat("\n--- Outcome-inflation check: Model A vs. Model B ---\n")
for (term_name in inflated_terms) {
  or_a <- odds_ratios_A[term == term_name, odds_ratio]
  or_b <- odds_ratios_B[term == term_name, odds_ratio]
  pct_reduction <- 100 * (or_a - or_b) / or_a
  cat(sprintf(
    "%s: Model A OR = %.2f, Model B OR = %.2f (%.0f%% lower, still far above Stage 1's ~6-13) - Model B is NOT free of the inflation, only modestly reduced; see comment block above for why.\n",
    term_name, or_a, or_b, pct_reduction
  ))
}

#' Build a forest plot of the full posterior odds-ratio distributions
#'
#' @param posterior_or_long long per-draw odds ratios (`odds_ratio_draws_long()`)
#' @param odds_ratios_summary matching per-term median + 95% CrI summary, used
#'   only to order terms by point estimate
#' @param title,subtitle plot labels
#' @return a ggplot object
plot_odds_ratio_forest <- function(posterior_or_long, odds_ratios_summary, title, subtitle) {
  term_order <- odds_ratios_summary[order(odds_ratio), term]
  plot_data <- copy(posterior_or_long)
  plot_data[, term := factor(term, levels = term_order)]

  ggplot(plot_data, aes(y = term, x = odds_ratio)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
    stat_halfeye(
      .width = 0.95, point_interval = "median_qi",
      fill = "#4C72B0", slab_alpha = 0.55,
      colour = "#2A4D77", linewidth = 1.1, point_size = 3
    ) +
    scale_x_log10() +
    labs(x = "Odds ratio (log10 scale)", y = NULL, title = title, subtitle = subtitle) +
    theme_minimal(base_size = 12)
}

plot_odds_ratio_forest(
  posterior_or_long_A, odds_ratios_A,
  title = "Stage 2 Model A odds ratios by predictor",
  subtitle = "Complete-row model (outcome-informed imputation); median & 95% CrI (dashed line: no effect, OR = 1)"
)
ggsave(
  "tables_figures/stage2_report/odds_ratios_forest_plot.png",
  width = 10, height = 7, bg = "white"
)

plot_odds_ratio_forest(
  posterior_or_long_B, odds_ratios_B,
  title = "Stage 2 Model B odds ratios by predictor",
  subtitle = "Incomplete-row model (outcome-free imputation); median & 95% CrI (dashed line: no effect, OR = 1)"
)
ggsave(
  "tables_figures/stage2_report/odds_ratios_forest_plot_modelB.png",
  width = 10, height = 7, bg = "white"
)

# Side-by-side comparison plot: both models' point estimates + 95% CrI on one
# axis, ordered by Model A's point estimate, so the outcome-inflation gap for
# stage_fear_bin/drained_after_socializing_bin is visible directly.
term_order_comparison <- odds_ratios_A[order(odds_ratio), term]
odds_ratios_comparison[, term := factor(term, levels = term_order_comparison)]

ggplot(odds_ratios_comparison, aes(y = term, x = odds_ratio, colour = model)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  geom_pointrange(
    aes(xmin = or_lower_95, xmax = or_upper_95),
    position = position_dodge(width = 0.5), linewidth = 0.9, size = 0.6
  ) +
  scale_x_log10() +
  scale_colour_manual(values = c(A = "#4C72B0", B = "#C44E52")) +
  labs(
    x = "Odds ratio (log10 scale)", y = NULL, colour = "Model",
    title = "Stage 2 odds ratios: Model A vs. Model B",
    subtitle = "Median & 95% CrI (dashed line: no effect, OR = 1); Model A = complete rows, Model B = incomplete rows"
  ) +
  theme_minimal(base_size = 12)
ggsave(
  "tables_figures/stage2_report/odds_ratios_comparison_plot.png",
  width = 10, height = 7, bg = "white"
)

# --- Derived quantity: stage_fear_bin + drained_after_socializing_bin posterior sum ---
# Same computation as report_stage1_inference.R's identically-named block: per-draw
# sum (not a sum of point estimates), so the CrI properly reflects the joint
# posterior rather than treating the two coefficients as independent. Per the
# design spec "Multicollinearity" section, this sum -- not the individual
# coefficients above -- is what "Reporting insights" should lead with for this
# pair, and for Model A it is the more important comparison point against
# Stage 1's own sum (tables_figures/stage1_report/stage_fear_drained_sum.csv)
# given the outcome-informed-imputation caveat documented above. Model B's sum
# is the fairer like-for-like comparison against Stage 1, since neither uses
# outcome-informed imputation.
#' Compute the posterior sum of stage_fear_bin + drained_after_socializing_bin
#'
#' @param draws_df a `posterior::as_draws_df()` result
#' @return one-row data.table: term (label), estimate (mean), q2.5, q97.5
stage_fear_drained_sum <- function(draws_df) {
  combined_effect <- draws_df$b_stage_fear_bin + draws_df$b_drained_after_socializing_bin
  data.table(
    term = "stage_fear_bin + drained_after_socializing_bin (sum)",
    estimate = mean(combined_effect),
    q2.5 = quantile(combined_effect, 0.025),
    q97.5 = quantile(combined_effect, 0.975)
  )
}

sum_A <- stage_fear_drained_sum(draws_df_A)
sum_B <- stage_fear_drained_sum(draws_df_B)
fwrite(sum_A, "tables_figures/stage2_report/stage_fear_drained_sum.csv")
fwrite(sum_B, "tables_figures/stage2_report/stage_fear_drained_sum_modelB.csv")

cat("\n--- Model A: stage_fear_bin + drained_after_socializing_bin posterior sum ---\n")
print(sum_A)
cat("\n--- Model B: stage_fear_bin + drained_after_socializing_bin posterior sum ---\n")
print(sum_B)
