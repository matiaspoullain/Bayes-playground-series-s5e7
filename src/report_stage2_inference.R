library(brms)
library(ggplot2)
library(posterior)
library(ggdist)
library(data.table)

# stage2_fit_full.rds is a brmsfit_multiple object (from brm_multiple() across 20
# mice-imputed copies), not a plain brmsfit -- but it inherits from brmsfit, and
# as_draws_df() returns the already-pooled posterior draws across all M imputations
# x 4 chains (20 x 4 x 1000 post-warmup = 80000 draws): brm_multiple() combines the
# M chain sets into a single draws object per Rubin's-rules logic internally, so no
# separate manual pooling step is needed here.
fit <- readRDS("models/stage2_fit_full.rds")
dir.create("tables_figures/stage2_report", showWarnings = FALSE, recursive = TRUE)

# --- Table + Figure: odds ratios (exp of fixed-effect coefficients), 50%/95% CrI ---
# Same two-width interval style as report_stage1_inference.R (thick bar = 50% CrI,
# thin line = 95% CrI, dot = median) -- see that script for the rationale.
#
# 6 predictors -- social_event_attendance_z dropped per Stage 1's DAG-motivated
# coefficient-shift check (fully explained by stage_fear_bin +
# drained_after_socializing_bin; see README.md "Stage 1 findings").
fixef_terms <- c(
  "time_spent_alone_z", "going_outside_z",
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
#' @return one row per term: `term`, `odds_ratio`, `or_lower_50`, `or_upper_50`,
#'   `or_lower_95`, `or_upper_95`
summarize_odds_ratios <- function(posterior_or_long) {
  posterior_or_long[, .(
    odds_ratio = median(odds_ratio),
    or_lower_50 = quantile(odds_ratio, 0.25),
    or_upper_50 = quantile(odds_ratio, 0.75),
    or_lower_95 = quantile(odds_ratio, 0.025),
    or_upper_95 = quantile(odds_ratio, 0.975)
  ), by = term]
}

draws_df <- as_draws_df(fit)
stopifnot(nrow(draws_df) == 80000)
posterior_or_long <- odds_ratio_draws_long(draws_df, fixef_terms)
odds_ratios <- summarize_odds_ratios(posterior_or_long)

# stage_fear_bin / drained_after_socializing_bin run substantially larger here
# (OR ~19/~14) than in Stage 1 (OR ~6.4/~12.7) for the same coefficients, and
# this is not just the usual "individual collinear coefficients are unstable"
# story -- the sum below is also much larger than Stage 1's, which a pure
# collinearity split would not produce. Investigated directly (an earlier design
# iteration compared this model against a second one whose imputation DID
# include the outcome -- see git history around the "two-model split" -- and
# found the two gave very similar odds ratios for this pair, ruling out
# outcome-informed imputation as the main driver here): only ~10.2%/6.2% of
# stage_fear_bin/drained_after_socializing_bin values are missing at all, so
# this model is fit predominantly on the ~90-94% of rows where these two
# predictors are genuinely observed. A naive 2x2 table on the OBSERVED
# (non-imputed) subset alone already shows a near-deterministic relationship
# (raw odds ratio ~670-676 for each predictor against personality_introvert) --
# so most of the coefficient magnitude is intrinsic to the genuinely-observed
# data, not an artifact of the imputation scheme. Stage 1's much lower OR is a
# separate story: its mode imputation sets every missing row to a constant,
# diluting the observed relationship, whereas this model imputes plausible
# varying values for the missing minority and leaves the observed majority
# untouched -- so the derived sum below (not the individual coefficients) is
# still the more robust quantity to lead with for this collinear pair (r =
# 0.995, see design spec "Multicollinearity").
odds_ratios[
  term %in% c("stage_fear_bin", "drained_after_socializing_bin"),
  note := "near-duplicate pair (r=0.995); large OR is mostly intrinsic to the observed data, not an imputation artifact - see sum in stage_fear_drained_sum.csv"
]
fwrite(odds_ratios, "tables_figures/stage2_report/odds_ratios_summary.csv")

cat("\n--- Stage 2 odds-ratio table (all 7 predictors, 50%/95% CrI) ---\n")
print(odds_ratios)

#' Build a half-eye plot (posterior density + median, 50%/95% CrI) of the odds ratios
#'
#' Same `ggdist::stat_halfeye()` idiom as Bayes-playground-series-s5e3's
#' `report_inference.R` -- the density "shadow" makes skew/multimodality in the
#' collinear pair's posteriors visible, not just a single summary pointrange.
#' `point_interval = "median_qi"` matches this project's existing
#' median-not-mean convention (see `summarize_odds_ratios()` roxygen above).
#'
#' @param posterior_or_long per-draw long odds ratios (`odds_ratio_draws_long()`)
#' @param odds_ratios_summary per-term median summary (`summarize_odds_ratios()`),
#'   used only to order the y-axis by point estimate
#' @return a ggplot object
plot_odds_ratio_forest <- function(posterior_or_long, odds_ratios_summary) {
  term_order <- odds_ratios_summary[order(odds_ratio), term]
  plot_data <- copy(posterior_or_long)
  plot_data[, term := factor(term, levels = term_order)]

  ggplot(plot_data, aes(y = term, x = odds_ratio)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
    stat_halfeye(
      .width = c(0.5, 0.95), point_interval = "median_qi",
      fill = "#2A4D77", slab_alpha = 0.55,
      colour = "#1B3554", linewidth = 1.1, point_size = 2.4
    ) +
    scale_x_log10() +
    labs(
      x = "Odds ratio (log10 scale)", y = NULL,
      title = "Stage 2 odds ratios by predictor",
      subtitle = paste0(
        "Posterior distribution, median, thick bar = 50% CrI, thin line = 95% CrI\n",
        "(dashed line marks no effect, OR = 1)"
      )
    ) +
    theme_minimal(base_size = 12)
}

plot_odds_ratio_forest(posterior_or_long, odds_ratios)
ggsave(
  "tables_figures/stage2_report/odds_ratios_forest_plot.png",
  width = 10, height = 7, bg = "white"
)

# --- Derived quantity: stage_fear_bin + drained_after_socializing_bin posterior sum ---
# Same computation as report_stage1_inference.R's identically-named block: per-draw
# sum (not a sum of point estimates), so the CrI properly reflects the joint
# posterior rather than treating the two coefficients as independent. Per the
# design spec "Multicollinearity" section, this sum -- not the individual
# coefficients above -- is what "Reporting insights" should lead with for this pair.
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
