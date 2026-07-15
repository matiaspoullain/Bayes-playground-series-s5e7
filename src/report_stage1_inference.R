library(brms)
library(ggplot2)
library(ggdist)
library(posterior)
library(data.table)

fit <- readRDS("models/stage1_fit_full.rds")
dir.create("tables_figures/stage1_report", showWarnings = FALSE, recursive = TRUE)

# --- Table + Figure: odds ratios (exp of fixed-effect coefficients), 95% CrI ---
# Uses the actual posterior draws (not just the summary) so the plot can show the
# full posterior density as a "shadow" behind each point + interval, via ggdist's
# half-eye geom -- not just a single summary pointrange. This matters here more
# than usual: several was_missing_* terms have very wide, near-prior-width
# posteriors, and the shape (not just the endpoints) is part of the finding.

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
# a wide, right-skewed posterior (e.g. was_missing_stage_fear, SE ~2.5 on the log-odds
# scale) mean(exp(draws)) is pulled far into the right tail by Jensen's inequality --
# it comes out around 26 here, which would misleadingly read as a huge effect for a
# coefficient that is in fact indistinguishable from its normal(0, 2.5) prior.
odds_ratios <- posterior_or_long[, .(
  odds_ratio = median(odds_ratio),
  or_lower_95 = quantile(odds_ratio, 0.025),
  or_upper_95 = quantile(odds_ratio, 0.975)
), by = term]

fwrite(odds_ratios, "tables_figures/stage1_report/odds_ratios_summary.csv")
cat("\n--- Stage 1 odds-ratio table (all 14 predictors, 95% CrI) ---\n")
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
    title = "Stage 1 odds ratios by predictor",
    subtitle = paste0(
      "Posterior distribution, median & 95% CrI (dashed line: no effect, OR = 1)\n",
      "was_missing_stage_fear / was_missing_drained_after_socializing: near-null medians,\n",
      "essentially prior-width CrIs -- individually non-identifiable"
    )
  ) +
  theme_minimal(base_size = 12)
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
non_identified <- c("was_missing_stage_fear", "was_missing_drained_after_socializing")
non_identified_flagged <- missing_summary[term %in% non_identified]

cat("\n--- Confirmed MNAR effects (95% CrI excludes OR = 1) ---\n")
if (length(confirmed_mnar) > 0) {
  print(confirmed_mnar)
} else {
  cat("None -- no was_missing_* term's 95% CrI excludes OR = 1.\n")
}

cat(
  "\n--- Non-identifiability finding: was_missing_stage_fear /",
  "was_missing_drained_after_socializing ---\n"
)
cat(
  "These two missingness indicators are almost certainly collinear with each other",
  "(a respondent who skips the Stage_fear question is very likely to also skip",
  "Drained_after_socializing), mirroring the same individual-vs-sum non-identifiability",
  "pattern the design spec anticipated for stage_fear_bin/drained_after_socializing_bin",
  "themselves. Unlike that pair, however, these two missingness coefficients came back",
  "essentially unchanged from their normal(0, 2.5) prior -- the likelihood contributed",
  "almost no information about how to split the shared missingness signal between them",
  "individually:\n"
)
print(non_identified_flagged)
cat(
  "\nInterpretation: we cannot separately attribute an MNAR effect to stage_fear",
  "missingness vs. drained_after_socializing missingness from this data -- their",
  "combined/joint missingness pattern may still be informative, but Stage 1's flat",
  "specification (independent was_missing_* terms) cannot resolve it. A joint",
  "missingness indicator or an explicit sum (as done above for the bin terms) would",
  "be needed to test that jointly, and is left for a later stage / follow-up.\n"
)
