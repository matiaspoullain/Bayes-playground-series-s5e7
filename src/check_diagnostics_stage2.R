library(brms)
library(bayesplot)
library(posterior)
library(ggplot2)
library(data.table)

fit <- readRDS("models/stage2_fit_full.rds")
dir.create("tables_figures/stage2_diagnostics", showWarnings = FALSE, recursive = TRUE)

# fit is a brmsfit_multiple object (from brm_multiple()). Per ?brm_multiple's own
# documentation, its combined Rhat/ESS (as reported by summary()) are "not
# informative" -- inflated by genuine between-imputation variability, not
# non-convergence. The documented way to check convergence is to subset the
# combined draws back into per-imputation chain blocks and run
# posterior::summarise_draws() on each block separately. This mirrors
# src/fit_stage2_full.R's own diagnostic block exactly (same chain-index
# arithmetic, same m_value/chains_per_imputation).
m_value <- 20
chains_per_imputation <- 4

draws <- as_draws_array(fit)
n_chains_total <- posterior::nchains(draws)
stopifnot(n_chains_total == m_value * chains_per_imputation)

per_imputation_summary <- rbindlist(lapply(seq_len(m_value), function(i) {
  chain_idx <- ((i - 1) * chains_per_imputation + 1):(i * chains_per_imputation)
  draws_i <- subset_draws(draws, chain = chain_idx)
  summ_i <- summarise_draws(draws_i, default_convergence_measures())
  data.table(
    imputation = i,
    max_rhat = max(summ_i$rhat, na.rm = TRUE),
    min_ess_bulk = min(summ_i$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(summ_i$ess_tail, na.rm = TRUE)
  )
}))

fwrite(per_imputation_summary, "tables_figures/stage2_diagnostics/rhat_ess_summary.csv")
print(per_imputation_summary)

cat("\nWorst max Rhat across all imputations:", max(per_imputation_summary$max_rhat), "(target: < 1.01)\n")
cat("Worst min Bulk ESS across all imputations:", min(per_imputation_summary$min_ess_bulk), "(target: > 400)\n")

# pp_check() works directly on the brmsfit_multiple object (posterior_predict()
# runs without error), but it does NOT pool across all M imputations: brms
# emits "Using only the first imputed data set. Please interpret the results
# with caution until a more principled approach has been implemented." So this
# check reflects imputation 1's completed dataset only, not the full ensemble.
pp_check(fit, type = "bars", ndraws = 1000) +
  labs(
    title = "Stage 2 posterior predictive check",
    subtitle = "Simulated 0/1 outcome proportions across 1000 posterior draws vs. observed (y)\n(brms uses only the first of 20 imputed datasets for posterior_predict())",
    x = "Introvert (1) / Extrovert (0)", y = "Count"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage2_diagnostics/pp_check_bars.png", width = 7, height = 6, bg = "white")
