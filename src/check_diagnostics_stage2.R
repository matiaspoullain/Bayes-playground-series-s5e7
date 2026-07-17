library(brms)
library(bayesplot)
library(posterior)
library(ggplot2)
library(data.table)
library(mice)

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

# pp_check(fit, type = "bars") called with default arguments (newdata = NULL)
# does NOT pool across all M imputations: brms's own warn_brmsfit_multiple()
# only fires that warning when newdata is NULL, and posterior_predict() with
# newdata = NULL silently reuses imputation 1's completed data for every draw.
# Passing an explicit per-imputation `newdata` (as below) sidesteps that branch
# entirely, so this block manually pools genuine posterior_predict() draws from
# all 20 imputations instead of just the first. This mirrors
# src/fit_stage2_full.R's completed-dataset-rebuild (mice::complete() + the
# same standardization stats) and its chain-index arithmetic (per-imputation
# chain blocks), extended here to flat draw_ids ranges for posterior_predict().
imp <- readRDS("models/stage2_mice_imp.rds")
standardize_stats <- readRDS("models/stage2_fit_full_standardize_stats.rds")
numeric_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency"
)

completed_list <- lapply(seq_len(m_value), function(i) {
  dt <- as.data.table(complete(imp, i))
  for (col in numeric_cols) {
    set(dt, j = paste0(col, "_z"), value = (dt[[col]] - standardize_stats[[col]]$mean) / standardize_stats[[col]]$sd)
  }
  dt[]
})

# draws_per_imputation = post-warmup draws/chain x chains/imputation, verified
# against the real fit object (niterations(draws) = 1000, chains_per_imputation
# = 4) rather than assumed. as_draws_matrix()/posterior_predict()'s draw_ids
# index in chain-major order (all draws of chain 1, then chain 2, ...), so
# imputation i's chain block ((i-1)*chains_per_imputation+1):(i*chains_per_imputation)
# maps directly onto the flat draw_ids range below.
draws_per_chain <- posterior::niterations(draws)
draws_per_imputation <- draws_per_chain * chains_per_imputation
stopifnot(draws_per_imputation * m_value == posterior::ndraws(draws))

# Subsample within each imputation's draw block (matching the previous
# ndraws = 1000 default's overall scale) so the combined y_rep matrix
# (pooled_draws x N observations) stays a manageable size.
ndraws_per_imputation <- 50
set.seed(1)
y_rep_list <- lapply(seq_len(m_value), function(i) {
  block_start <- (i - 1) * draws_per_imputation + 1
  block_end <- i * draws_per_imputation
  draw_ids_i <- sample(block_start:block_end, ndraws_per_imputation)
  posterior_predict(fit, newdata = completed_list[[i]], draw_ids = draw_ids_i)
})
y_rep <- do.call(rbind, y_rep_list)

# personality_introvert has zero missing values in the raw data, so it is
# identical across all 20 completed datasets -- any one of them gives the
# true observed outcome vector.
y_obs <- completed_list[[1]]$personality_introvert

ppc_bars(y = y_obs, yrep = y_rep) +
  labs(
    title = "Stage 2 posterior predictive check",
    subtitle = paste0(
      "Simulated 0/1 outcome proportions (", ndraws_per_imputation, " draws x ", m_value,
      " imputations, pooled) vs. observed (y)"
    ),
    x = "Introvert (1) / Extrovert (0)", y = "Count"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage2_diagnostics/pp_check_bars.png", width = 7, height = 6, bg = "white")
