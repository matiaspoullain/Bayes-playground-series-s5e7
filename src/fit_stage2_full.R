source("src/features.R")
source("src/model_spec_stage2.R")
library(brms)
library(mice)
library(data.table)

predictor_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency", "stage_fear_bin",
  "drained_after_socializing_bin"
)
numeric_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency"
)

train_raw <- read_competition_csv("data/raw/train.csv")
prepared <- prepare_stage2_data(train_raw)$data

# Standardization stats computed once, from each numeric predictor's OBSERVED
# (non-missing) values only -- applied identically to all M imputed copies below.
standardize_stats <- lapply(numeric_cols, function(col) {
  list(mean = mean(prepared[[col]], na.rm = TRUE), sd = sd(prepared[[col]], na.rm = TRUE))
})
names(standardize_stats) <- numeric_cols

# mice's input EXCLUDES personality_introvert entirely (not just left unused) -- it
# must not be a column at all, so it cannot inform imputation of any predictor. This
# is what makes the same trained model directly usable on test.csv, which never has
# an outcome column at all: the imputation process training relied on and the
# imputation process a real prediction-time row would need are identical. An earlier
# design also tried including the outcome in the imputation model (legitimate MI
# practice for training data, where the outcome is always known) and used a second
# "Model A" for rows with no missing predictors; an AUC comparison on complete rows
# (src/compare_modelA_modelB_on_complete_rows.R) found only a ~0.003 AUC edge for
# that approach, too small to justify keeping two models/two imputation passes -- see
# design spec's "Missing-data strategy" section for the full account.
mice_input <- as.data.frame(prepared[, predictor_cols, with = FALSE])

m_value <- 20
cat("Running mice(m =", m_value, ", maxit = 5) WITHOUT personality_introvert as a predictor...\n")
imp <- mice(mice_input, m = m_value, maxit = 5, seed = 1, printFlag = TRUE)

cat("\nBuilding", m_value, "standardized completed datasets (reattaching the real outcome for fitting)...\n")
completed_list <- lapply(seq_len(m_value), function(i) {
  dt <- as.data.table(complete(imp, i))
  dt[, personality_introvert := prepared$personality_introvert]
  for (col in numeric_cols) {
    set(dt, j = paste0(col, "_z"), value = (dt[[col]] - standardize_stats[[col]]$mean) / standardize_stats[[col]]$sd)
  }
  dt[]
})

cat("\nFitting brm_multiple() across", m_value, "imputations...\n")
chains_per_imputation <- 4
fit <- brm_multiple(
  formula = stage2_formula(),
  data = completed_list,
  family = stage2_family(),
  prior = stage2_priors(),
  backend = "cmdstanr",
  chains = chains_per_imputation, cores = chains_per_imputation, iter = 2000, warmup = 1000,
  seed = 1
)

# Save immediately, before any post-processing that could error and lose the fit
# (this order bit us once already: a bug in per-imputation diagnostic code crashed
# the script after a completed fit that was never saved).
dir.create("models", showWarnings = FALSE)
saveRDS(fit, "models/stage2_fit_full.rds")
saveRDS(imp, "models/stage2_mice_imp.rds")
saveRDS(standardize_stats, "models/stage2_fit_full_standardize_stats.rds")
cat("\nFit and imputation object saved to models/.\n")

print(summary(fit))

library(posterior)
draws <- as_draws_array(fit)
n_chains_total <- posterior::nchains(draws)
stopifnot(n_chains_total == m_value * chains_per_imputation)

cat("\n--- Per-imputation convergence (max Rhat, min ESS per imputation) ---\n")
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
print(per_imputation_summary)
cat("\nWorst max Rhat across all imputations:", max(per_imputation_summary$max_rhat), "\n")
cat("Worst min Bulk ESS across all imputations:", min(per_imputation_summary$min_ess_bulk), "\n")

fwrite(per_imputation_summary, "models/stage2_fit_full_per_imputation_convergence.csv")
