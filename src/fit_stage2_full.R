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
# (non-missing) values only -- applied identically to all M imputed copies below,
# so the standardized scale (and thus the pooled coefficients) is consistent
# across imputations, not re-derived per-copy.
standardize_stats <- lapply(numeric_cols, function(col) {
  list(mean = mean(prepared[[col]], na.rm = TRUE), sd = sd(prepared[[col]], na.rm = TRUE))
})
names(standardize_stats) <- numeric_cols

mice_input <- as.data.frame(prepared[, c(predictor_cols, "personality_introvert"), with = FALSE])

m_value <- 20
cat("Running mice(m =", m_value, ", maxit = 5)...\n")
imp <- mice(mice_input, m = m_value, maxit = 5, seed = 1, printFlag = TRUE)

cat("\nBuilding", m_value, "standardized completed datasets...\n")
completed_list <- lapply(seq_len(m_value), function(i) {
  dt <- as.data.table(complete(imp, i))
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
# (this order bit us once already: a bug in per-imputation diagnostic code below
# crashed the script and the completed fit, never having been saved, was lost).
dir.create("models", showWarnings = FALSE)
saveRDS(fit, "models/stage2_fit_full.rds")
saveRDS(imp, "models/stage2_mice_imp.rds")
saveRDS(standardize_stats, "models/stage2_fit_full_standardize_stats.rds")
cat("\nFit and imputation object saved to models/.\n")

print(summary(fit))

# Per-imputation convergence check, per ?brm_multiple's own documented approach
# (there is no fit$rhats field - brm_multiple's overall Rhat/ESS are explicitly
# "not informative" per its own summary() output; the documented way to check
# convergence is to subset the combined draws back into per-imputation chain
# blocks and run posterior::summarise_draws() on each block separately).
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
