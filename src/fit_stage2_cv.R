source("src/features.R")
source("src/model_spec_stage2.R")
source("src/metrics.R")
library(brms)
library(mice)
library(posterior)
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
target_for_folds <- prepared$personality_introvert
folds <- make_stratified_folds(target_for_folds, k = 5, seed = 1)

mice_input <- as.data.frame(prepared[, c(predictor_cols, "personality_introvert"), with = FALSE])

dir.create("models", showWarnings = FALSE)
dir.create("tables_figures/stage2_cv", showWarnings = FALSE, recursive = TRUE)

m_value <- 5
chains_per_imputation <- 4

fold_summaries <- list()
per_imputation_convergence <- list()
oof_predictions <- data.table(id = train_raw$id, actual = target_for_folds, oof_pred_prob = NA_real_)

for (held_out_fold in sort(unique(folds))) {
  ignore_vec <- folds == held_out_fold

  cat(sprintf("\n=== Fold %d: mice(m = %d) with ignore = held-out rows ===\n", held_out_fold, m_value))
  imp <- mice(mice_input, m = m_value, maxit = 5, seed = 1, ignore = ignore_vec, printFlag = TRUE)

  standardize_stats <- lapply(numeric_cols, function(col) {
    list(
      mean = mean(prepared[[col]][!ignore_vec], na.rm = TRUE),
      sd = sd(prepared[[col]][!ignore_vec], na.rm = TRUE)
    )
  })
  names(standardize_stats) <- numeric_cols

  standardize_cols <- function(dt) {
    for (col in numeric_cols) {
      set(dt, j = paste0(col, "_z"), value = (dt[[col]] - standardize_stats[[col]]$mean) / standardize_stats[[col]]$sd)
    }
    dt[]
  }

  completed_train_list <- lapply(seq_len(m_value), function(i) {
    standardize_cols(as.data.table(complete(imp, i))[!ignore_vec])
  })

  fit <- brm_multiple(
    formula = stage2_formula(),
    data = completed_train_list,
    family = stage2_family(),
    prior = stage2_priors(),
    backend = "cmdstanr",
    chains = chains_per_imputation, cores = chains_per_imputation, iter = 1500, warmup = 750,
    seed = 1
  )

  saveRDS(fit, sprintf("models/stage2_cv_fold_%d.rds", held_out_fold))
  cat(sprintf("Fold %d fit saved.\n", held_out_fold))

  # Predict on each of the M holdout completions (same mice() call, ignore=-produced
  # imputed values for the held-out rows), average across the M sets of predictions --
  # this propagates both training-side (pooled fit) and holdout-side (varying
  # imputations) uncertainty.
  holdout_pred_means <- lapply(seq_len(m_value), function(i) {
    holdout_i <- standardize_cols(as.data.table(complete(imp, i))[ignore_vec])
    draws_i <- posterior_epred(fit, newdata = holdout_i, allow_new_levels = TRUE)
    colMeans(draws_i)
  })
  pred_prob <- Reduce(`+`, holdout_pred_means) / m_value
  set(oof_predictions, i = which(ignore_vec), j = "oof_pred_prob", value = pred_prob)

  # Per-imputation convergence (same documented brm_multiple approach as the final fit)
  draws <- as_draws_array(fit)
  stopifnot(posterior::nchains(draws) == m_value * chains_per_imputation)
  fold_conv <- rbindlist(lapply(seq_len(m_value), function(i) {
    chain_idx <- ((i - 1) * chains_per_imputation + 1):(i * chains_per_imputation)
    summ_i <- summarise_draws(subset_draws(draws, chain = chain_idx), default_convergence_measures())
    data.table(
      held_out_fold = held_out_fold, imputation = i,
      max_rhat = max(summ_i$rhat, na.rm = TRUE), min_ess_bulk = min(summ_i$ess_bulk, na.rm = TRUE)
    )
  }))
  per_imputation_convergence[[as.character(held_out_fold)]] <- fold_conv

  actual_holdout <- prepared$personality_introvert[ignore_vec]
  key <- as.character(held_out_fold)
  fold_summaries[[key]] <- data.table(
    held_out_fold = held_out_fold,
    accuracy = accuracy(actual_holdout, pred_prob),
    brier = brier_score(actual_holdout, pred_prob),
    max_rhat = max(fold_conv$max_rhat),
    min_bulk_ess = min(fold_conv$min_ess_bulk)
  )

  cat(sprintf(
    "Fold %d held out: accuracy=%.4f brier=%.4f max_Rhat=%.4f min_ESS=%.0f\n",
    held_out_fold, fold_summaries[[key]]$accuracy, fold_summaries[[key]]$brier,
    fold_summaries[[key]]$max_rhat, fold_summaries[[key]]$min_bulk_ess
  ))

  # Save incrementally after each fold, so a crash on a later fold doesn't lose
  # earlier folds' results (lesson from Task 12's save-ordering incident).
  fwrite(rbindlist(fold_summaries), "tables_figures/stage2_cv/cv_summary.csv")
  fwrite(oof_predictions, "tables_figures/stage2_cv/oof_predictions.csv")
  fwrite(rbindlist(per_imputation_convergence), "tables_figures/stage2_cv/per_imputation_convergence.csv")
}

cv_summary <- rbindlist(fold_summaries)
cat("\nMean per-fold accuracy:", mean(cv_summary$accuracy), "SD:", sd(cv_summary$accuracy), "\n")
cat("Pooled out-of-fold accuracy:", accuracy(oof_predictions$actual, oof_predictions$oof_pred_prob), "\n")
