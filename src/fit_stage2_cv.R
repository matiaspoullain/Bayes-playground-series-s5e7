source("src/features.R")
source("src/model_spec_stage2.R")
source("src/metrics.R")
library(brms)
library(mice)
library(posterior)
library(data.table)

# Two-model CV: Model A is trained on outcome-informed imputation (legitimate for
# training data, where the outcome is always genuinely known) and predicts holdout
# rows with NO missing predictors directly (no imputation needed at prediction time,
# so no train/deployment mismatch). Model B is trained on outcome-free imputation
# (matching what's achievable on test.csv, which never has an outcome column) and
# predicts holdout rows that DO have missing predictors, imputed the same
# outcome-free way. See design spec's "Two-model split" (Missing-data strategy).

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

dir.create("models", showWarnings = FALSE)
dir.create("tables_figures/stage2_cv", showWarnings = FALSE, recursive = TRUE)

m_value <- 5
chains_per_imputation <- 4

fold_summaries <- list()
per_imputation_convergence <- list()
oof_predictions <- data.table(
  id = train_raw$id, actual = target_for_folds,
  oof_pred_prob = NA_real_, model_used = NA_character_
)

extract_per_imputation_conv <- function(fit, held_out_fold, model_label) {
  draws <- as_draws_array(fit)
  stopifnot(posterior::nchains(draws) == m_value * chains_per_imputation)
  rbindlist(lapply(seq_len(m_value), function(i) {
    chain_idx <- ((i - 1) * chains_per_imputation + 1):(i * chains_per_imputation)
    summ_i <- summarise_draws(subset_draws(draws, chain = chain_idx), default_convergence_measures())
    data.table(
      held_out_fold = held_out_fold, model = model_label, imputation = i,
      max_rhat = max(summ_i$rhat, na.rm = TRUE), min_ess_bulk = min(summ_i$ess_bulk, na.rm = TRUE)
    )
  }))
}

for (held_out_fold in sort(unique(folds))) {
  train_mask <- folds != held_out_fold
  holdout_mask <- folds == held_out_fold

  standardize_stats <- lapply(numeric_cols, function(col) {
    list(
      mean = mean(prepared[[col]][train_mask], na.rm = TRUE),
      sd = sd(prepared[[col]][train_mask], na.rm = TRUE)
    )
  })
  names(standardize_stats) <- numeric_cols
  standardize_cols <- function(dt) {
    for (col in numeric_cols) {
      set(dt, j = paste0(col, "_z"), value = (dt[[col]] - standardize_stats[[col]]$mean) / standardize_stats[[col]]$sd)
    }
    dt[]
  }

  ## --- Model A: outcome-informed imputation, trained on this fold's training rows ---
  cat(sprintf("\n=== Fold %d: Model A (outcome-informed imputation) ===\n", held_out_fold))
  mice_input_A <- as.data.frame(prepared[train_mask, c(predictor_cols, "personality_introvert"), with = FALSE])
  imp_A <- mice(mice_input_A, m = m_value, maxit = 5, seed = 1, printFlag = TRUE)
  completed_A_list <- lapply(seq_len(m_value), function(i) {
    standardize_cols(as.data.table(complete(imp_A, i)))
  })
  fit_A <- brm_multiple(
    formula = stage2_formula(), data = completed_A_list, family = stage2_family(),
    prior = stage2_priors(), backend = "cmdstanr",
    chains = chains_per_imputation, cores = chains_per_imputation, iter = 1500, warmup = 750, seed = 1
  )
  saveRDS(fit_A, sprintf("models/stage2_cv_fold_%d_modelA.rds", held_out_fold))
  cat(sprintf("Fold %d Model A saved.\n", held_out_fold))

  ## --- Model B: outcome-free imputation, trained on this fold's training rows ---
  cat(sprintf("\n=== Fold %d: Model B (outcome-free imputation) ===\n", held_out_fold))
  mice_input_B <- as.data.frame(prepared[train_mask, predictor_cols, with = FALSE])
  imp_B_train <- mice(mice_input_B, m = m_value, maxit = 5, seed = 1, printFlag = TRUE)
  train_outcome <- prepared$personality_introvert[train_mask]
  completed_B_list <- lapply(seq_len(m_value), function(i) {
    dt <- as.data.table(complete(imp_B_train, i))
    dt[, personality_introvert := train_outcome]
    standardize_cols(dt)
  })
  fit_B <- brm_multiple(
    formula = stage2_formula(), data = completed_B_list, family = stage2_family(),
    prior = stage2_priors(), backend = "cmdstanr",
    chains = chains_per_imputation, cores = chains_per_imputation, iter = 1500, warmup = 750, seed = 1
  )
  saveRDS(fit_B, sprintf("models/stage2_cv_fold_%d_modelB.rds", held_out_fold))
  cat(sprintf("Fold %d Model B saved.\n", held_out_fold))

  ## --- Route holdout rows by completeness ---
  holdout_indices <- which(holdout_mask)
  holdout_complete_mask <- complete.cases(prepared[holdout_indices, predictor_cols, with = FALSE])
  complete_indices <- holdout_indices[holdout_complete_mask]
  incomplete_indices <- holdout_indices[!holdout_complete_mask]

  if (length(complete_indices) > 0) {
    complete_holdout_dt <- standardize_cols(copy(prepared[complete_indices]))
    draws_complete <- posterior_epred(fit_A, newdata = complete_holdout_dt, allow_new_levels = TRUE)
    set(oof_predictions, i = complete_indices, j = "oof_pred_prob", value = colMeans(draws_complete))
    set(oof_predictions, i = complete_indices, j = "model_used", value = "A")
  }

  if (length(incomplete_indices) > 0) {
    combined_predictors <- rbind(
      prepared[train_mask, predictor_cols, with = FALSE],
      prepared[incomplete_indices, predictor_cols, with = FALSE]
    )
    ignore_vec_incomplete <- c(rep(FALSE, sum(train_mask)), rep(TRUE, length(incomplete_indices)))
    cat(sprintf(
      "\n=== Fold %d: outcome-free mice() for %d incomplete holdout rows ===\n",
      held_out_fold, length(incomplete_indices)
    ))
    imp_incomplete <- mice(
      as.data.frame(combined_predictors), m = m_value, maxit = 5, seed = 1,
      ignore = ignore_vec_incomplete, printFlag = TRUE
    )
    incomplete_pred_means <- lapply(seq_len(m_value), function(i) {
      holdout_i <- standardize_cols(as.data.table(complete(imp_incomplete, i))[ignore_vec_incomplete])
      draws_i <- posterior_epred(fit_B, newdata = holdout_i, allow_new_levels = TRUE)
      colMeans(draws_i)
    })
    pred_incomplete <- Reduce(`+`, incomplete_pred_means) / m_value
    set(oof_predictions, i = incomplete_indices, j = "oof_pred_prob", value = pred_incomplete)
    set(oof_predictions, i = incomplete_indices, j = "model_used", value = "B")
  }

  fold_conv <- rbind(
    extract_per_imputation_conv(fit_A, held_out_fold, "A"),
    extract_per_imputation_conv(fit_B, held_out_fold, "B")
  )
  per_imputation_convergence[[as.character(held_out_fold)]] <- fold_conv

  actual_holdout <- prepared$personality_introvert[holdout_mask]
  pred_holdout <- oof_predictions$oof_pred_prob[holdout_mask]
  key <- as.character(held_out_fold)
  fold_summaries[[key]] <- data.table(
    held_out_fold = held_out_fold,
    n_complete = length(complete_indices), n_incomplete = length(incomplete_indices),
    accuracy = accuracy(actual_holdout, pred_holdout),
    brier = brier_score(actual_holdout, pred_holdout),
    accuracy_modelA = if (length(complete_indices) > 0) {
      accuracy(prepared$personality_introvert[complete_indices], oof_predictions$oof_pred_prob[complete_indices])
    } else {
      NA_real_
    },
    accuracy_modelB = if (length(incomplete_indices) > 0) {
      accuracy(prepared$personality_introvert[incomplete_indices], oof_predictions$oof_pred_prob[incomplete_indices])
    } else {
      NA_real_
    },
    max_rhat = max(fold_conv$max_rhat), min_bulk_ess = min(fold_conv$min_ess_bulk)
  )

  cat(sprintf(
    "\nFold %d: overall accuracy=%.4f (n_complete=%d via A, n_incomplete=%d via B, acc_A=%.4f, acc_B=%.4f)\n",
    held_out_fold, fold_summaries[[key]]$accuracy, length(complete_indices), length(incomplete_indices),
    fold_summaries[[key]]$accuracy_modelA, fold_summaries[[key]]$accuracy_modelB
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
cat(
  "Pooled accuracy, complete rows (Model A):",
  accuracy(
    oof_predictions$actual[oof_predictions$model_used == "A"],
    oof_predictions$oof_pred_prob[oof_predictions$model_used == "A"]
  ), "\n"
)
cat(
  "Pooled accuracy, incomplete rows (Model B):",
  accuracy(
    oof_predictions$actual[oof_predictions$model_used == "B"],
    oof_predictions$oof_pred_prob[oof_predictions$model_used == "B"]
  ), "\n"
)
