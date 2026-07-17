source("src/features.R")
source("src/metrics.R")
library(brms)
library(data.table)

# Question: does Model B (outcome-free imputation) do at least as well as Model A
# (outcome-informed imputation) even on the COMPLETE rows Model A was designed for?
# If so, the two-model split adds complexity without adding predictive value, and
# Model B alone should replace both. Reuses the per-fold fits already saved by
# src/fit_stage2_cv.R - no refitting.

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

fold_results <- list()
pooled <- data.table(id = integer(0), actual = integer(0), pred_A = numeric(0), pred_B = numeric(0))

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

  holdout_indices <- which(holdout_mask)
  holdout_complete_mask <- complete.cases(prepared[holdout_indices, predictor_cols, with = FALSE])
  complete_indices <- holdout_indices[holdout_complete_mask]

  if (length(complete_indices) == 0) next

  fit_A <- readRDS(sprintf("models/stage2_cv_fold_%d_modelA.rds", held_out_fold))
  fit_B <- readRDS(sprintf("models/stage2_cv_fold_%d_modelB.rds", held_out_fold))

  complete_holdout_dt <- standardize_cols(copy(prepared[complete_indices]))
  pred_A <- colMeans(posterior_epred(fit_A, newdata = complete_holdout_dt, allow_new_levels = TRUE))
  pred_B <- colMeans(posterior_epred(fit_B, newdata = complete_holdout_dt, allow_new_levels = TRUE))

  actual <- prepared$personality_introvert[complete_indices]

  fold_results[[as.character(held_out_fold)]] <- data.table(
    held_out_fold = held_out_fold,
    n_complete = length(complete_indices),
    auc_A = auc(actual, pred_A),
    auc_B = auc(actual, pred_B)
  )

  pooled <- rbind(pooled, data.table(id = train_raw$id[complete_indices], actual = actual, pred_A = pred_A, pred_B = pred_B))

  cat(sprintf(
    "Fold %d (n_complete=%d): AUC Model A=%.4f  AUC Model B=%.4f\n",
    held_out_fold, length(complete_indices), auc(actual, pred_A), auc(actual, pred_B)
  ))
}

fold_summary <- rbindlist(fold_results)
dir.create("tables_figures/stage2_report", showWarnings = FALSE, recursive = TRUE)
fwrite(fold_summary, "tables_figures/stage2_report/modelA_vs_modelB_on_complete_rows.csv")

cat("\n--- Per-fold summary ---\n")
print(fold_summary)
cat("\nMean AUC Model A (complete rows):", mean(fold_summary$auc_A), "\n")
cat("Mean AUC Model B (complete rows):", mean(fold_summary$auc_B), "\n")
cat("\nPooled AUC Model A (complete rows):", auc(pooled$actual, pooled$pred_A), "\n")
cat("Pooled AUC Model B (complete rows):", auc(pooled$actual, pooled$pred_B), "\n")
