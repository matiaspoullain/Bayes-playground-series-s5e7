source("src/metrics.R")
library(data.table)

# Stage 1 vs. Stage 2 comparison, per the design spec's "Comparison" section:
# loo_compare() is not directly applicable here -- Stage 1 is a single brmsfit,
# Stage 2 is an M-fit-pooled brmsfit_multiple object (20 mice imputations), so
# there is no single log-likelihood matrix to compare across draws. Instead the
# primary comparison is predictive: mean per-fold accuracy/AUC +/- SD (fold-level
# variability) alongside pooled out-of-fold accuracy/AUC/Brier (computed by
# concatenating every fold's held-out predictions, i.e. one prediction per row
# of the full training set). AUC is the headline metric here (not accuracy alone)
# because both stages sit in a near-ceiling ~0.96-0.97 accuracy band where
# threshold-based accuracy is a coarse way to distinguish them; AUC uses the full
# ranking of predicted probabilities instead.

dir.create("tables_figures/stage_comparison", showWarnings = FALSE, recursive = TRUE)

stage1_cv_summary <- fread("tables_figures/stage1_cv/cv_summary.csv")
stage1_oof <- fread("tables_figures/stage1_cv/oof_predictions.csv")
stage2_cv_summary <- fread("tables_figures/stage2_cv/cv_summary.csv")
stage2_oof <- fread("tables_figures/stage2_cv/oof_predictions.csv")

#' Summarize one row of the accuracy/AUC-comparison table
#'
#' @param stage_name label for this row (e.g. "stage1", "stage2")
#' @param fold_accuracy,fold_auc numeric vectors of per-fold accuracy/AUC for
#'   this stage
#' @param oof_actual,oof_pred_prob concatenated out-of-fold actual/predicted
#'   vectors, used for the pooled accuracy/AUC/Brier figures
#' @return one-row data.table: stage, n_folds, n_rows, mean/sd fold
#'   accuracy/AUC, pooled OOF accuracy/AUC/Brier
summarize_stage <- function(stage_name, fold_accuracy, fold_auc, oof_actual, oof_pred_prob) {
  data.table(
    stage = stage_name,
    n_folds = length(fold_accuracy),
    n_rows = length(oof_actual),
    mean_accuracy = mean(fold_accuracy), sd_accuracy = sd(fold_accuracy),
    mean_auc = mean(fold_auc), sd_auc = sd(fold_auc),
    pooled_accuracy = accuracy(oof_actual, oof_pred_prob),
    pooled_auc = auc(oof_actual, oof_pred_prob),
    pooled_brier = brier_score(oof_actual, oof_pred_prob)
  )
}

# Stage 1's CV script predates the auc() metric, so per-fold AUC isn't in its
# committed cv_summary.csv -- reconstructed here from the deterministic fold
# assignment (same seed/k as fit_stage1_cv.R) applied to the already-saved
# oof_predictions.csv, no refit needed.
source("src/features.R")
stage1_folds <- make_stratified_folds(stage1_oof$actual, k = 10, seed = 1)
stage1_fold_auc <- sapply(sort(unique(stage1_folds)), function(f) {
  auc(stage1_oof$actual[stage1_folds == f], stage1_oof$oof_pred_prob[stage1_folds == f])
})

accuracy_comparison <- rbindlist(list(
  summarize_stage("stage1", stage1_cv_summary$accuracy, stage1_fold_auc, stage1_oof$actual, stage1_oof$oof_pred_prob),
  summarize_stage("stage2", stage2_cv_summary$accuracy, stage2_cv_summary$auc, stage2_oof$actual, stage2_oof$oof_pred_prob)
))

fwrite(accuracy_comparison, "tables_figures/stage_comparison/accuracy_comparison.csv")

cat("\n--- Stage 1 vs. Stage 2 comparison ---\n")
print(accuracy_comparison)

for (row_stage in accuracy_comparison$stage) {
  row <- accuracy_comparison[stage == row_stage]
  cat(sprintf(
    "\n%s: %d folds, n=%d, mean accuracy %.4f (SD %.4f), mean AUC %.4f (SD %.4f), pooled OOF accuracy %.4f, pooled OOF AUC %.4f, pooled OOF Brier %.4f\n",
    row_stage, row$n_folds, row$n_rows, row$mean_accuracy, row$sd_accuracy,
    row$mean_auc, row$sd_auc, row$pooled_accuracy, row$pooled_auc, row$pooled_brier
  ))
}
