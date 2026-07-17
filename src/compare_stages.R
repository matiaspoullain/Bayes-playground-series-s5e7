source("src/metrics.R")
library(data.table)

# Stage 1 vs. Stage 2 comparison, per the design spec's "Comparison" section:
# loo_compare() is not directly applicable here -- Stage 1 is a single brmsfit,
# Stage 2 is an M-fit-pooled brmsfit_multiple object (20 mice imputations), so
# there is no single log-likelihood matrix to compare across draws. Instead the
# primary comparison is predictive: mean per-fold accuracy +/- SD (fold-level
# variability) alongside pooled out-of-fold accuracy/Brier (computed by
# concatenating every fold's held-out predictions, i.e. one prediction per row
# of the full training set).

dir.create("tables_figures/stage_comparison", showWarnings = FALSE, recursive = TRUE)

stage1_cv_summary <- fread("tables_figures/stage1_cv/cv_summary.csv")
stage1_oof <- fread("tables_figures/stage1_cv/oof_predictions.csv")
stage2_cv_summary <- fread("tables_figures/stage2_cv/cv_summary.csv")
stage2_oof <- fread("tables_figures/stage2_cv/oof_predictions.csv")

#' Summarize one row of the accuracy-comparison table
#'
#' @param stage_name label for this row (e.g. "stage1", "stage2_modelA_complete")
#' @param fold_accuracy numeric vector of per-fold accuracies for this slice
#'   (may contain NA for folds where this slice had zero rows -- mean()/sd()
#'   below use na.rm = TRUE to tolerate that)
#' @param oof_actual,oof_pred_prob concatenated out-of-fold actual/predicted
#'   vectors for this slice, used for the pooled accuracy/Brier figures
#' @return one-row data.table: stage, n_folds, n_rows, mean/sd fold accuracy,
#'   pooled OOF accuracy, pooled OOF Brier
summarize_stage <- function(stage_name, fold_accuracy, oof_actual, oof_pred_prob) {
  data.table(
    stage = stage_name,
    n_folds = sum(!is.na(fold_accuracy)),
    n_rows = length(oof_actual),
    mean_accuracy = mean(fold_accuracy, na.rm = TRUE),
    sd_accuracy = sd(fold_accuracy, na.rm = TRUE),
    pooled_accuracy = accuracy(oof_actual, oof_pred_prob),
    pooled_brier = brier_score(oof_actual, oof_pred_prob)
  )
}

# Stage 2 is now two models routed by predictor completeness (see design spec's
# "Two-model split"): Model A predicts holdout rows with no missing predictors
# (outcome-informed training imputation, no imputation at prediction time), Model
# B predicts holdout rows with missing predictors (outcome-free training
# imputation, matching outcome-free imputation at prediction time). Reported here
# as four rows: Stage 1 (unchanged), Stage 2 combined/pooled (matching the old
# single-row figure), and the Model A / Model B breakdown -- cv_summary.csv's
# accuracy_modelA/accuracy_modelB columns give the per-fold figures, and
# oof_predictions.csv's model_used column gives the pooled ones.
accuracy_comparison <- rbindlist(list(
  summarize_stage("stage1", stage1_cv_summary$accuracy, stage1_oof$actual, stage1_oof$oof_pred_prob),
  summarize_stage("stage2_combined", stage2_cv_summary$accuracy, stage2_oof$actual, stage2_oof$oof_pred_prob),
  summarize_stage(
    "stage2_modelA_complete", stage2_cv_summary$accuracy_modelA,
    stage2_oof[model_used == "A", actual], stage2_oof[model_used == "A", oof_pred_prob]
  ),
  summarize_stage(
    "stage2_modelB_incomplete", stage2_cv_summary$accuracy_modelB,
    stage2_oof[model_used == "B", actual], stage2_oof[model_used == "B", oof_pred_prob]
  )
))

fwrite(accuracy_comparison, "tables_figures/stage_comparison/accuracy_comparison.csv")

cat("\n--- Stage 1 vs. Stage 2 (combined + Model A/B breakdown) comparison ---\n")
print(accuracy_comparison)

for (row_stage in accuracy_comparison$stage) {
  row <- accuracy_comparison[stage == row_stage]
  cat(sprintf(
    "\n%s: %d folds, n=%d, mean accuracy %.4f (SD %.4f), pooled OOF accuracy %.4f, pooled OOF Brier %.4f\n",
    row_stage, row$n_folds, row$n_rows, row$mean_accuracy, row$sd_accuracy,
    row$pooled_accuracy, row$pooled_brier
  ))
}

# Model B (incomplete rows, imputed outcome-free) notably outperforms Model A
# (complete rows, no imputation needed) on pooled accuracy -- worth surfacing
# explicitly, not just leaving as a number in the table above.
acc_a <- accuracy_comparison[stage == "stage2_modelA_complete", pooled_accuracy]
acc_b <- accuracy_comparison[stage == "stage2_modelB_incomplete", pooled_accuracy]
cat(sprintf(
  "\nNote: Model B (incomplete-predictor rows, pooled accuracy %.4f) outperforms Model A (complete-predictor rows, pooled accuracy %.4f) by %.4f -- see design spec's \"Two-model split\" for why this is plausible (Model B's outcome-free imputation still recovers useful signal, and its holdout rows are not inherently harder to classify).\n",
  acc_b, acc_a, acc_b - acc_a
))
