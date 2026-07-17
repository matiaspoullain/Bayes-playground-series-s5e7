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

summarize_stage <- function(stage_name, cv_summary, oof) {
  data.table(
    stage = stage_name,
    n_folds = nrow(cv_summary),
    mean_accuracy = mean(cv_summary$accuracy),
    sd_accuracy = sd(cv_summary$accuracy),
    pooled_accuracy = accuracy(oof$actual, oof$oof_pred_prob),
    pooled_brier = brier_score(oof$actual, oof$oof_pred_prob)
  )
}

accuracy_comparison <- rbindlist(list(
  summarize_stage("stage1", stage1_cv_summary, stage1_oof),
  summarize_stage("stage2", stage2_cv_summary, stage2_oof)
))

fwrite(accuracy_comparison, "tables_figures/stage_comparison/accuracy_comparison.csv")

cat("\n--- Stage 1 vs. Stage 2 comparison ---\n")
print(accuracy_comparison)

cat(sprintf(
  "\nStage 1: %d folds, mean accuracy %.4f (SD %.4f), pooled OOF accuracy %.4f, pooled OOF Brier %.4f\n",
  accuracy_comparison[stage == "stage1", n_folds],
  accuracy_comparison[stage == "stage1", mean_accuracy],
  accuracy_comparison[stage == "stage1", sd_accuracy],
  accuracy_comparison[stage == "stage1", pooled_accuracy],
  accuracy_comparison[stage == "stage1", pooled_brier]
))
cat(sprintf(
  "Stage 2: %d folds, mean accuracy %.4f (SD %.4f), pooled OOF accuracy %.4f, pooled OOF Brier %.4f\n",
  accuracy_comparison[stage == "stage2", n_folds],
  accuracy_comparison[stage == "stage2", mean_accuracy],
  accuracy_comparison[stage == "stage2", sd_accuracy],
  accuracy_comparison[stage == "stage2", pooled_accuracy],
  accuracy_comparison[stage == "stage2", pooled_brier]
))
