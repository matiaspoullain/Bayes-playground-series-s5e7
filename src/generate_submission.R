source("src/features.R")
library(brms)
library(mice)
library(data.table)

# Generates the Kaggle submission file (id, Personality) for test.csv, using the
# final Stage 2 model (models/stage2_fit_full.rds). test.csv is imputed the same
# outcome-free way training was: one mice() pass over train + test predictors,
# with train rows fitting the imputation model and test rows only receiving
# imputed values (ignore = TRUE), exactly as fit_stage2_cv.R does for CV holdout
# folds. The saved standardize_stats (from fit_stage2_full.R, computed on full
# train) are reused as-is -- NOT recomputed here -- since the fitted model's
# coefficients are defined on that exact z-scale.

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
test_raw <- read_competition_csv("data/raw/test.csv")
prepared_train <- prepare_stage2_data(train_raw)$data
prepared_test <- prepare_stage2_data(test_raw)$data

fit <- readRDS("models/stage2_fit_full.rds")
standardize_stats <- readRDS("models/stage2_fit_full_standardize_stats.rds")
standardize_cols <- function(dt) {
  for (col in numeric_cols) {
    set(dt, j = paste0(col, "_z"), value = (dt[[col]] - standardize_stats[[col]]$mean) / standardize_stats[[col]]$sd)
  }
  dt[]
}

combined_predictors <- rbind(
  prepared_train[, predictor_cols, with = FALSE],
  prepared_test[, predictor_cols, with = FALSE]
)
ignore_vec <- c(rep(FALSE, nrow(prepared_train)), rep(TRUE, nrow(prepared_test)))

m_value <- 20
cat("Running mice(m =", m_value, ") over train+test predictors (train fits the imputation model, test is imputed only)...\n")
imp <- mice(as.data.frame(combined_predictors), m = m_value, maxit = 5, seed = 1, ignore = ignore_vec, printFlag = TRUE)

# posterior_epred() on the pooled fit (20 imputations x 4 chains x 1000
# post-warmup = 80000 draws) against 6175 test rows would need an ~5GB draws
# matrix per call -- more than this machine has free. ndraws subsamples the
# pooled posterior for prediction only (the fit itself is untouched); 2000
# draws is still ample for a stable posterior mean on a near-separable
# classification problem, and explicit gc() keeps peak memory from stacking
# across the 20 imputation loops.
ndraws_predict <- 2000
cat("\nPredicting test.csv, averaging posterior_epred() (ndraws =", ndraws_predict, ") across all", m_value, "imputations...\n")
test_pred_means <- lapply(seq_len(m_value), function(i) {
  test_i <- standardize_cols(as.data.table(complete(imp, i))[ignore_vec])
  draws_i <- posterior_epred(fit, newdata = test_i, allow_new_levels = TRUE, ndraws = ndraws_predict)
  result <- colMeans(draws_i)
  rm(draws_i)
  gc(verbose = FALSE)
  result
})
pred_prob <- Reduce(`+`, test_pred_means) / m_value

submission <- data.table(
  id = test_raw$id,
  Personality = ifelse(pred_prob >= 0.5, "Introvert", "Extrovert")
)

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
fwrite(submission, "data/processed/submission_stage2.csv")

cat("\nSaved data/processed/submission_stage2.csv\n")
cat("Predicted class balance:\n")
print(submission[, .N, by = Personality][, pct := 100 * N / sum(N)][])
cat(sprintf("\n(Training set balance: 73.95%% Extrovert / 26.05%% Introvert, for comparison)\n"))
