source("src/features.R")
source("src/model_spec_stage1.R")
source("src/metrics.R")
library(brms)

train_raw <- read.csv("data/raw/train.csv")
target_for_folds <- as.integer(train_raw$Personality == "Introvert")
folds <- make_stratified_folds(target_for_folds, k = 10, seed = 1)

dir.create("models", showWarnings = FALSE)
dir.create("tables_figures/stage1_cv", showWarnings = FALSE, recursive = TRUE)

fold_summaries <- list()
oof_predictions <- data.frame(
  id = train_raw$id, actual = target_for_folds, oof_pred_prob = NA_real_
)

for (held_out_fold in sort(unique(folds))) {
  train_fold_raw <- train_raw[folds != held_out_fold, ]
  holdout_fold_raw <- train_raw[folds == held_out_fold, ]

  prepared_train <- prepare_stage1_data(train_fold_raw)
  prepared_holdout <- prepare_stage1_data(holdout_fold_raw, stats = prepared_train$stats)

  fit <- brm(
    formula = stage1_formula(),
    data = prepared_train$data,
    family = stage1_family(),
    prior = stage1_priors(),
    backend = "cmdstanr",
    chains = 4, cores = 4, iter = 2000, warmup = 1000,
    seed = 1
  )

  saveRDS(fit, sprintf("models/stage1_cv_fold_%d.rds", held_out_fold))

  pred_draws <- posterior_epred(fit, newdata = prepared_holdout$data, allow_new_levels = TRUE)
  pred_prob <- colMeans(pred_draws)
  oof_predictions$oof_pred_prob[folds == held_out_fold] <- pred_prob

  summ <- summary(fit)
  rhats <- summ$fixed[, "Rhat"]
  ess_bulk <- summ$fixed[, "Bulk_ESS"]
  divergences <- sum(subset(nuts_params(fit), Parameter == "divergent__")$Value)

  key <- as.character(held_out_fold)
  fold_summaries[[key]] <- data.frame(
    held_out_fold = held_out_fold,
    accuracy = accuracy(prepared_holdout$data$personality_introvert, pred_prob),
    brier = brier_score(prepared_holdout$data$personality_introvert, pred_prob),
    max_rhat = max(rhats),
    min_bulk_ess = min(ess_bulk),
    divergences = divergences
  )

  cat(sprintf(
    "Fold %d held out: accuracy=%.4f brier=%.4f max_Rhat=%.4f min_ESS=%.0f divergences=%d\n",
    held_out_fold, fold_summaries[[key]]$accuracy, fold_summaries[[key]]$brier,
    fold_summaries[[key]]$max_rhat, fold_summaries[[key]]$min_bulk_ess, divergences
  ))
}

cv_summary <- do.call(rbind, fold_summaries)
write.csv(cv_summary, "tables_figures/stage1_cv/cv_summary.csv", row.names = FALSE)
write.csv(oof_predictions, "tables_figures/stage1_cv/oof_predictions.csv", row.names = FALSE)

cat("\nMean per-fold accuracy:", mean(cv_summary$accuracy), "SD:", sd(cv_summary$accuracy), "\n")
cat("Pooled out-of-fold accuracy:", accuracy(oof_predictions$actual, oof_predictions$oof_pred_prob), "\n")
