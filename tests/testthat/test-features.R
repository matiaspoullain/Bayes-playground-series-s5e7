source(file.path("..", "..", "src", "features.R"))

test_that("recode_personality creates 0/1 target and binary predictor columns, preserving NA", {
  df <- data.frame(
    Personality = c("Introvert", "Extrovert", NA),
    Stage_fear = c("Yes", "No", NA),
    Drained_after_socializing = c("No", "Yes", NA)
  )
  result <- recode_personality(df)

  expect_equal(result$personality_introvert, c(1L, 0L, NA))
  expect_equal(result$stage_fear_bin, c(1L, 0L, NA))
  expect_equal(result$drained_after_socializing_bin, c(0L, 1L, NA))
})

test_that("add_missingness_indicators flags NA per column", {
  df <- data.frame(x = c(1, NA, 3), y = c(NA, NA, "a"))
  result <- add_missingness_indicators(df, cols = c("x", "y"))

  expect_equal(result$was_missing_x, c(0L, 1L, 0L))
  expect_equal(result$was_missing_y, c(1L, 1L, 0L))
})

test_that("impute_from_stats fills numeric NAs with median and binary NAs with mode", {
  df <- data.frame(num = c(1, 2, NA, 4), bin = c(0, 0, 1, NA))
  out <- impute_from_stats(df, numeric_cols = "num", binary_cols = "bin")

  expect_equal(out$data$num, c(1, 2, 2, 4))   # median of c(1,2,4) = 2
  expect_equal(out$data$bin, c(0, 0, 1, 0))   # mode of c(0,0,1) = 0
  expect_equal(out$stats$num$median, 2)
  expect_equal(out$stats$bin$mode, 0)
})

test_that("impute_from_stats reuses supplied stats instead of recomputing", {
  train <- data.frame(num = c(1, 2, 3))
  train_out <- impute_from_stats(train, numeric_cols = "num", binary_cols = character(0))

  new_data <- data.frame(num = c(NA))
  new_out <- impute_from_stats(new_data, numeric_cols = "num", binary_cols = character(0),
                                stats = train_out$stats)
  expect_equal(new_out$data$num, 2)  # training median, not new_data's own (undefined) median
})

test_that("standardize_predictors z-scores columns and returns reusable stats", {
  df <- data.frame(x = c(1, 2, 3, 4, 5))
  out <- standardize_predictors(df, predictor_cols = "x")

  expect_equal(mean(out$data$x_z), 0, tolerance = 1e-8)
  expect_equal(sd(out$data$x_z), 1, tolerance = 1e-8)
  expect_equal(out$stats$x$mean, 3)
})

test_that("make_stratified_folds preserves class proportions per fold", {
  y <- rep(c(1, 1, 1, 0), 25)  # 75/25 split, n = 100
  folds <- make_stratified_folds(y, k = 5, seed = 1)

  expect_equal(sort(unique(folds)), 0:4)
  for (f in 0:4) {
    fold_rate <- mean(y[folds == f])
    expect_equal(fold_rate, 0.75, tolerance = 0.05)
  }
})

test_that("make_stratified_folds is deterministic given the same seed", {
  y <- rep(c(1, 0), 50)
  expect_equal(make_stratified_folds(y, k = 5, seed = 7), make_stratified_folds(y, k = 5, seed = 7))
})
