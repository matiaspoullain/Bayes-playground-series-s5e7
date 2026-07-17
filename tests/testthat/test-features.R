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

test_that("recode_personality tolerates a missing Personality column (as in test.csv)", {
  df <- data.frame(
    Stage_fear = c("Yes", "No"),
    Drained_after_socializing = c("No", "Yes")
  )
  result <- recode_personality(df)

  expect_equal(result$personality_introvert, c(NA_integer_, NA_integer_))
  expect_equal(result$stage_fear_bin, c(1L, 0L))
  expect_equal(result$drained_after_socializing_bin, c(0L, 1L))
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

test_that("prepare_stage1_data produces snake_case modeling columns with NA propagated to was_missing_*", {
  df <- data.frame(
    Time_spent_Alone = c(1, NA, 3, 4),
    Social_event_attendance = c(2, 3, NA, 5),
    Going_outside = c(1, 2, 3, 4),
    Friends_circle_size = c(5, 6, 7, 8),
    Post_frequency = c(1, 1, 1, 1),
    Stage_fear = c("Yes", "No", NA, "No"),
    Drained_after_socializing = c("No", "Yes", "No", NA),
    Personality = c("Introvert", "Extrovert", "Introvert", "Extrovert")
  )
  out <- prepare_stage1_data(df)

  expected_cols <- c(
    "personality_introvert",
    "time_spent_alone_z", "social_event_attendance_z", "going_outside_z",
    "friends_circle_size_z", "post_frequency_z",
    "stage_fear_bin", "drained_after_socializing_bin",
    "was_missing_time_spent_alone", "was_missing_social_event_attendance",
    "was_missing_going_outside", "was_missing_friends_circle_size",
    "was_missing_post_frequency", "was_missing_stage_fear",
    "was_missing_drained_after_socializing"
  )
  expect_true(all(expected_cols %in% names(out$data)))

  expect_equal(out$data$was_missing_time_spent_alone, c(0L, 1L, 0L, 0L))
  expect_equal(out$data$was_missing_social_event_attendance, c(0L, 0L, 1L, 0L))
  expect_equal(out$data$was_missing_stage_fear, c(0L, 0L, 1L, 0L))
  expect_equal(out$data$was_missing_drained_after_socializing, c(0L, 0L, 0L, 1L))

  expect_false(anyNA(out$data$time_spent_alone_z))
  expect_false(anyNA(out$data$stage_fear_bin))

  expect_true(is.list(out$stats))
  expect_true(all(c("impute", "standardize") %in% names(out$stats)))
})

test_that("prepare_stage2_data recodes/renames but leaves NAs untouched (no impute, no standardize)", {
  df <- data.frame(
    Time_spent_Alone = c(1, NA, 3, 4),
    Social_event_attendance = c(2, 3, NA, 5),
    Going_outside = c(1, 2, 3, 4),
    Friends_circle_size = c(5, 6, 7, 8),
    Post_frequency = c(1, 1, 1, 1),
    Stage_fear = c("Yes", "No", NA, "No"),
    Drained_after_socializing = c("No", "Yes", "No", NA),
    Personality = c("Introvert", "Extrovert", "Introvert", "Extrovert")
  )
  out <- prepare_stage2_data(df)

  expected_cols <- c(
    "personality_introvert", "time_spent_alone", "social_event_attendance", "going_outside",
    "friends_circle_size", "post_frequency", "stage_fear_bin", "drained_after_socializing_bin"
  )
  expect_true(all(expected_cols %in% names(out$data)))

  # NAs preserved, not imputed
  expect_equal(out$data$time_spent_alone, c(1, NA, 3, 4))
  expect_equal(out$data$social_event_attendance, c(2, 3, NA, 5))
  expect_equal(out$data$stage_fear_bin, c(1L, 0L, NA, 0L))
  expect_equal(out$data$drained_after_socializing_bin, c(0L, 1L, 0L, NA))

  # not standardized: raw values, no *_z columns, no was_missing_* indicators
  expect_false(any(grepl("_z$", names(out$data))))
  expect_false(any(grepl("^was_missing_", names(out$data))))

  expect_equal(out$data$personality_introvert, c(1L, 0L, 1L, 0L))
})

test_that("read_competition_csv reads a blank field in a character column as NA", {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp))
  writeLines(c(
    "id,Stage_fear,Time_spent_Alone",
    "1,Yes,3",
    "2,,4"
  ), tmp)

  result <- read_competition_csv(tmp)

  expect_true(is.na(result$Stage_fear[2]))
  expect_equal(result$Stage_fear, c("Yes", NA))
})
