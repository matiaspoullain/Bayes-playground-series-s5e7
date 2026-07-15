source(file.path("..", "..", "src", "metrics.R"))

test_that("accuracy is 1.0 for perfect predictions at the default threshold", {
  expect_equal(accuracy(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9)), 1.0)
})

test_that("accuracy correctly thresholds at 0.5", {
  expect_equal(accuracy(c(0, 1), c(0.49, 0.51)), 1.0)
  expect_equal(accuracy(c(0, 1), c(0.51, 0.49)), 0.0)
})

test_that("accuracy respects a custom threshold (>= threshold counts as class 1)", {
  expect_equal(accuracy(c(1, 1), c(0.3, 0.3), threshold = 0.3), 1.0)
})

test_that("brier_score is zero for perfect predictions", {
  expect_equal(brier_score(c(1, 0, 1), c(1, 0, 1)), 0)
})

test_that("brier_score computes mean squared error", {
  actual <- c(1, 0)
  predicted <- c(0.7, 0.3)
  expect_equal(brier_score(actual, predicted), mean(c((0.7 - 1)^2, (0.3 - 0)^2)))
})
