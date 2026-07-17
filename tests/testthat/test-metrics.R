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

test_that("auc is 1.0 for perfect separation", {
  actual <- c(0, 0, 1, 1)
  predicted <- c(0.1, 0.2, 0.8, 0.9)
  expect_equal(auc(actual, predicted), 1.0)
})

test_that("auc is 0.5 for random/no-separation predictions", {
  actual <- c(0, 1, 0, 1)
  predicted <- c(0.5, 0.5, 0.5, 0.5)
  expect_equal(auc(actual, predicted), 0.5)
})

test_that("auc matches a hand-computed value for a small mixed case", {
  # positives: 0.9, 0.4 | negatives: 0.3, 0.6
  # pairs (pos,neg): (0.9,0.3)=win (0.9,0.6)=win (0.4,0.3)=win (0.4,0.6)=loss -> 3/4 = 0.75
  actual <- c(1, 0, 0, 1)
  predicted <- c(0.9, 0.3, 0.6, 0.4)
  expect_equal(auc(actual, predicted), 0.75)
})
