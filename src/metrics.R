#' Classification accuracy at a probability threshold
#'
#' @param actual numeric/integer vector of 0/1 true labels
#' @param predicted_prob numeric vector of predicted probabilities, same length as `actual`
#' @param threshold probability at/above which a prediction counts as class 1 (default 0.5,
#'   the Bayes-optimal threshold for plain accuracy regardless of class balance)
#' @return proportion of correctly classified rows
accuracy <- function(actual, predicted_prob, threshold = 0.5) {
  predicted_class <- as.integer(predicted_prob >= threshold)
  mean(predicted_class == actual)
}

#' Brier score (mean squared error between predicted probability and 0/1 outcome)
#'
#' @param actual numeric/integer vector of 0/1 true labels
#' @param predicted_prob numeric vector of predicted probabilities, same length as `actual`
#' @return mean squared error, lower is better, 0 is perfect
brier_score <- function(actual, predicted_prob) {
  mean((predicted_prob - actual)^2)
}

#' Area under the ROC curve, via the Mann-Whitney-U statistic
#'
#' Rank-based, threshold-free: the probability a randomly chosen positive case is
#' ranked above a randomly chosen negative case. No external dependency (no pROC/yardstick).
#'
#' @param actual numeric/integer vector of 0/1 true labels
#' @param predicted_prob numeric vector of predicted probabilities, same length as `actual`
#' @return AUC in \[0, 1\]; 0.5 is random, 1.0 is perfect separation
auc <- function(actual, predicted_prob) {
  pos <- predicted_prob[actual == 1]
  neg <- predicted_prob[actual == 0]

  ranks <- rank(c(pos, neg))
  n_pos <- length(pos)
  n_neg <- length(neg)

  sum_pos_ranks <- sum(ranks[1:n_pos])
  u_statistic <- sum_pos_ranks - n_pos * (n_pos + 1) / 2

  u_statistic / (n_pos * n_neg)
}
