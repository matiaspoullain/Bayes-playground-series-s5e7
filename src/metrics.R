accuracy <- function(actual, predicted_prob, threshold = 0.5) {
  predicted_class <- as.integer(predicted_prob >= threshold)
  mean(predicted_class == actual)
}

brier_score <- function(actual, predicted_prob) {
  mean((predicted_prob - actual)^2)
}
