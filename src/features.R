recode_personality <- function(df) {
  df$personality_introvert <- as.integer(df$Personality == "Introvert")
  df$stage_fear_bin <- as.integer(df$Stage_fear == "Yes")
  df$drained_after_socializing_bin <- as.integer(df$Drained_after_socializing == "Yes")
  df
}

add_missingness_indicators <- function(df, cols) {
  for (col in cols) {
    df[[paste0("was_missing_", col)]] <- as.integer(is.na(df[[col]]))
  }
  df
}

impute_from_stats <- function(df, numeric_cols, binary_cols, stats = NULL) {
  compute_mode <- function(x) {
    tab <- table(x)
    as.numeric(names(tab)[which.max(tab)])
  }

  if (is.null(stats)) {
    stats <- list()
    for (col in numeric_cols) {
      stats[[col]] <- list(median = median(df[[col]], na.rm = TRUE))
    }
    for (col in binary_cols) {
      stats[[col]] <- list(mode = compute_mode(df[[col]][!is.na(df[[col]])]))
    }
  }

  for (col in numeric_cols) {
    df[[col]][is.na(df[[col]])] <- stats[[col]]$median
  }
  for (col in binary_cols) {
    df[[col]][is.na(df[[col]])] <- stats[[col]]$mode
  }

  list(data = df, stats = stats)
}

standardize_predictors <- function(df, predictor_cols, stats = NULL) {
  if (is.null(stats)) {
    stats <- list()
    for (col in predictor_cols) {
      stats[[col]] <- list(mean = mean(df[[col]]), sd = sd(df[[col]]))
    }
  }

  for (col in predictor_cols) {
    z_col <- paste0(col, "_z")
    df[[z_col]] <- (df[[col]] - stats[[col]]$mean) / stats[[col]]$sd
  }

  list(data = df, stats = stats)
}

make_stratified_folds <- function(y, k, seed = 1) {
  set.seed(seed)
  folds <- integer(length(y))
  for (cls in unique(y)) {
    idx <- which(y == cls)
    idx <- sample(idx)
    folds[idx] <- rep(0:(k - 1), length.out = length(idx))
  }
  folds
}
