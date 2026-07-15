library(data.table)

recode_personality <- function(df) {
  dt <- as.data.table(df)
  dt[, `:=`(
    personality_introvert = as.integer(Personality == "Introvert"),
    stage_fear_bin = as.integer(Stage_fear == "Yes"),
    drained_after_socializing_bin = as.integer(Drained_after_socializing == "Yes")
  )]
  dt[]
}

add_missingness_indicators <- function(df, cols) {
  dt <- as.data.table(df)
  for (col in cols) {
    set(dt, j = paste0("was_missing_", col), value = as.integer(is.na(dt[[col]])))
  }
  dt[]
}

impute_from_stats <- function(df, numeric_cols, binary_cols, stats = NULL) {
  compute_mode <- function(x) {
    tab <- table(x)
    as.numeric(names(tab)[which.max(tab)])
  }

  dt <- as.data.table(df)

  if (is.null(stats)) {
    stats <- list()
    for (col in numeric_cols) {
      stats[[col]] <- list(median = median(dt[[col]], na.rm = TRUE))
    }
    for (col in binary_cols) {
      stats[[col]] <- list(mode = compute_mode(dt[[col]][!is.na(dt[[col]])]))
    }
  }

  for (col in numeric_cols) {
    col_data <- as.numeric(dt[[col]])
    col_data[is.na(col_data)] <- stats[[col]]$median
    set(dt, j = col, value = col_data)
  }
  for (col in binary_cols) {
    col_data <- as.numeric(dt[[col]])
    col_data[is.na(col_data)] <- stats[[col]]$mode
    set(dt, j = col, value = col_data)
  }

  list(data = dt[], stats = stats)
}

standardize_predictors <- function(df, predictor_cols, stats = NULL) {
  dt <- as.data.table(df)

  if (is.null(stats)) {
    stats <- setNames(
      lapply(predictor_cols, function(col) list(mean = mean(dt[[col]]), sd = sd(dt[[col]]))),
      predictor_cols
    )
  }

  for (col in predictor_cols) {
    set(dt, j = paste0(col, "_z"), value = (dt[[col]] - stats[[col]]$mean) / stats[[col]]$sd)
  }

  list(data = dt[], stats = stats)
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
