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

#' Prepare raw train/test data for the Stage 1 model
#'
#' Runs the full Stage 1 feature pipeline in the order that composes
#' correctly: recode the target/binary predictors while columns are still
#' PascalCase (`recode_personality()` expects raw `train.csv`/`test.csv`
#' naming) -> rename the raw PascalCase columns to the snake_case names the
#' Stage 1 model formula uses -> add missingness indicators -> impute ->
#' standardize. This is the one place the PascalCase -> snake_case rename
#' happens, so every later Stage 1 script should call this function rather
#' than re-deriving the rename.
#'
#' @param df Raw data.frame/data.table with PascalCase columns
#'   (`Time_spent_Alone`, `Stage_fear`, ...), as read from `train.csv` or
#'   `test.csv`. `test.csv` has no `Personality` column, which
#'   `recode_personality()` tolerates (the resulting `personality_introvert`
#'   column is simply all-`NA`).
#' @param stats Optional list with `impute` and `standardize` elements
#'   (as produced by a prior call's `$stats`), used to reuse training-fold
#'   statistics on held-out data instead of recomputing them. Defaults to
#'   `NULL`, which computes fresh stats from `df`.
#'
#' @return A list with `data` (the prepared data.table, including the
#'   snake_case predictor, `_z`-standardized, `_bin`, and `was_missing_*`
#'   columns `stage1_formula()` references) and `stats` (list with `impute`
#'   and `standardize` elements, reusable via the `stats` argument).
prepare_stage1_data <- function(df, stats = NULL) {
  df <- recode_personality(df)

  raw_to_snake <- c(
    Time_spent_Alone = "time_spent_alone", Social_event_attendance = "social_event_attendance",
    Going_outside = "going_outside", Friends_circle_size = "friends_circle_size",
    Post_frequency = "post_frequency", Stage_fear = "stage_fear",
    Drained_after_socializing = "drained_after_socializing"
  )
  dt <- as.data.table(df)
  setnames(dt, old = names(raw_to_snake), new = unname(raw_to_snake))

  numeric_cols <- c("time_spent_alone", "social_event_attendance", "going_outside",
                     "friends_circle_size", "post_frequency")
  dt <- add_missingness_indicators(dt, c(numeric_cols, "stage_fear", "drained_after_socializing"))

  imputed <- impute_from_stats(
    dt, numeric_cols = numeric_cols,
    binary_cols = c("stage_fear_bin", "drained_after_socializing_bin"),
    stats = if (is.null(stats)) NULL else stats$impute
  )
  std <- standardize_predictors(
    imputed$data, predictor_cols = numeric_cols,
    stats = if (is.null(stats)) NULL else stats$standardize
  )

  list(data = std$data, stats = list(impute = imputed$stats, standardize = std$stats))
}
