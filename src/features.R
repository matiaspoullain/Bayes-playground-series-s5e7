library(data.table)

recode_personality <- function(df) {
  dt <- as.data.table(df)
  if ("Personality" %in% names(dt)) {
    dt[, `:=`(
      personality_introvert = as.integer(Personality == "Introvert"),
      stage_fear_bin = as.integer(Stage_fear == "Yes"),
      drained_after_socializing_bin = as.integer(Drained_after_socializing == "Yes")
    )]
  } else {
    dt[, `:=`(
      personality_introvert = NA_integer_,
      stage_fear_bin = as.integer(Stage_fear == "Yes"),
      drained_after_socializing_bin = as.integer(Drained_after_socializing == "Yes")
    )]
  }
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
#'   `test.csv`. `recode_personality()` tolerates a missing `Personality`
#'   column (as in `test.csv`) — `personality_introvert` comes back all-`NA`
#'   in that case, rather than erroring.
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

#' Prepare raw train/test data for the Stage 2 (mice + brm_multiple) model
#'
#' Runs only the recode + rename steps of the Stage 1 pipeline
#' (`recode_personality()`, then the same PascalCase -> snake_case rename
#' `prepare_stage1_data()` uses) and stops there: no missingness
#' indicators, no imputation, no standardization. `NA`s in the 7
#' partially-missing predictor columns are left exactly as they appear in
#' the raw data, because Stage 2 imputes them downstream via `mice()` (see
#' `src/fit_stage2_full.R`/`src/fit_stage2_cv.R`) rather than having them
#' point-imputed away here. (An earlier design used `brms`'s own `mi()`
#' multivariate sub-models for this instead of external `mice` imputation;
#' that approach was abandoned — see the design spec's "Missing-data
#' strategy" section — but this function's recode/rename/leave-NAs-intact
#' contract is unchanged by that pivot, so it's still the right prep step
#' for Stage 2's current `mice`-based pipeline.)
#'
#' @param df Raw data.frame/data.table with PascalCase columns
#'   (`Time_spent_Alone`, `Stage_fear`, ...), as read from `train.csv` or
#'   `test.csv`. `recode_personality()` tolerates a missing `Personality`
#'   column (as in `test.csv`) — `personality_introvert` comes back all-`NA`
#'   in that case, rather than erroring.
#'
#' @return A list with a single `data` element: the prepared data.table,
#'   including `personality_introvert`, the 5 snake_case count-like
#'   predictors (`time_spent_alone`, `social_event_attendance`,
#'   `going_outside`, `friends_circle_size`, `post_frequency`), and the 2
#'   binary predictors (`stage_fear_bin`, `drained_after_socializing_bin`),
#'   with `NA`s intact wherever the raw data had them.
prepare_stage2_data <- function(df) {
  df <- recode_personality(df)

  raw_to_snake <- c(
    Time_spent_Alone = "time_spent_alone", Social_event_attendance = "social_event_attendance",
    Going_outside = "going_outside", Friends_circle_size = "friends_circle_size",
    Post_frequency = "post_frequency", Stage_fear = "stage_fear",
    Drained_after_socializing = "drained_after_socializing"
  )
  dt <- as.data.table(df)
  setnames(dt, old = names(raw_to_snake), new = unname(raw_to_snake), skip_absent = TRUE)

  list(data = dt[])
}

#' Read a competition raw CSV, treating blank fields as NA
#'
#' `fread()`'s default `na.strings = "NA"` only matches the literal text
#' "NA" — a blank field in a character column (e.g. an unanswered
#' `Stage_fear`/`Drained_after_socializing` question) is read back as `""`,
#' not `NA`, unless `na.strings` explicitly includes `""`. This wrapper is
#' the single place that decision gets made, so every script reading
#' `train.csv`/`test.csv` sees genuine missingness as `NA` consistently.
#'
#' @param path Path to a competition CSV file (e.g. `"data/raw/train.csv"`).
#'
#' @return A data.table, as returned by `fread()`, with blank fields and
#'   the literal text "NA" both read as `NA`.
read_competition_csv <- function(path) {
  fread(path, na.strings = c("NA", ""))
}

#' Compare a predictor's glm coefficient magnitude with vs. without a conditioning variable
#'
#' Fits `personality_introvert ~ predictor` and
#' `personality_introvert ~ predictor + conditioning_vars` (both
#' `family = binomial`, complete cases per model via `glm()`'s default
#' `na.action`), and reports how much `predictor`'s own coefficient shifts
#' between the two. This is an effect-size-based test (retained coefficient
#' magnitude / odds-ratio distance from 1), not a p-value threshold -- see
#' the design spec's "Method: coefficient-shift comparison, not p-values".
#' A predictor is a "drop candidate" if its coefficient falls to <=20% of
#' its unconditioned magnitude, or its conditioned odds ratio lands within
#' +-10% of 1 (practically no association once the conditioning variable is
#' known); otherwise it's "keep".
#'
#' @param data data.table containing `personality_introvert`, `predictor`,
#'   and every column named in `conditioning_vars`
#' @param predictor character, name of the predictor whose coefficient is tracked
#' @param conditioning_vars character vector, name(s) of the variable(s)
#'   added in the conditioned model
#'
#' @return one-row data.table: `predictor`, `conditioning_vars`
#'   (comma-joined), `coef_alone`, `coef_conditioned`, `retained_pct`,
#'   `or_alone`, `or_conditioned`, `call` (`"keep"` or `"drop candidate"`)
coefficient_shift_check <- function(data, predictor, conditioning_vars) {
  f_alone <- as.formula(paste("personality_introvert ~", predictor))
  f_full <- as.formula(paste(
    "personality_introvert ~",
    paste(c(predictor, conditioning_vars), collapse = " + ")
  ))

  fit_alone <- glm(f_alone, data = data, family = binomial)
  fit_full <- glm(f_full, data = data, family = binomial)

  coef_alone <- unname(coef(fit_alone)[predictor])
  coef_conditioned <- unname(coef(fit_full)[predictor])
  retained_pct <- 100 * abs(coef_conditioned) / abs(coef_alone)
  or_conditioned <- exp(coef_conditioned)

  call <- if (retained_pct <= 20 || abs(or_conditioned - 1) <= 0.10) "drop candidate" else "keep"

  data.table(
    predictor = predictor,
    conditioning_vars = paste(conditioning_vars, collapse = ", "),
    coef_alone = coef_alone,
    coef_conditioned = coef_conditioned,
    retained_pct = retained_pct,
    or_alone = exp(coef_alone),
    or_conditioned = or_conditioned,
    call = call
  )
}
