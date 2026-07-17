source("src/features.R")
library(brms)
library(bayesplot)
library(posterior)
library(ggplot2)
library(data.table)
library(mice)

dir.create("tables_figures/stage2_diagnostics", showWarnings = FALSE, recursive = TRUE)

# Both Stage 2 models are brmsfit_multiple objects (from brm_multiple()). Per
# ?brm_multiple's own documentation, their combined Rhat/ESS (as reported by
# summary()) are "not informative" -- inflated by genuine between-imputation
# variability, not non-convergence. The documented way to check convergence is
# to subset the combined draws back into per-imputation chain blocks and run
# posterior::summarise_draws() on each block separately. This mirrors
# src/fit_stage2_full.R's / src/fit_stage2_full_outcome_free.R's own
# diagnostic blocks exactly (same chain-index arithmetic, same
# m_value/chains_per_imputation), applied here to both fit objects.
m_value <- 20
chains_per_imputation <- 4

#' Per-imputation Rhat/ESS convergence summary for a brm_multiple() fit
#'
#' @param fit a brmsfit_multiple object fit across `m_value` mice imputations
#'   x `chains_per_imputation` chains each
#' @return data.table, one row per imputation: imputation, max_rhat,
#'   min_ess_bulk, min_ess_tail
per_imputation_convergence <- function(fit) {
  draws <- as_draws_array(fit)
  n_chains_total <- posterior::nchains(draws)
  stopifnot(n_chains_total == m_value * chains_per_imputation)

  rbindlist(lapply(seq_len(m_value), function(i) {
    chain_idx <- ((i - 1) * chains_per_imputation + 1):(i * chains_per_imputation)
    draws_i <- subset_draws(draws, chain = chain_idx)
    summ_i <- summarise_draws(draws_i, default_convergence_measures())
    data.table(
      imputation = i,
      max_rhat = max(summ_i$rhat, na.rm = TRUE),
      min_ess_bulk = min(summ_i$ess_bulk, na.rm = TRUE),
      min_ess_tail = min(summ_i$ess_tail, na.rm = TRUE)
    )
  }))
}

#' Report a per-imputation convergence summary: write CSV + print worst-case figures
#'
#' @param per_imp_summary output of `per_imputation_convergence()`
#' @param model_label short label used in the output filename and console messages
#'   (e.g. "modelA", "modelB")
report_convergence <- function(per_imp_summary, model_label) {
  out_path <- sprintf("tables_figures/stage2_diagnostics/rhat_ess_summary_%s.csv", model_label)
  fwrite(per_imp_summary, out_path)
  cat(sprintf("\n--- %s per-imputation convergence ---\n", model_label))
  print(per_imp_summary)
  cat(sprintf(
    "%s: worst max Rhat across all imputations = %.5f (target: < 1.01)\n",
    model_label, max(per_imp_summary$max_rhat)
  ))
  cat(sprintf(
    "%s: worst min Bulk ESS across all imputations = %.1f (target: > 400)\n",
    model_label, min(per_imp_summary$min_ess_bulk)
  ))
}

# --- Model A: outcome-informed imputation (models/stage2_fit_full.rds) ---
fit_A <- readRDS("models/stage2_fit_full.rds")
conv_A <- per_imputation_convergence(fit_A)
report_convergence(conv_A, "modelA")

# --- Model B: outcome-free imputation (models/stage2_fit_full_outcome_free.rds) ---
fit_B <- readRDS("models/stage2_fit_full_outcome_free.rds")
conv_B <- per_imputation_convergence(fit_B)
report_convergence(conv_B, "modelB")

# pp_check(fit, type = "bars") called with default arguments (newdata = NULL)
# does NOT pool across all M imputations: brms's own warn_brmsfit_multiple()
# only fires that warning when newdata is NULL, and posterior_predict() with
# newdata = NULL silently reuses imputation 1's completed data for every draw.
# Passing an explicit per-imputation `newdata` (as below) sidesteps that branch
# entirely, so this block manually pools genuine posterior_predict() draws from
# all M imputations instead of just the first. This mirrors
# src/fit_stage2_full.R's completed-dataset-rebuild (mice::complete() + the
# same standardization stats) and its chain-index arithmetic (per-imputation
# chain blocks), extended here to flat draw_ids ranges for posterior_predict(),
# and applied to both models.
numeric_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency"
)

#' Build M standardized completed datasets from a saved mice imputation object
#'
#' @param imp a `mice::mice()` result (already-run imputation)
#' @param standardize_stats named list (one entry per `numeric_cols` element)
#'   of `list(mean, sd)`, applied identically to every completed copy
#' @param outcome_vector if not NULL, the true `personality_introvert` values
#'   to attach as a column (needed for Model B, whose mice input excluded the
#'   outcome entirely -- see fit_stage2_full_outcome_free.R -- so
#'   `complete()` never reconstructs it). NULL for Model A, whose mice input
#'   already included the outcome, so `complete()` returns it unchanged.
#' @return list of `m_value` standardized data.tables
build_completed_list <- function(imp, standardize_stats, outcome_vector = NULL) {
  lapply(seq_len(m_value), function(i) {
    dt <- as.data.table(complete(imp, i))
    if (!is.null(outcome_vector)) {
      dt[, personality_introvert := outcome_vector]
    }
    for (col in numeric_cols) {
      set(dt, j = paste0(col, "_z"), value = (dt[[col]] - standardize_stats[[col]]$mean) / standardize_stats[[col]]$sd)
    }
    dt[]
  })
}

#' Pool posterior_predict() draws across all M imputations and write a pp_check bars plot
#'
#' @param fit a brmsfit_multiple object
#' @param completed_list output of `build_completed_list()`, one entry per imputation
#' @param model_label short label for filenames/plot titles (e.g. "Model A")
#' @param out_path PNG output path
pooled_pp_check_bars <- function(fit, completed_list, model_label, out_path) {
  draws <- as_draws_array(fit)
  draws_per_chain <- posterior::niterations(draws)
  draws_per_imputation <- draws_per_chain * chains_per_imputation
  stopifnot(draws_per_imputation * m_value == posterior::ndraws(draws))

  # Subsample within each imputation's draw block (matching the previous
  # ndraws = 1000 default's overall scale) so the combined y_rep matrix
  # (pooled_draws x N observations) stays a manageable size.
  ndraws_per_imputation <- 50
  set.seed(1)
  y_rep_list <- lapply(seq_len(m_value), function(i) {
    block_start <- (i - 1) * draws_per_imputation + 1
    block_end <- i * draws_per_imputation
    draw_ids_i <- sample(block_start:block_end, ndraws_per_imputation)
    posterior_predict(fit, newdata = completed_list[[i]], draw_ids = draw_ids_i)
  })
  y_rep <- do.call(rbind, y_rep_list)

  # personality_introvert has zero missing values in the raw data, so it is
  # identical across all M completed datasets -- any one of them gives the
  # true observed outcome vector.
  y_obs <- completed_list[[1]]$personality_introvert

  ppc_bars(y = y_obs, yrep = y_rep) +
    labs(
      title = sprintf("Stage 2 %s posterior predictive check", model_label),
      subtitle = paste0(
        "Simulated 0/1 outcome proportions (", ndraws_per_imputation, " draws x ", m_value,
        " imputations, pooled) vs. observed (y)"
      ),
      x = "Introvert (1) / Extrovert (0)", y = "Count"
    ) +
    theme_minimal(base_size = 12)
  ggsave(out_path, width = 7, height = 6, bg = "white")
}

# --- Model A pooled pp_check ---
imp_A <- readRDS("models/stage2_mice_imp.rds")
standardize_stats_A <- readRDS("models/stage2_fit_full_standardize_stats.rds")
completed_list_A <- build_completed_list(imp_A, standardize_stats_A)
pooled_pp_check_bars(
  fit_A, completed_list_A, "Model A",
  "tables_figures/stage2_diagnostics/pp_check_bars_modelA.png"
)

# --- Model B pooled pp_check ---
# Model B's mice input excluded personality_introvert entirely (see
# fit_stage2_full_outcome_free.R / design spec "Two-model split"), so
# complete() on its saved imputation object never reconstructs an outcome
# column -- it must be reattached from the raw training data, exactly as
# fit_stage2_full_outcome_free.R does when building Model B's own training
# data.
imp_B <- readRDS("models/stage2_mice_imp_outcome_free.rds")
standardize_stats_B <- readRDS("models/stage2_fit_full_outcome_free_standardize_stats.rds")
train_raw <- read_competition_csv("data/raw/train.csv")
prepared <- prepare_stage2_data(train_raw)$data
completed_list_B <- build_completed_list(imp_B, standardize_stats_B, outcome_vector = prepared$personality_introvert)
pooled_pp_check_bars(
  fit_B, completed_list_B, "Model B",
  "tables_figures/stage2_diagnostics/pp_check_bars_modelB.png"
)
