source("src/features.R")
library(ggdag)
library(ggplot2)
library(data.table)

# New Stage 1 (2026-07-20): replaces the point-imputation + was_missing_*
# baseline model. Uses an explicit causal DAG (Personality as latent trait,
# the 7 behaviors as its effects) plus coefficient-shift checks to decide
# which predictors carry independent signal about Personality vs. are
# redundant/mediated -- see the design spec's full reasoning:
# docs/superpowers/specs/Bayes-playground-series-s5e7/2026-07-20-causal-dag-variable-selection-design.md
# (in the parent repo, not this submodule).

train_raw <- read_competition_csv("data/raw/train.csv")
prepared <- prepare_stage2_data(train_raw)$data

# z-score the 5 numeric predictors from their own observed (non-missing)
# values, na.rm = TRUE -- same convention Stage 2 uses (standardize.R's
# stats are computed on observed values only, never a global constant that
# implicitly assumes 0 for missing rows).
numeric_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency"
)
for (col in numeric_cols) {
  z <- (prepared[[col]] - mean(prepared[[col]], na.rm = TRUE)) / sd(prepared[[col]], na.rm = TRUE)
  set(prepared, j = paste0(col, "_z"), value = z)
}

dir.create("tables_figures/stage1_dag", showWarnings = FALSE, recursive = TRUE)

# --- DAG: Personality is the root cause; the 7 behaviors are its effects.
# stage_fear_bin/drained_after_socializing_bin -> social_event_attendance_z
# and social_event_attendance_z -> friends_circle_size_z are the two
# plausible direct behavior-to-behavior edges (see spec "The DAG").
dag <- dagify(
  stage_fear_bin ~ personality_introvert,
  drained_after_socializing_bin ~ personality_introvert,
  time_spent_alone_z ~ personality_introvert,
  going_outside_z ~ personality_introvert,
  post_frequency_z ~ personality_introvert,
  social_event_attendance_z ~ personality_introvert + stage_fear_bin + drained_after_socializing_bin,
  friends_circle_size_z ~ personality_introvert + social_event_attendance_z,
  labels = c(
    personality_introvert = "Personality",
    stage_fear_bin = "stage_fear",
    drained_after_socializing_bin = "drained_after_socializing",
    time_spent_alone_z = "time_spent_alone",
    going_outside_z = "going_outside",
    post_frequency_z = "post_frequency",
    social_event_attendance_z = "social_event_attendance",
    friends_circle_size_z = "friends_circle_size"
  )
)

p_dag <- ggdag(dag, text = FALSE, use_labels = "label") +
  theme_dag_blank() +
  labs(
    title = "Causal DAG: Personality as latent trait, 7 behaviors as its effects",
    subtitle = "social_event_attendance and friends_circle_size are colliders (multiple parents)"
  )
ggsave("tables_figures/stage1_dag/dag.png", p_dag, width = 11, height = 8, bg = "white")

cat("Saved tables_figures/stage1_dag/dag.png\n")

# --- Coefficient-shift checks (spec "Method: coefficient-shift comparison, not p-values") ---
all_predictors <- c(
  "time_spent_alone_z", "social_event_attendance_z", "going_outside_z",
  "friends_circle_size_z", "post_frequency_z", "stage_fear_bin",
  "drained_after_socializing_bin"
)

# 3 DAG-motivated checks (spec "The DAG" / "Method", checks 1-3)
dag_checks <- rbindlist(list(
  coefficient_shift_check(prepared, "stage_fear_bin", "drained_after_socializing_bin"),
  coefficient_shift_check(prepared, "drained_after_socializing_bin", "stage_fear_bin"),
  coefficient_shift_check(
    prepared, "social_event_attendance_z",
    c("stage_fear_bin", "drained_after_socializing_bin")
  ),
  coefficient_shift_check(prepared, "friends_circle_size_z", "social_event_attendance_z")
))
dag_checks[, check := "dag_motivated"]

# --- Figure: % of bivariate coefficient retained after conditioning, per
# DAG-motivated check, against the 20% decision threshold. This single number
# (retained_pct) is what actually drives every "keep"/"drop candidate" call
# among the 4 checks above (see coefficient_shift_check()'s decision rule) --
# the general_screen rows are supporting context, not part of this figure.
#
#' Bar chart of retained_pct per DAG-motivated coefficient-shift check
#'
#' @param dag_checks output of `coefficient_shift_check()`, one row per
#'   DAG-motivated check, with columns `predictor`, `retained_pct`, `call`
#' @return a ggplot object
plot_retained_pct <- function(dag_checks) {
  term_order <- dag_checks[order(retained_pct), predictor]
  plot_data <- copy(dag_checks)
  plot_data[, predictor := factor(predictor, levels = term_order)]

  ggplot(plot_data, aes(y = predictor, x = retained_pct, fill = call)) +
    geom_col(width = 0.6) +
    geom_vline(xintercept = 20, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
    annotate(
      "text", x = 20, y = Inf, label = "20% decision threshold",
      colour = "grey40", size = 3.2, hjust = -0.05, vjust = 1.5
    ) +
    geom_text(
      aes(label = sprintf("%.1f%%", retained_pct)),
      hjust = -0.15, size = 3.5, colour = "grey20"
    ) +
    scale_fill_manual(values = c("keep" = "#2A4D77", "drop candidate" = "#C1666B")) +
    scale_x_continuous(limits = c(0, 65), expand = expansion(mult = c(0, 0.05))) +
    labs(
      x = "% of bivariate coefficient retained after conditioning", y = NULL, fill = "Call",
      title = "Stage 1: which predictors survive the DAG-motivated coefficient-shift checks",
      subtitle = paste0(
        "Below the 20% threshold, a predictor's association is judged\n",
        "fully explained by the conditioning variable(s)"
      )
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
}

ggsave(
  "tables_figures/stage1_dag/retained_pct_by_predictor.png",
  plot_retained_pct(dag_checks),
  width = 10, height = 5.5, bg = "white"
)
cat("Saved tables_figures/stage1_dag/retained_pct_by_predictor.png\n")

# General screen: each predictor's bivariate coefficient vs. its coefficient
# in the full 7-predictor model -- supporting context, not a primary
# decision (spec "Method", "General screen").
general_screen <- rbindlist(lapply(all_predictors, function(p) {
  others <- setdiff(all_predictors, p)
  coefficient_shift_check(prepared, predictor = p, conditioning_vars = others)
}))
general_screen[, check := "general_screen"]

results <- rbind(dag_checks, general_screen)
fwrite(results, "tables_figures/stage1_dag/coefficient_shift_checks.csv")

cat("\n--- DAG-motivated coefficient-shift checks ---\n")
print(dag_checks[, .(predictor, conditioning_vars, coef_alone, coef_conditioned, retained_pct, call)])
cat("\n--- General screen: bivariate vs. full 7-predictor model ---\n")
print(general_screen[, .(predictor, coef_alone, coef_conditioned, retained_pct, call)])

cat("\nSaved tables_figures/stage1_dag/coefficient_shift_checks.csv\n")
