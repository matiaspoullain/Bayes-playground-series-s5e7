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
