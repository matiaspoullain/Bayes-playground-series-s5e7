library(brms)

stage1_formula <- function() {
  bf(
    personality_introvert ~ 1 + time_spent_alone_z + social_event_attendance_z + going_outside_z +
      friends_circle_size_z + post_frequency_z + stage_fear_bin + drained_after_socializing_bin +
      was_missing_time_spent_alone + was_missing_social_event_attendance + was_missing_going_outside +
      was_missing_friends_circle_size + was_missing_post_frequency + was_missing_stage_fear +
      was_missing_drained_after_socializing
  )
}

stage1_family <- function() {
  bernoulli(link = "logit")
}

# Revised after prior predictive check (Task 5) revealed original directional priors were
# overconfident. See design spec "Priors" section for full rationale.
stage1_priors <- function() {
  c(
    set_prior("normal(-1.04, 1)", class = "Intercept"),
    set_prior("normal(0, 2.5)", class = "b")
  )
}
