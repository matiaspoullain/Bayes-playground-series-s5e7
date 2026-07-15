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

stage1_priors <- function() {
  c(
    set_prior("normal(-1.04, 1)", class = "Intercept"),
    set_prior("normal(2.5, 1.5)", class = "b", coef = "time_spent_alone_z"),
    set_prior("normal(-2.2, 1.5)", class = "b", coef = "social_event_attendance_z"),
    set_prior("normal(-2.2, 1.5)", class = "b", coef = "going_outside_z"),
    set_prior("normal(-2.0, 1.5)", class = "b", coef = "friends_circle_size_z"),
    set_prior("normal(-2.1, 1.5)", class = "b", coef = "post_frequency_z"),
    set_prior("normal(1.5, 1.5)", class = "b", coef = "stage_fear_bin"),
    set_prior("normal(1.5, 1.5)", class = "b", coef = "drained_after_socializing_bin"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_time_spent_alone"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_social_event_attendance"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_going_outside"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_friends_circle_size"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_post_frequency"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_stage_fear"),
    set_prior("normal(0, 1)", class = "b", coef = "was_missing_drained_after_socializing")
  )
}
