library(brms)

stage2_formula <- function() {
  bf(
    personality_introvert ~ 1 + time_spent_alone_z + going_outside_z +
      friends_circle_size_z + post_frequency_z + stage_fear_bin + drained_after_socializing_bin
  )
}

stage2_family <- function() {
  bernoulli(link = "logit")
}

# Same priors as the retired baseline model's stage1_priors() (src/old/): both encoded the same
# beliefs about the personality model. See design spec "Priors" section for full rationale.
stage2_priors <- function() {
  c(
    set_prior("normal(-1.04, 1)", class = "Intercept"),
    set_prior("normal(0, 2.5)", class = "b")
  )
}
