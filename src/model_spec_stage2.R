library(brms)

stage2_formula <- function() {
  bf_main <- bf(
    personality_introvert ~ mi(time_spent_alone) + mi(social_event_attendance) + mi(going_outside) +
      mi(friends_circle_size) + mi(post_frequency) + mi(stage_fear_bin) + mi(drained_after_socializing_bin),
    family = bernoulli(link = "logit")
  )

  bf_time_spent_alone <- bf(
    time_spent_alone | mi() ~ mi(social_event_attendance) + mi(going_outside) + mi(friends_circle_size) +
      mi(post_frequency) + mi(stage_fear_bin) + mi(drained_after_socializing_bin) + personality_introvert,
    family = gaussian()
  )

  bf_social_event_attendance <- bf(
    social_event_attendance | mi() ~ mi(time_spent_alone) + mi(going_outside) + mi(friends_circle_size) +
      mi(post_frequency) + mi(stage_fear_bin) + mi(drained_after_socializing_bin) + personality_introvert,
    family = gaussian()
  )

  bf_going_outside <- bf(
    going_outside | mi() ~ mi(time_spent_alone) + mi(social_event_attendance) + mi(friends_circle_size) +
      mi(post_frequency) + mi(stage_fear_bin) + mi(drained_after_socializing_bin) + personality_introvert,
    family = gaussian()
  )

  bf_friends_circle_size <- bf(
    friends_circle_size | mi() ~ mi(time_spent_alone) + mi(social_event_attendance) + mi(going_outside) +
      mi(post_frequency) + mi(stage_fear_bin) + mi(drained_after_socializing_bin) + personality_introvert,
    family = gaussian()
  )

  bf_post_frequency <- bf(
    post_frequency | mi() ~ mi(time_spent_alone) + mi(social_event_attendance) + mi(going_outside) +
      mi(friends_circle_size) + mi(stage_fear_bin) + mi(drained_after_socializing_bin) + personality_introvert,
    family = gaussian()
  )

  bf_stage_fear_bin <- bf(
    stage_fear_bin | mi() ~ mi(time_spent_alone) + mi(social_event_attendance) + mi(going_outside) +
      mi(friends_circle_size) + mi(post_frequency) + mi(drained_after_socializing_bin) + personality_introvert,
    family = gaussian()
  )

  bf_drained_after_socializing_bin <- bf(
    drained_after_socializing_bin | mi() ~ mi(time_spent_alone) + mi(social_event_attendance) + mi(going_outside) +
      mi(friends_circle_size) + mi(post_frequency) + mi(stage_fear_bin) + personality_introvert,
    family = gaussian()
  )

  bf_main + bf_time_spent_alone + bf_social_event_attendance + bf_going_outside +
    bf_friends_circle_size + bf_post_frequency + bf_stage_fear_bin + bf_drained_after_socializing_bin +
    set_rescor(FALSE)
}

# Priors chosen against the exact resp/class/coef names brms::get_prior() reports for
# stage2_formula() (see design spec "Missing-data strategy - full detail" and Task 11 report for
# the full get_prior() table). Main-model (resp = "personalityintrovert") priors match the
# current stage1_priors() so Stage 1 and Stage 2 encode the same beliefs about the personality
# model. Each of the 7 predictor sub-models gets a generic normal(0, 2) on its own b-class
# coefficients (coef left unspecified so it applies to every b-class coefficient for that resp);
# sub-model Intercepts are left at brms's own default (not customized).
stage2_priors <- function() {
  c(
    set_prior("normal(-1.04, 1)", class = "Intercept", resp = "personalityintrovert"),
    set_prior("normal(0, 2.5)", class = "b", resp = "personalityintrovert"),
    set_prior("normal(0, 2)", class = "b", resp = "timespentalone"),
    set_prior("normal(0, 2)", class = "b", resp = "socialeventattendance"),
    set_prior("normal(0, 2)", class = "b", resp = "goingoutside"),
    set_prior("normal(0, 2)", class = "b", resp = "friendscirclesize"),
    set_prior("normal(0, 2)", class = "b", resp = "postfrequency"),
    set_prior("normal(0, 2)", class = "b", resp = "stagefearbin"),
    set_prior("normal(0, 2)", class = "b", resp = "drainedaftersocializingbin")
  )
}

stage2_family <- function() {
  # Families are per-submodel, declared inline in stage2_formula() above: bernoulli for the main
  # model, gaussian for all 7 predictor sub-models (including the two binary ones - brms/Stan's
  # mi() response-side term does not support discrete-family responses, so the binary sub-models
  # use gaussian() as a pragmatic approximation, same as the five count-like sub-models; see
  # design spec "Missing-data strategy - full detail" for the full rationale). This function
  # exists only for interface symmetry with stage1_family(); brms ignores an explicit top-level
  # family argument when every bf() component already declares its own.
  NULL
}
