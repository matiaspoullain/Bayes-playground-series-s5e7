source("src/features.R")
source("src/model_spec_stage1.R")
library(brms)
library(data.table)

train_raw <- fread("data/raw/train.csv")
prepared <- prepare_stage1_data(train_raw)
train <- prepared$data

fit_prior <- brm(
  formula = stage1_formula(),
  data = train,
  family = stage1_family(),
  prior = stage1_priors(),
  sample_prior = "only",
  backend = "cmdstanr",
  chains = 2, iter = 500, warmup = 250,
  seed = 1
)

prior_probs <- posterior_epred(fit_prior, ndraws = 200)
cat("Prior predictive probability range: [", min(prior_probs), ",", max(prior_probs), "]\n")
cat("Prior predictive probability median:", median(prior_probs), "\n")
cat("Observed base Introvert rate:", mean(train$personality_introvert), "\n")
