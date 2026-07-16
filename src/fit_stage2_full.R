source("src/features.R")
source("src/model_spec_stage2.R")
library(brms)
library(data.table)

train_raw <- read_competition_csv("data/raw/train.csv")
prepared <- prepare_stage2_data(train_raw)
train <- prepared$data

missingness_cols <- c("time_spent_alone", "social_event_attendance", "going_outside",
                       "friends_circle_size", "post_frequency", "stage_fear_bin",
                       "drained_after_socializing_bin")
train <- add_missingness_indicators(train, missingness_cols)

fit <- brm(
  formula = stage2_formula(),
  data = train,
  prior = stage2_priors(),
  backend = "cmdstanr",
  chains = 4, cores = 4, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95),
  seed = 1
)

print(summary(fit))

dir.create("models", showWarnings = FALSE)
saveRDS(fit, "models/stage2_fit_full.rds")
