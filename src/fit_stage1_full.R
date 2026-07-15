source("src/features.R")
source("src/model_spec_stage1.R")
library(brms)
library(data.table)

train_raw <- fread("data/raw/train.csv")
prepared <- prepare_stage1_data(train_raw)
train <- prepared$data

fit <- brm(
  formula = stage1_formula(),
  data = train,
  family = stage1_family(),
  prior = stage1_priors(),
  backend = "cmdstanr",
  chains = 4, cores = 4, iter = 2000, warmup = 1000,
  seed = 1
)

print(summary(fit))

dir.create("models", showWarnings = FALSE)
saveRDS(fit, "models/stage1_fit_full.rds")
saveRDS(prepared$stats, "models/stage1_fit_full_stats.rds")
