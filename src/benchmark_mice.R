source("src/features.R")
library(mice)
library(data.table)

train_raw <- read_competition_csv("data/raw/train.csv")
prepared <- prepare_stage2_data(train_raw)$data

impute_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency", "stage_fear_bin",
  "drained_after_socializing_bin", "personality_introvert"
)
mice_input <- as.data.frame(prepared[, ..impute_cols])

cat("Rows:", nrow(mice_input), " Cols:", ncol(mice_input), "\n")
cat("Missing cells per column:\n")
print(colSums(is.na(mice_input)))

m_value <- 5
cat("\nRunning mice(m =", m_value, ", maxit = 5, default method selection)...\n")

start_time <- Sys.time()
imp <- mice(mice_input, m = m_value, maxit = 5, seed = 1, printFlag = TRUE)
end_time <- Sys.time()

elapsed <- end_time - start_time
cat("\n--- Timing result ---\n")
cat("mice() with m =", m_value, "took:", format(elapsed), "\n")
cat("Estimated per-imputation time:", as.numeric(elapsed, units = "secs") / m_value, "seconds\n")

saveRDS(imp, "models/mice_benchmark_test.rds")
cat("\nSaved test object to models/mice_benchmark_test.rds (gitignored, for inspection only)\n")
