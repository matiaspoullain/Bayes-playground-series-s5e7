library(brms)
library(bayesplot)
library(ggplot2)
library(data.table)

fit <- readRDS("models/stage1_fit_full.rds")
dir.create("tables_figures/stage1_diagnostics", showWarnings = FALSE, recursive = TRUE)

summ <- summary(fit)
diagnostics <- as.data.table(summ$fixed, keep.rownames = "term")[, .(term, Rhat, Bulk_ESS, Tail_ESS)]

divergences <- as.data.table(nuts_params(fit))[Parameter == "divergent__", sum(Value)]
diagnostics_summary <- rbindlist(list(
  diagnostics,
  data.table(term = "TOTAL_DIVERGENCES", Rhat = NA, Bulk_ESS = NA, Tail_ESS = divergences)
))

fwrite(diagnostics_summary, "tables_figures/stage1_diagnostics/rhat_ess_summary.csv")
print(diagnostics_summary)

cat("\nMax Rhat:", max(diagnostics$Rhat, na.rm = TRUE), "(target: < 1.01)\n")
cat("Min Bulk ESS:", min(diagnostics$Bulk_ESS, na.rm = TRUE), "(target: > 400)\n")
cat("Divergent transitions:", divergences, "(target: 0)\n")

pp_check(fit, type = "bars", ndraws = 1000) +
  labs(
    title = "Stage 1 posterior predictive check",
    subtitle = "Simulated 0/1 outcome proportions across 1000 posterior draws vs. observed (y)",
    x = "Introvert (1) / Extrovert (0)", y = "Count"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage1_diagnostics/pp_check_bars.png", width = 7, height = 6, bg = "white")
