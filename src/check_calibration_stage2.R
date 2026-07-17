source("src/metrics.R")
library(ggplot2)
library(data.table)

oof <- fread("tables_figures/stage2_cv/oof_predictions.csv")
dir.create("tables_figures/stage2_calibration", showWarnings = FALSE, recursive = TRUE)

oof[, bin := cut(oof_pred_prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE, labels = FALSE)]

reliability <- oof[, .(
  observed_rate = mean(actual),
  mean_predicted = mean(oof_pred_prob),
  n = .N
), by = bin][order(bin)]

overall_accuracy <- accuracy(oof$actual, oof$oof_pred_prob)
overall_brier <- brier_score(oof$actual, oof$oof_pred_prob)

cat("Pooled out-of-fold accuracy:", overall_accuracy, "\n")
cat("Pooled out-of-fold Brier score:", overall_brier, "\n")

fwrite(
  data.table(metric = c("accuracy", "brier_score"), value = c(overall_accuracy, overall_brier)),
  "tables_figures/stage2_calibration/calibration_summary.csv"
)

ggplot(reliability, aes(x = mean_predicted, y = observed_rate)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  geom_line(colour = "#4C72B0", linewidth = 0.8) +
  geom_point(colour = "#2A4D77", fill = "white", shape = 21, size = 3.5, stroke = 1.2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  labs(
    x = "Mean predicted probability (per 10-point-wide bin)", y = "Observed introvert rate",
    title = "Stage 2 reliability diagram",
    subtitle = "Out-of-fold predictions (5-fold stratified CV, mice + brm_multiple), 10 equally-spaced bins"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage2_calibration/reliability_diagram.png", width = 7, height = 7, bg = "white")
