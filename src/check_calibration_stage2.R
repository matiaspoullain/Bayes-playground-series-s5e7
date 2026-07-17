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

calibration_summary <- data.table(
  accuracy = accuracy(oof$actual, oof$oof_pred_prob),
  auc = auc(oof$actual, oof$oof_pred_prob),
  brier_score = brier_score(oof$actual, oof$oof_pred_prob),
  n = nrow(oof)
)
fwrite(calibration_summary, "tables_figures/stage2_calibration/calibration_summary.csv")
fwrite(reliability, "tables_figures/stage2_calibration/reliability_by_bin.csv")

cat("\n--- Stage 2 calibration summary ---\n")
print(calibration_summary)

# Cross-check: recomputed pooled accuracy/AUC should match fit_stage2_cv.R's own
# printed pooled figures.
cat(sprintf(
  "\nCross-check: recomputed pooled accuracy = %.4f, pooled AUC = %.4f\n",
  calibration_summary$accuracy, calibration_summary$auc
))

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
    subtitle = "Out-of-fold predictions (5-fold stratified CV), 10 equally-spaced bins"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage2_calibration/reliability_diagram.png", width = 7, height = 7, bg = "white")
