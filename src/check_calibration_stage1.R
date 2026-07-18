source("src/metrics.R")
library(ggplot2)
library(data.table)

oof <- fread("tables_figures/stage1_cv/oof_predictions.csv")
dir.create("tables_figures/stage1_calibration", showWarnings = FALSE, recursive = TRUE)

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
  "tables_figures/stage1_calibration/calibration_summary.csv"
)

# Grouped bars (predicted vs. observed per bin) rather than a predicted-vs-observed
# line: with only 10 bins, paired bars make the per-bin gap between "what the model
# said" and "what actually happened" a directly readable height difference, and the
# bin's row count (labelled above each pair) speaks to how much each gap should be
# trusted.
reliability_long <- melt(
  reliability, id.vars = c("bin", "n"),
  measure.vars = c("mean_predicted", "observed_rate"),
  variable.name = "quantity", value.name = "rate"
)
reliability_long[, quantity := factor(
  quantity,
  levels = c("mean_predicted", "observed_rate"),
  labels = c("Mean predicted", "Observed")
)]
bin_labels <- reliability[, sprintf("[%.1f, %.1f)", (bin - 1) / 10, bin / 10)]
n_labels <- reliability[, .(bin, n, y = pmax(mean_predicted, observed_rate) + 0.03)]

ggplot(reliability_long, aes(x = factor(bin), y = rate, fill = quantity)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(
    data = n_labels, aes(x = factor(bin), y = y, label = paste0("n=", n)),
    inherit.aes = FALSE, size = 2.8, colour = "grey30"
  ) +
  scale_x_discrete(labels = bin_labels) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1), limits = c(0, 1.05), expand = c(0, 0)) +
  scale_fill_manual(values = c("Mean predicted" = "#4C72B0", "Observed" = "#B23A48"), name = NULL) +
  labs(
    x = "Predicted-probability bin", y = "Rate",
    title = "Stage 1 calibration by bin",
    subtitle = "Out-of-fold predictions (10-fold stratified CV); bars should match within each bin if well-calibrated"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("tables_figures/stage1_calibration/reliability_diagram.png", width = 9, height = 6.5, bg = "white")
