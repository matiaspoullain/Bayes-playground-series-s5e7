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

# Diagonal reliability curve -- same idiom as Bayes-playground-series-s5e3's
# check_calibration.R (mean predicted vs. observed rate per bin, open circles
# connected by a line, dashed y = x reference), with point size added to
# encode each bin's N so the sparser middle bins read as visibly less certain
# than the two dense extreme bins.
ggplot(reliability, aes(x = mean_predicted, y = observed_rate)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey45", linewidth = 0.5) +
  geom_line(colour = "#2A4D77", linewidth = 0.8) +
  geom_point(aes(size = n), colour = "#2A4D77", fill = "white", shape = 21, stroke = 1.2) +
  geom_text(aes(label = paste0("n=", n)), vjust = -2.6, size = 2.9, colour = "grey30") +
  scale_size_continuous(range = c(3, 14), guide = "none") +
  coord_cartesian(xlim = c(0, 1), ylim = c(-0.03, 1.08)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  labs(
    x = "Mean predicted probability (per 10-point-wide bin)", y = "Observed rate",
    title = "Stage 2 reliability diagram",
    subtitle = "Out-of-fold predictions (5-fold stratified CV), 10 equally-spaced bins; dot size = N per bin"
  ) +
  theme_bw(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "grey88", linewidth = 0.3),
    panel.border = element_rect(colour = "grey70", fill = NA, linewidth = 0.6),
    plot.title = element_text(face = "bold", size = rel(1.15), colour = "grey10", margin = margin(b = 6)),
    plot.subtitle = element_text(colour = "grey35", margin = margin(b = 10)),
    axis.title = element_text(face = "bold", colour = "grey20"),
    axis.text = element_text(colour = "grey30")
  )
ggsave("tables_figures/stage2_calibration/reliability_diagram.png", width = 8, height = 7.5, bg = "white")
