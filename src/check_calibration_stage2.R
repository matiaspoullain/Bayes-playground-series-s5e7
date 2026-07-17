source("src/metrics.R")
library(ggplot2)
library(data.table)

# oof_predictions.csv now has a model_used column ("A"/"B") alongside
# id/actual/oof_pred_prob -- see design spec's "Two-model split". Model A
# predicts holdout rows with no missing predictors, Model B predicts holdout
# rows with missing predictors (imputed outcome-free). The combined/pooled
# reliability diagram and summary below are unchanged from before; a
# per-model breakdown is added alongside it, since the two models could in
# principle be calibrated differently even if their combined accuracy looks
# fine pooled together.
oof <- fread("tables_figures/stage2_cv/oof_predictions.csv")
dir.create("tables_figures/stage2_calibration", showWarnings = FALSE, recursive = TRUE)

oof[, bin := cut(oof_pred_prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE, labels = FALSE)]

#' Compute a decile reliability table and accuracy/Brier summary for a slice of OOF predictions
#'
#' @param oof_slice data.table subset with columns `bin`, `actual`, `oof_pred_prob`
#' @param group_label label identifying this slice (e.g. "combined", "modelA")
#' @return list(reliability = per-bin data.table, summary = one-row accuracy/Brier data.table)
summarize_calibration <- function(oof_slice, group_label) {
  reliability <- oof_slice[, .(
    observed_rate = mean(actual),
    mean_predicted = mean(oof_pred_prob),
    n = .N
  ), by = bin][order(bin)]
  reliability[, group := group_label]

  summary_row <- data.table(
    group = group_label,
    accuracy = accuracy(oof_slice$actual, oof_slice$oof_pred_prob),
    brier_score = brier_score(oof_slice$actual, oof_slice$oof_pred_prob),
    n = nrow(oof_slice)
  )
  list(reliability = reliability, summary = summary_row)
}

combined <- summarize_calibration(oof, "combined")
model_a <- summarize_calibration(oof[model_used == "A"], "modelA")
model_b <- summarize_calibration(oof[model_used == "B"], "modelB")

calibration_summary <- rbindlist(list(combined$summary, model_a$summary, model_b$summary))
fwrite(calibration_summary, "tables_figures/stage2_calibration/calibration_summary.csv")

cat("\n--- Stage 2 calibration summary (combined + Model A/B breakdown) ---\n")
print(calibration_summary)

# Cross-check: recomputed combined accuracy should match the CV script's own
# reported pooled accuracy (0.9690).
cat(sprintf(
  "\nCross-check: recomputed combined pooled accuracy = %.4f (expected ~0.9690)\n",
  combined$summary$accuracy
))

reliability_by_group <- rbindlist(list(combined$reliability, model_a$reliability, model_b$reliability))
fwrite(reliability_by_group, "tables_figures/stage2_calibration/reliability_by_group.csv")

# --- Overall combined reliability diagram (unchanged from before: all OOF
# predictions pooled, regardless of which model produced them) ---
ggplot(combined$reliability, aes(x = mean_predicted, y = observed_rate)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  geom_line(colour = "#4C72B0", linewidth = 0.8) +
  geom_point(colour = "#2A4D77", fill = "white", shape = 21, size = 3.5, stroke = 1.2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  labs(
    x = "Mean predicted probability (per 10-point-wide bin)", y = "Observed introvert rate",
    title = "Stage 2 reliability diagram",
    subtitle = "Out-of-fold predictions (5-fold stratified CV, Model A + Model B pooled), 10 equally-spaced bins"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage2_calibration/reliability_diagram.png", width = 7, height = 7, bg = "white")

# --- Model A vs. Model B calibration breakdown, faceted on one plot ---
by_model <- rbindlist(list(model_a$reliability, model_b$reliability))
by_model[, model_label := ifelse(group == "modelA", "Model A (complete rows)", "Model B (incomplete rows)")]

ggplot(by_model, aes(x = mean_predicted, y = observed_rate)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40", linewidth = 0.5) +
  geom_line(colour = "#4C72B0", linewidth = 0.8) +
  geom_point(colour = "#2A4D77", fill = "white", shape = 21, size = 3.5, stroke = 1.2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  facet_wrap(~model_label) +
  labs(
    x = "Mean predicted probability (per 10-point-wide bin)", y = "Observed introvert rate",
    title = "Stage 2 reliability diagram, by model",
    subtitle = "Out-of-fold predictions (5-fold stratified CV), 10 equally-spaced bins per panel"
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/stage2_calibration/reliability_diagram_by_model.png", width = 12, height = 6.5, bg = "white")
