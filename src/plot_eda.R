source("src/features.R")
library(ggplot2)
library(data.table)

train_raw <- read_competition_csv("data/raw/train.csv")
prepared <- prepare_stage2_data(train_raw)$data

dir.create("tables_figures/eda", showWarnings = FALSE, recursive = TRUE)

# --- Correlation matrix: 7 predictors + target, complete-case per pair ---
vars <- c(
  "personality_introvert", "time_spent_alone", "social_event_attendance",
  "going_outside", "friends_circle_size", "post_frequency",
  "stage_fear_bin", "drained_after_socializing_bin"
)
cor_mat <- cor(as.matrix(prepared[, vars, with = FALSE]), use = "pairwise.complete.obs")

cor_dt <- melt(
  as.data.table(cor_mat, keep.rownames = "var1"),
  id.vars = "var1", variable.name = "var2", value.name = "r"
)
cor_dt[, `:=`(
  var1 = factor(var1, levels = vars),
  var2 = factor(var2, levels = rev(vars))
)]

p_corr <- ggplot(cor_dt, aes(x = var1, y = var2, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3.1, colour = "grey15") +
  scale_fill_gradient2(
    low = "#B23A48", mid = "white", high = "#1B6CA8",
    midpoint = 0, limits = c(-1, 1), name = "Pearson r"
  ) +
  labs(
    x = NULL, y = NULL,
    title = "Correlation matrix: all 7 predictors + target",
    subtitle = "Complete-case pairwise correlation (stage_fear_bin & drained_..._bin are near-duplicates, r=0.995)"
  ) +
  coord_fixed() +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank())
ggsave("tables_figures/eda/correlation_matrix.png", p_corr, width = 11, height = 8.5, bg = "white")

# --- Missingness by column ---
missing_cols <- c(
  "time_spent_alone", "social_event_attendance", "going_outside",
  "friends_circle_size", "post_frequency", "stage_fear_bin", "drained_after_socializing_bin"
)
missing_pct <- sapply(missing_cols, function(col) mean(is.na(prepared[[col]])) * 100)
missing_dt <- data.table(column = missing_cols, pct_missing = missing_pct)
missing_dt[, column := factor(column, levels = column[order(pct_missing)])]

p_missing <- ggplot(missing_dt, aes(x = column, y = pct_missing)) +
  geom_col(fill = "#4C72B0") +
  geom_text(aes(label = sprintf("%.1f%%", pct_missing)), hjust = -0.15, size = 3.5) +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    x = NULL, y = "% missing",
    title = "Missingness by column (train.csv)",
    subtitle = sprintf("%.1f%% of rows have at least one missing predictor", 100 * mean(!complete.cases(prepared[, missing_cols, with = FALSE])))
  ) +
  theme_minimal(base_size = 12)
ggsave("tables_figures/eda/missingness_by_column.png", p_missing, width = 8, height = 5, bg = "white")

# --- Missingness by column, split by target (MNAR check) ---
missing_by_target_dt <- rbindlist(lapply(missing_cols, function(col) {
  dt <- prepared[, .(pct_missing = mean(is.na(get(col))) * 100), by = personality_introvert]
  dt[, column := col]
}))
missing_by_target_dt[, `:=`(
  column = factor(column, levels = levels(missing_dt$column)),
  target = factor(personality_introvert, levels = c(0, 1), labels = c("Extrovert", "Introvert"))
)]

p_missing_by_target <- ggplot(missing_by_target_dt, aes(x = column, y = pct_missing, fill = target)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.1f%%", pct_missing)), hjust = -0.15, size = 3) +
  coord_flip(clip = "off") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  scale_fill_manual(values = c(Extrovert = "#1B6CA8", Introvert = "#B23A48"), name = NULL) +
  facet_wrap(~target) +
  labs(
    x = NULL, y = "% missing",
    title = "Missingness by column, split by target",
    subtitle = "Missingness rates differ by class — evidence the data is MNAR, not MCAR"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
ggsave("tables_figures/eda/missingness_by_column_by_target.png", p_missing_by_target, width = 10, height = 5, bg = "white")

cat("Saved tables_figures/eda/correlation_matrix.png, missingness_by_column.png, and missingness_by_column_by_target.png\n")
