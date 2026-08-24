# SMOTE on the full 1,001-row golden set, run BEFORE any analysis, per the
# approved Task 1 order and the rubric's 7,000-row minimum. Needs numeric
# inputs, so it runs on [DistilBERT embeddings + sentiment_score + one-hot
# product dummies] from 02_get_embeddings.R. Synthetic rows have no real
# narrative text (they're interpolated points, not new complaints), so
# `narrative`/`complaint_id` are NA for them and `is_synthetic` flags them.

library(dplyr)
library(smotefamily)

df  <- readRDS("data/processed/model_frame.rds")
emb <- readRDS("data/processed/embeddings_cache.rds")

embed_df <- as.data.frame(emb$embeddings)
emb_cols <- names(embed_df)

full <- df %>%
  mutate(sentiment_score = emb$sentiment_score[match(complaint_id, emb$complaint_id)]) %>%
  bind_cols(embed_df)

product_dummies <- as.data.frame(model.matrix(~ product, data = full))[, -1, drop = FALSE]
names(product_dummies) <- make.names(names(product_dummies))
dummy_cols <- names(product_dummies)

X <- cbind(full[, emb_cols], full[, "sentiment_score", drop = FALSE], product_dummies)

n_real      <- nrow(full)
minority_n  <- sum(full$churn_signal == 1)
target_total <- 7500
dup_size <- max(1, round((target_total - n_real) / minority_n))  # lands close to ~7,500

cat("Pre-SMOTE:", n_real, "rows | dup_size =", dup_size, "\n")

smote_out <- SMOTE(X = X, target = full$churn_signal, K = 5, dup_size = dup_size)
sm_data <- smote_out$data
names(sm_data)[names(sm_data) == "class"] <- "churn_signal"
sm_data$churn_signal <- as.integer(as.character(sm_data$churn_signal))

n_total <- nrow(sm_data)
sm_data$is_synthetic <- c(rep(FALSE, n_real), rep(TRUE, n_total - n_real))
sm_data$row_id       <- seq_len(n_total)   # stable id used by every later join
sm_data$complaint_id <- c(full$complaint_id, rep(NA_real_, n_total - n_real))

# Reconstruct a categorical `product` for synthetic rows via argmax of the
# interpolated one-hot dummies -- the only way back to a discrete label.
baseline_level <- levels(full$product)[1]
other_levels   <- levels(full$product)[-1]
reconstruct_product <- function(vals) {
  if (all(vals < 0.5)) baseline_level else other_levels[which.max(vals)]
}
synthetic_dummies  <- as.matrix(sm_data[(n_real + 1):n_total, dummy_cols])
synthetic_products <- apply(synthetic_dummies, 1, reconstruct_product)
sm_data$product <- factor(c(as.character(full$product), synthetic_products),
                          levels = levels(full$product))

cat("Post-SMOTE:", n_total, "rows\n")
print(table(sm_data$churn_signal))
print(table(sm_data$is_synthetic))
print(table(sm_data$product))

saveRDS(sm_data, "data/processed/expanded_dataset.rds")