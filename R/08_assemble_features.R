# Combines PCA components + K-Means cluster + product dummies + sentiment
# score into the final train/test feature tables used by the model.

library(dplyr)

train_set <- readRDS("data/processed/train_set.rds")
test_set  <- readRDS("data/processed/test_set.rds")
train_pcs <- readRDS("data/processed/train_pcs.rds")
test_pcs  <- readRDS("data/processed/test_pcs.rds")

build_features <- function(base_set, pcs_set) {
  base <- base_set %>% select(row_id, product, sentiment_score, churn_signal, is_synthetic)
  merged <- left_join(base, pcs_set, by = "row_id")
  
  product_dummies <- as.data.frame(model.matrix(~ product, data = merged))[, -1, drop = FALSE]
  cluster_dummies <- as.data.frame(model.matrix(~ cluster, data = merged))[, -1, drop = FALSE]
  pc_cols <- grep("^PC", names(merged), value = TRUE)
  
  out <- cbind(row_id = merged$row_id, merged[, pc_cols], product_dummies, cluster_dummies,
               sentiment_score = merged$sentiment_score, is_synthetic = merged$is_synthetic,
               churn_signal = merged$churn_signal)
  names(out) <- make.names(names(out))
  out
}

saveRDS(build_features(train_set, train_pcs), "data/processed/train_features_full.rds")
saveRDS(build_features(test_set,  test_pcs),  "data/processed/test_features_full.rds")