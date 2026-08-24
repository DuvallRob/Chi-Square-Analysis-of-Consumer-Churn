# 80/20 split of the full ~7,500-row expanded dataset, stratified by
# churn_signal. The original cv_folds.csv fold labels don't extend to
# synthetic rows, so this uses a fresh stratified split instead; k-fold CV
# for hyperparameter tuning is regenerated inside the training portion in
# 09_hyperparam_cv.R.

library(caret)
set.seed(606)

full <- readRDS("data/processed/expanded_dataset.rds")
train_idx <- createDataPartition(full$churn_signal, p = 0.8, list = FALSE)

train_set <- full[train_idx, ]
test_set  <- full[-train_idx, ]

cat("Train:", nrow(train_set), "| churn:", mean(train_set$churn_signal),
    "| synthetic:", mean(train_set$is_synthetic), "\n")
cat("Test: ", nrow(test_set), "| churn:", mean(test_set$churn_signal),
    "| synthetic:", mean(test_set$is_synthetic), "\n")

saveRDS(train_set, "data/processed/train_set.rds")
saveRDS(test_set,  "data/processed/test_set.rds")