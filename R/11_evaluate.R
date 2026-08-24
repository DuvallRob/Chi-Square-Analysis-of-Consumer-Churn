# R/11_evaluate.R
# Final evaluation on the held-out test set (20% of the post-SMOTE ~7,500-row
# dataset). Reported two ways:
#   1. Full test set, as split -- this is your primary, rubric-compliant
#      result (min. 7,000 rows satisfied, honest 80/20 split of that set).
#   2. Real-rows-only subset (is_synthetic == FALSE) -- a supplementary,
#      more conservative check. Because SMOTE ran before the split, some
#      synthetic test rows were interpolated from real rows that landed in
#      train, so (1) alone can run a little optimistic. Report both in
#      Section E, and name the gap between them as your limitation -- that's
#      the honest write-up of the tradeoff you chose, not a reason to second-
#      guess the pipeline order.

library(xgboost)
library(pROC)
library(mlflow)

test_data    <- readRDS("data/processed/test_features_full.rds")
feature_cols <- readRDS("models/feature_cols.rds")
model        <- xgb.load("models/final_xgb_model.model")

score <- function(data) {
  dmat <- xgb.DMatrix(as.matrix(data[, feature_cols]))
  pred_prob  <- predict(model, dmat)
  pred_class <- as.integer(pred_prob > 0.5)
  actual     <- data$churn_signal
  
  cm <- table(predicted = pred_class, actual = actual)
  
  tp <- sum(pred_class == 1 & actual == 1); tn <- sum(pred_class == 0 & actual == 0)
  fp <- sum(pred_class == 1 & actual == 0); fn <- sum(pred_class == 0 & actual == 1)
  
  accuracy  <- (tp + tn) / length(actual)
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1        <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  auc_val   <- if (length(unique(actual)) < 2) NA_real_ else
    as.numeric(auc(roc(actual, pred_prob, quiet = TRUE)))
  
  list(cm = cm, accuracy = accuracy, precision = precision, recall = recall,
       f1 = f1, auc = auc_val, pred_prob = pred_prob, actual = actual, n = length(actual))
}

full_eval <- score(test_data)
real_only <- test_data[test_data$is_synthetic == FALSE, ]
real_eval <- score(real_only)

report <- function(label, r) {
  cat(sprintf("\n--- %s (n = %d) ---\n", label, r$n))
  print(r$cm)
  cat(sprintf("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %s\n",
              r$accuracy, r$precision, r$recall, r$f1,
              ifelse(is.na(r$auc), "NA", sprintf("%.4f", r$auc))))
}

report("Full test set (real + synthetic)", full_eval)
report("Real rows only",                    real_eval)

mlflow_set_experiment("consumer-churn-xgboost-final")
with(mlflow_start_run(), {
  mlflow_log_metric("test_full_accuracy",  full_eval$accuracy)
  mlflow_log_metric("test_full_precision", full_eval$precision)
  mlflow_log_metric("test_full_recall",    full_eval$recall)
  mlflow_log_metric("test_full_f1",        full_eval$f1)
  if (!is.na(full_eval$auc)) mlflow_log_metric("test_full_auc", full_eval$auc)
  
  mlflow_log_metric("test_real_only_accuracy",  real_eval$accuracy)
  mlflow_log_metric("test_real_only_precision", real_eval$precision)
  mlflow_log_metric("test_real_only_recall",    real_eval$recall)
  mlflow_log_metric("test_real_only_f1",        real_eval$f1)
  if (!is.na(real_eval$auc)) mlflow_log_metric("test_real_only_auc", real_eval$auc)
})

saveRDS(list(full = full_eval, real_only = real_eval),
        "data/processed/test_evaluation.rds")