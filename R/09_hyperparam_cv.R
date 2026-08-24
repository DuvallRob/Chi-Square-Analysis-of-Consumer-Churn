# Manual 4-fold CV using the ORIGINAL fold labels (0-3) inside train. 
library(dplyr)
library(xgboost)
library(caret)
library(mlflow)

train_data <- readRDS("data/processed/train_features_full.rds")
feature_cols <- setdiff(names(train_data), c("churn_signal", "row_id", "is_synthetic"))

set.seed(606)
folds <- createFolds(train_data$churn_signal, k = 4)
grid <- expand.grid(max_depth = c(3, 5), eta = c(0.05, 0.1), nrounds = c(100, 200))

# --- MLFLOW WINDOWS PATH & TIMEOUT FIX ---
py_bin <- reticulate::virtualenv_python("d606-nlp")
Sys.setenv(MLFLOW_BIN = file.path(dirname(py_bin), "mlflow.exe"))

if (!file.exists(Sys.getenv("MLFLOW_BIN"))) {
  reticulate::py_install("mlflow", envname = "d606-nlp")
}

# YOU MISSED THIS PART: Bypass the 10-second background server timeout
dir.create("mlruns", showWarnings = FALSE)
mlflow_set_tracking_uri(paste0("file://", normalizePath("mlruns", winslash = "/")))
# -----------------------------------------

mlflow_set_tracking_uri("http://127.0.0.1:5000")
mlflow_set_experiment("consumer-churn-xgboost-cv")

cv_results <- lapply(seq_len(nrow(grid)), function(i) {
  p <- grid[i, ]
  fold_f1 <- sapply(folds, function(val_idx) {
    cv_train <- train_data[-val_idx, ]; cv_val <- train_data[val_idx, ]
    dtrain <- xgb.DMatrix(as.matrix(cv_train[, feature_cols]), label = cv_train$churn_signal)
    dval   <- xgb.DMatrix(as.matrix(cv_val[, feature_cols]))
    model <- xgb.train(list(objective = "binary:logistic", eval_metric = "auc",
                            max_depth = p$max_depth, eta = p$eta),
                       dtrain, nrounds = p$nrounds, verbose = 0)
    pred_class <- as.integer(predict(model, dval) > 0.5)
    tp <- sum(pred_class == 1 & cv_val$churn_signal == 1)
    fp <- sum(pred_class == 1 & cv_val$churn_signal == 0)
    fn <- sum(pred_class == 0 & cv_val$churn_signal == 1)
    precision <- ifelse(tp+fp==0,0,tp/(tp+fp)); recall <- ifelse(tp+fn==0,0,tp/(tp+fn))
    ifelse(precision+recall==0,0,2*precision*recall/(precision+recall))
  })
  mean_f1 <- mean(fold_f1)
  with(mlflow_start_run(), {
    mlflow_log_param("max_depth", p$max_depth); mlflow_log_param("eta", p$eta)
    mlflow_log_param("nrounds", p$nrounds); mlflow_log_metric("cv_mean_f1", mean_f1)
  })
  cat(sprintf("depth=%d eta=%.2f nrounds=%d -> F1=%.4f\n", p$max_depth, p$eta, p$nrounds, mean_f1))
  data.frame(p, mean_f1 = mean_f1)
}) %>% bind_rows()

best <- cv_results[which.max(cv_results$mean_f1), ]
saveRDS(best, "data/processed/best_params.rds")
saveRDS(cv_results, "data/processed/cv_results.rds")