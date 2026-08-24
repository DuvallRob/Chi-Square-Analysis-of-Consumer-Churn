# Refit on the FULL training split (folds 0-3) with the winning
# hyperparameters, SMOTE applied once to that full training set. Test
# (fold 4) stays untouched until 10_evaluate.R.

library(xgboost)
library(mlflow)

train_data <- readRDS("data/processed/train_features_full.rds")
best <- readRDS("data/processed/best_params.rds")
feature_cols <- setdiff(names(train_data), c("churn_signal", "row_id", "is_synthetic"))

dtrain <- xgb.DMatrix(as.matrix(train_data[, feature_cols]), label = train_data$churn_signal)
final_model <- xgb.train(list(objective = "binary:logistic", eval_metric = "auc",
                              max_depth = best$max_depth, eta = best$eta),
                         dtrain, nrounds = best$nrounds, verbose = 1)

dir.create("models", showWarnings = FALSE)
xgb.save(final_model, "models/final_xgb_model.model")
saveRDS(feature_cols, "models/feature_cols.rds")

mlflow_set_tracking_uri("http://127.0.0.1:5000")
mlflow_set_experiment("consumer-churn-xgboost-cv")

with(mlflow_start_run(), {
  mlflow_log_param("max_depth", best$max_depth); mlflow_log_param("eta", best$eta)
  mlflow_log_param("nrounds", best$nrounds); mlflow_log_artifact("models/final_xgb_model.model")
})