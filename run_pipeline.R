# run_pipeline.R — runs the whole thing top to bottom from a terminal.
scripts <- c(
  "R/01_load_data.R",
  "R/02_get_embeddings.R",
  "R/03_smote_expand.R",
  "R/04_chi_square.R",
  "R/05_train_test_split.R",
  "R/06_pca.R",
  "R/07_kmeans_clustering.R",
  "R/08_assemble_features.R",
  "R/09_hyperparam_cv.R",
  "R/10_final_train.R",
  "R/11_evaluate.R"
)

for (s in scripts) {
  cat("\n=====", s, "=====\n")
  source(s, echo = TRUE)
}