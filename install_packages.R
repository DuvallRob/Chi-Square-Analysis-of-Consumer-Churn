pkgs <- c("shiny", "reticulate", "xgboost", "smotefamily", "cluster", "caret",
          "ggplot2", "dplyr", "readr", "stringr", "tidyr", "scales",
          "mlflow", "pROC", "tibble")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, repos = "https://cloud.r-project.org")
