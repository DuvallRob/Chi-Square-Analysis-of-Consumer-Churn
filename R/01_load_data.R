# Loads golden_set.csv + cv_folds.csv, merges on complaint_id, and keeps only
# the true IV/DV columns per the approved Task 1 variable table.
#
# EXCLUDED ON PURPOSE: proposed_label, evidence_quote, confidence,
# evidence_verbatim, labeler_model, ambiguous, human_overrode. These are
# metadata from the labeling process that PRODUCED churn_signal itself --
# proposed_label matches churn_signal in 995/1001 rows. Using any of them as
# a feature is leakage, the same failure mode as running SMOTE/K-Means before
# the train/test split.

library(readr)
library(dplyr)
library(stringr)

golden <- read_csv("data/golden_set.csv", show_col_types = FALSE)
folds  <- read_csv("data/cv_folds.csv",  show_col_types = FALSE)

stopifnot(
  n_distinct(golden$complaint_id) == nrow(golden),
  n_distinct(folds$complaint_id)  == nrow(folds)
)

df <- golden %>%
  inner_join(folds, by = "complaint_id") %>%
  transmute(
    complaint_id,
    product = factor(product,
                     levels = c("Checking or savings account",
                                "Credit card",
                                "Debt collection")),
    narrative = str_squish(narrative),   # collapse the embedded \n\n / whitespace
    fold,
    churn_signal = as.integer(churn_signal)
  )

stopifnot(nrow(df) == nrow(golden))  # confirms the join dropped nothing

dir.create("data/processed", showWarnings = FALSE)
saveRDS(df, "data/processed/model_frame.rds")

cat("Loaded", nrow(df), "rows.\n")
print(table(df$product))
cat("Overall churn rate:", mean(df$churn_signal), "\n")