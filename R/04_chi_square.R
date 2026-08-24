# Chi-Square Test of Independence: product x churn_signal, on the full
# ~7,500-row post-SMOTE dataset, per the approved order.
#
# One note for your write-up, not a blocker: synthetic rows are interpolated
# from real minority rows, so this p-value will run lower than the same test
# on the original 1,001 rows would -- that's the SMOTE volume talking, not a
# stronger population signal. Worth naming as your Section E limitation.

df <- readRDS("data/processed/expanded_dataset.rds")

ct <- table(df$product, df$churn_signal)
dimnames(ct) <- list(product = dimnames(ct)[[1]], churn_signal = c("0", "1"))
print(ct)

test <- chisq.test(ct)
print(test)

expected <- test$expected
cat("\nExpected counts:\n"); print(round(expected, 2))
cat("Minimum expected cell count:", round(min(expected), 2), "\n")

saveRDS(list(contingency_table = ct, chisq = test),
        "data/processed/chi_square_results_expanded.rds")