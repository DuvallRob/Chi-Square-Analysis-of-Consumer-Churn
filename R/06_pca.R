# PCA on the 768-dim embeddings, FIT ON TRAIN ONLY. Needed before K-Means
# (curse of dimensionality) and before XGBoost given n=801 training rows.
# 90% variance-explained threshold is a reasonable default -- adjust if you
# want more/less compression.

train_set <- readRDS("data/processed/train_set.rds")
test_set  <- readRDS("data/processed/test_set.rds")

emb_cols <- grep("^emb_", names(train_set), value = TRUE)

pca_fit <- prcomp(train_set[, emb_cols], center = TRUE, scale. = TRUE)

var_explained <- cumsum(pca_fit$sdev^2) / sum(pca_fit$sdev^2)
n_components <- which(var_explained >= 0.90)[1]
cat("Retaining", n_components, "PCs (",
    round(var_explained[n_components] * 100, 1), "% variance explained)\n")

pc_cols <- paste0("PC", seq_len(n_components))

train_pcs <- as.data.frame(predict(pca_fit, train_set[, emb_cols])[, seq_len(n_components), drop = FALSE])
names(train_pcs) <- pc_cols
test_pcs <- as.data.frame(predict(pca_fit, test_set[, emb_cols])[, seq_len(n_components), drop = FALSE])
names(test_pcs) <- pc_cols

# Shapiro-Wilk normality check on PC1, per the Task 1 commitment to test
# normality of the continuous embedding-derived variables.
# Base R's shapiro.test has a limit of 5000 observations. 
set.seed(606)
n_sample <- min(length(train_pcs$PC1), 5000)
sw <- shapiro.test(sample(train_pcs$PC1, n_sample))

cat("\nShapiro-Wilk on PC1: W =", round(sw$statistic, 4),
    " p =", format.pval(sw$p.value, digits = 4), "\n")

saveRDS(pca_fit, "data/processed/pca_fit.rds")
saveRDS(cbind(row_id = train_set$row_id, train_pcs), "data/processed/train_pcs.rds")
saveRDS(cbind(row_id = test_set$row_id, test_pcs), "data/processed/test_pcs.rds")