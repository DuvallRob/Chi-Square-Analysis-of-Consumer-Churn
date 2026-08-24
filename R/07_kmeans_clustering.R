# K-Means on the PCA-reduced embeddings, FIT ON TRAIN ONLY, to group
# narratives by complaint pattern. k chosen via elbow (WSS) + silhouette.
# Test rows are assigned to their nearest fitted centroid, never used to
# refit -- K-Means never sees the target, so this is the same "fit on train,
# apply to test" rule as PCA and SMOTE.

library(cluster)

train_pcs <- readRDS("data/processed/train_pcs.rds")
test_pcs  <- readRDS("data/processed/test_pcs.rds")
pc_cols   <- grep("^PC", names(train_pcs), value = TRUE)

X_train <- as.matrix(train_pcs[, pc_cols])

k_range <- 2:8
wss <- sapply(k_range, function(k) kmeans(X_train, centers = k, nstart = 25)$tot.withinss)
sil <- sapply(k_range, function(k) {
  km <- kmeans(X_train, centers = k, nstart = 25)
  mean(silhouette(km$cluster, dist(X_train))[, 3])
})

cat("k  :", paste(k_range, collapse = " "), "\n")
cat("wss:", paste(round(wss, 1), collapse = " "), "\n")
cat("sil:", paste(round(sil, 3), collapse = " "), "\n")

best_k <- k_range[which.max(sil)]
cat("\nSelected k =", best_k, "(max avg silhouette =", round(max(sil), 3), ")\n")

final_km <- kmeans(X_train, centers = best_k, nstart = 25)

assign_clusters <- function(newdata, centers) {
  d <- as.matrix(dist(rbind(centers, as.matrix(newdata))))
  k <- nrow(centers)
  d_sub <- d[(k + 1):nrow(d), 1:k, drop = FALSE]
  apply(d_sub, 1, which.min)
}

train_pcs$cluster <- factor(final_km$cluster, levels = seq_len(best_k))
test_pcs$cluster  <- factor(assign_clusters(test_pcs[, pc_cols], final_km$centers),
                            levels = seq_len(best_k))

saveRDS(list(model = final_km, k = best_k, wss = wss, sil = sil, k_range = k_range),
        "data/processed/kmeans_fit.rds")
saveRDS(train_pcs, "data/processed/train_pcs.rds")
saveRDS(test_pcs,  "data/processed/test_pcs.rds")