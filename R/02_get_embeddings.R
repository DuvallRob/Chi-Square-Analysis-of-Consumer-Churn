# Bridges to Python via reticulate to compute DistilBERT embeddings +
# sentiment for every narrative, cached to disk since CPU inference over
# 1,001 rows is the slowest step in the pipeline.
#
# ONE-TIME LOCAL SETUP (skip inside Docker -- the Dockerfile builds its own
# venv and sets RETICULATE_PYTHON directly):
#   reticulate::virtualenv_create("d606-nlp", packages = c(
#     "torch==2.3.1", "transformers==4.41.2", "numpy==1.26.4"))

library(reticulate)

if (Sys.getenv("RETICULATE_PYTHON") == "") {
  reticulate::use_virtualenv("d606-nlp", required = TRUE)
}

cache_path <- "data/processed/embeddings_cache.rds"
df <- readRDS("data/processed/model_frame.rds")

if (file.exists(cache_path)) {
  cat("Loading cached embeddings...\n")
  cached <- readRDS(cache_path)
} else {
  cat("Computing DistilBERT embeddings + sentiment (this can take a while)...\n")
  reticulate::source_python("python/embed_narratives.py")
  
  result <- get_narrative_features(df$narrative, batch_size = 16L)
  
  embed_matrix <- do.call(rbind, lapply(result$embeddings, as.numeric))
  colnames(embed_matrix) <- paste0("emb_", seq_len(ncol(embed_matrix)))
  
  cached <- list(
    complaint_id = df$complaint_id,
    embeddings = embed_matrix,
    sentiment_label = unlist(result$sentiment_label),
    sentiment_score = unlist(result$sentiment_score)
  )
  saveRDS(cached, cache_path)
}

stopifnot(identical(cached$complaint_id, df$complaint_id))
cat("Embedding matrix:", nrow(cached$embeddings), "x", ncol(cached$embeddings), "\n")