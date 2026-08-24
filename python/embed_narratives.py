"""
Called from R via reticulate::source_python(). For each narrative, produces:
  1. A 768-dim mean-pooled DistilBERT embedding (distilbert-base-uncased) --
     feeds K-Means (complaint-pattern clustering) and XGBoost.
  2. A signed sentiment score in [-1, 1] from a sentiment-finetuned DistilBERT
     checkpoint -- fulfills the Task 1 commitment to "NLP sentiment analysis
     on the unstructured narratives" as an engineered feature.

Truncates at 256 tokens. Narratives run 35-375 words; DistilBERT's ceiling is
512 tokens, so 256 keeps inference fast without cutting off the substance of
the complaint for the large majority of rows.
"""

import torch
from transformers import AutoTokenizer, AutoModel, pipeline

_EMBED_MODEL_NAME = "distilbert-base-uncased"
_SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = AutoTokenizer.from_pretrained(_EMBED_MODEL_NAME)
_embed_model = AutoModel.from_pretrained(_EMBED_MODEL_NAME).to(_device)
_embed_model.eval()

_sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=_SENTIMENT_MODEL_NAME,
    tokenizer=_SENTIMENT_MODEL_NAME,
    truncation=True,
    max_length=256,
    device=0 if _device == "cuda" else -1,
)


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def get_narrative_features(narratives, batch_size=16):
    """
    Parameters
    ----------
    narratives : list[str]

    Returns
    -------
    dict: 'embeddings' (n x 768 list of lists), 'sentiment_label' (list[str]),
          'sentiment_score' (list[float], signed, [-1, 1])
    """
    narratives = list(narratives)
    all_embeddings, all_labels, all_scores = [], [], []

    for start in range(0, len(narratives), batch_size):
        batch = narratives[start:start + batch_size]

        encoded = _tokenizer(
            batch, padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            output = _embed_model(**encoded)

        pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        all_embeddings.extend(pooled.cpu().numpy().tolist())

        sentiments = _sentiment_pipe(batch)
        for s in sentiments:
            all_labels.append(s["label"])
            signed = s["score"] if s["label"] == "POSITIVE" else -s["score"]
            all_scores.append(float(signed))

        print(f"embedded {min(start + batch_size, len(narratives))}/{len(narratives)}")

    return {
        "embeddings": all_embeddings,
        "sentiment_label": all_labels,
        "sentiment_score": all_scores,
    }
