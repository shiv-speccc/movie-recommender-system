"""
evaluate.py
===========
Offline evaluation metrics for the recommendation system.

Metrics implemented
-------------------
* RMSE  – Root Mean Squared Error  (rating prediction quality)
* MAE   – Mean Absolute Error      (rating prediction quality)
* Precision@K  – fraction of top-K recommendations the user actually liked
* Recall@K     – fraction of liked items captured in top-K
* NDCG@K       – Normalised Discounted Cumulative Gain (ranking quality)
* Coverage     – fraction of catalogue a model can recommend
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Rating-prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics (top-K)
# ─────────────────────────────────────────────────────────────────────────────

def _dcg(relevances: list[int]) -> float:
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Normalised DCG@K.

    Parameters
    ----------
    recommended : ordered list of recommended item ids/titles
    relevant    : set of items the user actually liked (relevance threshold applied externally)
    k           : cutoff rank
    """
    top_k = recommended[:k]
    gains = [1 if item in relevant else 0 for item in top_k]
    ideal = sorted(gains, reverse=True)
    dcg   = _dcg(gains)
    idcg  = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue coverage
# ─────────────────────────────────────────────────────────────────────────────

def catalogue_coverage(all_recommendations: list[list], catalogue_size: int) -> float:
    """
    Fraction of total catalogue items that appear in at least one recommendation list.
    """
    unique_recommended = len({item for recs in all_recommendations for item in recs})
    return unique_recommended / catalogue_size


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline (leave-one-out)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cf_model(recommender, matrix: pd.DataFrame, stats: pd.DataFrame,
                       sample_movies: int = 50, top_k: int = 10,
                       relevance_threshold: float = 4.0) -> dict:
    """
    Evaluate a fitted recommender via leave-one-out sampling.

    For each sampled query movie:
    - 'relevant' items = movies that the users who liked the query also gave ≥ threshold
    - Measures ranking quality against this proxy ground truth.

    Returns dict with mean Precision@K, Recall@K, NDCG@K, and Coverage.
    """
    movies_with_enough = stats[stats["rating_count"] >= 50]["title"].tolist()
    query_movies = pd.Series(movies_with_enough).sample(
        min(sample_movies, len(movies_with_enough)), random_state=42
    ).tolist()

    p_scores, r_scores, ndcg_scores, all_recs = [], [], [], []

    for movie in query_movies:
        if movie not in matrix.columns:
            continue
        try:
            recs_df = recommender.recommend(movie, top_n=top_k * 3)
        except Exception:
            continue

        # Proxy ground truth: movies with high avg correlation among fans of query
        fans = matrix[movie].dropna()
        fans = fans[fans >= relevance_threshold].index   # users who liked it
        if len(fans) < 5:
            continue

        # Among those fans, movies they also rated highly
        fan_matrix = matrix.loc[fans]
        relevant_movies = set(
            fan_matrix.mean().dropna()
            .loc[lambda s: s >= relevance_threshold]
            .index
        ) - {movie}

        if not relevant_movies:
            continue

        recommended_titles = recs_df["title"].tolist()
        all_recs.append(recommended_titles[:top_k])

        p_scores.append(precision_at_k(recommended_titles, relevant_movies, top_k))
        r_scores.append(recall_at_k(recommended_titles,    relevant_movies, top_k))
        ndcg_scores.append(ndcg_at_k(recommended_titles,  relevant_movies, top_k))

    coverage = catalogue_coverage(all_recs, len(matrix.columns))

    return {
        f"Precision@{top_k}":  round(np.mean(p_scores),    4) if p_scores    else 0.0,
        f"Recall@{top_k}":     round(np.mean(r_scores),    4) if r_scores    else 0.0,
        f"NDCG@{top_k}":       round(np.mean(ndcg_scores), 4) if ndcg_scores else 0.0,
        "Coverage":            round(coverage,              4),
        "Queries evaluated":   len(p_scores),
    }
