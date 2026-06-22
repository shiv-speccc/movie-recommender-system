"""
recommender.py
==============
Three recommendation strategies:

1. CollaborativeFilteringRecommender   – item-item Pearson correlation (memory-based CF)
2. SVDRecommender                      – matrix factorisation via TruncatedSVD  (model-based CF)
3. HybridRecommender                   – blend of both, weighted by confidence
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MIN_COMMON_RATINGS = 30   # minimum overlapping users for Pearson correlation
MIN_RATING_COUNT   = 50   # minimum ratings a movie must have to appear in results


# ─────────────────────────────────────────────────────────────────────────────
# 1. Memory-based Collaborative Filtering (Item-Item Pearson)
# ─────────────────────────────────────────────────────────────────────────────

class CollaborativeFilteringRecommender:
    """
    Item-item collaborative filter using Pearson correlation.

    Methodology
    -----------
    * Builds a user × movie rating matrix.
    * For a query movie, correlates its rating vector against every other
      movie's rating vector across users who rated both.
    * Filters out movies with fewer than `min_ratings` total ratings to
      avoid spurious high correlations on tiny samples.

    Parameters
    ----------
    min_ratings : int
        Minimum number of ratings a movie must have to be included in results.
    """

    def __init__(self, min_ratings: int = MIN_RATING_COUNT):
        self.min_ratings = min_ratings
        self._matrix: pd.DataFrame | None = None
        self._stats:  pd.DataFrame | None = None

    def fit(self, matrix: pd.DataFrame, stats: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        """
        Parameters
        ----------
        matrix : pd.DataFrame
            User-item rating matrix (rows = users, cols = movie titles).
        stats : pd.DataFrame
            Movie statistics with columns [title, rating_count].
        """
        self._matrix = matrix
        self._stats  = stats
        return self

    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return top-N movies most similar to `movie_title`.

        Returns
        -------
        pd.DataFrame with columns [title, correlation, rating_count]
        sorted by correlation descending.
        """
        if self._matrix is None:
            raise RuntimeError("Call .fit() before .recommend()")

        if movie_title not in self._matrix.columns:
            raise ValueError(f"'{movie_title}' not found in the rating matrix. "
                             "Check spelling or try a different title.")

        query_vec = self._matrix[movie_title]
        corr_series = self._matrix.corrwith(query_vec)

        result = (
            pd.DataFrame({"title": corr_series.index, "correlation": corr_series.values})
            .dropna()
            .merge(self._stats[["title", "rating_count"]], on="title", how="left")
            .query("rating_count >= @self.min_ratings")
            .query("title != @movie_title")
            .sort_values("correlation", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        result.index += 1
        return result

    def save(self, name: str = "cf_model"):
        joblib.dump(self, MODELS_DIR / f"{name}.pkl")

    @classmethod
    def load(cls, name: str = "cf_model") -> "CollaborativeFilteringRecommender":
        return joblib.load(MODELS_DIR / f"{name}.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model-based CF via Matrix Factorisation (TruncatedSVD)
# ─────────────────────────────────────────────────────────────────────────────

class SVDRecommender:
    """
    Latent factor model using Truncated SVD (a.k.a. LSA / collaborative LSA).

    Methodology
    -----------
    * Fills NaN ratings with each user's mean rating (mean-centered imputation).
    * Decomposes the matrix into k latent factors with TruncatedSVD.
    * Computes cosine similarity between movie vectors in latent space.
    * Recommendations are movies with highest cosine similarity to the query.

    Parameters
    ----------
    n_components : int
        Number of latent factors (SVD rank). Typical range: 20–100.
    min_ratings : int
        Minimum number of ratings a movie must have to be included.
    """

    def __init__(self, n_components: int = 50, min_ratings: int = MIN_RATING_COUNT):
        self.n_components = n_components
        self.min_ratings  = min_ratings
        self._svd:          TruncatedSVD | None = None
        self._movie_vecs:   np.ndarray   | None = None
        self._movie_titles: list[str]    | None = None
        self._stats:        pd.DataFrame | None = None

    def fit(self, matrix: pd.DataFrame, stats: pd.DataFrame) -> "SVDRecommender":
        self._stats        = stats
        self._movie_titles = list(matrix.columns)

        # Mean-centered fill: replace NaN with user's mean rating
        filled = matrix.T.fillna(matrix.mean(axis=1)).T.fillna(3.0)

        self._svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        # Shape after fit: (n_movies, n_components)
        self._movie_vecs = self._svd.fit_transform(filled.T)
        self._movie_vecs = normalize(self._movie_vecs)
        return self

    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        if self._movie_vecs is None:
            raise RuntimeError("Call .fit() before .recommend()")

        if movie_title not in self._movie_titles:
            raise ValueError(f"'{movie_title}' not found. Check spelling.")

        idx        = self._movie_titles.index(movie_title)
        query_vec  = self._movie_vecs[idx].reshape(1, -1)
        sims       = cosine_similarity(query_vec, self._movie_vecs).flatten()

        result_df = pd.DataFrame({
            "title":       self._movie_titles,
            "svd_score":   sims,
        })
        result = (
            result_df
            .merge(self._stats[["title", "rating_count"]], on="title", how="left")
            .query("rating_count >= @self.min_ratings")
            .query("title != @movie_title")
            .sort_values("svd_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        result.index += 1
        return result

    def explained_variance(self) -> float:
        if self._svd is None:
            raise RuntimeError("Model not fitted.")
        return float(self._svd.explained_variance_ratio_.sum())

    def save(self, name: str = "svd_model"):
        joblib.dump(self, MODELS_DIR / f"{name}.pkl")

    @classmethod
    def load(cls, name: str = "svd_model") -> "SVDRecommender":
        return joblib.load(MODELS_DIR / f"{name}.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Recommender
# ─────────────────────────────────────────────────────────────────────────────

class HybridRecommender:
    """
    Weighted ensemble of CollaborativeFilteringRecommender and SVDRecommender.

    Parameters
    ----------
    cf_weight  : float, weight for Pearson-CF correlation score (0–1)
    svd_weight : float, weight for SVD cosine-similarity score   (0–1)
    min_ratings: int, minimum rating count filter applied before blending
    """

    def __init__(
        self,
        cf_weight:   float = 0.5,
        svd_weight:  float = 0.5,
        min_ratings: int   = MIN_RATING_COUNT,
    ):
        self.cf_weight   = cf_weight
        self.svd_weight  = svd_weight
        self.min_ratings = min_ratings
        self._cf  = CollaborativeFilteringRecommender(min_ratings=min_ratings)
        self._svd = SVDRecommender(min_ratings=min_ratings)

    def fit(self, matrix: pd.DataFrame, stats: pd.DataFrame) -> "HybridRecommender":
        self._cf.fit(matrix, stats)
        self._svd.fit(matrix, stats)
        return self

    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        cf_rec  = self._cf.recommend(movie_title,  top_n=200)
        svd_rec = self._svd.recommend(movie_title, top_n=200)

        # Normalise both score columns to [0, 1] before blending
        def minmax(s: pd.Series) -> pd.Series:
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo + 1e-9)

        cf_rec  = cf_rec.rename(columns={"correlation": "score"})
        svd_rec = svd_rec.rename(columns={"svd_score":  "score"})

        cf_rec["norm_score"]  = minmax(cf_rec["score"])  * self.cf_weight
        svd_rec["norm_score"] = minmax(svd_rec["score"]) * self.svd_weight

        merged = (
            cf_rec[["title", "norm_score", "rating_count"]]
            .merge(
                svd_rec[["title", "norm_score"]],
                on="title", how="outer", suffixes=("_cf", "_svd")
            )
            .fillna(0)
        )
        merged["hybrid_score"] = merged["norm_score_cf"] + merged["norm_score_svd"]

        result = (
            merged
            .sort_values("hybrid_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        result.index += 1
        return result[["title", "hybrid_score", "rating_count"]]

    def save(self, name: str = "hybrid_model"):
        joblib.dump(self, MODELS_DIR / f"{name}.pkl")

    @classmethod
    def load(cls, name: str = "hybrid_model") -> "HybridRecommender":
        return joblib.load(MODELS_DIR / f"{name}.pkl")
