"""
tests/test_recommender.py
==========================
Unit tests using synthetic data — no actual MovieLens files required.

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recommender import (
    CollaborativeFilteringRecommender,
    SVDRecommender,
    HybridRecommender,
)
from evaluate import precision_at_k, recall_at_k, ndcg_at_k, catalogue_coverage
from data_loader import popularity_stats


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N_USERS  = 150
N_MOVIES = 40

MOVIES = [f"Movie_{i}" for i in range(N_MOVIES)]
USERS  = list(range(1, N_USERS + 1))

# Dense synthetic rating matrix (some NaNs to mimic sparsity)
_raw = np.random.choice([1, 2, 3, 4, 5, np.nan], size=(N_USERS, N_MOVIES),
                        p=[0.05, 0.10, 0.15, 0.25, 0.25, 0.20])
MATRIX = pd.DataFrame(_raw, index=USERS, columns=MOVIES)

# Synthetic ratings DataFrame
_records = []
for u in USERS:
    for m_idx, m in enumerate(MOVIES):
        val = MATRIX.loc[u, m]
        if not np.isnan(val):
            _records.append({"user_id": u, "title": m, "rating": val})
DF_RATINGS = pd.DataFrame(_records)

# Popularity stats
STATS = (
    DF_RATINGS.groupby("title")["rating"]
    .agg(mean_rating="mean", rating_count="count")
    .reset_index()
)
STATS["bayesian_avg"] = STATS["mean_rating"]   # simplified for tests


# ─────────────────────────────────────────────────────────────────────────────
# CF Recommender tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCFRecommender:
    def setup_method(self):
        self.model = CollaborativeFilteringRecommender(min_ratings=1).fit(MATRIX, STATS)

    def test_returns_dataframe(self):
        recs = self.model.recommend("Movie_0", top_n=5)
        assert isinstance(recs, pd.DataFrame)

    def test_correct_columns(self):
        recs = self.model.recommend("Movie_0", top_n=5)
        assert "title" in recs.columns and "correlation" in recs.columns

    def test_no_self_in_results(self):
        recs = self.model.recommend("Movie_0", top_n=10)
        assert "Movie_0" not in recs["title"].values

    def test_top_n_respected(self):
        recs = self.model.recommend("Movie_0", top_n=5)
        assert len(recs) <= 5

    def test_sorted_descending(self):
        recs = self.model.recommend("Movie_0", top_n=10)
        assert recs["correlation"].is_monotonic_decreasing

    def test_invalid_movie_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.model.recommend("DoesNotExist", top_n=5)

    def test_not_fitted_raises(self):
        m = CollaborativeFilteringRecommender()
        with pytest.raises(RuntimeError):
            m.recommend("Movie_0")


# ─────────────────────────────────────────────────────────────────────────────
# SVD Recommender tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSVDRecommender:
    def setup_method(self):
        self.model = SVDRecommender(n_components=10, min_ratings=1).fit(MATRIX, STATS)

    def test_returns_dataframe(self):
        recs = self.model.recommend("Movie_1", top_n=5)
        assert isinstance(recs, pd.DataFrame)

    def test_no_self_in_results(self):
        recs = self.model.recommend("Movie_1", top_n=10)
        assert "Movie_1" not in recs["title"].values

    def test_scores_between_0_and_1(self):
        recs = self.model.recommend("Movie_1", top_n=10)
        assert (recs["svd_score"] >= -0.01).all() and (recs["svd_score"] <= 1.01).all()

    def test_explained_variance_positive(self):
        ev = self.model.explained_variance()
        assert 0 < ev <= 1.0

    def test_not_fitted_raises(self):
        m = SVDRecommender()
        with pytest.raises(RuntimeError):
            m.recommend("Movie_0")


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRecommender:
    def setup_method(self):
        self.model = HybridRecommender(min_ratings=1).fit(MATRIX, STATS)

    def test_returns_dataframe(self):
        recs = self.model.recommend("Movie_2", top_n=5)
        assert isinstance(recs, pd.DataFrame)

    def test_hybrid_score_column_exists(self):
        recs = self.model.recommend("Movie_2", top_n=5)
        assert "hybrid_score" in recs.columns

    def test_no_self_in_results(self):
        recs = self.model.recommend("Movie_2", top_n=10)
        assert "Movie_2" not in recs["title"].values


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metric tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_precision_perfect(self):
        rec = ["a", "b", "c"]
        rel = {"a", "b", "c"}
        assert precision_at_k(rec, rel, k=3) == 1.0

    def test_precision_zero(self):
        rec = ["a", "b", "c"]
        rel = {"d", "e"}
        assert precision_at_k(rec, rel, k=3) == 0.0

    def test_recall_perfect(self):
        rec = ["a", "b", "c", "d"]
        rel = {"a", "b"}
        assert recall_at_k(rec, rel, k=4) == 1.0

    def test_ndcg_perfect(self):
        rec = ["a", "b", "c"]
        rel = {"a", "b", "c"}
        assert abs(ndcg_at_k(rec, rel, k=3) - 1.0) < 1e-9

    def test_ndcg_no_hits(self):
        rec = ["x", "y"]
        rel = {"a", "b"}
        assert ndcg_at_k(rec, rel, k=2) == 0.0

    def test_coverage(self):
        all_recs = [["a", "b"], ["b", "c"], ["d"]]
        cov = catalogue_coverage(all_recs, catalogue_size=4)
        assert cov == 1.0   # a, b, c, d all covered
