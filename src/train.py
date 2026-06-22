"""
train.py
========
End-to-end training pipeline.

Run:
    python src/train.py

Outputs
-------
models/cf_model.pkl
models/svd_model.pkl
models/hybrid_model.pkl
outputs/evaluation_results.csv
outputs/eda/  (all EDA plots as .png)
"""

import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

# Make imports work whether we run from project root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader  import load_and_merge, rating_matrix, popularity_stats
from recommender  import CollaborativeFilteringRecommender, SVDRecommender, HybridRecommender
from evaluate     import evaluate_cf_model
import eda as eda_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
EDA_DIR    = OUTPUT_DIR / "eda"
OUTPUT_DIR.mkdir(exist_ok=True)
EDA_DIR.mkdir(exist_ok=True)


def save_fig(fig: plt.Figure, name: str):
    path = EDA_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved plot → {path.relative_to(ROOT)}")


def run():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Loading data …")
    df = load_and_merge()
    log.info(f"  {len(df):,} ratings  |  {df['user_id'].nunique():,} users  |  {df['item_id'].nunique():,} movies")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info("Computing features …")
    stats  = popularity_stats(df)
    matrix = rating_matrix(df)

    sparsity = 1 - matrix.notna().mean().mean()
    log.info(f"  Matrix: {matrix.shape[0]} users × {matrix.shape[1]} movies  |  sparsity {sparsity:.2%}")

    # ── 3. EDA plots ──────────────────────────────────────────────────────────
    log.info("Generating EDA plots …")
    save_fig(eda_module.plot_rating_distribution(df),        "01_rating_distribution")
    save_fig(eda_module.plot_top_movies(stats),              "02_top_movies")
    save_fig(eda_module.plot_rating_vs_popularity(stats),    "03_rating_vs_popularity")
    save_fig(eda_module.plot_genre_distribution(df),         "04_genre_distribution")
    save_fig(eda_module.plot_user_activity(df),              "05_user_activity")
    save_fig(eda_module.plot_sparsity_heatmap(matrix),       "06_sparsity_heatmap")

    # ── 4. Train models ───────────────────────────────────────────────────────
    log.info("Training CollaborativeFilteringRecommender …")
    cf_model = CollaborativeFilteringRecommender(min_ratings=50)
    cf_model.fit(matrix, stats)
    cf_model.save()

    log.info("Training SVDRecommender (50 latent factors) …")
    svd_model = SVDRecommender(n_components=50, min_ratings=50)
    svd_model.fit(matrix, stats)
    svd_model.save()
    save_fig(eda_module.plot_svd_variance(svd_model), "07_svd_variance")
    log.info(f"  SVD explains {svd_model.explained_variance():.2%} of variance with 50 factors")

    log.info("Training HybridRecommender …")
    hybrid_model = HybridRecommender(cf_weight=0.5, svd_weight=0.5, min_ratings=50)
    hybrid_model.fit(matrix, stats)
    hybrid_model.save()

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    log.info("Evaluating all models (leave-one-out, 50 sample queries) …")

    results = {}
    for name, model in [("CF", cf_model), ("SVD", svd_model), ("Hybrid", hybrid_model)]:
        log.info(f"  Evaluating {name} …")
        metrics = evaluate_cf_model(model, matrix, stats, sample_movies=50, top_k=10)
        results[name] = metrics
        log.info(f"    {metrics}")

    eval_df = pd.DataFrame(results).T
    eval_path = OUTPUT_DIR / "evaluation_results.csv"
    eval_df.to_csv(eval_path)
    log.info(f"Evaluation results saved → {eval_path.relative_to(ROOT)}")

    # ── 6. Quick sanity-check recommendations ─────────────────────────────────
    log.info("Sanity-check recommendations for 'Star Wars (1977)' …")
    sample_query = "Star Wars (1977)"
    for name, model in [("CF", cf_model), ("SVD", svd_model), ("Hybrid", hybrid_model)]:
        try:
            recs = model.recommend(sample_query, top_n=5)
            log.info(f"\n  [{name}] Top-5 for '{sample_query}':\n{recs.to_string()}\n")
        except Exception as e:
            log.warning(f"  [{name}] failed: {e}")

    log.info("✓ Training pipeline complete.")
    return cf_model, svd_model, hybrid_model, matrix, stats


if __name__ == "__main__":
    run()
