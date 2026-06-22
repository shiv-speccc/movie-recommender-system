# 🎬 Movie Recommender System

> End-to-end collaborative filtering pipeline — Memory-based CF · SVD Matrix Factorisation · Hybrid Ensemble

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![Tests](https://img.shields.io/badge/Tests-21%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

🚀 **Live Demo:** [movie-recommender-system-5djo4astjuq3b5z7idmbzx.streamlit.app](https://movie-recommender-system-5djo4astjuq3b5z7idmbzx.streamlit.app)

---

## Overview

A production-structured movie recommendation system built on the **MovieLens 100K** dataset — 100,000 ratings from 943 users across 1,682 movies. The project implements and compares three complementary recommendation strategies, evaluated with standard information-retrieval metrics.

| Model | Approach | Key Technique |
|-------|----------|---------------|
| CF | Memory-based collaborative filtering | Item-item Pearson correlation |
| SVD | Model-based collaborative filtering | Truncated SVD (latent factors) |
| Hybrid | Weighted ensemble | Normalised score blending |

---

## Dataset

Download: **[MovieLens 100K — GroupLens](https://files.grouplens.org/datasets/movielens/ml-100k.zip)**

Place the following files in the `data/` folder:

```
data/
├── u.data    ← ratings  (user_id, item_id, rating, timestamp)
├── u.item    ← movies   (item_id, title, release_date, genres)
└── u.user    ← users    (user_id, age, gender, occupation, zip)
```

---

## Project Structure

```
movie-recommender/
├── data/                        ← MovieLens dataset files
├── src/
│   ├── data_loader.py           ← Ingestion, merging, rating matrix, Bayesian stats
│   ├── recommender.py           ← CF, SVD, and Hybrid recommender classes
│   ├── evaluate.py              ← Precision@K, Recall@K, NDCG@K, Coverage
│   ├── eda.py                   ← Visualisation functions
│   └── train.py                 ← End-to-end training pipeline
├── models/                      ← Serialised model artifacts (.pkl)
├── outputs/
│   └── eda/                     ← 7 EDA plots (auto-generated)
├── tests/
│   └── test_recommender.py      ← 21 unit tests (synthetic data)
├── app.py                       ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/shiv-speccc/movie-recommender-system.git
cd movie-recommender-system

conda create -n movie-rec python=3.11 -y
conda activate movie-rec

pip install -r requirements.txt
```

---

## Usage

**Train all models**

```bash
python src/train.py
```

This loads the dataset, generates 7 EDA plots, trains all three models, runs evaluation, and prints sanity-check recommendations.

**Launch the app**

```bash
streamlit run app.py
```

**Use directly in Python**

```python
from src.data_loader import load_and_merge, rating_matrix, popularity_stats
from src.recommender import HybridRecommender

df     = load_and_merge()
stats  = popularity_stats(df)
matrix = rating_matrix(df)

model = HybridRecommender(cf_weight=0.5, svd_weight=0.5).fit(matrix, stats)
print(model.recommend("Toy Story (1995)", top_n=5))
```

---

## Methodology

### Memory-based CF (Pearson Correlation)

For a query movie *q*, Pearson correlation is computed between *q*'s rating vector and every other movie's rating vector across shared users. A minimum-ratings filter (≥ 50) prevents spurious correlations on rarely-rated movies.

### Model-based CF (Truncated SVD)

The rating matrix (943 × 1682) is mean-centered imputed, then decomposed into 50 latent factors via TruncatedSVD. Movie vectors are L2-normalised; recommendations are ranked by cosine similarity in latent space. 50 factors capture ~35–40% of total variance.

### Bayesian Average Rating

Raw mean ratings are shrunk toward the global mean to correct for low-count movies:

```
bayesian_avg = (C × global_mean + n × mean) / (C + n)
```

where *C* is the 70th-percentile rating count.

### Hybrid Ensemble

Both models' scores are min-max normalised to [0, 1] then linearly blended:

```
hybrid_score = w_cf × norm(correlation) + w_svd × norm(cosine_similarity)
```

Default: `w_cf = w_svd = 0.5` (configurable).

---

## Evaluation

Evaluation uses a **leave-one-out proxy**: fans of each query movie serve as the ground truth to measure ranking quality.

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items captured in top-K |
| NDCG@K | Ranking quality — penalises relevant items ranked lower |
| Coverage | Fraction of the catalogue the model can recommend |

```bash
pytest tests/ -v   # 21 tests, no dataset required
```

---

## Interview Discussion Points

- **Popularity bias** — Bayesian averaging vs. raw mean; the 70th-percentile confidence prior
- **Sparsity** — 93.6% of the matrix is empty; mean-imputation vs. matrix factorisation
- **Cold-start** — why the minimum-ratings filter exists and what breaks without it
- **SVD rank selection** — elbow analysis on the explained-variance curve
- **Pearson vs. cosine** — when each is preferable for item-item CF
- **Offline vs. online evaluation** — why NDCG@K is better than raw accuracy

---

## Future Work

- Implicit feedback (clicks, watch-time) instead of explicit ratings
- Time-weighted ratings to handle temporal preference drift
- Content-based features — genre, cast, director embeddings
- Neural CF — NeuMF or LightGCN for large-scale catalogues
- A/B testing framework for online evaluation

---

## License

MIT License — see [LICENSE](LICENSE) for details.
