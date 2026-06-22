🎬 Movie Recommender System
> **End-to-end collaborative filtering pipeline — Memory-based CF · SVD Matrix Factorisation · Hybrid Ensemble**
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![Tests](https://img.shields.io/badge/Tests-21%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
---
Overview
A production-structured movie recommendation system built on the MovieLens 100K dataset (100,000 ratings from 943 users across 1,682 movies). The project implements and compares three complementary recommendation strategies, evaluated with standard information-retrieval metrics.
Model	Approach	Key Technique
CF	Memory-based collaborative filtering	Item-item Pearson correlation
SVD	Model-based collaborative filtering	Truncated SVD (latent factors)
Hybrid	Weighted ensemble	Normalised score blending
---
Dataset
Download from Kaggle: MovieLens 100K Dataset
Place the following files in the `data/` folder:
```
data/
├── u.data    ← ratings  (user_id, item_id, rating, timestamp) [TSV]
├── u.item    ← movies   (item_id, title, release_date, genres…) [pipe-sep]
└── u.user    ← users    (user_id, age, gender, occupation, zip) [pipe-sep]
```
Why this dataset?
100K ratings is small enough to run locally but large enough to produce meaningful statistical results.
Rich metadata (genres, user demographics, timestamps) enables multiple analysis angles.
Industry-standard benchmark — every recommendation paper cites it.
---
Project Structure
```
movie-recommender/
│
├── data/                   ← Place MovieLens files here (not committed)
│
├── src/
│   ├── data_loader.py      ← Data ingestion, merging, rating matrix, popularity stats
│   ├── recommender.py      ← CF, SVD, and Hybrid recommender classes
│   ├── evaluate.py         ← Precision@K, Recall@K, NDCG@K, Coverage metrics
│   ├── eda.py              ← All visualisation functions
│   └── train.py            ← End-to-end training pipeline
│
├── models/                 ← Serialised models (auto-created by train.py)
│   ├── cf_model.pkl
│   ├── svd_model.pkl
│   └── hybrid_model.pkl
│
├── outputs/                ← EDA plots + evaluation CSV (auto-created)
│   ├── eda/
│   │   ├── 01_rating_distribution.png
│   │   ├── 02_top_movies.png
│   │   ├── 03_rating_vs_popularity.png
│   │   ├── 04_genre_distribution.png
│   │   ├── 05_user_activity.png
│   │   ├── 06_sparsity_heatmap.png
│   │   └── 07_svd_variance.png
│   └── evaluation_results.csv
│
├── tests/
│   └── test_recommender.py ← 21 unit tests (synthetic data, no dataset needed)
│
├── app.py                  ← Streamlit interactive web app
├── requirements.txt
└── README.md
```
---
Installation
```bash
git clone https://github.com/shiv-speccc/movie-recommender.git
cd movie-recommender

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```
---
Usage
1. Train all models
```bash
python src/train.py
```
This will:
Load and merge all three data files
Generate 7 EDA plots → `outputs/eda/`
Train CF, SVD, and Hybrid models → `models/`
Run evaluation on all three models → `outputs/evaluation_results.csv`
Print sanity-check recommendations for Star Wars (1977)
2. Launch the Streamlit app
```bash
streamlit run app.py
```
3. Use the API directly
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
Methodology
Memory-based CF (Pearson Correlation)
For a query movie q, we compute the Pearson correlation between q's rating vector and every other movie's rating vector across users who rated both:
```
corr(q, m) = Pearson(ratings_q, ratings_m)  over shared users
```
A minimum-ratings filter (≥ 50 ratings) prevents spurious high correlations on movies with very few common raters — a known cold-start artefact.
Model-based CF (Truncated SVD)
The rating matrix R (943 × 1682) is:
Mean-centered imputed: NaNs replaced with each user's mean rating.
Decomposed: `R ≈ U Σ Vᵀ` using Truncated SVD with k = 50 latent factors.
Movie vectors (rows of Vᵀ) are L2-normalised; similarity is cosine distance in latent space.
With 50 factors the model captures roughly 35–40% of total variance — sufficient to surface genre-level and quality-level similarity.
Bayesian Average Rating
To rank movies fairly, raw mean ratings are shrunk toward the global mean using:
```
bayesian_avg = (C × global_mean + n × mean) / (C + n)
```
where C is the 70th-percentile rating count. This prevents a movie with 2 five-star ratings from outranking a movie with 500 four-star ratings.
Hybrid Ensemble
Both models' scores are independently min-max normalised to [0, 1], then linearly blended:
```
hybrid_score = w_cf × norm(corr) + w_svd × norm(cosine_sim)
```
Default weights: `w_cf = w_svd = 0.5`. Both weights are configurable.
---
Evaluation
Evaluation uses a leave-one-out proxy approach: for each sampled query movie, we identify its "fans" (users who rated it ≥ 4) and treat movies those fans also rated highly as the ground-truth relevant set.
Metric	Description
Precision@K	Fraction of top-K recommendations that are relevant
Recall@K	Fraction of relevant items captured in top-K
NDCG@K	Ranking quality (penalises relevant items ranked lower)
Coverage	Fraction of the catalogue the model can recommend
Run evaluation independently:
```python
from src.evaluate import evaluate_cf_model
metrics = evaluate_cf_model(hybrid_model, matrix, stats, sample_movies=50, top_k=10)
print(metrics)
```
---
Testing
```bash
pytest tests/ -v
```
21 unit tests covering CF, SVD, Hybrid, and all evaluation metrics. Tests use synthetic data — no dataset download required to run the test suite.
---
Interview Discussion Points
This project was deliberately designed to surface real ML engineering decisions:
Popularity bias correction — Bayesian averaging vs. raw mean; why the 70th-percentile confidence prior.
Sparsity handling — 93–94% of the matrix is empty; mean-imputation vs. matrix factorisation vs. implicit feedback models.
Cold-start problem — why the minimum-ratings filter exists; what happens without it.
SVD rank selection — elbow analysis on the explained-variance curve; overfitting at high k.
Pearson vs. cosine similarity — when each is preferable for item-item CF.
Hybrid weighting — how to tune cf_weight/svd_weight; cross-validation approach.
Offline vs. online evaluation — why NDCG@K is better than raw accuracy for recommenders.
---
Limitations and Future Work
Implicit feedback (clicks, watch-time) would produce a richer signal than explicit 1–5 ratings.
Temporal drift — user preferences change; time-weighted ratings or session-based models (GRU4Rec) address this.
Content-based features — genre, cast, director embeddings could power a true content-CF hybrid.
Neural CF — NeuMF or LightGCN would outperform linear SVD for large catalogues.
A/B testing framework — online evaluation via interleaving or bandit algorithms.
---
License
MIT License — see LICENSE for details.
