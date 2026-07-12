# 💹 Dynamic Pricing Optimization Model

An end-to-end machine learning system that predicts product demand from price and
market context, and recommends the **revenue-optimal** or **profit-optimal** price
for an e-commerce product — complete with EDA, feature engineering, three tuned
regression models, price elasticity analysis, and an interactive Streamlit
dashboard.

Built as a production-style capstone project: modular code, reproducible
pipeline, tuned models, saved artifacts, and a polished web app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset](#-dataset)
4. [Project Architecture](#-project-architecture)
5. [Methodology](#-methodology)
6. [Results](#-results)
7. [Repository Structure](#-repository-structure)
8. [Installation & Setup](#-installation--setup)
9. [Running the Pipeline](#-running-the-pipeline)
10. [Running the Streamlit App](#-running-the-streamlit-app)
11. [Deployment (Streamlit Community Cloud)](#-deployment-streamlit-community-cloud)
12. [Model Details](#-model-details)
13. [Price Elasticity Analysis](#-price-elasticity-analysis)
14. [Future Improvements](#-future-improvements)
15. [Tech Stack](#-tech-stack)
16. [License](#-license)

---

## 🎯 Project Overview

Retailers lose revenue every day by pricing products **statically** — ignoring
competitor moves, seasonality, weather, promotions, and how sensitive
customers actually are to price. This project builds a **dynamic pricing
engine** that:

- Learns a demand function `units_sold = f(price, competitor_price, weather,
  season, day-of-week, promotions, ratings, ...)` using regression models.
- Simulates demand across a grid of candidate prices for any product context.
- Recommends the price that **maximizes revenue or profit**, using the
  learned demand curve.
- Quantifies **price elasticity** so pricing/marketing teams understand *why*
  a price is recommended, not just *what* it is.
- Ships as an interactive **Streamlit** app any stakeholder can use without
  touching code.

## 🧩 Business Problem

> *"What price should we set for Product X today, given today's competitor
> price, the weather, the season, and whether we're running a promotion —
> to maximize revenue (or profit) without guessing?"*

This is the exact problem behind dynamic pricing systems used by Amazon,
Uber, airlines, and hotel chains. This project reproduces that decision loop
end-to-end on realistic retail data.

## 📊 Dataset

**Source:** A custom-built, economically-consistent synthetic dataset
(`src/data_generation.py`), generated because public Kaggle "dynamic
pricing" datasets do not simultaneously provide daily competitor prices,
weather, promotions, seasonality **and** a valid ground-truth demand
response to price in one place — the combination this project's
requirements call for.

The generator is **not** random noise: it implements a constant-elasticity
demand curve (`log(units) = a + elasticity·log(price/base_price) + seasonal
+ weather + promo + competitor + trend + noise`) per product category, with
category-specific elasticities calibrated to realistic ranges (e.g. Fashion
≈ -2.4 "elastic", Groceries ≈ -1.2 "inelastic"), so every EDA insight,
elasticity number, and model coefficient in this project reflects genuine
microeconomic behaviour, fully reproducible via a fixed random seed. This
design choice is documented in detail at the top of `src/data_generation.py`
and in the [Project Report](reports/Project_Report.docx).

**Scale:** 80 products across 8 categories × ~1.5 years of daily records ≈
**27,000 rows**, 23 raw columns (tunable — increase `N_PRODUCTS_PER_CATEGORY`
and `N_DAYS` in `src/data_generation.py` for a larger dataset on more
powerful hardware).

| Column | Description |
|---|---|
| `date`, `product_id`, `category`, `brand_tier` | Identifiers |
| `base_price`, `cost_price`, `price`, `competitor_price` | Pricing |
| `discount_pct`, `promotion_flag` | Promotions |
| `day_of_week`, `month`, `is_weekend`, `is_holiday`, `season` | Time |
| `temperature_c`, `weather_condition` | Weather |
| `rating`, `review_count`, `inventory_level`, `website_traffic` | Product/demand signals |
| `units_sold` | **Target** |

> Swapping in a real Kaggle dataset (e.g. *Online Retail II*, *Retail Price
> Optimization*) only requires re-pointing `src/preprocessing.py` at the new
> raw file and adjusting `NUMERIC_FEATURE_COLS` in `src/feature_engineering.py`
> — the rest of the pipeline is dataset-agnostic.

## 🏗 Project Architecture

```
Raw Data → Preprocessing → Feature Engineering → EDA
                                   │
                                   ▼
        ┌──────────────────────────────────────────┐
        │   Model Training (GridSearchCV)           │
        │   Linear Regression | Random Forest |     │
        │   XGBoost                                  │
        └──────────────────────────────────────────┘
                                   │
                                   ▼
              Evaluation (RMSE / MAE / R² / MAPE)
                                   │
                                   ▼
         Price Elasticity & Optimal Price Simulation
                                   │
                                   ▼
                 Joblib Model Artifacts (models/)
                                   │
                                   ▼
                    Streamlit Web Application
```

## 🔬 Methodology

1. **Data Generation** (`src/data_generation.py`) — simulate ~1.5 years of
   daily transactions across 80 products / 8 categories (tunable up on
   multi-core hardware).
2. **Preprocessing** (`src/preprocessing.py`) — dedupe, validate ranges,
   impute missing values, IQR-clip outliers.
3. **Feature Engineering** (`src/feature_engineering.py`) — 40+ engineered
   features: cyclical time encodings, price-vs-competitor ratios/gaps,
   markup %, 7/30-day rolling demand, lagged price, weather buckets,
   one-hot categoricals.
4. **EDA** (`src/eda.py`) — 8 charts: price distributions, demand vs price,
   seasonality trend, weather effect, promotion lift, correlation heatmap,
   competitor-price comparison, day-of-week pattern. Saved to `/assets`.
5. **Modeling** (`src/train_models.py`) — time-based train/test split (last
   20% of each product's history withheld, to avoid leakage from
   lag/rolling features), three models tuned with `GridSearchCV` (3-fold,
   RMSE-scored):
   - Linear Regression (baseline, standardized features)
   - Random Forest Regressor
   - XGBoost Regressor
6. **Evaluation** — RMSE, MAE, R², MAPE on held-out (future) data.
7. **Price Elasticity & Optimization** (`src/elasticity_analysis.py`) — for
   each category, simulate demand across a price grid (±40–60% of base
   price) using the best model, compute arc price elasticity, and find the
   revenue-/profit-maximizing price.
8. **Persistence** — every model, scaler, and metadata bundle saved with
   `joblib` to `/models`.
9. **Deployment** — modern Streamlit dashboard (`app.py`) for interactive,
   non-technical use.

## 📈 Results

Actual results from a full pipeline run on this repository's generated
dataset (27,009 rows, time-based 80/20 split, 1-core CI environment —
see [Model Details](#-model-details) for how to widen the search on a
multi-core machine):

| Model | RMSE ↓ | MAE ↓ | R² ↑ | MAPE ↓ |
|---|---|---|---|---|
| Linear Regression | 6.711 | 5.081 | 0.885 | 10.54% |
| Random Forest (tuned) | 6.554 | 5.130 | 0.891 | 10.57% |
| **XGBoost (tuned)** ⭐ | **6.444** | **5.018** | **0.894** | **10.29%** |

**XGBoost** is automatically selected as the production model (lowest RMSE)
and is pre-loaded by the Streamlit app, with the option to switch models
interactively. Full numbers regenerate at `reports/model_metrics.json` and
category-level elasticity at `reports/price_elasticity_by_category.csv`
every time you re-run the pipeline.

**Price elasticity by category** (excerpt — see full table in the app):

| Category | Elasticity | Interpretation | Revenue-Optimal Price |
|---|---|---|---|
| Electronics | -1.09 | Elastic | $381.78 (from $279.35 median) |
| Fashion | -1.10 | Elastic | $83.54 (from $60.62 median) |
| Groceries | -1.19 | Elastic | $21.94 (from $15.64 median) |
| Beauty | -0.43 | Inelastic | $54.05 (from $37.20 median) |
| Books | -0.60 | Inelastic | $29.39 (from $21.61 median) |

## 📁 Repository Structure

```
dynamic_pricing_project/
├── app.py                          # Streamlit web application
├── requirements.txt
├── README.md
├── data/
│   ├── raw/dynamic_pricing_raw.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── engineered_data.csv
├── src/
│   ├── data_generation.py          # Synthetic dataset generator
│   ├── preprocessing.py            # Cleaning & validation
│   ├── feature_engineering.py      # Feature pipeline
│   ├── eda.py                      # Exploratory Data Analysis
│   ├── train_models.py             # Training + GridSearchCV + evaluation
│   └── elasticity_analysis.py      # Elasticity & optimal price simulation
├── models/                         # Saved joblib artifacts
│   ├── linear_regression_model.pkl
│   ├── linear_scaler.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.pkl
├── reports/
│   ├── model_metrics.json
│   ├── feature_importance_RandomForest.csv
│   ├── feature_importance_XGBoost.csv
│   ├── linear_regression_coefficients.csv
│   ├── price_elasticity_by_category.csv
│   ├── Project_Report.docx
│   └── Dynamic_Pricing_Presentation.pptx
├── assets/                         # EDA charts (PNG)
└── notebooks/                      # (optional) exploratory notebook
```

> **Note:** `models/random_forest_model.pkl` is ~32MB (well under GitHub's
> 100MB file limit, no Git LFS needed). If you widen the hyperparameter grid
> in `src/train_models.py`, re-check the file size before committing.

## 💻 Installation & Setup

```bash
git clone https://github.com/<your-username>/dynamic-pricing-optimization.git
cd dynamic-pricing-optimization

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ Running the Pipeline

Run each stage in order (each writes its output to `data/` / `models/` /
`reports/` for the next stage):

```bash
python src/data_generation.py        # -> data/raw/dynamic_pricing_raw.csv
python src/preprocessing.py          # -> data/processed/cleaned_data.csv
python src/feature_engineering.py    # -> data/processed/engineered_data.csv
python src/eda.py                    # -> assets/*.png
python src/train_models.py           # -> models/*.pkl, reports/model_metrics.json
python src/elasticity_analysis.py    # -> reports/price_elasticity_by_category.csv
```

> Total runtime: ~5–10 minutes on a standard laptop (GridSearchCV is the
> slowest step).

## 🖥 Running the Streamlit App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

**In the app you can:**
- Select a product category, brand tier, and pricing context
- Set competitor price, weather, season, promotions, ratings, inventory
- Choose which trained model powers the recommendation
- Optimize for **Revenue** or **Profit**
- View the simulated price–demand curve, elasticity, model comparison, and
  feature importance — all interactively, with no code required

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"** → select your repo, branch (`main`), and set
   **Main file path** to `app.py`.
4. Under **Advanced settings**, set the Python version to match your
   `requirements.txt` (3.10+ recommended).
5. Click **Deploy**. Streamlit Cloud will `pip install -r requirements.txt`
   and launch `app.py` automatically.
6. **Important:** commit the `models/`, `data/processed/`, and `reports/`
   directories (or run the pipeline as a build step) so the deployed app has
   the trained artifacts available — the app will show a clear error message
   and instructions if they're missing.

This repo already includes `runtime.txt` (pins Python 3.11) and
`.streamlit/config.toml` (custom navy/gold theme) for a smoother deploy.

## 🤖 Model Details

| Model | Tuning | Key hyperparameters searched |
|---|---|---|
| Linear Regression | — (baseline) | Standardized features |
| Random Forest | `GridSearchCV`, 2-fold | `n_estimators` [80,150], `max_depth` [12,18] (`min_samples_leaf=2` fixed) |
| XGBoost | `GridSearchCV`, 2-fold | `n_estimators` [120,220], `max_depth` [4,6] (`learning_rate=0.08` fixed) |

> The grids above are intentionally compact to train quickly on a single
> CPU core (~2.5 minutes total). On a multi-core machine, widen them freely
> in `src/train_models.py` — e.g. `n_estimators` up to 500, deeper trees,
> more folds — for a marginal accuracy gain at a longer runtime cost.

Model selection is automatic: the model with the lowest test-set RMSE is
marked `best_model` in `models/model_metadata.pkl` and pre-selected in the
Streamlit app (users can still switch manually).

## 📐 Price Elasticity Analysis

For each category, `src/elasticity_analysis.py`:
1. Builds a representative context row (median feature values).
2. Simulates predicted demand across a price grid.
3. Computes **arc elasticity**: `E = %Δ Quantity / %Δ Price`.
4. Flags the category **Elastic** (|E| > 1, demand highly price-sensitive)
   or **Inelastic** (|E| < 1).
5. Finds the **revenue-optimal** and **profit-optimal** price on the
   simulated curve.

Results are saved to `reports/price_elasticity_by_category.csv` and
rendered live in the Streamlit app's "Elasticity Insight" tab.

## 🚀 Future Improvements

- Swap in real transactional data (e.g. from a retailer's data warehouse)
  once available, keeping the same pipeline.
- Add a causal / uplift model (e.g. double machine learning) to separate
  correlation from the true causal effect of price on demand.
- Add competitor-price forecasting (currently competitor price is an input,
  not predicted).
- Multi-armed bandit / reinforcement learning layer for continuous online
  price optimization with real-time feedback.
- Add confidence intervals around the recommended price using quantile
  regression or bootstrapped predictions.

## 🛠 Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `joblib` ·
`Matplotlib` / `Seaborn` · `Plotly` · `Streamlit`

## 📄 License

This project is released under the MIT License — see [`LICENSE`](LICENSE).

---

*Built as a Machine Learning capstone project. Contributions and forks are
welcome — open an issue or pull request.*
