# 🚗 Ensemble Learning on Sri Lankan Car Prices (ikman.lk)

Hands-on project — **Ensemble Learning Dashboard** (Bagging · Boosting · Stacking · Hybrid).

## What is in this folder

| File | Purpose |
|---|---|
| `ensemble_cars_sri_lanka.ipynb` | **Main deliverable** – data cleaning, EDA, 12 models, comparison, interpretation, saving artefacts (already executed) |
| `app.py` | Streamlit dashboard (price predictor · model comparison · "why it works" · market explorer) |
| `scraper/scrape_ikman.py` | Polite, resumable scraper for public ikman.lk car ads |
| `data/ikman_cars_raw.jsonl` | Raw scraped ads (654) · `data/cars_clean.csv` cleaned (633) |
| `models/` | Trained pipelines (`*.joblib`) + metrics / importance / predictions / meta |
| `VIDEO_SCRIPT.md` | 3-minute voice-over script |

## Run it

```bash
pip install -r requirements.txt

# 1) dashboard (uses the models already in models/)
python3 -m streamlit run app.py

# 2) re-run the notebook end-to-end (~3 min)
jupyter nbconvert --to notebook --execute --inplace ensemble_cars_sri_lanka.ipynb

# 3) (optional) re-scrape — the site is slow (~0.3 req/s), be patient and polite
python3 scraper/scrape_ikman.py --pages 400 --page-step 8 --workers 3 --limit 1500
```

## Data
* **Source:** [ikman.lk](https://ikman.lk) car ads, scraped 23 Nov 2026 – Sri Lanka's biggest classifieds site (a Sri Lankan context dataset, as required).
* Only public pages that `robots.txt` allows, low request rate, random sample across the catalogue (every 8th list page, shuffled ads).
* 654 ads scraped → **633** after cleaning (vintage < 1985, future-dated and absurd prices removed). Target: asking price in LKR, modelled as `log(price)`.
* Features: brand, model, year → age, mileage, engine cc, fuel, transmission, body, condition, district, dealer flag.

## Models compared (5-fold CV on 506 train ads, 127 held-out test ads)

| Family | Models |
|---|---|
| Baselines | Ridge, single Decision Tree |
| **Bagging** | Bagged Trees, Random Forest, Extra Trees |
| **Boosting** | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Neural | ANN (MLP) |
| **Hybrid** | Random Forest + ANN (average) |
| **Stacking** | RF + ET + LightGBM + CatBoost + ANN → Ridge meta-learner |

Headline results (test, log-price RMSE): Extra Trees 0.270 · Stacking 0.270 · XGBoost 0.271 · CatBoost 0.272 … single tree 0.374 · ANN 0.413. Top four are a statistical tie; ensembles cut error ≈ 25–28 % vs a single tree. See the notebook for the *why* (variance vs bias curves, residual correlation, importance, residuals).

## Limitations
Small sample (633 ads, one day), asking not sale prices, few EVs/rare models, no heavy hyper-parameter search.
