"""
Ensemble Learning Dashboard — Sri Lankan car prices (ikman.lk)

Run:  python3 -m streamlit run app.py

It only *loads* models trained in `ensemble_cars_sri_lanka.ipynb`; nothing is re-trained at runtime.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
MODELS, DATA = ROOT / "models", ROOT / "data"

FAMILY_COLORS = {"Baseline": "#9D9D9D", "Bagging": "#4C78A8", "Boosting": "#E45756",
                 "Neural": "#B279A2", "Hybrid": "#54A24B", "Stacking": "#F58518"}
MAROON = "#8D153A"

st.set_page_config(page_title="Ensemble Car-Price Lab · Sri Lanka", page_icon="🚗", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
.hero {{ background: linear-gradient(120deg, {MAROON} 0%, #C2185B 55%, #F5A623 130%); color: #fff;
        padding: 1.6rem 1.9rem; border-radius: 16px; margin-bottom: 1.2rem; }}
.hero h1 {{ margin: 0 0 .25rem 0; font-size: 2rem; color: #fff; }}
.hero p {{ margin: 0; opacity: .92; font-size: 1.02rem; }}
.price-card {{ border: 2px solid {MAROON}; border-radius: 16px; padding: 1.2rem 1.4rem; text-align: center; }}
.price-card .big {{ font-size: 2.6rem; font-weight: 800; color: {MAROON}; line-height: 1.1; }}
.price-card .sub {{ opacity: .75; font-size: .95rem; }}
.pill {{ display:inline-block; padding:.12rem .6rem; border-radius:999px; color:#fff; font-size:.78rem; margin-right:.3rem; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_data
def load_tables():
    meta = json.load(open(MODELS / "meta.json"))
    metrics = pd.read_csv(MODELS / "metrics.csv")
    importance = pd.read_csv(MODELS / "importance.csv")
    preds = pd.read_csv(MODELS / "test_predictions.csv")
    cars = pd.read_csv(DATA / "cars_clean.csv")
    return meta, metrics, importance, preds, cars


@st.cache_resource(show_spinner="Loading trained ensembles…")
def load_model(slug):
    return joblib.load(MODELS / f"{slug}.joblib")


def rs(v, short=True):
    if short:
        return f"Rs {v/1e6:,.2f} M" if v >= 1e6 else f"Rs {v/1e3:,.0f} K"
    return f"Rs {v:,.0f}"


if not (MODELS / "meta.json").exists():
    st.error("Models not found. Run `ensemble_cars_sri_lanka.ipynb` first to train and save them.")
    st.stop()

meta, metrics, importance, preds, cars = load_tables()
best = metrics.sort_values("test_rmse").iloc[0]
cur_year = meta["current_year"]

st.markdown(f"""<div class="hero"><h1>🚗 Ensemble Car-Price Lab</h1>
<p>What is a car <b>really</b> worth in Sri Lanka? {len(cars):,} live ikman.lk listings · Bagging vs Boosting vs Stacking vs Hybrid · best model:
<b>{best.model}</b> (typical error ≈ {best.test_mape_pct:.0f}%)</p></div>""", unsafe_allow_html=True)

tab_pred, tab_cmp, tab_why, tab_mkt, tab_about = st.tabs(
    ["🔮 Price predictor", "📊 Model comparison", "🧠 Why it works", "🗺️ Market explorer", "ℹ️ About"])

# ----------------------------------------------------------------------------- predictor
with tab_pred:
    st.subheader("Describe a car — all models price it, then the ensembles vote")
    c1, c2, c3 = st.columns(3)
    brands = sorted(meta["brand_models"], key=lambda b: -meta["brand_counts"].get(b, 0))
    brand = c1.selectbox("Brand", brands, index=0)
    model_name = c2.selectbox("Model", meta["brand_models"][brand])
    yr_lo, yr_hi = meta["ranges"]["year"]
    year = c3.slider("Year of manufacture", yr_lo, yr_hi, 2016)

    c4, c5, c6 = st.columns(3)
    cond = c4.selectbox("Condition", meta["categories"]["condition"],
                        index=meta["categories"]["condition"].index("Used") if "Used" in meta["categories"]["condition"] else 0)
    new_car = cond == "Brand New"
    mileage = c5.number_input("Mileage (km)", 0, 800_000, 0 if new_car else max(0, (cur_year - year) * 10_000), step=1_000)
    default_cc = int(meta["median_engine_by_brand_model"].get(f"{brand} {model_name}", 1300))
    cc = c6.number_input("Engine capacity (cc)", 500, 8000, default_cc, step=50)

    c7, c8, c9, c10 = st.columns(4)
    fuel = c7.selectbox("Fuel", meta["categories"]["fuel"])
    trans = c8.selectbox("Transmission", meta["categories"]["transmission"])
    body = c9.selectbox("Body type", meta["categories"]["body"])
    district = c10.selectbox("District", meta["categories"]["district"],
                             index=meta["categories"]["district"].index("Colombo") if "Colombo" in meta["categories"]["district"] else 0)
    dealer = st.toggle("Sold by a dealer / business account", value=False)

    age = max(0, cur_year - year)
    row = pd.DataFrame([{
        "age": age, "mileage_km": float(mileage), "engine_cc": float(cc),
        "km_per_year": mileage / max(age, 1), "log_mileage": np.log1p(mileage),
        "brand_model": f"{brand} {model_name}", "brand": brand, "body": body, "fuel": fuel,
        "transmission": trans, "condition": cond, "district": district, "is_dealer": int(dealer)}])

    out = []
    for _, m in metrics.iterrows():
        out.append({"model": m.model, "family": m.family, "price": float(np.exp(load_model(m.slug).predict(row)[0]))})
    out = pd.DataFrame(out)
    head = out[out.model == best.model].price.iloc[0]
    ens = out[out.family.isin(["Bagging", "Boosting", "Hybrid", "Stacking"])]
    lo, hi = head / (1 + best.test_mape_pct / 100 * 1.25), head * (1 + best.test_mape_pct / 100 * 1.25)

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown(f"""<div class="price-card"><div class="sub">Estimated asking price · {best.model}</div>
        <div class="big">{rs(head)}</div>
        <div class="sub">likely range {rs(lo)} – {rs(hi)}</div></div>""", unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        a.metric("Ensemble median", rs(ens.price.median()))
        b.metric("Model disagreement", f"±{(ens.price.std() / ens.price.mean() * 100):.0f}%",
                 help="Standard deviation of the ensemble predictions relative to their mean. High disagreement = unusual car, treat with caution.")
        st.caption("The range is the best model's typical test error (MAPE × 1.25). Asking prices are negotiable — this is a market estimate, not a valuation.")
    with right:
        o = out.sort_values("price")
        fig = px.bar(o, x="price", y="model", color="family", color_discrete_map=FAMILY_COLORS, orientation="h", height=420)
        fig.add_vline(x=head, line_dash="dash", line_color="black")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="predicted price (LKR)", yaxis_title="", legend_title="")
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- comparison
with tab_cmp:
    st.subheader("Bagging vs Boosting vs Stacking vs Hybrid")
    fams = st.multiselect("Model families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    m = metrics[metrics.family.isin(fams)].copy()
    m["gap"] = m.test_rmse - m.train_rmse
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", best.model)
    k2.metric("Test R² (log-price)", f"{best.test_r2:.3f}")
    k3.metric("Typical error (MAPE)", f"{best.test_mape_pct:.1f}%")
    k4.metric("Avg. error", rs(best.test_mae_lkr))

    c1, c2 = st.columns(2)
    f1 = px.bar(m.sort_values("test_rmse", ascending=False), x="test_rmse", y="model", color="family", color_discrete_map=FAMILY_COLORS,
                orientation="h", title="Test RMSE (log-price) — lower is better", height=430)
    f1.update_layout(yaxis_title="", xaxis_title="", legend_title="", margin=dict(t=40, b=0))
    c1.plotly_chart(f1, width="stretch")
    f2 = px.scatter(m, x="train_rmse", y="test_rmse", color="family", color_discrete_map=FAMILY_COLORS, text="model",
                    title="Over-fitting map: far below the diagonal = memorising", height=430)
    mx = max(m.test_rmse.max(), m.train_rmse.max()) * 1.05
    f2.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False))
    f2.update_traces(textposition="top center", textfont_size=9, selector=dict(mode="markers+text"))
    f2.update_layout(margin=dict(t=40, b=0), legend_title="")
    c2.plotly_chart(f2, width="stretch")

    st.dataframe(m[["model", "family", "cv_rmse_mean", "cv_rmse_std", "test_rmse", "test_r2", "test_mape_pct", "test_mae_lkr", "train_rmse", "gap", "seconds"]]
                 .sort_values("test_rmse").rename(columns={"cv_rmse_mean": "CV RMSE", "cv_rmse_std": "CV std", "test_rmse": "Test RMSE", "test_r2": "Test R²",
                                                           "test_mape_pct": "MAPE %", "test_mae_lkr": "MAE (Rs)", "train_rmse": "Train RMSE", "gap": "Over-fit gap", "seconds": "Time (s)"}),
                 hide_index=True, width="stretch")

    sel = st.selectbox("Predicted vs actual for…", metrics.sort_values("test_rmse").model.tolist())
    slug = metrics.set_index("model").loc[sel, "slug"]
    f3 = px.scatter(x=preds.actual_price / 1e6, y=preds[slug] / 1e6, log_x=True, log_y=True, opacity=.55,
                    labels={"x": "actual price (Rs M)", "y": "predicted price (Rs M)"}, height=420)
    lim = [preds.actual_price.min() / 1e6, preds.actual_price.max() / 1e6]
    f3.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="black", dash="dash"), showlegend=False))
    st.plotly_chart(f3, width="stretch")

# ----------------------------------------------------------------------------- why
with tab_why:
    st.subheader("Why do the ensembles behave the way they do?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🟦 Bagging cuts **variance**")
        rf = pd.DataFrame({"trees": np.arange(1, len(meta["rf_curve"]) + 1), "test RMSE": meta["rf_curve"]})
        fig = px.line(rf, x="trees", y="test RMSE", log_x=True, height=330)
        fig.update_traces(line_color="#4C78A8"); fig.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("One deep tree memorises noise. Averaging hundreds of trees trained on bootstrap samples cancels that noise — error drops fast, then plateaus.")
    with c2:
        st.markdown("#### 🟥 Boosting cuts **bias**")
        gb = pd.DataFrame({"round": range(len(meta["gb_test_curve"])), "train": meta["gb_train_curve"], "test": meta["gb_test_curve"]}).melt("round", var_name="set", value_name="RMSE")
        fig = px.line(gb, x="round", y="RMSE", color="set", height=330, color_discrete_map={"train": "#E45756", "test": "#4C78A8"})
        fig.update_layout(margin=dict(t=10, b=0), legend_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each shallow tree fixes the previous trees' mistakes. Train error keeps falling; test error flattens once the signal is used up — the gap is over-fitting.")
    st.markdown("#### 🔑 What drives the price? (permutation importance)")
    imp = importance.sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std", height=360, color_discrete_sequence=[MAROON])
    fig.update_layout(margin=dict(t=10, b=0), xaxis_title="increase in test RMSE when this feature is shuffled", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.info("**Stacking** blends diverse models (Random Forest, Extra Trees, LightGBM, CatBoost, ANN) with a Ridge meta-learner trained on out-of-fold predictions. "
            "**Hybrid RF + ANN** averages a rule-like forest with a smooth neural net. Both only help if the members make *different* mistakes — see the notebook's residual-correlation heat-map.")

# ----------------------------------------------------------------------------- market
with tab_mkt:
    st.subheader("Sri Lankan car market snapshot (ikman.lk, scraped 23 Nov 2026)")
    f1c, f2c, f3c = st.columns(3)
    sel_fuel = f1c.multiselect("Fuel", sorted(cars.fuel.dropna().unique()), default=[])
    sel_body = f2c.multiselect("Body", sorted(cars.body.dropna().unique()), default=[])
    ymin, ymax = int(cars.year.min()), int(cars.year.max())
    yr = f3c.slider("Model year", ymin, ymax, (max(ymin, 2005), ymax))
    d = cars[cars.year.between(*yr)]
    if sel_fuel: d = d[d.fuel.isin(sel_fuel)]
    if sel_body: d = d[d.body.isin(sel_body)]
    st.caption(f"{len(d):,} listings match")
    c1, c2 = st.columns(2)
    top = d.brand.value_counts().head(12).index
    bp = d[d.brand.isin(top)].groupby("brand").price.median().sort_values().reset_index()
    f = px.bar(bp, x="price", y="brand", orientation="h", title="Median asking price by brand", color_discrete_sequence=[MAROON], height=420)
    f.update_layout(yaxis_title="", xaxis_title="LKR", margin=dict(t=40, b=0)); c1.plotly_chart(f, width="stretch")
    g = d.groupby("year").price.median().reset_index()
    f = px.line(g, x="year", y="price", markers=True, title="Median asking price by model year", height=420)
    f.update_layout(yaxis_title="LKR", margin=dict(t=40, b=0)); c2.plotly_chart(f, width="stretch")
    s = d.dropna(subset=["mileage_km"])
    f = px.scatter(s, x="mileage_km", y="price", color="fuel", log_y=True, hover_data=["brand", "model", "year"], opacity=.6, height=420, title="Price vs mileage")
    st.plotly_chart(f, width="stretch")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
**Goal.** Build an *ensemble-learning dashboard* on a **Sri Lankan** dataset and compare **Bagging, Boosting, Stacking and Hybrid** models, interpreting *why* each behaves as it does.

**Data.** {meta['n_rows']:,} car ads (new, reconditioned and used) scraped from [ikman.lk](https://ikman.lk) on {meta['scraped']} (public pages only, rate-limited, robots.txt respected). Target = asking price in LKR (modelled on a log scale).
{meta['n_train']:,} ads for training / cross-validation and {meta['n_test']:,} held-out for the numbers shown here.

**Models.** Ridge & single Decision Tree (baselines) · Bagged Trees, Random Forest, Extra Trees (bagging) · Gradient Boosting, XGBoost, LightGBM, CatBoost (boosting) ·
ANN (MLP) · Hybrid RF+ANN · Stacking (RF, ET, LightGBM, CatBoost, ANN → Ridge).

**Caveats.** Asking prices ≠ sale prices (negotiation typically lowers them); the data is a single-day snapshot; rare brands/models have little data, so their predictions are less reliable
(check the *model disagreement* indicator).

""")
