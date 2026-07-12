"""
app.py
======
Dynamic Pricing Optimization — Streamlit Web Application

A modern, interactive dashboard that:
  - Lets a user describe a product/context (category, competitor price,
    weather, season, promotions, day of week, ratings, etc.)
  - Uses the trained demand model to simulate a full price-response curve
  - Recommends the revenue-optimal and profit-optimal price
  - Shows price elasticity, expected demand, and feature importance
  - Lets the user pick which trained model (Linear / Random Forest / XGBoost)
    powers the recommendation

Run locally:  streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_engineering import build_model_matrix, CATEGORICAL_COLS

# ----------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B2545"      # deep navy
ACCENT = "#F4B400"       # gold/amber (price & value)
ACCENT_2 = "#13A89E"     # teal (secondary, demand/positive)
BG_CARD = "#13294B"

st.markdown(f"""
<style>
    .stApp {{
        background-color: #F7F9FC;
    }}
    .main-header {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #16406E 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }}
    .main-header h1 {{
        color: white;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .main-header p {{
        color: #C9D6E8;
        font-size: 1rem;
        margin: 0;
    }}
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 4px rgba(11,37,69,0.08);
        border: 1px solid #E7ECF3;
    }}
    .price-highlight {{
        background: linear-gradient(135deg, {ACCENT} 0%, #FFD873 100%);
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
    }}
    .price-highlight .price-value {{
        font-size: 2.6rem;
        font-weight: 800;
        color: {PRIMARY};
    }}
    .price-highlight .price-label {{
        color: #6B4E00;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-size: 0.8rem;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E7ECF3 !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: white;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    meta = joblib.load("models/model_metadata.pkl")
    models = {
        "Linear Regression": ("models/linear_regression_model.pkl", "models/linear_scaler.pkl"),
        "Random Forest": ("models/random_forest_model.pkl", None),
        "XGBoost": ("models/xgboost_model.pkl", None),
    }
    loaded = {}
    for name, (mpath, spath) in models.items():
        if os.path.exists(mpath):
            model = joblib.load(mpath)
            scaler = joblib.load(spath) if spath and os.path.exists(spath) else None
            loaded[name] = (model, scaler)
    return meta, loaded


@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/processed/engineered_data.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_metrics():
    with open("reports/model_metrics.json") as f:
        return json.load(f)


def predict(model, scaler, X):
    if scaler is not None:
        X = scaler.transform(X)
    return np.clip(model.predict(X), 0, None)


def simulate_curve(model, scaler, meta, context: dict, base_price, cost_price,
                    price_min_ratio=0.5, price_max_ratio=1.6, n_points=45):
    prices = np.linspace(base_price * price_min_ratio, base_price * price_max_ratio, n_points)
    rows = []
    for p in prices:
        r = context.copy()
        r["price"] = p
        r["price_vs_competitor_ratio"] = p / max(r["competitor_price"], 0.01)
        r["price_vs_competitor_gap"] = p - r["competitor_price"]
        r["price_vs_base_ratio"] = p / base_price
        r["markup_pct"] = (p - cost_price) / cost_price * 100
        r["is_cheaper_than_competitor"] = int(p < r["competitor_price"])
        r["price_lag1"] = p
        r["price_change_pct"] = 0.0
        r["units_sold"] = 0
        rows.append(r)
    sim_df = pd.DataFrame(rows)
    X, _, _ = build_model_matrix(sim_df)
    X = X.reindex(columns=meta["feature_cols"], fill_value=0)
    demand = predict(model, scaler, X)
    curve = pd.DataFrame({
        "price": prices,
        "predicted_demand": demand,
        "predicted_revenue": prices * demand,
        "predicted_profit": (prices - cost_price) * demand,
    })
    return curve


def point_elasticity(curve, ref_price):
    idx = (curve["price"] - ref_price).abs().idxmin()
    idx = max(1, min(idx, len(curve) - 2))
    p1, p2 = curve.loc[idx - 1, "price"], curve.loc[idx + 1, "price"]
    q1, q2 = curve.loc[idx - 1, "predicted_demand"], curve.loc[idx + 1, "predicted_demand"]
    if (q1 + q2) == 0 or (p1 + p2) == 0:
        return 0.0
    pct_q = (q2 - q1) / ((q1 + q2) / 2)
    pct_p = (p2 - p1) / ((p1 + p2) / 2)
    return round(pct_q / pct_p, 3) if pct_p != 0 else 0.0


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>💹 Dynamic Pricing Optimization Engine</h1>
    <p>ML-powered demand forecasting &amp; price recommendation for e-commerce retail</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists("models/model_metadata.pkl"):
    st.error("Model artifacts not found. Please run the training pipeline first: "
             "`python src/data_generation.py && python src/preprocessing.py && "
             "python src/feature_engineering.py && python src/train_models.py`")
    st.stop()

meta, loaded_models = load_artifacts()
ref_df = load_reference_data()
metrics = load_metrics()

# ----------------------------------------------------------------------
# Sidebar — Inputs
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Pricing Scenario")
    st.caption("Describe the product & market context to get a price recommendation.")

    model_choice = st.selectbox("Prediction model", list(loaded_models.keys()),
                                 index=list(loaded_models.keys()).index(meta["best_model_name"])
                                 if meta["best_model_name"] in loaded_models else 0)

    category = st.selectbox("Product category", sorted(ref_df["category"].unique()))
    cat_df = ref_df[ref_df["category"] == category]

    brand_tier = st.selectbox("Brand tier", ["Budget", "Standard", "Premium"], index=1)

    default_base = float(cat_df["base_price"].median())
    base_price = st.number_input("Base / list price ($)", min_value=1.0,
                                  value=round(default_base, 2), step=1.0)
    cost_price = st.number_input("Unit cost ($)", min_value=0.1,
                                  value=round(base_price * 0.55, 2), step=1.0)
    competitor_price = st.number_input("Competitor price ($)", min_value=0.1,
                                        value=round(default_base * 1.02, 2), step=1.0)

    st.markdown("---")
    st.markdown("**📅 Timing**")
    day_of_week = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday",
                                                 "Friday","Saturday","Sunday"])
    dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
    month = st.selectbox("Month", list(range(1, 13)), index=6,
                          format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
    is_holiday = st.checkbox("Holiday period", value=False)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

    st.markdown("---")
    st.markdown("**🌦️ Weather**")
    weather_condition = st.selectbox("Weather condition",
                                      ["Sunny", "Cloudy", "Rainy", "Snowy", "Stormy"])
    temperature_c = st.slider("Temperature (°C)", -10, 45, 20)

    st.markdown("---")
    st.markdown("**🏷️ Promotion & Product Quality**")
    promotion_flag = st.checkbox("Active promotion", value=False)
    discount_pct = st.slider("Promotion discount (%)", 0, 50, 10, disabled=not promotion_flag)
    rating = st.slider("Product rating", 1.0, 5.0, float(round(cat_df["rating"].median(),1)), 0.1)
    review_count = st.number_input("Review count", min_value=0,
                                    value=int(cat_df["review_count"].median()))
    inventory_level = st.number_input("Inventory level (units)", min_value=0,
                                       value=int(cat_df["inventory_level"].median()))
    website_traffic = st.number_input("Expected daily website traffic", min_value=0,
                                       value=int(cat_df["website_traffic"].median()))

    objective = st.radio("Optimize for", ["Revenue", "Profit"], horizontal=True)

    run_btn = st.button("🚀 Generate Price Recommendation", use_container_width=True, type="primary")

# ----------------------------------------------------------------------
# Build context row & run
# ----------------------------------------------------------------------
if "run_once" not in st.session_state:
    st.session_state.run_once = True  # auto-run on first load with defaults

if run_btn or st.session_state.get("first_load", True):
    st.session_state.first_load = False

    is_weekend = int(day_of_week in ["Saturday", "Sunday"])
    week_of_year = int(pd.Timestamp(2024, month, 15).isocalendar().week)

    context = {
        "date": pd.Timestamp(2024, month, 15),
        "product_id": "SIM001",
        "category": category,
        "brand_tier": brand_tier,
        "base_price": base_price,
        "cost_price": cost_price,
        "price": base_price,
        "competitor_price": competitor_price,
        "discount_pct": discount_pct if promotion_flag else 0,
        "promotion_flag": int(promotion_flag),
        "day_of_week": dow_map[day_of_week],
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": int(is_holiday),
        "season": season,
        "temperature_c": temperature_c,
        "weather_condition": weather_condition,
        "rating": rating,
        "review_count": review_count,
        "inventory_level": inventory_level,
        "website_traffic": website_traffic,
        "days_since_launch": int(cat_df["days_since_launch"].median()),
        "week_of_year": week_of_year,
        "is_month_start": 0,
        "is_month_end": 0,
        "dow_sin": np.sin(2*np.pi*dow_map[day_of_week]/7),
        "dow_cos": np.cos(2*np.pi*dow_map[day_of_week]/7),
        "month_sin": np.sin(2*np.pi*month/12),
        "month_cos": np.cos(2*np.pi*month/12),
        "units_sold_lag1": float(cat_df["units_sold"].median()),
        "units_sold_roll7": float(cat_df["units_sold"].median()),
        "units_sold_roll30": float(cat_df["units_sold"].median()),
        "temp_bucket": pd.cut([temperature_c], bins=[-100,0,10,20,30,100],
                               labels=["Freezing","Cold","Mild","Warm","Hot"])[0],
    }

    model, scaler = loaded_models[model_choice]
    curve = simulate_curve(model, scaler, meta, context, base_price, cost_price)

    obj_col = "predicted_revenue" if objective == "Revenue" else "predicted_profit"
    best_row = curve.loc[curve[obj_col].idxmax()]
    optimal_price = float(best_row["price"])
    optimal_demand = float(best_row["predicted_demand"])
    optimal_metric = float(best_row[obj_col])

    current_demand_row = curve.iloc[(curve["price"] - base_price).abs().idxmin()]
    current_demand = float(current_demand_row["predicted_demand"])
    current_revenue = base_price * current_demand
    elasticity = point_elasticity(curve, base_price)

    price_delta_pct = (optimal_price - base_price) / base_price * 100

    # ------------------------------------------------------------------
    # Top recommendation row
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="price-highlight">
            <div class="price-label">Recommended Optimal Price ({objective})</div>
            <div class="price-value">${optimal_price:,.2f}</div>
            <div style="color:#6B4E00; font-weight:600;">
                {'+' if price_delta_pct>=0 else ''}{price_delta_pct:.1f}% vs. your list price
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><b>Predicted Demand</b><br>
        <span style="font-size:1.8rem;color:{PRIMARY};font-weight:700">{optimal_demand:,.0f}</span>
        <span style="color:#888"> units/day</span></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><b>Expected {objective}</b><br>
        <span style="font-size:1.8rem;color:{PRIMARY};font-weight:700">${optimal_metric:,.0f}</span>
        <span style="color:#888"> /day</span></div>""", unsafe_allow_html=True)
    with col4:
        elastic_label = "Elastic" if abs(elasticity) > 1 else "Inelastic"
        st.markdown(f"""<div class="metric-card"><b>Price Elasticity</b><br>
        <span style="font-size:1.8rem;color:{PRIMARY};font-weight:700">{elasticity:.2f}</span>
        <span style="color:#888"> ({elastic_label})</span></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Tabs: Price curve / Elasticity / Model performance / Feature importance
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Price–Demand Curve", "🎯 Elasticity Insight", "📊 Model Performance", "🔍 Feature Importance"]
    )

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["price"], y=curve["predicted_demand"],
                                  mode="lines", name="Predicted Demand",
                                  line=dict(color=ACCENT_2, width=3)))
        fig.add_vline(x=base_price, line_dash="dot", line_color="#888",
                      annotation_text="Your list price")
        fig.add_vline(x=optimal_price, line_dash="dash", line_color=ACCENT,
                      annotation_text="Optimal price")
        fig.update_layout(title="Simulated Demand Curve vs. Price",
                           xaxis_title="Price ($)", yaxis_title="Predicted Demand (units/day)",
                           template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=curve["price"], y=curve["predicted_revenue"],
                                   mode="lines", name="Revenue", line=dict(color=PRIMARY, width=3)))
        fig2.add_trace(go.Scatter(x=curve["price"], y=curve["predicted_profit"],
                                   mode="lines", name="Profit", line=dict(color=ACCENT, width=3)))
        fig2.add_vline(x=optimal_price, line_dash="dash", line_color="#666")
        fig2.update_layout(title="Revenue & Profit vs. Price",
                            xaxis_title="Price ($)", yaxis_title="$ / day",
                            template="plotly_white", height=420, legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### What elasticity means here")
            st.write(
                f"At your current list price of **${base_price:,.2f}**, the estimated price "
                f"elasticity of demand is **{elasticity:.2f}**. "
                + ("This category is **price elastic** — a 1% price increase is expected to "
                   "reduce demand by more than 1%, so small discounts can grow revenue."
                   if abs(elasticity) > 1 else
                   "This category is **price inelastic** — demand is relatively insensitive "
                   "to price changes, so a moderate price increase is unlikely to sharply "
                   "reduce units sold.")
            )
            st.markdown("#### Category elasticity benchmark")
            if os.path.exists("reports/price_elasticity_by_category.csv"):
                elastic_ref = pd.read_csv("reports/price_elasticity_by_category.csv")
                st.dataframe(elastic_ref, use_container_width=True, hide_index=True)
        with c2:
            fig3 = px.line(curve, x="price", y="predicted_demand", template="plotly_white")
            fig3.add_vline(x=base_price, line_dash="dot", line_color="#888")
            fig3.update_layout(title="Demand Sensitivity Zoom", height=350)
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("#### Trained model comparison (held-out test set)")
        met_df = pd.DataFrame(metrics).T
        display_cols = [c for c in ["RMSE", "MAE", "R2", "MAPE_%"] if c in met_df.columns]
        st.dataframe(met_df[display_cols].style.highlight_min(subset=["RMSE","MAE"], color="#D9F5E5")
                     .highlight_max(subset=["R2"], color="#D9F5E5"),
                     use_container_width=True)
        st.caption("RMSE / MAE lower is better; R² higher is better. "
                   f"Currently serving predictions with **{model_choice}**.")

        fig4 = go.Figure()
        for name in met_df.index:
            fig4.add_trace(go.Bar(name=name, x=["RMSE","MAE"],
                                   y=[met_df.loc[name,"RMSE"], met_df.loc[name,"MAE"]]))
        fig4.update_layout(barmode="group", template="plotly_white", height=380,
                            title="Error Metrics by Model")
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.markdown("#### What drives demand predictions")
        imp_path = f"reports/feature_importance_{'RandomForest' if model_choice=='Random Forest' else 'XGBoost'}.csv"
        if model_choice == "Linear Regression":
            imp_path = "reports/linear_regression_coefficients.csv"
            imp_df = pd.read_csv(imp_path).head(15)
            imp_df["importance"] = imp_df["coefficient"].abs()
            imp_df = imp_df.sort_values("importance", ascending=True)
            fig5 = px.bar(imp_df, x="importance", y="feature", orientation="h",
                          title="Top 15 Standardized Linear Regression Coefficients (|value|)",
                          template="plotly_white", color_discrete_sequence=[ACCENT_2])
        elif os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path).head(15).sort_values("importance", ascending=True)
            fig5 = px.bar(imp_df, x="importance", y="feature", orientation="h",
                          title=f"Top 15 Feature Importances — {model_choice}",
                          template="plotly_white", color_discrete_sequence=[ACCENT_2])
        else:
            fig5 = None
        if fig5:
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚠️ This tool simulates demand using a machine-learning model trained on historical "
        "(synthetic, economically-modeled) transaction data. Treat recommendations as decision "
        "support, not an automated pricing action — validate with A/B testing before deploying "
        "price changes in production."
    )
