# -*- coding: utf-8 -*-
"""
Insurance Premium Prediction Pro AI
====================================
A production-ready, AI-powered Insurance Premium Prediction application
built with Streamlit featuring a modern glassmorphism UI, synthetic data
generation, ensemble machine learning, explainable AI, lifestyle
simulation, fairness analysis, and PDF/QR reporting.

Run with:
    streamlit run app.py

No external dataset or pickle files are required. Everything (data
generation, model training) happens automatically on first launch and is
cached for performance.
"""

import io
import time
import base64
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# OPTIONAL LIBRARIES - never crash if unavailable
# --------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    )
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import qrcode
    from PIL import Image
    QRCODE_AVAILABLE = True
except Exception:
    QRCODE_AVAILABLE = False


# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================
st.set_page_config(
    page_title="Insurance Premium Prediction Pro AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# CUSTOM CSS - GLASSMORPHISM / DARK MODE / ANIMATIONS
# ==========================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, #1f1147 0%, #0b0c2a 35%, #05060f 100%);
    background-attachment: fixed;
    color: #EDEDF7;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#7f5af0,#2cb67d); border-radius: 10px; }

/* ---------- Glass card ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 24px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 40px rgba(127,90,240,0.35);
    border: 1px solid rgba(127,90,240,0.55);
}

/* ---------- Hero ---------- */
.hero-wrap {
    background: linear-gradient(120deg, rgba(127,90,240,0.28), rgba(44,182,125,0.18));
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 28px;
    padding: 46px 40px;
    text-align: center;
    margin-bottom: 28px;
    animation: fadeInUp 0.8s ease;
    box-shadow: 0 10px 50px rgba(127,90,240,0.25);
}
.hero-title {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(90deg, #7f5af0, #2cb67d, #ff8906);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shimmer 6s linear infinite;
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 17px;
    color: #c8c8e0;
    font-weight: 400;
    max-width: 760px;
    margin: 0 auto;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- KPI Card ---------- */
.kpi-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(14px);
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease;
}
.kpi-card:hover {
    transform: scale(1.04);
    border: 1px solid #7f5af0;
    box-shadow: 0 0 26px rgba(127,90,240,0.45);
}
.kpi-value {
    font-size: 30px;
    font-weight: 800;
    background: linear-gradient(90deg,#7f5af0,#2cb67d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 13px;
    color: #b7b7d1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ---------- Section title ---------- */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #f2f2fb;
    margin: 18px 0 14px 0;
    border-left: 5px solid #7f5af0;
    padding-left: 14px;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px;
}
.badge-green { background: rgba(44,182,125,0.18); color: #2cb67d; border: 1px solid #2cb67d;}
.badge-orange { background: rgba(255,137,6,0.18); color: #ff8906; border: 1px solid #ff8906;}
.badge-red { background: rgba(239,71,111,0.18); color: #ef476f; border: 1px solid #ef476f;}
.badge-purple { background: rgba(127,90,240,0.18); color: #7f5af0; border: 1px solid #7f5af0;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120a2e 0%, #05060f 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 10px 26px;
    font-weight: 700;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 18px rgba(127,90,240,0.35);
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(127,90,240,0.55);
}

/* ---------- Inputs ---------- */
.stSlider, .stSelectbox, .stNumberInput { border-radius: 12px; }

/* ---------- Progress ring text ---------- */
.ring-label {
    text-align: center;
    font-size: 14px;
    color: #c8c8e0;
    margin-top: -8px;
}

hr { border-color: rgba(255,255,255,0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# CONSTANTS
# ==========================================================================
REGIONS = ["North", "South", "East", "West", "Central"]
OCCUPATION_RISK = ["Low", "Medium", "High"]
DIET_QUALITY = ["Poor", "Average", "Good", "Excellent"]
GENDERS = ["Male", "Female", "Other"]

RANDOM_STATE = 42


# ==========================================================================
# SYNTHETIC DATA GENERATION
# ==========================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n_records: int = 6500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic synthetic insurance dataset with no external files."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n_records)
    gender = rng.choice(GENDERS, n_records, p=[0.48, 0.48, 0.04])
    height_cm = rng.normal(168, 10, n_records).clip(140, 205)
    base_weight = 22 + (age / 100) * 5
    weight_kg = (base_weight * (height_cm / 100) ** 2 + rng.normal(0, 8, n_records)).clip(40, 160)
    bmi = weight_kg / ((height_cm / 100) ** 2)

    smoking = rng.choice(["Yes", "No"], n_records, p=[0.22, 0.78])
    alcohol = rng.choice(["None", "Occasional", "Regular", "Heavy"], n_records, p=[0.4, 0.35, 0.18, 0.07])
    exercise_days = rng.integers(0, 8, n_records)
    occupation_risk = rng.choice(OCCUPATION_RISK, n_records, p=[0.5, 0.35, 0.15])
    income = rng.lognormal(mean=10.8, sigma=0.55, size=n_records).clip(8000, 500000)
    region = rng.choice(REGIONS, n_records)

    blood_pressure = rng.choice(["Normal", "Elevated", "High"], n_records, p=[0.55, 0.25, 0.20])
    diabetes = rng.choice(["Yes", "No"], n_records, p=[0.14, 0.86])
    heart_disease = rng.choice(["Yes", "No"], n_records, p=[0.09, 0.91])
    family_history = rng.choice(["Yes", "No"], n_records, p=[0.3, 0.7])
    dependents = rng.integers(0, 6, n_records)
    health_checkups = rng.integers(0, 5, n_records)
    sleep_hours = rng.normal(6.8, 1.3, n_records).clip(3, 11)
    stress_level = rng.integers(1, 11, n_records)
    diet_quality = rng.choice(DIET_QUALITY, n_records, p=[0.2, 0.4, 0.3, 0.1])

    # ---- Realistic premium formula (target variable) ----
    premium = 4000.0
    premium += age * 42
    premium += np.where(bmi > 30, (bmi - 30) * 180, 0)
    premium += np.where(bmi < 18.5, (18.5 - bmi) * 120, 0)
    premium += np.where(smoking == "Yes", 4200, 0)
    premium += np.where(alcohol == "Heavy", 1800, np.where(alcohol == "Regular", 700, 0))
    premium -= exercise_days * 90
    premium += np.where(occupation_risk == "High", 2600, np.where(occupation_risk == "Medium", 1000, 0))
    premium += income * 0.008
    premium += np.where(blood_pressure == "High", 1900, np.where(blood_pressure == "Elevated", 700, 0))
    premium += np.where(diabetes == "Yes", 2400, 0)
    premium += np.where(heart_disease == "Yes", 3600, 0)
    premium += np.where(family_history == "Yes", 1400, 0)
    premium += dependents * 320
    premium -= health_checkups * 150
    premium += np.where(sleep_hours < 6, (6 - sleep_hours) * 220, 0)
    premium += stress_level * 110
    premium += np.where(diet_quality == "Poor", 1300,
                 np.where(diet_quality == "Average", 500,
                 np.where(diet_quality == "Good", -300, -700)))
    premium += rng.normal(0, 900, n_records)  # natural noise
    premium = premium.clip(1800, 60000)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Height_cm": height_cm.round(1),
        "Weight_kg": weight_kg.round(1),
        "BMI": bmi.round(2),
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Exercise_Days": exercise_days,
        "Occupation_Risk": occupation_risk,
        "Income": income.round(2),
        "Region": region,
        "Blood_Pressure": blood_pressure,
        "Diabetes": diabetes,
        "Heart_Disease": heart_disease,
        "Family_History": family_history,
        "Dependents": dependents,
        "Health_Checkups": health_checkups,
        "Sleep_Hours": sleep_hours.round(1),
        "Stress_Level": stress_level,
        "Diet_Quality": diet_quality,
        "Premium": premium.round(2),
    })
    return df


# ==========================================================================
# FEATURE ENGINEERING
# ==========================================================================
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot / ordinal encode categorical features for ML models."""
    data = df.copy()

    binary_maps = {
        "Smoking": {"No": 0, "Yes": 1},
        "Diabetes": {"No": 0, "Yes": 1},
        "Heart_Disease": {"No": 0, "Yes": 1},
        "Family_History": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_maps.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    ordinal_maps = {
        "Alcohol": {"None": 0, "Occasional": 1, "Regular": 2, "Heavy": 3},
        "Occupation_Risk": {"Low": 0, "Medium": 1, "High": 2},
        "Blood_Pressure": {"Normal": 0, "Elevated": 1, "High": 2},
        "Diet_Quality": {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3},
    }
    for col, mapping in ordinal_maps.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)

    data = pd.get_dummies(data, columns=["Gender", "Region"], drop_first=False)
    return data


FEATURE_COLUMNS_CACHE = {}


def prepare_ml_dataset(df: pd.DataFrame):
    """Encode dataset and split into X, y aligned with a stable column order."""
    encoded = encode_features(df)
    y = encoded["Premium"] if "Premium" in encoded.columns else None
    X = encoded.drop(columns=["Premium"], errors="ignore")
    return X, y


def align_columns(X_input: pd.DataFrame, reference_columns) -> pd.DataFrame:
    """Ensure a single-row/inference dataframe has the same columns as training data."""
    for col in reference_columns:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[reference_columns]
    return X_input


# ==========================================================================
# MODEL TRAINING (CACHED)
# ==========================================================================
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """Train Random Forest plus any available optional models. Returns dict of
    trained models, scaler, feature columns, and evaluation metrics."""
    X, y = prepare_ml_dataset(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    metrics = {}

    # ---- Random Forest (always available) ----
    rf = RandomForestRegressor(
        n_estimators=220, max_depth=14, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = {"model": rf, "scaled": False}
    pred = rf.predict(X_test)
    metrics["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }

    # ---- XGBoost ----
    if XGBOOST_AVAILABLE:
        try:
            xgb = XGBRegressor(
                n_estimators=260, max_depth=6, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.85,
                random_state=RANDOM_STATE, verbosity=0
            )
            xgb.fit(X_train, y_train)
            models["XGBoost"] = {"model": xgb, "scaled": False}
            pred = xgb.predict(X_test)
            metrics["XGBoost"] = {
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred),
            }
        except Exception:
            pass

    # ---- LightGBM ----
    if LIGHTGBM_AVAILABLE:
        try:
            lgbm = LGBMRegressor(
                n_estimators=260, max_depth=-1, learning_rate=0.06,
                random_state=RANDOM_STATE, verbose=-1
            )
            lgbm.fit(X_train, y_train)
            models["LightGBM"] = {"model": lgbm, "scaled": False}
            pred = lgbm.predict(X_test)
            metrics["LightGBM"] = {
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred),
            }
        except Exception:
            pass

    # ---- CatBoost ----
    if CATBOOST_AVAILABLE:
        try:
            cat = CatBoostRegressor(
                iterations=260, depth=6, learning_rate=0.06,
                random_state=RANDOM_STATE, verbose=False
            )
            cat.fit(X_train, y_train)
            models["CatBoost"] = {"model": cat, "scaled": False}
            pred = cat.predict(X_test)
            metrics["CatBoost"] = {
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred),
            }
        except Exception:
            pass

    # ---- KNN for "similar customers" ----
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(X_train_scaled)

    return {
        "models": models,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "knn": knn,
        "X_train": X_train,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def ensemble_predict(bundle: dict, X_row: pd.DataFrame):
    """Run every trained model on a single input row and combine results."""
    X_aligned = align_columns(X_row.copy(), bundle["feature_columns"])
    preds = {}
    start = time.time()
    for name, info in bundle["models"].items():
        try:
            p = info["model"].predict(X_aligned)[0]
            preds[name] = max(float(p), 500.0)
        except Exception:
            continue
    elapsed = time.time() - start

    if not preds:
        return 0.0, {}, 0.0, elapsed

    values = np.array(list(preds.values()))
    ensemble_value = float(np.mean(values))
    spread = float(np.std(values))
    agreement = float(max(0.0, 100.0 - (spread / max(ensemble_value, 1)) * 100))
    return ensemble_value, preds, agreement, elapsed


# ==========================================================================
# AI HELPER FUNCTIONS (Health Score, Risk, BMI, Fraud, Explainability)
# ==========================================================================
def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def compute_health_score(inputs: dict) -> float:
    """A 0-100 composite health score derived from lifestyle & medical inputs."""
    score = 100.0
    bmi = inputs["BMI"]
    if bmi < 18.5 or bmi >= 30:
        score -= 18
    elif bmi >= 25:
        score -= 8

    if inputs["Smoking"] == "Yes":
        score -= 20
    if inputs["Alcohol"] == "Heavy":
        score -= 12
    elif inputs["Alcohol"] == "Regular":
        score -= 6

    score += min(inputs["Exercise_Days"], 5) * 2.4
    score -= max(0, 5 - inputs["Exercise_Days"]) * 1.5

    if inputs["Blood_Pressure"] == "High":
        score -= 14
    elif inputs["Blood_Pressure"] == "Elevated":
        score -= 6

    if inputs["Diabetes"] == "Yes":
        score -= 14
    if inputs["Heart_Disease"] == "Yes":
        score -= 18
    if inputs["Family_History"] == "Yes":
        score -= 6

    score += min(inputs["Health_Checkups"], 4) * 2
    if inputs["Sleep_Hours"] < 6:
        score -= 10
    elif inputs["Sleep_Hours"] > 9:
        score -= 4
    else:
        score += 4

    score -= (inputs["Stress_Level"] - 5) * 1.6

    diet_bonus = {"Poor": -10, "Average": 0, "Good": 8, "Excellent": 14}
    score += diet_bonus.get(inputs["Diet_Quality"], 0)

    return float(np.clip(score, 0, 100))


def compute_risk_score(health_score: float) -> float:
    return float(np.clip(100 - health_score, 0, 100))


def risk_category(risk_score: float) -> str:
    if risk_score < 25:
        return "Low Risk"
    elif risk_score < 50:
        return "Moderate Risk"
    elif risk_score < 75:
        return "High Risk"
    else:
        return "Critical Risk"


def recommend_plan(risk_score: float, income: float) -> str:
    if risk_score < 25 and income > 60000:
        return "Platinum Comprehensive Plan"
    elif risk_score < 25:
        return "Gold Value Plan"
    elif risk_score < 50:
        return "Silver Balanced Plan"
    elif risk_score < 75:
        return "Bronze Essential Plan"
    else:
        return "High-Risk Managed Care Plan"


def detect_fraud_signal(inputs: dict, predicted_premium: float) -> dict:
    """A simple rule-based anomaly / fraud heuristic (not a real fraud model)."""
    flags = []
    if inputs["Income"] < 10000 and predicted_premium > 20000:
        flags.append("Premium disproportionately high relative to reported income")
    if inputs["Smoking"] == "No" and inputs["Alcohol"] == "Heavy" and inputs["Exercise_Days"] >= 6:
        flags.append("Unusual combination of heavy alcohol use with very high exercise frequency")
    if inputs["Age"] < 25 and inputs["Heart_Disease"] == "Yes" and inputs["Diabetes"] == "Yes":
        flags.append("Rare combination of multiple chronic conditions at a young age")
    if inputs["Health_Checkups"] == 0 and inputs["Family_History"] == "Yes" and inputs["Age"] > 55:
        flags.append("No health checkups despite family history and advanced age")

    risk_level = "Low" if len(flags) == 0 else ("Medium" if len(flags) == 1 else "High")
    return {"flags": flags, "risk_level": risk_level}


def manual_feature_explanation(bundle: dict, X_row: pd.DataFrame, top_n: int = 6):
    """Fallback explainability using Random Forest feature_importances_ combined
    with the direction of each feature relative to the training mean."""
    rf_info = bundle["models"].get("Random Forest")
    if rf_info is None:
        return []
    importances = rf_info["model"].feature_importances_
    cols = bundle["feature_columns"]
    train_means = bundle["X_train"].mean()

    X_aligned = align_columns(X_row.copy(), cols)
    contributions = []
    for i, col in enumerate(cols):
        diff = X_aligned.iloc[0][col] - train_means[col]
        contributions.append((col, importances[i], diff))

    contributions.sort(key=lambda x: x[1], reverse=True)
    top = contributions[:top_n]
    explanations = []
    for col, importance, diff in top:
        direction = "increases" if diff > 0 else "decreases" if diff < 0 else "is near average, minor effect on"
        explanations.append({
            "feature": col,
            "importance": round(float(importance) * 100, 2),
            "direction": direction,
        })
    return explanations


def shap_explanation(bundle: dict, X_row: pd.DataFrame, top_n: int = 6):
    """Use SHAP TreeExplainer on the Random Forest model when available."""
    try:
        rf_info = bundle["models"].get("Random Forest")
        if rf_info is None:
            return None
        X_aligned = align_columns(X_row.copy(), bundle["feature_columns"])
        explainer = shap.TreeExplainer(rf_info["model"])
        shap_values = explainer.shap_values(X_aligned)
        contributions = list(zip(bundle["feature_columns"], shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top = contributions[:top_n]
        return [{"feature": f, "shap_value": round(float(v), 2)} for f, v in top]
    except Exception:
        return None


def find_similar_customers(bundle: dict, df_raw: pd.DataFrame, X_row: pd.DataFrame, k: int = 5):
    """Use KNN to find the k most similar customers in the training population."""
    try:
        X_aligned = align_columns(X_row.copy(), bundle["feature_columns"])
        X_scaled = bundle["scaler"].transform(X_aligned)
        distances, indices = bundle["knn"].kneighbors(X_scaled, n_neighbors=k + 1)
        idx = bundle["X_train"].index[indices[0]]
        similar = df_raw.loc[df_raw.index.intersection(idx)]
        return similar
    except Exception:
        return pd.DataFrame()


def forecast_premiums(base_inputs: dict, years: int = 5):
    """Generate a naive 5-year premium forecast for three lifestyle scenarios."""
    current, improved, poor = [], [], []
    base = base_inputs["base_premium"]
    for y in range(years + 1):
        inflation = (1.045 ** y)
        current.append(base * inflation)
        improved.append(base * inflation * (0.97 ** y))
        poor.append(base * inflation * (1.06 ** y))
    return list(range(datetime.now().year, datetime.now().year + years + 1)), current, improved, poor


# ==========================================================================
# CHART BUILDERS
# ==========================================================================
def gauge_chart(value: float, title: str, max_value: float = 100,
                 color_stops=(("#2cb67d", 0, 40), ("#ff8906", 40, 70), ("#ef476f", 70, 100))):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18, "color": "#EDEDF7"}},
        number={"font": {"size": 34, "color": "#EDEDF7"}},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": "#c8c8e0"},
            "bar": {"color": "#7f5af0"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [a, b], "color": c} for c, a, b in
                      [(s[0], s[1] * max_value / 100, s[2] * max_value / 100) for s in color_stops]],
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EDEDF7"},
    )
    return fig


def progress_ring(value: float, label: str, color: str = "#7f5af0"):
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        hole=0.72,
        marker_colors=[color, "rgba(255,255,255,0.08)"],
        textinfo="none",
        sort=False,
        direction="clockwise",
    ))
    fig.update_layout(
        showlegend=False, height=180, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{value:.0f}%</b>", x=0.5, y=0.5, font_size=22,
                           font_color="#EDEDF7", showarrow=False)],
    )
    return fig


def themed_layout(fig, height=420, title=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EDEDF7"}, height=height,
        title=title, margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


# ==========================================================================
# PDF / TXT REPORT GENERATION
# ==========================================================================
def generate_pdf_report(inputs: dict, prediction_result: dict) -> bytes:
    """Generate a downloadable PDF report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#7f5af0"))
    heading_style = ParagraphStyle("HeadX", parent=styles["Heading2"], textColor=colors.HexColor("#2cb67d"))
    normal = styles["Normal"]

    elements = [
        Paragraph("Insurance Premium Prediction Pro AI", title_style),
        Paragraph("AI-Generated Insurance Report", normal),
        Spacer(1, 14),
        Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal),
        Spacer(1, 18),
        Paragraph("Customer Profile", heading_style),
    ]

    profile_data = [["Field", "Value"]]
    for k in ["Age", "Gender", "BMI", "Smoking", "Exercise_Days", "Region",
              "Blood_Pressure", "Diabetes", "Heart_Disease", "Income"]:
        profile_data.append([k.replace("_", " "), str(inputs.get(k, "-"))])

    table = Table(profile_data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f5af0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("Prediction Results", heading_style))
    elements.append(Paragraph(f"Predicted Annual Premium: Rs. {prediction_result['ensemble']:.2f}", normal))
    elements.append(Paragraph(f"Model Agreement: {prediction_result['agreement']:.1f}%", normal))
    elements.append(Paragraph(f"Health Score: {prediction_result['health_score']:.1f} / 100", normal))
    elements.append(Paragraph(f"Risk Category: {prediction_result['risk_category']}", normal))
    elements.append(Paragraph(f"Recommended Plan: {prediction_result['plan']}", normal))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Recommendations", heading_style))
    for rec in prediction_result.get("recommendations", []):
        elements.append(Paragraph(f"- {rec}", normal))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("5-Year Forecast (Current Lifestyle)", heading_style))
    years, current, improved, poor = prediction_result["forecast"]
    forecast_data = [["Year", "Current", "Improved", "Poor"]]
    for i, yr in enumerate(years):
        forecast_data.append([str(yr), f"{current[i]:.0f}", f"{improved[i]:.0f}", f"{poor[i]:.0f}"])
    ftable = Table(forecast_data, colWidths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
    ftable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2cb67d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(ftable)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_txt_report(inputs: dict, prediction_result: dict) -> bytes:
    """Fallback plain-text report when ReportLab is unavailable."""
    lines = [
        "INSURANCE PREMIUM PREDICTION PRO AI",
        "=" * 45,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "CUSTOMER PROFILE",
        "-" * 45,
    ]
    for k in ["Age", "Gender", "BMI", "Smoking", "Exercise_Days", "Region",
              "Blood_Pressure", "Diabetes", "Heart_Disease", "Income"]:
        lines.append(f"{k.replace('_', ' ')}: {inputs.get(k, '-')}")

    lines += [
        "",
        "PREDICTION RESULTS",
        "-" * 45,
        f"Predicted Annual Premium: Rs. {prediction_result['ensemble']:.2f}",
        f"Model Agreement: {prediction_result['agreement']:.1f}%",
        f"Health Score: {prediction_result['health_score']:.1f} / 100",
        f"Risk Category: {prediction_result['risk_category']}",
        f"Recommended Plan: {prediction_result['plan']}",
        "",
        "RECOMMENDATIONS",
        "-" * 45,
    ]
    for rec in prediction_result.get("recommendations", []):
        lines.append(f"- {rec}")

    lines += ["", "5-YEAR FORECAST", "-" * 45]
    years, current, improved, poor = prediction_result["forecast"]
    for i, yr in enumerate(years):
        lines.append(f"{yr}: Current={current[i]:.0f}  Improved={improved[i]:.0f}  Poor={poor[i]:.0f}")

    return "\n".join(lines).encode("utf-8")


def generate_qr_code(payload: str):
    """Generate a QR code image (bytes) for the digital health passport."""
    try:
        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#7f5af0", back_color="white").convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ==========================================================================
# SESSION STATE INITIALIZATION
# ==========================================================================
def init_session_state():
    defaults = {
        "page": "🏠 Home",
        "last_prediction": None,
        "last_inputs": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ==========================================================================
# DATA & MODEL BOOTSTRAP
# ==========================================================================
with st.spinner("🔄 Initializing AI engine and training models..."):
    raw_df = generate_synthetic_dataset()
    model_bundle = train_models(raw_df)


# ==========================================================================
# SIDEBAR NAVIGATION
# ==========================================================================
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 10px 0;'>"
        "<h2 style='color:#7f5af0; font-weight:800;'>🛡️ InsurAI Pro</h2>"
        "<p style='color:#8f8fb0; font-size:13px;'>Premium Prediction Engine</p>"
        "</div>", unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🧮 Premium Prediction",
            "📊 Dashboard",
            "📈 Analytics",
            "🧠 AI Insights",
            "🎯 Lifestyle Simulator",
            "❤️ Health Score",
            "📄 PDF Report",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<p style='color:#8f8fb0; font-size:12px;'>Models Active:</p>", unsafe_allow_html=True)
    active_models = ", ".join(model_bundle["models"].keys())
    st.markdown(f"<p style='color:#2cb67d; font-size:12px;'>{active_models}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#8f8fb0; font-size:11px;'>Dataset: {len(raw_df):,} synthetic records</p>",
                unsafe_allow_html=True)


# ==========================================================================
# INPUT FORM (shared across pages that need customer inputs)
# ==========================================================================
def render_input_form(key_prefix: str = "form"):
    """Render the full customer input form and return a dict of values."""
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 90, 35, key=f"{key_prefix}_age")
        gender = st.selectbox("Gender", GENDERS, key=f"{key_prefix}_gender")
        height_cm = st.slider("Height (cm)", 130, 220, 170, key=f"{key_prefix}_height")
        weight_kg = st.slider("Weight (kg)", 35, 180, 70, key=f"{key_prefix}_weight")
        income = st.number_input("Annual Income (Rs.)", 5000, 1000000, 45000, step=1000,
                                  key=f"{key_prefix}_income")
        region = st.selectbox("Region", REGIONS, key=f"{key_prefix}_region")
        dependents = st.slider("Dependents", 0, 8, 1, key=f"{key_prefix}_dependents")

    with col2:
        smoking = st.selectbox("Smoking", ["No", "Yes"], key=f"{key_prefix}_smoking")
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Regular", "Heavy"],
                                key=f"{key_prefix}_alcohol")
        exercise_days = st.slider("Exercise Days / Week", 0, 7, 3, key=f"{key_prefix}_exercise")
        occupation_risk = st.selectbox("Occupation Risk", OCCUPATION_RISK, key=f"{key_prefix}_occrisk")
        blood_pressure = st.selectbox("Blood Pressure", ["Normal", "Elevated", "High"],
                                       key=f"{key_prefix}_bp")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], key=f"{key_prefix}_diabetes")
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], key=f"{key_prefix}_heart")

    with col3:
        family_history = st.selectbox("Family History of Illness", ["No", "Yes"],
                                       key=f"{key_prefix}_family")
        health_checkups = st.slider("Health Checkups / Year", 0, 6, 1, key=f"{key_prefix}_checkups")
        sleep_hours = st.slider("Average Sleep Hours", 3.0, 11.0, 7.0, step=0.5,
                                 key=f"{key_prefix}_sleep")
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, key=f"{key_prefix}_stress")
        diet_quality = st.selectbox("Diet Quality", DIET_QUALITY, key=f"{key_prefix}_diet")

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    st.markdown(
        f"<div class='badge badge-purple'>Calculated BMI: {bmi} ({bmi_category(bmi)})</div>",
        unsafe_allow_html=True,
    )

    return {
        "Age": age, "Gender": gender, "Height_cm": height_cm, "Weight_kg": weight_kg,
        "BMI": bmi, "Smoking": smoking, "Alcohol": alcohol, "Exercise_Days": exercise_days,
        "Occupation_Risk": occupation_risk, "Income": income, "Region": region,
        "Blood_Pressure": blood_pressure, "Diabetes": diabetes, "Heart_Disease": heart_disease,
        "Family_History": family_history, "Dependents": dependents,
        "Health_Checkups": health_checkups, "Sleep_Hours": sleep_hours,
        "Stress_Level": stress_level, "Diet_Quality": diet_quality,
    }


def run_full_prediction(inputs: dict) -> dict:
    """Run the ensemble model + all AI features for a given input dict."""
    input_df = pd.DataFrame([inputs])
    X_row, _ = prepare_ml_dataset(input_df.assign(Premium=0))
    ensemble_value, individual_preds, agreement, elapsed = ensemble_predict(model_bundle, X_row)

    health_score = compute_health_score(inputs)
    risk_score = compute_risk_score(health_score)
    r_category = risk_category(risk_score)
    plan = recommend_plan(risk_score, inputs["Income"])
    fraud = detect_fraud_signal(inputs, ensemble_value)

    recommendations = []
    if inputs["Smoking"] == "Yes":
        recommendations.append("Consider a smoking cessation program to significantly reduce your premium.")
    if inputs["Exercise_Days"] < 3:
        recommendations.append("Increase weekly exercise to at least 3-4 days to improve your health score.")
    if inputs["Sleep_Hours"] < 6:
        recommendations.append("Aim for 7-8 hours of sleep per night to reduce health risk.")
    if inputs["Stress_Level"] > 7:
        recommendations.append("Explore stress management techniques such as meditation or counseling.")
    if inputs["Diet_Quality"] in ["Poor", "Average"]:
        recommendations.append("Improve diet quality with more whole foods to lower long-term risk.")
    if inputs["Health_Checkups"] == 0:
        recommendations.append("Schedule at least one annual health checkup for early risk detection.")
    if not recommendations:
        recommendations.append("Your lifestyle profile looks great — keep maintaining these healthy habits!")

    forecast = forecast_premiums({"base_premium": ensemble_value})

    return {
        "ensemble": ensemble_value,
        "individual": individual_preds,
        "agreement": agreement,
        "elapsed": elapsed,
        "health_score": health_score,
        "risk_score": risk_score,
        "risk_category": r_category,
        "plan": plan,
        "fraud": fraud,
        "recommendations": recommendations,
        "forecast": forecast,
        "X_row": X_row,
    }


# ==========================================================================
# PAGE: HOME
# ==========================================================================
def page_home():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">🛡️ Insurance Premium Prediction Pro AI</div>
            <div class="hero-sub">
                An advanced AI-powered platform that predicts personalized insurance premiums,
                evaluates your health &amp; risk profile, simulates lifestyle changes, and
                generates professional reports — all in real time, powered by an ensemble
                of machine learning models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (f"{len(raw_df):,}", "Training Records"),
        (f"{len(model_bundle['models'])}", "Active AI Models"),
        (f"Rs. {raw_df['Premium'].mean():,.0f}", "Avg. Premium"),
        (f"{model_bundle['metrics'].get('Random Forest', {}).get('R2', 0) * 100:.1f}%", "Model Accuracy (R²)"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-value'>{val}</div>"
                f"<div class='kpi-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>✨ Platform Capabilities</div>", unsafe_allow_html=True)
    features = [
        ("🧮", "Ensemble Prediction", "Combines multiple ML models for accurate, confident premium estimates."),
        ("❤️", "Health & Risk Scoring", "0-100 health score with automatic risk categorization."),
        ("🎯", "Lifestyle Simulator", "See how quitting smoking, exercising more, or sleeping better saves money."),
        ("🧠", "Explainable AI", "Understand exactly which factors drive your premium up or down."),
        ("📊", "Interactive Dashboard", "Deep dive into population trends, correlations, and distributions."),
        ("📄", "Instant Reports", "Download a professional PDF/QR health passport in one click."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"<div class='glass-card'><h4>{icon} {title}</h4>"
                f"<p style='color:#c8c8e0; font-size:14px;'>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>📈 Premium Distribution Snapshot</div>", unsafe_allow_html=True)
    fig = px.histogram(raw_df, x="Premium", nbins=40, color_discrete_sequence=["#7f5af0"])
    fig = themed_layout(fig, height=380, title="Population Premium Distribution")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE: PREMIUM PREDICTION
# ==========================================================================
def page_prediction():
    st.markdown("<div class='section-title'>🧮 Premium Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    inputs = render_input_form("predict")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔮 Predict My Premium", use_container_width=True):
        try:
            result = run_full_prediction(inputs)
            st.session_state["last_prediction"] = result
            st.session_state["last_inputs"] = inputs
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

    result = st.session_state.get("last_prediction")
    inputs_saved = st.session_state.get("last_inputs")

    if result and inputs_saved:
        st.markdown("<div class='section-title'>💰 Prediction Results</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            (f"Rs. {result['ensemble']:,.0f}", "Predicted Annual Premium"),
            (f"{result['agreement']:.1f}%", "Model Agreement"),
            (f"{result['elapsed']*1000:.1f} ms", "Prediction Time"),
            (f"{len(result['individual'])}", "Models Used"),
        ]
        for col, (val, label) in zip([c1, c2, c3, c4], kpis):
            with col:
                st.markdown(
                    f"<div class='kpi-card'><div class='kpi-value'>{val}</div>"
                    f"<div class='kpi-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

        colA, colB = st.columns(2)
        with colA:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(result["risk_score"], "Risk Score"), use_container_width=True)
            badge_class = "badge-green" if result["risk_score"] < 25 else \
                "badge-orange" if result["risk_score"] < 50 else "badge-red"
            st.markdown(f"<span class='badge {badge_class}'>{result['risk_category']}</span>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(result["health_score"], "Health Score",
                             color_stops=(("#ef476f", 0, 40), ("#ff8906", 40, 70), ("#2cb67d", 70, 100))),
                             use_container_width=True)
            st.markdown(f"<span class='badge badge-purple'>Recommended: {result['plan']}</span>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>🤖 Model-by-Model Breakdown</div>", unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "Model": list(result["individual"].keys()),
            "Predicted Premium": list(result["individual"].values()),
        })
        fig = px.bar(comp_df, x="Model", y="Predicted Premium", color="Model",
                     color_discrete_sequence=px.colors.qualitative.Bold, text_auto=".0f")
        fig = themed_layout(fig, height=380, title="Individual Model Predictions")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>🚩 Fraud / Anomaly Signal</div>", unsafe_allow_html=True)
        fraud = result["fraud"]
        badge_class = {"Low": "badge-green", "Medium": "badge-orange", "High": "badge-red"}[fraud["risk_level"]]
        st.markdown(f"<span class='badge {badge_class}'>Anomaly Risk: {fraud['risk_level']}</span>",
                    unsafe_allow_html=True)
        if fraud["flags"]:
            for f in fraud["flags"]:
                st.warning(f"⚠️ {f}")
        else:
            st.success("✅ No anomaly signals detected in this profile.")

        st.markdown("<div class='section-title'>💡 Personalized Recommendations</div>", unsafe_allow_html=True)
        for rec in result["recommendations"]:
            st.info(f"💡 {rec}")


# ==========================================================================
# PAGE: DASHBOARD
# ==========================================================================
def page_dashboard():
    st.markdown("<div class='section-title'>📊 Population Dashboard</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (f"{len(raw_df):,}", "Total Population"),
        (f"Rs. {raw_df['Premium'].mean():,.0f}", "Average Premium"),
        (f"{raw_df['BMI'].mean():.1f}", "Average BMI"),
        (f"{(raw_df['Smoking']=='Yes').mean()*100:.1f}%", "Smoking Rate"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-value'>{val}</div>"
                f"<div class='kpi-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Regional & Occupation", "Correlations", "Composition"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            fig = px.histogram(raw_df, x="Age", nbins=30, color_discrete_sequence=["#2cb67d"])
            st.plotly_chart(themed_layout(fig, title="Age Distribution"), use_container_width=True)
        with colB:
            fig = px.histogram(raw_df, x="BMI", nbins=30, color_discrete_sequence=["#ff8906"])
            st.plotly_chart(themed_layout(fig, title="BMI Distribution"), use_container_width=True)

        fig = px.scatter(raw_df.sample(min(1200, len(raw_df))), x="BMI", y="Premium", color="Smoking",
                          color_discrete_sequence=["#2cb67d", "#ef476f"], opacity=0.6)
        st.plotly_chart(themed_layout(fig, height=440, title="BMI vs Premium (colored by Smoking)"),
                         use_container_width=True)

    with tab2:
        colA, colB = st.columns(2)
        with colA:
            region_avg = raw_df.groupby("Region")["Premium"].mean().reset_index()
            fig = px.bar(region_avg, x="Region", y="Premium", color="Region",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(themed_layout(fig, title="Average Premium by Region"), use_container_width=True)
        with colB:
            occ_avg = raw_df.groupby("Occupation_Risk")["Premium"].mean().reset_index()
            fig = px.bar(occ_avg, x="Occupation_Risk", y="Premium", color="Occupation_Risk",
                         color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(themed_layout(fig, title="Average Premium by Occupation Risk"), use_container_width=True)

        fig = px.treemap(raw_df, path=["Region", "Occupation_Risk"], values="Premium",
                          color="Premium", color_continuous_scale="Purples")
        fig = themed_layout(fig, height=460, title="Premium Treemap: Region → Occupation Risk")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_cols = ["Age", "BMI", "Exercise_Days", "Income", "Dependents",
                         "Health_Checkups", "Sleep_Hours", "Stress_Level", "Premium"]
        corr = raw_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig = themed_layout(fig, height=520, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        colA, colB = st.columns(2)
        with colA:
            gender_counts = raw_df["Gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            fig = px.pie(gender_counts, names="Gender", values="Count", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(themed_layout(fig, height=380, title="Gender Composition"), use_container_width=True)
        with colB:
            fig = px.sunburst(raw_df, path=["Diet_Quality", "Blood_Pressure"], values="Premium",
                               color="Premium", color_continuous_scale="Tealgrn")
            st.plotly_chart(themed_layout(fig, height=380, title="Diet & Blood Pressure Sunburst"),
                             use_container_width=True)


# ==========================================================================
# PAGE: ANALYTICS
# ==========================================================================
def page_analytics():
    st.markdown("<div class='section-title'>📈 Advanced Analytics</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🏆 Model Performance Comparison")
    metrics_df = pd.DataFrame(model_bundle["metrics"]).T.reset_index()
    metrics_df.columns = ["Model", "MAE", "R2"]
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(metrics_df, x="Model", y="MAE", color="Model", text_auto=".0f",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(themed_layout(fig, title="Mean Absolute Error (lower is better)"),
                         use_container_width=True)
    with colB:
        fig = px.bar(metrics_df, x="Model", y="R2", color="Model", text_auto=".3f",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(themed_layout(fig, title="R² Score (higher is better)"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🌟 Global Feature Importance (Random Forest)")
    try:
        rf = model_bundle["models"]["Random Forest"]["model"]
        importance_df = pd.DataFrame({
            "Feature": model_bundle["feature_columns"],
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=True).tail(15)
        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Purples")
        st.plotly_chart(themed_layout(fig, height=520, title="Top 15 Most Influential Features"),
                         use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚖️ Fairness Analysis</div>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    with colA:
        gender_fair = raw_df.groupby("Gender")["Premium"].mean().reset_index()
        fig = px.bar(gender_fair, x="Gender", y="Premium", color="Gender",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(themed_layout(fig, title="Avg Premium by Gender"), use_container_width=True)
    with colB:
        region_fair = raw_df.groupby("Region")["Premium"].mean().reset_index()
        fig = px.bar(region_fair, x="Region", y="Premium", color="Region",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(themed_layout(fig, title="Avg Premium by Region"), use_container_width=True)
    with colC:
        age_bins = pd.cut(raw_df["Age"], bins=[17, 30, 45, 60, 90],
                           labels=["18-30", "31-45", "46-60", "61-90"])
        age_fair = raw_df.groupby(age_bins, observed=True)["Premium"].mean().reset_index()
        fig = px.bar(age_fair, x="Age", y="Premium", color="Age",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(themed_layout(fig, title="Avg Premium by Age Group"), use_container_width=True)

    max_gap = gender_fair["Premium"].max() - gender_fair["Premium"].min()
    gap_pct = (max_gap / gender_fair["Premium"].mean()) * 100
    if gap_pct < 5:
        st.success(f"✅ Fairness summary: Gender premium gap is minimal ({gap_pct:.1f}%), suggesting balanced pricing.")
    elif gap_pct < 15:
        st.info(f"ℹ️ Fairness summary: Moderate gender premium gap detected ({gap_pct:.1f}%). Monitor for bias.")
    else:
        st.warning(f"⚠️ Fairness summary: Significant gender premium gap detected ({gap_pct:.1f}%). Review model inputs.")


# ==========================================================================
# PAGE: AI INSIGHTS
# ==========================================================================
def page_ai_insights():
    st.markdown("<div class='section-title'>🧠 AI Insights &amp; Explainability</div>", unsafe_allow_html=True)

    result = st.session_state.get("last_prediction")
    inputs_saved = st.session_state.get("last_inputs")

    if not result:
        st.info("ℹ️ Run a prediction on the **🧮 Premium Prediction** page first to unlock personalized AI insights.")
        return

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Why This Premium? (Explainable AI)")
    if SHAP_AVAILABLE:
        explanation = shap_explanation(model_bundle, result["X_row"])
        if explanation:
            exp_df = pd.DataFrame(explanation)
            fig = px.bar(exp_df, x="shap_value", y="feature", orientation="h",
                         color="shap_value", color_continuous_scale="RdBu_r")
            st.plotly_chart(themed_layout(fig, height=420, title="SHAP Feature Contributions"),
                             use_container_width=True)
        else:
            st.warning("SHAP explanation unavailable for this input; falling back to manual explanation.")
            explanation = None
    else:
        explanation = None

    if not SHAP_AVAILABLE or explanation is None:
        manual_exp = manual_feature_explanation(model_bundle, result["X_row"])
        for e in manual_exp:
            st.markdown(
                f"<div class='badge badge-purple'>{e['feature']} ({e['importance']}%) — "
                f"{e['direction']} the premium</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 👥 Similar Customer Finder (KNN)")
    similar = find_similar_customers(model_bundle, raw_df, result["X_row"])
    if not similar.empty:
        st.dataframe(
            similar[["Age", "Gender", "BMI", "Smoking", "Region", "Premium"]].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.warning("Could not find similar customers for this profile.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📆 5-Year Premium Forecast")
    years, current, improved, poor = result["forecast"]
    forecast_df = pd.DataFrame({
        "Year": years * 3,
        "Premium": current + improved + poor,
        "Scenario": ["Current"] * len(years) + ["Improved Lifestyle"] * len(years) + ["Poor Lifestyle"] * len(years),
    })
    fig = px.line(forecast_df, x="Year", y="Premium", color="Scenario", markers=True,
                  color_discrete_sequence=["#7f5af0", "#2cb67d", "#ef476f"])
    st.plotly_chart(themed_layout(fig, height=420, title="5-Year Forecast by Lifestyle Scenario"),
                     use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🕸️ Risk Profile Radar")
    if inputs_saved:
        radar_categories = ["BMI Risk", "Smoking Risk", "Lifestyle Risk", "Medical Risk", "Stress Risk"]
        bmi_r = min(100, abs(inputs_saved["BMI"] - 22) * 6)
        smoke_r = 100 if inputs_saved["Smoking"] == "Yes" else 10
        lifestyle_r = max(0, 100 - inputs_saved["Exercise_Days"] * 14)
        medical_r = (30 if inputs_saved["Diabetes"] == "Yes" else 0) + \
                    (35 if inputs_saved["Heart_Disease"] == "Yes" else 0) + \
                    (20 if inputs_saved["Blood_Pressure"] == "High" else 0)
        stress_r = inputs_saved["Stress_Level"] * 10
        values = [bmi_r, smoke_r, lifestyle_r, min(medical_r, 100), stress_r]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=radar_categories, fill="toself",
                                       line_color="#7f5af0", fillcolor="rgba(127,90,240,0.35)"))
        fig.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.15)")),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#EDEDF7", height=440,
            title="Personal Risk Radar",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE: LIFESTYLE SIMULATOR
# ==========================================================================
def page_lifestyle_simulator():
    st.markdown("<div class='section-title'>🎯 Lifestyle Simulator</div>", unsafe_allow_html=True)

    inputs_saved = st.session_state.get("last_inputs")
    if not inputs_saved:
        st.info("ℹ️ Run a prediction on the **🧮 Premium Prediction** page first, then come back to simulate changes.")
        return

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Adjust Your Lifestyle")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_weight = st.slider("New Weight (kg)", 35, 180, int(inputs_saved["Weight_kg"]), key="sim_weight")
        new_smoking = st.selectbox("Smoking", ["No", "Yes"],
                                    index=["No", "Yes"].index(inputs_saved["Smoking"]), key="sim_smoking")
    with col2:
        new_exercise = st.slider("Exercise Days / Week", 0, 7, inputs_saved["Exercise_Days"], key="sim_exercise")
        new_sleep = st.slider("Sleep Hours", 3.0, 11.0, float(inputs_saved["Sleep_Hours"]), step=0.5,
                               key="sim_sleep")
    with col3:
        new_diet = st.selectbox("Diet Quality", DIET_QUALITY,
                                 index=DIET_QUALITY.index(inputs_saved["Diet_Quality"]), key="sim_diet")
        new_stress = st.slider("Stress Level", 1, 10, inputs_saved["Stress_Level"], key="sim_stress")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⚡ Simulate New Premium", use_container_width=True):
        try:
            new_inputs = dict(inputs_saved)
            new_inputs["Weight_kg"] = new_weight
            new_inputs["BMI"] = round(new_weight / ((inputs_saved["Height_cm"] / 100) ** 2), 2)
            new_inputs["Smoking"] = new_smoking
            new_inputs["Exercise_Days"] = new_exercise
            new_inputs["Sleep_Hours"] = new_sleep
            new_inputs["Diet_Quality"] = new_diet
            new_inputs["Stress_Level"] = new_stress

            old_result = run_full_prediction(inputs_saved)
            new_result = run_full_prediction(new_inputs)

            old_premium = old_result["ensemble"]
            new_premium = new_result["ensemble"]
            saved = old_premium - new_premium
            saved_pct = (saved / old_premium) * 100 if old_premium else 0

            st.markdown("<div class='section-title'>💰 Simulation Results</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            for col, (val, label) in zip(
                [c1, c2, c3],
                [
                    (f"Rs. {old_premium:,.0f}", "Old Premium"),
                    (f"Rs. {new_premium:,.0f}", "New Premium"),
                    (f"Rs. {saved:,.0f} ({saved_pct:.1f}%)", "Money Saved"),
                ],
            ):
                with col:
                    st.markdown(
                        f"<div class='kpi-card'><div class='kpi-value'>{val}</div>"
                        f"<div class='kpi-label'>{label}</div></div>",
                        unsafe_allow_html=True,
                    )

            comp_df = pd.DataFrame({"Scenario": ["Old Lifestyle", "New Lifestyle"],
                                     "Premium": [old_premium, new_premium]})
            fig = px.bar(comp_df, x="Scenario", y="Premium", color="Scenario", text_auto=".0f",
                         color_discrete_sequence=["#ef476f", "#2cb67d"])
            st.plotly_chart(themed_layout(fig, height=400, title="Premium Comparison"), use_container_width=True)

            if saved > 0:
                st.success(f"🎉 Great choice! These lifestyle changes could save you Rs. {saved:,.0f} per year "
                           f"({saved_pct:.1f}%).")
            elif saved < 0:
                st.warning(f"⚠️ These changes would increase your premium by Rs. {abs(saved):,.0f} per year.")
            else:
                st.info("ℹ️ These changes have negligible impact on your premium.")
        except Exception as e:
            st.error(f"⚠️ Simulation failed: {e}")


# ==========================================================================
# PAGE: HEALTH SCORE
# ==========================================================================
def page_health_score():
    st.markdown("<div class='section-title'>❤️ Health Score Analyzer</div>", unsafe_allow_html=True)

    result = st.session_state.get("last_prediction")
    if not result:
        st.info("ℹ️ Run a prediction on the **🧮 Premium Prediction** page first to view your health score.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(progress_ring(result["health_score"], "Health Score", "#2cb67d"),
                         use_container_width=True)
        st.markdown("<p class='ring-label'>Overall Health Score</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(progress_ring(result["risk_score"], "Risk Score", "#ef476f"),
                         use_container_width=True)
        st.markdown("<p class='ring-label'>Overall Risk Score</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(progress_ring(result["agreement"], "Model Agreement", "#7f5af0"),
                         use_container_width=True)
        st.markdown("<p class='ring-label'>AI Confidence</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='glass-card' style='text-align:center;'>"
        f"<h3>Risk Category: <span class='badge badge-purple'>{result['risk_category']}</span></h3>"
        f"<p style='color:#c8c8e0;'>Recommended Plan: <b>{result['plan']}</b></p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>💡 Personalized Recommendations</div>", unsafe_allow_html=True)
    for rec in result["recommendations"]:
        st.info(f"💡 {rec}")


# ==========================================================================
# PAGE: PDF REPORT
# ==========================================================================
def page_pdf_report():
    st.markdown("<div class='section-title'>📄 Downloadable Report</div>", unsafe_allow_html=True)

    result = st.session_state.get("last_prediction")
    inputs_saved = st.session_state.get("last_inputs")

    if not result or not inputs_saved:
        st.info("ℹ️ Run a prediction on the **🧮 Premium Prediction** page first to generate a report.")
        return

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Report Preview")
    st.write(f"**Predicted Premium:** Rs. {result['ensemble']:,.2f}")
    st.write(f"**Health Score:** {result['health_score']:.1f} / 100")
    st.write(f"**Risk Category:** {result['risk_category']}")
    st.write(f"**Recommended Plan:** {result['plan']}")
    st.markdown("</div>", unsafe_allow_html=True)

    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = generate_pdf_report(inputs_saved, result)
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"insurance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation failed ({e}); falling back to text report.")
            txt_bytes = generate_txt_report(inputs_saved, result)
            st.download_button(
                "⬇️ Download TXT Report",
                data=txt_bytes,
                file_name=f"insurance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.info("ℹ️ ReportLab not available — generating a plain-text report instead.")
        txt_bytes = generate_txt_report(inputs_saved, result)
        st.download_button(
            "⬇️ Download TXT Report",
            data=txt_bytes,
            file_name=f"insurance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("<div class='section-title'>📱 Digital Health Passport QR Code</div>", unsafe_allow_html=True)
    if QRCODE_AVAILABLE:
        try:
            payload = (
                f"InsurAI Pro Health Passport\n"
                f"Health Score: {result['health_score']:.1f}\n"
                f"Risk Category: {result['risk_category']}\n"
                f"Premium: Rs. {result['ensemble']:,.0f}\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d')}"
            )
            qr_bytes = generate_qr_code(payload)
            if qr_bytes:
                st.image(qr_bytes, width=220, caption="Scan for a summary of your health passport")
            else:
                st.warning("QR code generation failed unexpectedly.")
        except Exception as e:
            st.warning(f"QR generation failed: {e}")
    else:
        st.info("ℹ️ QR code library not available — this feature is hidden.")


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
def page_about():
    st.markdown("<div class='section-title'>ℹ️ About This Application</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='glass-card'>
        <h4>🛡️ Insurance Premium Prediction Pro AI</h4>
        <p style='color:#c8c8e0;'>
        This application is a fully self-contained, production-ready demonstration of an
        AI-powered insurance premium prediction platform. It automatically generates a
        realistic synthetic dataset, trains an ensemble of machine learning models, and
        provides explainable, actionable insights — with no external dataset or pickle
        files required.
        </p>
        <p style='color:#c8c8e0;'>
        <b>Core Technologies:</b> Streamlit, scikit-learn, Plotly, Random Forest, and
        optionally XGBoost, LightGBM, CatBoost, and SHAP when available in the environment.
        </p>
        <p style='color:#c8c8e0;'>
        <b>Disclaimer:</b> All predictions are generated from synthetic data for
        demonstration purposes only and should not be used for actual insurance
        underwriting decisions.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>🧩 Optional Library Status</div>", unsafe_allow_html=True)
    status_data = [
        ("XGBoost", XGBOOST_AVAILABLE),
        ("LightGBM", LIGHTGBM_AVAILABLE),
        ("CatBoost", CATBOOST_AVAILABLE),
        ("SHAP", SHAP_AVAILABLE),
        ("ReportLab (PDF)", REPORTLAB_AVAILABLE),
        ("QR Code", QRCODE_AVAILABLE),
    ]
    cols = st.columns(3)
    for i, (name, available) in enumerate(status_data):
        with cols[i % 3]:
            badge = "badge-green" if available else "badge-red"
            label = "Available" if available else "Not Installed"
            st.markdown(
                f"<div class='glass-card' style='text-align:center;'>"
                f"<h5>{name}</h5><span class='badge {badge}'>{label}</span></div>",
                unsafe_allow_html=True,
            )


# ==========================================================================
# PAGE ROUTER
# ==========================================================================
try:
    if page == "🏠 Home":
        page_home()
    elif page == "🧮 Premium Prediction":
        page_prediction()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "📈 Analytics":
        page_analytics()
    elif page == "🧠 AI Insights":
        page_ai_insights()
    elif page == "🎯 Lifestyle Simulator":
        page_lifestyle_simulator()
    elif page == "❤️ Health Score":
        page_health_score()
    elif page == "📄 PDF Report":
        page_pdf_report()
    elif page == "ℹ️ About":
        page_about()
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred while rendering this page: {e}")
    st.info("Please try navigating to another page or refreshing the application.")
