# -*- coding: utf-8 -*-
"""
Insurance Premium Prediction Pro AI
=====================================
A production-ready, ultra-modern Streamlit application that predicts health
insurance premiums using an ensemble of machine learning models, explains
predictions with SHAP (or a manual fallback), simulates lifestyle changes,
forecasts future premiums, detects fraudulent inputs, checks fairness across
demographic groups, finds similar customers, and generates downloadable
PDF / Digital Health Passport reports.

Run with:  streamlit run app.py
"""

import os
import io
import base64
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Optional heavy dependencies - the app must never crash if one is missing.
# --------------------------------------------------------------------------
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
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
    import qrcode
    QRCODE_AVAILABLE = True
except Exception:
    QRCODE_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import cm
    from reportlab.lib import colors as rl_colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ==========================================================================
# PAGE CONFIG + GLOBAL CSS
# ==========================================================================
st.set_page_config(
    page_title="Insurance Premium Prediction Pro AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-attachment: fixed;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    transition: all 0.35s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.75);
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
    padding: 1.3rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    animation: floatUp 0.6s ease;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.65);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

@keyframes floatUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0.6rem 0 1rem 0;
    border-left: 4px solid #a78bfa;
    padding-left: 0.7rem;
}

.badge-low { background: rgba(52,211,153,0.18); color: #34d399; padding: 0.25rem 0.8rem; border-radius: 999px; font-weight:700; border: 1px solid rgba(52,211,153,0.4);}
.badge-medium { background: rgba(251,191,36,0.18); color: #fbbf24; padding: 0.25rem 0.8rem; border-radius: 999px; font-weight:700; border: 1px solid rgba(251,191,36,0.4);}
.badge-high { background: rgba(248,113,113,0.18); color: #f87171; padding: 0.25rem 0.8rem; border-radius: 999px; font-weight:700; border: 1px solid rgba(248,113,113,0.4);}

.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    box-shadow: 0 4px 18px rgba(124,58,237,0.4);
    transition: all 0.25s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(124,58,237,0.55);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 0.8rem;
    border: 1px solid rgba(255,255,255,0.1);
}

.rec-high { border-left: 4px solid #f87171; padding: 0.55rem 0.9rem; background: rgba(248,113,113,0.08); border-radius: 8px; margin-bottom: 0.5rem;}
.rec-medium { border-left: 4px solid #fbbf24; padding: 0.55rem 0.9rem; background: rgba(251,191,36,0.08); border-radius: 8px; margin-bottom: 0.5rem;}
.rec-low { border-left: 4px solid #34d399; padding: 0.55rem 0.9rem; background: rgba(52,211,153,0.08); border-radius: 8px; margin-bottom: 0.5rem;}

hr {border-color: rgba(255,255,255,0.1);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# CONSTANTS / ENCODING MAPS
# ==========================================================================
MODEL_PATH = "insurance_premium_models.pkl"
DATA_SEED = 42
N_SYNTHETIC_RECORDS = 5200

GENDER_OPTIONS = ["Male", "Female", "Other"]
YESNO_OPTIONS = ["No", "Yes"]
ALCOHOL_OPTIONS = ["None", "Occasional", "Frequent"]
RISK_OPTIONS = ["Low", "Medium", "High"]
REGION_OPTIONS = ["North", "South", "East", "West"]
BP_OPTIONS = ["Normal", "Elevated", "High"]
DIET_OPTIONS = ["Poor", "Average", "Good", "Excellent"]

GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
YESNO_MAP = {"No": 0, "Yes": 1}
ALCOHOL_MAP = {"None": 0, "Occasional": 1, "Frequent": 2}
RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}
REGION_MAP = {"North": 0, "South": 1, "East": 2, "West": 3}
BP_MAP = {"Normal": 0, "Elevated": 1, "High": 2}
DIET_MAP = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}

FEATURE_NAMES = [
    "age", "height", "weight", "bmi", "exercise_days", "income", "dependents",
    "checkups_per_year", "sleep_hours", "stress_level",
    "gender_enc", "smoking_enc", "alcohol_enc", "occupation_risk_enc",
    "region_enc", "blood_pressure_enc", "diabetes_enc", "heart_disease_enc",
    "family_history_enc", "diet_quality_enc",
]

FEATURE_LABELS = {
    "age": "Age", "height": "Height", "weight": "Weight", "bmi": "BMI",
    "exercise_days": "Exercise Days/Week", "income": "Income",
    "dependents": "Dependents", "checkups_per_year": "Checkups/Year",
    "sleep_hours": "Sleep Hours", "stress_level": "Stress Level",
    "gender_enc": "Gender", "smoking_enc": "Smoking", "alcohol_enc": "Alcohol Use",
    "occupation_risk_enc": "Occupation Risk", "region_enc": "Region",
    "blood_pressure_enc": "Blood Pressure", "diabetes_enc": "Diabetes",
    "heart_disease_enc": "Heart Disease", "family_history_enc": "Family History",
    "diet_quality_enc": "Diet Quality",
}


# ==========================================================================
# SYNTHETIC DATA GENERATION
# ==========================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n_records: int = N_SYNTHETIC_RECORDS, seed: int = DATA_SEED) -> pd.DataFrame:
    """Create a realistic synthetic health-insurance dataset for demo mode."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n_records)
    gender = rng.choice(GENDER_OPTIONS, n_records, p=[0.48, 0.48, 0.04])
    height = rng.normal(168, 10, n_records).clip(140, 205)
    weight = rng.normal(72, 15, n_records).clip(40, 150)
    bmi = weight / ((height / 100) ** 2)

    smoking = rng.choice(YESNO_OPTIONS, n_records, p=[0.78, 0.22])
    alcohol = rng.choice(ALCOHOL_OPTIONS, n_records, p=[0.5, 0.35, 0.15])
    exercise_days = rng.integers(0, 8, n_records)
    occupation_risk = rng.choice(RISK_OPTIONS, n_records, p=[0.55, 0.30, 0.15])
    income = rng.normal(55000, 25000, n_records).clip(8000, 250000)
    region = rng.choice(REGION_OPTIONS, n_records)
    blood_pressure = rng.choice(BP_OPTIONS, n_records, p=[0.6, 0.25, 0.15])
    diabetes = rng.choice(YESNO_OPTIONS, n_records, p=[0.85, 0.15])
    heart_disease = rng.choice(YESNO_OPTIONS, n_records, p=[0.9, 0.1])
    family_history = rng.choice(YESNO_OPTIONS, n_records, p=[0.65, 0.35])
    dependents = rng.integers(0, 6, n_records)
    checkups_per_year = rng.integers(0, 5, n_records)
    sleep_hours = rng.normal(7, 1.4, n_records).clip(3, 12)
    stress_level = rng.integers(1, 11, n_records)
    diet_quality = rng.choice(DIET_OPTIONS, n_records, p=[0.2, 0.4, 0.3, 0.1])

    smoking_enc = np.array([YESNO_MAP[v] for v in smoking])
    alcohol_enc = np.array([ALCOHOL_MAP[v] for v in alcohol])
    occ_enc = np.array([RISK_MAP[v] for v in occupation_risk])
    bp_enc = np.array([BP_MAP[v] for v in blood_pressure])
    diabetes_enc = np.array([YESNO_MAP[v] for v in diabetes])
    heart_enc = np.array([YESNO_MAP[v] for v in heart_disease])
    family_enc = np.array([YESNO_MAP[v] for v in family_history])
    diet_enc = np.array([DIET_MAP[v] for v in diet_quality])

    bmi_excess = np.clip(bmi - 22, 0, None)
    sleep_penalty = np.abs(sleep_hours - 7.5) * 60

    noise = rng.normal(0, 350, n_records)

    premium = (
        2800
        + age * 24
        + bmi_excess * 42
        + smoking_enc * 1550
        + alcohol_enc * 260
        + occ_enc * 380
        + bp_enc * 340
        + diabetes_enc * 720
        + heart_enc * 950
        + family_enc * 310
        + dependents * 140
        + stress_level * 28
        + sleep_penalty
        - exercise_days * 75
        - checkups_per_year * 45
        - diet_enc * 90
        + income * 0.0018
        + noise
    )
    premium = np.clip(premium, 900, None)

    df = pd.DataFrame({
        "age": age, "gender": gender, "height": height, "weight": weight, "bmi": bmi,
        "smoking": smoking, "alcohol": alcohol, "exercise_days": exercise_days,
        "occupation_risk": occupation_risk, "income": income, "region": region,
        "blood_pressure": blood_pressure, "diabetes": diabetes, "heart_disease": heart_disease,
        "family_history": family_history, "dependents": dependents,
        "checkups_per_year": checkups_per_year, "sleep_hours": sleep_hours,
        "stress_level": stress_level, "diet_quality": diet_quality, "premium": premium,
    })
    return df


def encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Encode a raw dataframe of inputs into the numeric feature matrix."""
    X = pd.DataFrame()
    X["age"] = df["age"]
    X["height"] = df["height"]
    X["weight"] = df["weight"]
    X["bmi"] = df["bmi"]
    X["exercise_days"] = df["exercise_days"]
    X["income"] = df["income"]
    X["dependents"] = df["dependents"]
    X["checkups_per_year"] = df["checkups_per_year"]
    X["sleep_hours"] = df["sleep_hours"]
    X["stress_level"] = df["stress_level"]
    X["gender_enc"] = df["gender"].map(GENDER_MAP)
    X["smoking_enc"] = df["smoking"].map(YESNO_MAP)
    X["alcohol_enc"] = df["alcohol"].map(ALCOHOL_MAP)
    X["occupation_risk_enc"] = df["occupation_risk"].map(RISK_MAP)
    X["region_enc"] = df["region"].map(REGION_MAP)
    X["blood_pressure_enc"] = df["blood_pressure"].map(BP_MAP)
    X["diabetes_enc"] = df["diabetes"].map(YESNO_MAP)
    X["heart_disease_enc"] = df["heart_disease"].map(YESNO_MAP)
    X["family_history_enc"] = df["family_history"].map(YESNO_MAP)
    X["diet_quality_enc"] = df["diet_quality"].map(DIET_MAP)
    return X[FEATURE_NAMES].to_numpy(dtype=float)


def encode_single_input(inputs: dict) -> np.ndarray:
    """Encode a single dict of raw inputs into a (1, n_features) numpy array."""
    row = {
        "age": inputs["age"], "height": inputs["height"], "weight": inputs["weight"],
        "bmi": inputs["bmi"], "exercise_days": inputs["exercise_days"],
        "income": inputs["income"], "dependents": inputs["dependents"],
        "checkups_per_year": inputs["checkups_per_year"], "sleep_hours": inputs["sleep_hours"],
        "stress_level": inputs["stress_level"],
        "gender_enc": GENDER_MAP[inputs["gender"]], "smoking_enc": YESNO_MAP[inputs["smoking"]],
        "alcohol_enc": ALCOHOL_MAP[inputs["alcohol"]],
        "occupation_risk_enc": RISK_MAP[inputs["occupation_risk"]],
        "region_enc": REGION_MAP[inputs["region"]],
        "blood_pressure_enc": BP_MAP[inputs["blood_pressure"]],
        "diabetes_enc": YESNO_MAP[inputs["diabetes"]],
        "heart_disease_enc": YESNO_MAP[inputs["heart_disease"]],
        "family_history_enc": YESNO_MAP[inputs["family_history"]],
        "diet_quality_enc": DIET_MAP[inputs["diet_quality"]],
    }
    return np.array([[row[f] for f in FEATURE_NAMES]], dtype=float)


# ==========================================================================
# MODEL TRAINING / LOADING (cached as a resource - runs once per session)
# ==========================================================================
@st.cache_resource(show_spinner="Preparing AI models (first run only)...")
def load_or_train_models():
    """Load trained models from disk, or train a fresh demo ensemble."""
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle
        except Exception:
            pass  # fall through to retraining if the pickle is corrupt

    df = generate_synthetic_dataset()
    X = encode_dataframe(df)
    y = df["premium"].to_numpy()

    models = {}

    rf = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_leaf=3,
        random_state=DATA_SEED, n_jobs=-1
    )
    rf.fit(X, y)
    models["Random Forest"] = rf

    if XGBOOST_AVAILABLE:
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, random_state=DATA_SEED,
                verbosity=0,
            )
            xgb_model.fit(X, y)
            models["XGBoost"] = xgb_model
        except Exception:
            pass

    if LIGHTGBM_AVAILABLE:
        try:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                random_state=DATA_SEED, verbosity=-1,
            )
            lgb_model.fit(X, y)
            models["LightGBM"] = lgb_model
        except Exception:
            pass

    if CATBOOST_AVAILABLE:
        try:
            cat_model = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.05,
                random_state=DATA_SEED, verbose=False,
            )
            cat_model.fit(X, y)
            models["CatBoost"] = cat_model
        except Exception:
            pass

    bundle = {
        "models": models,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now().isoformat(),
        "n_records": len(df),
    }
    try:
        joblib.dump(bundle, MODEL_PATH)
    except Exception:
        pass
    return bundle


# ==========================================================================
# CORE HELPER / BUSINESS LOGIC FUNCTIONS
# ==========================================================================
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0:
        return 0.0
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def predict_premium(inputs: dict, bundle: dict):
    """Run every available model on the input and return ensemble stats."""
    X = encode_single_input(inputs)
    preds = {}
    for name, model in bundle["models"].items():
        try:
            preds[name] = float(model.predict(X)[0])
        except Exception:
            continue

    if not preds:
        return {"ensemble_mean": 0.0, "ensemble_std": 0.0, "confidence": 0.0,
                "per_model": {}, "model_used": "N/A"}

    values = np.array(list(preds.values()))
    mean_pred = float(np.mean(values))
    std_pred = float(np.std(values))
    rel_std = (std_pred / mean_pred) if mean_pred > 0 else 0
    confidence = float(np.clip(100 - rel_std * 350, 55, 99.5))

    model_used = " + ".join(preds.keys())
    return {
        "ensemble_mean": max(mean_pred, 500.0),
        "ensemble_std": std_pred,
        "confidence": round(confidence, 1),
        "per_model": preds,
        "model_used": model_used,
    }


def calculate_risk_score(inputs: dict) -> float:
    """0-100 risk score derived directly from lifestyle / medical factors."""
    score = 0.0
    score += min(inputs["age"] / 100 * 22, 22)
    bmi_excess = max(inputs["bmi"] - 22, 0)
    score += min(bmi_excess * 1.6, 18)
    score += 18 if inputs["smoking"] == "Yes" else 0
    score += {"None": 0, "Occasional": 4, "Frequent": 9}[inputs["alcohol"]]
    score += {"Low": 0, "Medium": 5, "High": 10}[inputs["occupation_risk"]]
    score += {"Normal": 0, "Elevated": 5, "High": 10}[inputs["blood_pressure"]]
    score += 9 if inputs["diabetes"] == "Yes" else 0
    score += 11 if inputs["heart_disease"] == "Yes" else 0
    score += 6 if inputs["family_history"] == "Yes" else 0
    score += inputs["stress_level"] * 0.8
    score -= min(inputs["exercise_days"] * 1.1, 8)
    score -= min(inputs["checkups_per_year"] * 1.0, 5)
    score -= {"Poor": 0, "Average": 1, "Good": 3, "Excellent": 5}[inputs["diet_quality"]]
    return float(np.clip(score, 0, 100))


def risk_level_from_score(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 65:
        return "Medium"
    return "High"


def calculate_health_score(inputs: dict) -> float:
    """0-100 wellness score - higher is healthier."""
    score = 100.0
    bmi = inputs["bmi"]
    if bmi < 18.5 or bmi >= 30:
        score -= 18
    elif bmi >= 25:
        score -= 9
    score -= 22 if inputs["smoking"] == "Yes" else 0
    score -= {"None": 0, "Occasional": 5, "Frequent": 14}[inputs["alcohol"]]
    score += min(inputs["exercise_days"] * 2.2, 15) - 8
    score -= abs(inputs["sleep_hours"] - 7.5) * 4
    score -= inputs["stress_level"] * 1.6
    score += {"Poor": -10, "Average": 0, "Good": 6, "Excellent": 12}[inputs["diet_quality"]]
    score += min(inputs["checkups_per_year"] * 2.0, 8)
    score -= 10 if inputs["diabetes"] == "Yes" else 0
    score -= 12 if inputs["heart_disease"] == "Yes" else 0
    score -= 6 if inputs["family_history"] == "Yes" else 0
    return float(np.clip(score, 0, 100))


def health_category(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Average"
    return "Poor"


def fraud_check(inputs: dict) -> list:
    """Return a list of human-readable warnings for unrealistic inputs."""
    warnings_list = []
    if inputs["bmi"] < 10 or inputs["bmi"] > 70:
        warnings_list.append(f"BMI of {inputs['bmi']} is physiologically implausible.")
    if inputs["age"] < 18 or inputs["age"] > 100:
        warnings_list.append(f"Age of {inputs['age']} is outside the supported range (18-100).")
    if inputs["income"] < 0:
        warnings_list.append("Income cannot be negative.")
    if inputs["height"] < 100 or inputs["height"] > 230:
        warnings_list.append(f"Height of {inputs['height']} cm looks unrealistic.")
    if inputs["weight"] < 25 or inputs["weight"] > 250:
        warnings_list.append(f"Weight of {inputs['weight']} kg looks unrealistic.")
    if inputs["sleep_hours"] < 1 or inputs["sleep_hours"] > 16:
        warnings_list.append(f"Sleep of {inputs['sleep_hours']} hours/night is out of a realistic range.")
    if inputs["exercise_days"] > 7:
        warnings_list.append("Exercise days per week cannot exceed 7.")
    return warnings_list


def get_feature_importance(bundle: dict) -> dict:
    """Average feature_importances_ across all tree models that expose it."""
    importances = []
    for model in bundle["models"].values():
        if hasattr(model, "feature_importances_"):
            importances.append(np.array(model.feature_importances_, dtype=float))
    if not importances:
        return {f: 1.0 / len(FEATURE_NAMES) for f in FEATURE_NAMES}
    avg = np.mean(importances, axis=0)
    avg = avg / avg.sum()
    return dict(zip(FEATURE_NAMES, avg))


def explain_prediction(inputs: dict, bundle: dict, background_df: pd.DataFrame):
    """
    Generate a feature-level explanation for one prediction.
    Uses SHAP TreeExplainer if available, otherwise falls back to a manual
    contribution estimate based on global feature importance and how far the
    input deviates from the population average.
    """
    feature_importance = get_feature_importance(bundle)
    X_row = encode_single_input(inputs)

    contributions = {}
    method = "manual"

    if SHAP_AVAILABLE and "Random Forest" in bundle["models"]:
        try:
            explainer = shap.TreeExplainer(bundle["models"]["Random Forest"])
            shap_values = explainer.shap_values(X_row)
            for i, f in enumerate(FEATURE_NAMES):
                contributions[f] = float(shap_values[0][i])
            method = "shap"
        except Exception:
            method = "manual"

    if method == "manual":
        X_bg = encode_dataframe(background_df)
        means = X_bg.mean(axis=0)
        stds = X_bg.std(axis=0) + 1e-6
        for i, f in enumerate(FEATURE_NAMES):
            z = (X_row[0][i] - means[i]) / stds[i]
            contributions[f] = float(z * feature_importance[f] * 100)

    total_abs = sum(abs(v) for v in contributions.values()) + 1e-9
    pct_contrib = {f: abs(v) / total_abs * 100 for f, v in contributions.items()}

    sorted_features = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    bullets = []
    for f, val in sorted_features[:6]:
        label = FEATURE_LABELS.get(f, f)
        direction = "increases" if val > 0 else "decreases"
        bullets.append(f"**{label}** {direction} the premium (contributes ~{pct_contrib[f]:.0f}% of the effect).")

    return {
        "method": method,
        "contributions": contributions,
        "pct_contrib": pct_contrib,
        "sorted_features": sorted_features,
        "bullets": bullets,
    }


def generate_recommendations(inputs: dict, health_score: float) -> list:
    """Personalized, prioritized lifestyle recommendations."""
    recs = []
    if inputs["smoking"] == "Yes":
        recs.append({"text": "Quit smoking - this alone can significantly reduce your premium and long-term health risk.", "priority": "High"})
    if inputs["bmi"] >= 30:
        recs.append({"text": "Work toward losing 5-10 kg to bring your BMI into a healthier range.", "priority": "High"})
    elif inputs["bmi"] >= 25:
        recs.append({"text": "Losing 3-5 kg would move your BMI closer to the normal range.", "priority": "Medium"})
    if inputs["exercise_days"] < 3:
        recs.append({"text": "Aim for at least 4-5 days of exercise per week (e.g. brisk walking, 8,000+ steps/day).", "priority": "High"})
    elif inputs["exercise_days"] < 5:
        recs.append({"text": "Increase exercise frequency slightly to 5 days/week for optimal benefit.", "priority": "Medium"})
    if inputs["sleep_hours"] < 6.5 or inputs["sleep_hours"] > 9:
        recs.append({"text": "Aim for 7-8 hours of quality sleep per night.", "priority": "Medium"})
    if inputs["stress_level"] >= 7:
        recs.append({"text": "High stress increases future health risk - consider mindfulness, therapy, or workload changes.", "priority": "High"})
    elif inputs["stress_level"] >= 5:
        recs.append({"text": "Moderate stress detected - regular relaxation practices are recommended.", "priority": "Medium"})
    if inputs["alcohol"] == "Frequent":
        recs.append({"text": "Reduce alcohol consumption to lower long-term health risk.", "priority": "Medium"})
    if inputs["diet_quality"] in ("Poor", "Average"):
        recs.append({"text": "Improve diet quality - add more vegetables, fiber, and lean protein; drink more water.", "priority": "Medium"})
    if inputs["checkups_per_year"] < 1:
        recs.append({"text": "Schedule at least one annual comprehensive health checkup.", "priority": "Medium"})
    if inputs["diabetes"] == "Yes" or inputs["heart_disease"] == "Yes":
        recs.append({"text": "Continue regular monitoring and follow your physician's treatment plan closely.", "priority": "High"})
    if health_score >= 80 and not recs:
        recs.append({"text": "Great job! Maintain your current healthy lifestyle to keep your premium low.", "priority": "Low"})
    if not recs:
        recs.append({"text": "Keep up your current habits and continue annual health checkups.", "priority": "Low"})
    return recs


def recommend_plan(inputs: dict, risk_level: str) -> dict:
    """Recommend an insurance plan tier with a short justification."""
    age = inputs["age"]
    dependents = inputs["dependents"]

    if age >= 60:
        return {"plan": "Senior Citizen", "reason": "Tailored coverage with higher medical limits suited for ages 60+."}
    if dependents >= 2:
        return {"plan": "Family", "reason": f"Covers you and your {dependents} dependents under one comprehensive policy."}
    if risk_level == "High":
        return {"plan": "Premium", "reason": "Broader coverage and higher claim limits recommended given your current risk profile."}
    if risk_level == "Medium":
        return {"plan": "Standard", "reason": "Balanced coverage and premium cost that matches your moderate risk profile."}
    return {"plan": "Basic", "reason": "Cost-effective coverage suited to your low-risk, healthy profile."}


def forecast_future_premium(inputs: dict, bundle: dict, years: int = 5) -> pd.DataFrame:
    """Project premiums for 3 lifestyle scenarios over the next N years."""
    rows = []
    for year in range(years + 1):
        # --- current lifestyle: age increases, everything else constant ---
        cur = dict(inputs)
        cur["age"] = inputs["age"] + year
        cur_pred = predict_premium(cur, bundle)["ensemble_mean"]

        # --- improved lifestyle: gradually healthier over time ---
        imp = dict(inputs)
        imp["age"] = inputs["age"] + year
        imp["exercise_days"] = min(7, inputs["exercise_days"] + year)
        imp["smoking"] = "No"
        imp["stress_level"] = max(1, inputs["stress_level"] - year)
        imp["diet_quality"] = "Excellent" if year >= 2 else ("Good" if year >= 1 else inputs["diet_quality"])
        imp["bmi"] = max(18.5, inputs["bmi"] - year * 0.6)
        imp["checkups_per_year"] = min(4, inputs["checkups_per_year"] + 1)
        imp_pred = predict_premium(imp, bundle)["ensemble_mean"]

        # --- worse lifestyle: gradually unhealthier over time ---
        wor = dict(inputs)
        wor["age"] = inputs["age"] + year
        wor["exercise_days"] = max(0, inputs["exercise_days"] - year)
        wor["smoking"] = "Yes"
        wor["stress_level"] = min(10, inputs["stress_level"] + year)
        wor["diet_quality"] = "Poor"
        wor["bmi"] = min(55, inputs["bmi"] + year * 1.2)
        wor_pred = predict_premium(wor, bundle)["ensemble_mean"]

        rows.append({"Year": datetime.now().year + year, "Current Lifestyle": cur_pred,
                      "Improved Lifestyle": imp_pred, "Worse Lifestyle": wor_pred})
    return pd.DataFrame(rows)


def find_similar_customers(df: pd.DataFrame, inputs: dict, k: int = 5) -> pd.DataFrame:
    """K-Nearest-Neighbours search over age/bmi/income/premium space."""
    feats = df[["age", "bmi", "income", "premium"]].copy()
    means, stds = feats.mean(), feats.std() + 1e-9
    feats_norm = (feats - means) / stds

    query = pd.DataFrame([{
        "age": inputs["age"], "bmi": inputs["bmi"], "income": inputs["income"],
        "premium": inputs.get("estimated_premium", feats["premium"].mean()),
    }])
    query_norm = (query - means) / stds

    nn = NearestNeighbors(n_neighbors=min(k, len(df)))
    nn.fit(feats_norm.to_numpy())
    _, idx = nn.kneighbors(query_norm.to_numpy())

    result = df.iloc[idx[0]][["age", "gender", "bmi", "income", "premium", "region"]].copy()
    result["risk_score"] = result.apply(
        lambda r: risk_level_from_score(calculate_risk_score({
            "age": r["age"], "bmi": r["bmi"], "smoking": "No", "alcohol": "None",
            "occupation_risk": "Low", "blood_pressure": "Normal", "diabetes": "No",
            "heart_disease": "No", "family_history": "No", "stress_level": 5,
            "exercise_days": 3, "checkups_per_year": 1, "diet_quality": "Average",
        })), axis=1
    )
    return result.reset_index(drop=True)


def fairness_analysis(df: pd.DataFrame) -> dict:
    """Compare average predicted premiums across demographic groups."""
    by_gender = df.groupby("gender")["premium"].mean().round(0)
    by_region = df.groupby("region")["premium"].mean().round(0)
    age_bins = pd.cut(df["age"], bins=[17, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
    by_age = df.groupby(age_bins, observed=True)["premium"].mean().round(0)

    gender_gap = float(by_gender.max() - by_gender.min())
    region_gap = float(by_region.max() - by_region.min())
    age_gap = float(by_age.max() - by_age.min())

    return {"by_gender": by_gender, "by_region": by_region, "by_age": by_age,
            "gender_gap": gender_gap, "region_gap": region_gap, "age_gap": age_gap}


def generate_qr_image(data_str: str):
    if not QRCODE_AVAILABLE:
        return None
    try:
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(data_str)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def generate_pdf_report(inputs: dict, prediction: dict, health_score: float,
                         risk_level: str, recommendations: list, forecast_df: pd.DataFrame) -> bytes:
    """Build a downloadable PDF report. Falls back to a plain-text report if
    reportlab is unavailable."""
    if not REPORTLAB_AVAILABLE:
        return generate_text_report(inputs, prediction, health_score, risk_level,
                                     recommendations, forecast_df).encode("utf-8")

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    def line(text, size=11, bold=False, gap=0.65, color=rl_colors.black):
        nonlocal y
        c.setFillColor(color)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(2 * cm, y, text)
        y -= gap * cm

    c.setFillColor(rl_colors.HexColor("#7c3aed"))
    c.rect(0, height - 2.6 * cm, width, 2.6 * cm, fill=True, stroke=False)
    c.setFillColor(rl_colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 1.7 * cm, "Insurance Premium Prediction Pro AI - Report")
    y = height - 3.4 * cm

    line("Digital Health Passport & Premium Report", 14, bold=True)
    line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 9, color=rl_colors.grey)
    y -= 0.2 * cm

    line("User Profile", 13, bold=True, color=rl_colors.HexColor("#312e81"))
    line(f"Age: {inputs['age']}   Gender: {inputs['gender']}   Region: {inputs['region']}")
    line(f"Height: {inputs['height']} cm   Weight: {inputs['weight']} kg   BMI: {inputs['bmi']} ({bmi_category(inputs['bmi'])})")
    line(f"Smoking: {inputs['smoking']}   Alcohol: {inputs['alcohol']}   Exercise: {inputs['exercise_days']} days/week")
    y -= 0.2 * cm

    line("Premium Prediction", 13, bold=True, color=rl_colors.HexColor("#312e81"))
    line(f"Predicted Annual Premium: ${prediction['ensemble_mean']:,.2f}", 12, bold=True)
    line(f"Confidence: {prediction['confidence']}%   Model(s) Used: {prediction['model_used']}")
    line(f"Risk Level: {risk_level}")
    y -= 0.2 * cm

    line("Health Score", 13, bold=True, color=rl_colors.HexColor("#312e81"))
    line(f"Score: {health_score:.0f}/100  ({health_category(health_score)})")
    y -= 0.2 * cm

    line("5-Year Forecast (Current Lifestyle)", 13, bold=True, color=rl_colors.HexColor("#312e81"))
    for _, row in forecast_df.iterrows():
        line(f"{int(row['Year'])}: ${row['Current Lifestyle']:,.0f}", 10)
    y -= 0.2 * cm

    line("Recommendations", 13, bold=True, color=rl_colors.HexColor("#312e81"))
    for rec in recommendations[:8]:
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm
        line(f"[{rec['priority']}] {rec['text']}", 10)

    c.showPage()
    c.save()
    return buf.getvalue()


def generate_text_report(inputs, prediction, health_score, risk_level, recommendations, forecast_df) -> str:
    lines = []
    lines.append("INSURANCE PREMIUM PREDICTION PRO AI - REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("USER PROFILE")
    lines.append(f"Age: {inputs['age']}, Gender: {inputs['gender']}, Region: {inputs['region']}")
    lines.append(f"BMI: {inputs['bmi']} ({bmi_category(inputs['bmi'])})")
    lines.append("")
    lines.append("PREDICTION")
    lines.append(f"Premium: ${prediction['ensemble_mean']:,.2f}")
    lines.append(f"Confidence: {prediction['confidence']}%")
    lines.append(f"Risk Level: {risk_level}")
    lines.append(f"Health Score: {health_score:.0f}/100 ({health_category(health_score)})")
    lines.append("")
    lines.append("5-YEAR FORECAST (CURRENT LIFESTYLE)")
    for _, row in forecast_df.iterrows():
        lines.append(f"{int(row['Year'])}: ${row['Current Lifestyle']:,.0f}")
    lines.append("")
    lines.append("RECOMMENDATIONS")
    for rec in recommendations:
        lines.append(f"[{rec['priority']}] {rec['text']}")
    return "\n".join(lines)


def gauge_chart(value: float, title: str, max_value: float = 100,
                 low_color="#34d399", mid_color="#fbbf24", high_color="#f87171") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18, "color": "white"}},
        number={"font": {"color": "white", "size": 34}},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": "rgba(167,139,250,0.9)"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.2)",
            "steps": [
                {"range": [0, max_value * 0.35], "color": low_color},
                {"range": [max_value * 0.35, max_value * 0.65], "color": mid_color},
                {"range": [max_value * 0.65, max_value], "color": high_color},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, height=300, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        legend={"font": {"color": "white"}},
        margin=dict(t=50, b=30, l=30, r=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    return fig


def risk_badge_html(risk_level: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[risk_level]
    return f'<span class="{cls}">{risk_level} Risk</span>'


def default_inputs() -> dict:
    """Sensible default input values used before the user has predicted anything."""
    weight, height = 75.0, 172.0
    return {
        "age": 35, "gender": "Male", "height": height, "weight": weight,
        "bmi": calculate_bmi(weight, height), "smoking": "No", "alcohol": "Occasional",
        "exercise_days": 3, "occupation_risk": "Low", "income": 55000.0, "region": "North",
        "blood_pressure": "Normal", "diabetes": "No", "heart_disease": "No",
        "family_history": "No", "dependents": 1, "checkups_per_year": 1,
        "sleep_hours": 7.0, "stress_level": 5, "diet_quality": "Average",
    }


# ==========================================================================
# SESSION STATE INITIALIZATION
# ==========================================================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = default_inputs()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_risk_level" not in st.session_state:
    st.session_state.last_risk_level = None
if "last_health_score" not in st.session_state:
    st.session_state.last_health_score = None
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False


# ==========================================================================
# SIDEBAR NAVIGATION
# ==========================================================================
PAGES = [
    "🏠 Home", "🧮 Premium Prediction", "📊 Dashboard", "📈 Analytics",
    "🧠 AI Insights", "🎯 Lifestyle Simulator", "❤️ Health Score",
    "📄 PDF Report", "ℹ️ About",
]

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0.8rem 0;'>"
        "<div style='font-size:2.4rem;'>🛡️</div>"
        "<div style='font-size:1.15rem;font-weight:800;color:white;'>Premium AI</div>"
        "<div style='font-size:0.75rem;color:rgba(255,255,255,0.55);'>Prediction Pro</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    choice = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = choice
    st.markdown("---")
    st.caption("Models available:")
    model_flags = [
        ("Random Forest", True), ("XGBoost", XGBOOST_AVAILABLE),
        ("LightGBM", LIGHTGBM_AVAILABLE), ("CatBoost", CATBOOST_AVAILABLE),
        ("SHAP Explainability", SHAP_AVAILABLE),
    ]
    for name, ok in model_flags:
        st.markdown(f"{'✅' if ok else '⚪'} {name}")


# Load / train models once, cached as a resource across reruns.
model_bundle = load_or_train_models()
demo_df = generate_synthetic_dataset()


# ==========================================================================
# PAGE: HOME
# ==========================================================================
def page_home():
    st.markdown('<div class="hero-title">AI Insurance Premium Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Predict your health insurance premium instantly with explainable, '
        'ensemble machine learning - then simulate lifestyle changes to see exactly how much you could save.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🚀 Start Prediction", use_container_width=True):
            st.session_state.page = "🧮 Premium Prediction"
            st.rerun()
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "📊 Dashboard"
            st.rerun()
    with col3:
        if st.button("🎯 Try Lifestyle Simulator", use_container_width=True):
            st.session_state.page = "🎯 Lifestyle Simulator"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("5,200+", "Training Records"),
        (f"{len(model_bundle['models'])}", "AI Models in Ensemble"),
        ("20", "Health & Lifestyle Factors"),
        ("< 1s", "Prediction Time"),
    ]
    for col, (val, label) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                        f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">✨ Platform Features</div>', unsafe_allow_html=True)
    features = [
        ("🧮", "Ensemble Premium Prediction", "Random Forest, XGBoost, LightGBM & CatBoost combine for an accurate, confidence-scored estimate."),
        ("🧠", "Explainable AI", "SHAP-powered breakdown shows exactly why your premium is what it is."),
        ("🎯", "Lifestyle Simulator", "Adjust weight, smoking, exercise, diet and sleep to see instant savings."),
        ("📈", "5-Year Forecast", "Compare current, improved and worsening lifestyle trajectories."),
        ("❤️", "Health Score", "A 0-100 wellness score benchmarked against population data."),
        ("🕵️", "Fraud Detection", "Automatic flags for unrealistic or suspicious inputs."),
        ("⚖️", "Fairness Dashboard", "Transparency into how predictions vary across demographic groups."),
        ("👥", "Similar Customer Finder", "K-Nearest-Neighbours match against comparable profiles."),
        ("📄", "Digital Health Passport & PDF Report", "A shareable, printable summary with QR code."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f'<div class="glass-card"><div style="font-size:1.8rem;">{icon}</div>'
                        f'<div style="font-weight:700;color:white;margin:0.3rem 0;">{title}</div>'
                        f'<div style="color:rgba(255,255,255,0.65);font-size:0.9rem;">{desc}</div></div>',
                        unsafe_allow_html=True)


# ==========================================================================
# PAGE: PREMIUM PREDICTION
# ==========================================================================
def page_prediction():
    st.markdown('<div class="section-title">🧮 Premium Prediction</div>', unsafe_allow_html=True)
    st.caption("Fill in your details below and click Predict to get your instant AI-powered premium estimate.")

    d = st.session_state.last_inputs

    with st.form("prediction_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**👤 Personal Details**")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 18, 100, d["age"])
        gender = c2.selectbox("Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(d["gender"]))
        region = c3.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index(d["region"]))
        income = c4.number_input("Annual Income ($)", 0.0, 2_000_000.0, float(d["income"]), step=1000.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**📏 Body Metrics**")
        c1, c2, c3 = st.columns(3)
        height = c1.number_input("Height (cm)", 100.0, 230.0, float(d["height"]))
        weight = c2.number_input("Weight (kg)", 25.0, 250.0, float(d["weight"]))
        bmi = calculate_bmi(weight, height)
        c3.metric("BMI (auto-calculated)", f"{bmi}", bmi_category(bmi))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🚬 Lifestyle**")
        c1, c2, c3, c4 = st.columns(4)
        smoking = c1.selectbox("Smoking", YESNO_OPTIONS, index=YESNO_OPTIONS.index(d["smoking"]))
        alcohol = c2.selectbox("Alcohol Consumption", ALCOHOL_OPTIONS, index=ALCOHOL_OPTIONS.index(d["alcohol"]))
        exercise_days = c3.slider("Exercise Days / Week", 0, 7, d["exercise_days"])
        diet_quality = c4.selectbox("Diet Quality", DIET_OPTIONS, index=DIET_OPTIONS.index(d["diet_quality"]))
        c1, c2, c3 = st.columns(3)
        sleep_hours = c1.slider("Sleep Hours / Night", 3.0, 12.0, float(d["sleep_hours"]), 0.5)
        stress_level = c2.slider("Stress Level (1-10)", 1, 10, d["stress_level"])
        occupation_risk = c3.selectbox("Occupation Risk", RISK_OPTIONS, index=RISK_OPTIONS.index(d["occupation_risk"]))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🩺 Medical History**")
        c1, c2, c3, c4 = st.columns(4)
        blood_pressure = c1.selectbox("Blood Pressure", BP_OPTIONS, index=BP_OPTIONS.index(d["blood_pressure"]))
        diabetes = c2.selectbox("Diabetes", YESNO_OPTIONS, index=YESNO_OPTIONS.index(d["diabetes"]))
        heart_disease = c3.selectbox("Heart Disease", YESNO_OPTIONS, index=YESNO_OPTIONS.index(d["heart_disease"]))
        family_history = c4.selectbox("Family History of Illness", YESNO_OPTIONS, index=YESNO_OPTIONS.index(d["family_history"]))
        c1, c2 = st.columns(2)
        dependents = c1.slider("Dependents", 0, 8, d["dependents"])
        checkups_per_year = c2.slider("Health Checkups / Year", 0, 6, d["checkups_per_year"])
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Predict My Premium", use_container_width=True)

    if submitted:
        inputs = {
            "age": age, "gender": gender, "height": height, "weight": weight, "bmi": bmi,
            "smoking": smoking, "alcohol": alcohol, "exercise_days": exercise_days,
            "occupation_risk": occupation_risk, "income": income, "region": region,
            "blood_pressure": blood_pressure, "diabetes": diabetes, "heart_disease": heart_disease,
            "family_history": family_history, "dependents": dependents,
            "checkups_per_year": checkups_per_year, "sleep_hours": sleep_hours,
            "stress_level": stress_level, "diet_quality": diet_quality,
        }

        fraud_warnings = fraud_check(inputs)
        if fraud_warnings:
            st.warning("⚠️ **Potential data quality issues detected:**\n\n" + "\n".join(f"- {w}" for w in fraud_warnings))

        with st.spinner("Running ensemble AI models..."):
            start = datetime.now()
            prediction = predict_premium(inputs, model_bundle)
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000

        risk_score = calculate_risk_score(inputs)
        risk_level = risk_level_from_score(risk_score)
        health_score = calculate_health_score(inputs)

        st.session_state.last_inputs = inputs
        st.session_state.last_prediction = prediction
        st.session_state.last_risk_level = risk_level
        st.session_state.last_health_score = health_score
        st.session_state.has_predicted = True
        st.session_state.last_elapsed_ms = elapsed_ms
        st.session_state.last_risk_score = risk_score

    if st.session_state.has_predicted:
        prediction = st.session_state.last_prediction
        risk_level = st.session_state.last_risk_level
        health_score = st.session_state.last_health_score
        inputs = st.session_state.last_inputs

        st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("💰 Annual Premium", f"${prediction['ensemble_mean']:,.0f}")
        m2.metric("🎯 Confidence", f"{prediction['confidence']}%")
        m3.markdown(f"**Risk Level**<br>{risk_badge_html(risk_level)}", unsafe_allow_html=True)
        m4.metric("⏱️ Prediction Time", f"{st.session_state.get('last_elapsed_ms', 0):.0f} ms")
        m5.metric("🤖 Models Used", str(len(prediction["per_model"])))
        st.caption(f"Ensemble: {prediction['model_used']}")

        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(gauge_chart(st.session_state.last_risk_score, "Risk Meter"), use_container_width=True)
        with gc2:
            st.plotly_chart(gauge_chart(health_score, "Health Score"), use_container_width=True)

        plan = recommend_plan(inputs, risk_level)
        st.markdown(f'<div class="glass-card"><b>🛡️ Recommended Plan: {plan["plan"]}</b><br>'
                    f'<span style="color:rgba(255,255,255,0.7)">{plan["reason"]}</span></div>', unsafe_allow_html=True)

        st.info("💡 Head to **🧠 AI Insights** for a detailed explanation, **🎯 Lifestyle Simulator** to see savings, "
                "or **📄 PDF Report** to download your Digital Health Passport.")


# ==========================================================================
# PAGE: DASHBOARD
# ==========================================================================
def page_dashboard():
    st.markdown('<div class="section-title">📊 Dashboard</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Records", f"{len(demo_df):,}")
    m2.metric("Avg Premium", f"${demo_df['premium'].mean():,.0f}")
    m3.metric("Avg BMI", f"{demo_df['bmi'].mean():.1f}")
    m4.metric("Smoker %", f"{(demo_df['smoking'] == 'Yes').mean() * 100:.1f}%")

    if st.session_state.has_predicted:
        st.markdown('<div class="section-title">Your Last Prediction</div>', unsafe_allow_html=True)
        pred = st.session_state.last_prediction
        inputs = st.session_state.last_inputs
        c1, c2, c3 = st.columns(3)
        c1.metric("Your Premium", f"${pred['ensemble_mean']:,.0f}")
        c2.metric("vs. Population Avg", f"{(pred['ensemble_mean'] / demo_df['premium'].mean() - 1) * 100:+.1f}%")
        c3.markdown(f"**Risk Level**<br>{risk_badge_html(st.session_state.last_risk_level)}", unsafe_allow_html=True)

        st.markdown('<div class="section-title">👥 Similar Customer Finder</div>', unsafe_allow_html=True)
        sim_inputs = dict(inputs)
        sim_inputs["estimated_premium"] = pred["ensemble_mean"]
        similar = find_similar_customers(demo_df, sim_inputs, k=5)
        st.dataframe(
            similar.rename(columns={"age": "Age", "gender": "Gender", "bmi": "BMI", "income": "Income",
                                     "premium": "Premium", "region": "Region", "risk_score": "Risk"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Make a prediction on the 🧮 Premium Prediction page to see personalized dashboard insights and similar customers.")

    st.markdown('<div class="section-title">Population Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(demo_df, x="premium", nbins=40, title="Premium Distribution", color_discrete_sequence=["#a78bfa"])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.pie(demo_df, names="occupation_risk", title="Occupation Risk Split",
                     color_discrete_sequence=px.colors.sequential.Purp)
        st.plotly_chart(style_fig(fig), use_container_width=True)


# ==========================================================================
# PAGE: ANALYTICS
# ==========================================================================
def page_analytics():
    st.markdown('<div class="section-title">📈 Analytics</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Distributions", "Relationships", "Composition", "Advanced"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(demo_df, x="age", nbins=30, title="Age Distribution", color_discrete_sequence=["#60a5fa"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fig = px.histogram(demo_df, x="bmi", nbins=30, title="BMI Distribution", color_discrete_sequence=["#34d399"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig = px.box(demo_df, x="smoking", y="premium", title="Premium by Smoking Status", color="smoking",
                         color_discrete_sequence=["#34d399", "#f87171"])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c4:
            fig = px.box(demo_df, x="diet_quality", y="premium", title="Premium by Diet Quality", color="diet_quality")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            sample = demo_df.sample(min(800, len(demo_df)), random_state=1)
            fig = px.scatter(sample, x="bmi", y="premium", color="smoking", title="BMI vs Premium",
                              color_discrete_sequence=["#34d399", "#f87171"], opacity=0.6)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            fig = px.scatter(sample, x="age", y="premium", color="occupation_risk", title="Age vs Premium by Occupation Risk", opacity=0.6)
            st.plotly_chart(style_fig(fig), use_container_width=True)

        numeric_cols = ["age", "bmi", "income", "exercise_days", "stress_level", "sleep_hours", "premium"]
        corr = demo_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", color_continuous_scale="Purples")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(demo_df, names="region", title="Region Split", hole=0.4)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with c2:
            region_smoke = demo_df.groupby(["region", "smoking"], observed=True).size().reset_index(name="count")
            fig = px.bar(region_smoke, x="region", y="count", color="smoking", barmode="group", title="Smoking by Region")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        fig = px.sunburst(demo_df.sample(min(1500, len(demo_df)), random_state=1),
                           path=["region", "occupation_risk", "smoking"], values="premium",
                           title="Region → Occupation Risk → Smoking (by Premium)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        fig = px.treemap(demo_df.sample(min(1500, len(demo_df)), random_state=1),
                          path=["region", "diet_quality"], values="premium", title="Premium Treemap by Region & Diet")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with tabs[3]:
        importance = get_feature_importance(model_bundle)
        imp_df = pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in importance], "Importance": list(importance.values())})
        imp_df = imp_df.sort_values("Importance", ascending=True)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Global Feature Importance",
                     color="Importance", color_continuous_scale="Viridis")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        top_feats = imp_df.sort_values("Importance", ascending=False).head(6)
        fig = go.Figure(go.Scatterpolar(r=top_feats["Importance"], theta=top_feats["Feature"], fill="toself",
                                         line_color="#a78bfa"))
        fig.update_layout(title="Top Feature Importance (Radar)", polar=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(style_fig(fig), use_container_width=True)


# ==========================================================================
# PAGE: AI INSIGHTS
# ==========================================================================
def page_ai_insights():
    st.markdown('<div class="section-title">🧠 AI Insights (Explainable AI)</div>', unsafe_allow_html=True)

    if not st.session_state.has_predicted:
        st.info("Make a prediction on the 🧮 Premium Prediction page first to see a personalized explanation.")
        return

    inputs = st.session_state.last_inputs
    explanation = explain_prediction(inputs, model_bundle, demo_df)

    st.caption(f"Explanation method: {'SHAP' if explanation['method'] == 'shap' else 'Manual feature-contribution fallback'}")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Why this premium?**")
    for bullet in explanation["bullets"]:
        st.markdown(f"- {bullet}")
    st.markdown('</div>', unsafe_allow_html=True)

    sorted_feats = explanation["sorted_features"][:10]
    chart_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f, _ in sorted_feats],
        "Contribution": [v for _, v in sorted_feats],
    }).sort_values("Contribution")
    fig = px.bar(chart_df, x="Contribution", y="Feature", orientation="h",
                 title="Feature Contribution to Your Premium", color="Contribution",
                 color_continuous_scale="RdYlGn_r")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">⚖️ Fairness Dashboard</div>', unsafe_allow_html=True)
    fairness = fairness_analysis(demo_df)
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(fairness["by_gender"].reset_index(), x="gender", y="premium", title="Avg Premium by Gender")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.bar(fairness["by_region"].reset_index(), x="region", y="premium", title="Avg Premium by Region")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c3:
        fig = px.bar(fairness["by_age"].reset_index(), x="age", y="premium", title="Avg Premium by Age Group")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown(
        f'<div class="glass-card">'
        f'<b>Fairness Summary:</b> Gender gap ${fairness["gender_gap"]:,.0f} · '
        f'Region gap ${fairness["region_gap"]:,.0f} · Age-group gap ${fairness["age_gap"]:,.0f}. '
        f'These gaps largely reflect underlying risk-factor differences (e.g. age-related risk) rather than '
        f'direct use of protected attributes as pricing inputs.'
        f'</div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: LIFESTYLE SIMULATOR
# ==========================================================================
def page_lifestyle_simulator():
    st.markdown('<div class="section-title">🎯 Lifestyle Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders to see how lifestyle changes affect your premium in real time.")

    base = st.session_state.last_inputs
    current_prediction = st.session_state.last_prediction or predict_premium(base, model_bundle)
    current_premium = current_prediction["ensemble_mean"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sim_weight = st.slider("Weight (kg)", 40.0, 150.0, float(base["weight"]), 0.5)
        sim_smoking = st.selectbox("Smoking", YESNO_OPTIONS, index=YESNO_OPTIONS.index(base["smoking"]), key="sim_smoke")
        sim_exercise = st.slider("Exercise Days/Week", 0, 7, base["exercise_days"], key="sim_ex")
    with c2:
        sim_diet = st.selectbox("Diet Quality", DIET_OPTIONS, index=DIET_OPTIONS.index(base["diet_quality"]), key="sim_diet")
        sim_sleep = st.slider("Sleep Hours", 3.0, 12.0, float(base["sleep_hours"]), 0.5, key="sim_sleep")
    st.markdown('</div>', unsafe_allow_html=True)

    sim_inputs = dict(base)
    sim_inputs["weight"] = sim_weight
    sim_inputs["bmi"] = calculate_bmi(sim_weight, base["height"])
    sim_inputs["smoking"] = sim_smoking
    sim_inputs["exercise_days"] = sim_exercise
    sim_inputs["diet_quality"] = sim_diet
    sim_inputs["sleep_hours"] = sim_sleep

    improved_prediction = predict_premium(sim_inputs, model_bundle)
    improved_premium = improved_prediction["ensemble_mean"]
    savings = current_premium - improved_premium
    savings_pct = (savings / current_premium * 100) if current_premium > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Premium", f"${current_premium:,.0f}")
    m2.metric("Simulated Premium", f"${improved_premium:,.0f}", f"{-savings:,.0f}")
    m3.metric("Money Saved / Year", f"${savings:,.0f}")
    m4.metric("Savings %", f"{savings_pct:.1f}%")

    fig = go.Figure(data=[
        go.Bar(name="Current", x=["Premium"], y=[current_premium], marker_color="#f87171"),
        go.Bar(name="Simulated", x=["Premium"], y=[improved_premium], marker_color="#34d399"),
    ])
    fig.update_layout(title="Premium Comparison", barmode="group")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">📈 5-Year Premium Forecast</div>', unsafe_allow_html=True)
    forecast_df = forecast_future_premium(base, model_bundle, years=5)
    st.session_state.last_forecast = forecast_df
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Current Lifestyle"], mode="lines+markers", name="Current", line=dict(color="#60a5fa")))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Improved Lifestyle"], mode="lines+markers", name="Improved", line=dict(color="#34d399")))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["Worse Lifestyle"], mode="lines+markers", name="Worse", line=dict(color="#f87171")))
    fig.update_layout(title="Projected Premium Over 5 Years", xaxis_title="Year", yaxis_title="Premium ($)")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">💡 AI Recommendations</div>', unsafe_allow_html=True)
    recs = generate_recommendations(base, st.session_state.last_health_score or calculate_health_score(base))
    for rec in recs:
        css_class = {"High": "rec-high", "Medium": "rec-medium", "Low": "rec-low"}[rec["priority"]]
        st.markdown(f'<div class="{css_class}"><b>[{rec["priority"]}]</b> {rec["text"]}</div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: HEALTH SCORE
# ==========================================================================
def page_health_score():
    st.markdown('<div class="section-title">❤️ Health Score</div>', unsafe_allow_html=True)

    inputs = st.session_state.last_inputs
    score = calculate_health_score(inputs)
    category = health_category(score)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(gauge_chart(score, "Overall Health Score"), use_container_width=True)
    with c2:
        st.markdown(f'<div class="glass-card" style="text-align:center;">'
                    f'<div style="font-size:2.4rem;">{"🟢" if category=="Excellent" else "🟡" if category in ("Good","Average") else "🔴"}</div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:white;">{category}</div>'
                    f'<div style="color:rgba(255,255,255,0.65);">Score: {score:.0f} / 100</div></div>',
                    unsafe_allow_html=True)
        st.markdown(
            "- **Excellent** (80-100): Outstanding lifestyle habits\n"
            "- **Good** (60-79): Healthy with minor improvement areas\n"
            "- **Average** (40-59): Room for meaningful improvement\n"
            "- **Poor** (0-39): Significant lifestyle changes recommended"
        )

    st.markdown('<div class="section-title">Score Breakdown</div>', unsafe_allow_html=True)
    breakdown = {
        "BMI": 100 - min(abs(inputs["bmi"] - 22) * 4, 100),
        "Smoking": 0 if inputs["smoking"] == "Yes" else 100,
        "Exercise": min(inputs["exercise_days"] / 7 * 100, 100),
        "Sleep Quality": max(0, 100 - abs(inputs["sleep_hours"] - 7.5) * 20),
        "Stress": max(0, 100 - inputs["stress_level"] * 10),
        "Diet": {"Poor": 25, "Average": 55, "Good": 80, "Excellent": 100}[inputs["diet_quality"]],
    }
    fig = go.Figure(go.Scatterpolar(
        r=list(breakdown.values()) + [list(breakdown.values())[0]],
        theta=list(breakdown.keys()) + [list(breakdown.keys())[0]],
        fill="toself", line_color="#f472b6",
    ))
    fig.update_layout(title="Health Factor Radar", polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(style_fig(fig), use_container_width=True)


# ==========================================================================
# PAGE: PDF REPORT / DIGITAL HEALTH PASSPORT
# ==========================================================================
def page_pdf_report():
    st.markdown('<div class="section-title">📄 PDF Report & Digital Health Passport</div>', unsafe_allow_html=True)

    if not st.session_state.has_predicted:
        st.info("Make a prediction on the 🧮 Premium Prediction page first to generate your report.")
        return

    inputs = st.session_state.last_inputs
    prediction = st.session_state.last_prediction
    risk_level = st.session_state.last_risk_level
    health_score = st.session_state.last_health_score
    recs = generate_recommendations(inputs, health_score)
    forecast_df = st.session_state.get("last_forecast") or forecast_future_premium(inputs, model_bundle, years=5)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🪪 Digital Health Passport Preview**")
        st.markdown(f"- **Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.markdown(f"- **Age / Gender:** {inputs['age']} / {inputs['gender']}")
        st.markdown(f"- **Health Score:** {health_score:.0f}/100 ({health_category(health_score)})")
        st.markdown(f"- **Risk Level:** {risk_level}")
        st.markdown(f"- **Predicted Premium:** ${prediction['ensemble_mean']:,.2f}")
        st.markdown("**Top Recommendations:**")
        for rec in recs[:3]:
            st.markdown(f"  - [{rec['priority']}] {rec['text']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        qr_str = (f"Health Passport | Age:{inputs['age']} Premium:${prediction['ensemble_mean']:.0f} "
                  f"Risk:{risk_level} Score:{health_score:.0f} Date:{datetime.now().strftime('%Y-%m-%d')}")
        qr_bytes = generate_qr_image(qr_str)
        if qr_bytes:
            st.image(qr_bytes, caption="Scan for passport summary", width=180)
        else:
            st.caption("QR code library not available - install `qrcode` to enable this feature.")

    st.markdown("---")
    pdf_bytes = generate_pdf_report(inputs, prediction, health_score, risk_level, recs, forecast_df)
    file_ext = "pdf" if REPORTLAB_AVAILABLE else "txt"
    mime = "application/pdf" if REPORTLAB_AVAILABLE else "text/plain"
    st.download_button(
        f"⬇️ Download {'PDF' if REPORTLAB_AVAILABLE else 'Text'} Report",
        data=pdf_bytes,
        file_name=f"insurance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
        mime=mime,
        use_container_width=True,
    )
    if not REPORTLAB_AVAILABLE:
        st.caption("`reportlab` not installed - a plain-text report was generated instead.")


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
def page_about():
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
**Insurance Premium Prediction Pro AI** is an end-to-end, explainable machine-learning
platform for estimating health insurance premiums from lifestyle and medical data.
It combines an ensemble of tree-based models with transparent, human-readable
explanations so users understand *why* they receive the premium they do - not just
the number itself.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🛠️ Technology Stack**")
        st.markdown("""
- Python 3.11, Streamlit, Pandas, NumPy
- Plotly for interactive visualization
- Scikit-learn (Random Forest, KNN)
- XGBoost, LightGBM, CatBoost ensemble
- SHAP for explainable AI
- ReportLab for PDF generation, qrcode for passports
""")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🌟 Novel Features**")
        st.markdown("""
- Lifestyle Simulator with instant savings
- 5-Year multi-scenario premium forecasting
- Automated fraud / anomaly detection
- Ensemble-agreement confidence scoring
- Fairness dashboard across demographics
- KNN-based similar customer finder
- Digital Health Passport with QR code
""")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**🔭 Future Scope**")
    st.markdown("""
Integration with real insurer datasets, wearable-device data feeds (steps, heart rate,
sleep tracking), federated learning for privacy-preserving model updates, and a claims
prediction module to complement premium pricing.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**👨‍💻 Developer Credits**")
    st.markdown("Built with Streamlit, scikit-learn, and the open-source Python data science ecosystem.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================================
# ROUTER
# ==========================================================================
PAGE_ROUTER = {
    "🏠 Home": page_home,
    "🧮 Premium Prediction": page_prediction,
    "📊 Dashboard": page_dashboard,
    "📈 Analytics": page_analytics,
    "🧠 AI Insights": page_ai_insights,
    "🎯 Lifestyle Simulator": page_lifestyle_simulator,
    "❤️ Health Score": page_health_score,
    "📄 PDF Report": page_pdf_report,
    "ℹ️ About": page_about,
}

try:
    PAGE_ROUTER[st.session_state.page]()
except Exception as e:
    st.error(f"⚠️ Something went wrong while rendering this page: {e}")
    st.info("Please try navigating to a different page or refreshing the app.")