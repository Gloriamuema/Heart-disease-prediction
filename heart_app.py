# heart_app.py
#import the libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
import plotly.graph_objects as go

# -----------------------------
# Page config & basic styling
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fafafa;
    }
    .big-title {
        font-size:32px;
        font-weight:700;
        color:#b30000;
        margin-bottom: 0;
    }
    .subtitle {
        color: #333333;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .card {
        background: white;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">❤️ Heart Disease Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details and get an easy-to-understand risk assessment and explanation.</div>', unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(path="HeartDiseaseTrain-Test.csv"):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read dataset file: {e}")
        return None

    # unify columns
    df.rename(columns={"cholestoral": "cholesterol", "Max_heart_rate": "max_heart_rate"}, inplace=True)
    return df

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(df):
    # If df is None, return None
    if df is None:
        return None

    # Copy to avoid modifying original cached df
    data = df.copy()

    # Fill or drop missing values (simple strategy)
    data = data.dropna().reset_index(drop=True)

    # Categorical encoding (using the same mapping approach)
    categorical_cols = [
        "sex", "chest_pain_type", "fasting_blood_sugar",
        "rest_ecg", "exercise_induced_angina",
        "slope", "vessels_colored_by_flourosopy", "thalassemia"
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes

    if 'target' not in data.columns:
        st.error("Dataset must contain a 'target' column.")
        return None

    X = data.drop('target', axis=1)
    y = data['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    # Return objects needed for predictions and SHAP
    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "X_train_scaled": X_train_scaled,
        "feature_names": X.columns.tolist(),
        "X_train_df": X  # needed to get feature order/values for SHAP
    }

# -----------------------------
# Load and train
# -----------------------------
df = load_data()
model_bundle = train_model(df)

if model_bundle is None:
    st.stop()

model = model_bundle["model"]
scaler = model_bundle["scaler"]
accuracy = model_bundle["accuracy"]
X_train_scaled = model_bundle["X_train_scaled"]
feature_names = model_bundle["feature_names"]
X_train_df = model_bundle["X_train_df"]

# -----------------------------
# Sample data viewer
# -----------------------------
with st.expander(" View Sample Data"):
    if df is not None:
        st.dataframe(df.head(8))
    else:
        st.write("No dataset loaded.")

# -----------------------------
# Input Form (Two columns)
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(" Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=52, key="age_input")
    sex = st.selectbox("Sex", ["Male", "Female"], key="sex_input")
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], key="cp_input")
    resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130, key="bp_input")
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, key="chol_input")

with col2:
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"], key="fbs_input")
    rest_ecg = st.selectbox("Rest ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], key="ecg_input")
    max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, key="hr_input")
    exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"], key="angina_input")
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, key="oldpeak_input")

slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"], key="slope_input")
vessels_colored_by_flourosopy = st.selectbox("Vessels Colored by Flourosopy", ["Zero", "One", "Two", "Three"], key="vessels_input")
thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"], key="thal_input")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Encode Inputs
# -----------------------------
input_dict = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "chest_pain_type": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(chest_pain_type),
    "resting_blood_pressure": resting_blood_pressure,
    "cholesterol": cholesterol,
    "fasting_blood_sugar": 1 if fasting_blood_sugar == "Greater than 120 mg/ml" else 0,
    "rest_ecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(rest_ecg),
    "max_heart_rate": max_heart_rate,
    "exercise_induced_angina": 1 if exercise_induced_angina == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
    "vessels_colored_by_flourosopy": ["Zero", "One", "Two", "Three"].index(vessels_colored_by_flourosopy),
    "thalassemia": ["Normal", "Fixed Defect", "Reversable Defect"].index(thalassemia)
}

input_df = pd.DataFrame([input_dict])[feature_names]  # enforce column order

# -----------------------------
# Predictions & UI Display
# -----------------------------
if st.button(" Predict Heart Disease", key="predict_button"):

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    # probability of class '1'
    proba = model.predict_proba(input_scaled)[0, 1]  
    pred = int(model.predict(input_scaled)[0])
    risk_pct = proba * 100

    # Display a clean result card
    if pred == 1:
        st.markdown("<h3 style='color:#b30000;'>⚠️ High likelihood of heart disease</h3>", unsafe_allow_html=True)
        st.markdown(f"**Risk probability:** {risk_pct:.1f}%")
    else:
        st.markdown("<h3 style='color:#2a9d8f;'>✅ Low likelihood of heart disease</h3>", unsafe_allow_html=True)
        st.markdown(f"**Risk probability:** {risk_pct:.1f}%")

    st.caption(f"Model accuracy (on held-out test set): {accuracy*100:.2f}%")

    # -----------------------------
    # Gauge (Plotly)
    # -----------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': "%", "valueformat": ".1f"},
        title={'text': "<b>Heart Disease Risk</b>"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#b30000" if pred == 1 else "#2a9d8f"},
            'steps': [
                {'range': [0, 40], 'color': "#d4f5e6"},
                {'range': [40, 70], 'color': "#fff2cc"},
                {'range': [70, 100], 'color': "#ffd6d6"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': risk_pct
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Recommendations (human-friendly)
    # -----------------------------
    st.subheader(" Personalized Recommendations")
    if pred == 1:
        recommendations = [
            "Reduce cholesterol with a heart-healthy diet (more vegetables, lean proteins, less saturated fat).",
            "Monitor and control blood pressure regularly talk to a clinician about medications if elevated.",
            "Avoid strenuous exercise until cleared by a doctor; start low-impact activity after check-up.",
            "Manage stress, stop smoking, and limit alcohol intake."
        ]
        st.markdown("<div style='color:#b30000;font-weight:600;'>Immediate steps:</div>", unsafe_allow_html=True)
    else:
        recommendations = [
            "Maintain a healthy lifestyle with regular moderate exercise (150 min/week recommended).",
            "Keep a balanced diet focus on whole foods and reduce processed foods.",
            "Continue regular health check-ups and monitor blood pressure and cholesterol.",
            "Keep body weight and stress levels in check."
        ]
        st.markdown("<div style='color:#2a9d8f;font-weight:600;'>Good job — keep it up:</div>", unsafe_allow_html=True)

    for rec in recommendations:
        st.write(f"• {rec}")

    # -----------------------------
    # SHAP Explainability - Text Only (Top factors)
    # -----------------------------
    st.subheader(" Key Factors Influencing This Prediction")

    # Create SHAP explainer (LinearExplainer is appropriate for linear models)
    try:
        explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(input_scaled)  # returns array shaped (n_samples, n_features)
        contributions = dict(zip(feature_names, shap_values[0]))
    except Exception as e:
        st.write("Could not compute SHAP values. Showing fallback feature importance instead.")
        # Fallback: use coef * value (approximate linear influence)
        coefs = model.coef_[0]
        contributions = {f: float(coefs[idx] * input_df.iloc[0, idx]) for idx, f in enumerate(feature_names)}

    # Build a table for the top features
    contrib_df = pd.DataFrame({
        "Feature": list(contributions.keys()),
        "SHAP_Impact": list(contributions.values()),
        "Value": [input_df[c].iloc[0] for c in feature_names]
    })

    # Choose top 8 by absolute impact
    contrib_df["abs_impact"] = contrib_df["SHAP_Impact"].abs()
    top_contrib = contrib_df.sort_values("abs_impact", ascending=False).head(8).copy()

    # Make friendly labels and effects
    def friendly_label(name):
        return name.replace("_", " ").title()

    top_contrib["Effect"] = top_contrib["SHAP_Impact"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
    top_contrib["Impact Strength"] = top_contrib["abs_impact"].round(3)
    top_contrib_display = top_contrib[["Feature", "Value", "Effect", "Impact Strength"]].copy()
    top_contrib_display["Feature"] = top_contrib_display["Feature"].apply(friendly_label)
    top_contrib_display = top_contrib_display.reset_index(drop=True)

    st.table(top_contrib_display)

# -----------------------------
# Footer / Extras
# -----------------------------
st.markdown("---")
