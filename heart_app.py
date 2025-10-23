import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HeartDiseaseTrain-Test.csv")

    # Clean and consistent column names
    df.rename(columns={
        "cholestoral": "cholesterol",
        "Max_heart_rate": "max_heart_rate"
    }, inplace=True)

    return df

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model():
    df = load_data()

    # Encode categorical columns
    categorical_cols = [
        "sex", "chest_pain_type", "fasting_blood_sugar",
        "rest_ecg", "exercise_induced_angina",
        "slope", "vessels_colored_by_flourosopy", "thalassemia"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    return model, scaler, accuracy, X_train_scaled, X_train.columns.tolist()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's health details to predict the likelihood of heart disease.")

# Show sample data
with st.expander("📊 View Sample Data"):
    st.dataframe(load_data().head())

# Load model
model, scaler, accuracy, X_train_scaled, feature_names = train_model()

# -----------------------------
# Input Form
# -----------------------------
st.subheader("🔹 Patient Information")
age = st.number_input("Age", min_value=1, max_value=120, value=52, key="age_input")
sex = st.selectbox("Sex", ["Male", "Female"], key="sex_input")
chest_pain_type = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], key="cp_input")
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130, key="bp_input")
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, key="chol_input")
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"], key="fbs_input")
rest_ecg = st.selectbox("Rest ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], key="ecg_input")
max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, key="hr_input")
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"], key="angina_input")
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, key="oldpeak_input")
slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"], key="slope_input")
vessels_colored_by_flourosopy = st.selectbox("Vessels Colored by Flourosopy", ["Zero", "One", "Two", "Three"], key="vessels_input")
thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"], key="thal_input")

# -----------------------------
# Encode Inputs
# -----------------------------
input_dict = {
    "age": age,
    "sex": 1 if sex=="Male" else 0,
    "chest_pain_type": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(chest_pain_type),
    "resting_blood_pressure": resting_blood_pressure,
    "cholesterol": cholesterol,
    "fasting_blood_sugar": 1 if fasting_blood_sugar=="Greater than 120 mg/ml" else 0,
    "rest_ecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(rest_ecg),
    "max_heart_rate": max_heart_rate,
    "exercise_induced_angina": 1 if exercise_induced_angina=="Yes" else 0,
    "oldpeak": oldpeak,
    "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
    "vessels_colored_by_flourosopy": ["Zero", "One", "Two", "Three"].index(vessels_colored_by_flourosopy),
    "thalassemia": ["Normal", "Fixed Defect", "Reversable Defect"].index(thalassemia)
}

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction and SHAP
# -----------------------------
if st.button("🔍 Predict Heart Disease", key="predict_button"):
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0,1]

    # Feedback & recommendations
    if prediction == 1:
        st.error(f"⚠️ High likelihood of heart disease ({prediction_proba*100:.2f}%)")
        feedback = "You have risk factors that increase the likelihood of heart disease."
        recommendations = [
            "Reduce cholesterol with a heart-healthy diet.",
            "Monitor and control blood pressure regularly.",
            "Consult a doctor before intense exercise.",
            "Manage stress and maintain a healthy lifestyle."
        ]
    else:
        st.success(f"✅ Low likelihood of heart disease ({prediction_proba*100:.2f}%)")
        feedback = "Good news! Your risk of heart disease is currently low."
        recommendations = [
            "Maintain a healthy lifestyle with regular exercise.",
            "Follow a balanced diet.",
            "Continue regular health check-ups.",
            "Monitor key indicators like blood pressure and cholesterol."
        ]

    st.subheader("Feedback:")
    st.write(feedback)

    st.subheader("Recommendations:")
    for rec in recommendations:
        st.write(f"• {rec}")

    st.caption(f"Model Accuracy: {accuracy*100:.2f}%")
 # -----------------------------
# SHAP Explainability (Text Only)
# -----------------------------
st.subheader("🔍 Key Factors Influencing This Prediction")

# Use SHAP to understand feature influence
explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(input_scaled)

# Pair feature names with their impact
feature_contributions = dict(zip(feature_names, shap_values[0]))

# Sort by strongest effect
sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

# Clear text explanation
st.write("The model based its prediction mainly on these factors:")
for feat, val in sorted_features[:5]:
    if val > 0:
        st.write(f"• **{feat.replace('_', ' ').title()}** — increased the risk of heart disease.")
    else:
        st.write(f"• **{feat.replace('_', ' ').title()}** — decreased the risk of heart disease.")

st.caption("Positive values indicate higher risk factors, while negative values indicate protective factors.")
