import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Data Function
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HeartDiseaseTrain-Test.csv")
    return df

# -----------------------------
# Train Model Function
# -----------------------------
@st.cache_resource
def train_model():
    df = load_data()

    # Identify categorical columns (based on your input form)
    categorical_cols = [
        "sex", "chest_pain_type", "fasting_blood_sugar",
        "rest_ecg", "exercise_induced_angina",
        "slope", "vessels_colored_by_flourosopy", "thalassemia"
    ]

    # Encode categorical columns (convert text → numbers)
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, accuracy


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's health details to predict the likelihood of heart disease.")

# Display dataset sample (optional)
with st.expander("📊 View Sample Data"):
    st.dataframe(load_data().head())

# Load model
model, scaler, accuracy = train_model()

# -----------------------------
# Input Form
# -----------------------------
st.subheader("🔹 Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=52)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130)
cholestoral = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"])
rest_ecg = st.selectbox("Rest ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
Max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
vessels_colored_by_flourosopy = st.selectbox("Vessels Colored by Flourosopy", ["Zero", "One", "Two", "Three"])
thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

# -----------------------------
# Encoding
# -----------------------------
sex = 1 if sex == "Male" else 0
chest_pain_type = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(chest_pain_type)
fasting_blood_sugar = 1 if fasting_blood_sugar == "Greater than 120 mg/ml" else 0
rest_ecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(rest_ecg)
exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
vessels_colored_by_flourosopy = ["Zero", "One", "Two", "Three"].index(vessels_colored_by_flourosopy)
thalassemia = ["Normal", "Fixed Defect", "Reversable Defect"].index(thalassemia)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Heart Disease"):
    input_data = np.array([[age, sex, chest_pain_type, resting_blood_pressure, cholestoral,
                            fasting_blood_sugar, rest_ecg, Max_heart_rate,
                            exercise_induced_angina, oldpeak, slope,
                            vessels_colored_by_flourosopy, thalassemia]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts a **high likelihood** of heart disease.")
    else:
        st.success("✅ The model predicts a **low likelihood** of heart disease.")

st.caption(f"Model Accuracy: {accuracy*100:.2f}%")
