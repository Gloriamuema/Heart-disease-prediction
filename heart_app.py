# heart_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('HeartDiseaseTrain-Test.csv')
    return df

print(df.head())
# --- Encode categorical columns manually ---
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['chest_pain_type'] = df['chest_pain_type'].map({
    'Typical angina': 0,
    'Atypical angina': 1,
    'Non-anginal pain': 2,
    'Asymptomatic': 3
})

df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
    'Lower than 120 mg/ml': 0,
    'Greater than 120 mg/ml': 1
})

df['rest_ecg'] = df['rest_ecg'].map({
    'Normal': 0,
    'ST-T wave abnormality': 1,
    'Left ventricular hypertrophy': 2
})

df['exercise_induced_angina'] = df['exercise_induced_angina'].map({
    'No': 0,
    'Yes': 1
})

df['slope'] = df['slope'].map({
    'Upsloping': 0,
    'Flat': 1,
    'Downsloping': 2
})

df['vessels_colored_by_flourosopy'] = df['vessels_colored_by_flourosopy'].map({
    'Zero': 0,
    'One': 1,
    'Two': 2,
    'Three': 3
})

df['thalassemia'] = df['thalassemia'].map({
    'Normal': 0,
    'Fixed Defect': 1,
    'Reversable Defect': 2
})

# --- Save clean numeric version ---
df.to_csv('HeartDiseaseTrain-Test.csv', index=False)

print(" Cleaned data saved to HeartDiseaseTrain-Test.csv")
print(df.head())

# --- Train Model ---
@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, accuracy

# --- Streamlit UI ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

model, scaler, acc = train_model()

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=45)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type (0-3)", (0, 1, 2, 3))
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))

with col2:
    restecg = st.selectbox("Resting ECG (0-2)", (0, 1, 2))
    thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", (0, 1))
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of ST Segment (0-2)", (0, 1, 2))
    ca = st.selectbox("Major Vessels Colored (0-4)", (0, 1, 2, 3, 4))
    thal = st.selectbox("Thal (0 = normal, 1 = fixed defect, 2 = reversible defect)", (0, 1, 2))

# --- Prediction ---

if st.button("Predict"):
    try:
        # Convert categorical fields to numeric
        sex_numeric = 1 if sex == "Male" else 0
        fbs_numeric = int(fbs)
        restecg_numeric = int(restecg)
        exang_numeric = int(exang)
        slope_numeric = int(slope)
        ca_numeric = int(ca)
        thal_numeric = int(thal)
        cp_numeric = int(cp)

        # Combine all numeric inputs
        input_data = np.array([[
            float(age), sex_numeric, cp_numeric, float(trestbps),
            float(chol), fbs_numeric, restecg_numeric, float(thalach),
            exang_numeric, float(oldpeak), slope_numeric, ca_numeric, thal_numeric
        ]])

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # Display result
        if prediction[0] == 1:
            st.error("⚠️ High Risk: The person may have heart disease.")
        else:
            st.success("✅ Low Risk: The person is unlikely to have heart disease.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

