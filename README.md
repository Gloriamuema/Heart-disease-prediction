# Heart-disease-prediction Project

A Streamlit-based machine learning web application that predicts the likelihood of heart disease based on patient health indicators.
The model uses Logistic Regression for prediction and provides personalized feedback and recommendations for heart health improvement.

# Overview
Heart disease is one of the leading causes of death globally.
This app helps identify individuals who may be at higher risk by analyzing clinical parameters such as:

1. Age
2. Blood pressure
3. Cholesterol levels
4. Heart rate
5. ECG results
6. Exercise-induced angina

It uses a trained logistic regression model to output a probability score and provides insightful, easy-to-understand recommendations.

# Key Features
1. Interactive UI – Input patient details easily through Streamlit widgets.
2. Instant Prediction – Get real-time results on heart disease likelihood.
3. Health Recommendations – Personalized advice based on prediction results.
4. Accuracy Display – View the trained model’s performance.
5. Clean Visual Design – A minimal and user-friendly layout.

# Tech Stack
1. Component	# Technology
2. Frontend	# Streamlit
3. Data Handling	# Pandas
4. Machine Learning	# Scikit-Learn
5. Visualization	# Matplotlib
7. Language       # Python 3.8+

# Project Structure
# project Name: heart_disease_prediction
1. HeartDiseaseTrain-Test.csv   # Dataset file
2. heart_app.py                 # Main Streamlit app
3. requirements.txt             # Dependencies
4. README.md                    # Documentation
5. Heart_Disease_Notebook.ipynb # Notebook

# Installation & Setup
Clone the repository
git clone https://github.com/GloriamuemaHeart-disease-prediction.git
cd Heart-disease-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add the dataset
Ensure you have the HeartDiseaseTrain-Test.csv file in the project root directory.

# Run the app
python -m streamlit run heart_app.py

#  Model Details

Algorithm: Logistic Regression
Feature Scaling: StandardScaler
Test Size: 20%
Evaluation Metric: Accuracy Score
Explainability: SHAP (removed graph for simplicity; textual feature impact provided)

# Example Output

✅ Low likelihood of heart disease (12.45%)

“Good news! Your risk of heart disease is currently low.”

⚠️ High likelihood of heart disease (85.73%)

“You have risk factors that increase the likelihood of heart disease.”

# Future Improvements

 Add more advanced models (Random Forest, XGBoost).
 Deploy on Streamlit Cloud or Hugging Face Spaces.
 Include SHAP visualizations for feature importance.
 Add user authentication and patient history tracking.
 Integrate with wearable health data APIs.

# Author
Gloria Muema