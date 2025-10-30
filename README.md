# Heart-disease-prediction Project

A Streamlit-based machine learning web application that predicts the likelihood of heart disease based on patient health indicators.
The model uses Logistic Regression for prediction and provides personalized feedback and recommendations for heart health improvement.

# Overview
Heart disease is one of the leading causes of death globally.
This app helps identify individuals who may be at higher risk by analyzing clinical parameters such as:

Age
Blood pressure
Cholesterol levels
Heart rate
ECG results
Exercise-induced angina

It uses a trained logistic regression model to output a probability score and provides insightful, easy-to-understand recommendations.

# Key Features
✅ Interactive UI – Input patient details easily through Streamlit widgets.
✅ Instant Prediction – Get real-time results on heart disease likelihood.
✅ Health Recommendations – Personalized advice based on prediction results.
✅ Model Accuracy Display – View the trained model’s performance.
✅ Clean Visual Design – A minimal and user-friendly layout.

# Tech Stack
Component	Technology
Frontend	Streamlit
Data Handling	Pandas
Machine Learning	Scikit-Learn
Visualization	Matplotlib
Language Python 3.8+

📁 Project Structure
heart_disease_app/
HeartDiseaseTrain-Test.csv   # Dataset file
heart_app.py                 # Main Streamlit app
requirements.txt             # Dependencies
README.md                    # Documentation
Heart_Disease_Notebook.ipynb # Notebook

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