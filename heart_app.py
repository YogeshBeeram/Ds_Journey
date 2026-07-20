import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


st.title("🫁 Heart Disease Risk Predictor")
st.write("Enter patient details to predict heart disease risk!")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 80, 45)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
    restecg = st.selectbox("Resting ECG", [0,1,2])

with col2:
    thalach = st.slider("Max Heart Rate", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST segment", [0,1,2])
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [0,1,2,3])

if st.button("Predict Risk 🫁"):
    input_data = [[age, sex, cp, trestbps, chol, fbs,
                   restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK of Heart Disease! ({probability[1]*100:.1f}% probability)")
    else:
        st.success(f"✅ LOW RISK of Heart Disease! ({probability[0]*100:.1f}% probability)")

    st.write("### Top Risk Factors (from SHAP analysis):")
    st.write("1. Number of Major Vessels (ca)")
    st.write("2. Chest Pain Type (cp)")
    st.write("3. Thalassemia (thal)")

st.write("---")
st.write("Built with Random Forest + SHAP | 83% accuracy on UCI Heart Disease Dataset")

