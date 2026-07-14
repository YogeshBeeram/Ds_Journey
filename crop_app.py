import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('crop_production.csv')
df.dropna(inplace=True)


le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df['State_Name'] = le_state.fit_transform(df['State_Name'])
df['Crop'] = le_crop.fit_transform(df['Crop'])
df['Season'] = le_season.fit_transform(df['Season'])


X = df[['State_Name', 'Crop', 'Season', 'Area']]
y = df['Production']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


# Strealit App

st.title("Indian Crop Yield Predictor")
st.write("Predict crop production based on state, crop and area!")


# user Input
state = st.selectbox("Select State", le_state.classes_)
crop = st.selectbox("Select Crop", le_crop.classes_)
season = st.selectbox("Select Season", le_season.classes_)
area = st.slider("Area (hectares)", 1, 100000,1000)

if st.button("Predict Production "):
    state_enc = le_state.transform([state])[0]
    crop_enc = le_crop.transform([crop])[0]
    season_enc = le_season.transform([season])[0]

    prediction = model.predict([[state_enc, crop_enc, season_enc, area]])
    st.success(f"Predicted Production: {prediction[0]:,.0f} tonnes")

    st.write("### Your Selection:")
    st.table(pd.DataFrame({
        'Input': ['State', 'Crop', 'Season', 'Area'],
        'Value': [state, crop, season, f"{area} hectares"]
    }))

st.write("---")
st.write("Built with Random Forest + Real Indian Data 🇮🇳")