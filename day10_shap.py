import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load crop data
df = pd.read_csv('crop_production.csv')
df.dropna(inplace=True)

# Encode categories
le = LabelEncoder()
df['State_Name'] = le.fit_transform(df['State_Name'])
df['Crop'] = le.fit_transform(df['Crop'])
df['Season'] = le.fit_transform(df['Season'])

# Features
X = df[['State_Name', 'Crop', 'Season', 'Area']]
y = df['Production']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# ===== SHAP EXPLANATION =====
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])

# SHAP Summary Plot
print("Creating SHAP plot...")
shap.summary_plot(shap_values, X_test[:100],
                  feature_names=X.columns.tolist(),
                  show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()
print("SHAP plot saved as shap_summary.png!")

# Most important features from SHAP
print("\n=== SHAP FEATURE IMPORTANCE ===")
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'SHAP Importance': abs(shap_values).mean(axis=0)
}).sort_values('SHAP Importance', ascending=False)
print(shap_importance)

print("\nSHAP analysis complete!")