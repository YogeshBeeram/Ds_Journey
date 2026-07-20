import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  classification_report
import warnings
warnings.filterwarnings("ignore")
import shap
import matplotlib.pyplot as plt
import numpy as np



url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)

print(df.head())
print(df.info())
print(df.describe())
print(df.shape)

# mISSING VALUES
print(df.isnull().sum())
# No missing values in the dataset

print(df["target"].value_counts())

X = df.drop("target", axis = 1)
y = df["target"]

X_train,X_test,y_train , y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

model =  RandomForestClassifier(n_estimators = 100, random_state = 42)
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))



explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values type: {type(shap_values)}")
print(f"SHAP values length: {len(shap_values)}")

# Get importance directly
if isinstance(shap_values, list):
    # Multiple classes — take class 1 (has disease)
    vals = np.array(shap_values[1])
else:
    vals = np.array(shap_values)

print(f"vals shape: {vals.shape}")
importance_vals = np.abs(vals).mean(axis=0).mean(axis=1)
print(f"importance_vals shape: {importance_vals.shape}")

shap_importance = pd.DataFrame(
    {
        'Feature': X.columns.tolist(),
        'SHAP Importance': importance_vals,
    }
).sort_values('SHAP Importance', ascending=False).reset_index(drop=True)
print(shap_importance)

print("\n=== TOP FEATURES FOR HEART DISEASE ===")
print(shap_importance)