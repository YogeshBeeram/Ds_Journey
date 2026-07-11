from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target


# Random Forest

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf_model, X,y,cv = 5)

print("Random Forest scores on 5 different splits:")
for i, score in enumerate(rf_scores):
    print(f"  Split {i+1}: {score*100:.2f}%")
print(f"Average: {rf_scores.mean()*100:.2f}%")
print(f"Std Dev: {rf_scores.std()*100:.2f}%")


#Xgboost

xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_scores = cross_val_score(xgb_model,X,y, cv=5)

print("XGBoost scores on 5 different splits:")
for i, score in enumerate(xgb_scores):
    print(f"  Split {i+1}: {score*100:.2f}%")
print(f"Average: {xgb_scores.mean()*100:.2f}%")
print(f"Std Dev: {xgb_scores.std()*100:.2f}%")


if rf_scores.mean() > xgb_scores.mean():
    print(f"Random Forest wins! ({rf_scores.mean()*100:.2f}%)")
else:
    print(f"XGBoost wins! ({xgb_scores.mean()*100:.2f}%)")