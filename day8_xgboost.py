from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import time

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# Random Forest 
start = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_time = time.time() - start


print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}%")
print(f"Random Forest Time: {rf_time:.4f} sec")


# Xgboost

start = time.time()
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_time = time.time() - start


print(f"\nXGBoost Accuracy: {xgb_accuracy*100:.2f}%")
print(f"XGBoost Time: {xgb_time:.4f} sec")


print("\n=== XGBOOST FEATURE IMPORTANCE ===")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.head(5))

print("\n=== COMPARISON ===")
print(f"Random Forest: {rf_accuracy*100:.2f}% in {rf_time:.4f}s")
print(f"XGBoost:       {xgb_accuracy*100:.2f}% in {xgb_time:.4f}s")
winner = "XGBoost" if xgb_accuracy > rf_accuracy else "Random Forest"
print(f"\n🏆 Winner: {winner}")