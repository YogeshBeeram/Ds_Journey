from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

wine = load_wine()
X = pd.DataFrame(wine.data, columns = wine.feature_names)
y = wine.target

# Hyperparameter tuning for Random Forest  

# Smaller grid — less combinations!
param_grid = {
    'n_estimators': [50, 100],    # only 2 options
    'max_depth': [3, None],       # only 2 options
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=3)  # cv=3 instead of 5
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_*100:.2f}%")
print(f"Combinations tried: {2*2*3} tests total")
