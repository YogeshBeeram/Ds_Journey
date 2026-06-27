import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

X = pd.DataFrame(wine.data, columns = wine.feature_names)
y = wine.target


print(X.shape)
print(X.head(3))
print(wine.target_names)
print(X.columns)
print(X.describe())
print(X.columns)

#Split the data into Train and Test sets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

# Training Model for Randim forest, Logidtic Regression, Decision tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

models = {
    "Decision Tree" : DecisionTreeClassifier(random_state = 42),
    "Random Forest" : RandomForestClassifier(n_estimators = 100, random_state = 42),
    "Logistic Regression" : LogisticRegression(max_iter = 1000, random_state = 42)
}


print("Model comparison for wine dataset")

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy*100:.2f}%")


best = max(results, key=results.get)
print(f"\n🏆 Best model: {best} ({results[best]*100:.2f}%)")


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance.head(5))
print("\nTop feature:", importance.iloc[0]['Feature'])

