import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df["species"] = iris.target

print(df.head(10))

print(df.describe())

#Single Conditions
big_flower = df[df[ "sepal length (cm)"] >6.0]
print(f"Flower with sepal> 6cm:{len(big_flower)} flowers")

# multiple conditions
filtered = df[(df["sepal length (cm)"] >6.0) & (df["species"] == 2)]
print(f"large flowers of species 2: {len(filtered)}")

# group by
grouped = df.groupby("species").mean()
print("Avg measurements per species:")
print(grouped)

# sorting
top5 = df.sort_values("sepal length (cm)", ascending=False).head()
print(top5)

# New column
df["sepal_ratio"] = df['sepal length (cm)'] / df["sepal width (cm)"]
print(df[['sepal length (cm)', 'sepal width (cm)', 'sepal_ratio']].head())

print("Advanced pandas done!")

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



iris = load_iris()
X = iris.data      # Features (input)
y = iris.target    # Target (what we predict)

print(f"Total samples: {len(X)}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")

# 2. SPLIT — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


print(f"Testing samples: {len(X_test)}")



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=iris.target_names))



new_flower = [[5.1, 3.5, 1.4, 0.2]]  
prediction = model.predict(new_flower)
print(f"New flower measurements: {new_flower[0]}")
print(f"Predicted species: {iris.target_names[prediction[0]]}")
