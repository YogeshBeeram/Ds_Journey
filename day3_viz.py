import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df["species"] = iris.target

plt.figure(figsize = (8,4))
df["sepal length (cm)"].hist(bins =20, color = "orange")
plt.title("sepal length distribution")
plt.savefig("hist.png")
print("Histogram saved!")

plt.figure(figsize = (8,4))
colors = ["red", "green", "blue"]
for i in range(3):
    subset = df[df["species"] == i]
    plt.scatter(subset['sepal length (cm)'],
                subset['petal length (cm)'],
                label=iris.target_names[i],
                color=colors[i])
plt.xlabel("Sepal Length ")
plt.ylabel("Petal Length")
plt.title("sepal vs petal length by species")
plt.legend()
plt.savefig("scatter.png")
print("Scatter plot saved!")

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('heatmap.png')
print("Heatmap saved!")

print("\nAll 3 charts saved!")

plt