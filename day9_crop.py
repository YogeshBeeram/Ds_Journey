import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("crop_production.csv")

print("Indian crop production dataset")
print(df.head())
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

print(df.isnull().sum())

#Basic analysis
print(df.groupby("State_Name")["Production"].sum().sort_values(ascending=False).head(10))

print(df["Season"].value_counts())

#Visualization
plt.figure(figsize=(12,5))
df.groupby("State_Name")["Production"].sum().sort_values(ascending=False).head(10).plot(kind="bar",color = "green")
plt.title("Top 10 states by crop production in India")
plt.xlabel("State")
plt.ylabel("Total Production")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('crop_production.png')
