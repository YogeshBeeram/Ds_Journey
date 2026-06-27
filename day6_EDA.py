import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = sns.load_dataset("tips")

print(df.shape)
#(244, 7)
print(df.head(10))

# 'total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'

print(df.columns.tolist())
print(df.describe())

#checking any missing data
print(df.isnull().sum())
# No missing data

# Avg Tip by day
print(df.groupby("sex")['tip'].mean())
# Male people are giving more tips while compared to female customers

# Avg smoker tip
print(df.groupby("smoker")['tip'].mean())
# there is no significant gap between Smoker and Non Smoker  

# Avg day tip
print(df.groupby("day")['tip'].mean().sort_values(ascending= False))
# Sunday we are getting more tips

# Data visualization

plt.figure(figsize=(8,5))
df.groupby("day")['tip'].mean().plot(kind = "bar", color = "orange")
plt.title("Avg tip by day")
plt.tight_layout()
plt.savefig('tips_by_day.png')
print("\nChart 1 saved!")


plt.figure(figsize=(8,5))
sns.scatterplot(x='total_bill', y='tip', hue='sex', data=df)
plt.title('Total Bill vs Tip')
plt.tight_layout()
plt.savefig('bill_vs_tip.png')
print("Chart 2 saved!")


plt.figure(figsize=(8,5))
sns.boxplot(x='day', y='tip', data=df)
plt.title('Tip Distribution by Day')
plt.tight_layout()
plt.savefig('tip_boxplot.png')
print("Chart 3 saved!")


