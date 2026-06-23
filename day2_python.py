# 1.list
players = ["Virat","Rohit","Dhoni","Bumrah"]
print("First player:", players[0])

# 2.Dictionaies
player_info = {"name":"Virat Kohi","Runs":12000}
print("player:", player_info["name"])

# 3.Functions
def great_player(name,runs):
    return f"{name} has scored {runs} runs!"

print(great_player ("Rohit Sharma", 9500))


#4 loops
for player in players:
    print(f"Player: {player}")


#5. If/else

score = 85
if score >= 90:
    print("Excellent!")
elif score >=70:
    print("Good score!")
else:
    print("Need to improve")


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

print("\nSpecies count:")
print(df['species'].value_counts())

# Chart
df['sepal length (cm)'].plot(kind='hist', color='orange', bins=20)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.savefig('iris_chart.png')
print("\nChart saved as iris_chart.png!")