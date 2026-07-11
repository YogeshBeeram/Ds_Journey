import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# load data

df =pd.read_csv("crop_production.csv")

df.dropna(inplace=True)
print(f"Clean data shape: {df.shape}")

le = LabelEncoder()
df['State_Name'] = le.fit_transform(df['State_Name'])
df['Crop'] = le.fit_transform(df['Crop'])
df['Season'] = le.fit_transform(df['Season'])

X = df[['State_Name', 'Crop', 'Season', 'Area']]
y = df['Production']

print(f"Features: {X.columns.tolist()}")
print(f"Target: Production")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)
print("\nCrop yield model done!")