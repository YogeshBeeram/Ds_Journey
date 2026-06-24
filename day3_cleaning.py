import pandas as pd
import numpy as np

# Create a messy dataset — like real world data!
data = {
    'name': ['Virat', 'Rohit', None, 'Dhoni', 'Virat'],
    'runs': [12000, 9500, 8000, None, 12000],
    'age': [34, 36, 28, 42, 34],
    'country': ['India', 'India', 'India', 'India', 'India']
}

df = pd.DataFrame(data)
print("ORIGINAL MESSY DATA:")
print(df)

# 1. CHECK MISSING VALUES
print("\nMissing values:")
print(df.isnull().sum())

# 2. FILL MISSING VALUES
df['runs'] = df['runs'].fillna(df['runs'].mean())
df['name'] = df['name'].fillna('Unknown')
print("\nAfter filling missing values:")
print(df)

# 3. REMOVE DUPLICATES
print(f"\nBefore removing duplicates: {len(df)} rows")
df.drop_duplicates(inplace=True)
print(f"After removing duplicates: {len(df)} rows")

# 4. RENAME COLUMNS
df.columns = ['Player', 'Runs', 'Age', 'Country']
print("\nFinal clean data:")
print(df)
print("\nData cleaning done!")