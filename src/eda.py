import pandas as pd

print("=== EDA ===")

df = pd.read_csv('data/raw/test.csv')

print("\nDistribusi Kepuasan:")
print(df['satisfaction'].value_counts())

print("\nMissing Value:")
print(df.isnull().sum())