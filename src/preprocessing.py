import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=== PREPROCESSING ===")

df = pd.read_csv('data/raw/test.csv')

df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df.to_csv('data/processed/clean.csv', index=False)

print("Data bersih berhasil disimpan!")