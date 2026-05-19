import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=== MODELING ===")

# =====================================
# LOAD DATA BERSIH
# =====================================

df = pd.read_csv('data/processed/clean.csv')

# =====================================
# SPLIT DATA
# =====================================

X = df[[
    'Gender',
    'Age',
    'Flight Distance',
    'Inflight wifi service',
    'Online boarding'
]]
y = df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# SETUP MLFLOW
# =====================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Airline Satisfaction")

best_acc = 0
best_model = None
best_n = 0

# =====================================
# TRAINING MODEL
# =====================================

for n in [50, 100, 150]:

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # logging MLflow
        mlflow.log_param(
            "n_estimators",
            n
        )

        mlflow.log_metric(
            "accuracy",
            acc
        )

        mlflow.sklearn.log_model(
            model,
            "random_forest_model"
        )

        print(f"Model {n} → Accuracy: {acc}")

        # simpan model terbaik
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_n = n

# =====================================
# HASIL MODEL TERBAIK
# =====================================

print("\n=== MODEL TERBAIK ===")

print("n_estimators :", best_n)
print("Accuracy     :", best_acc)

# =====================================
# EVALUASI MODEL
# =====================================

y_pred = best_model.predict(X_test)

print("\n=== CONFUSION MATRIX ===")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\n=== CLASSIFICATION REPORT ===")

print(classification_report(y_test, y_pred))

# =====================================
# FEATURE IMPORTANCE
# =====================================

print("\n=== FEATURE IMPORTANCE ===")

importances = best_model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

print(feature_df.head(10))

# =====================================
# BUAT FOLDER MODELS
# =====================================

os.makedirs('models', exist_ok=True)

# =====================================
# SIMPAN FEATURE IMPORTANCE
# =====================================

feature_df.to_csv(
    'models/feature_importance.csv',
    index=False
)

# =====================================
# SIMPAN MODEL
# =====================================

joblib.dump(
    best_model,
    'models/best_model.pkl'
)

# =====================================
# VISUALISASI FEATURE IMPORTANCE
# =====================================

top_feature = feature_df.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top_feature['Feature'],
    top_feature['Importance']
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Top 10 Feature Importance")

plt.gca().invert_yaxis()

plt.show()

print("\nModel berhasil disimpan!")

print("\nFeature importance berhasil disimpan!")