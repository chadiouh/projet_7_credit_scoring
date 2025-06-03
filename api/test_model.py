import joblib
import json
import pandas as pd
import os

# === Chargement des artefacts ===
model = joblib.load("model_final.pkl")
preprocessor = joblib.load("preprocessor.pkl")

with open("baseline_row.json", "r") as f:
    baseline_row = json.load(f)

# === Conversion en DataFrame
X = pd.DataFrame([baseline_row])

# === Test de transformation
try:
    X_transformed = preprocessor.transform(X)
    print("✅ Transformation réussie - shape:", X_transformed.shape)
except Exception as e:
    print("❌ Erreur lors de la transformation :", e)

# === Test de prédiction
try:
    y_proba = model.predict_proba(X_transformed)[:, 1][0]
    print("✅ Prédiction réussie - proba :", round(y_proba, 4))
except Exception as e:
    print("❌ Erreur lors de la prédiction :", e)
