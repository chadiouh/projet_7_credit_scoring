from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os

# === Initialisation FastAPI ===
app = FastAPI(title="Credit Scoring API", description="API de prédiction LightGBM optimisé", version="1.0")

# === Détection du chemin absolu du fichier ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Chargement sécurisé des artefacts ===
def safe_load(path, mode="pkl"):
    try:
        if mode == "pkl":
            return joblib.load(path)
        elif mode == "json":
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement de {os.path.basename(path)} : {e}")

try:
    model = safe_load(os.path.join(BASE_DIR, "model_final.pkl"))
    imputer = safe_load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    baseline_row = safe_load(os.path.join(BASE_DIR, "baseline_row.json"), mode="json")
    top_features = safe_load(os.path.join(BASE_DIR, "top_features.json"), mode="json")
except RuntimeError as err:
    raise HTTPException(status_code=500, detail=str(err))

BEST_THRESHOLD = 0.42  # ajustable

class InputData(BaseModel):
    data: dict

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Credit Scoring"}

@app.post("/predict")
def predict(input: InputData):
    try:
        input_data = input.data

        # Remplissage avec valeurs par défaut
        complete_row = baseline_row.copy()
        complete_row.update(input_data)

        X = pd.DataFrame([complete_row])
        X_imputed = imputer.transform(X)

        y_proba = model.predict_proba(X_imputed)[:, 1][0]
        y_pred = int(y_proba >= BEST_THRESHOLD)

        return {
            "proba": round(y_proba, 4),
            "prediction": y_pred,
            "threshold": BEST_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction : {e}")
