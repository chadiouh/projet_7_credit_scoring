# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialisation FastAPI
app = FastAPI(
    title="Credit Scoring API",
    description="API de prediction de defaut client avec LightGBM",
    version="1.0.0"
)

# Chemins
ROOT_DIR = r"C:\Users\chouh\Projet7"
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model_lgbm_optimized.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_custom.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold_metier.txt")
TEMPLATE_PATH = os.path.join(ARTIFACTS_DIR, "template_input.pkl")

# Chargement des artefacts
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    template = joblib.load(TEMPLATE_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read())
except Exception as e:
    raise RuntimeError(f"Erreur au chargement des artefacts : {e}")

# Schema d'entree
class ClientData(BaseModel):
    data: dict

@app.get("/")
def root():
    return {"message": "API operationnelle. Utilisez POST /predict pour faire une prediction."}

@app.post("/predict")
def predict(client_data: ClientData):
    try:
        input_df = pd.DataFrame([client_data.data])
        full_input = template.copy()
        full_input.iloc[0] = np.nan

        # Injection des valeurs utilisateur
        for col in input_df.columns:
            if col in full_input.columns:
                full_input[col] = input_df[col].values[0]
            else:
                raise HTTPException(status_code=400, detail=f"Colonne inconnue : {col}")

        # Transformation
        X_transformed = preprocessor.transform(full_input)
        proba = model.predict_proba(X_transformed)[:, 1][0]
        prediction = int(proba >= threshold)

        return {
            "probability": round(proba, 4),
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")




