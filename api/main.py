import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Chargement des fichiers
with open("top_features.json", "r") as f:
    top_features = json.load(f)

with open("baseline_row.json", "r") as f:
    baseline_row = json.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("model_final.pkl", "rb") as f:
    model = pickle.load(f)

# Initialisation de l'API
app = FastAPI()

# Modèle de données (uniquement les 15 variables)
class InputData(BaseModel):
    values: List[float]

@app.get("/")
def read_root():
    return {"message": "API de scoring opérationnelle."}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Vérification de la taille d'entrée
        if len(input_data.values) != len(top_features):
            raise HTTPException(status_code=400, detail=f"Attendu {len(top_features)} valeurs, reçu {len(input_data.values)}")

        # Construction d'une ligne complète avec toutes les features
        full_input = baseline_row.copy()
        for i, feature in enumerate(top_features):
            full_input[feature] = input_data.values[i]

        # Transformation en DataFrame
        X = pd.DataFrame([full_input])

        # Préprocessing
        X_processed = preprocessor.transform(X)

        # Prédiction
        prediction = model.predict_proba(X_processed)[0, 1]
        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



