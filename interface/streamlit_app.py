# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import os

# === Chargement des artefacts ===
model_path = os.path.join("artifacts", "best_model_lgbm_optimized.pkl")
preprocessor_path = os.path.join("artifacts", "custom_preprocessor.pkl")
features_path = os.path.join("artifacts", "top_10_features.pkl")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
top_features = joblib.load(features_path)

st.set_page_config(page_title="Scoring Credit", layout="centered")
st.title("Scoring de crédit client")
st.markdown("Remplissez les champs ci-dessous pour obtenir une prédiction de défaut.")

# === Formulaire dynamique ===
with st.form("formulaire_credit"):
    input_data = {}
    for feature in top_features:
        value = st.text_input(f"{feature} :", key=feature)
        input_data[feature] = value

    submit = st.form_submit_button("Obtenir une prédiction")

# === Fonction de prédiction locale ===
def predict_credit_risk(data_dict):
    df = pd.DataFrame([data_dict])
    X = preprocessor.transform(df)
    y_pred_proba = model.predict_proba(X)[0, 1]
    return y_pred_proba

# === Lancement de la prédiction ===
if submit:
    try:
        # Conversion des entrées utilisateur
        for key in input_data:
            try:
                input_data[key] = float(input_data[key])
            except:
                pass

        proba = predict_credit_risk(input_data)
        prediction = int(proba >= 0.5)

        st.success("Prédiction effectuée avec succès.")
        st.metric("Probabilité de défaut", f"{proba:.2%}")
        st.metric("Décision",
                  "Client à risque ❌" if prediction == 1 else "Client acceptable ✅")

    except Exception as e:
        st.error(f"Erreur technique : {e}")







