# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import joblib
import os

# Chargement des top features
root_path = r"C:\Users\chouh\Projet7"
features_path = os.path.join(root_path, "artifacts", "top_10_features.pkl")
top_features = joblib.load(features_path)

st.set_page_config(page_title="Scoring Credit", layout="centered")
st.title("Scoring de credit client")
st.markdown("Remplissez les champs ci-dessous pour obtenir une prediction de defaut.")

# === Formulaire dynamique ===
with st.form("formulaire_credit"):
    input_data = {}
    for feature in top_features:
        value = st.text_input(f"{feature} :", key=feature)
        input_data[feature] = value

    submit = st.form_submit_button("Obtenir une prediction")

# === Envoi à l'API ===
if submit:
    try:
        # Conversion automatique : float si possible
        for key in input_data:
            try:
                input_data[key] = float(input_data[key])
            except:
                pass

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"data": input_data}
        )

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction effectuee avec succes.")
            st.metric("Probabilite de defaut", f"{result['probability'] * 100:.2f} %")
            st.metric("Decision",
                      "Client a risque ❌" if result["prediction"] == 1 else "Client acceptable ✅")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Erreur technique : {e}")






