import streamlit as st
import pandas as pd
import requests
import json

# === Chargement des features ===
with open("top_features.json", "r") as f:
    top_features = json.load(f)

API_URL = "https://projet-7-credit-scoring-api.onrender.com/predict"

st.title("🧠 Interface Credit Scoring")
st.write("Remplissez les variables principales pour prédire si un client est solvable.")

# === Interface utilisateur dynamique
user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(feature, step=1.0)

if st.button("Prédire"):
    try:
        # Envoi de la requête
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"data": [user_input]},
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            if prediction is not None:
                if prediction == 1:
                    st.success("✅ Ce client est solvable.")
                else:
                    st.error("❌ Ce client n'est pas solvable.")
            else:
                st.warning("Réponse inattendue de l'API.")
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur réseau : {e}")


