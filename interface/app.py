import streamlit as st
import json
import requests

# === CONFIGURATION ===
API_URL = "https://projet-7-credit-scoring-api.onrender.com/predict"  # Remplace ici par ton URL exacte
SEUIL = 0.5  # Seuil de décision métier

# === CHARGEMENT DES VARIABLES ===
with open("top_features.json", "r") as f:
    top_features = json.load(f)

# === TITRE ===
st.title("Prédiction de scoring client")

# === FORMULAIRE UTILISATEUR ===
st.markdown("**Remplis les variables principales :**")
user_input = {}

for feature in top_features:
    user_input[feature] = st.number_input(feature, value=0.0)

# === BOUTON DE PREDICTION ===
if st.button("Lancer la prédiction"):
    try:
        values_list = [user_input[feature] for feature in top_features]
        response = requests.post(API_URL, json={"values": values_list})

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"📊 Probabilité d’insolvabilité : {round(prediction, 4)}")

            # Affichage explicite de la décision
            if prediction >= SEUIL:
                st.error("❌ Crédit refusé (risque élevé)")
            else:
                st.success("✅ Crédit accordé (risque acceptable)")

        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la requête : {e}")








