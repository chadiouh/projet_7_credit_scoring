import streamlit as st
import requests
import json

# === Titre et description ===
st.title("🧠 Interface Credit Scoring")
st.write("Remplissez les variables principales pour prédire si un client est solvable.")

# === Chargement des top features ===
with open("top_features.json", "r") as f:
    top_features = json.load(f)

# === Création dynamique du formulaire ===
user_input = {}
for feature in top_features:
    user_input[feature] = st.text_input(f"{feature} :")

# === Bouton de prédiction ===
if st.button("Prédire"):
    try:
        # Nettoyage et conversion
        input_data = {k: float(v) for k, v in user_input.items() if v.strip() != ""}
    except ValueError:
        st.error("Toutes les entrées doivent être numériques.")
    else:
        payload = {"data": input_data}

        try:
            # ✅ Appel vers le bon endpoint
            response = requests.post(
                "https://projet-7-credit-scoring-api.onrender.com/predict",
                json=payload
            )
            response.raise_for_status()  # Gestion erreurs HTTP

            result = response.json()
            st.success(f"✅ Probabilité d'insolvabilité : {result['proba']}")
            st.info(f"Prédiction finale : {'Non solvable' if result['prediction'] == 1 else 'Solvable'}")
            st.caption(f"Seuil utilisé : {result['threshold']}")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur réseau : {e}")
        except ValueError:
            st.error("Erreur : réponse inattendue de l’API (non JSON).")

