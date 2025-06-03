import requests
import json

def test_api_prediction():
    # 🔗 Remplace par ton URL réelle de l'API déployée
    url = "https://ton-api-sur-render.com/predict"

    # 📄 Exemple de données d’entrée (remplace par un vrai extrait de baseline_row.json)
    input_data = {
        "CODE_GENDER": 0,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.4,
        "FLAG_OWN_REALTY": 1,
        "FLAG_PHONE": 1,
        "NAME_FAMILY_STATUS": 1,
        "FLAG_DOCUMENT_3": 1,
        "REG_CITY_NOT_WORK_CITY": 0,
        "FLAG_OWN_CAR": 0,
        "NAME_EDUCATION_TYPE": 1,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "FLAG_WORK_PHONE": 0,
        "CNT_CHILDREN": 1,
        "prevapp_NFLAG_INSURED_ON_APPROVAL_max": 0,
        "OCCUPATION_TYPE": 3,
        "prevapp_NFLAG_INSURED_ON_APPROVAL_mean": 0.5,
        "prevapp_CNT_PAYMENT_max": 48,
        "DAYS_EMPLOYED": -1200,
        "DEF_30_CNT_SOCIAL_CIRCLE": 1,
        "bureau_AMT_CREDIT_SUM_DEBT_sum": 50000
    }

    # Envoi de la requête
    response = requests.post(url, json=input_data)

    # ✅ Test 1 : la requête répond
    assert response.status_code == 200

    # ✅ Test 2 : le corps contient une prédiction
    result = response.json()
    assert "score" in result or "prediction" in result or "proba" in result
