# projet_7_credit_scoring

## 🔌 API – Service de prédiction de score de crédit
## 🎯 Objectif de l’API
L’API de ce projet a été développée avec FastAPI dans le but de fournir un service de scoring de crédit accessible via une requête HTTP.
Elle permet de prédire la probabilité de défaut d’un client à partir de ses informations personnelles et financières.

## 🧩 Fonctionnement de l’API
L’utilisateur envoie une requête POST contenant un JSON avec uniquement les 15 variables les plus importantes du modèle (déterminées par l’analyse SHAP).
Ces variables sont les plus pertinentes pour évaluer le risque client.

Les autres variables nécessaires au modèle sont automatiquement complétées à l’aide d’un fichier de base (baseline_row.json) qui contient des valeurs médianes ou modalités les plus fréquentes.

Une fois les données complètes :

Elles sont transformées selon le préprocesseur du projet.

Elles sont passées au modèle entraîné (chargé via un fichier .pkl).

Le modèle retourne une probabilité de défaut entre 0 et 1.

Cette probabilité est renvoyée à l’utilisateur au format JSON.

## 💻 Intégration avec l’interface utilisateur
Une interface Streamlit a été développée pour permettre une utilisation simple et intuitive de cette API.

L’utilisateur remplit un formulaire contenant les 15 variables importantes.
Au moment de valider, l’interface envoie une requête à l’API déployée sur le cloud (Render).

L’interface affiche ensuite :

La probabilité de défaut prédit par le modèle.

Un message clair indiquant si la demande de crédit serait acceptée ou refusée selon le seuil de décision défini.

## 🌐 Déploiement
L’API est hébergée sur Render.

L’interface Streamlit est également déployée sur Render et communique avec l’API en ligne.

L’ensemble peut être utilisé depuis n’importe quel navigateur, sans besoin d’installation locale.
