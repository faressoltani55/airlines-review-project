# **Analyse des avis de voyageurs aériens avec PySpark et Streamlit**

## **Description du projet**
Ce projet utilise des outils de traitement de données tels que **PySpark** et **Streamlit** pour analyser et prédire les sentiments des avis de passagers des compagnies aériennes.
Il comprend plusieurs étapes d'analyse de données, de traitement de texte, et de création de modèles prédictifs pour obtenir des informations sur la satisfaction des clients.

### **Structure du projet**
1. **Tab: Base de Données** : Aperçu des données.
2. **Analyse et ingénierie des caractéristiques** : Analyse des données de la base par compagnie et type de voyageur.
3. **Page des prédictions** : Utilisation de **PySpark MLlib** pour effectuer l'analyse de sentiment basée sur des avis,
et création d'un modèle prédictif pour prédire des résultats comme la note globale ou la satisfaction des clients.

## **Installation**

### Prérequis
- Python 3.10
- PySpark
- Streamlit
- pandas
- matplotlib
- scikit-learn (pour certains outils de visualisation ou évaluation)

### Installation des dépendances

Installez les dépendances en utilisant `pip` :

```bash
pip install -r requirements.txt
```

## **Description**

#### **Objectifs principaux :**
- Nettoyage et prétraitement des données.
- Analyse des tendances, corrélations et schémas dans les avis.
- Extraction des sentiments des avis à l'aide d'une analyse basée sur un modèle pré-entraîné.
- Prétraitement des textes des avis.
- Extraction de caractéristiques avec **TF-IDF**.
- Application d'un modèle de régression logistique pour la classification des avis (positif ou négatif).
- Calcul de la précision du modèle (Accuracy) pour évaluer ses performances.
- Entraîner un modèle prédictif pour prédire des notes globales ou des sentiments.
- Évaluer les performances du modèle (RMSE).

## **Exécution du projet**

### Lancer Streamlit

Pour lancer l'application Streamlit et visualiser les pages de votre projet, utilisez la commande suivante dans votre terminal :

```bash
python -m streamlit run app.py
```

Cela ouvrira l'application dans votre navigateur où vous pourrez interagir avec les différentes pages du projet.

## **Structure du Code**

- `Home.py`: Fichier principal qui contient la logique Streamlit pour la navigation entre les tabs.
- `util.py`: Contient la logique pour entraîner le modèle de machine learning avec PySpark et évaluer ses performances, ainsi que les fonctions de pré-traitements des données.
- `data/`: Dossier contenant le jeu de données des avis des passagers.
- `requirements.txt`: Liste des bibliothèques nécessaires pour faire fonctionner le projet.

## **Améliorations futures**

- Améliorer la précision du modèle avec des algorithmes plus complexes comme **XGBoost** ou **Random Forest**.
- Ajouter une interface pour exporter les modèles vers des fichiers pickle.
- Implémenter un modèle de sentiment plus avancé en utilisant des embeddings de mots comme **Word2Vec**.
