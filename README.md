# 🛳️ Titanic Survival Prediction – Logistic Regression ML Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 🎯 **Obiettivo del Progetto**

Predire la sopravvivenza dei passeggeri del Titanic utilizzando tecniche di **regressione logistica** e analisi statistica, con una pipeline chiara dal caricamento dati alla valutazione del modello.
---

## 📋 Table of Contents
- [📖 Introduction](#-introduction)
- [🧹 Data Preprocessing](#-data-preprocessing)
- [🧩 Feature Engineering](#-feature-engineering)
- [⚙️ Model Training](#️-model-training)
- [📈 Model Evaluation](#-model-evaluation)
- [🧠 Explainability (SHAP)](#-explainability-shap)
- [🎮 Interactive Prediction](#-interactive-prediction)

---
## 📊 **Pipeline Analitica**

- **Analisi Esplorativa**: Statistiche e visualizzazioni sui dati
- **Preprocessing**: Pulizia, gestione dei missing values con KNNImputer
- **Feature Engineering**: Creazione FamilySize, codifica variabili categoriali
- **Modellazione**: Addestramento e selezione automatica delle features
- **Valutazione**: Accuracy, precision, recall, F1-score e interpretabilità con SHAP
## 📖 Introduction

This notebook-based project explores the famous **Kaggle Titanic dataset** to predict passenger survival using **Logistic Regression**.

It covers the full **data science pipeline**:
1. Data loading & cleaning  
2. Feature engineering  
3. Model training with class balancing  
4. Evaluation (accuracy, recall, F1, confusion matrix)  
5. Explainability using **SHAP**  
6. Interactive prediction for new passengers  

This project is ideal as a **portfolio example** of applied data science and model interpretability.

---



## 🧹 Data Preprocessing

Key preprocessing steps:
- Missing values handled via **KNNImputer** (`Age`)
- Dropped non-informative columns: `Cabin`, `Name`, `Ticket`, `PassengerId`
- One-hot encoding of categorical variables (`Sex`, `Embarked`)
- Removal of remaining missing rows after cleaning

Example:
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df[['Age']] = imputer.fit_transform(df[['Age']])



---

## 🔍 Feature Engineering

### Feature Originali
- **Pclass**: Classe passeggero (1, 2, 3)
- **Age**: Età (imputata con KNN)
- **SibSp**: # Fratelli/Coniugi a bordo
- **Parch**: # Genitori/Figli a bordo
- **Fare**: Tariffa pagata
- **Sex**: Sesso (codificato: male=1, female=0)
- **Embarked**: Porto imbarco (C, Q, S)

### Feature Ingegnerizzate
- **FamilySize**: `SibSp + Parch + 1` - Dimensione nucleo familiare

### Feature Selezionate (RFE)
Le 8 feature più rilevanti identificate tramite **Recursive Feature Elimination**:
1. Sex (male)
2. Pclass
3. Age
4. Fare
5. SibSp
6. FamilySize
7. Embarked_S
8. Parch

---

## 🏆 **Metriche del Modello**

| **Metrica**        | **Score**  |
|:-------------------|:----------:|
| Accuracy           | 0.81       |
| Precision          | 0.78       |
| Recall             | 0.72       |
| F1-score           | 0.75       |
| ROC-AUC            | 0.84       |

---

## 🎨 **Visualizzazioni**

- Distribuzioni delle variabili con **Seaborn**
- Heatmap delle correlazioni con palette *coolwarm*
- Importanza delle feature (permutation e SHAP)
- Grafici salvatati in `results/figures` pronti per essere integrati nel README

---

## 💡 **Esempio di Predizione**

from src.model import TitanicSurvivalModel

model = TitanicSurvivalModel.load('models/logistic_model.pkl')
result = model.predict_passenger(Pclass=3, Age=30, SibSp=3, Parch=0, Fare=80, FamilySize=4, male=0, Q=0, S=1)
print(result)

