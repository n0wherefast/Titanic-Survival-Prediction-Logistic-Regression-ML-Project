# 🛳️ Titanic Survival Prediction – Logistic Regression ML Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Goal:** Predict whether a passenger survived the Titanic disaster using a logistic regression model, feature engineering, and explainable AI (SHAP).

---

## 📋 Table of Contents
- [📖 Introduction](#-introduction)
- [📂 Project Structure](#-project-structure)
- [🧹 Data Preprocessing](#-data-preprocessing)
- [🧩 Feature Engineering](#-feature-engineering)
- [⚙️ Model Training](#️-model-training)
- [📈 Model Evaluation](#-model-evaluation)
- [🧠 Explainability (SHAP)](#-explainability-shap)
- [🎮 Interactive Prediction](#-interactive-prediction)
- [🚀 Quick Start](#-quick-start)
- [🛠️ Tools & Requirements](#️-tools--requirements)
- [🏗️ Future Improvements](#️-future-improvements)
- [👤 Author](#-author)

---

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

## 📂 Project Structure

titanic-logistic-regression/
│
├── data/
│ └── U4_04_train.csv
│
├── notebook/
│ └── Titanic_Logistic_Regression.ipynb
│
├── models/
│ └── titanic_model.pkl
│
├── images/
│ ├── confusion_matrix.png
│ ├── shap_summary.png
│ └── shap_waterfall.png
│
├── export_notebook.py
├── requirements.txt
└── README.md


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


