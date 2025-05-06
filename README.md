# 🍷 Wine Dataset Classification with RandomForest & MLflow

This project trains a **RandomForestClassifier** on the classic **Wine dataset** using scikit-learn, and logs parameters, metrics, and the model itself with **MLflow** for experiment tracking.

---

## 📦 Project Overview

- **Goal:** Classify different types of wine using chemical features.
- **Model:** RandomForestClassifier
- **Tracking:** MLflow
- **Dataset:** [Scikit-learn Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)

---

## 🔧 Environment Setup

1. **Create a virtual environment**

```bash
python -m venv venv

```
```bash
venv\Scripts\activate
```
```bash
pip install -r requirements.txt

```
.
├── train_wine_rf.py          # Training script with MLflow integration
├── requirements.txt          # List of dependencies
└── README.md                 # This file
