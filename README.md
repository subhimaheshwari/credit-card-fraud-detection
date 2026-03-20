# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using supervised classification techniques. The notebook covers the full data science pipeline — from exploratory analysis and preprocessing to model training, evaluation, and exporting flagged transactions.

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
│
├── credit-card-fraud-analysis.ipynb   # Main analysis and modeling notebook
├── best_fraud_model.pkl               # Serialized best-performing ML model
└── flagged_transactions.csv           # Output: transactions predicted as fraudulent
```

---

## 🔍 Project Overview

Credit card fraud is a significant challenge for financial institutions, with fraudulent transactions often being rare but costly. This project tackles the problem as a **binary classification task** (fraud vs. legitimate) on an imbalanced dataset.

Key steps in the pipeline:

- **Exploratory Data Analysis (EDA)** — understanding class imbalance, feature distributions, and correlations
- **Data Preprocessing** — scaling, handling imbalance (e.g., SMOTE or undersampling)
- **Model Training** — training and comparing multiple classifiers
- **Model Evaluation** — using metrics suitable for imbalanced data (Precision, Recall, F1-score, ROC-AUC)
- **Model Export** — saving the best model as `best_fraud_model.pkl`
- **Output Generation** — writing flagged (predicted fraudulent) transactions to `flagged_transactions.csv`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Jupyter Notebook | Development environment |
| pandas / NumPy | Data manipulation |
| scikit-learn | ML models and preprocessing |
| imbalanced-learn | Handling class imbalance |
| matplotlib / seaborn | Visualization |
| pickle / joblib | Model serialization |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/subhimaheshwari/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### 3. Add the dataset

This project uses a credit card transactions dataset (e.g., the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)). Download `creditcard.csv` and place it in the project root before running the notebook.

### 4. Run the notebook

```bash
jupyter notebook credit-card-fraud-analysis.ipynb
```

---

## 📊 Model Output

- **`best_fraud_model.pkl`** — the trained model with the best evaluation score, ready for inference
- **`flagged_transactions.csv`** — a list of transactions the model has predicted as fraudulent, useful for review or downstream alerting

### Loading the saved model

```python
import pickle

with open("best_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_new)
```

---

## ⚠️ Notes on Class Imbalance

Fraud detection datasets are typically highly imbalanced (e.g., <1% fraud). Standard accuracy is misleading in such cases. This project focuses on **Precision**, **Recall**, and **F1-score** as primary evaluation metrics, with the goal of minimising false negatives (missed fraud).

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♀️ Author

**Subhi Maheshwari**  
[GitHub](https://github.com/subhimaheshwari)
