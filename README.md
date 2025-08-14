
# Initialize git if not already
```bash
git init
```
# Add all project files
```bash
git add .
```
# Commit changes
```bash
git commit -m "Initial commit - Bankruptcy Prediction ML Pipeline"
```
# Push to GitHub main branch
git branch -M main
git push -u origin main

If your repo already exists and you’ve made changes:
```bash
git add .
git commit -m "Updated ML pipeline and README"
git push
```

# Bankruptcy Prediction - End-to-End ML Pipeline

## 📌 Overview
This project implements a complete **Machine Learning pipeline** for predicting company bankruptcy based on financial indicators.  
It covers **data preprocessing, feature engineering, model selection, hyperparameter tuning, class imbalance handling, explainability, and interpretation**.

---

## 📂 Project Structure

```bash
.
├── data.csv # Input dataset
├── pipeline.ipynb # Full ML pipeline notebook
├── app.py # Optional script for deployment
├── README.md # Project documentation
└── requirements.txt # Python dependencies

```
---

## 🚀 Features
- **Choosing Initial Models**: Logistic Regression (baseline) and advanced models like XGBoost, Random Forest, CatBoost.
- **Data Preprocessing**: Scaling (for LogReg), encoding, missing value handling.
- **Class Imbalance Handling**: Class weighting and SMOTE.
- **Outlier Detection**: Isolation Forest, Winsorization (for scale-sensitive models).
- **Data Normalization**: StandardScaler for linear models.
- **Normality Testing**: Skewness analysis and log-transform for extreme skew.
- **Dimensionality Reduction**: PCA (optional).
- **Feature Engineering**: Domain-specific ratios, trend aggregates.
- **Multicollinearity Testing**: VIF analysis and correlation filtering.
- **Feature Selection**: Correlation threshold, XGBoost gain, Lasso regression.
- **Hyperparameter Tuning**: Randomized Search + Bayesian Optimization.
- **Model Interpretation**: SHAP values, feature importance plots.

---

## 📊 Dataset
The dataset contains **financial ratios and indicators** for companies with a binary target:
- **`Bankrupt?`** → `1` = bankrupt, `0` = not bankrupt.

Example features:
- ROA(C) before interest and depreciation
- Debt ratio %
- Operating gross margin
- Working capital to total assets

---

## 🛠 Installation

### 1. Clone this repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
▶️ Usage

Run the pipeline:
```bash
python pipeline.py
```

If you have a notebook:
```bash
jupyter notebook pipeline.ipynb
```
### ⚙️ Models Used

- Logistic Regression – Baseline, interpretable.

- Random Forest – Handles non-linearities.

- XGBoost – Powerful gradient boosting.

- CatBoost – Handles categorical features natively.

### 📈 Performance Metrics

We use:

- Precision

- Recall

- F1-Score

- ROC-AUC

### 📌 Explainability

- SHAP plots for feature impact.

- Permutation Importance for model-agnostic insights.

- Coefficients for Logistic Regression interpretability.

### 🧠 Methodology Summary

- Load & clean data.

- Train/test split (stratified).

- Handle missing values.

- Scale numeric data for linear models.

- Check for imbalance and apply SMOTE/class weights.

- Outlier detection & treatment.

- Feature engineering & selection.

- Train multiple models and tune hyperparameters.

- Evaluate with cross-validation.

- Interpret results with SHAP & feature importance.
