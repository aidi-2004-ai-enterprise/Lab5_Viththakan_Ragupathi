# bankruptcy_full_pipeline_enhanced.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# =====================
# 1. LOAD DATA
# =====================
df = pd.read_csv("data.csv")

# =====================
# 2. BASIC EDA
# =====================
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Class Distribution ---")
print(df["Bankrupt?"].value_counts())
print(df["Bankrupt?"].value_counts(normalize=True))

# Target distribution plot
plt.figure(figsize=(5,4))
sns.countplot(x="Bankrupt?", data=df, palette="coolwarm")
plt.title("Target Class Distribution")
plt.show()

# Summary stats
print("\n--- Summary Statistics ---")
print(df.describe())

# Top correlated features with target
corr = df.corr()
target_corr = corr["Bankrupt?"].drop("Bankrupt?").sort_values(key=abs, ascending=False)
print("\n--- Top 10 Correlated Features ---")
print(target_corr.head(10))

plt.figure(figsize=(6,5))
target_corr.head(10).plot(kind="bar", color="teal")
plt.title("Top 10 Features Correlated with Bankruptcy")
plt.ylabel("Correlation Coefficient")
plt.show()

# Heatmap for top 10 correlations
plt.figure(figsize=(8,6))
sns.heatmap(df[["Bankrupt?"] + list(target_corr.head(10).index)].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Top 10 Features)")
plt.show()

# =====================
# 3. DATA CLEANING
# =====================
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)  # Drop rows with missing values

# =====================
# 4. FEATURES / TARGET SPLIT
# =====================
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

# =====================
# 5. TRAIN-TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 6. FEATURE SCALING
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# 7. HANDLE CLASS IMBALANCE WITH SMOTE
# =====================
print("\nBefore SMOTE:", y_train.value_counts())
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE:", y_train_res.value_counts())

# =====================
# 8. MODELS & TRAINING
# =====================

# Benchmark: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_res, y_train_res)
y_pred_lr = log_reg.predict(X_test_scaled)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test_scaled)

# XGBoost
xgb = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)  # XGBoost handles imbalance differently
y_pred_xgb = xgb.predict(X_test)

# =====================
# 9. EVALUATION
# =====================
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)

print("\nEnhanced pipeline complete.")
