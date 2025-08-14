"""
Lab 4 — Bankruptcy Risk Assessment: Full Data Pipeline (16 Areas) + EDA
Dataset file expected as: ./data.csv
Target column: "Bankrupt?"

This single script implements code for ALL 16 areas required in the lab:
1) Load data & overview
2) EDA (class imbalance + correlation heatmap)
3) Stratified train/test split
4) Outlier detection & optional treatment (winsorization)
5) Class imbalance handling (class weights + SMOTE)
6) Data normalization strategy
7) Testing for normality (skew) & optional log transform
8) PCA (optional) with explained variance
9) Feature engineering (minimal, domain-safe examples)
10) Multicollinearity checks (correlation + VIF)
11) Feature selection (corr filter, L1-LR, model-based importance)
12) Hyperparameter tuning (RandomizedSearchCV on LR & RF)
13) Cross-validation strategy (StratifiedKFold) with ROC-AUC/PR-AUC/F1
14) Evaluation metrics on test (ROC-AUC, PR-AUC, F1, Brier) + ROC/PR curves
15) Drift monitoring: PSI implementation (train vs test) + top-PSI bar plot
16) Interpretability: LR coefficients, permutation importance; optional SHAP

Notes:
- Uses matplotlib only (no seaborn) for plots.
- XGBoost/CatBoost/SHAP are optional (guarded with try/except); falls back gracefully.
- Heavy steps are capped for speed; increase caps for deeper runs.
"""

# ======================= 0) Imports & Setup =======================
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, precision_recall_curve, roc_curve, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Paths / toggles
DATA_PATH = "./data.csv"        # <-- change path if needed
FIG_DIR = "./figs"
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_COL = "Bankrupt?"

# Toggles (flip to True to enable optional steps)
DO_WINSORIZE = False
DO_LOG_TRANSFORM = False
DO_PCA = False
TRY_XGBOOST = True
TRY_CATBOOST = True
TRY_SHAP = False  # set True if SHAP installed and you want SHAP plots

# Light caps for speed; raise for more thorough runs
RSCV_N_ITER = 8          # hyperparam search iterations
RF_N_EST_CHOICES = [200, 400]
PERM_N_REPEATS = 5
VIF_MAX_COLS = 30        # compute VIF on top-variance subset to avoid timeouts


def savefig(name: str):
    """Helper to save figures to FIG_DIR and close the figure."""
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ======================= 1) Load Data & Basic Overview =======================
df = pd.read_csv(DATA_PATH)
print("[1] Loaded data:", df.shape)
assert TARGET_COL in df.columns, f"Target column {TARGET_COL!r} not found. Columns: {df.columns.tolist()[:5]}..."

print("[1] Head (first 3 rows):")
print(df.head(3).to_string())

missing = df.isna().sum().sort_values(ascending=False)
print("[1] Missing values (top 10):\n", missing.head(10).to_string())


# ======================= 2) EDA: Class Imbalance + Correlation =======================
counts = df[TARGET_COL].value_counts(dropna=False)
ratios = counts / counts.sum()
print("[2] Class counts:\n", counts.to_string())
print("[2] Class ratios:\n", ratios.to_string())

# Class distribution bar
plt.figure()
counts.sort_index().plot(kind="bar")
plt.title("Class Distribution (0=Non-bankrupt, 1=Bankrupt)")
plt.xlabel("Class")
plt.ylabel("Count")
savefig("eda_class_distribution.png")

# Correlation heatmap (matplotlib only)
num_df = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
corr_mat = num_df.corr().values
plt.figure(figsize=(8, 6))
plt.imshow(corr_mat, interpolation="nearest", aspect="auto")
plt.title("Correlation Matrix (numeric features)")
plt.colorbar()
plt.xticks([]); plt.yicks = []  # hide ticks to avoid clutter
savefig("eda_corr_matrix.png")


# ======================= 3) Stratified Train/Test Split =======================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("[3] Train/Test shapes:", X_train.shape, X_test.shape,
      "| positive ratio train:", round(y_train.mean(), 4),
      "test:", round(y_test.mean(), 4))

# Identify numeric columns (dataset is numeric-heavy)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()


# ======================= 4) Outlier Detection & Optional Winsorization =======================
Q1 = X_train[num_cols].quantile(0.25)
Q3 = X_train[num_cols].quantile(0.75)
IQR = (Q3 - Q1).replace(0, np.nan)

outlier_mask = (X_train[num_cols] < (Q1 - 1.5 * IQR)) | (X_train[num_cols] > (Q3 + 1.5 * IQR))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)
print("[4] Outlier counts (top 10):\n", outlier_counts.head(10).to_string())

if DO_WINSORIZE:
    lower = X_train[num_cols].quantile(0.01)
    upper = X_train[num_cols].quantile(0.99)
    X_train[num_cols] = X_train[num_cols].clip(lower=lower, upper=upper, axis=1)
    X_test[num_cols]  = X_test[num_cols].clip(lower=lower,  upper=upper,  axis=1)
print("[4] Winsorization applied:", DO_WINSORIZE)


# ======================= 5) Class Imbalance: Class Weights + SMOTE =======================
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
CLASS_WEIGHT = {int(c): float(w) for c, w in zip(classes, class_weights)}
print("[5] Class weights:", CLASS_WEIGHT)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("[5] SMOTE class counts:", y_train_smote.value_counts().to_dict())


# ======================= 6) Data Normalization Strategy =======================
scaler = StandardScaler()
scale_transformer = ColumnTransformer(
    transformers=[('num', scaler, num_cols)],
    remainder='passthrough'
)
X_train_scaled = scale_transformer.fit_transform(X_train)
X_test_scaled = scale_transformer.transform(X_test)
print("[6] Scaling ready for LR/SVM; tree models will use unscaled features.")


# ======================= 7) Testing for Normality (Skew) & Optional Log Transform =======================
skews = X_train[num_cols].skew().sort_values(ascending=False)
print("[7] Top 10 skewed features:\n", skews.head(10).to_string())

SKEW_THRESHOLD = 2.0
skewed_pos_cols = [c for c in num_cols if (skews.get(c, 0) > SKEW_THRESHOLD) and (X_train[c].min() >= 0)]
if DO_LOG_TRANSFORM and len(skewed_pos_cols) > 0:
    for c in skewed_pos_cols:
        X_train[c] = np.log1p(X_train[c])
        X_test[c]  = np.log1p(X_test[c])
print("[7] Log transform applied:", DO_LOG_TRANSFORM, "| columns:", len(skewed_pos_cols))


# ======================= 8) PCA (Optional) =======================
if DO_PCA:
    pca = PCA(n_components=min(20, X_train.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(StandardScaler().fit_transform(X_train[num_cols]))
    X_test_pca  = pca.transform(StandardScaler().fit_transform(X_test[num_cols]))
    evr = pca.explained_variance_ratio_

    print("[8] PCA explained variance (first 10):", np.round(evr[:10], 4).tolist())
    plt.figure()
    plt.plot(np.cumsum(evr))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    savefig("pca_explained_variance.png")
else:
    X_train_pca = None
    X_test_pca = None
print("[8] PCA enabled:", DO_PCA)


# ======================= 9) Feature Engineering (minimal, domain-safe) =======================
X_train_eng = X_train.copy()
X_test_eng  = X_test.copy()

# Example engineered features:
X_train_eng["num_positive_feats"] = (X_train_eng[num_cols] > 0).sum(axis=1)
X_test_eng["num_positive_feats"]  = (X_test_eng[num_cols]  > 0).sum(axis=1)

train_mean = X_train[num_cols].mean()
train_std  = X_train[num_cols].std().replace(0, np.nan)
z_scores   = (X_train[num_cols] - train_mean) / train_std
X_train_eng["num_extreme_z"] = (z_scores.abs() > 3).sum(axis=1)

z_scores_test = (X_test[num_cols] - train_mean) / train_std
X_test_eng["num_extreme_z"] = (z_scores_test.abs() > 3).sum(axis=1)

print("[9] Feature engineering added: ['num_positive_feats','num_extreme_z']")


# ======================= 10) Multicollinearity: Correlation & VIF =======================
corr_abs = X_train[num_cols].corr().abs()
high_corr_pairs = [(i, j, corr_abs.loc[i, j])
                   for i in corr_abs.columns for j in corr_abs.columns
                   if i < j and corr_abs.loc[i, j] > 0.9]
print("[10] Highly correlated pairs (>0.9):", len(high_corr_pairs))

# VIF on top-variance subset (to avoid singularities/compute blowup)
vif_df = None
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    var_series = X_train[num_cols].var().sort_values(ascending=False)
    vif_cols = var_series.head(min(VIF_MAX_COLS, len(var_series))).index.tolist()
    X_for_vif = X_train[vif_cols].fillna(X_train[vif_cols].median())
    X_for_vif = X_for_vif + np.random.normal(0, 1e-9, X_for_vif.shape)
    vifs = [(col, float(variance_inflation_factor(X_for_vif.values, i)))
            for i, col in enumerate(X_for_vif.columns)]
    vif_df = pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
    print("[10] VIF (top 10):\n", vif_df.head(10).to_string(index=False))
except Exception as e:
    print("[10] VIF skipped:", e)


# ======================= 11) Feature Selection: Corr Filter, L1-LR, Model-Based =======================
# (a) Correlation filter: drop one from any pair > 0.95
to_drop_corr = set()
thr = 0.95
for i in corr_abs.columns:
    for j in corr_abs.columns:
        if i < j and corr_abs.loc[i, j] > thr and j not in to_drop_corr:
            to_drop_corr.add(j)
print("[11] Correlation-filter drop count:", len(to_drop_corr))

# (b) L1-regularized Logistic Regression selection
pipe_l1 = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=2000,
                               class_weight="balanced", random_state=42))
])
pipe_l1.fit(X_train[num_cols], y_train)
coef = pipe_l1.named_steps["clf"].coef_.ravel()
selected_l1 = [col for col, c in zip(num_cols, coef) if abs(c) > 1e-6]
print("[11] L1-LR selected features:", len(selected_l1))

# (c) Model-based importance (XGBoost preferred; fallback RF; optionally CatBoost)
feature_importances = None
selected_model_features = None

used_model_for_importance = None
if TRY_XGBOOST:
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
        )
        xgb.fit(X_train, y_train)
        feature_importances = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        selected_model_features = feature_importances.head(30).index.tolist()
        used_model_for_importance = "XGBoost"
    except Exception as e:
        print("[11] XGBoost not available:", e)

if feature_importances is None and TRY_CATBOOST:
    try:
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier(
            depth=6, learning_rate=0.1, iterations=300, verbose=False,
            loss_function="Logloss", random_state=42,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
        )
        cat.fit(X_train, y_train)
        feature_importances = pd.Series(cat.get_feature_importance(), index=X_train.columns).sort_values(ascending=False)
        selected_model_features = feature_importances.head(30).index.tolist()
        used_model_for_importance = "CatBoost"
    except Exception as e:
        print("[11] CatBoost not available:", e)

if feature_importances is None:
    rf_tmp = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_tmp.fit(X_train, y_train)
    feature_importances = pd.Series(rf_tmp.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    selected_model_features = feature_importances.head(30).index.tolist()
    used_model_for_importance = "RandomForest (fallback)"

print(f"[11] Model-based importance using: {used_model_for_importance}")
print("[11] Top 5 important features:\n", feature_importances.head(5).to_string())


# ======================= 12) Hyperparameter Tuning (RandomizedSearchCV) =======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression search (penalty + C)
pipe_lr = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=42))
])
param_lr = {
    "clf__C": np.logspace(-3, 2, 20),
    "clf__penalty": ["l1", "l2"]
}
rs_lr = RandomizedSearchCV(pipe_lr, param_distributions=param_lr, n_iter=RSCV_N_ITER,
                           cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42)
rs_lr.fit(X_train, y_train)
print("[12] Best LR params:", rs_lr.best_params_, "ROC-AUC (CV):", round(rs_lr.best_score_, 4))

# RandomForest search
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
param_rf = {
    "n_estimators": RF_N_EST_CHOICES,
    "max_depth": [None, 4, 6, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}
rs_rf = RandomizedSearchCV(rf, param_distributions=param_rf, n_iter=RSCV_N_ITER,
                           cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42)
rs_rf.fit(X_train, y_train)
print("[12] Best RF params:", rs_rf.best_params_, "ROC-AUC (CV):", round(rs_rf.best_score_, 4))


# ======================= 13) Cross-Validation Strategy (scores) =======================
auc_scores = cross_val_score(rs_rf.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
pr_scores  = cross_val_score(rs_rf.best_estimator_, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
f1_scores  = cross_val_score(rs_rf.best_estimator_, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

print("[13] CV ROC-AUC:", round(auc_scores.mean(), 4), "+/-", round(auc_scores.std(), 4))
print("[13] CV PR-AUC:",  round(pr_scores.mean(), 4),  "+/-", round(pr_scores.std(), 4))
print("[13] CV F1:",      round(f1_scores.mean(), 4),  "+/-", round(f1_scores.std(), 4))


# ======================= 14) Evaluation Metrics on Test + Curves =======================
best_model = rs_rf.best_estimator_
best_model.fit(X_train, y_train)

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)
f1      = f1_score(y_test, y_pred)
brier   = brier_score_loss(y_test, y_proba)

print("[14] Test ROC-AUC:", round(roc_auc, 4))
print("[14] Test PR-AUC:",  round(pr_auc, 4))
print("[14] Test F1:",      round(f1, 4))
print("[14] Test Brier:",   round(brier, 4))
print("[14] Classification report:\n", classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
savefig("eval_roc_curve.png")

# PR curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
savefig("eval_pr_curve.png")

# Save metrics to JSON for report
with open("./test_metrics.json", "w") as f:
    json.dump({
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "brier": float(brier)
    }, f, indent=2)


# ======================= 15) Drift Monitoring: PSI (Train vs Test) =======================
def psi(expected, actual, buckets=10):
    """
    Population Stability Index between two numeric arrays.
    Returns a single PSI value (higher => more shift).
    """
    expected = pd.Series(expected).dropna().values
    actual   = pd.Series(actual).dropna().values
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    if np.all(expected == expected[0]) or np.all(actual == actual[0]):
        return 0.0

    quantiles = np.linspace(0, 100, buckets + 1)
    bins = np.unique(np.percentile(expected, quantiles))  # use expected to set bins
    if len(bins) <= 2:
        return 0.0

    e_hist, _ = np.histogram(expected, bins=bins)
    a_hist, _ = np.histogram(actual,   bins=bins)

    e_perc = e_hist / max(e_hist.sum(), 1)
    a_perc = a_hist / max(a_hist.sum(), 1)

    e_perc = np.clip(e_perc, 1e-6, None)
    a_perc = np.clip(a_perc, 1e-6, None)

    return float(np.sum((e_perc - a_perc) * np.log(e_perc / a_perc)))

psi_scores = {col: psi(X_train[col], X_test[col]) for col in X.columns}
psi_df = pd.DataFrame({"Feature": list(psi_scores.keys()), "PSI": list(psi_scores.values())}).sort_values("PSI", ascending=False)
print("[15] PSI (top 15):\n", psi_df.head(15).to_string(index=False))

psi_df.to_csv("./psi_scores.csv", index=False)

# Plot top PSI features
top_n = min(15, psi_df.shape[0])
plt.figure(figsize=(8, 5))
plt.barh(psi_df["Feature"].head(top_n)[::-1], psi_df["PSI"].head(top_n)[::-1])
plt.xlabel("PSI")
plt.title("Top PSI Features (Train vs Test)")
savefig("psi_top_features.png")


# ======================= 16) Interpretability: LR Coefs, Permutation Importance, SHAP (optional) =======================
# Logistic Regression (interpretable baseline)
lr_interp = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42))
])
lr_interp.fit(X_train, y_train)
coef_series = pd.Series(lr_interp.named_steps["clf"].coef_.ravel(), index=X_train.columns).sort_values()

k = min(10, max(2, coef_series.shape[0] // 10))  # show up to 10 each side
plt.figure(figsize=(8, 6))
coef_series.tail(k).plot(kind="barh")
plt.title("LR: Most Positive Coefficients")
savefig("interpret_lr_pos_coeffs.png")

plt.figure(figsize=(8, 6))
coef_series.head(k).plot(kind="barh")
plt.title("LR: Most Negative Coefficients")
savefig("interpret_lr_neg_coeffs.png")

# Permutation importance for the tuned RF model
perm = permutation_importance(best_model, X_test, y_test, n_repeats=PERM_N_REPEATS, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
perm_imp.head(15).plot(kind="barh")
plt.title("Permutation Importance (Top 15)")
savefig("interpret_permutation_importance.png")

# Optional: SHAP for tree models (if installed and TRY_SHAP=True)
if TRY_SHAP:
    try:
        import shap
        explainer = None
        # Try TreeExplainer for RF-like models
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_vals = explainer.shap_values(X_test)
            # Summary plot (if multiclass/array, handle first)
            plt.figure()
            shap.summary_plot(shap_vals, X_test, show=False)
            savefig("shap_summary.png")
        except Exception as e:
            print("[16] SHAP TreeExplainer failed:", e)
            # Fallback to KernelExplainer (slow) — we’ll use a small sample
            small_X = X_test.sample(min(200, X_test.shape[0]), random_state=42)
            explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train, 100))
            shap_vals = explainer.shap_values(small_X, nsamples=100)
            plt.figure()
            shap.summary_plot(shap_vals[1], small_X, show=False)  # class 1
            savefig("shap_summary_kernel.png")
        print("[16] SHAP plots generated.")
    except Exception as e:
        print("[16] SHAP not available:", e)

print("\n[DONE] All 16 areas executed.")
print(f"[FILES] Figures saved under: {FIG_DIR}")
print("[FILES] PSI table: ./psi_scores.csv")
print("[FILES] Test metrics: ./test_metrics.json")
