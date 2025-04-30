# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin

import joblib

# Data for cls 4
df = pd.read_csv("cluster_4.csv")
df.head()

X = df.drop(columns=["Index", "Cluster", "Bankrupt?"])
y = df["Bankrupt?"]


#  Original Class Balance
plt.figure(figsize=(6,6))
y.value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'], labels=["Not Bankrupt", "Bankrupt"])
plt.title("Original Class Distribution (Subgroup 4)")
plt.ylabel("")
plt.show()

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Custom FeatureSelector class for consistent feature selection
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.features]

# Remove High Correlation Features (>0.95)
corr = X_scaled_df.corr()
high_corr_features = set()
for j in range(len(corr.columns)):
    for i in range(j):
        if abs(corr.iloc[j, i]) > 0.95:
            high_corr_features.add(corr.columns[j])
X_reduced = X_scaled_df.drop(columns=high_corr_features)

# Correlation Matrix
plt.figure(figsize=(18,15))
sns.heatmap(X_reduced.corr(), cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix After Removing High Correlation (>0.95)")
plt.show()

# Mutual Information
mi_scores = mutual_info_classif(X_reduced, y, random_state=42)
mi_df = pd.DataFrame({"Feature": X_reduced.columns, "MI": mi_scores})
selected_features = mi_df[mi_df["MI"] > 0.01].head(12)["Feature"].tolist()

# Final Selected Features
X_selected = X_reduced[selected_features]
print(f"Selected {len(selected_features)} final features.")

# MI Scores
plt.figure(figsize=(10,8))
sns.barplot(x="MI", y="Feature", data=mi_df.sort_values(by="MI", ascending=False), palette="viridis")
plt.title("Mutual Information Scores for Features")
plt.tight_layout()
plt.show()

# Base Models
base_models = [
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42))
]

# Meta Model (Logistic Regression)
meta_model = LogisticRegression(class_weight='balanced', random_state=42)

# Stacking Classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train Stacking Classifier
stacked_model.fit(X_selected, y)


# Eval Base Models
print("\n--- Base Models Evaluation ---\n")
for name, model in base_models:
    model.fit(X_selected, y)
    y_pred_base = model.predict(X_selected)
    cm_base = confusion_matrix(y, y_pred_base)
    TT_base = cm_base[1, 1]
    TF_base = cm_base[1, 0]
    acc_base = TT_base / (TT_base + TF_base) if (TT_base + TF_base) > 0 else 0
    print(f"{name.upper()} - TT: {TT_base}, TF: {TF_base}, Accuracy (acc): {acc_base:.4f}")



# Eval Meta Model
print("\n--- Meta Model Evaluation ---\n")
y_pred_meta = stacked_model.predict(X_selected)
cm_meta = confusion_matrix(y, y_pred_meta)
TT_meta = cm_meta[1, 1]
TF_meta = cm_meta[1, 0]
acc_meta = TT_meta / (TT_meta + TF_meta) if (TT_meta + TF_meta) > 0 else 0
print(classification_report(y, y_pred_meta))
print(f"\nMeta Model - TT: {TT_meta}, TF: {TF_meta}, Accuracy (acc): {acc_meta:.4f}")

# Final Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_meta, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Final Meta Model Prediction (Subgroup 4)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

from sklearn.pipeline import Pipeline


# Pipeline
pipeline = Pipeline([
    ('selector', FeatureSelector(selected_features)),
    ('scaler', StandardScaler()),
    ('model', StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5))
])
pipeline.fit(X, y)
joblib.dump(pipeline, "subgroup_4_model.pkl")
print("\n Pipeline saved as 'subgroup_4_model.pkl'")


# import pickle

# with open('subgroup_4_model.pkl', 'wb') as f:
    # pickle.dump(stacked_model, f)
