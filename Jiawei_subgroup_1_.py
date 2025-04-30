#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_train = pd.read_csv('train_data.csv')
df_train.shape


# In[3]:


df_train.info()


# In[4]:


null_sum = df_train.isnull().sum()
print(null_sum.values)


# In[5]:


# Checking for duplicates
df_train.duplicated().sum()


# In[6]:


target = df_train['Bankrupt?']
index = df_train['Index']
df_train = df_train.drop(columns=['Bankrupt?','Index'])


# In[7]:


target.value_counts(normalize=True).plot(kind='bar')
print(target.value_counts())


# In[8]:


# Compute the correlation matrix
corr = df_train.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(20, 20))

# Draw the heatmap with the mask
sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Features (Lower Triangle)", fontsize=16)
plt.show()


# In[9]:


# Identify features with high correlation
high_corr_features = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.95:
            colname = corr.columns[i]
            high_corr_features.add(colname)

# Drop the highly correlated features
df_train_reduced = df_train.drop(columns=high_corr_features)

df_train_reduced.shape


# In[10]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_train_scaled = scaler.fit_transform(df_train_reduced)
df_train_scaled = pd.DataFrame(df_train_scaled , columns=df_train_reduced.columns)
df_train_scaled.head()


# In[11]:


df_train_scaled.describe()


# In[12]:


from sklearn.feature_selection import mutual_info_classif

mi_score = mutual_info_classif(df_train_scaled, target, random_state=42)

mi_df = pd.DataFrame({
    "Feature": df_train_scaled.columns,
    "MI SCORE": mi_score
}).sort_values(by="MI SCORE",ascending=False)

# Plotting the mutual information scores
plt.figure(figsize=(10, 6))
plt.bar(mi_df["Feature"], mi_df["MI SCORE"])
plt.xticks(rotation=90)
plt.title("Mutual Information Scores")
plt.xlabel("Features")
plt.ylabel("MI Score")
plt.show()


# In[13]:


# Display the full mutual information scores DataFrame
pd.set_option('display.max_rows', None)  # Set to display all rows
print(mi_df.sort_values(by="MI SCORE", ascending=False))

# Drop rows where "MI SCORE" is less than 0.0x
mi_01 = mi_df[mi_df["MI SCORE"] >= 0.01]
mi_02 = mi_df[mi_df["MI SCORE"] >= 0.02]
mi_03 = mi_df[mi_df["MI SCORE"] >= 0.03]


# ## >0.02

# In[14]:


df_02 = df_train_scaled.drop(columns=[col for col in df_train_scaled.columns if col not in mi_02["Feature"].values])
df_02.shape


# In[15]:


z_score_mask = (df_02.applymap(abs) <= 3).all(axis=1)
df_02 = df_02[z_score_mask]
target_02 = target[z_score_mask]

print(target_02.shape)
print(df_02.shape)


# In[16]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
kmeans = KMeans(n_clusters=6, random_state=42 , n_init=10)
gmm = GaussianMixture(n_components=5, random_state=42)


# In[17]:


model_km_02 = kmeans.fit_predict(df_02)

# Use df_02 as the selected dataset for PCA
train_selected_df_02 = df_02

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_selected_df_02)

# Add cluster labels to the PCA result
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Cluster"] = model_km_02  # Cluster labels from KMeans

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in sorted(pca_df["Cluster"].unique()):
    subset = pca_df[pca_df["Cluster"] == cluster]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.6)

plt.title("KMeans Clusters Visualized using PCA 20F")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[18]:


#Attach cluster labels and summarize
cleaned_clustered_df_02 = df_train.loc[df_02.index].copy()
cleaned_clustered_df_02["Cluster"] = model_km_02

# Reattachinng the 'Bankrupt' column to the cleaned_clustered_df_02 DataFrame
cleaned_clustered_df_02["Bankrupt?"] = target_02.values

# Summary of clusters
kmeans_summary_02 = cleaned_clustered_df_02.groupby("Cluster")["Bankrupt?"].value_counts().unstack(fill_value=0)
kmeans_summary_02.columns = ["Not Bankrupt (y=0)", "Bankrupt (y=1)"]
kmeans_summary_02["Total"] = kmeans_summary_02.sum(axis=1)
kmeans_summary_02["Bankruptcy Rate (%)"] = (kmeans_summary_02["Bankrupt (y=1)"] / kmeans_summary_02["Total"]) * 100

# Display the KMeans Cluster Summary using pandas
from IPython.display import display
display(kmeans_summary_02)


# In[19]:


model_gm_02 = gmm.fit_predict(train_selected_df_02)

# PCA for visualization
pca_result = pca.transform(train_selected_df_02)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Cluster"] = model_gm_02


plt.figure(figsize=(8, 6))
for cluster in sorted(pca_df["Cluster"].unique()):
    subset = pca_df[pca_df["Cluster"] == cluster]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.6)

plt.title("Gaussian Mixture Clusters (Visualized with PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout() 
plt.show()


# In[20]:


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
#  Load data
df = pd.read_csv("cluster_1.csv")
df.columns = df.columns.str.strip()

#  Select features
features = mi_02["Feature"].str.strip().tolist()
X = df[features]
y = df["Bankrupt?"]

#  Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Define unfitted base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ('svm', SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42))
]

#  Build each base model
for name, model in base_models:
    model.fit(X_scaled, y)
    score = model.score(X_scaled, y)
    print(f'{name} accuracy: {round(score, 3)}')

#  Build the stacking
stack_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=5,
    stack_method='predict_proba'
)

#  Fit the stacking model
stack_clf.fit(X_scaled, y)


# In[21]:


class FeatureSelector1(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        X_sel = X[self.columns]
        self.scaler.fit(X_sel)
        return self
    
    def transform(self, X):
        X_sel = X[self.columns]
        X_scaled = self.scaler.transform(X_sel)
        return pd.DataFrame(X_scaled, columns=self.columns, index=X.index)


# In[22]:


#  Evaluate on the same data
stack_preds = stack_clf.predict(X_scaled)
stack_acc = accuracy_score(y, stack_preds)
stack_cm = confusion_matrix(y, stack_preds)

print(f"Stacking Accuracy: {stack_acc:.4f}")
print(f"Stacking Confusion Matrix:\n{stack_cm}\n")

#  Compute TT, TF and acc for the stacking (meta) model
TN_m, FP_m, FN_m, TP_m = stack_cm.ravel()
meta_acc = TP_m / (TP_m + FN_m) if (TP_m + FN_m) > 0 else 0
print(f"Meta: TT={TP_m}, TF={FN_m}, acc (y=1)={meta_acc:.4f}")

#  Evaluate each base model individually
print("\n--- Base Models Evaluation on Training Data ---\n")

for name, _ in base_models:
    model = stack_clf.named_estimators_[name]
    y_pred_base = model.predict(X_scaled)
    cm_base = confusion_matrix(y, y_pred_base)
    TN, FP, FN, TP = cm_base.ravel()
    acc_base = TP / (TP + FN) if (TP + FN) > 0 else 0
    print(f"{name.upper()} - TT: {TP}, TF: {FN}, Accuracy (acc) for bankrupt companies: {acc_base:.4f}")



# In[24]:


import joblib
columns=X.columns
pipeline = Pipeline([
    ('feature_select', FeatureSelector1(columns)),  # Feature selection step
    ('model', stack_clf)  # Stacking model
])

print(pipeline) 

y = df['Bankrupt?']
X = df.drop(columns=['Bankrupt?','Index'])

pipeline.fit(X, y)

joblib.dump(pipeline, 'subgroup_1_model.pkl')

