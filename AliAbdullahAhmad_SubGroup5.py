#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif
import joblib


# In[2]:


df = pd.read_csv('C:/Users/abdul/OneDrive/Documents/GitHub/CS559_project/cluster_5.csv')
df.head()


# In[3]:


df.shape


# In[4]:


index = df['Index']
target = df['Bankrupt?']
cluster = df['Cluster']
X = df.drop(columns=['Index','Bankrupt?','Cluster'])


# In[5]:


# Compute the correlation matrix
corr = X.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(20, 20))

# Draw the heatmap with the mask
sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Features (Lower Triangle)", fontsize=16)
plt.show()


# In[6]:


high_corr_f = set()

for j in range(len(corr.columns)):
    for i in range(j):
        if abs(corr.iloc[j,i])>0.95:
            colname = corr.columns[j]
            high_corr_f.add(colname)

df_reduced = X.drop(columns=high_corr_f)

df_reduced.shape


# In[7]:


mi_score = mutual_info_classif(df_reduced , target, random_state=42)

mi_df = pd.DataFrame({
    "Feature":df_reduced.columns,
    "Mi Score": mi_score
}).sort_values(by="Mi Score",ascending=False)

# Improved plotting of mutual information scores
plt.figure(figsize=(12, 8))  # Set the figure size
sns.barplot(x="Mi Score", y="Feature", data=mi_df, palette="viridis")  # Create a horizontal bar plot
plt.title("Mutual Information Scores", fontsize=16)  # Set the plot title
plt.xlabel("MI Score", fontsize=14)  # Label for the x-axis
plt.ylabel("Features", fontsize=14)  # Label for the y-axis
plt.tight_layout()  # Adjust layout for better fit
plt.show()  # Display the plot


# In[8]:


# Display the full mutual information scores DataFrame
pd.set_option('display.max_rows', None)  # Set to display all rows
print(mi_df.sort_values(by="Mi Score", ascending=False))

# Drop rows where "Mi Score" is less than 0.01
mi_01 = mi_df[mi_df["Mi Score"] >= 0.001]


# In[9]:


df_01 = df_reduced.drop(columns=[col for col in df_reduced.columns if col not in mi_01["Feature"].values])
df_01.shape


# In[10]:


columns = df_01.columns

print(columns)


# In[11]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np


class FeatureSelector5(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """
        Initialize the feature selector with the column names to be selected.
        
        :param columns: List of column names to select from the DataFrame.
        """
        self.columns = columns
        self.scaler = StandardScaler()  # Create a scaler object
    
    def fit(self, X, y=None):
        """
        Fit the scaler on the selected columns.
        """
        # Select only the columns specified by the user
        X_selected = X[self.columns]
        # Fit the scaler only on the selected columns
        self.scaler.fit(X_selected)
        return self
    
    def transform(self, X):
        """
        Select the specified columns and scale them.
        
        :param X: Input DataFrame.
        :return: Scaled DataFrame with selected columns.
        """
        # Select only the columns specified by the user
        X_selected = X[self.columns]
        # Scale the selected columns
        X_scaled = self.scaler.transform(X_selected)
        
        # Create a DataFrame with the scaled columns and original column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.columns, index=X.index)
        return X_scaled_df


# In[12]:


base_models = [
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),  
    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42))
]
for name, model in base_models:
    model.fit(df_01, target)
    score = model.score(df_01, target)
    print(f'{name} accuracy: {round(score, 3)}')

# Meta model
meta_model = LogisticRegression(class_weight='balanced', random_state=42)

# Define stacking classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
result = stacked_model.fit(df_01,target).score(df_01,target)
print(f'cv: {round(result,3)}') 

y_pred = stacked_model.predict(df_01)
cm = confusion_matrix(target, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(target, y_pred))

# Calculate acc = TT / (TT + TF) and display TT(TF)
for name, model in base_models:
    y_pred_base = model.predict(df_01)
    cm_base = confusion_matrix(target, y_pred_base)
    TT_base = cm_base[1, 1]
    TF_base = cm_base[1, 0]
    acc_base = TT_base / (TT_base + TF_base) if (TT_base + TF_base) > 0 else 0
    print(f"\n{name} - TT: {TT_base}, TF: {TF_base}, Accuracy (acc) for bankrupt companies: {acc_base:.4f}")

# For meta model
y_pred_meta = stacked_model.predict(df_01)
cm_meta = confusion_matrix(target, y_pred_meta)
TT_meta = cm_meta[1, 1]
TF_meta = cm_meta[1, 0]
acc_meta = TT_meta / (TT_meta + TF_meta) if (TT_meta + TF_meta) > 0 else 0
print(f"\nMeta model - TT: {TT_meta}, TF: {TF_meta}, Accuracy (acc) for bankrupt companies: {acc_meta:.4f}")


# In[13]:


import joblib
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('feature_select', FeatureSelector5(columns)),  # Feature selection step
    ('model', stacked_model)  # Stacking model
])

y = df['Bankrupt?']
X = df.drop(columns=['Bankrupt?','Index'])

pipeline.fit(X, y)

joblib.dump(pipeline, 'subgroup_5_model.pkl')

