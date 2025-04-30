#!/usr/bin/env python
# coding: utf-8

# In[170]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.pipeline import Pipeline

df = pd.read_csv('cluster_7.csv')
y = df['Bankrupt?']
X = df.drop(columns=['Bankrupt?','Index'])


# In[171]:


corr = X.corr()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.75:
            column_to_drop = corr.columns[i]
            if column_to_drop in X.columns:
                X = X.drop(columns=column_to_drop)


# In[172]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df = X_scaled_df.loc[:, X_scaled_df.std() != 0]
X = X_scaled_df


# In[173]:


from sklearn.feature_selection import mutual_info_classif


mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi_scores})
selected_features = mi_df[mi_df['MI'] > .02]['Feature'].tolist()
X = X[selected_features]

columns = X.columns

print(columns)


# In[174]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np


class FeatureSelector7(BaseEstimator, TransformerMixin):
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


# In[175]:


from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

base_models = [
    #('knn',KNeighborsClassifier(weights='uniform',n_neighbors=1,metric='manhattan')),
    ('rf', RandomForestClassifier(class_weight='balanced',max_depth=17,min_samples_split=3,min_samples_leaf=2)),
    #('svc', SVC(class_weight='balanced',C=1,gamma=1)),  
    ('dt', DecisionTreeClassifier(class_weight='balanced',max_depth=17,min_samples_split=2,min_samples_leaf=1,criterion='entropy')),
    #('gpc', GaussianProcessClassifier()),
    ('hgb', HistGradientBoostingClassifier(class_weight='balanced',max_depth=6,learning_rate=.1,l2_regularization=4,min_samples_leaf=10, max_leaf_nodes=None,early_stopping=False))

]
for name, model in base_models:
    model.fit(X, y)
    score = model.score(X, y)
    print(f'{name} accuracy: {round(score, 3)}')

meta_model = LogisticRegression(class_weight='balanced', solver='liblinear',C=.3,penalty='l2')
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
result = stacked_model.fit(X,y).score(X,y)
print(f'cv: {round(result,3)}') 

y_pred = stacked_model.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n------------------------------------\n", conf_matrix)
print("\nClassification Report:\n------------------------------------\n", classification_report(y, y_pred))

for name, model in base_models:
    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    TT = conf_matrix[1, 1] 
    TF = conf_matrix[1, 0] 

    if (TT+TF) != 0:
        ratio = TT / (TT+TF)
    else:
        ratio = 1
    
    print(f"{name}:\nTT: {TT}\nTF: {TF}\nacc: {ratio}\n")

y_pred_meta = stacked_model.predict(X)
conf_matrix = confusion_matrix(y, y_pred_meta)
TT = conf_matrix[1, 1]
TF = conf_matrix[1, 0]
ratio = TT / (TT+ TF)
print(f"\nMeta model - TT:\nTT: {TT}\nTF: {TF}\nacc: {ratio}\n")
print(f"n_features: {len(X.columns)}")


# In[ ]:





# In[176]:


import joblib
pipeline = Pipeline([
    ('feature_select', FeatureSelector7(columns)),  # Feature selection step
    ('model', stacked_model)  # Stacking model
])

y = df['Bankrupt?']
X = df.drop(columns=['Bankrupt?','Index'])

pipeline.fit(X, y)

joblib.dump(pipeline, 'subgroup_7_model.pkl')


