#!/usr/bin/env python
# coding: utf-8

# ### Extreme Gradient Boosting (XGBoost) Classification for Binding Affinity Prediction

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# #### Load ChEMBL bioactivity data

# In[2]:


from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import PandasTools

chembl_bioactivity_df = pd.read_pickle('../data/chembl_bioactivity_data.pkl')


# In[3]:


chembl_bioactivity_df.shape


# In[4]:


from rdkit.Chem.SaltRemover import SaltRemover

def unsalt(smiles):
    remover = SaltRemover()
    #print(remover.salts)
    mol = Chem.MolFromSmiles(smiles)
    mol, deleted = remover.StripMolWithDeleted(mol)
    #print([Chem.MolToSmarts(s) for s in deleted])
    return Chem.MolToSmiles(mol, True)

chembl_bioactivity_ml_df = chembl_bioactivity_df[['molecule_chembl_id', 'canonical_smiles', 'mean_pchembl_value']].copy()

#remove salts
smiles = list(map(lambda i: unsalt(i), list(chembl_bioactivity_ml_df['canonical_smiles'])))

chembl_bioactivity_ml_df['smiles']  = smiles
#chembl_bioactivity_ml_df.head()

mols = [Chem.MolFromSmiles(smi) for smi in chembl_bioactivity_ml_df['smiles']]  #sanitize=True default


# #### Featurize the ChEMBL dataset
# 
# #### Calculate Molecular Descriptors

# In[5]:


import deepchem as dc

#if use_fragment = True, a total of 208 descriptors are returned to include fragment binary descriptors like 'fr_'
md_featurizer = dc.feat.RDKitDescriptors(use_fragment = False)

features_md = md_featurizer.featurize(mols)
#features_md is a N x 123 array containing the 123 molecular descriptors(physiochemical properties) for the 1502 molecules
print(features_md.shape)
features_md[:5]


# In[6]:


#from rdkit.ML.Descriptors import MoleculeDescriptors

#calculated rdkit descriptors
descriptors = []
descList = []
from rdkit.Chem import Descriptors
for descriptor, function in Descriptors.descList:
    if descriptor.startswith('fr_'):
        continue
    descriptors.append(descriptor)
    descList.append((descriptor, function))
print(descriptors)
print(len(descriptors))


# #### Dataset preparation

# In[7]:


dataset = pd.DataFrame(data=features_md, columns=descriptors)
dataset['Activity'] = chembl_bioactivity_ml_df['mean_pchembl_value'].astype(float)


# In[8]:


#### Summary of the dataset
dataset.info()


# In[9]:


#### Summary statistics of the dataset
dataset.describe()


# In[10]:


#### Correlation matrix
corr=dataset.corr()
mask=np.triu(np.ones_like(corr,dtype=bool))
f,ax = plt.subplots(figsize=(15,12))
cmap = sns.diverging_palette(230,20, as_cmap=True)
sns.heatmap(corr,cmap=cmap,mask=mask,linewidth=0.5,square=True,center=0)


# In[11]:


#### Distributions of descriptors
#color = '#a9cce3'
#dataset.hist(bins=15,figsize=(25,15),color=color)
#plt.rcParams['font.size'] = 18
#plt.show()


# In[12]:


#### Make Activity categorical
dataset['Activity'] = np.where(dataset['Activity'] >=7.0, 1, 0)
dataset.head()


# #### Create feature matrix (X) and target vector (y)

# In[13]:


X = dataset.drop('Activity', axis=1)
y = dataset['Activity']


# In[14]:


X.head()


# In[15]:


y.head()


# #### Split dataset in train and test datasets

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)


# In[17]:


X_train.shape


# In[18]:


X_test.shape


# #### XGBoost Classifier model

# In[21]:


get_ipython().system(' pip install xgboost')


# In[22]:


from xgboost import XGBClassifier

params = {'objective':'binary:logistic',
          'max_depth':5, 
          'alpha':10, 
          'learning_rate':0.1, 
          'n_estimators':250,
          'random_state':42}

xgb_clf = XGBClassifier(**params)

xgb_clf.fit(X_train, y_train)


# In[23]:


y_pred = xgb_clf.predict(X_test)
y_pred


# In[24]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[25]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[1,0])
print(cm)


# In[26]:


from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)
print(cr)


# #### Feature Importance

# In[27]:


from xgboost import plot_importance
plot_importance(xgb_clf, height=0.4, max_num_features=14)


# #### XGBoost k-fold cross validation model

# In[28]:


#### convert the dataset into the Dmatrix optimized data structure supported by XGBoost 
#### that gives its performance and efficiency gains.

from xgboost import DMatrix

data_dmatrix = DMatrix(data=X,label=y)


# In[29]:


from xgboost import cv

params = {'objective':'binary:logistic',
          'max_depth':5, 
          'alpha':10, 
          'learning_rate':0.1, 
          'random_state':42, 
          'colsample_bytree':0.3}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=10,
            num_boost_round=50, early_stopping_rounds=10, metrics="auc", 
            as_pandas=True, seed=42)


# In[30]:


xgb_cv.head()


# In[31]:


xgb_cv.describe()


# #### XGBoost hyperparameters tuning with HYPEROPT Bayesian optimization

# In[32]:


get_ipython().system(' pip install hyperopt')


# In[33]:


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# In[34]:


space={'max_depth': hp.quniform("max_depth", 4, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 10,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.3, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'eval_metric':"auc",
        'early_stopping_rounds': hp.uniform('early_stopping_rounds', 2, 20),
        'n_estimators': hp.uniform('n_estimators', 50, 500),
        'seed': 42}


# In[35]:


def objective(space):
    clf=XGBClassifier(
                    n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    early_stopping_rounds= int(space['early_stopping_rounds']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [ ( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation,
            verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


# In[36]:


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 200,
                        trials = trials)


# In[37]:


best_hyperparams


# In[38]:


import sklearn
print("Scikit-Learn: ", sklearn.__version__)

import rdkit
print("RDKit: ", rdkit.__version__)

from platform import python_version
print("Python: ", python_version())
print("Numpy: ", np.__version__)
print("Pandas: ", pd.__version__)


# In[ ]:




