#!/usr/bin/env python
# coding: utf-8

# ### Random Forest Regression for Binding Affinity Prediction with Scikit-Learn/DeepChem/RDKit

# In[1]:


import pandas as pd
import numpy as np


# #### Load ChEMBL bioactivity data

# In[2]:


from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import PandasTools

chembl_bioactivity_df = pd.read_pickle('../data/chembl_bioactivity_data.pkl')
chembl_bioactivity_df.iloc[:,:-7].head(2)


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
# #### 1. Use Molecular Descriptors

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


# #### 2. Use Fingerprints

# In[7]:


fp_featurizer = dc.feat.CircularFingerprint(size=2048)

features_fp = fp_featurizer.featurize(mols)
#features_fp is a N x 2048 array containing the fingerprints for the 1502 molecules
print(features_fp.shape)
features_fp[:5]


# #### 3. Use Graph Convolutions

# In[8]:


gc_featurizer = dc.feat.ConvMolFeaturizer()
features_graphs = gc_featurizer.featurize(mols)
features_graphs


# #### Dataset preparation

# In[9]:


features = features_fp
labels = chembl_bioactivity_ml_df['mean_pchembl_value']
ids = chembl_bioactivity_ml_df['molecule_chembl_id']

dataset = dc.data.NumpyDataset(X=features, y=labels, ids=ids)

train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(dataset, seed=42)


# In[10]:


train_dataset.get_shape()


# In[11]:


test_dataset.get_shape()


# #### RandomForestRegressor model

# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


# In[13]:


seed = 42
rf_model = RandomForestRegressor()
rf_model.random_state = seed

param_grid = {'oob_score': [True], 'n_estimators':[50, 100, 150, 200, 250]}

grid_search = GridSearchCV(rf_model, param_grid, cv=10, verbose = 1, refit = True, 
                           return_train_score=True, n_jobs = -2)

grid_search.fit(train_dataset.X, train_dataset.y)


# In[14]:


grid_search.best_params_


# In[15]:


grid_search.best_estimator_


# In[16]:


grid_search.cv_results_


# In[17]:


grid_search.best_score_


# In[18]:


y_pred_train = grid_search.predict(train_dataset.X)
y_pred_test = grid_search.predict(test_dataset.X)


# In[19]:


R2_cv_train = r2_score(train_dataset.y, y_pred_train)
R2_cv_test = r2_score(test_dataset.y, y_pred_test)

print("RF Train set R2 %f" % R2_cv_train)
print("RF Test set R2 %f" % R2_cv_test)


# In[20]:


import deepchem as dc
print("DeepChem: ", dc.__version__)

#deepchem is enabled by/running on TensorFlow GPU platform
import tensorflow as tf
print("TensorFlow: ", tf.__version__)
print("GPUs available: ", tf.config.list_physical_devices('GPU'))

import sklearn
print("Scikit-Learn: ", sklearn.__version__)

import rdkit
print("RDKit: ", rdkit.__version__)

from platform import python_version
print("Python: ", python_version())
print("Numpy: ", np.__version__)
print("Pandas: ", pd.__version__)


# In[21]:


get_ipython().system(' conda list')

