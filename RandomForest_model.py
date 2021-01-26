import pandas as pd, os, gc
import copy
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List

train_path = '/insurance_recommend/DATA/train.csv'
test_path = '/insurance_recommend/DATA/Test.csv'
submission_path = 'E:\\pycharmproject\\insurance_recommend\\SampleSubmission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)
# print(train.shape, test.shape, submission.shape)

X_train = []
X_train_columns = train.columns
c = 0
for v in train.values:
  info = v[:8]
  binary = v[8:]
  index = [k for k, i in enumerate(binary) if i == 1]
  for i in index:
    c+=1
    for k in range(len(binary)):
      if k == i:
        binary_transformed = list(copy.copy(binary))
        binary_transformed[i] = 0
        X_train.append(list(info) + binary_transformed + [X_train_columns[8+k]] + [c])

# format
X_train = pd.DataFrame(X_train)
X_train.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
       'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'product_pred', 'ID2']

X_test = []
true_values = []
c = 0
for v in test.values:
  c += 1
  info = v[:8]
  binary = v[8:]
  index = [k for k, i in enumerate(binary) if i == 1]
  X_test.append(list(info) + list(binary) + [c])
  for k in test.columns[8:][index]:
    true_values.append(v[0] + ' X ' + k)

X_test = pd.DataFrame(X_test)
X_test.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
       'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'ID2']

# transform data
features_train = []
features_test = []
columns = []

append_features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9',
'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX',
'ECY3', 'ID', 'ID2', 'join_date', 'sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code',
'birth_year']
for v in append_features:
  features_train.append(X_train[v].values.reshape(-1, 1))
  features_test.append(X_test[v].values.reshape(-1, 1))
  columns.append(np.array([v]))

y_train = X_train[['product_pred']]
# print('--------------\n', y_train)

features_train = np.concatenate(features_train, axis=1)
features_test = np.concatenate(features_test, axis=1)
columns = np.concatenate(np.array(columns))

X_train = pd.DataFrame(features_train)
X_train.columns = columns
X_test = pd.DataFrame(features_test)
X_test.columns = columns

# 新特征
X_train['month'] = X_train['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)
X_train['day'] = X_train['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
X_train['year'] = X_train['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
X_train.drop('join_date', axis=1, inplace=True)

X_test['month'] = X_test['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)
X_test['day'] = X_test['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
X_test['year'] = X_test['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
X_test.drop('join_date', axis=1, inplace=True)

X_train['age'] = X_train['year'] - X_train['birth_year']
X_test['age'] = X_test['year'] - X_test['birth_year']

# 缺失值处理
a = X_train.isnull().any()
b = X_test.isnull().any()
# print(a,b)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.fillna(0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data = X_train.append(X_test)
for v in ['sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code']:
  data.loc[:,v] = le.fit_transform(data.loc[:,v])
X_train = data[:X_train.shape[0]]
X_test = data[-X_test.shape[0]:]

le.fit(y_train.iloc[:,0])
y_train = pd.DataFrame(le.transform(y_train.iloc[:,0]))
y_train.columns = ['target']


# 模型
model = RandomForestClassifier()

# print('y_train:', y_train)
# print('X_train:', X_train)
# print('X_test:', X_test)

train_X = X_train.drop(columns=['ID', 'ID2'])
train_y = y_train
model.fit(train_X, train_y)

proba = model.predict_proba(X_test.drop(columns=['ID','ID2'], axis=1))
y_test = pd.DataFrame(proba)
y_test.columns = le.inverse_transform(y_test.columns)

mass = []
for i in range(X_test.shape[0]):
    id = X_test['ID'].iloc[i]
    for c in y_test.columns:
        mass.append([id + ' X ' + c, y_test[c].iloc[i]])

df_answer = pd.DataFrame(mass)
df_answer.columns = ['ID X PCODE', 'Label']
for i in range(df_answer.shape[0]):
    if df_answer['ID X PCODE'].iloc[i] in true_values:
        df_answer['Label'].iloc[i] = 1.0

df_answer.reset_index(drop=True, inplace=True)
df_answer.to_csv('submission11.csv', index=False)

# y_pred = y_test
# y_true = submission['Label']
# loss =  log_loss(y_true, y_pred)
# print('loss:', loss)