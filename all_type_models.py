import pandas as pd, gc
import copy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
import warnings
import numpy as np
from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


train_path = '/insurance_recommend/DATA/train.csv'
test_path = '/insurance_recommend/DATA/Test.csv'
submission_path = 'E:\\pycharmproject\\insurance_recommend\\SampleSubmission.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)
# print(train_data.shape, test_data.shape, submission.shape)

# 21个产品
pro_train = train_data[['P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']]
pro_test = test_data[['P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']]

#
# print(pro_train.shape)
# print(pro_test.shape)

train = train_data.melt(id_vars=train_data.columns[:8], value_vars=pro_train, var_name = "PCODE", value_name="Label" )
test = test_data.melt(id_vars=test_data.columns[:8], value_vars=pro_test, var_name = "PCODE", value_name="Label" )

train['combiner']='x'

data=pd.concat([train,test],sort=False).reset_index(drop=True)

train=data[data.combiner.notnull()].reset_index(drop=True)
test=data[data.combiner.isna()].reset_index(drop=True)
train.drop('combiner', inplace=True, axis=1)
test.drop(['Label','combiner'], inplace=True, axis=1)
# print(train.shape, test.shape, submission.shape)

train['ID X PCODE'] = train['ID'] + ' X ' + train['PCODE']
test['ID X PCODE'] = test['ID'] + ' X ' + test['PCODE']

# 类别型转换为数值型
def encode_LE(train,test,cols,verbose=True):
    for col in cols:
        df_comb = pd.concat([train[col],test[col]],axis=0)
        df_comb,_ = df_comb.factorize(sort=True)
        nm = col
        if df_comb.max()>32000:
            train[nm] = df_comb[:len(train)].astype('int32')
            test[nm] = df_comb[len(train):].astype('int32')
        else:
            train[nm] = df_comb[:len(train)].astype('int16')
            test[nm] = df_comb[len(train):].astype('int16')
        del df_comb; x=gc.collect()
        if verbose: print(nm,', ',end='')

encode_LE(train, test, ['ID','branch_code', 'occupation_code','occupation_category_code','PCODE','sex','marital_status'])

# print('---------train_shape-------------:', train.shape)
# print('---------train_head()------------:', train.head())
# print('---------test_shape--------------:', test.shape)
# print('---------test_head()-------------:', test.head())

# 相关系数
# 其中，birth_year,marital_status,occupation_code(客户端操作代码，这里我理解为购买保险前填写的问卷信息内容所获得的风险等级)
# print(train.corr())



train['month'] = train['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)
train['day'] = train['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
train['year'] = train['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
train.drop('join_date', axis=1, inplace=True)

test['month'] = test['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)
test['day'] = test['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
test['year'] = test['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
test.drop('join_date', axis=1, inplace=True)

train['age'] = train['year'] - train['birth_year'] +1
test['age'] = test['year'] - test['birth_year'] + 1
# print('-----------------------------train---------------------------:\n',train)

# true_values = []
# for v in test.values:
#     binary = v[8:]
#     index = [k for k, i in enumerate(binary) if i == 1]
#     for k in test.columns[8:][index]:
#         true_values.append(v[0] + 'X' + k)


# train
test = test.fillna(0)
X = train.fillna(0)
y = X['Label']
X = X.drop(columns=['ID', 'Label', 'ID X PCODE'])

submission = pd.read_csv('SampleSubmission.csv')
y_true = submission['Label']

# 查看缺失值
# print(X.isnull().any())
# print(y.isnull().any())
# print(test.isnull().any())
# 查看数据集
# print('train_shape:', train.shape)
# print('test_shape:', test.shape)
# print('submission_shape:', submission.shape)

# KNN
# clf1 = KNeighborsClassifier()
# clf1.fit(X, y)
# prob1 = clf1.predict_proba(test.drop(columns=['ID','ID X PCODE']))
# print('prob1:', prob1)
# a1 = prob1[:,1]
# print(a1)
# b1 = log_loss(y_true, a1)
# print('log_loss:',b1)

# GBDT
# clf2 = GradientBoostingClassifier()
# clf2.fit(X, y)
# prob2 = clf2.predict_proba(test.drop(columns=['ID','ID X PCODE']))
# a2 = prob2[:,1]
# b2 = log_loss(y_true,a2)
# print('log_loss_gbdt:',b2)

# RandomForest
# clf3 = RandomForestClassifier()
# clf3.fit(X,y)
# prob3 = clf3.predict_proba(test.drop(columns=['ID', 'ID X PCODE']))
# a3 = prob3[:,1]
# b3 = log_loss(y_true, a3)
# print('log_loss_RandomForest:',b3)

# AdaBoost
# clf4 = AdaBoostClassifier()
# clf4.fit(X,y)
# prob4 = clf4.predict_proba(test.drop(columns=['ID', 'ID X PCODE']))
# a4 = prob4[:,1]
# b4 = log_loss(y_true, a4)
# print('log_loss_adabt:',b4)

# SVC
# 这里要把svc里面的probability改为true
# clf5 = SVC()
# clf5.fit(X, y)
# prob5 = clf5.predict_proba(test.drop(columns=['ID','ID X PCODE']))
# a5 = prob5[:,1]
# b5 = log_loss(y_true, a5)
# print('log_loss_svm:',b5)

