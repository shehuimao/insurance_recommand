import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 進制轉換
def binary_to_int(x):
    num = 0
    m = len(x)-1
    for i in x:
        num += i*2**m
        m-=1
    return num

# 編碼
def encode(data, cols):
    encoder = LabelEncoder()
    for col in cols:
        data[col] = encoder.fit_transform(data[col])
    return data

# 測試和訓練數據劃分
def tt_split(data):
    train, test = train_test_split(data,  test_size=0.1, train_size=0.9, random_state=1)
    return train, test

# 數據標準化
def scale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

    return X_train, X_test

# 預處理
train_data = pd.read_csv("DATA/train.csv")
test_data = pd.read_csv("DATA/Test.csv")
sample = pd.read_csv("samplesubmission.csv" )


# print(test_data.shape, train_data.shape, sample.shape)

# 缺失值查看
# print(test_data.isnull().any())
# print(train_data.isnull().any())

# 缺失值統計
# print(train_data.isnull().sum())
# print(test_data.isnull().sum())

# 填充缺失值，這裡用就近的數值填充
train_data.join_date = train_data.join_date.fillna("1/5/2018")
test_data.join_date = train_data.join_date.fillna("1/5/2018")

# 新增特征
train_data["subscribed"] = binary_to_int([train_data['P5DA'], train_data['RIBP'],train_data['8NN1'],
                                          train_data['7POT'], train_data['66FJ'],train_data['GYSR'],
                                          train_data['SOP4'], train_data['RVSZ'],train_data['PYUQ'],
                                          train_data['LJR9'], train_data['N2MW'], train_data['AHXO'],
                                          train_data['BSTQ'], train_data['FM3X'],train_data['K6QO'],
                                          train_data['QBOL'], train_data['JWFN'],train_data['JZ9D'],
                                          train_data['J9JW'], train_data['GHYX'], train_data['ECY3']])

test_data["subscribed"] = binary_to_int([test_data['P5DA'], test_data['RIBP'],test_data['8NN1'],
                                          test_data['7POT'], test_data['66FJ'],test_data['GYSR'],
                                          test_data['SOP4'], test_data['RVSZ'],test_data['PYUQ'],
                                          test_data['LJR9'], test_data['N2MW'], test_data['AHXO'],
                                          test_data['BSTQ'], test_data['FM3X'],test_data['K6QO'],
                                          test_data['QBOL'], test_data['JWFN'],test_data['JZ9D'],
                                          test_data['J9JW'], test_data['GHYX'], test_data['ECY3']])

train_data['join_month'] = train_data['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
train_data['join_year'] = train_data['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
train_data.drop('join_date', axis=1, inplace=True)

test_data['join_month'] = test_data['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)
test_data['join_year'] = test_data['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)
test_data.drop('join_date', axis=1, inplace=True)

train_data['join_age'] = train_data['join_year'] - train_data['birth_year']
test_data['join_age'] = test_data['join_year'] - test_data['birth_year']

# 32個新特征
train_data = train_data[['ID', 'join_month','join_year','join_age', 'sex', 'marital_status', 'birth_year', 'branch_code',
       'occupation_code', 'occupation_category_code','subscribed', 'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']]

# 32個新特征
test_data = test_data[['ID', 'join_month','join_year','join_age', 'sex', 'marital_status', 'birth_year', 'branch_code',
       'occupation_code', 'occupation_category_code','subscribed', 'P5DA', 'RIBP', '8NN1',
       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']]

# 目標轉換，轉換為21個分類問題
train_data = train_data.melt(id_vars=train_data.columns[:11], value_vars=train_data.columns[11:],
                     var_name = "PCODE", value_name="Label" )
test_data = test_data.melt(id_vars=test_data.columns[:11], value_vars=test_data.columns[11:],
                   var_name = "PCODE", value_name="Label" )
melted_test = test_data[["ID","PCODE"]]

print(train_data.shape, test_data.shape, sample.shape)


# 類別型變量編碼轉換
train_data = encode(train_data, ['sex', 'marital_status', 'branch_code', 'occupation_code',
                          'occupation_category_code', 'PCODE'])

test_data = encode(test_data, ['sex', 'marital_status', 'branch_code', 'occupation_code',
                        'occupation_category_code', 'PCODE'])

# 在訓練集裡面劃分訓練和測試
train, test = tt_split(train_data.iloc[:, 1:])
print(train.shape, test.shape)

# 標準化
train, test = scale(train, test)

scaler = MinMaxScaler()
test_data = pd.DataFrame(scaler.fit_transform(test_data.iloc[:, 1:-1]), columns=test_data.iloc[:,1:-1].columns)

# 切片
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# 模型
def train_model(classifier, X_train, y_train):
    """

    :param classifier: 這裡可以放sklearn裡面的各種分類模型
    :param X_train:
    :param y_train:
    :return:
    """
    accuracy = []
    skf = StratifiedKFold(n_splits=5, random_state=None)
    skf.get_n_splits(X_train, y_train)

    for train_index, test_index in skf.split(X_train, y_train):
        print("TRAIN:", train_index.min(), "to", train_index.max(), "TEST:",
              test_index.min(), "to", test_index.max())

        X1_train, X1_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y1_train, y1_test = y_train.iloc[train_index], y_train.iloc[test_index]

        classifier.fit(X1_train, y1_train)
        prediction = classifier.predict(X1_test)
        score = metrics.log_loss(prediction, y1_test)
        accuracy.append(score)

    print("\nrank loss: ", accuracy)
    print("\nAverage rank loss =", np.array(accuracy).mean())

    return classifier

# 返回log_loss
def get_accuracy(model, X_test, y_test):
    prediction = model.predict(X_test)
    score = metrics.log_loss(prediction,y_test)
    return score

# 預測的概率值
def get_proba(model, z_test):
    proba = model.predict_proba(z_test)[:, 1]
    proba = pd.DataFrame(proba)

    return proba

# 提交
def get_sub(proba):
    final = melted_test
    final["ID X PCODE"] = final["ID"] + " X " + final["PCODE"]
    final["Label"] = proba

    return final[["ID X PCODE", "Label"]]

# 把0值填充
def fill_ones(sub):
    ones = sample[sample.Label == 1]
    ones_index = ones.index
    for i in range(len(sub)):
        if i in ones_index:
            sub['Label'].iloc[i] = 1.0
    return sub

def export(df, name):
    df.to_csv(name, index=False)

# 下面是導入模型的樣例
import xgboost as xgb
xg_boost = xgb.XGBClassifier()
xg_boost = train_model(xg_boost, X_train, y_train)
get_accuracy(xg_boost, X_test,y_test)
proba = get_proba(xg_boost, test_data)
sub = get_sub(proba)
sub = fill_ones(sub)

export(sub,'xgboost_sub.csv')

