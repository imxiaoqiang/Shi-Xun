import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


df_train = pd.read_csv('C:/Users/lincanyuan/Downloads/Big-Data-Competition-Project/02_共享单车(Bike Sharing Demand)/input/train.csv',header = 0)

df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
df_train['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour
df_train_origin = df_train #保存原数据集
df_train = df_train.drop(['datetime','casual','registered'], axis = 1)


df_train_target = df_train['count'].values
df_train_data = df_train.drop(['count'],axis = 1).values




spliter = model_selection.ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)


print("支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)")
cv = spliter.split(df_train_data)
for train, test in cv:
    svc = svm.SVR(kernel='rbf', C=10, gamma=.001).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))

print("随机森林回归/Random Forest(n_estimators = 100)")
cv = spliter.split(df_train_data)
for train, test in cv:
    svc = RandomForestRegressor(n_estimators=100).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))