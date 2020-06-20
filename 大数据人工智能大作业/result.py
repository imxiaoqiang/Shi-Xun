import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


# 读入原始数据
train_df = pd.read_csv("C:/Users/lincanyuan/Downloads/Big-Data-Competition-Project/02_共享单车(Bike Sharing Demand)/input/train.csv", header=0)
test_df = pd.read_csv("C:/Users/lincanyuan/Downloads/Big-Data-Competition-Project/02_共享单车(Bike Sharing Demand)/input/test.csv", header=0)


# 选取特征值
selected_features = ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']

X_train_df = train_df[selected_features]
y_train_df = train_df['count']

X_test_df = test_df[selected_features]


# 特征值处理
X_train_df['month'] = pd.DatetimeIndex(X_train_df.datetime).month
X_train_df['day'] = pd.DatetimeIndex(X_train_df.datetime).dayofweek
X_train_df['hour'] = pd.DatetimeIndex(X_train_df.datetime).hour
X_train_df = X_train_df.drop(['datetime'], axis=1)  # axis=1意思为每行执行，axis=0表示每列执行

X_test_df['month'] = pd.DatetimeIndex(X_test_df.datetime).month
X_test_df['day'] = pd.DatetimeIndex(X_test_df.datetime).dayofweek
X_test_df['hour'] = pd.DatetimeIndex(X_test_df.datetime).hour
X_test_df = X_test_df.drop(['datetime'], axis=1)



# 采用DictVectorizer进行特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train_df = dict_vec.fit_transform(X_train_df.to_dict(orient='record'))
X_test_df = dict_vec.transform(X_test_df.to_dict(orient='record'))


# 使用RandomForestRegressor进行回归预测
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
gbr = RandomForestRegressor()
gbr.fit(X_train_df, y_train_df)
gbr_y_predict = gbr.predict(X_test_df)


# 输出结果
gbr_submission = pd.DataFrame({'datetime': test_df['datetime'], 'count': gbr_y_predict})
gbr_submission.to_csv('C:/Users/lincanyuan/Downloads/Big-Data-Competition-Project/02_共享单车(Bike Sharing Demand)/input/submission.csv', index=False)
