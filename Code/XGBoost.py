from datetime import datetime
start_time = datetime.now()
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.model_selection import KFold, cross_validate as CVS, cross_val_predict as CVP, train_test_split as TTS
from matplotlib import pyplot
import joblib
import os

# 数据读取
X = pd.read_excel("D:/NH3xifu.xlsx", 
                  sheet_name="Sheet4", usecols="B:H")
Y = pd.read_excel("D:/NH3xifu.xlsx", 
                  sheet_name="Sheet4", usecols=['N'])

## 对“亨利系数”一列进行对数化处理，假设“亨利系数"命名为“K”
if 'K' in X.columns:
    X['K'] = np.where(X['K'] > 0, np.log(X['K']), 0)  # 对G=0使用0，其他情况计算log()

# 查找缺失值并填充
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
Y = my_imputer.fit_transform(Y.values.reshape(-1, 1)).ravel()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# 配置XGBoost回归器
xgb = XGBRegressor(max_depth=36, 
                   random_state=26, 
                   min_child_weight=16, 
                   learning_rate=0.01, 
                   subsample=0.9, 
                   booster='gbtree', 
                   objective='reg:squarederror', 
                   colsample_bytree=0.9, 
                   importance_type='gain', 
                   gamma=0.01, 
                   reg_lambda=0.1, 
                   reg_alpha=0.03, 
                   n_estimators=1400)

# 训练模型
xgb.fit(X_train, y_train)
y_pred1 = xgb.predict(X_train)
y_pred2 = xgb.predict(X_test)

## save model
os.makedirs("../model", exist_ok=True) 
joblib.dump(xgb, "../model/xgb.pt")  # 保存模型

# 训练集性能评估
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train, y_pred1))
MAE_train = metrics.mean_absolute_error(y_train, y_pred1)
R2_train = r2_score(y_train, y_pred1)
print(f'R2_1={R2_train}')
print(f'MAE={MAE_train}')
print(f'RMSE={RMSE_train}')

# 测试集性能评估
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
MAE_test = metrics.mean_absolute_error(y_test, y_pred2)
R2_test = r2_score(y_test, y_pred2)
print(f'R2_2={R2_test}')
print(f'MAE={MAE_test}')
print(f'RMSE={RMSE_test}')

'''
# 计算特征重要性并显示
plot_importance(xgb, importance_type='weight')
pyplot.show()
'''
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()
# 运行时间的计算
print(f"训练时间: {training_time} 秒")
