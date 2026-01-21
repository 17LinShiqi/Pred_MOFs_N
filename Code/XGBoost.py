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


X = pd.read_excel("D:/NH3xifu.xlsx", 
                  sheet_name="Sheet4", usecols="B:H")
Y = pd.read_excel("D:/NH3xifu.xlsx", 
                  sheet_name="Sheet4", usecols=['N'])


if 'K' in X.columns:
    X['K'] = np.where(X['K'] > 0, np.log(X['K']), 0) 


my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
Y = my_imputer.fit_transform(Y.values.reshape(-1, 1)).ravel()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


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


xgb.fit(X_train, y_train)
y_pred1 = xgb.predict(X_train)
y_pred2 = xgb.predict(X_test)

## save model
os.makedirs("../model", exist_ok=True) 
joblib.dump(xgb, "../model/xgb.pt")  


RMSE_train = np.sqrt(metrics.mean_squared_error(y_train, y_pred1))
MAE_train = metrics.mean_absolute_error(y_train, y_pred1)
R2_train = r2_score(y_train, y_pred1)
print(f'R2_1={R2_train}')
print(f'MAE={MAE_train}')
print(f'RMSE={RMSE_train}')


RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
MAE_test = metrics.mean_absolute_error(y_test, y_pred2)
R2_test = r2_score(y_test, y_pred2)
print(f'R2_2={R2_test}')
print(f'MAE={MAE_test}')
print(f'RMSE={RMSE_test}')

'''

plot_importance(xgb, importance_type='weight')
pyplot.show()
'''
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

print(f"training time: {training_time} seconds")

