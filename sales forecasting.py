# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 01:14:27 2022

@author: laxmi
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
 
train = pd.read_csv("C://Users//laxmi//OneDrive//Documents//BA//Sales Forecasting//train.csv")
train =  train.fillna(0)

features = pd.read_csv("C://Users//laxmi//OneDrive//Documents//BA//Sales Forecasting//features.csv")
features = features.fillna(0)

train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])
ad_data = pd.merge(features, train, on=["Store", "Date"],how='inner')
ad_data = ad_data.fillna(0)
ad_data

ad_data.head()
print(ad_data.head())
ad_data.info()
ad_data.describe()
ad_data.columns
ad_data.corr()

x = ad_data[['MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI']]
y = ad_data.Weekly_Sales

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X,y)

y_pred = regr.predict(X)
print(X[1:10])

regr.intercept_
regr.coef_

# y = b0 + b1 x   -- weeekly sales = 15981 + 540 * markdown4 + 1018 * m5 -543 * cpi     x = ad_data[['MarkDown4', 'MarkDown5', 'CPI']]
Sales = 15981+732*(ad_data["MarkDown1"])+241*(ad_data["MarkDown2"])+849*(ad_data["MarkDown3"])-30*(ad_data["MarkDown4"])+819*(ad_data["MarkDown5"])-533*(ad_data["CPI"])
Sales

print("R squared: {}".format(r2_score(y_true=y,y_pred=y_pred)))
