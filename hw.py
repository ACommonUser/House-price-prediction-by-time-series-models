#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:32:34 2018

@author: zhouxiyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import holtwinters as hw
house_price = pd.read_csv('house price.csv')
#use 10 years as time span, starting from Apr 2007 
y = house_price['house price'][378:]
months = house_price['monthyear'][378:]
x = np.array([dt.datetime.strptime(d, '%b-%Y') for d in months])
xp = np.array([dt.datetime.strptime(d, '%b-%Y') for d in('Apr-2017','May-2017','Jun-2017','Jul-2017','Aug-2017','Sep-2017',
 'Oct-2017','Nov-2017','Dec-2017','Jan-2018','Feb-2018','Mar-2018')])

plt.figure(figsize = (12,8))
plt.plot(x,y,label='train set')
plt.xlabel('time span')
plt.ylabel('house price')
plt.title('Additive Holt winters forecasts')
# use the data from Apr 2007 to Mar 2017 as training set, Apr 2017 to Mar 2018 as testing set
from holtwinters import linear
from holtwinters import additive
from holtwinters import multiplicative
# Now we define how many predictions we wish to predict
fc = 12 # One year forecasting
# Now we define season length
h = 12
#split the original time series by train set and test set
train = y[:-12]
test = y[-12:]
x1 = x[:-12]
#the additive method forecasting
y_smoothed_additive, y_forecast_vals_additive, alpha, beta, gamma,rmse = hw.additive(train.tolist(), fc=fc, m=h)
plt.plot(np.hstack((x1,xp)),y_smoothed_additive[:-1], c = 'red',label="Additive forecasts")
plt.plot(xp,test,c = 'green',label='test set')
plt.legend()
print(alpha,beta,gamma)
#the multiplicative method forecasting
plt.figure(figsize = (12,8))
plt.plot(x,y,label='train set')
plt.xlabel('time span')
plt.ylabel('house price')
plt.title('Multiplicative Holt winters forecasts')
y_smoothed_mult, y_forecast_vals_Multi, alpha, beta, gamma,rmse = hw.multiplicative(train.tolist(), fc=fc, m=h)
plt.plot(np.hstack((x1,xp)),y_smoothed_mult[:-1], c = 'red',label="Multiplicitive forecasts")
plt.plot(xp,test,c = 'green',label='test set')
plt.legend()
print(alpha,beta,gamma)
print(rmse)
# model accuracy for the testing set
yf1 = y_forecast_vals_additive
yf2 = y_forecast_vals_Multi
def mae(yf1, test):
    return np.average(np.abs(yf1-test))
def mae(yf2, test):
    return np.average(np.abs(yf2-test))
def rmse(yf1,test):
    return np.sqrt(np.average(np.power(yf1-test,2)))
def rmse(yf2,test):
    return np.sqrt(np.average(np.power(yf2-test,2)))
#model accuracy for the training set
ys1 = y_smoothed_additive[:-13]
ys2 = y_smoothed_mult[:-13]
def mae(ys1, train):
    return np.average(np.abs(ys1-train))
def mae(ys2, train):
    return np.average(np.abs(ys2-train))
def rmse(ys1,train):
    return np.sqrt(np.average(np.power(ys1-train,2)))
def rmse(ys1,train):
    return np.sqrt(np.average(np.power(ys1-train,2)))
#final answers
print(mae(yf1, test),mae(yf2, test),rmse(yf1,test),rmse(yf2,test),mae(ys1, train),mae(ys2, train),rmse(ys1,train),rmse(ys2,train))