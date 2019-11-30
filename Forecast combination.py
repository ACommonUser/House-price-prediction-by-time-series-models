#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:40:17 2018

@author: zhouxiyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
fc = pd.read_csv('FC.csv')
true_price = fc['orignial data']
months = fc['monthyear']
hw = fc['HW'][-12:]
arima = fc['ARIMA'][-12:]
x = np.array([dt.datetime.strptime(d, '%b-%Y') for d in months])
xp = np.array([dt.datetime.strptime(d, '%b-%Y') for d in('Apr-2017','May-2017','Jun-2017','Jul-2017','Aug-2017','Sep-2017',
 'Oct-2017','Nov-2017','Dec-2017','Jan-2018','Feb-2018','Mar-2018')])
plt.figure(figsize=(12,8))
plt.plot(x,true_price, label='true house price')
plt.plot(xp,hw,label='HW prediction')
plt.plot(xp,arima,label='ARIMA prediction')

#the residual
y = true_price[-12:]
residuals1 = y - hw
residuals2 = y - arima
#the covariance
cov = np.cov(residuals1,residuals2)
# the variance of two error series
var1 = cov[0,0]
var2 = cov[1,1]
#correlation coefficient rho
rho = cov[0,1]/(np.sqrt(var1*var2))
#find the optimal w
optimal_weights = (np.power(var2,2)-rho*var1*var2)/(np.power(var1,2)+np.power(var2,2)-2*rho*var1*var2)
print(optimal_weights)

combine = optimal_weights*hw + (1-optimal_weights)*arima
plt.plot(xp,combine,label='combined forecast')
plt.legend()

#comparison
plt.figure(figsize=(20,12))
plt.plot(xp,y, label='true house price')
plt.plot(xp,hw,label='HW prediction')
plt.plot(xp,arima,label='ARIMA prediction')
plt.plot(xp,combine,label='combined forecast')
plt.legend()