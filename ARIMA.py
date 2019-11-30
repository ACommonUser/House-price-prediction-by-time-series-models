# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:06:46 2018

@author: Serina
"""

#import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm 
import statsmodels.api as smt
from statsmodels.tsa.arima_model import ARIMA
#read data
data = pd.read_csv('purchase_price.csv')

# Seasonal period
M = 12
# 2D-array
data_v = data['price'].iloc[:120].values
print(data_v)

data_log = np.log(data_v)
             
# format and plot the data
fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(211)
ax1.plot(data_v)
plt.title('Purchase price',fontsize=20)
ax1.tick_params(labelsize=22,length=10,width=3)


plt.tight_layout()

from scipy.optimize import brute
def objfunc(order, data):
    try:
        fit = ARIMA(data, order[0:3]).fit()
        return fit.aic
    except:
        return np.inf

grid = (slice(0, 4, 1), slice(1,2,1), slice(0, 4, 1))
optimal = brute(objfunc, grid, args=(data_v,), finish=None)

optimal_converted3 = [int(x) for x in optimal]

data_df1m=np.array(data_v,dtype=np.float) 

#optimal_result = ARIMA(data_df1m, optimal_converted).fit(disp=False)
#print(optimal_result.summary())
arima_model = sm.tsa.arima_model.ARIMA(data_df1m, order=(2,1,1))
result_a = arima_model.fit(disp=False)
print(result_a.summary())

data_qr = np.power(data_v,0.25)
data_qr_d = np.diff(data_qr)  

# format and calculate the data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(data_qr)
ax1.tick_params(labelsize=15,length=10,width=3)
plt.title('Quartic Roots of Purchase price',fontsize=20)
ax2 = fig.add_subplot(212)
ax2.plot(data_qr_d)
ax2.tick_params(labelsize=15,length=10,width=3)
plt.title('Differenced Quartic Roots of Purchase price',fontsize=20)
plt.tight_layout()


# Do seasonally differencing of quartic root
data_ds = data_qr[12:] - data_qr[:-12]
print(data_ds.size)

# first order difference
data_dsd = np.diff(data_ds);
# plot data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(data_ds)
plt.title('Seasonally Differenced price')
ax2 = fig.add_subplot(212)
ax2.plot(data_dsd)
plt.title('Regularlly Differenced price')

# Draw ACF and PACF for the quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_qr, lags=40, ax=ax1)
ax1.set_title("ACF: Quartic Root Data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_qr, lags=40, ax=ax2)
ax2.set_title("PACF: Quartic Root Data")

# For the ordinary difference of quartic root data
fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_qr_d, lags=40, ax=ax1)
ax1.set_title("ACF: first order difference of quartic root data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_qr_d, lags=40, ax=ax2)
ax2.set_title("PACF: first order difference of quartic root data")

# For the seasonal difference of quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_ds, lags=40, ax=ax1)
ax1.set_title("ACF: seasonal difference of quartic root data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_ds, lags=40, ax=ax2)  
ax2.set_title("PACF: seasonal difference of quartic root data")   


# For first order difference of seasonally differenced quartic root data
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(data_dsd, lags=40, ax=ax1)
ax1.set_title("ACF: differencing the seasonally differenced quartic root data")
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(data_dsd, lags=40, ax=ax2)  
ax2.set_title("PACF: differencing the seasonally differenced quartic root data")
"""
the ACF is dying down However this dying-down behavior at the nonseasonal level
and cutting-off behaviour at the seasonal level do not appear to be as
quick as the dying-down behaviour at the nonseasonal level and the
cutting-off behavior at the seasonal level as the last illustration
"""                      

"""
Conclusion
We will build a model based on the Seasonally differenced quartic roots
of price

1. At the nonseasonal level the Sample PACF has spikes at lags 1, 3 and 5,
and cuts off after lag 5 and the the Sample ACF dies down. Therefore we
tentatively identify the following nonseasonal autoregressive model
Z_ds(t) = c + \phi_1 Z_ds(t-1) + \phi_3Z_ds(t-3) + \phi_5Z_ds(t-5) + e_t
 
2. At the seasonal level the sample ACF has a spike at lag 12 and cuts
off after lag 12, and Sample PACF dies down. Therefore, we tentatively
identify the seasonal moving average model of oder 1
   Z_ds(t) = c + e(t) + \Theta_1 e(t-12)

3. Combining these models, we obtain the overall tentatively identified
 model
  Z_ds(t) = c + \phi_1 Z_ds(t-1) + \phi_3Z_ds(t-3) + \phi_5Z_ds(t-5) + e(t) + \Theta_1 e(t-12)
"""
from scipy.optimize import brute
def objfunc(order, data):
    try:
        fit = ARIMA(data, order[0:3]).fit()
        return fit.aic
    except:
        return np.inf

grid = (slice(0, 4, 1), slice(1,2,1), slice(0, 4, 1))
optimal = brute(objfunc, grid, args=(data_dsd,), finish=None)

optimal_converted3 = [int(x) for x in optimal]

data_df3m=np.array(data_dsd,dtype=np.float) 

#optimal_result = ARIMA(data_df1m, optimal_converted).fit(disp=False)
#print(optimal_result.summary())
arima_model = sm.tsa.arima_model.ARIMA(data_df3m, order=(2,1,1))
result_c = arima_model.fit(disp=False)
print(result_c.summary())
# Define the model according to the identificated pattern
# We use p = (1,3,5) to indicate that only lags 1, 3, 5 in AR process
sarima_model = smt.tsa.statespace.SARIMAX(data_qr, order=(5,0,0), seasonal_order=(0,1,1,12))   


# sarima_model = smt.tsa.statespace.SARIMAX(data_log, order=((1,3,5),0,0), seasonal_order=(0,1,1,12))   
# In the second format, we just say p = 5 including AR lags 1, 2, 3, 4, 5

# Estimating the model
result = sarima_model.fit(disp=False)
print(result.summary())



# Forecasting
forecasts = result.forecast(12)
print(forecasts)

# Display forecasting
fig = plt.figure(figsize=(10,8)) 
data_original=data['price'].values
plt.plot(np.arange(1,133),data_original)
forecast=np.power(forecasts,4)
plt.plot(np.arange(121,133), forecast)


#model performance testing
def rmse(x,y):
    return np.sqrt(np.average(np.power(x-y,2)))

def mad(x,y):
    return np.average(np.abs(x-y))

def mape(x,y):
    return np.average(((np.abs(x-y))/x)*100)

x=data_original[120:132]
y=forecast

print(rmse(x,y))
print(mad(x,y))
print(mape(x,y))


# Validation
forecasts = result.forecast(12)

# Display forecasting
fig = plt.figure(figsize=(10,8)) 
plt.plot(data_qr)
plt.plot(np.arange(121,133),forecasts)

