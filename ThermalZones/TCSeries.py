#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:46:49 2018

@author: batela
"""
import warnings
import logging
import numpy as np
import pandas as pd

from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm

class TCSeries (object):

    def __init__(self, params, cols):
        '''
        Constructor esto es un comenari
        '''
        self.id         = params
        self.logger     = logging.getLogger("tzzipper")  
        self.coNames    = cols
         
    def clean (self):        
        self.data       = None
        
        
    def initialize (self,data):
        
        self.logger.info ("Inicializamos los valores..")
        
        self.data   = data
        self.data_log = np.log(self.data)
    
#        decomposition = seasonal_decompose(ts_log)        
        self.logger.info ("fin de inicializacion.")
        return self.data_log


    def checkStationarity(self,timeseries):
        res = False, 0
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        self.logger.info (dfoutput)
        if (dfoutput[0] <= dfoutput[4]):
            res = True, 1
        elif (dfoutput[0] <= dfoutput[5]):
            res = True , 5
        elif (dfoutput[0] <= dfoutput[5]):
            res = True , 10    
        
        return res


    def removeOutliers(self,data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/(mdev if mdev else 1.)
        return data[s<m]
    
    def smoothSerie(self,ts,box_pts):

        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(ts, box, mode='same')
        return y_smooth    
        

    
    def doForecasting  (self, data):
        self.logger.info ("Comenzamos la prediccion")
      
        self.logger.debug ("Fase #1: Descomposicion...")
        decomposition = seasonal_decompose(data['Pbld'])

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        residual.dropna(inplace=True)
        lag_acf = acf(residual, nlags=30)
        lag_pacf = pacf(residual, nlags=30, method='ols')

       
        self.logger.info ("Fin de la prediccion")
        return residual,lag_acf,lag_pacf
        
    # Evalua una tripleta individual
    def evaluate_arima_model(self,X, arima_order):
        	# prepare training dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        	# make predictions
        predictions = list()
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=-1)        		
        for t in range(len(test)):
        		yhat = model_fit.forecast()[0]
        		predictions.append(yhat)
        		history.append(test[t])
        	# calculate out of sample error
        error = mean_squared_error(test, predictions)
        aic = model_fit.aic
        return error,aic
 
    # Esta funcion evalua los diferentes posibles valores de d ,q ,d
    # para posteriormente quedarse con el mejor
    def evaluate_models(self,dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score,best_score_aic, best_cfg, best_cfg_aic = float("inf"),float("inf"),None, None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse, aic =  self.evaluate_arima_model(dataset, order)
                        
                        if mse < best_score:
                            best_cfg = mse, order
                            best_score = mse
                        #self.logger.info ("'ARIMA%s MSE=" + str((order,mse)))
                        if aic < best_score_aic:
                            best_cfg_aic = aic, order
                            best_score_aic = aic
                        
                        self.logger.info ("'ARIMA%s AIC=" + str((order,aic)))
                    except:
                        continue
        #self.logger.info ("BEST ARIMA= "+ str (best_cfg) + "AIC= " + str(best_score_aic))      
        return best_cfg , best_cfg_aic
        	
## Esta funcion analiza de forma stadistica la dependiencia de los valores entre si    
    def checkRegresionModel (self,fulldata):
        fulldata['const']=1
        olsModel=sm.OLS(endog=fulldata['Pbld'],exog=fulldata[['Toutdoor','const']])
        results1=olsModel.fit()
        print(results1.summary())    
        