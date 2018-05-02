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
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        self.logger.info (dfoutput)

    def doForecasting  (self, data):
        self.logger.info ("Comenzamos la prediccion")
      
        self.logger.debug ("Fase #1: Descomposicion...")
        decomposition = seasonal_decompose(data)

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
        	for t in range(len(test)):
        		model = ARIMA(history, order=arima_order)
        		model_fit = model.fit(disp=-1)
        		yhat = model_fit.forecast()[0]
        		predictions.append(yhat)
        		history.append(test[t])
        	# calculate out of sample error
        	error = mean_squared_error(test, predictions)
        	return error
 
    # Esta funcion evalua los diferentes posibles valores de d ,q ,d
    # para posteriormente quedarse con el mejor
    def evaluate_models(self,dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse =  self.evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_cfg = mse, order
                        self.logger.info ("'ARIMA%s MSE=" + str((order,mse)))
                    except:
                        continue
        self.logger.info ("BEST ARIMA= "+ str (best_cfg) + "MSE= " + str(best_score))      
        return best_cfg
        	
