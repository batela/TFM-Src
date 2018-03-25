#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:46:49 2018

@author: batela
"""
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


#from sklearn.cluster import KMeans
#from scipy.cluster.hierarchy import cophenet

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
        print (dfoutput)

    def doForecasting  (self, data):
        self.logger.info ("Comenzamos la prediccion")
      
        self.logger.debug ("Fase #1: Descomposicion...")
        decomposition = seasonal_decompose(data)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
       
        self.logger.info ("Fin de la prediccion")
        return trend,seasonal,residual