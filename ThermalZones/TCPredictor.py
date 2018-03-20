#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:26:57 2018

@author: deba
"""

import logging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from scipy.signal import savgol_filter
from sklearn.cross_validation import cross_val_score


from matplotlib import pyplot as plt
#from sklearn.cluster import KMeans
#from scipy.cluster.hierarchy import cophenet

class TCPredictor (object):

    def __init__(self, params, coNames):
        '''
        Constructor esto es un comenari
        '''
        self.id     = params
        self.consumptionNames = coNames
        self.logger = logging.getLogger("tzzipper")  
        self.data       = None
         
    def clean (self):        
        self.data       = None
        self.k=None
        self.period = None
        self.days= None
        
        
    def initialize (self,data,period,days):
        
        self.logger.info ("Inicializamos los valores..")
        
        rowData = []
        dayItems = (24*60//period)   
        self.data       = data
        self.days       = days
        
        for co in self.consumptionNames:
            rowItem = self.data[[co]]
            if (self.days > 0):
                rowData = np.append(rowData,rowItem[:dayItems*days])
            else:
                rowData = np.append(rowData,rowItem)
        
    
        
        self.logger.info ("fin de inicializacion.")
        return np.resize (rowData,(len(rowData)//dayItems,dayItems))
    
    
    def integrate (self,data,periodInt):
        
        self.logger.info ("Comenzamos integracion..")
        
        dayDataInt = []
        
        ids = np.repeat(np.arange(24//periodInt), 96//(24//periodInt)) 
        for i in np.arange (data.shape[0]): 
            dayDataInt = np.append(dayDataInt,np.bincount(ids, weights=data[i])//np.bincount(ids))
#            if (i < 90):
#                dayDataInt = np.append(dayDataInt,np.bincount(ids, weights=data[i])//np.bincount(ids))
#            else :
#                dayDataInt = np.append(dayDataInt,np.bincount(ids, weights=data[i]))
        self.logger.info ("Finalizamos integracion..")
#        readyData = np.resize (dayDataInt,(len(dayDataInt)//(24//periodInt),(24//periodInt)))
        readyData = np.resize (dayDataInt,(4,len(dayDataInt)//(4)))
        tmp = readyData[3]
        tmp[tmp<500] = 0
        tmp[np.where ((tmp>=500) & (tmp<1000))] = 1
        tmp[np.where ((tmp>=1000) & (tmp<1500))] = 2
        tmp[np.where ((tmp>=1500) & (tmp<2000))] = 3
        tmp[np.where ((tmp>=2000) & (tmp<2500))] = 4
        tmp[np.where (tmp>=2500)] = 5
        self.TCPloter (readyData[0], tmp)
        return readyData[0], tmp
    
    def doForecasting  (self,  X,Y):
        y = np.array(Y)
        
        ids = np.repeat(np.arange(180//3), 3) 
        ysm= np.bincount(ids, y)//np.bincount(ids)
        
        self.logger.info ("Testing with cross-validation")
        
        mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
        self.logger.info((cross_val_score(mlp, X.reshape(-1,1).astype(int), ysm.repeat(3).reshape(-1,).astype(int), scoring='accuracy', cv = 3)))
        
#        scaler = StandardScaler()
#        Fit only to the training data
#        scaler.fit(X_train
#        X_train = scaler.transform(X_train)
#       X_test = scaler.transform(X_test)
        
        self.logger.info ("Testing with split test set..")
        
        mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
        X_train, X_test, y_train, y_test = train_test_split(X.astype(int), ysm.repeat(3).astype(int))  
        mlp.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
        predictions = mlp.predict(X_test.reshape(-1,1))
        self.logger.info((confusion_matrix(y_test,predictions)))    
        self.logger.info((classification_report(y_test,predictions)))
        
        return None
    
    def TCPloter (self,x,y):
        
        plt.close()
        yhat = savgol_filter(y, 5, 3) # window size 51, polynomial order 3
        
        ids = np.repeat(np.arange(180//3), 3) 
        ysm= np.bincount(ids, y)//np.bincount(ids)
        
        
        plt.plot(x)
        plt.plot(y)
        plt.plot(ysm.repeat(3))     
        plt.show()