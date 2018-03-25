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
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
 
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
<<<<<<< HEAD
        tmp = readyData[3]
        tmp[tmp<300] = 0
        tmp[np.where ((tmp>=300) & (tmp<600))] = 1
        tmp[np.where ((tmp>=600) & (tmp<900))] = 2
        tmp[np.where ((tmp>=900) & (tmp<1200))] = 3
        tmp[np.where ((tmp>=1200) & (tmp<1500))] = 4
        tmp[np.where (tmp>=1500)] = 5
        self.TCPloter (readyData[0], tmp)
        return readyData[0], tmp
=======
        return readyData[0], readyData[3]
>>>>>>> 54bebe4ffc6b3dbcb3a327f617f483f4e2092637
    
    def labelize (self,data,n, bins, patches):
        
        tmp = data
        #tmp[tmp< (434 + 195//2)] = 0
        #tmp[np.where ((tmp>=(434 + 195//2)) & (tmp<(1019 + 195//2)))] = 1
        #tmp[np.where ((tmp>=(1019 + 195//2)) & (tmp<(1214 + 195//2)))] = 2
        #tmp[np.where ((tmp>=(1214 + 195//2)) & (tmp<(1409 + 195//2)))] = 3
#        tmp[np.where ((tmp>=(1409 + 195//2)) & (tmp<(1214 + 195//2)))] = 4
        #tmp[np.where (tmp>=(1409 + 195//2))] = 4
        
        tmp[tmp< (500)] = 0
        tmp[np.where ((tmp>=(500)) & (tmp<(750)))] = 1
        tmp[np.where ((tmp>=(750)) & (tmp<(1000)))] = 2
        tmp[np.where ((tmp>=(1000)) & (tmp<(1500)))] = 3
#        tmp[np.where ((tmp>=(1409 + 195//2)) & (tmp<(1214 + 195//2)))] = 4
        tmp[np.where (tmp>=(1500))] = 4
        
        
        return tmp
        
        
    def doForecasting  (self,  X,Y):
        
        y = np.array(Y).astype(int)
#        y=shift(y, -0, cval=y[-1]) # Valido para desplazar la curva, no se usa
        
#       Se prueban diferente regresores con crosvalidation
        ids = np.repeat(np.arange(180//3), 3) 
        ysm= np.bincount(ids, y)//np.bincount(ids)        
        xsm= np.bincount(ids, X)//np.bincount(ids)
        
        data_input =  X.reshape(-1,1).astype(int) #xsm.repeat(3).reshape(-1,1).astype(int)
        data_output = ysm.repeat(3).reshape(-1,).astype(int)
        
        self.logger.info ("Testing with cross-validation")
        
        mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
        self.logger.info((cross_val_score(mlp, data_input, data_output, scoring='accuracy', cv = 5)))
                
        rf_class = RandomForestClassifier(n_estimators=10)
        log_class = LogisticRegression()
        svm_class = svm.SVC()

        self.logger.info((cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 5)))
        accuracy = cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 5).mean() * 100
        self.logger.info("Accuracy of Random Forests is: " , accuracy)
         
        self.logger.info((cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv = 5)))
        accuracy = cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv = 5).mean() * 100
        self.logger.debug("Accuracy of SVM is: " , accuracy)
         
        self.logger.debug((cross_val_score(log_class, data_input, data_output, scoring='accuracy', cv = 5)))
        accuracy = cross_val_score(log_class, data_input, data_output, scoring='accuracy', cv = 5).mean() * 100
        self.logger.debug("Accuracy of SVM is: " , accuracy)
        
#        scaler = StandardScaler()
#        Fit only to the training data
#        scaler.fit(X_train
#        X_train = scaler.transform(X_train)
#       X_test = scaler.transform(X_test)

#       Se prueba el MLP dividiendo la muestra
        self.logger.info ("Testing with split test set..")
        mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
        X_train, X_test, y_train, y_test = train_test_split( data_input, data_output)  

        mlp.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
        predictions = mlp.predict(X_test.reshape(-1,1))
        
        self.logger.info((confusion_matrix(y_test,predictions)))    
        self.logger.info((classification_report(y_test,predictions)))
        
        return None
    
    def TCPloter (self,x,y):
        
        plt.close()
<<<<<<< HEAD
=======
      
>>>>>>> 54bebe4ffc6b3dbcb3a327f617f483f4e2092637
        ids = np.repeat(np.arange(180//3), 3) 
        
        Xsm = np.bincount(ids, x)//np.bincount(ids)
        ysm= np.bincount(ids, y)//np.bincount(ids)
        xsm= np.bincount(ids, x)//np.bincount(ids)
        
        y = ysm.repeat(3)
        y= shift(y, -15, cval=y[-1])
        plt.plot(x)
        plt.plot(y)
<<<<<<< HEAD
        plt.plot(ysm.repeat(3)) 
        plt.plot(Xsm.repeat(3)) 
=======
        
        plt.plot(ysm.repeat(3))       
>>>>>>> 54bebe4ffc6b3dbcb3a327f617f483f4e2092637
        plt.show()