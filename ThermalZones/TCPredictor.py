#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:26:57 2018

@author: deba
"""

import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


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
        
        
        
        self.logger.info ("fin de inicializacion..")
        
        return rowData
    
    def TCPloter (self, data):
        
        plt.close()
        plt.plot(data["Php.1"])          
        plt.show()