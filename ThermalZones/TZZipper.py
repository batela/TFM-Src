#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:26:57 2018

@author: deba
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import fcluster

class TZZipper (object):

    def __init__(self, params):
        '''
        Constructor esto es un comenario
        '''
        self.id     = params
        self.data   = None
        self.logger = logging.getLogger("tzzipper")    
        #self.roomNames =['R0T','R1','R2','R3','R4','R5','R6','R7T']
        self.roomNames =['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11']    
        self.k=None
        self.period = None
        self.days= None
        
        
    def clean (self):        
        self.data       = None
        self.k=None
        self.period = None
        self.days= None
        
        
    def initialize (self,data,k,period,days):
        
        self.logger.info ("Inicializamos los valores..")
        
        self.k=7
        self.period = period
        #self.days= 17
        self.days= days
        
        self.data       = data
        dayItems = (24*60//self.period)        
        rowData = []
        
        for ro in self.roomNames:
            rowItem = self.data[[ro]]
            if (self.days > 0):
                rowData = np.append(rowData,rowItem[:dayItems*self.days])
            else:
                rowData = np.append(rowData,rowItem)
        
        self.logger.info ("fin de inicializacion..")
        
        return np.resize (rowData,(len(rowData)//dayItems,dayItems))
    
    def clusterizeKmeans (self,groups,X):
        self.logger.info ("Inicio clusterizaccion..")
    
        # Para revisar
        #kmeans = KMeans(n_clusters=groups, random_state=0).fit(X)
        #ll = kmeans.labels_
   
        self.logger.info ("Fin cluseterizacion")
#   BTL estorr revisar        
    def elbowMethod  (self,Z):
        last = Z[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        plt.plot(idxs, last_rev)
        
        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        plt.plot(idxs[:-2] + 1, acceleration_rev)
        plt.show()
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
        self.logger.info ("Cluster calculation" + str(k))
        
        
    def clusterizeHClust (self,X):
        self.logger.info ("Inicio clusterizaccion por hierarchical..")
        Z = linkage(X, 'ward')
        
        self.elbowMethod (Z)
        
        #c, coph_dists = cophenet(Z, pdist(X))
        cl = fcluster(Z, self.k, criterion='maxclust')
        
        self.plotter(Z)
        return cl
    
    
    def plotter (self,Z):
        
        self.logger.info ("Inicio ploting dendongram..")
        
        #plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=5,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show()
        self.logger.info ("Fin plotting..")