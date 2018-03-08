#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:01:08 2018

@author: deba
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:48:36 2017

@author: 105083
"""
import logging
import pandas as pd
import numpy
import imp
import TZZipper
from matplotlib import pyplot as plt2


def loadFileData (filepath):
                
        #self.logger.info ("Loading data from file....")         
        
        #filename = 'PumpGround_allData.csv'
        #data = pd.read_csv(filepath+filename, sep=';', decimal = ',',thousands = '.', usecols=['DATE', 'Timpout36', 'Tretout37','Timpin22','Tretin23','Econsump', 'ONOFF'], parse_dates=['DATE'])
        
        
        filename = 'FHP-20141008.csv'        
        filename = 'datosvivienda_test.csv'
        filename = 'ed700.csv'
        #usecols = ['Fecha','geotech_raw_36_Temperatura_T1','geotech_raw_37_Temperatura_T2','geotech_raw_256_EnergiaActiva','geotech_raw_2_T_Aire_Exterior','geotech_raw_22_EP_Impulsion_T','geotech_raw_23_EP_Retorno_T','geotech_raw_24_Caudal']
        data = pd.read_csv(filepath+filename, sep=';', decimal = '.')
        return data
    

def saveFileData (filepath,data):
                
        #self.logger.info ("Loading data from file....")            di =     
        filename = 'pydata.csv'        
        numpy.savetxt(filepath+filename, data, fmt='%.4f', delimiter=';')
        
        return data

def calculaStadist(data,cl):
        dias = 579
        rooms = 12
        idx = -1
        di = None
        for ro in list(range (0,rooms)):
            idx +=1
            l = list(cl[dias*idx:dias*(idx+1)])
            for it in list(range (1,numpy.amax(cl)+1)):
                di = dict((it,l.count(it)) for it in set(l))
            logger.info ("Found data for room " + str (ro) +" : "+ str(di))
                
def auxPlotter(data,cl):
                
    idx = -1
    count = -1;
    ptidx = -1
    
    colour=['blue','green','red','orange','cyan','black','pink','magenta']
    # Two subplots, the axes array is 1-d
    #f, ptarr = plt2.subplots(2, sharex=True)
    for it in data:
        idx+=1 
        #count+=1
        #if ((count %579) == 0):
        #    ptidx+=1
        plt2.plot(it,color=colour[cl[idx-1]])
        plt2.savefig('../Images/myfigdv')
        ##if ((idx %579) == 578):
        ##    count+=1    
            #plt2.savefig('../Images/mybfigdv_'+str(count))

#f, ptarr = plt2.subplots(12, sharex=True)
#    for it in data:
#        idx+=1 
#        count+=1
#        if ((count %579) == 0):
#            ptidx+=1
#        ptarr[ptidx].plot(it,color=colour[cl[idx-1]])        
#    plt2.savefig('../Images/myfig')

if __name__ == "__main__":
    
    filepath='../Repo/'
#    imp.reload(IHPPlanner)
    imp.reload(logging)
    
    
    
    FORMAT = '%(levelname)s - %(asctime)s - %(filename)s::%(funcName)s - %(message)s'
    #logging.basicConfig(level=logging.DEBUG, format = '%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(message)s')
    logging.basicConfig(level=logging.DEBUG, format = FORMAT)
    logger = logging.getLogger("tzzipper")
    
    handler = logging.FileHandler('hello.log')
    handler.setLevel(logging.DEBUG)
    # create a logging format
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    
    logger.info ("Starting process..")
    
    hpp = TZZipper.TZZipper("mytest")
    data = hpp.initialize(loadFileData (filepath))
            
    saveFileData (filepath,data)
    #hpp.clusterize (4,data)
    cl = hpp.clusterizeHClust (data)
#    auxPlotter(data,cl)    de momentono hago las graficas
    calculaStadist(data,cl)
        
    logger.debug("Process ended...")
    