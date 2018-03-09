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

## Esto no debería estar aqui.... pero por simplificar..
period = 10
days=0
k = 4

def loadFileData (filepath):
                
        logger.info ("Loading data from file....")         

        filename = 'FHP-20141008.csv'                
        filename = 'datosvivienda_test.csv'
        filename = 'ed700.csv'        
                
        fulldata = pd.read_csv(filepath+filename, sep=';', decimal = '.')
        
        logger.debug ("Looking for proper daytypes..")         
        data = fulldata[(pd.to_datetime(fulldata['Time']).dt.weekday < 5)]

        logger.info ("Loaded data from file: " + filepath+filename)   
        return data
    

def saveFileData (filepath,data):
                
        logger.info ("Saving data to file....")
        filename = 'pydata.csv'        
        numpy.savetxt(filepath+filename, data, fmt='%.4f', delimiter=';')
        
        return data

def calculaStadist(data,cl):
        
        rooms = 12
        dias = len(cl)//(rooms)
        
        idx = -1
        di = None
        for ro in list(range (0,rooms)):
            idx +=1
            l = list(cl[dias*idx:dias*(idx+1)])
            for it in list(range (1,numpy.amax(cl)+1)):
                di = dict((it,(l.count(it)*100//dias)) for it in set(l))
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
    data = hpp.initialize(loadFileData (filepath),k,period,days) # Para RV 3,15,17 Para 700 7,10,0
            
    saveFileData (filepath,data)
    #hpp.clusterize (4,data)
    cl = hpp.clusterizeHClust (data)
#    auxPlotter(data,cl)    de momentono hago las graficas
    calculaStadist(data,cl)
        
    logger.debug("Process ended...")
    