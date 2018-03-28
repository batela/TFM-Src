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
import TCPredictor
import TCSeries

from matplotlib import pyplot as plt2
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.arima_model import ARIMA


## Esto no debería estar aqui.... pero por simplificar..
## BTL Variables de entrada
period = 15 ## Periodo entre datos en minutos
days=0      ## Si 0 se toman todos los datos del fichero, sino lo "dias" primeros
k = 4       ## Numero de clusteres 
rooms = 8   ## Numero de zonas/estancias
roomNames =['R0T','R1','R2','R3','R4','R5','R6','R7T']
#roomNames =['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11']    


coNames =['Toutdoor','Tindoor','Fbld','Pbld']        
coSeriesNames =['Control','Pbld']  

## BTL Variables de salida
distances   = None ## Matrix de distancias por dia
clusteres   = None ## Clusters encontrados 
distribut   = None ## Distribcion de cada zona en clusteres


"""
Funciones auxiliares para la lectura de datos
"""


def loadFileData (filepath,rooms):
                
        logger.info ("Loading data from file....")         
#        filename = 'ed700.csv'
#        filename = 'FHP-20141008.csv'                
        filename = 'datosvivienda_testw.csv'
                
        fulldata = pd.read_csv(filepath+filename, sep=',', decimal = '.')
        
        logger.debug ("Looking for proper daytypes..")         
#        data = fulldata[(pd.to_datetime(fulldata['Time']).dt.weekday < 5)]
        data = fulldata
        logger.info ("Loaded data from file: " + filepath+filename)   
        return data
    

def loadFileDataWithTime (filepath):
                
        logger.info ("Loading data from file....")         

#        filename = 'ed700.csv'
#        filename = 'FHP-20141008.csv'                
        filename = 'datosvivienda_testwc.csv'
                
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M:%S')
        fulldata = pd.read_csv(filepath+filename, sep=',', decimal = '.', index_col='Control',date_parser=dateparse ,usecols=coSeriesNames)
    
        logger.debug ("Looking for proper daytypes..")         
        #data = fulldata[(pd.to_datetime(fulldata.index).dt.weekday < 5)]

        logger.info ("Loaded data from file: " + filepath+filename)   
        return fulldata



def saveFileData (filepath,data):
                
        logger.info ("Saving data to file....")
        filename = 'pydata.csv'        
        numpy.savetxt(filepath+filename, data, fmt='%.4f', delimiter=';')
        
        return data

def groupByDays(data):
        
        logger.info ("Starting groupByDays")
        dias = len(data)//(rooms)
        dayItems = (24*60//period) 
        dayzones = numpy.zeros((dias, rooms, dayItems))
        
        logger.debug ("Creating aggrupation")       
        for idD in range (0,dias):
            for idR in range (0,rooms):
#                logger.debug ("Values:" + str( data[(dias*idR)+idD]))
                dayzones[idD][idR] =  data[(dias*idR)+idD]
        
        logger.debug ("Calculating distance matrix")       
        distances = numpy.zeros((dias, rooms, rooms))
        for idD in range (0,dias):
            distances[idD] = squareform(pdist(dayzones[idD], 'seuclidean', V=None) )       
        
        logger.info ("Finish groupByDays")

    
def calculaStadist(data,cl):
        
        dias = len(cl)//(rooms)
        idx = -1
        for ro in list(range (0,rooms)):
            idx +=1
            l = list(cl[dias*idx:dias*(idx+1)])
            for it in list(range (1,numpy.amax(cl)+1)):
                distribut = dict((it,(l.count(it)*100//dias)) for it in set(l))
            logger.info ("Found data for room " + str (ro) +" : "+ str(distribut))


"""
Funciones auxiliares para plotear..
"""
                
def auxPlotter(data,cl):
                
        idx = -1
        idxRo = 0
        count = -1;
        ptidx = -1
        dias = len(cl)//(rooms)
        colour=['blue','green','red','orange','cyan','black','pink','magenta']
        
        #f, ptarr = plt2.subplots(rooms, sharex=True)
        
        plt2.close()
        for it in data:
            idx+=1 
            count+=1
            if ((count %dias) == 0):
                ptidx+=1
            #ptarr[ptidx].plot(it,color=colour[cl[idx-1]])
            plt2.plot(it,color=colour[cl[idx-1]])        
            ## Prints a chart per room
            if ((idx %dias) == (dias-1)):   
                plt2.savefig('../Images/zone_'+str(idxRo))
                plt2.close()
                idxRo+=1

def auxPlotterHisto(data):                
     plt2.close()
     n, bins, patches = plt2.hist(data, 10, facecolor='green', alpha=0.75)
     plt2.show()
     return n, bins, patches 

def doPlotSingleToFile (data, fname,title):
        
        plt2.plot(data)
        plt2.grid()
        plt2.title(title)
    
        plt2.savefig('../Images/'+fname)
        plt2.close()
        

def doPlotDoubleToFile (data1,data2, fname,title):
        
        plt2.plot(data1)
        plt2.plot(data2)
        plt2.grid()
        plt2.title(title)
    
        plt2.savefig('../Images/'+fname)
        plt2.close()


def doMultizone ():
        ## BTL creamos la clase que utilizaremos    
        hpp = TZZipper.TZZipper("mytest",roomNames)
        
    ## BTL Inicializacion y creacion de estructura de datos que utilizaremos.
    ## "data" es una matriz en el que se ordenan por cada zona los datos corres-
    ## pondientes a sus "dias" de forma consecutiva. Es decir las primeras n filas 
    ## pertenecen a los n "dias" de la primera zona
        data = hpp.initialize(loadFileData (filepath),k,period,days) # Para RV 3,15,17 Para 700 7,10,0
    
    ## BTL funcion auxiliar            
        saveFileData (filepath,data)
    ## BTL funcion que calcula la distancia por dia para cada una de las zonas
        groupByDays (data)
        
    #   hpp.clusterize (4,data)
    
    ## BTL realiza la clusterizacion , plotea y calculo de estadisticas...
        clusteres = hpp.clusterizeHClust (data)
        auxPlotter(data,clusteres)    
        calculaStadist(data,clusteres)


def doClassForecasting ():
        hpf = TCPredictor.TCPredictor("mytest",coNames)
        data = hpf.initialize(loadFileData(filepath,coNames),period,30)
        X,y= hpf.integrate(data,4)
        n, bins, patches = auxPlotterHisto(y)
        yl = hpf.labelize(y,n, bins, patches)
        hpf.doForecasting( X,yl)
        hpf.TCPloter (X, yl)
        
       # hpf.TCPloter (data)


def doSelectBestARIMA  (tcs,data):
        p_values = [0, 1, 2, 4, 6, 8, 10]
        
        d_values = range(0,3)
        q_values = range(0,3)
        best_sol =tcs.evaluate_models(data['Pbld'], p_values, d_values, q_values)
        p = best_sol[1][0]
        d = best_sol[1][1]
        q = best_sol[1][2]
        
        return p,d,q


def doTimeSeriesForecasting ():
        
        
        tcs = TCSeries.TCSeries("mytest",coNames)
        fulldata = loadFileDataWithTime(filepath)
        aggData = fulldata.resample('4H').mean()
        
        data_log = tcs.initialize(aggData)
        tcs.checkStationarity(data_log['Pbld'])
#        plt2.subplot(411)
#        plt2.plot (data_log)
        
# Estas graficas solo representan los valores medios 
# contra el algoritmos de los mismo        
        
        doPlotSingleToFile (aggData,"ts_base","Base data ")
        doPlotSingleToFile (data_log,"ts_base_log","Logarimic data")

# Calculo la media pondera realizar la diferencia y verificar
# si esta diferencia es una serie estacionaria        
        expwighted_avg = pd.ewma(data_log, halflife=12)
        
        #data_log_diff = data_log - data_log.shift()
        data_log_diff = data_log - expwighted_avg
        data_log_diff.dropna(inplace=True)
        tcs.checkStationarity(data_log_diff['Pbld'])
        #plt2.subplot(411)
        #plt2.plot (data_log_diff)
        
        
# Calulo las funciones de autocorrelacion y autocorrelacion parciase.
# ademas la funcion me devuelve la parte residuo de los datos
# tras aplicarles el logaritmos. Los valores optimos deberian ser aquellos
# que en la grafica cortan con 0.2 la primera vez        
        #trend,season,residual = tcs.doForecasting(data_log)
        residual,lag_acf,lag_pacf = tcs.doForecasting(data_log)
        tcs.checkStationarity(residual['Pbld'])
        #Plot ACF: 
        doPlotSingleToFile (lag_acf,"ts_ac","Autocorrelation function")
        doPlotSingleToFile (lag_pacf,"ts_pac","Partial Autocorrelation function")
        
# Trato de determinar cuales son los mejore valore de p,q y d
# OJOOOO !!! Esta funcion tarda mucho....
        p,d,q =doSelectBestARIMA  (tcs,residual)
               
        model = ARIMA(residual, order=(p,d,q))  
        results_AR = model.fit(disp=-1)  
        doPlotDoubleToFile (residual,results_AR.fittedvalues,"ts_AR","AR model")
        #plt2.title('RSS: %.4f'% sum((results_AR.fittedvalues-residual)**2))
        
        model = ARIMA(residual, order=(p,d,q))  
        results_MA = model.fit(disp=-1)
        doPlotDoubleToFile (residual,results_MA.fittedvalues,"ts_MA","MA model")
        
        model = ARIMA(residual, order=(p,d,q))  
        results_ARIMA = model.fit(disp=-1)  
        doPlotDoubleToFile (residual,results_ARIMA.fittedvalues,"ts_MA","MA model")
        
    
if __name__ == "__main__":
    
    filepath='../Repo/'
    imp.reload(TZZipper)
    imp.reload(TCPredictor)
    imp.reload(TCSeries)
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
    

#    doMultizone();
#    doClassForecasting();
    doTimeSeriesForecasting();
    logger.debug("Process ended...")
    