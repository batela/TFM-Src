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
                
# the histogram of the data
     plt2.close()
     n, bins, patches = plt2.hist(data, 10, facecolor='green', alpha=0.75)
     plt2.show()
     return n, bins, patches 




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


def doForecasting ():
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
        d_values = range(0, 3)
        q_values = range(0, 3)
        warnings.filterwarnings("ignore")
        evaluate_models(data, p_values, d_values, q_values)

def doSeriesForecasting ():
        
        
        tcs = TCSeries.TCSeries("mytest",coNames)
        fulldata = loadFileDataWithTime(filepath)
        aggData = fulldata.resample('4H').mean()
        
        data_log = tcs.initialize(aggData)
        tcs.checkStationarity(data_log['Pbld'])
#        plt2.subplot(411)
#        plt2.plot (data_log)
        
        
        plt2.subplot(121) 
        plt2.plot(aggData)
        
        plt2.subplot(122)
        plt2.plot(data_log)
        
        plt2.savefig('../Images/ts_base')
        plt2.close()
        
        
        expwighted_avg = pd.ewma(data_log, halflife=12)
        
        #data_log_diff = data_log - data_log.shift()
        data_log_diff = data_log - expwighted_avg
        data_log_diff.dropna(inplace=True)
        tcs.checkStationarity(data_log_diff['Pbld'])
        #plt2.subplot(411)
        #plt2.plot (data_log_diff)
        
        
        #trend,season,residual = tcs.doForecasting(data_log)
        residual,lag_acf,lag_pacf = tcs.doForecasting(data_log)
        tcs.checkStationarity(residual['Pbld'])
        #Plot ACF: 
        plt2.subplot(121) 
        plt2.plot(lag_acf)
        plt2.axhline(y=0,linestyle='--',color='gray')
        plt2.axhline(y=-1.96/numpy.sqrt(len(residual)),linestyle='--',color='gray')
        plt2.axhline(y=1.96/numpy.sqrt(len(residual)),linestyle='--',color='gray')
        plt2.grid()
        plt2.title('Autocorrelation Function')
        
        plt2.subplot(122)
        plt2.plot(lag_pacf)
        plt2.axhline(y=0,linestyle='--',color='gray')
        plt2.axhline(y=-1.96/numpy.sqrt(len(residual)),linestyle='--',color='gray')
        plt2.axhline(y=1.96/numpy.sqrt(len(residual)),linestyle='--',color='gray')
        plt2.title('Partial Autocorrelation Function')
        plt2.tight_layout()
        plt2.grid(color='r', linestyle='-', linewidth=2)
        
        plt2.savefig('../Images/ts')
        plt2.close()
        
        model = ARIMA(residual, order=(2, 1, 0))  
        results_AR = model.fit(disp=-1)  
        plt2.plot(residual)
        plt2.plot(results_AR.fittedvalues, color='red')
        #plt2.title('RSS: %.4f'% sum((results_AR.fittedvalues-residual)**2))
        plt2.savefig('../Images/ts_AR')
        plt2.close()

        
        model = ARIMA(residual, order=(0, 1, 2))  
        results_MA = model.fit(disp=-1)  
        plt2.plot(residual)
        plt2.plot(results_MA.fittedvalues, color='red')
        #plt2.title('RSS: %.4f'% sum((results_MA.fittedvalues-residual)**2))
        plt2.savefig('../Images/ts_MA')
        plt2.close()

        model = ARIMA(data_log, order=(2, 1, 2))  
        results_ARIMA = model.fit(disp=-1)  
        plt2.plot(residual)
        plt2.plot(results_ARIMA.fittedvalues, color='red')
        #plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))        
        
        
        plt2.savefig('../Images/ts_ARMA')
        plt2.close()

#        logger.info ("Checking stationary trend....") 
#        trend.dropna(inplace=True)
#        tcs.checkStationarity(trend['Pbld'])    
#        plt2.subplot(412)
#        plt2.plot (trend)
        
#        logger.info ("Checking stationary season....")   
#        season.dropna(inplace=True)
#        tcs.checkStationarity(season['Pbld'])    
#        plt2.subplot(413)
#        plt2.plot (season)
#        residual.dropna(inplace=True)
#        logger.info ("Checking stationary residual....")         
#        tcs.checkStationarity(residual['Pbld'])    
#        plt2.subplot(414)
#        plt2.plot (residual)
        #plt2.show()
        
        

    
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
#    doForecasting();
    doSeriesForecasting();
    logger.debug("Process ended...")
    