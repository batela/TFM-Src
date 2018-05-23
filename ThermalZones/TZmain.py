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
from sklearn.preprocessing import MinMaxScaler

## Esto no debería estar aqui.... pero por simplificar..
## BTL Variables de entrada
period = 15 ## Periodo entre datos en minutos
days=0      ## Si 0 se toman todos los datos del fichero, sino lo "dias" primeros
k = 4       ## Numero de clusteres 
rooms = 8   ## Numero de zonas/estancias
roomNames =['R0T','R1','R2','R3','R4','R5','R6','R7T']
#roomNames =['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11']    


coNames =['Toutdoor','Tindoor','Fbld','Pbld']        
coSeriesNames =['Control','Pbld','Toutdoor','Tindoor']  

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
        filename = 'datosvivienda_test.csv'
                
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
        filename = 'datosvivienda_tfm.csv'
                
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

## BTL Calcula una matriz tridiemnsionale en la cadauna de los planos YZ representa un día,
## para cada día se disponen de 96 valores, se calcula la distancia de dichos arrays.
        for idD in range (0,dias):
            distances[idD] = squareform(pdist(dayzones[idD], 'seuclidean', V=None) )       
        
        distancesAvg = numpy.zeros((rooms,rooms))
        for idR in range (0,rooms):
            distancesAvg[idR] = distances[:,idR,:].mean(0)       
        
        logger.info ("Finish groupByDays")
        return distancesAvg
    
    
    
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
Funciones auxiliares para plotear..si se quiere


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

## BTL: Habia in problema con el indice, se restaba 1 a idx,
## no hay que hacerlo, estaba cogiendo un valor que no era            
            plt2.plot(it,color=colour[cl[idx]])        
            ## Prints a chart per room
            if ((idx %dias) == (dias-1)):   
                plt2.savefig('../Images/zone_'+str(idxRo))
                idxRo+=1
                plt2.close()
                

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


def doPlotDistance (data, fname,title):
        
        plt2.matshow(data,cmap="Reds")
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
        data = hpp.initialize(loadFileData (filepath,roomNames),k,period,days) # Para RV 3,15,17 Para 700 7,10,0
    
## BTL funcion auxiliar            
        saveFileData (filepath,data)
## BTL funcion que calcula la distancia por dia para cada una de las zonas y realiza
## el grafico correspondiente
        distances = groupByDays (data)
        doPlotDistance(distances,"dists","Distancias")
    #   hpp.clusterize (4,data)
    
## BTL realiza la clusterizacion , plotea y calculo de estadisticas...
        clusteres = hpp.clusterizeHClust (data)
        auxPlotter(data,clusteres)    
## BTL la funcion indica en que clusters se encuentra cada elemento y con 
## que probabilidad
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


#  BTL  esta funcion esta por terminar.
def doRegressionForecasting ():
        hpf = TCPredictor.TCPredictor("mytest",coNames)
        data = hpf.initialize(loadFileData(filepath,coNames),period,30)
        X,y= hpf.integrate(data,4)
        hpf.doForecastingRegressor( X,y)
        hpf.TCPloter (X, y)
        
       # hpf.TCPloter (data)


def doSelectBestARIMA  (tcs,data):
        p_values = [0, 1, 2, 4, 6, 8, 10]
        
        d_values = range(0,3)
        q_values = range(0,3)
        
        best_sol, best_sol_aic =tcs.evaluate_models(data, p_values, d_values, q_values)
        
        return best_sol, best_sol_aic


def doTimeSeriesARIMAXForecasting ():
        
# BTL: En primer termino instancio la clase TCSeries, que viene de
# ThermalComfortSeries... es decir tratamiento por series numericas
# del problema del comfort termico. De todo el dataset solo voy a utilizar
# las columnas  coSeriesNames =['Control','Pbld','Toutdoor']     
        tcs = TCSeries.TCSeries("mytest",coSeriesNames)

# BTL: Cargamos los datos en el dataframe y realizamos un resample en periodos
# de cuatro horas tomando la media        
        fulldata = loadFileDataWithTime(filepath)
        aggData = fulldata.resample('4H').mean()

# BTL: Inicializamos el objeto TCSeries esta funcion ademas calcula
# el logaritmo de los valores, es una forma de penalizar los valores
# mas altos y homogeneizar la serie. Posteriormente verificamos si 
# estamos ante una serie estacionaria         
        data_log = tcs.initialize(aggData)
        tcs.checkStationarity(data_log['Pbld'])
 
                
        residual,lag_acf,lag_pacf = tcs.doForecasting(data_log)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        dataAR = pd.DataFrame()
        dataAR['ToutdoorRef'] = aggData['Toutdoor']
        dataAR['Pbld'] = aggData['Pbld']
        
## BTL Realizamos el estudio sin normalizar los valores
        p = 8 #10
        d = 0
        q = 2 #0
        model = ARIMA(endog=dataAR['Pbld'],exog=dataAR['ToutdoorRef'] ,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
        doPlotDoubleToFile (dataAR['Pbld'],results_AR.fittedvalues,"ts_ARIMAX_"+str(p)+str(d)+str(q),"ARIMAX model")

## BTL Realizamos el estudio habiendo normalizadoo los valores
                
        scaler = scaler.fit(dataAR['Pbld'].reshape(-1,1))
        normalizedPbld = scaler.transform(dataAR['Pbld'].reshape(-1,1))
        
        scaler = scaler.fit(dataAR['ToutdoorRef'].reshape(-1,1))
        normalizedOutdoor = scaler.transform(dataAR['ToutdoorRef'].reshape(-1,1))
        
        p = 8 #10
        d = 0
        q = 2 #0
        model = ARIMA(endog=normalizedPbld,exog=normalizedOutdoor,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
        doPlotDoubleToFile (normalizedPbld,results_AR.fittedvalues,"ts_ARIMAX_Normaliz_"+str(p)+str(d)+str(q),"ARIMAX model")
        


    
def doTimeSeriesForecasting ():
        
# BTL: En primer termino instancio la clase TCSeries, que viene de
# ThermalComfortSeries... es decir tratamiento por series numericas
# del problema del comfort termico. De todo el dataset solo voy a utilizar
# las columnas  coSeriesNames =['Control','Pbld']     
        tcs = TCSeries.TCSeries("mytest",coSeriesNames)

# BTL: Cargamos los datos en el dataframe y realizamos un resample en periodos
# de cuatro horas tomando la media        
        fulldata = loadFileDataWithTime(filepath)
        aggData = fulldata.resample('4H').mean()

# BTL: Inicializamos el objeto TCSeries esta funcion ademas calcula
# el logaritmo de los valores, es una forma de penalizar los valores
# mas altos y homogeneizar la serie. Posteriormente verificamos si 
# estamos ante una serie estacionaria         
        data_log = tcs.initialize(aggData)
        tcs.checkStationarity(data_log['Pbld'])
        
# BTL: Estas graficas solo representan los valores medios 
# contra el logarimo de los mismos, solo tiene propositos ilustarativos
                
        doPlotSingleToFile (aggData,"ts_base","Base data ")
        doPlotSingleToFile (data_log,"ts_base_log","Logarimic data")

# BTL: Otro calculo auxiliar,Calculo la media pondera de manera exponencial
# Exponentially Weighted Moving Average realizar la diferencia y verificar
# si esta diferencia es una serie estacionaria, probamos a coger una ventana
# de un dia es decir como hemos agrupado cada 4h cogemos halflife de 6        
        expwighted_avg = pd.ewma(data_log, halflife=6)
        
        #data_log_diff = data_log - data_log.shift()
        data_log_diff = data_log - expwighted_avg
        data_log_diff.dropna(inplace=True)
        tcs.checkStationarity(data_log_diff['Pbld'])
        

# BTL:Calulo las funciones de autocorrelacion y autocorrelacion parciase.
# ademas la funcion me devuelve la parte residuo de los datos
# tras aplicarles el logaritmos. Los valores optimos deberian ser aquellos
# que en la grafica cortan con 0.2 la primera vez, se trata de un metodo
# para identificar los valores de p-q-d a aplicar al modelo ARIMA. Se propone
# aplicar el modelo ARIMA a la parte residual 
#trend,season,residual = tcs.doForecasting(data_log)
        
        residual,lag_acf,lag_pacf = tcs.doForecasting(data_log)
        tcs.checkStationarity(residual)
        #Plot ACF: 
        doPlotSingleToFile (lag_acf,"ts_ac","Autocorrelation function")
        doPlotSingleToFile (lag_pacf,"ts_pac","Partial Autocorrelation function")

        # BTL: Trato de determinar cuales son los mejore valore de p,q y d de forma iterativa
# es decir prueba-error. Este procedimiento puede tardar mucho
# OJOOOO !!! Esta funcion tarda mucho....

        best_sol_mse,best_sol_aic =doSelectBestARIMA  (tcs,residual)
# BTL realizamos la grafica con la mejor opcion segun el metodo AIC
        p = best_sol_aic[1][0]
        d = best_sol_aic[1][1]
        q = best_sol_aic[1][2]
        model = ARIMA(residual, order=(p,d,q))  
        results_AR = model.fit(disp=-1)
        doPlotDoubleToFile (residual,results_AR.fittedvalues,"ts_ARIMA_"+str(p)+str(d)+str(q),"ARIMA model")

# BTL realizamos la grafica con la mejor opcion segun el metodo MSE                
        p = best_sol_mse[1][0]
        d = best_sol_mse[1][1]
        q = best_sol_mse[1][2]
        model = ARIMA(residual, order=(p,d,q))  
        results_AR = model.fit(disp=-1)  
        doPlotDoubleToFile (residual,results_AR.fittedvalues,"ts_ARIMA_"+str(p)+str(d)+str(q),"ARIMA model")
 
if __name__ == "__main__":
    
    filepath='../Repo/'
    imp.reload(TZZipper)
    imp.reload(TCPredictor)
    imp.reload(TCSeries)
    imp.reload(logging)
    
# BTL Inicializar le log    

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

# BTL Comienza la ejecucion en sí
    
    logger.info ("Starting process..")
## La funcion multizona realiza la clusterizacion ademas de 
## realizar las graficas
##    doMultizone();
#    doClassForecasting();

# BTL: Modelado por series temporales tradicionales ARIMA    
#    doTimeSeriesForecasting()
# BTL: Incorporando una variable exogena, utilizo los mismos valores
# pdq que he obtenido anteriormente
    doTimeSeriesARIMAXForecasting()
    
    logger.debug("Process ended...")
    