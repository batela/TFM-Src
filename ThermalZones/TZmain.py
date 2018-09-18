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
import matplotlib.dates as mdates
import IndoorModel
import auxTFM as ptfm

from matplotlib import pyplot as plt2
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from numpy.polynomial import polynomial as P

## BTL para el tratamiento de errores
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


### Esto no debería estar aqui.... pero por simplificar..
### BTL Variables de entrada
#period = 15 ## Periodo entre datos en minutos
#days=0      ## Si 0 se toman todos los datos del fichero, sino lo "dias" primeros
#k = 4       ## Numero de clusteres 
#rooms = 8   ## Numero de zonas/estancias
#roomNames =['R0T','R1','R2','R3','R4','R5','R6','R7T']
##roomNames =['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11']    
#
#
coNames =['Toutdoor','Tindoor','Fbld','Pbld']        
coSeriesNames =['Control','Pbld','Toutdoor','Tindoor']  

filepath='../Repo/'



def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100


def doSelectBestARIMA  (tcs,data):
    
    p_values = [0, 1, 2, 4, 6, 8, 10] 
    d_values = range(0,2)
    q_values = range(0,3)
    
    best_sol, best_sol_aic =tcs.evaluate_models(data, p_values, d_values, q_values) 
    return best_sol, best_sol_aic


def calculateTrendModel (decomposition):
    logger.debug("Comienza calculo de tendencia")
    tendencia = decomposition.trend.dropna()
    X = numpy.array(range (0,len(tendencia.index)))
    Xoutbound =numpy.append(X,[162,163,164,165,166,167,168])
    coeff, stats = P.polyfit(X,tendencia.values,9,full=True)
    fitpoly = P.Polynomial(coeff)
    auxPoli = pd.DataFrame()
    auxPoli["vals"]=fitpoly(X)
    auxPoli = auxPoli.set_index(pd.DatetimeIndex(tendencia.index))
    
    ptfm.doPlotDoubleToFileDate (auxPoli,tendencia,"ts_OLS_InBound_POLI","POLINOMIAL Inbound Prediction","Modelo","Real")
    mse = mean_squared_error(auxPoli,tendencia)
    mae = mean_absolute_error(auxPoli,tendencia)
    mape = mean_absolute_percentage_error(auxPoli,tendencia)    
        
#    doPlotDoubleToFile (tendencia.values,fitpoly(Xoutbound),"ts_OLS_Outbound_POLI","POLINOMIAL Outbound Prediction")
    
    X = sm.add_constant(X)
    Xoutbound = sm.add_constant(Xoutbound)
    #X=X.reshape(-1,1)
    #tendencia=tendencia.values.reshape(-1,1)
    olsmodel = sm.OLS(numpy.array(tendencia.values),X)
    fitols = olsmodel.fit()
    auxOLS = pd.DataFrame()
    auxOLS["vals"]=fitols.predict(X)
    auxOLS = auxOLS.set_index(pd.DatetimeIndex(tendencia.index))
    
    mse = mean_squared_error(auxOLS,tendencia)
    mae = mean_absolute_error(auxOLS,tendencia)
    mape = mean_absolute_percentage_error(auxOLS,tendencia)    
    
    ptfm.doPlotDoubleToFileDate (auxOLS,tendencia,"ts_OLS_InBound","OLS Inbound Prediction","Modelo","Real")
#    doPlotDoubleToFile (tendencia.values,fitols.predict(Xoutbound),"ts_OLS_OutBound","OLS Outbound Prediction")
    
    logger.debug("Fin calculo de tendencia")

    return fitols.predict(Xoutbound),fitpoly(Xoutbound),auxOLS, auxPoli


def doTimeSeriesARIMAXForecastingWithResiduals ():
        
        p = 8 #10
        d = 0
        q = 2 #0
        
# BTL: En primer termino instancio la clase TCSeries, que viene de
# ThermalComfortSeries... es decir tratamiento por series numericas
# del problema del comfort termico. De todo el dataset solo voy a utilizar
# las columnas  coSeriesNames =['Control','Pbld','Toutdoor']     
        tcs = TCSeries.TCSeries("mytest",coSeriesNames)

# BTL: Cargamos los datos en el dataframe y realizamos un resample en periodos
# de cuatro horas tomando la media        

        fulldata    = ptfm.loadFileDataWithTime(filepath,'datosvivienda_testwc.csv')
        predictdata = ptfm.loadFileDataWithTime(filepath,'datosvivienda_testwc_predict_largo.csv')
        
        
        aggData = fulldata.resample('4H').mean()
        aggPredictData = predictdata.resample('4H').mean()
        tt = aggPredictData["Toutdoor"][aggPredictData["Toutdoor"]>100]=aggPredictData["Toutdoor"]/100
        
# BTL: Inicializamos el objeto TCSeries esta funcion ademas calcula
# el logaritmo de los valores, es una forma de penalizar los valores
# mas altos y homogeneizar la serie. Posteriormente verificamos si 
# estamos ante una serie estacionaria         
        #data_log = tcs.initialize(aggData)
        
        outAddFuller = tcs.checkStationarity(aggData['Pbld'])
        logger.debug("El test de fuller indica que es estacionario: " + str (outAddFuller[0]))
        outAddFuller = tcs.checkStationarity(numpy.log(aggData['Pbld']))
        logger.debug("El test de fuller indica que es estacionario: " + str (outAddFuller[0])) 
        
        decomposition = seasonal_decompose(numpy.log(aggData['Pbld']))
        
        fitols, fitpoly,olsInB, poliOutB = calculateTrendModel(decomposition)
        
        residual = decomposition.resid
        residual.dropna(inplace=True)
        
        PackAggData = pd.DataFrame()
        PackAggData=aggData[aggData.index.isin(residual.index)]
        outAddFuller = tcs.checkStationarity(residual)
        
        logger.debug("El test de fuller indica que es estacionario: " +str (outAddFuller[0]))
        
        
        plot_acf(residual,lags=50)
        plt2.savefig('../Images/'+"ts_ac_res")
        plt2.close()

        plot_pacf(residual, lags=50)
        plt2.savefig('../Images/'+"ts_pac_res")
        plt2.close()

        scaler = MinMaxScaler(feature_range=(0, 1))

## BTL de aqui en adelante se usan la estructuras dataAR y nos quedamos con el residuo      
        dataAR = pd.DataFrame()
        dataAR['ToutdoorRef'] = PackAggData['Toutdoor']
        dataAR['Pbld'] = residual
        dataAR['ToutdoorRef'][abs(dataAR['ToutdoorRef']>100)]=0 
        dataARValid = dataAR['Pbld'][dataAR.index>='2017-03-05']
        dataARValid= dataARValid[dataARValid.index<='2017-03-06']
        
        model = ARIMA(endog=dataAR['Pbld'],exog=dataAR['ToutdoorRef'] ,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
        ptfm.doPlotDoubleToFileDate (results_AR.fittedvalues,dataAR['Pbld'],"ts_ARIMAX_RESIDUAL_"+str(p)+str(d)+str(q),"ARIMAX model RESIDUAL","Modelo","Real")
        
        mse = mean_squared_error(dataAR['Pbld'],results_AR.fittedvalues)
        mae = mean_absolute_error(dataAR['Pbld'],results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(dataAR['Pbld'],results_AR.fittedvalues)    
        
        pred = results_AR.predict(start="2017-03-01", end="2017-03-05",exog=tt, dynamic=False)
        ptfm.doPlotDoubleToFileDate ( pred , dataAR['Pbld'][dataAR.index<='2017-03-05'] ,"Predict_ts_ARIMAX_RESIDUAL_InBound_"+str(p)+str(d)+str(q),"Predict ARIMAX model log(Pbld)","Modelo","Real")
#        Esto filtra solo los dias de validacion
        mask = (dataAR.index<='2017-03-05') & (dataAR.index>='2017-03-01')
        mape = mean_absolute_percentage_error(dataAR.loc[mask]['Pbld'],pred)
        mse = mean_squared_error(dataAR.loc[mask]['Pbld'],pred)
        mae = mean_absolute_error(dataAR.loc[mask]['Pbld'],pred)
        ptfm.doPlotDoubleToFileDate ( pred ,dataAR.loc[mask]['Pbld'],"Predict_ts_ARIMAX_RESIDUAL_InBound_Validation_"+str(p)+str(d)+str(q),"Validation ARIMAX model log(Pbld)","Modelo","Real")
        
    
        olsInB = olsInB[mask] 
        poliOutB = poliOutB[mask]
        auxdataARValid = PackAggData['Pbld'][mask]
        ptfm.doPlotDoubleToFileDate ( numpy.exp(poliOutB.vals+pred),auxdataARValid,"Predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY_"+str(p)+str(d)+str(q),"Full predict Predict ARIMAX model Pbld - Residuals","Modelo","Real")
        predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY = numpy.exp(poliOutB.vals+pred)
        predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY_Real = auxdataARValid
        
        mape = mean_absolute_percentage_error(numpy.exp(poliOutB.vals+pred),auxdataARValid)
        mse = mean_squared_error(numpy.exp(poliOutB.vals+pred),auxdataARValid)
        mae = mean_absolute_error(numpy.exp(poliOutB.vals+pred),auxdataARValid)
        
        
        pred = results_AR.predict(start="2017-03-05", end="2017-03-06",exog=tt, dynamic=True)
        ptfm.doPlotDoubleToFile ( pred ,dataARValid,"Predict_ts_ARIMAX_RESIDUAL_OutBound_"+str(p)+str(d)+str(q),"Predict ARIMAX model log(Pbld)")
        

## BTL Realizamos el estudio habiendo normalizadoo los valores
                
        scaler = scaler.fit(dataAR['Pbld'].reshape(-1,1))
        normalizedPbld = scaler.transform(dataAR['Pbld'].reshape(-1,1))
        
        scaler = scaler.fit(dataAR['ToutdoorRef'].reshape(-1,1))
        normalizedOutdoor = scaler.transform(dataAR['ToutdoorRef'].reshape(-1,1))
        
        model = ARIMA(endog=normalizedPbld,exog=normalizedOutdoor,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
#        doPlotDoubleToFileDate (results_AR.fittedvalues,normalizedPbld,"ts_ARIMAX_RESIDUAL_NORMALIZADO_"+str(p)+str(d)+str(q),"ARIMAX model RESIDUAL NORMALIZADO","Modelo","Real")
        
        mse = mean_squared_error(normalizedPbld,results_AR.fittedvalues)
        mae = mean_absolute_error(normalizedPbld,results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(normalizedPbld,results_AR.fittedvalues)    
        
        
###################################################################
## BTL ahora realizamos la convolucion y el suavizado
        
        dataAR['Pbld'] = tcs.smoothSerie (dataAR['Pbld'],3)
        dataARValid = dataAR['Pbld'][dataAR.index>='2017-03-05']
        dataARValid= dataARValid[dataARValid.index<='2017-03-06']
        
        model = ARIMA(endog=dataAR['Pbld'],exog=dataAR['ToutdoorRef'] ,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
        ptfm.doPlotDoubleToFileDate (results_AR.fittedvalues,dataAR['Pbld'],"ts_ARIMAX_RESIDUAL_SUAVIZADO_"+str(p)+str(d)+str(q),"ARIMAX model RESIDUAL SUAVIZADO","Modelo","Real")

        mse = mean_squared_error(dataAR['Pbld'],results_AR.fittedvalues)
        mae = mean_absolute_error(dataAR['Pbld'],results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(dataAR['Pbld'],results_AR.fittedvalues)    
        
        pred = results_AR.predict(start="2017-03-01", end="2017-03-05",exog=tt, dynamic=False)
#        Aqui ahora
        mask = (dataAR.index<='2017-03-05')
        ptfm.doPlotDoubleToFileDate ( pred , dataAR['Pbld'][mask] ,"Predict_ts_ARIMAX_RESIDUAL_SUAVIZADO_InBound_"+str(p)+str(d)+str(q),"Predict ARIMAX model Pbld - Residuals","Modelo","Real")
        mask = (dataAR.index<='2017-03-05') & (dataAR.index>='2017-03-01')
        ptfm.doPlotDoubleToFileDate ( pred , dataAR['Pbld'][mask] ,"Predict_ts_ARIMAX_RESIDUAL_SUAVIZADO_InBound_Validation_"+str(p)+str(d)+str(q),"Predict ARIMAX model Pbld - Residuals","Modelo","Real")
        
        
        mape = mean_absolute_percentage_error(dataAR['Pbld'][mask],pred)
        mse = mean_squared_error(dataAR['Pbld'][mask],pred)
        mae = mean_absolute_error(dataAR['Pbld'][mask],pred)
    
        auxDataARSmooth = pd.DataFrame()
        auxDataARSmooth ["vals"]=tcs.smoothSerie (auxdataARValid,3)
        auxDataARSmooth  = auxDataARSmooth.set_index(pd.DatetimeIndex(auxdataARValid.index))
    
    
        ptfm.doPlotDoubleToFileDate ( numpy.exp(poliOutB.vals+pred),auxDataARSmooth ,"Predict_ts_ARIMAX_RESIDUAL_SUAVIZADO_FULL_InBound_POLY_"+str(p)+str(d)+str(q),"Full predict Predict ARIMAX model Pbld - Residuals","Modelo","Real")
        mape = mean_absolute_percentage_error(numpy.exp(poliOutB.vals+pred),auxdataARValid)
        predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY = numpy.exp(poliOutB.vals+pred)
        predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY_Real = auxDataARSmooth
        mse = mean_squared_error(numpy.exp(poliOutB.vals+pred),auxdataARValid)
        mae = mean_absolute_error(numpy.exp(poliOutB.vals+pred),auxdataARValid)
        
        
        pred = results_AR.predict(start="2017-03-05", end="2017-03-06",exog=tt, dynamic=True)
        ptfm.doPlotDoubleToFile ( pred ,dataARValid,"Predict_ts_ARIMAX_RESIDUAL_SUAVIZADO_OutBound_"+str(p)+str(d)+str(q),"Predict ARIMAX model Pbld - Residuals")
       
        #fitols fitpoly
        dataARValid = PackAggData['Pbld'][PackAggData.index>='2017-03-05']
        dataARValid= dataARValid[dataARValid.index<='2017-03-06']
        
        ptfm.doPlotDoubleToFileDate ( numpy.exp(fitpoly[-7:,1]+pred),dataARValid,"Predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_POLY_"+str(p)+str(d)+str(q),"Full predict Predict ARIMAX model Pbld - Residuals","Modelo","Real")
        ptfm.doPlotDoubleToFileDate ( numpy.exp(fitols[-7:]+pred),dataARValid,"Predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_OLS_"+str(p)+str(d)+str(q),"Full predict Predict ARIMAX model Pbld - Residuals","Modelo","Real")
        predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_POLY = numpy.exp(fitpoly[-7:,1]+pred)
        predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_OLS = numpy.exp(fitols[-7:]+pred)
        predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_Real = dataARValid

## BTL Realizamos el estudio habiendo normalizadoo los valores
                
        scaler = scaler.fit(dataAR['Pbld'].reshape(-1,1))
        normalizedPbld = scaler.transform(dataAR['Pbld'].reshape(-1,1))
        
        scaler = scaler.fit(dataAR['ToutdoorRef'].reshape(-1,1))
        normalizedOutdoor = scaler.transform(dataAR['ToutdoorRef'].reshape(-1,1))

        model = ARIMA(endog=normalizedPbld,exog=normalizedOutdoor,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
        ptfm.doPlotDoubleToFile (normalizedPbld,results_AR.fittedvalues,"ts_ARIMAX_RESIDUAL_SUAVIZADO_NORMALIZADO_"+str(p)+str(d)+str(q),"ARIMAX model RESIDUAL NORMALIZADO SUAVIZADO")
        
        mse = mean_squared_error(normalizedPbld,results_AR.fittedvalues)
        mae = mean_absolute_error(normalizedPbld,results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(normalizedPbld,results_AR.fittedvalues)    
        
        return predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY,predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY_Real,predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_POLY,predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_OLS,predict_ts_ARIMAX_RESIDUAL_FULL_OutBound_Real


def doTimeSeriesARIMAXForecasting ():
        
# BTL: En primer termino instancio la clase TCSeries, que viene de
# ThermalComfortSeries... es decir tratamiento por series numericas
# del problema del comfort termico. De todo el dataset solo voy a utilizar
# las columnas  coSeriesNames =['Control','Pbld','Toutdoor']     
        tcs = TCSeries.TCSeries("mytest",coSeriesNames)

# BTL: Cargamos los datos en el dataframe y realizamos un resample en periodos
# de cuatro horas tomando la media                
        fulldata            = ptfm.loadFileDataWithTime(filepath,'datosvivienda_testwc.csv')
        predictdata         = ptfm.loadFileDataWithTime(filepath,'datosvivienda_testwc_predict_largo.csv')
        predictdataCorto    = ptfm.loadFileDataWithTime(filepath,'datosvivienda_testwc_predict.csv')
        
        
        aggData = fulldata.resample('4H').mean()
        aggPredictData = predictdata.resample('4H').mean()
        aggPredictDataCorto = predictdataCorto.resample('4H').mean()
        tt = aggPredictData["Toutdoor"][aggPredictData["Toutdoor"]>100]=aggPredictData["Toutdoor"]/100
        
# BTL: Inicializamos el objeto TCSeries esta funcion ademas calcula
# el logaritmo de los valores, es una forma de penalizar los valores
# mas altos y homogeneizar la serie. Posteriormente verificamos si 
# estamos ante una serie estacionaria         
        #data_log = tcs.initialize(aggData)
        
        outAddFuller = tcs.checkStationarity(aggData['Pbld'])
        logger.debug("El test de fuller indica que es estacionario: " + str(outAddFuller[0]))
        outAddFuller = tcs.checkStationarity(numpy.log(aggData['Pbld']))
        logger.debug("El test de fuller indica que es estacionario: " + str(outAddFuller[0])) 
                
        
        scaler = MinMaxScaler(feature_range=(0, 1))

## BTL de aqui en adelante se usan la estructuras dataAR        
        dataAR = pd.DataFrame()
        dataAR['ToutdoorRef'] = aggData['Toutdoor']
        dataAR['Pbld'] = numpy.log(aggData['Pbld'])
        dataAR['ToutdoorRef'][abs(dataAR['ToutdoorRef']>100)]=0 
        
## BTL Realizamos el estudio sin normalizar los valores
        p = 8 #10
        d = 0
        q = 2 #0
        model = ARIMA(endog=dataAR['Pbld'],exog=dataAR['ToutdoorRef'] ,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
#        doPlotDoubleToFile (dataAR['Pbld'],results_AR.fittedvalues,"ts_ARIMAX_"+str(p)+str(d)+str(q),"ARIMAX model")
        ptfm.doPlotDoubleToFileDate(results_AR.fittedvalues,dataAR['Pbld'],"ts_ARIMAX_"+str(p)+str(d)+str(q),"ARIMAX model","Modelo","Real")
        
        mse = mean_squared_error(dataAR['Pbld'],results_AR.fittedvalues)
        mae = mean_absolute_error(dataAR['Pbld'],results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(dataAR['Pbld'],results_AR.fittedvalues)    
        
        pred = results_AR.predict(start="2017-03-01", end="2017-03-08",exog=tt, dynamic=False)

        ptfm.doPlotDoubleToFile ( pred , dataAR['Pbld'][dataAR.index<='2017-03-08'],"Predict_ts_ARIMAX_Pbld_InBound_"+str(p)+str(d)+str(q),"Predict ARIMAX model log(Pbld)")
        
        
        pred = results_AR.predict(start="2017-03-08", end="2017-03-09",exog=tt, dynamic=True)
        ptfm.doPlotDoubleToFile ( pred ,numpy.log(aggPredictDataCorto["Pbld"]),"Predict_ts_ARIMAX_Pbld_OutBound_"+str(p)+str(d)+str(q),"Predict ARIMAX model log(Pbld)")
        ptfm.doPlotDoubleToFile ( numpy.exp(pred) ,(aggPredictDataCorto["Pbld"]),"Predict_ts_ARIMAX_Pbld_OutBound_REAL_"+str(p)+str(d)+str(q),"Predict ARIMAX model log(Pbld)")
        
        # Azul prediccicion
        
        
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
        ptfm.doPlotDoubleToFile (normalizedPbld,results_AR.fittedvalues,"ts_ARIMAX_NORMALIZADO_"+str(p)+str(d)+str(q),"ARIMAX model NORMALIZADO")
        
        mse = mean_squared_error(normalizedPbld,results_AR.fittedvalues)
        mae = mean_absolute_error(normalizedPbld,results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(normalizedPbld,results_AR.fittedvalues)    
        
        
###################################################################
## BTL ahora realizamos la convolucion y el suavizado
        
        dataAR['Pbld'] = tcs.smoothSerie (dataAR['Pbld'],3)
        outAddFuller = tcs.checkStationarity(dataAR['Pbld'])
        logger.debug("El test de fuller indica que es estacionario: " + str(outAddFuller[0]))
        
        
        ## BTL Realizamos el estudio sin normalizar los valores
        p = 8 #10
        d = 0
        q = 2 #0
        model = ARIMA(endog=dataAR['Pbld'],exog=dataAR['ToutdoorRef'] ,order=[p,d,q]) 
        results_AR = model.fit(disp=-1)  
        ptfm.doPlotDoubleToFileDate (results_AR.fittedvalues,dataAR['Pbld'],"ts_ARIMAX_SUAVIZADO_"+str(p)+str(d)+str(q),"ARIMAX model SUAVIZADO","Modelo","Real")

        mse = mean_squared_error(dataAR['Pbld'],results_AR.fittedvalues)
        mae = mean_absolute_error(dataAR['Pbld'],results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(dataAR['Pbld'],results_AR.fittedvalues)    
        
        pred = results_AR.predict(start="2017-03-01", end="2017-03-08",exog=tt, dynamic=False)
        ptfm.doPlotDoubleToFile ( pred , dataAR['Pbld'][dataAR.index<='2017-03-08'] ,"Predict_ts_ARIMAX_Pbld_SAUVIZADO_InBound_"+str(p)+str(d)+str(q),"Predict ARIMAX SUAVIZADO model log(Pbld)")
        
        
        pred = results_AR.predict(start="2017-03-08", end="2017-03-09",exog=tt, dynamic=True)
        aggPredictDataCorto["Pbld"] = tcs.smoothSerie (numpy.log(aggPredictDataCorto["Pbld"]),3)
        ptfm.doPlotDoubleToFile (pred, aggPredictDataCorto["Pbld"] ,"Predict_ts_ARIMAX_Pbld_SUAVIZADO_OutBound_"+str(p)+str(d)+str(q),"Predict ARIMAX SUAVIZADO model log(Pbld)")
        ptfm.doPlotDoubleToFile (numpy.exp(pred), numpy.exp(aggPredictDataCorto["Pbld"]) ,"Predict_ts_ARIMAX_Pbld_SUAVIZADO_OutBound_REAL_"+str(p)+str(d)+str(q),"Predict ARIMAX SUAVIZADO model log(Pbld)")
        
        
        
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
        ptfm.doPlotDoubleToFile (normalizedPbld,results_AR.fittedvalues,"ts_ARIMAX_SUAVIZADO_NORMALIZADO_"+str(p)+str(d)+str(q),"ARIMAX model SUAVIZADO NORMALIZADO")
        mse = mean_squared_error(normalizedPbld,results_AR.fittedvalues)
        mae = mean_absolute_error(normalizedPbld,results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(normalizedPbld,results_AR.fittedvalues)    
        

    
def doTimeSeriesForecasting ():
        
# BTL: En primer termino instancio la clase TCSeries, que viene de
# ThermalComfortSeries... es decir tratamiento por series numericas
# del problema del comfort termico. De todo el dataset solo voy a utilizar
# las columnas  coSeriesNames =['Control','Pbld']     
        tcs = TCSeries.TCSeries("mytest",coSeriesNames)

# BTL: Cargamos los datos en el dataframe y realizamos un resample en periodos
# de cuatro horas tomando la media        
    
        fulldata = ptfm.loadFileDataWithTime(filepath,'datosvivienda_testwc.csv')
        aggData = fulldata.resample('4H').mean()

# BTL: Inicializamos el objeto TCSeries esta funcion ademas calcula
# el logaritmo de los valores, es una forma de penalizar los valores
# mas altos y homogeneizar la serie. Posteriormente verificamos si 
# estamos ante una serie estacionaria         
        data_log = tcs.initialize(aggData)
        tcs.checkStationarity(data_log['Pbld'])
        
# BTL: Estas graficas solo representan los valores medios 
# contra el logarimo de los mismos, solo tiene propositos ilustarativos
                
        ptfm.doPlotSingleToFile (aggData,"ts_base","Base data ")
        ptfm.doPlotSingleToFile (data_log,"ts_base_log","Logarimic data")

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
        ptfm.doPlotSingleToFile (lag_acf,"ts_ac","Autocorrelation function")
        ptfm.doPlotSingleToFile (lag_pacf,"ts_pac","Partial Autocorrelation function")

        # BTL: Trato de determinar cuales son los mejore valore de p,q y d de forma iterativa
# es decir prueba-error. Este procedimiento puede tardar mucho
# OJOOOO !!! Esta funcion tarda mucho....

# BTL realizamos la grafica con la mejor opcion segun el metodo AIC
        best_sol_mse,best_sol_aic =doSelectBestARIMA  (tcs,residual)

        #p = best_sol_aic[1][0]
        #d = best_sol_aic[1][1]
        #q = best_sol_aic[1][2]
        
        p = 8
        d = 0
        q = 2
       
        model = ARIMA(residual, order=(p,d,q))  
        results_AR = model.fit(disp=-1)
        ptfm.doPlotDoubleToFileDate (results_AR.fittedvalues,residual,"ts_ARIMA_"+str(p)+str(d)+str(q),"ARIMA model","Modelo","Real")
        mse = mean_squared_error(residual.values,results_AR.fittedvalues)
        mae = mean_absolute_error(residual.values,results_AR.fittedvalues)
        mape = mean_absolute_percentage_error(residual.values,results_AR.fittedvalues)
        #mse_values=mean_squared_error(residual.values,results_AR.fittedvalues,multioutput='raw_values')
        #doPlotSingleToFile (mse_values, "MSE_ts_ARIMA_"+str(p)+str(d)+str(q),"MSE ARIMA: " + str (mse))
        
#        model.predict()

        
# BTL realizamos la grafica con la mejor opcion segun el metodo MSE                
        p = best_sol_mse[1][0]
        d = best_sol_mse[1][1]
        q = best_sol_mse[1][2]
        model = ARIMA(residual, order=(p,d,q))  
        results_AR = model.fit(disp=-1)  
         
        ptfm.doPlotDoubleToFileDate (results_AR.fittedvalues,residual,"ts_ARIMA_"+str(p)+str(d)+str(q),"ARIMA model","Modelo","Real")
    
def doTimeSeriesEnergyForecasting (hpc,dataP,dataR):    
    
    efor=[]
    ereal=[]
    tout = evaldf["Toutdoor"][(0*96):(5*96)+96,] # take one day as reference    
    for i in range (0,len(dataP)):
        logger.debug("Period.." + str(i))        
        et=hpc.predictHP( tout[4*i],dataP[i],21.5)
        efor=numpy.append (efor,et)    
#        et=hpc.predictHP( tout[(k*96)+4*i],dataR[i],21.5)
        
#        et=hpc.predictHP( tout[4*i],dataR["vals"][i],21.5)
        et=hpc.predictHP( tout[4*i],dataR[i],21.5)
#        aa = random.randint (-25,75)
        ereal=numpy.append (ereal,et )
        
    auxFor = pd.DataFrame()
    auxReal = pd.DataFrame()
        
    auxFor["vals"]=efor
#    dl=pd.date_range(start='1/3/2018', periods=25,freq='4h')
    dl=pd.date_range(start='05/3/2018', periods=7,freq='4h')
    auxFor=auxFor.set_index (pd.DatetimeIndex(dl))
    
    auxReal["vals"]=ereal

#    dl=pd.date_range(start='1/3/2018', periods=25,freq='4h')
    dl=pd.date_range(start='05/3/2018', periods=7,freq='4h')
    auxReal=auxReal.set_index (pd.DatetimeIndex(dl))
           
    ptfm.doPlotDoubleToFileDate (auxFor,auxReal,"ts_ARIMA_Energy_Outbound","ARIMAX Energia Electrica","Modelo","Real")
    logger.debug("Fin time series energy forecasting..")                 



if __name__ == "__main__":
    
    imp.reload(TZZipper)
    imp.reload(TCPredictor)
    imp.reload(TCSeries)
    imp.reload(logging)
    imp.reload(IndoorModel)
    
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

# BTL: Modelado por series temporales tradicionales ARIMA    
#    doTimeSeriesForecasting()

# BTL: Incorporando una variable exogena, utilizo los mismos valores
# pdq que he obtenido anteriormente
#    doTimeSeriesARIMAXForecasting()

#   predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY,predict_ts_ARIMAX_RESIDUAL_FULL_InBound_POLY_Real,ARIMAX_RESIDUAL_FULL_OutBound_POLY,ARIMAX_RESIDUAL_FULL_OutBound_OLS,ARIMAX_RESIDUAL_FULL_OutBound_Real
    AX_RES_FULL_InBound_POLY,AX_RES_FULL_InBound_POLY_Real,AX_RES_FULL_OutBound_POLY,AX_RES_FULL_OutBound_OLS,AX_RES_FULL_OutBound_Real= doTimeSeriesARIMAXForecastingWithResiduals()
    
    hpc = IndoorModel.IndoorModel() 
    evaldf = hpc.loadEvaluationData()
    hpc.createTrainHPModel(evaldf)    
#    doTimeSeriesEnergyForecasting (hpc,AX_RES_FULL_InBound_POLY,AX_RES_FULL_InBound_POLY_Real)
    doTimeSeriesEnergyForecasting (hpc,AX_RES_FULL_OutBound_OLS,AX_RES_FULL_OutBound_Real)
    logger.debug("Process ended...")
    