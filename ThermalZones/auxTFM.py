#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 18:47:52 2018

@author: batela
"""

import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt2
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

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

#def loadFileData (filepath,rooms):
#                
#    filename = 'datosvivienda_test.csv'            
#    fulldata = pd.read_csv(filepath+filename, sep=',', decimal = '.')
#    data = fulldata
#
#    return data


def loadFileDataWithTime (filepath,filename):
                
    dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M:%S')
    fulldata = pd.read_csv(filepath+filename, sep=',', decimal = '.', index_col='Control',date_parser=dateparse ,usecols=coSeriesNames)
    return fulldata



def doPlotDoubleToFileDate (data1,data2, fname,title,label1,label2):
        
    fig, ax = plt2.subplots()    
#        ax.plot(df.index, df.values)
#        ax.set_xticks(data2.index)
    ax.xaxis.set_tick_params(labelsize=7)        
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d:%H"))
    
#        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))
    plt2.xticks(rotation=90) 
    plt2.plot(data1,label=label1)
    plt2.plot(data2,label=label2)
    plt2.ylabel('Potencia (W)')
    plt2.grid()
    plt2.title(title)
    ax.legend()
    plt2.savefig('../Images/'+fname)
    plt2.close()


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
