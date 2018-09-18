#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:21:14 2018

@author: gorbea
"""
import logging
import numpy as np
import pandas as pd
import pickle
import datetime
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib

import os.path

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cross_validation import cross_val_score



fileNames =['data_DOFF.p','data_D0.0.p','data_D0.1.p','data_D0.2.p','data_D0.3.p','data_D0.4.p','data_D0.5.p','data_D0.6.p','data_D0.7.p','data_D0.8.p','data_D-0.1.p','data_D-0.2.p','data_D-0.3.p','data_D-0.4.p','data_D-0.5.p','data_D-0.6.p','data_D-0.7.p','data_D-0.8.p']


class IndoorModel (object):
    

    def __init__(self):
        '''
        Constructor esto es un comenario
        '''
        self.trainDF = None
        self.model   = None
        self.data    = None
        self.hponTR = None   
        self.hponHP = None
        self.logger = logging.getLogger("indooropt")
#        self.loadFullDada()
        
    def clean (self):        
        self.model       = None
        self.data = None
        self.trainDF = None
        self.hponTR = None

    def loadEvaluationData (self):
        filepath='../Repo/'
        filename = "datosvivienda_hp_model.csv"
        evaldata = pd.read_csv (filepath+filename,sep=",",decimal=".")
        return evaldata


    def savitzky_golay(self,y, window_size, order, deriv=0, rate=1):
          
       window_size = np.abs(np.int(window_size))
       order = np.abs(np.int(order))
       if window_size % 2 != 1 or window_size < 1:
       raise TypeError("window_size size must be a positive odd number")
       if window_size < order + 2:
       raise TypeError("window_size is too small for the polynomials order")
       order_range = range(order+1)
       half_window = (window_size -1) // 2
       # precompute coefficients
       b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
       m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
       # pad the signal at the extremes with
       # values taken from the signal itself
       firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
       lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
       y = np.concatenate((firstvals, y, lastvals))
       return np.convolve( m[::-1], y, mode='valid')

    def loadFullDada (self):
    
        dffull = None
        dfaux = pd.DataFrame()
        for fn in fileNames:
            with open('../Repo/7day/'+fn, 'rb') as f:
                dfaux = pickle.load(f)
                dfaux["tintiff"]=np.insert(np.diff(dfaux.iloc[:,0]),0,np.nan)
                dfaux["tzoniff"]=np.insert(np.diff(dfaux.iloc[:,1]),0,np.nan)
                if (dffull is None):
                    dffull = dfaux
                else:
                    dffull=dffull.append(dfaux)
        return dffull

    def createTrainHPModel (self,data):
        
        res = True    
        if os.path.isfile("../indoormodel_hp.pkl") :       
            self.hponHP= joblib.load('../indoormodel_hp.pkl')
        else :    
            X = data['Toutdoor']
            X = np.append(X,data['Pbld'])
            X = np.append(X,data['Tindoor'])
            X.shape = (3,len (data['Toutdoor'].index))
            X=X.transpose()
            y=data['Php']
        
            X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.25)  

            self.hponHP = ExtraTreesRegressor(n_estimators=3,
                              max_features=3,
                              n_jobs=5,
                              criterion="mae",
                              random_state=0)
            model=self.hponHP.fit(X_train, y_train)
            self.logger.info ('Accuracy training : {:.3f}'.format(self.hponHP.score(X_test, y_test))) 
            
            dt_scores = cross_val_score(model, X_test, y_test, cv = 5)
            self.logger.info("mean cross validation score: {}".format(dt_scores))
            self.logger.info("score without cv: {}".format(model.score(X_test, y_test)))
            
            dt_scores = cross_val_score(model, X_test, y_test, cv = 10)
            self.logger.info("mean cross validation score: {}".format(dt_scores))
            self.logger.info("score without cv: {}".format(model.score(X_test, y_test)))
            
            
            
            importances = self.hponHP.feature_importances_
            std = np.std([tree.feature_importances_ for tree in self.hponHP.estimators_],
             axis=0)
            indices = np.argsort(importances)[::-1]

            
            self.logger.debug("Feature ranking:")
            
            for f in range(X.shape[1]):
                self.logger.info("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                
            pickle.dump(model, open('../indoormodel_hp.p', 'wb'))
            joblib.dump( model, '../indoormodel_hp.pkl')


        if ( self.hponHP is None):
            res = False

        return res
    
    def predictHP(self, a,b,c):
        #self.logger.debug ("Objective function called.." )        
        
        values = np.array([a,b,c])
        res = self.hponHP.predict(values.reshape(1,-1))
        
        #self.logger.debug ("Result for .." + str (a) + ":" + str (b) + ":" +str (c) +"->"+ str(res))        
        
        return res 
    
    def createNotLocalizedDataFrame (self):
        
        self.logger.info ("Start of Not Localized DF creation...")  
        
        self.trainDF = pd.DataFrame()
        
        self.trainDF['text'] = self.data["z_1.capZon.heaPor.T"]
        self.trainDF['tslm'] = self.data["Tamb"]
        self.trainDF['inctslm'] = self.data["tzoniff"]
        self.trainDF['potA'] = self.data["Qheat"]
        
        #dff=dff.drop (dff['inctslm'][dff['inctslm']<0].index)
        self.logger.info ("... end of Not Localized DF creation")
       
    
    
    def createTrainModel (self):

        
        res = True
    
        if os.path.isfile("../indoormodel.pkl") :       
            self.hponTR= joblib.load('../indoormodel.pkl')
        else :    
            X = self.trainDF['text']
            X = np.append(X,self.trainDF['tslm'])
            X = np.append(X,self.trainDF['potA'])
            X.shape = (3,len (self.trainDF['text'].index))
            X=X.transpose()
            Y=self.trainDF['inctslm']
        
            ids = np.repeat(np.arange(180//3), 3) 
            data_input =  X.reshape(-1,1).astype(int) #xsm.repeat(3).reshape(-1,1).astype(int)
            ysm= np.bincount(ids, Y)//np.bincount(ids)        
            data_output = ysm.repeat(3).reshape(-1,).astype(int)
        
            X_train, X_test, y_train, y_test = train_test_split( data_input, data_output)  
            x = np.arange(0.0, 1, 0.01).reshape(-1, 1)
            y = np.sin(2 * np.pi * x).ravel()

            nn = MLPRegressor(
                hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            n = nn.fit(x, y)
        
        
            #pickle.dump(model, open('../indoormodel.p', 'wb'))
            joblib.dump( model, '../indoormodel.pkl')

        if ( self.hponTR is None):
            res = False

        return res
    

    def predict(self, a,b,c):
        #self.logger.debug ("Objective function called.." )        
        
        values = np.array([a,b,c])
        res = self.hponTR.predict(values.reshape(1,-1))
        
        #self.logger.debug ("Result for .." + str (a) + ":" + str (b) + ":" +str (c) +"->"+ str(res))        
        
        return res 


    def calculateFlexibility(self, a,b,c):   
        
        t00=21.0+273.0
        t0=t00
        tsetp0 = 21.0 +273.0
        tsetp1 = 22.0 +273.0
        period = 51
        k=1
        evaldf= self.loadEvaluationData ()
    
        fulldf = self.loadFullDada ()
    
        tout = fulldf["Tamb"][(k*96):(k*96)+96,] # take one day as reference
        
        tfor=[]
        qfor=[]
        qrange = []
        qrange = np.append (np.linspace(0, 1000, num=5),np.linspace(3000, 5000, num=5))
    
        periodL=[35,51,35,51,35,51,35,51]
    #     
        k=0
        evaldf["Toutdoor"][evaldf["Toutdoor"] > 100]=2
        evaldf["Toutdoor"][evaldf["Toutdoor"] < -50]=0
        #    tout = fulldf["Tamb"][(k*96):(k*96)+96,] # take one day as reference    
        tout = evaldf["Toutdoor"][(k*96):(k*96)+96,] # take one day as reference    
        tsetp0L=[(21.0+273.0),(21.0+273.0)]
    #    tsetp0L=[(21.0+273.0)]
        tsetp1L=[(20.5+273.0),(20.5+273.0)]
        tsetp2L=[(22.0+273.0),(22.0+273.0)]
        periodL=[30,45]
        for k in range (5,6):
            tout = evaldf["Toutdoor"][(k*96):(k*96)+96,] # take one day as reference    
            for j in range (0,len (tsetp0L)):
                tsetp0 = tsetp0L[j]
                tsetp1 = tsetp1L[j]
                tsetp2 = tsetp2L[j]
                period = periodL[j]
                tfor=[]
                qfor=[]
                t0=t00
                efor=[]
                for i in range (0,96):
                    logger.debug("Period.." + str(i))        
                    tfor=np.append (tfor,t0)
                    tevol = []
                    for qq in qrange:
                        dt=self.predict( tout[(k*96)+i],t0,qq)
                        tevol=np.append (tevol,qq); tevol=np.append (tevol,t0+dt); tevol=np.append (tevol,dt)
                    
                    tevol.shape = (len(tevol)/3,3)
                    if (i<period):
    #                    positiva
                        idx =np.where(min (abs((tsetp0)-tevol[:,1]))==abs(((tsetp0)-tevol[:,1])))
    #                    negativa
    #                    idx =np.where(min (abs((tsetp2)-tevol[:,1]))==abs(((tsetp2)-tevol[:,1])))
                    elif (i>period and i < 70)  :
                        idx =np.where(min (abs((tsetp1)-tevol[:,1]))==abs(((tsetp1)-tevol[:,1])))
    #                    idx =np.where(min (abs((tsetp0)-tevol[:,1]))==abs(((tsetp0)-tevol[:,1])))
                    else:
                        idx =np.where(min (abs((tsetp0)-tevol[:,1]))==abs(((tsetp0)-tevol[:,1])))
    #                    idx =np.where(min (abs((tsetp2)-tevol[:,1]))==abs(((tsetp2)-tevol[:,1])))
                    t0 = tevol[idx,1]
                    qfor=np.append (qfor,tevol[idx,0])
                    #logger.debug("New temperatues.." + str(tfor))
                    et=self.predictHP( tout[(k*96)+i],tevol[idx,0],t0-273)
                    efor=np.append (efor,et)
#                logger.info("Initial T0: " + str(t00-273) + " Set point: " + str(tsetp0-273) +" - "+str(tsetp1-273))
        #        logger.info("Outdoor temperatues.." + str(tout.tolist()))
        #        logger.info("New temperatues.." + str(tfor))
        #        logger.info("New energies filter.." + str(savitzky_golay(qfor, 51, 3)))
        #        logger.info("New energies discre.." + str(qfor))
                
                dl=pd.date_range(start='1/1/2018', periods=96,freq='15min')                    
                pdd = pd.DataFrame()
                pdd["qfor"]=qfor
                pdd["qforSG"]=savitzky_golay(qfor, 51, 3)
                pdd["efor"]=efor            
                pdd["eforSG"]=savitzky_golay(efor, 51, 3)
                pdd=pdd.set_index (pd.DatetimeIndex(dl))
                pdd=pdd.resample("1H").mean()
                pdd.to_csv("../Outputs/Energy-"+str(tsetp0-273) + "-" + str(tsetp1-273)+"-" + str(period)+"-" + str(k)+".csv",sep='\t', encoding='utf-8')
                
#                logger.info(">>>>> ToutM:" + str(max(tout)) + " Toutm:" + str(min(tout)) + " ToutA:" + str(np.mean(tout)) +  
#                            " TinM:" + str(max(tfor-273.0)) + " Tinm:" + str(min(tfor-273.0)) + " ToutA:" + str(np.mean(tfor-273.0)) +
#                            " qM:" + str(max(qfor)) + " qTot:" + str(simps(qfor*0.25, dx=1)) + " qMSG:" + str(max(savitzky_golay(qfor, 51, 3))) + " qTotSG:"  + str(simps(savitzky_golay(qfor, 51, 3)*0.25, dx=1))  +
#                            " eTot:" + str(simps(efor*0.25, dx=1)) + " ETotSG:"  + str(simps(savitzky_golay(efor, 51, 3)*0.25, dx=1))  )
#                
##                doPlotEnergyToFile (efor,"Energy-"+str(tsetp0-273) + "-" + str(tsetp1-273)+"-" + str(period)+"-" + str(k)+".jpg","Seed "+str(t00-273) + " Target " + str(tsetp0))
#                doPlotDoubleToFile (tfor,qfor,tout.tolist(),str(tsetp0-273) + "-" + str(tsetp1-273)+"-" + str(period)+"-" + str(k)+".jpg","Seed "+str(t00-273) + " Target " + str(tsetp0))