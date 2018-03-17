# -*- coding: utf-8 -*-

import numpy as np
import math
#from scipy import stats
#from scipy.interpolate import griddata
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import RegularGridInterpolator
from scipy import optimize
import pickle
import os
import pandas as pd
import datetime as dt
import time
#import statsmodels.tsa.stattools as tsa

class densityEstimator:
    def __init__(self):
        self.interpolators = []
        self.ratioInterpolators = []
        self.drifts = np.zeros(41)
        i = 0
        for aDrift in np.linspace(0.0,2,41 ):
#            print('drift = ' + '{:1.2f}'.format(aDrift))
#            filename ='DensityInterpolators/DensityInterpolatorDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
            filename ='DensityInterpolators/fastKDEInterpolator'+'{:1.0f}'.format(aDrift*100)+'.obj'
#            print(filename)
            afile = open(filename, 'rb')
            aInterp = pickle.load(afile)
            self.interpolators.append(aInterp)
            self.drifts[i] = aDrift     
            i += 1
            afile.close()
            filename = 'DensityInterpolators/cwRatioDensityInterpolatorDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
            bfile = open(filename, 'rb')
            aInterp = pickle.load(bfile)
            self.ratioInterpolators.append(aInterp)
        self.mw = []
        self.mc = []
        self.weights = []
    
    def getDensity(self, aW: float, aC: float, drift: float):
        mu = drift
        myClose = aC        
        if (drift < 0):
            mu = -drift
            myClose = -aC
        
        if aW<0:
            return 0.0
        if myClose>aW:
            return 0.0

        idx = (np.abs(self.drifts-mu)).argmin()
        
        if mu == self.drifts[idx]:
            return self.interpolators[idx](np.array([aW, myClose]).T)
        elif mu > self.drifts[idx]:
            if idx == self.drifts.size-1:
                return self.interpolators[idx](np.array([aW, myClose]).T)
            else:
                weight = (mu-self.drifts[idx])/(self.drifts[idx+1] - self.drifts[idx])
                return self.interpolators[idx](np.array([aW, myClose]).T)*(1.0-weight)+self.interpolators[idx+1](np.array([aW, myClose]).T)*weight
        else:
            weight = (self.drifts[idx]-mu)/(self.drifts[idx] - self.drifts[idx-1])
            return self.interpolators[idx](np.array([aW, myClose]).T)*(1.0-weight)+self.interpolators[idx-1](np.array([aW, myClose]).T)*weight
    
    
    def getDensityRawInput(self, w_in : float, c_in: float, drift: float, vol: float):
        newDrift = drift/vol
        newW = w_in/vol
        newC = c_in/vol
#        aValue = vol
#        print('{:1.5f}'.format(w_in) + ' ' + '{:1.5f}'.format(c_in))
#        print( '{:1.5f}'.format(drift) )
#        print( str(aValue) )
        result = self.getDensity(newW, newC, newDrift)
        if result <= 1e-8:
            result = 1e-8
        return result
    
    def setWC(self, w_in:list, c_in:list, weights:list = None):
        self.mw = w_in
        self.mc = c_in
        if not(weights):
            self.weights = [1.0]*len(self.mw)
        else:
            self.weights = weights
        
        
    
    def evaluateSumLogDensity(self, driftvol):
        if not len(self.mw): #check if the list is empty
            return None;
        
        result = 0.0
        for aw,ac,weight in zip(self.mw, self.mc, self.weights):
#            print(aw)
#            print(ac)
#            print('W = ''{:1.5f}'.format(aw) +" C = " + '{:1.5f}'.format(ac) + " drift = "+ '{:1.5f}'.format(driftvol[0]) + ' vol = ' + '{:1.5f}'.format(driftvol[1]))
            result += math.log(self.getDensityRawInput(aw, ac, driftvol[0], driftvol[1]))*weight
            
        
        return -result #return negative log, for minimizing
    
    def evaluateSumLogDensity2(self, drift: float, vol: float):
        if not len(self.mw): #check if the list is empty
            return None;
        
        result = 0.0
        for aw,ac in zip(self.mw, self.mc):
#            print(aw)
#            print(ac)
#            print('W = ''{:1.5f}'.format(aw) +" C = " + '{:1.5f}'.format(ac) + " drift = "+ '{:1.5f}'.format(driftvol[0]) + ' vol = ' + '{:1.5f}'.format(driftvol[1]))
            result += math.log(self.getDensityRawInput(aw, ac, drift, vol))
            
        
        return -result #return negative log, for minimizing
        
    def evaluateSumLogDensity3(self, vol: float, driftvolratio: float):
        return self.evaluateSumLogDensity2(vol*driftvolratio, vol)
        
    def evaluateSumLogDensityNoDrift(self, vol: float):
        if not len(self.mw): #check if the list is empty
            return None;
        
        result = 0.0
        for aw,ac in zip(self.mw, self.mc):
#            print(aw)
#            print(ac)
#            print('W = ''{:1.5f}'.format(aw) +" C = " + '{:1.5f}'.format(ac) + " drift = "+ '{:1.5f}'.format(driftvol[0]) + ' vol = ' + '{:1.5f}'.format(driftvol[1]))
            result += math.log(self.getDensityRawInput(aw, ac, 0.0, vol))
            
        
        return -result #return negative log, for minimizing
    
    def getLikelyDriftAndVol(self, met = 'SLSQP'):
#        bnds = ((None,None), (1e-5, None))
        avgW = np.average(self.mw)
        avgC = np.average(self.mc)
        
        guessVol = avgW/1.5
#        guessDrift = avgC/guessVol
        
#        results = optimize.minimize(self.evaluateSumLogDensity, [guessDrift, guessVol], method=met, bounds = bnds)
        results = optimize.brute(self.evaluateSumLogDensity, ((-1,1),(0.0001,0.5)), Ns = 50)

        return results
        
    def getLikelyVolNoDrift(self):
        
        avgW = np.average(self.mw)
        avgC = np.average(self.mc)
        
        guessVol = avgW/1.5
        guessDrift = avgC/guessVol

#        vol = optimize.fmin(self.evaluateSumLogDensityNoDrift, guessVol, xtol = 1e-5, disp = 0)
        vol = [0.00055]
        drift = optimize.fmin(self.evaluateSumLogDensity2, guessDrift, xtol = 1e-5, disp = 0, args=(vol[0],))
        results = [drift[0],vol[0]]
        return results
    
    def getLikelyRatioDrift(self, driftGuess = -100.0): #actually returning drift/vol ratio, normalized drift
        
        if driftGuess >-190.0:        
            avgW = np.average(self.mw)
            avgC = np.average(self.mc)
        
            guessVol = avgW/1.5
            guessDrift = avgC/guessVol
        else:
            guessDrift = driftGuess

#        drift = optimize.fmin(self.evaluateSumLogDensityRatio, guessDrift, xtol = 1e-4, disp = 0)
        drift = optimize.fminbound(self.evaluateSumLogDensityRatio, -2.0, 2.0, xtol = 1e-4, disp = 0)
        return drift
        
    def getLikelyRatioDrift2(self): #actually returning drift/vol ratio, normalized drift
        ######
        #  returning drift value at only the anchor values, i.e 0, 0.05, 0.1, etc.
        ######
        idx = 0
        i=0
        minvalue = self.evaluateSumLogDensityRatio(0.0)
        ispositive = True
        avalue = 0
        for dr in self.drifts:
            avalue = self.evaluateSumLogDensityRatio(dr)
            if avalue<minvalue:
                minvalue = avalue
                idx = i
                ispositive = True
            avalue = self.evaluateSumLogDensityRatio(-dr)
            if avalue<minvalue:
                minvalue = avalue
                idx = i
                ispositive = False
            i += 1
                

        drift = self.drifts[idx]*(ispositive-0.5)*2.0
        return drift
    
    def getLikelyVolDriftGivenRatio(self, ratio: float): #actually returning drift/vol ratio, normalized drift
        
       
        vol = optimize.fminbound(self.evaluateSumLogDensity3, 0.00005, 0.50, xtol = 1e-5, disp = 0, args=(ratio,))
        return vol

    def evaluateSumLogDensityRatio(self, drift: float):
        result = 0.0
        for aw,ac in zip(self.mw, self.mc):
#            print(aw)
#            print(ac)
#            print('W = ''{:1.5f}'.format(aw) +" C = " + '{:1.5f}'.format(ac) + " drift = "+ '{:1.5f}'.format(driftvol[0]) + ' vol = ' + '{:1.5f}'.format(driftvol[1]))
            result += math.log(self.getRatioDensity(ac/aw, drift))
        return -result #return negative log, for minimizing
    
    def getRatioDensity(self, ratio: float, drift: float): #ratio is w/c and drift is actually drift/vol ratio
        mu = np.abs(drift)
        myClose = ratio
        ispositive = True        
        if (drift < 0):
            ispositive = False
        
        if ratio<-1.0:
            return 1e-9
        if ratio>1.0:
            return 1e-9

        idx = (np.abs(self.drifts-mu)).argmin()

        
        if mu == self.drifts[idx]:
            if (ispositive):
                return self.ratioInterpolators[idx](myClose)
            elif idx==0:
                return self.ratioInterpolators[0](myClose)
            else:
                return self.ratioInterpolators[idx](-myClose)
        elif mu > self.drifts[idx]:
            if ispositive:
                if idx == self.drifts.size-1:
                    return self.ratioInterpolators[idx](myClose)
                else:
                    weight = (mu-self.drifts[idx])/(self.drifts[idx+1] - self.drifts[idx])                
                    return self.ratioInterpolators[idx](myClose)*(1.0-weight)+self.ratioInterpolators[idx+1](myClose)*weight
            elif idx == 0:
                weight = (mu-self.drifts[idx])/(self.drifts[idx+1] - self.drifts[idx])                
                return self.ratioInterpolators[idx](myClose)*(1.0-weight)+self.ratioInterpolators[idx+1](-myClose)*weight
            else:
                weight = (mu-self.drifts[idx])/(self.drifts[idx+1] - self.drifts[idx])                
                return self.ratioInterpolators[idx](-myClose)*(1.0-weight)+self.ratioInterpolators[idx+1](-myClose)*weight
        else:
            weight = (self.drifts[idx]-mu)/(self.drifts[idx] - self.drifts[idx-1])
            if ispositive:
                return self.ratioInterpolators[idx](myClose)*(1.0-weight)+self.ratioInterpolators[idx-1](myClose)*weight
            elif idx==1:
                return self.ratioInterpolators[idx](-myClose)*(1.0-weight)+self.ratioInterpolators[idx-1](myClose)*weight
            else:
                return self.ratioInterpolators[idx](-myClose)*(1.0-weight)+self.ratioInterpolators[idx-1](-myClose)*weight

if __name__ == "__main__":
    
    start_time = time.time()
    myDen = densityEstimator()
#    aDrift = 0.1
#    vol = 3.0
    
#    aw = [0.3, 0.3, 0.3, 0.3, 0.3]
#    ac = [0.1, 0.1, 0.1, 0.1, 0.1]
#    
#    myDen.setWC(aw, ac)
##    result = myDen.getLikelyDriftAndVol('TNC')
##
##    print(result)
##    dr=-result.x[0]
##    vo=result.x[1]
##    for i in range(4):
##        print(myDen.getDensityRawInput(aw[i],ac[i],result[0],result[1]))
#    dr = -1.07831903
#    vo = 0.07923818
#    for i in range(4):
#        print(myDen.getDensityRawInput(aw[i],ac[i],dr,vo))
    
    filename = os.path.join(os.environ['QT_MKTDATA_PATH'], 'Commod//testBR1MinBars.csv')
    data = pd.read_csv(filename)
    ts = data["time"].tolist()
    ts = [dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in ts]
    data["time"] = ts
    
    ts = data['time'].tolist()
    opens = np.array(data["open"])
    highs = np.array(data["high"])
    lows = np.array(data["low"])
    closes = np.array(data["close"])
    volumes = np.array(data["volume"])
    waps = np.array(data["wap"])
    
    ws = np.log(highs/lows)
    cs = np.log(closes/opens)
    

    
    estnum = 100 # rolling window size
    count = 0
    
    aw = [ws[i] for i in range(estnum)]
    ac = [cs[i] for i in range(estnum)]
    
    filename = os.path.join(os.environ['QT_MKTDATA_PATH'], 'Commod//testBR1MinVolRatio.csv')
    output = open(filename, 'w')
#    output.write('time,drift,vol\n')
#    output.write('time,vol,open, high, low, close, volume\n')
#    output.write('time,drift, vol,open, high, low, close, volume\n')
#    output.write('time, driftvolratio,open, high, low, close, volume\n')
    output.write('time, driftvolratio,vol,open, high, low, close, volume\n')
    totrows = len(ws)    
    lotunit = 300
    
    print(time.time()-start_time)
    totlots = 0
    ahigh = 0.0
    alow = 1.0e5
    aclose = 0.0
    aopen = -1.0
    scale = 1.0
    drifts = []
    for i in range(len(ws)):
        totlots += volumes[i]
        if aopen < 0.0:
            aopen = opens[i]
        if highs[i]>ahigh:
            ahigh = highs[i]
        if lows[i]<alow:
            alow = lows[i]
        aclose = closes[i]
        if totlots > lotunit:
            aw.pop(0)
            ac.pop(0)
            scale = 1.0/math.sqrt(1.0*totlots/lotunit)
            myw = math.log(ahigh/alow)*scale
            myc = math.log(aclose/aopen)*scale
            aw.append(myw)
            ac.append(myc)
            count += 1
            
            if count>=estnum:
                myDen.setWC(aw,ac)
#                results = myDen.getLikelyDriftAndVol()
#                aline = ts[i].strftime("%Y-%m-%d %H:%M:%S") + ',' + '{:1.5f}'.format(results[0]) + ',' + '{:1.5f}'.format(results[1])+ ',' + str(totlots) + '\n'
#                results = myDen.getLikelyVolNoDrift()
                if (len(drifts)<1):
                    iniDrift = -100
                else:
                    iniDrift = drifts[-1]
                results = myDen.getLikelyRatioDrift(iniDrift)
                vol = myDen.getLikelyVolDriftGivenRatio(results)
               
                drifts.append(results)
                aline = ts[i].strftime("%Y-%m-%d %H:%M:%S") + ',' + '{:1.5f}'.format(results)+ ',' + '{:1.5f}'.format(vol)+ ',' + '{:1.5f}'.format(aopen) +',' + '{:1.5f}'.format(ahigh) +',' + '{:1.5f}'.format(alow) +',' + '{:1.5f}'.format(aclose) +',' + str(totlots) + '\n'
                output.write(aline)
            totlots = 0.0
            ahigh = 0.0
            alow = 1e5
            aclose = 0.0
            aopen = -1.0
        if (i%100 == 0):
            print(str(i)+' out of '+str(totrows))
            print(time.time()-start_time)
    
    output.close()    