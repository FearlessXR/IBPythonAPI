# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import stats
#from scipy.interpolate import griddata
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import pickle
#import os
from fastkde import fastKDE
import pylab as PP

import time

def getDensityInterpolator(drift, numSim = 1000000, nSteps = 100):

    numSteps = nSteps
    
    timeStep = 1.0/numSteps    
    
    numSimulations = numSim
    
    hlc = np.zeros((numSimulations, 3), dtype = float)
    
    drift = drift
    
    minDensity = 1.0/numSimulations/10
    
    for re in hlc:
        arr = np.array(np.random.normal(0,math.sqrt(timeStep),numSteps))
        arr = arr+drift*timeStep
        myHLC = np.cumsum(arr)
        re[0] = np.max(myHLC) 
        re[1] = np.min(myHLC) 
        re[2] = myHLC[-1]
        if re[0]<0.0:
            re[0] = 0
        if re[1]>0.0:
            re[1] = 0
    
    #hmax = np.max(hlc[:,0]) #max high
    #lmin = np.min(hlc[:,1]) #min low
#    cmin = np.min(hlc[:,2]) #min close
#    cmax = np.max(hlc[:,2]) #max close
#    
#    
    w = hlc[:,0]-hlc[:,1]
    c = hlc[:,2]
#    wmax = np.max(w)
#    W, C = np.mgrid[0:wmax:128j, cmin:cmax:128j]
    
    
#    positions = np.vstack([W.ravel(), C.ravel()])
#    values = np.vstack([w,c])
#    kernel = stats.gaussian_kde(values)
#    di = kernel(np.vstack([W.flatten(), C.flatten()]))
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.pcolormesh(W, C, di.reshape(W.shape))
    #plt.show() 
    
    wgrid = np.linspace(0.0, 6, 101)
    cgrid = np.linspace(-6, 6, 201)
    
    counts, edges = np.histogramdd([w,c], bins=(wgrid, cgrid))
    myden = counts/np.sum(counts)
    myden[myden<minDensity]=minDensity
    
#    mw, mc = np.mgrid[0.06:5.94:100j, -5.94:5.94:200j] 
#    fig2 = plt.figure()
#    bx = fig2.add_subplot(111)
#    bx.pcolormesh(mw,mc,myden)
    
    
    myInterp = RegularGridInterpolator([np.mgrid[0.06:5.94:100j], np.mgrid[-5.94:5.94:200j]], myden, bounds_error = False, fill_value = 1e-11)
    
    return myInterp
#    pts = np.vstack([mw.ravel(), mc.ravel()])
#    re=myInterp(pts.T)
#    re = np.reshape(re, (mw.shape[0], mw.shape[1]))
#    fig3 = plt.figure()
#    cx = fig3.add_subplot(111)
#    cx.pcolormesh(mw,mc,re)
#    
#    fig4 = plt.figure()
#    dx = fig4.add_subplot(111)
#    re = kernel(pts)
#    re = np.reshape(re, (mw.shape[0], mw.shape[1]))
#    dx.pcolormesh(mw,mc,re)
    
    
#    plt.show()

def getDensityInterpolatorFastkde(drift, numSim = 2**21+1, nSteps = 100):

    numSteps = nSteps
    
    timeStep = 1.0/numSteps    
    
    numSimulations = numSim
    
    hlc = np.zeros((numSimulations, 3), dtype = float)
    
    drift = drift
    
    minDensity = 1.0/numSimulations/10
    
    for re in hlc:
        arr = np.array(np.random.normal(0,math.sqrt(timeStep),numSteps))
        arr = arr+drift*timeStep
        myHLC = np.cumsum(arr)
        re[0] = np.max(myHLC) 
        re[1] = np.min(myHLC) 
        re[2] = myHLC[-1]
        if re[0]<0.0:
            re[0] = 0
        if re[1]>0.0:
            re[1] = 0
    
#    
    w = hlc[:,0]-hlc[:,1]
    c = hlc[:,2]
    
    akernel = fastKDE.fastKDE([w, c],  \
                                                beVerbose=False, \
                                                doSaveMarginals = False, \
                                                numPoints=129)
    
#    wmax = np.max(w)
#    W, C = np.mgrid[0:wmax:128j, cmin:cmax:128j]
    
    
#    positions = np.vstack([W.ravel(), C.ravel()])
#    values = np.vstack([w,c])
#    kernel = stats.gaussian_kde(values)
#    di = kernel(np.vstack([W.flatten(), C.flatten()]))
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.pcolormesh(W, C, di.reshape(W.shape))
    #plt.show() 
    
    return akernel

    
def getDensityInterpolatorFastkdeWCRatio(drift, numSim = 2**22+1, nSteps = 50):

    numSteps = nSteps
    
    timeStep = 1.0/numSteps    
    
    numSimulations = numSim
    
    hlc = np.zeros((numSimulations, 3), dtype = float)
    
    drift = drift
    
    minDensity = 1.0/numSimulations/10
    
    for re in hlc:
        arr = np.array(np.random.normal(0,math.sqrt(timeStep),numSteps))
        arr = arr+drift*timeStep
        myHLC = np.cumsum(arr)
        re[0] = np.max(myHLC) 
        re[1] = np.min(myHLC) 
        re[2] = myHLC[-1]
        if re[0]<0.0:
            re[0] = 0
        if re[1]>0.0:
            re[1] = 0
    
#    
    w = hlc[:,0]-hlc[:,1]
    c = hlc[:,2]
    ratios = c/w
    
#    akernel = fastKDE.fastKDE(ratios,  \
#                                                beVerbose=False, \
#                                                doSaveMarginals = False, \
#                                                numPoints=513)
    akernel = stats.gaussian_kde(ratios)
    
#    wmax = np.max(w)
#    W, C = np.mgrid[0:wmax:128j, cmin:cmax:128j]
    
    
#    positions = np.vstack([W.ravel(), C.ravel()])
#    values = np.vstack([w,c])
#    kernel = stats.gaussian_kde(values)
#    di = kernel(np.vstack([W.flatten(), C.flatten()]))
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.pcolormesh(W, C, di.reshape(W.shape))
    #plt.show() 
    
    return akernel
    

    
if __name__ == "__main__":
    
    aDrift = 0.0
    numSteps = 50
    numSim = 2**21+1
    
#    mw, mc = np.mgrid[0.06:5.94:100j, -5.94:5.94:200j] 
#    pts = np.vstack([mw.ravel(), mc.ravel()])
    
#testing load interpolators
#    for aDrift in np.linspace(0.0,2,41 ):
#        print('drift = ' + '{:1.2f}'.format(aDrift))
#        filename ='DensityInterpolators/DensityInterpolatorDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        aInterp = pickle.load(open(filename, 'rb'))
#        afile.close()
#        print(filename)
#        re = aInterp(pts.T)
#        re = np.reshape(re, (mw.shape[0], mw.shape[1]))
#        fig3 = plt.figure()
#        cx = fig3.add_subplot(111)
#        cx.pcolormesh(mw,mc,re)

# generating density interpolators
#    for aDrift in np.linspace(0.0,2,41 ):
#        print('drift = ' + '{:1.2f}'.format(aDrift))
#        aInterp = getDensityInterpolator(aDrift, numSim, numSteps)
#        filename ='DensityInterpolators/DensityInterpolatorDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        afile = open(filename, 'wb')
#        pickle.dump(aInterp, afile)
#        afile.close()
#        print(filename)



# generating fastKDE objects and save
#    for aDrift in np.linspace(0.0,2,41 ):
#        start_time = time.time()
#        akernel = getDensityInterpolatorFastkde(aDrift, numSim, numSteps)
#        v1,v2 = akernel.axes
#        
#        print('drift = ' + '{:1.2f}'.format(aDrift))
#        filename ='DensityInterpolators/fastKDEKernelDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        afile = open(filename, 'wb')
#        pickle.dump(akernel, afile)
#        afile.close()
#        print(filename)
#        PP.contour(v1,v2,akernel.pdf)
#        PP.show()
#        print("--- %s seconds ---" % (time.time() - start_time))    
#        x2,y2=np.meshgrid(v1,v2)
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.pcolormesh(x2, y2, akernel.pdf.reshape(x2.shape))

# generating fastKDE interpolators and save

#    v3,v4 = np.mgrid[0:5:501j, -5:5:1001j ]
#    aDrift = 0.25
#    start_time = time.time()
##    pts = np.vstack([v3.flatten(), v4.flatten()])
#    
#    for aDrift in np.linspace(0.0,2,41 ):
#        start_time = time.time()
#                
#        print('drift = ' + '{:1.2f}'.format(aDrift))
#        filename ='DensityInterpolators/fastKDEKernelDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        afile = open(filename, 'rb')
#        akernel = pickle.load(afile)
#        afile.close()
#        pdfs = akernel.pdf        
#        v1,v2 = akernel.axes
#        pdfs[pdfs<1e-7]=1e-8
##        aInterp = RegularGridInterpolator((v1, v2), pdfs, method = 'linear', bounds_error = False, fill_value = 1e-8)
#        PP.contour(v1,v2,akernel.pdf)
#        PP.show()
        
#        x1,x2 = np.meshgrid(v1,v2)        
#        pts = np.vstack((x1.flatten(), x2.flatten())
#        newPdfs = aInterp(pts.T)
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#    
#    ax.pcolormesh(x1, x2, newPdfs.reshape(x2.shape).T)
#        ax.pcolormesh(v3, v4, newPdfs.reshape(v3.shape))
        
#        print(filename+" closed")
#        filename ='DensityInterpolators/fastKDEInterpolator'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        afile = open(filename, 'wb')
#        pickle.dump(aInterp, afile)
#        afile.close()
#        print('Interpolator saved to '+filename)
#        print("--- %s seconds ---" % (time.time() - start_time))    

#testing fastKDE interpolators
#    
#    v1,v2 = np.mgrid[0:5:501j, -5:5:1001j ]
#    aDrift = 0.25
#    start_time = time.time()
    
#    filename ='DensityInterpolators/fastKDEKernelDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#    afile = open(filename, 'rb')
#    akernel = pickle.load(afile)
#    afile.close()
            
#    print('drift = ' + '{:1.2f}'.format(aDrift))
#    filename ='DensityInterpolators/fastKDEInterpolator'+'{:1.0f}'.format(aDrift*100)+'.obj'
#    afile = open(filename, 'rb')    
#    aInterp = pickle.load(afile)
#    afile.close()
#    
#    pts = np.vstack([v1.ravel(), v2.ravel()])
#    pdfs = aInterp(pts.T)
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.pcolormesh(v1, v2, pdfs.reshape(v1.shape))
#    
#    print("--- %s seconds ---" % (time.time() - start_time))

# generating wcratio density interpolators
    x=np.mgrid[-1:1:513j]
    start_time = time.time()
#    for aDrift in np.linspace(0.0,2,41 ):
#        akernel = getDensityInterpolatorFastkdeWCRatio(aDrift)
#        filename ='DensityInterpolators/cwRatioKernelDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        afile = open(filename, 'wb')
#        pickle.dump(akernel, afile)        
#        afile.close()
#        print('drift = ' + '{:1.2f}'.format(aDrift) +' done')
#        print('kernel done')
#        print('------ %s seconds-----' % (time.time()-start_time))
#        pdfs = akernel(x)
#        
#        filename = 'DensityInterpolators/cwRatioDensityInterpolatorDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
#        aInterp = interp1d(x,pdfs)        
#        bfile = open(filename, 'wb')
#        pickle.dump(aInterp, bfile)
#        bfile.close()
#        print('interpolator done')
#        print('------ %s seconds-----' % (time.time()-start_time))
        
    for aDrift in np.linspace(1.,2,11 ):
        filename = 'DensityInterpolators/cwRatioDensityInterpolatorDrift'+'{:1.0f}'.format(aDrift*100)+'.obj'
               
        bfile = open(filename, 'rb')
        aInterp=pickle.load(bfile)
        bfile.close()

        pdfs = aInterp(x)
        plt.plot(x,pdfs)
        plt.show()        
        print('interpolator done')
        print('------ %s seconds-----' % (time.time()-start_time))        
