import numpy as np
import matplotlib.pyplot as plt
from ..feature_calibration import *

def plotABRatio(sig, shortWsize, bigWsize, noverlap,fs, beta1,beta2,thr,buffer_dur=0.5,step=1,rmvf=None, plot=False, tauU_outBound =0.4, tauD_outBound = 0.1, figname = '' ,alarm_threshold=0.5,alarm_dur =1,tst=None, savePlot = False, saveFolder = '.\\'):
    """
    plot ABratio for given signal
    Args:
    sig: signal 
    figname: figure name 
    shortWsize: short window size
    bigWsize: big window size
    noverlap:the number of overlapping points
    fs: sample frequency
    beta1: time constant of adaptive mean
    beta2: time constant of adaptive standard deviation
    thr: threshold
    buffer_dur: duration for buffer to calculate standard deviance for ABratio
    step: small window move step
    rmvf: remove frequency list
    NoiseLevel: noise levels

    Outputs: (time, abratio, out of bound rate)

    """
  
    
    # outBound_rate = []

    # time step for Abratio
    dt = (bigWsize - noverlap)/fs

    Label=['Max abRatio', 'Var abRatio']

    # calculate Max ABratio and Var abRatio
    abRatio = feature_ABRatio(sig,fs,shortWsize,bigWsize,noverlap,step,rmvf=rmvf)
    abRatio = np.array(abRatio)

    # fig, axes = plt.subplots(2,2)
    # fig.suptitle(figname)
    DelayTime =[]
    buffer_length = int(buffer_dur/dt)
    for i in range(abRatio.shape[1]):
        delayT=[]
        # NoiseLevel = np.abs(np.quantile(abRatio[0:40,i],quantile_Level)-np.quantile([abRatio[0:40,i]],1-quantile_Level))

        adapted,std_abratio = adaptiveThreshold(abRatio[:,i],beta1,beta2,thr,buffer_length)
        if not isinstance(adapted, np.ndarray):
            adapted = np.array(adapted)
        if not isinstance(std_abratio, np.ndarray):
            std_abratio = np.array(std_abratio)



        if plot:
            T_= np.arange(len(adapted))*dt
           
            delay = plotOutBound(T_, abRatio[:,i],adapted,std_abratio,thr,tauU_outBound =tauU_outBound, tauD_outBound = tauD_outBound, Label ='{},{}'.format(Label[i],figname), alarm_threshold=alarm_threshold,alarm_dur =alarm_dur,tst=tst,savePlot = savePlot, saveFolder = saveFolder, figname = figname.replace(',','_') )
            delayT.append(delay)
       
       

        # delayT = np.nanmin(np.array(delayT))

        try:
            minDelay = delayT[0]
        except:
            minDelay = delayT
        DelayTime.append(minDelay)
        

    
    return DelayTime
