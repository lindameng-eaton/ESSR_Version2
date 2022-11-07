import numpy as np
import pywt
from .calculate_metrics import *
from ..utility import *


def deltaEnergy(Energy, dt, tauUp, tauDown):
    """
    Calculate the difference between the original signal and the smoothed one
    Args:
    Energy: energy series
    dt: time step
    tauUp: time constant for upside integration
    tauDown: time constant for down integration
    Outputs: delta Energy and smoothed Energy

    """
    # low pass Energy signals
    smoothed = lowPassFilter(tauUp, tauDown, Energy, dt, Energy[0])

    # subtract the smoothed Energy from the original sig
    delta = Energy - smoothed

    # return delta Energy and smoothed Energy
    return delta, smoothed

def calculate_CRF(sig, noiseLevel=1.e-6):
    """calculate crest factor of given signal
    Args:
    sig: signal to be analyzed
    noiseLevel: noise level
    output:
    crest factor

    """
    # take the absolute value
    abs_sig = np.abs(sig)
    return np.max(abs_sig) / max(np.mean(abs_sig), noiseLevel)

def wavedec_std(sig, wsize, wstep, fs, wavletname, decompLevel, level=4, plot=False, beta1=0.99, beta2=0.99, buffer_dur=0.5, thr=2, tauU_outBound=0.4, tauD_outBound=0.1, figname=[''], alarm_threshold=0.5, alarm_dur=1, tst=None, savePlot=False, saveFolder='.\\'):
    """calculate standard deviance of the wavelet decomposition components of sigals within moving window
    Args:
    sig: numpy array or array-like matrix
    wsize: window size for moving window
    wstep: step to move the window
    fs: sample frequency
    wavletname:the name of wavelet kernel
    decompLevel: the level to use for calculate standard deviance
    level: the max level for dwt 
    plot: boolean. True: plot time correlation, False: not plot 
    beta1 and beta2: time constant for adaptiveMean and adaptive variance
    buffer_dur: duration for the buffer storing std of the dwt components
    thr: threshold factor of the baseline variance
    figname: name list for the plotted signal matrix
    tauU_outBound, tauD_outBound: time constant for the decision logic
    alarm_threshold: the threshold for the decision logic 
    alarm_dur: duration for alarm continuously being out of boundary to report fault
    savePlot: boolean. True: save plot; False: not save


    """
    # make sure sig is np.ndarray
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)

    L = len(sig)
    sig = sig.reshape(L, -1)
    Std_WV = []
    FF = []
    delayT = []

    for i in range(sig.shape[1]):

        data_M = [sig[j:j+wsize, i] for j in np.arange(0, len(sig), wstep)]

        w = pywt.Wavelet(wavletname)

        std_wv = []
        deltat = (wstep)/fs
        buffer_length = int(buffer_dur/deltat)

        for k in np.arange(len(data_M)):
            mx = np.max(np.abs(data_M[k]))
            coeffs = pywt.wavedec(data_M[k]/mx, w, level=level)

            std_wv.append(np.std(coeffs[decompLevel]))

        std_wv = np.array(std_wv)

        T_ = np.arange(len(data_M)) * deltat
        # stdWV_noiseLevel= np.abs(np.quantile(std_wv[0:20],quantile_Level)-np.quantile([std_wv[0:20]],1-quantile_Level))
        adapted_stdWV, std_stdWV = adaptiveThreshold(
            std_wv, beta1, beta2, thr, buffer_length)
        delay = plotOutBound(T_, std_wv, adapted_stdWV, std_stdWV, thr, tauU_outBound=tauU_outBound, plot=plot, tauD_outBound=tauD_outBound, Label='wavelet decomposition lvl {},{}'.format(
            decompLevel, figname[i]), alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot, saveFolder=saveFolder, figname=figname[i].replace(',', '_'))
        delayT.append(delay)
        Std_WV.append(std_wv)

    delayT = np.nanmin(np.array(delayT))
    try:
        minDelay = delayT[0]
    except:
        minDelay = delayT
    return minDelay


def computeMI(x, y):
    """calculate mutrual information of x and y

    """
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([len(x[x == xval])/float(len(x))
                  for xval in x_value_list])  # P(x)
    Py = np.array([len(y[y == yval])/float(len(y))
                  for yval in y_value_list])  # P(y)
    for i in range(len(x_value_list)):
        if Px[i] == 0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy) == 0:
            continue
        pxy = np.array([len(sy[sy == yval])/float(len(y))
                       for yval in y_value_list])  # p(x,y)
        t = pxy[Py > 0.]/Py[Py > 0.] / Px[i]  # log(P(x,y)/( P(x)*P(y))
        # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
        sum_mi += sum(pxy[t > 0]*np.log2(t[t > 0]))
    return sum_mi
