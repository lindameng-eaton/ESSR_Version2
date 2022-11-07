import numpy as np
from ..utility import *
from .calculate_metrics import *

def plotTimeCorrelation(sig, column_list, wsize, fs, thr, rmvf=None, saveFolder='.\\', figname='',  savePlot=False, tauU_outBound=0.4, tauD_outBound=0.1, alarm_threshold=0.5, alarm_dur=1, tst=None, plot=False):
    """plot time correlation of the given signal or signal matrix
    Args:
    sig: numpy array or array-like matrix
    column_list: list of column names
    wsize: window size for moving window
    fs: sample frequency
    thr: threshold factor of the baseline variance
    rmvf:remove frequency for the high pass filter. Default=None
    savePlot: boolean. True: save plot; False: not save
    tauU_outBound, tauD_outBound: time constant for the decision logic
    plot: boolean. True: plot time correlation, False: not plot 
    figname: string for the signal file
    alarm_threshold: the threshold for the decision logic 
    alarm_dur: duration for alarm continuously being out of boundary to report fault
    tst: start time

    Output: the minimal time for identify fault along all the chanels in the signal matrix
    """

    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)

    noverlap = 0
    # reshape sig
    L = len(sig)
    sig = sig.reshape(L, -1)
    nrow = sig.shape[-1]
    deltat = wsize/fs

    delayT = []

    for i in range(nrow):
        dt = sig[:, i]

        dt_matrix = dataRearrange(dt, noverlap, wsize)

        corr = [pearson_corr(highPassFilter(dt_matrix[j, :], fs, rmvf), highPassFilter(
            dt_matrix[j+1, :], fs, rmvf)) for j in range(dt_matrix.shape[0]-1)]
        corr = np.array(corr)
        # corr_noiseLevel = np.abs(np.quantile(corr[0:10],quantile_Level)-np.quantile([corr[0:10]],1-quantile_Level))
        # adapted_corr, std_corr = adaptiveThreshold(corr, beta1,beta2,thr, NoiseLevel = corr_noiseLevel)
        # adapted_corr = np.array(adapted_corr)
        # std_corr =np.array(std_corr)
        # T = np.arange(len(corr))*deltat
        # plotOutBound(T,corr,adapted_corr,std_corr,corr_noiseLevel, thr, Label ='Time Correlation {}'.format(column_list[i]),ax = ax[i])
        T = np.arange(len(corr))*deltat
        # N = int(0.5/deltat)
        # corr_noiseLevel = np.abs(np.quantile(corr[0:N],quantile_Level)-np.quantile([corr[0:N]],1-quantile_Level))
        # adapted_corr, std_corr = adaptiveThreshold(corr, beta1,beta2,thr, NoiseLevel =corr_noiseLevel)
        # adapted_corr = np.array(adapted_corr)
        adapted_corr = np.ones((len(corr),))
        std_corr = np.ones((len(corr),))
        delay_time = plotOutBound(T, corr, adapted_corr, std_corr,  thr, tauU_outBound=tauU_outBound, tauD_outBound=tauD_outBound, Label='Time correlation_{}, {}'.format(
            column_list[i], figname), alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot, saveFolder=saveFolder, figname='Timecorr_{}'.format(column_list[i])+figname, plot=plot)
        delayT.append(delay_time)

    delayT = np.array(delayT)
    min_delay = np.nanmin(delayT)
    if isinstance(min_delay, np.ndarray):
        min_delay = min_delay[0]

    return min_delay
