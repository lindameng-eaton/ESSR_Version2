import numpy as np
from .calculate_metrics import *
from ..utility import *


def plotEnergyCorrelation(sig, column_list, bigwsize, smallwsize, step, fs, fmin, fmax, thr=2, buffer_dur=0.5, beta1=0.99, beta2=0.99, tauU_outBound=0.4, tauD_outBound=0.1):
    """plot the correlation of the energy changes between two big adjant time-window signals
    sig: numpy array or array like signal maxtrix
    column_list: list of columns for the signal matrix
    bigwsize: the big window size for calculate energy change
    smallwsize: small window size for calculate fft components
    step: step for moving small window within the big window
    fs: sample frequency
    fmin: lower side of the frequency range
    fmax: upper side of the frequency range
    thr: threshold factor for baseline variance
    beta1 and beta2: time constant for adaptiveMean and adaptive variance
    buffer_dur: duration for the buffer storing std of the dwt components
    tauU_outBound, tauD_outBound: time constant for the decision logic


    """
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)

    noverlap = 0
    # reshape sig5
    L = len(sig)
    sig = sig.reshape(L, -1)
    nrow = sig.shape[-1]
    deltat = bigwsize/fs
    buffer_length = int(buffer_dur/deltat)

    delta_f = fs/smallwsize
    freq_list = np.arange(0, fs/2, delta_f)
    indx = (freq_list >= fmin) & (freq_list <= fmax)
    harmonicComponents = freq_list[indx]

    for i in range(nrow):
        dt = sig[:, i]
        dt_matrix = dataRearrange(dt, noverlap, bigwsize)
        corr = []
        for j in range(dt_matrix.shape[0]-1):
            sig1 = dt_matrix[j, :]
            sig2 = dt_matrix[j+1, :]
            energy1 = calculate_signal_HarmonicEnergy(
                sig1, smallwsize, step, fs, harmonicComponents, smallwsize, 1)
            energy2 = calculate_signal_HarmonicEnergy(
                sig2, smallwsize, step, fs, harmonicComponents, smallwsize, 1)
            corr.append(pearson_corr(energy1, energy2))
        corr = np.array(corr)
        # corr_noiseLevel = np.abs(np.quantile(corr[0:10],quantile_Level)-np.quantile([corr[0:10]],1-quantile_Level))
        adapted_corr, std_corr = adaptiveThreshold(
            corr, beta1, beta2, thr, buffer_length)
        adapted_corr = np.array(adapted_corr)
        std_corr = np.array(std_corr)
        T = np.arange(len(corr))*deltat
        plotOutBound(T, corr, adapted_corr, std_corr, thr, tauU_outBound=tauU_outBound,
                     tauD_outBound=tauD_outBound, Label='Energy Correlation {}'.format(column_list[i]))

