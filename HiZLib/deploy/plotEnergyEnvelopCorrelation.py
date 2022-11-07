import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .calculate_metrics import *
from ..utility import *


def plotEnergyEnvelopCorrelation(sig, column_list, wsize, fs, fmin, fmax):
    """plot the correlation of the energy envelop between two adjant time-window signals
    sig: numpy array or array like signal maxtrix
    column_list: list of columns for the signal matrix
    wsize: window size
    fs: sample frequency
    fmin: lower side of the frequency range
    fmax: upper side of the frequency range

    """

    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)

    noverlap = 0
    # reshape sig5
    L = len(sig)
    sig = sig.reshape(L, -1)
    nrow = sig.shape[-1]
    deltat = wsize/fs
    fig, ax = plt.subplots(nrow, 1, figsize=(6, 12))
    fig.suptitle('Energy correlation')
    clr = sns.color_palette(None, nrow)

    for i in range(nrow):
        dt = sig[:, i]
        dt_matrix = dataRearrange(dt, noverlap, wsize)
        corr = [calculate_FFtCorr(dt_matrix[j, :], dt_matrix[j+1, :], fs, fmin, fmax)
                for j in range(dt_matrix.shape[0]-1)]
        corr = np.array(corr)

        T = np.arange(len(corr))*deltat
        try:
            ax[i].plot(T, corr, color=clr[i], label=column_list[i])
            ax[i].set_ylim([0, 1.1])
            ax[i].legend()
        except:
            ax.plot(T, corr, color=clr[i], label=column_list[i])
            ax.set_ylim([0, 1.1])
            ax.legend()
