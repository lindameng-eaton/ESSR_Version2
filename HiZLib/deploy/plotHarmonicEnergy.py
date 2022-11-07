import numpy as np
import matplotlib.pyplot as plt
from .calculate_metrics import *
from ..utility import *


def plotHarmonicEnergy(sig, figname, wsize, noverlap, fs, compname, selectedComp, Npad, P, beta1, beta2, thr, buffer_dur=0.5, ymin=None, ymax=None, foldername=".\\", plot=True, plotSave=False):
    """
    plot the time series of harmonic energy with adaptive mean and variance
    Args:
    sig: current or voltage signals
    figname: figure name
    wsize: window size for STFT 
    noverlap: number of overlapping of moving window in STFT 
    fs: sample frequency
    compname: name of frequency components, such as 3rd, odd, even, ...
    selectedComp: frequency list/array 
    Npad: the number of zero padding
    P: RV window parameter
    beta1: time constant for adaptive mean
    beta2: time constant for adaptive variance
    thr: threshold 
    buffer_dur: buffer duration for noise calculation
    ymin: minimum of y axis. if None, it will be the minimum of harmonic energy
    ymax: maximum of y axis. if None, it will be the maximum of harmonic energy

    Outputs: tuple list  the outboundary rate N

    """

    # if not NoiseLevel_f:
    #     NoiseLevel_f =100

    # energy for the selected frequency components
    energy = calculate_signal_HarmonicEnergy(
        sig, wsize, noverlap, fs, selectedComp, Npad, P)

    # cast energy into numpy array
    energy = np.array(energy)
    energy_db = 20*np.log10(energy)

    # NoiseLevel = np.abs(np.quantile(energy_db[0:10], quantile_Level) - np.quantile(energy_db[0:10], 1-quantile_Level))
    # time steps for energy series
    deltaT = (wsize-noverlap)/fs
    T = np.arange(1, len(energy_db)+1)*deltaT

    buffer_length = int(buffer_dur/deltaT)
    # adaptive mean and adaptive standard deviation
    adaptedEnergy, std_Energy = adaptiveThreshold(
        energy_db, beta1, beta2, thr, buffer_length)

    if not ymin:
        ymin_ = energy_db.min()
    if not ymax:
        ymax_ = energy_db.max()

    # upper bound and lower bound
    threshold = std_Energy*thr
    # threshold[threshold < NoiseLevel] = NoiseLevel
    std_Energy = np.array(std_Energy)
    upb = threshold + adaptedEnergy
    lwb = adaptedEnergy-threshold

    # out of boundary rate
    outBound = np.abs(energy_db-adaptedEnergy) > threshold
    T_ = T[outBound]

    t0 = T[0]
    if len(T_) > 0:
        N = sum(T_ > t0)/(T[-1]-t0)
    else:
        N = 0
    energy_ = energy_db[outBound]
    energy_ = energy_[T_ > t0]
    T_ = T_[T_ > t0]

    # plot
    fig = plt.figure()
    plt.plot(T, energy_db, color='black', linewidth=0.5, label='energy')
    # plt.plot(T, smoothedEnergy, color='green', linewidth=1,label='smoothed')
    if plot:
        plt.plot(T_, energy_, 'g*')

        plt.plot(T, upb, 'r--', linewidth=1)
        plt.plot(T, lwb, 'r--', linewidth=1)
        plt.plot(T, adaptedEnergy, color='red', linewidth=1, label='adapted')
    plt.xlim(t0, T.max())
    plt.ylim([ymin_-std_Energy.max()*2, ymax_+std_Energy.max()*2])

    plt.title(figname+'{}, OBR: {:.2f}'.format(compname, N))
    plt.legend()
    plt.show()

    # save plot
    if plotSave:

        figname = figname + '.png'
        savename = os.path.join(foldername, figname)
        fig.savefig(savename, format='png')

    return N
