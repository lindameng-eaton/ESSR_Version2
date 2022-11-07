import numpy as np
from ..utility import *


def pearson_corr(x1, x2):
    """return correlationcoefficient of x1 and x2
        input:
            x1, x2: signal vectors
        output:
            scalor: correlation coefficiente
    """
    return np.abs(np.corrcoef(x1, x2)[0, 1])


def calculate_FFtCorr(dt1, dt2, fs, fmin=None, fmax=None):
    """calculate the correlation of the fft envelops within [fmin, fmax] from dt1 and dt2 
    Args:
    dt1, dt2: time signals with same lengths
    fmin, fmax: frequency boundary
    fs: sample frequency

    output:
    Pearson correlation between two fft amplitude envelops with minimum frequency =fmin and maximum frequency=fmax
    """
    # calculate fft components for dt1, dt2
    fft_freq1, fft_dt1 = myfft(dt1, fs)
    fft_freq2, fft_dt2 = myfft(dt2, fs)

    # extract the fft components within frequency range: [fmin, fmax]
    if (not fmin) and (not fmax):
        fft_amp1 = np.abs(fft_dt1)
        fft_amp2 = np.abs(fft_dt2)
    elif not fmin:
        fft_amp1 = np.abs(fft_dt1[fft_freq1 <= fmax])
        fft_amp2 = np.abs(fft_dt2[fft_freq2 <= fmax])
    elif not fmax:
        fft_amp1 = np.abs(fft_dt1[fft_freq1 >= fmin])
        fft_amp2 = np.abs(fft_dt2[fft_freq2 >= fmin])
    else:
        fft_amp1 = np.abs(fft_dt1[(fft_freq1 >= fmin) & (fft_freq1 <= fmax)])
        fft_amp2 = np.abs(fft_dt2[(fft_freq2 >= fmin) & (fft_freq2 <= fmax)])
    # return correlation between fft_amp1 and fft_amp2

    return pearson_corr(fft_amp1, fft_amp2)


def harmonicEnergy(sig,  fs, selectedHarmonic,  P=1, paddingM=None):
    """
        calculate the power of selected harmonics of sig (DB/Hz)
        Args:
            sig: input signal for analysis
            fs: sample freq
            selectedHarmonic: the seelcted harmonic list for calculating Energy
            window: window kernel for sig
            paddingM: zero padding
        output:
            total power for the harmonic components


    """
    if not paddingM:
        paddingM = len(sig)

   # DFT
    _, fftH = myfft(sig, fs, selectedHarmonic, P, paddingM)

    return np.sum(np.abs(fftH)**2/2)


def calculate_signal_HarmonicEnergy(sig, wsize, noverlap, fs, harmonicComponents, paddingM, P):
    """
    calculate the selected harmoniceComponents'Energy with time
    Input:
        sig: the signal array or series

        wsize: window size 
        noverlap: the number of points of overlapping between two windows
        fs: sample frequency
        harmonicComponents: the selected frequency list
        paddingM: the zero padding
        P: for window function
    Output: list of energy


    """
    # data rearrange into matrix
    sig_new = dataRearrange(sig, noverlap, wsize)

    tempEnergy = []

    for i in range(sig_new.shape[0]):

        sig_ = sig_new[i, :]

        # calculate the total energy corresponding to the harmonicComponents
        temp = harmonicEnergy(sig_, fs, harmonicComponents, P, paddingM)

        # append temp into tempEnergy list
        tempEnergy.append(temp)

    return tempEnergy
