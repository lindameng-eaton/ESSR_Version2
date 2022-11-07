import numpy as np
from ..utility import *


def rms(sig):
    """
    calculate the RMS of the given sig and return the value
    """
    return np.sqrt(np.mean(sig**2))

def calculate_skewness(sig, mu, sigma):
    """
    calculate skewness of given sig to quantify the symmetric of given signal.
    Args:
    sig: signal to be calculated
    mu: mean of sig
    sigma: std of sig
    Output: the skewness of sig

    """
    skew = np.mean(((sig-mu)/sigma)**3)
    return skew

def calculate_kurtosis(sig, mu, sigma):
    """calculate kurtosis of given sig to quantify the tailness 
    Args:
    sig: signal to be analyzed
    mu: mean
    sigma: standard deviation

    Output:
    kurtosis

    """
    kurtosis = np.mean(((sig-mu)/sigma)**4)
    return kurtosis


def feature_ABRatio(sig, fs, shortWsize, bigWsize, noverlap, step=1, rmvf=None):
    """
    calculate ABRatio of sig using moving window strategy
    Args:
    sig: signal to be calculated
    fs: sample frequency
    shortWsize: short window size
    bigWsize: big window size
    noverlap: number of overlapping
    rmvf: removed frequency list 
    outputs: tuple list each element of which is tuple (peakAbratio, VarAbratio)

    """

    sig_new = dataRearrange(sig, noverlap, bigWsize)

    abRatio = [_shortWindow_rms(highPassFilter(sig_new[i, :], fs, rmvf), max(
        1, np.int32(shortWsize)), step) for i in range(sig_new.shape[0])]

    return abRatio

def _shortWindow_rms(sig, wsize, step=1, noiseLevel=1.e-6):
    """
    calculate the peak and variance small window RMS
    Args:
    sig: big window signal
    wsize: small window size
    step: sliding window step
    noiseLevel: noise level
    Output: tuple (peakRMS, varRMS)
    """

    temp = [rms(sig[i:i+wsize]) for i in range(0, len(sig)-wsize+1, step)]

    return (max(temp)/max(rms(sig), noiseLevel), np.std(temp)/max(rms(sig), noiseLevel))

def Sum_Of_Difference(sig):
    """
    calculate the sum of the difference between adjant rows of given signal matrix
    Args:
    sig: two dimension numpy array
    Ouput:
    a numpy array with the summed difference between adjant rows of the given sig.
    """
    diff_sig = np.abs(np.diff(sig, axis=0))

    return np.sum(diff_sig, axis=-1)
