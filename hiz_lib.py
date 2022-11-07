
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import Counter
import pywt
import pickle
import os
from skimage import measure
import math
import sys
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold,train_test_split
# import seaborn as sns   
import tensorflow as tf
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from livelossplot import PlotLossesKeras

from scipy import signal
from scipy.signal import butter

# from sklearn.externals import joblib
import joblib
# from ast_lib import iir_filter
# sys.path.append("C:\\Users\\E0644597\\OneDrive - Eaton\\CIP\\PythonCode_Linda\\azure-storage-repo\\azure-data-storage")
# sys.path.append("C:\\Users\\E0644597\\OneDrive - Eaton\\CIP\\PythonCode_Linda\\ESSR-Repo\\Denoiser\\denoise")
# from denoise import Denoiser


from azure_utils.azure_lib import *
from HiZLib.data_preprocessing import data_acquisition, TIME_COLUMN, read_data, estimate_breaker_status
from HiZLib.utility.calculate_metrics import calculate_sample_freq
from HiZLib.utility.filter import filter_butterworth


# temporary parameters
PLOT_SAVE_PATH = r".\plots"
# PLOT_SAVE_PATH = r".\\plot_timecorr_franskville"


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')


def _resample(sig, fs_curr, fs_target):
    """resample data to the sample rate = fs_target from fs_curr
    Args:
    sig: array like the data to be resampled
    fs_curr: int, current sample frequency
    fs_target: int, target sample frequency
    Outputs:
    resampled data, numpy array
    """
    sig_ = signal.resample_poly(sig, fs_target, fs_curr)

    return sig_


def next_power_of_2(x):  
    """
    calculate the nearest power 2 num for given value x
    Args: 
        x: value
    return: nearest power 2
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()
    

def _rv_win(P, N):
    '''rifevincent window
    when P=0, it is a square window; when P==1, it is hanning window
    Args:
    P: order for rifevincent window
    N: window length
    Output: a numpy array for rifevincent window
    '''

    n = np.arange(N)
    
    if P == 0:
        win = np.ones((N,))
    elif P > 0:
        # p = 0 outsde loop
        p = 0
        C = math.comb(2 * P, P)
        DL = C / (2 ** (2 * P))
        win = DL * np.cos(2 * np.pi * n * p / N)
        # p > 0 inside loop
        for i in np.arange(1, P + 1):
            p = i
            C = math.comb(2 * P, P - p)
            
            DL = ((-1) ** p) * (C / (2 ** (2 * P - 1)))
            win = win + DL * np.cos(2 * np.pi * n * p / N)
        
    else:
        print('P must be nonnegative')
        win = np.nan

    return win

    
def _calculate_sigma(delta, P):
    sigma_term = 1
    for g in np.arange(1, P + 1):
        sigma_term = sigma_term * ((g * g) - (delta * delta))
    return sigma_term
    

def myfft(sig, fs, FreqList=[], P=1, paddingM=None):
    """
    calculate harmonic component for given sig and return the power for freqs
    where freqs = (0:1:fix(fs/s/f0) *f0
    
    Args:
        sig: signal to be analyzed
        fs: sample frequency
        FreqList: numpy array or list of interested frequencies
        P: parameter for rv window function. P==1 hamming window; P==0: square window
        paddingM: zero padding.
    return tuple : (freqs, the fft components of harmonic frequencies)
    """
    N = len(sig)
    if not paddingM:
        paddingM = N

    if not isinstance(FreqList, np.ndarray):
        FreqList = np.array(FreqList)
    
    # window function, P==1 hamming window, P==0: square window, N is the length of the window
    window = _rv_win(P, N)

    # multiply signals with window function
    sig = sig * window

    # calculate frequency components 
    fftsig = np.fft.rfft(sig, paddingM) / N
    freqsig = np.fft.rfftfreq(paddingM, 1 / fs)
    fftsig = fftsig * 2
    fftsig[0] /= 2
    
    # calculate correct factor for window function
    factor = ((2 ** (2 * P)) / math.factorial(2 * P)) * _calculate_sigma(0, P)
    fftsig = fftsig * factor
    
    # find the frequency spectrum for given FreqList using nearist method
    if not len(FreqList):
        FreqList = freqsig
        y2 = fftsig
    else:
        indx = np.argmin(abs(FreqList.reshape(1, -1) - freqsig.reshape(-1, 1)), axis=0)
        y2 = fftsig[indx]
    # return 
    return (FreqList, y2)


def find_index(freq_array, f_target):
    """find the index in freq_array that freq_array[index] is the closest to f_target
    """
    if not isinstance(freq_array, np.ndarray):
        freq_array = np.array(freq_array)
    if not isinstance(f_target, np.ndarray):
        f_target = np.array(f_target)
    indx = np.argmin(abs(f_target.reshape(1, -1) - freq_array.reshape(-1, 1)), axis=0)
    return indx


def lowPassFilter(tauU, tauD, xin, dt, y0=None):
    """
        lowpass filter:
        dy/dt = -(y-xin)/tau
        Args:
            tauU is the time constant on the rising phase
            tauD is the time constant on the down phase
            xin is the input signal
            dt is the time step
            y0 is the initial state for y
        output: the lowpassed data for xin
    """
    if not y0:
        try:
            y0 = xin[0]
        except:
            y0 = xin
    
    tempy = y0
    Y = []
    for i in xin:
    
        if i >= tempy:
            y = tempy + (i - tempy) * min(1, dt / tauU)
        else:
            y = tempy + (i - tempy) * min(1, dt / tauD)
    
        Y.append(y)
        tempy = y
        
    return Y


def lowPassFilter_1(tauU, tauD, xin, dt, tempy):

    """
        lowpass filter:
        dy/dt = -(y-xin)/tau
        Args:
            tauU is the time constant on the rising phase
            tauD is the time constant on the down phase

            xin is the input 
            dt is the time step
            y0 is the initial state for y
        output: the lowpassed data for xin
    """
    
    if xin >= tempy:
        y = tempy + (xin - tempy) * min(1, dt / tauU)
    else:
        y = tempy + (xin - tempy) * min(1, dt / tauD)

    return y   
    

def highPassFilter(sig, fs, rmvf=None):

    """
    high pass signal with fft components in rmvf removed
    return high passed signal
    Args:
        sig: signal to be analyzed
        rmvf: frequency list to be removed
    Output:
        a numpy array with filtered signal
    """
    # if nothing to remove, return sig
    if not rmvf:
        return sig
    # if not isinstance(rmvf,np.ndarray):
    #     rmvf = np.array(rmvf)

    # calculate fft component using myfft
    N = len(sig)
    freq = np.fft.fftfreq(N, 1 / fs)
    freq = freq[0:N // 2]
    
    freq, X_fft = myfft(sig, fs, freq, P=0)

    # find the freq indices that are going to remove
    indx = np.argwhere(freq <= rmvf)
    
    # add up all the removing components
    temp = np.zeros(N)
    n = np.arange(N)
    for k in indx:
        temp = temp + (X_fft[k] * np.exp(1j * 2 * np.pi * n * k / N)).real

    # subtract the removing sum from the given sig    
    r = sig - temp
    return r


def pearson_corr(x1, x2, perc=0.9):
    """return correlationcoefficient of x1 and x2
        input:
            x1, x2: signal vectors, one-dimension
            NoiseLevel: Noise Level
        output:
            scalor: correlation coefficiente
    """

       
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)
    if not isinstance(x2, np.ndarray):
        x2 = np.array(x2)

    corr_x = np.abs(np.corrcoef(x1, x2)[0, 1])
    
    return corr_x


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


def IIRfilter(y, yt, beta):
    """
    Args: 
    y: previous y
    yt: current y
    beta: parameter for the integration time constant
    Output: updated y
    
    """
    return beta * y + (1 - beta) * yt


def rms(sig):
    """
    calculate the RMS of the given sig and return the value
    """
    return np.sqrt(np.mean(sig ** 2))


# def adaptiveThreshold(sig, beta1, beta2,T,NoiseLevel=1e-5, y0=None,df0=None,biasCorrect=True):
#     # approximately average 1/(1-beta) data points since the average winow will drop around 1/3 
#     avg=[]
#     sd =[]

#     if not y0:
#         y0=sig[0]
#         corrected_y = y0
#     if not df0:
#         df0=0
#         corrected_df0 = 0
        
#     for i,yt in enumerate(sig):

            
#         corrected_y = adaptiveMean(yt, beta1, i, corrected_y)
#         corrected_df0 = adaptiveVar(yt, beta2, i, corrected_y, corrected_df0, T, NoiseLevel)
        

#         avg.append(corrected_y)
#         sd.append(corrected_df0)
        
#     return (avg,sd) 

def adaptiveMean(yt, beta1, i, y0=None, biasCorrect=False):
    """
    Calculate the mean value of y(t) using moving window average
    Argus:
    yt: current value of y
    beta1: integration time constant
    i: current step
    y0: initial value of y
    biasCorrect: True: correct the first few steps by (1-beta1**i);
                 False: no correction
    Output: updated mean

    """
    # if not y0:
    #     y0=yt
    
    y0 = IIRfilter(y0, yt, beta1)
    
    if biasCorrect:
        corrected_y = y0 / (1 - beta1 ** i)
        
    else:
        corrected_y = y0
    
    return y0, corrected_y


def adaptiveVar(yt, beta2, i, ymean, df0, thr, NoiseLevel=1e-5, biasCorrect=False):
    """
    calculate the Variance of y(t) using moving window average. Only integrate variance under threshold
    Args:
    yt: current value
    beta2: time constant for integration window
    i: current step
    ymean: previous mean value
    df0: previous std
    thr: threshold 
    
    Outputs: current std
    """

    # calculate the current difference
    v = np.abs(ymean - yt)

    # if v is less than noiselevel and the threshold or current step i is less than 1/(1-beta2), 
    # adaptive variance is continuously integrated, otherwise freeze integration
    if v <= max(df0 * thr, NoiseLevel) or i <= round(1 / (1 - beta2)):

        # update df0 using iirfilter   
        df0 = IIRfilter(df0, v, beta2)

        # correct df0 if biasCorrect is True
        if biasCorrect:
            corrected_df0 = df0 / (1 - beta2 ** i)
        else:
            corrected_df0 = df0
    else:
        corrected_df0 = df0
            
    return corrected_df0


def calculate_accumulative_avg(CA_k, xt, k):
    """
    calculate accumulative avg
    CA_k+1 = CA_k + (xt-CA_k)/(k+1)
    Args:
    CA_k: float32, previous avg
    xt: float32, current value
    k: current step
    """
    return CA_k + (xt - CA_k) / (k + 1)
  

def adaptiveThreshold(sig, beta1, beta2, thr, buffer_length, y0=None, df0=None, biasCorrect=False):
    """
    Approximately average 1/(1-beta) data points for mean and std when the average winow will drop around 1/3 
    Args:
    sig: signal for estimating mean and std
    beta1: time constant for mean estimation
    beta2: time constant for std estimation
    thr: threshold
    buffer_length: buffer length for calibrating baseline variance
    y0: initial value of sig
    df0: initial value of std
    biasCorrect: True: correct the first few steps by (1-beta1**i);
                 False: no correction 
    
    Outputs: estimated adaptive mean and std
    """

    # init
    avg = []
    sd = []

    if not y0:
        y0 = sig[0]
        corrected_y = y0
    if not df0:
        df0 = 0
        corrected_df0 = 0
   
    sig_buffer = np.ones((buffer_length, 1)) * np.nan
    sig_var = 0

    # loop through all the data points    
    for i, yt in enumerate(sig):
            
        corrected_y = adaptiveMean(yt, beta1, i, corrected_y, biasCorrect=biasCorrect)
        if np.abs(yt - corrected_y) < sig_var * thr or i < buffer_length:
            sig_buffer = np.append(sig_buffer, yt)
            sig_buffer = sig_buffer[1:]
            v = np.nanstd(sig_buffer)
            sig_var = adaptiveMean(v, beta2, i, sig_var)
        # corrected_df0 = adaptiveVar(yt, beta2, i, corrected_y, corrected_df0, thr, NoiseLevel, biasCorrect = biasCorrect)

        avg.append(corrected_y)
        sd.append(sig_var)

    # output avg and sd
    avg = np.array(avg)
    sd = np.array(sd)    
    return (avg, sd)


def calculate_skewness(sig, mu, sigma):
    """
    calculate skewness of given sig to quantify the symmetric of given signal.
    Args:
    sig: signal to be calculated
    mu: mean of sig
    sigma: std of sig
    Output: the skewness of sig
    
    """

    skew = np.mean(((sig - mu) / sigma) ** 3)
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
    
    kurtosis = np.mean(((sig - mu) / sigma) ** 4)
    return kurtosis


def rms(sig):
    """
    calculate the RMS of the given sig and return the value
    Args:
    sig: signal, numpy array
    output:
    root mean square
    """
    return np.sqrt(np.mean(sig ** 2))


def _shortWindow_rms(sig, wsize, step=1, NoiseLevel=1.e-6):
    """
    calculate  the small window RMSs
    Args:
    sig: big window signal
    wsize: small window size
    step: sliding window step
    NoiseLevel: noise level
    Output: peak RMS
    """
    mx = max(rms(sig), NoiseLevel)
    temp = [rms(sig[i:i + wsize]) / mx for i in range(0, len(sig) - wsize, step)]
    
    return temp


def dataRearrange(sig, noverlap, wsize):
    """
    rearrange the sig into a array list with each array as the segment of sig
    Args:
        sig: signal array
        wsize: the window size for each segment
        noverlap: the number of overlaps for nearby data segment
    output: numpy array that constains sig segments
    
    """
    L = len(sig)
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)
    sig_ = []
    step = wsize - noverlap
    wsize = np.int32(wsize)
    if noverlap == 0:
        nRow = np.int32(np.floor(L / wsize))
        sig_ = sig[0:nRow * wsize].reshape(nRow, -1)
        
        return sig_
    else:
        step = wsize - noverlap
        sig_ = [sig[i:i + wsize] for i in range(0, L - wsize + 1, step)]
        sig_ = np.array(sig_)

        return sig_
    

def feature_ABRatio(sig, fs, shortWsize, bigWsize, noverlap, step=1, NoiseLevel=1.e-6, rmvf=None):
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

    RMS_shortWindow = [_shortWindow_rms(highPassFilter(sig_new[i, :], fs, rmvf), max(1, np.int32(shortWsize)), step) for
                       i in range(sig_new.shape[0])]

    Peak_abRatio = [np.max(temp) for temp in RMS_shortWindow]
    Std_abRatio = [np.std(temp) for temp in RMS_shortWindow]

    return Peak_abRatio, Std_abRatio


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


def calculate_crossing(sig, threshold=0, minstep=0, nodirection=True):
    """
    compute the indices of the elements in sig that cross the threshold
    Args:
    sig: signal to be analyzed
    threshold: threshold for crossing
    minstep: the minimal steps between two crossing points
    nodirection: True: the algorithm will return all indices for crossing threshold 
                 False: only return the indices which crossing threshold from bottom to above
    Outputs: indices of the crossing points            
    """
    # if sig is not numpy array, cast it to np array
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)
        
    # compute the sign of the signal, if signal > threshold it will be 1 otherwise 0
    sign_sig = sig > threshold

    # if the length of the segments of 1s is less than minstep, set sign_sign =0
    if minstep > 0:
        # label connected regions
        label = measure.label(sign_sig)

        # unique number of labels
        unique_label = np.unique(label)

        # explore all the regions, if the length of the region is less than minstep, set the sign_sig within that region 0
        for i in unique_label:
            if sum(label == i) < minstep:
                sign_sig[label == i] = 0
    
    # extract the indices of the nonzeros
    if nodirection:
        crossing_indices = np.nonzero(np.diff(1 * sign_sig))[0]
    else:
        crossing_indices = np.nonzero(np.diff(1 * sign_sig) > 0)[0]
    
    # return induces
    return crossing_indices


def calculate_entropy(list_values):
    """calculate the entropy of the list values
    Args:
    list_value: list of the values for calculation
    Output: entropy
    """
    # count values 
    counter_values = Counter(list_values).most_common()
    # calculate probabilities
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    # calculate entropy
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    """
    calculate statistics of given data array including 5%, 25%,75%, 95% quantiles,
    mean, median, std, and fano factor (1/ SNR)
    Args:
    list_values: np array 
    output: A list that contains 5%, 25%,75%, 95% quantiles,
    mean, median, std, and fano factor
    """
    # 5%, 25%, 75%, 95%, 50% quantiles, mean, std, fanofactor
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)      
    var = np.nanstd(list_values)
    fano = var / mean
    
    # return a list
    return [n5, n25, n75, n95, median, mean, var, fano]


def harmonicEnergy(sig, fs, selectedHarmonic, P=1, paddingM=None):
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

    return np.sum(np.abs(fftH) ** 2 / 2)


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


def deltaEnergy(Energy, dt, tauUp, tauDown):
    """
    Calculate the difference between the original signal and the smoothed one
    Args:
    Energy: energy series
    dt: time step
    tauUp: time constant for upside integration
    tauDown: time constant for down integration
    Outputs: delta Energy and smoothed Energy
    Outputs:
    tuple, (delta energy, energy of the signal, smoothed)
    
    """
    # low pass Energy signals
    smoothed = lowPassFilter(tauUp, tauDown, Energy, dt, Energy[0])

    # subtract the smoothed Energy from the original sig
    delta = Energy - smoothed

    # return delta Energy and smoothed Energy
    return delta, smoothed


def feature_deltaEnergy(sig, wsize, noverlap, fs, harmonicComponents, tauUp, tauDown, paddingM=None, P=1):
    """calculate delta energy of given signal
    Args:
    sig: one-dimension array or array like
    wsize: window size, int
    noverlap: number of overlaping, int
    fs: sample frequency, int
    harmonicComponents: the harmonice components within deltaEnergy
    paddingM: number of padding, int
    tauUp: time constant for upside integration
    tauDown: time constant for down integration
    P: the order of the RV window
    Output:
    1-D list contains the delta energy time series
    
    """
    energy_list = calculate_signal_HarmonicEnergy(sig, wsize, noverlap, fs, harmonicComponents, paddingM, P)
    energy = np.array(energy_list)
    energy_db = 20 * np.log10(energy)
    dt = (wsize - noverlap) / fs
    delta_list, _ = deltaEnergy(energy_db, dt, tauUp, tauDown)
    delta_list = list(delta_list)

    return delta_list

   
def plotHarmonicEnergy(sig, figname, wsize, noverlap, fs, compname, selectedComp, Npad, P, beta1, beta2, thr,
                       buffer_dur=0.5, ymin=None, ymax=None, foldername=".\\", plot=True, plotSave=False):
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
    delta_energy, energy_db, smoothed = feature_deltaEnergy(sig, wsize, noverlap, fs, selectedComp, tauUp, tauDown,
                                                            paddingM=Npad, P=p)

    # # cast energy into numpy array
    # energy = np.array(energy)
    # energy_db = 20*np.log10(energy)

    # NoiseLevel = np.abs(np.quantile(energy_db[0:10], quantile_Level) - np.quantile(energy_db[0:10], 1-quantile_Level))
    # time steps for energy series
    deltaT = (wsize - noverlap) / fs
    T = np.arange(1, len(energy_db) + 1) * deltaT

    buffer_length = int(buffer_dur / deltaT)
    # adaptive mean and adaptive standard deviation
    adaptedEnergy, std_Energy = adaptiveThreshold(energy_db, beta1, beta2, thr, buffer_length)
    
    if not ymin:
        ymin_ = energy_db.min()
    if not ymax:
        ymax_ = energy_db.max()

    # upper bound and lower bound 
    threshold = std_Energy * thr
    # threshold[threshold < NoiseLevel] = NoiseLevel
    std_Energy = np.array(std_Energy)
    upb = threshold + adaptedEnergy
    lwb = adaptedEnergy - threshold

    # out of boundary rate
    outBound = np.abs(energy_db - adaptedEnergy) > threshold
    T_ = T[outBound]
  
    t0 = T[0]
    if len(T_) > 0:
        N = sum(T_ > t0) / (T[-1] - t0)
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
    plt.ylim([ymin_ - std_Energy.max() * 2, ymax_ + std_Energy.max() * 2])

    plt.title(figname + '{}, OBR: {:.2f}'.format(compname, N))
    plt.legend()
    plt.show()

    # save plot
    if plotSave:
        figname = figname + '.png'
        savename = os.path.join(foldername, figname)
        fig.savefig(savename, format='png')

    return N


def plotSpecgram(sig, P, figname, wsize, fs, noverlap, cmap='inferno', Npad=None, plotSave=False):
    """
    short-time fourial spectram of the signal
    Args:
        sig: signal 
        window: window for stfs, here using hamming window
        figname: the name for saving stsf figure
        wsize: window size 
        freq:sample frequency
        noverlap: number of overlapping
        fsize: figure size
        Npad: zeropadding  
        plotSave: default = False
    """
    if not Npad:
        Npad = len(sig)
    
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)

    # rv window    
    window = _rv_win(P, wsize)

    try:   
        fig, axes = plt.subplots(sig.shape[1], 1)
        axes[0].set_title(figname)
        for i in range(sig.shape[1]):
            y = sig[:, i]
            vmin = 20 * np.log10(np.max(y)) - 300
            _, _, _, im = axes[i].specgram(y, window=window, NFFT=wsize, Fs=fs, pad_to=Npad, noverlap=noverlap,
                                           vmin=vmin, cmap=cmap)
            axes[i].set_xlabel('t (s)')
            axes[i].set_ylabel('Hz')
            plt.colorbar(im, ax=axes[i])
        plt.tight_layout()

    except:
        fig = plt.figure()
        plt.title(figname)
        y = sig
        vmin = 20 * np.log10(np.max(y)) - 300
        _, _, _, im = plt.specgram(y, window=window, NFFT=wsize, Fs=fs, pad_to=Npad, noverlap=noverlap, vmin=vmin,
                                   cmap=cmap)
        plt.colorbar(im)
        plt.xlabel('t (s)')
        plt.ylabel('Hz')
    
    plt.show()
    if plotSave:
        figname = figname + '_STFT' + '.png'
        fig.savefig(figname, format='png', dpi=600)


def plotABRatio(sig, shortWsize, bigWsize, noverlap, fs, beta1, beta2, thr, buffer_dur=0.5, step=1, rmvf=None,
                plot=False, tauU_outBound=0.4, tauD_outBound=0.1, figname='', alarm_threshold=0.5, alarm_dur=1,
                tst=None, savePlot=False, saveFolder='.\\'):
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
    dt = (bigWsize - noverlap) / fs

    Label = ['Max abRatio', 'Var abRatio']

    # calculate Max ABratio and Var abRatio
    Peak_abRatio, Std_abRatio = feature_ABRatio(sig, fs, shortWsize, bigWsize, noverlap, step, rmvf=rmvf)
    Peak_abRatio = np.array(Peak_abRatio).reshape(-1, 1)
    Std_abRatio = np.array(Std_abRatio).reshape(-1, 1)
    abRatio = np.hstack(Peak_abRatio, Std_abRatio)

    # fig, axes = plt.subplots(2,2)
    # fig.suptitle(figname)
    DelayTime = []
    buffer_length = int(buffer_dur / dt)
    for i in range(abRatio.shape[1]):
        delayT = []
        # NoiseLevel = np.abs(np.quantile(abRatio[0:40,i],quantile_Level)-np.quantile([abRatio[0:40,i]],1-quantile_Level))

        adapted, std_abratio = adaptiveThreshold(abRatio[:, i], beta1, beta2, thr, buffer_length)
        if not isinstance(adapted, np.ndarray):
            adapted = np.array(adapted)
        if not isinstance(std_abratio, np.ndarray):
            std_abratio = np.array(std_abratio)

        if plot:
            T_ = np.arange(len(adapted)) * dt
           
            delay = plotOutBound(T_, abRatio[:, i], adapted, std_abratio, thr, tauU_outBound=tauU_outBound,
                                 tauD_outBound=tauD_outBound, Label='{},{}'.format(Label[i], figname),
                                 alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                                 saveFolder=saveFolder, figname=figname.replace(',', '_'))
            delayT.append(delay)
       
        # delayT = np.nanmin(np.array(delayT))

        try:
            minDelay = delayT[0]
        except:
            minDelay = delayT
        DelayTime.append(minDelay)
        
    return DelayTime


def plot_VoltageVsCurrent(d, chanel_name, fs, N, ncycle, thrp=0, figname='', savefolder_trace='', plotSave=False):
    """snapshot of voltage vs current trajectories
    Args:
      d: data matrix
      chanel_name: name of the chanel of voltage and current signals
      fs: sample frequency
      N: number of point per cycle
      ncycle: number of cycle per snapshot
      savefolder_trace: if plotSave == True, the snapshots of the trajectory will be saved into savefolder_trace
      thrp: percentage of the max value of Voltage
      figname: data file name
      plotSave: boolean. True: save figure False: not save
    
    """
    deltaT = 4  # s
    deltaN = int(deltaT * fs)

    if not isinstance(d, np.ndarray):
        d = np.array(d)

    if d.shape[1] > 5:
        dt = d
        plt.figure()
        for st in np.arange(0, d.shape[0], deltaN):
            thr = max(dt[:, 2]) * thrp
            plt.plot([thr, thr], [min(dt[:, 5]), max(dt[:, 5])])
            # for i, ax in enumerate(axes):
            if st < d.shape[0] - N * ncycle:
                plt.plot(dt[st:st + N * ncycle, 2], dt[st:st + N * ncycle, 5], label='st={}s'.format(st / fs))

                plt.legend()
                plt.xlabel('Voltage (V)')
                plt.ylabel('Current (A)')

        plt.title(figname)
        plt.show()

        if plotSave:
            savefile = os.path.join(savefolder_trace, 'Trace_' + figname + '_{}.png'.format(chanel_name))
            
            plt.savefig(savefile, format='png')

        fig, axes = plt.subplots(3, 2)
        T = d[:, 0]
        for i in range(3):
            v = d[:, i + 1]
            I = d[:, i + 4]
            thr = max(v) * thrp
            indx0 = calculate_crossing(v, threshold=thr, nodirection=False)
            T_ = T[indx0]
            I0 = I[indx0]
            axes[i, 0].plot(T_, I0, 'b.')
            axes[i, 1].plot(T_[1:], np.diff(I0), 'g.')

            axes[i, 1].set_ylim([-0.1, 0.1])
        plt.tight_layout()
        plt.show()



 
    else:
        print('this case doesnt contain both voltage and current')


def plot_EnergyRatio(d, fs, f0, wsize, freqComp, savefolder_energy=".\\", P=1, plotSave=False, Npad=None,
                     noverlap=None):
    """
    plot energyRatio for different frequency components in freqComp
    Args:
    d: table of different current or voltage channels
    fs: sample frequency
    f0: fundamental frequency
    wsize: window size for DFT
    freqComp: dictionary with different frequency components
    savefolder_energy: the folder path to save the energy ratio plots
    P: window function parameter. When p ==1, window function is hamming window
    plotSave: boolean. True: start to save the plots
    Npad: zero padding. Default is the same as wsize
    noverlap: number of overlap of sliding windows
    """
    if not noverlap:
        noverlap = int(np.fix(wsize * 1 / 2))
    if not Npad:
        Npad = wsize

    dt = d.copy()

    column_list = d.columns
    if not isinstance(dt, np.ndarray):
        dt = np.array(dt)
    
    nrow = len(freqComp)
    palette = sns.color_palette(None, nrow)
    deltaT = (wsize - noverlap) / fs

    for i in np.arange(0, dt.shape[1]):
        
        fig, axes = plt.subplots(nrow, 1)
        f_0 = [f0]
        Energy0 = calculate_signal_HarmonicEnergy(dt[:, i], wsize, noverlap, fs, f_0, Npad, P)
        
        T = np.arange(len(Energy0)) * deltaT
        Energy0 = np.array(Energy0)

        for j, comp in enumerate(freqComp):
            selectedComp = freqComp[comp]
            Energy = calculate_signal_HarmonicEnergy(dt[:, i], wsize, noverlap, fs, selectedComp, Npad, P)
            Energy = np.array(Energy)
            eRatio = np.log10(Energy / Energy0)

            axes[j].plot(T, eRatio, color=palette[j], label=comp)
            axes[j].set_xlim([0, T.max()])
            axes[j].legend()

        plt.title('EnergyRatio_{}'.format(column_list[i]))
        plt.show()
        if plotSave:
            savefile = os.path.join(savefolder_energy, 'EnergyRatio_{}'.format(column_list[i]))
            fig.savefig(savefile + '.png', format='png', dpi=600)


def plot_skewness_Kurtosis(dt, fs, column_list, wsize_statistic, noverlap_statistic, thr, buffer_dur=0.5,
                           saveFolder='.\\', figname='', beta1=0.99, beta2=0.99, savePlot=False, tauU_outBound=0.4,
                           tauD_outBound=0.1, alarm_threshold=0.5, alarm_dur=1, tst=None):
    """
    plot skewness and kurtosis of signal dt
    Args:
    dt: table matrix with current or voltage signals 
    fs: sample rate
    column_list: list of the column names in dt
    wsize_statistic: window size for calculate skewness and kurtosis
    noverlap_statistic: number of overlap
    thr: threshold parameter
    buffer_dur: buffer duration for calibrating the baseline variance
    savefolder_statistic: the folder path for saving skewness and kurtosis plots
    figname: name for the figure to be saved
    beta1 and beta2: the time constant for adaptive mean and adaptive variance calibration
    tauU_outBound, tauD_outBound: the time constant for the rising and falling phase of the fault indicator
    alarm_threshold: the threshold for the decision logic 
    alarm_dur: duration for alarm continuously being out of boundary to report fault
    tst: start time
    """
    deltat = (wsize_statistic - noverlap_statistic) / fs
    buffer_length = int(buffer_dur / deltat)
    
    if not isinstance(dt, np.ndarray):
        dt = np.array(dt)

    L = len(dt)
    dt = dt.reshape(L, -1)
    # column_list = dt.columns
    DTC_sk = []
    DTC_crf = []
    DTC_kurtosis = []
    
    for i in np.arange(dt.shape[1]):
        
        sig_modified = dataRearrange(dt[:, i], noverlap_statistic, wsize_statistic)
       
        sk = []
        kurtosis = []
        crf = []

        for nrow in range(len(sig_modified)):
            sig_ = np.abs(sig_modified[nrow])
            mu = np.mean(sig_)
            sigma = np.std(sig_)
            
            s = calculate_skewness(sig_, mu, sigma)
            k = calculate_kurtosis(sig_, mu, sigma)
            c = calculate_CRF(sig_)
            sk.append(s)
            kurtosis.append(k)
            crf.append(c)
        
        # sk_noiseLevel = np.abs(np.median(sk[0:10]))/NoiseLevel_f
        # sk_noiseLevel = np.abs(np.quantile(sk[0:50],quantile_Level)-np.quantile([sk[0:50]],1-quantile_Level))
        adapted_sk, std_sk = adaptiveThreshold(sk, beta1, beta2, thr, buffer_length)
        # crf_noiseLevel = np.abs(np.median(crf[0:10]))/NoiseLevel_f
        # crf_noiseLevel = np.abs(np.quantile(crf[0:50],quantile_Level)-np.quantile([crf[0:50]],1-quantile_Level))
        adapted_crf, std_crf = adaptiveThreshold(crf, beta1, beta2, thr, buffer_length)
        # kurtosis_noiseLevel = np.abs(np.median(kurtosis[0:10]))/NoiseLevel_f
        # kurtosis_noiseLevel =  np.abs(np.quantile(kurtosis[0:50],quantile_Level)-np.quantile([kurtosis[0:50]],1-quantile_Level))

        adapted_kurtosis, std_kurtosis = adaptiveThreshold(kurtosis, beta1, beta2, thr, buffer_length)
        adapted_sk = np.array(adapted_sk)
        std_sk = np.array(std_sk)
        sk = np.array(sk)

        T = np.arange(len(sk)) * deltat
        
        adapted_crf = np.array(adapted_crf)
        std_crf = np.array(std_crf)
        crf = np.array(crf)

        adapted_kurtosis = np.array(adapted_kurtosis)
        std_kurtosis = np.array(std_kurtosis)
        kurtosis = np.array(kurtosis)

        delayT_sk = plotOutBound(T, sk, adapted_sk, std_sk, thr, tauU_outBound=tauU_outBound,
                                 tauD_outBound=tauD_outBound, Label='skewness_{}, {}'.format(column_list[i], figname),
                                 alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                                 saveFolder=saveFolder, figname='skewness_' + figname)
        delayT_crf = plotOutBound(T, crf, adapted_crf, std_crf, thr, tauU_outBound=tauU_outBound,
                                  tauD_outBound=tauD_outBound, Label='CRF_{}, {}'.format(column_list[i], figname),
                                  alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                                  saveFolder=saveFolder, figname='crf_' + figname)
        delayT_kur = plotOutBound(T, kurtosis, adapted_kurtosis, std_kurtosis, thr, tauU_outBound=tauU_outBound,
                                  tauD_outBound=tauD_outBound, Label='kurtosis_{}, {}'.format(column_list[i], figname),
                                  alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                                  saveFolder=saveFolder, figname='kurtosis_' + figname)
        DTC_sk.append(delayT_sk)
        DTC_crf.append(delayT_crf)
        DTC_kurtosis.append(delayT_kur)

    DTC_sk = np.array(DTC_sk)
    DTC_crf = np.array(DTC_crf)
    DTC_kurtosis = np.array(DTC_kurtosis)
    min_delay_sk = np.nanmin(DTC_sk)
    if isinstance(min_delay_sk, np.ndarray):
        min_delay_sk = min_delay_sk[0]
    min_delay_crf = np.nanmin(DTC_crf)
    if isinstance(min_delay_crf, np.ndarray):
        min_delay_crf = min_delay_crf[0]
    min_delay_kurtosis = np.nanmin(DTC_kurtosis)
    if isinstance(min_delay_kurtosis, np.ndarray):
        min_delay_kurtosis = min_delay_kurtosis[0]
  
    return min_delay_sk, min_delay_crf, min_delay_kurtosis


def plotTimeCorrelation(sig, column_list, wsize, fs, thr, rmvf=None, saveFolder='.\\', figname='', savePlot=False,
                        tauU_outBound=0.4, tauD_outBound=0.1, alarm_threshold=0.5, alarm_dur=1, tst=None, plot=False):
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
    deltat = wsize / fs

    delayT = []

    for i in range(nrow):
        dt = sig[:, i]
        
        dt_matrix = dataRearrange(dt, noverlap, wsize)

        corr = [pearson_corr(highPassFilter(dt_matrix[j, :], fs, rmvf), highPassFilter(dt_matrix[j + 1, :], fs, rmvf))
                for j in range(dt_matrix.shape[0] - 1)]
        corr = np.array(corr)
        # corr_noiseLevel = np.abs(np.quantile(corr[0:10],quantile_Level)-np.quantile([corr[0:10]],1-quantile_Level))
        # adapted_corr, std_corr = adaptiveThreshold(corr, beta1,beta2,thr, NoiseLevel = corr_noiseLevel)
        # adapted_corr = np.array(adapted_corr)
        # std_corr =np.array(std_corr)
        # T = np.arange(len(corr))*deltat
        # plotOutBound(T,corr,adapted_corr,std_corr,corr_noiseLevel, thr, Label ='Time Correlation {}'.format(column_list[i]),ax = ax[i])
        T = np.arange(len(corr)) * deltat
        # N = int(0.5/deltat)
        # corr_noiseLevel = np.abs(np.quantile(corr[0:N],quantile_Level)-np.quantile([corr[0:N]],1-quantile_Level))
        # adapted_corr, std_corr = adaptiveThreshold(corr, beta1,beta2,thr, NoiseLevel =corr_noiseLevel)
        # adapted_corr = np.array(adapted_corr)
        adapted_corr = np.ones((len(corr),))
        std_corr = np.ones((len(corr),))
        delay_time = plotOutBound(T, corr, adapted_corr, std_corr, thr, tauU_outBound=tauU_outBound,
                                  tauD_outBound=tauD_outBound,
                                  Label='Time correlation_{}, {}'.format(column_list[i], figname),
                                  alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                                  saveFolder=saveFolder, figname='Timecorr_{}'.format(column_list[i]) + figname,
                                  plot=plot)
        delayT.append(delay_time)

    delayT = np.array(delayT)
    min_delay = np.nanmin(delayT)
    if isinstance(min_delay, np.ndarray):
        min_delay = min_delay[0]
   
    return min_delay


def plotOutBound(T, sig, adapted_sig, std_sig, thr, tauU_outBound=0.4, tauD_outBound=0.1, plot=True, ax=None, Label='',
                 savePlot=False, saveFolder='.\\', figname='', alarm_threshold=0.5, alarm_dur=1, tst=None, mingap=0.2):
    """
    plot the upper and lower boundaries of the given sig and outBound indicator
    Args:
    T: time
    sig: signal, numpy array
    adapted_sig, std_sig: are the output from adaptiveThreshold function
    thr: threshold hold factor
    tauU_outBound: rising time constant for outBound indicator
    tauD_outBound: falling phase time constant for outBound indicator
    plot: boolean. True: plot the signal with upper bound and lowerbound and out of bound indicator
    ax: axis to plot
    Label: label for the sig to plot
    savePlot: boolean. if ax is None and if savePlot is True: save the plots;  False: do not save plots
    saveFolder: Folder path to save plots
    figname: figure name. The final name will be figname + Label
    output:
    detect time 
    """
    threshold = std_sig * thr
    # threshold[threshold < NoiseLevel] = NoiseLevel
    upb_sig = adapted_sig + threshold
    Lwb_sig = adapted_sig - threshold
    outBound_ind = np.abs(adapted_sig - sig) > threshold
    sig_outBound = sig[outBound_ind]
    T_outBound = T[outBound_ind]
    deltat = T[1] - T[0]
    outBound_avg = lowPassFilter(tauU_outBound, tauD_outBound, outBound_ind * 1, T[1] - T[0], outBound_ind[0])
    outBound_avg = np.array(outBound_avg)
    mean_outBound_avg = np.array([np.sum(outBound_avg[:i]) * deltat for i in range(1, len(outBound_avg) + 1)])
    Alarm_indx = (outBound_avg > alarm_threshold) * 1
    label = measure.label(Alarm_indx)
    for lb in np.unique(label):
        if sum(label == lb) * deltat < mingap and np.sum(Alarm_indx[label == lb]) == 0:
            Alarm_indx[label == lb] = 1
    label = measure.label(Alarm_indx)       

        # unique number of labels
    unique_label = np.unique(label)

    Fault_indicator = 0
    alrm = np.zeros(outBound_avg.shape)
    # explore all the regions, if the length of the region is less than minstep, set the sign_sig within that region 0
    for i in unique_label:
        if (sum(label == i) * deltat > alarm_dur) and np.sum(Alarm_indx[label == i]) > 1:
            Fault_indicator = 1
            alrm[label == i] = 1

            break
    
    detection_time = np.argwhere(alrm == 1)
    if len(detection_time) and tst:
  
        delay_time = T[detection_time[0]] - tst
    else:
        delay_time = np.nan

    if plot:
        if ax:
            ax.plot(T, sig, 'k', label=Label)
            ax.plot(T, adapted_sig, 'b', label='adapted')
            ax.plot(T, upb_sig, 'b--', label='Boundary')
            ax.plot(T, Lwb_sig, 'b--')
            ax.plot(T_outBound, sig_outBound, 'g*', label='out of boundary(OB)')
            
            # ax.plot(T,outBound_avg, 'r--', label ='outBound indicator')
        else:
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(T, sig, 'k', label=Label)
            ax[0].plot(T, adapted_sig, 'b')
            ax[0].plot(T, upb_sig, 'b--')
            ax[0].plot(T, Lwb_sig, 'b--')
            ax[0].plot(T_outBound, sig_outBound, 'g*', label='out of boundary(OB)')
            ax[0].legend()
            ax[1].plot(T, outBound_avg, 'k', label='outBound indicator, Fault={}'.format(Fault_indicator))
            ax[1].plot(T, alrm, 'r--')
            if tst:
                ax[1].plot([tst, tst], [0, 1], 'b--')
            ax[2].plot(T, mean_outBound_avg, 'g', label='mean of the outBound indicator')
            if tst:
                ax[2].plot([tst, tst], [0, 1], 'b--')
            
            ax[1].legend()
            plt.show()

            if savePlot:
                figname = figname + '.png'
                filename = os.path.join(saveFolder, figname)

                fig.savefig(filename, format='png', dpi=600)

    return delay_time


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
    deltat = wsize / fs
    fig, ax = plt.subplots(nrow, 1, figsize=(6, 12))
    fig.suptitle('Energy correlation')
    clr = sns.color_palette(None, nrow)

    for i in range(nrow):
        dt = sig[:, i]
        dt_matrix = dataRearrange(dt, noverlap, wsize)
        corr = [calculate_FFtCorr(dt_matrix[j, :], dt_matrix[j + 1, :], fs, fmin, fmax) for j in
                range(dt_matrix.shape[0] - 1)]
        corr = np.array(corr)
        
        T = np.arange(len(corr)) * deltat
        try:
            ax[i].plot(T, corr, color=clr[i], label=column_list[i])
            ax[i].set_ylim([0, 1.1])
            ax[i].legend()
        except:
            ax.plot(T, corr, color=clr[i], label=column_list[i])
            ax.set_ylim([0, 1.1])
            ax.legend()


def plotEnergyCorrelation(sig, column_list, bigwsize, smallwsize, step, fs, fmin, fmax, thr=2, buffer_dur=0.5,
                          beta1=0.99, beta2=0.99, tauU_outBound=0.4, tauD_outBound=0.1):
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
    deltat = bigwsize / fs
    buffer_length = int(buffer_dur / deltat)

    delta_f = fs / smallwsize
    freq_list = np.arange(0, fs / 2, delta_f)
    indx = (freq_list >= fmin) & (freq_list <= fmax)
    harmonicComponents = freq_list[indx]

    for i in range(nrow):
        dt = sig[:, i]
        dt_matrix = dataRearrange(dt, noverlap, bigwsize)
        corr = []
        for j in range(dt_matrix.shape[0] - 1):
            sig1 = dt_matrix[j, :]
            sig2 = dt_matrix[j + 1, :]
            energy1 = calculate_signal_HarmonicEnergy(sig1, smallwsize, step, fs, harmonicComponents, smallwsize, 1)
            energy2 = calculate_signal_HarmonicEnergy(sig2, smallwsize, step, fs, harmonicComponents, smallwsize, 1)
            corr.append(pearson_corr(energy1, energy2))
        corr = np.array(corr)
        # corr_noiseLevel = np.abs(np.quantile(corr[0:10],quantile_Level)-np.quantile([corr[0:10]],1-quantile_Level))
        adapted_corr, std_corr = adaptiveThreshold(corr, beta1, beta2, thr, buffer_length)
        adapted_corr = np.array(adapted_corr)
        std_corr = np.array(std_corr)
        T = np.arange(len(corr)) * deltat
        plotOutBound(T, corr, adapted_corr, std_corr, thr, tauU_outBound=tauU_outBound, tauD_outBound=tauD_outBound,
                     Label='Energy Correlation {}'.format(column_list[i]))


def reconstruction_waveletrec_plot(waveletrec, xmax, **kwargs):
    """Plot reconstruction from wavelet coeff on x [0,xmax] independently of amount of values it contains."""
    # plt.figure()
    # plt.plot(np.linspace(0, 1, len(yyy)), yyy, **kwargs)

    plt.plot(np.linspace(0, 1., num=len(waveletrec)) * xmax, waveletrec, **kwargs)


def reconstruction_waveletcoeff_stem(waveletcoeff, xmax, **kwargs):
    """Plot coefficient vector on x [0,xmax] independently of amount of values it contains."""
    # ymax = waveletcoeff.max()
    plt.stem(np.linspace(0, 1., num=len(waveletcoeff)) * xmax, waveletcoeff, **kwargs)


def wavedec_std(sig, wsize, wstep, fs, wavletname, decompLevel, level=4, plot=False, beta1=0.99, beta2=0.99,
                buffer_dur=0.5, thr=2, tauU_outBound=0.4, tauD_outBound=0.1, figname=[''], alarm_threshold=0.5,
                alarm_dur=1, tst=None, savePlot=False, saveFolder='.\\'):
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
        std_wv = feature_stdDWT(sig[:, i], wsize, wstep, wavletname, level, decompLevel)
    
        # data_M = [sig[j:j+wsize,i] for j in np.arange(0,len(sig),wstep)]

        # w = pywt.Wavelet(wavletname)
        
        # std_wv=[]
        deltat = (wstep) / fs
        buffer_length = int(buffer_dur / deltat)

        # for k in np.arange(len(data_M)):
        #     mx = np.max(np.abs(data_M[k]))
        #     coeffs = pywt.wavedec(data_M[k]/mx,w,level =level)
       
        #     std_wv.append(np.std(coeffs[decompLevel]))

        std_wv = np.array(std_wv)
    
        T_ = np.arange(len(std_wv)) * deltat
        # stdWV_noiseLevel= np.abs(np.quantile(std_wv[0:20],quantile_Level)-np.quantile([std_wv[0:20]],1-quantile_Level))
        adapted_stdWV, std_stdWV = adaptiveThreshold(std_wv, beta1, beta2, thr, buffer_length)
        delay = plotOutBound(T_, std_wv, adapted_stdWV, std_stdWV, thr, tauU_outBound=tauU_outBound, plot=plot,
                             tauD_outBound=tauD_outBound,
                             Label='wavelet decomposition lvl {},{}'.format(decompLevel, figname[i]),
                             alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                             saveFolder=saveFolder, figname=figname[i].replace(',', '_'))
        delayT.append(delay)
        Std_WV.append(std_wv)
       
    delayT = np.nanmin(np.array(delayT))
    try:
        minDelay = delayT[0]
    except:
        minDelay = delayT
    return minDelay


def feature_stdDWT(sig, wsize, wstep, wavletname, level, decompLevel, NoiseLevel=1e-6):
    """calculate standard deviance of the wavelet decomposition components of a sigal within moving window
    Args:
    sig: one-dimension array or array like
    wsize: window size for moving window
    wstep: step to move the window
  
    wavletname:the name of wavelet kernel
    decompLevel: the level to use for calculate standard deviance
    level: the max level for dwt 
    
    Outputs: list contains standard deviance of the wavelet decomposition components
    
    """
    # make sure sig is np.ndarray
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)

    data_M = [sig[j:j + wsize] for j in np.arange(0, len(sig) - wsize, wstep)]

    w = pywt.Wavelet(wavletname)
    
    std_wv = [np.std(pywt.wavedec(dt / np.max([np.max(np.abs(dt)), NoiseLevel]), w, level=level)[decompLevel]) for dt in
              data_M]
   
    return std_wv


def windowed_dataset(series, window_size, nshift, col=-1):
    """Generates dataset windows
    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
  
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=nshift, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:, :-1], window[-1, col:]))
    dataset = dataset.map(lambda win, tg: (tf.cast(win, tf.float32), tf.cast(tg, tf.int32)))
    
    return dataset


def run_ABRatio():
    """automatically run ABratio testing using leaky integrator as decision logic through datasets in cloud storage"""

    # analysis_type = 'Peak ABratio' or 'Standard ABratio'
    analysis_type = 'peak ABRatio'
    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)

    temp_file = os.path.join('.', 'temp_file.h5')
    delayT = []
    Label = np.array([0] * 21 + [1] * 68)

    f0 = 50

    # parameter for adaptive mean and variance
    buffer_dur = 1
    beta1_dur = 2  # mean time constant
    beta2_dur = 5  # variance time constant

    # parameter for integrator
    alarm_threshold = 0.5
    alarm_dur = 0.8
    thr = 4
    tauUp = 0.4
    tauDown = 0.4
    perc = 0.05

    Delaytime = []
    Fault_predict = []
    Alarm_duration = []

    for blob_name in blob_list[1:]:
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5'):
            data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
            fs = calculate_sample_freq(data[TIME_COLUMN])
            file = os.path.basename(blob_name)
            figname = file.split('.')[0]

            # delete the first 2 s of data
            t0 = 2
            t = data.loc[:, 'Time']
            data = data.loc[(t > t0) & (t < 25), :]
            I0 = pd.DataFrame(data.iloc[:, 4:7].mean(axis=1).values, columns=['I0'])
            t = t[(t > t0) & (t < 25)].reset_index().drop("index", axis='columns').values
            t = t - t[0]

            # create new dataset with interested current as columns
            data_new = I0
            print(data_new.columns)

            # set up parameters for DWT 
            Npoints = int(fs / f0)
            big_window_size = Npoints * 2
            shortWsize = int(Npoints // 4)
            nshift = int(Npoints // 4)
            noverlap = big_window_size - nshift

            # extract features
            data_seg = np.array([])

            for i in range(data_new.shape[1]):
                Peak_abRatio, Std_abRatio = feature_ABRatio(data_new.iloc[:, i].values, fs, shortWsize, big_window_size,
                                                            noverlap, nshift, rmvf=60)

                if i == 0:
                    if analysis_type.lower().__contains__('peak'):

                        data_seg = np.array(Peak_abRatio).reshape(-1, 1)
                    else:
                        data_seg = np.array(Std_abRatio).reshape(-1, 1)

                else:
                    if analysis_type.lower().__contains__('peak'):

                        data_seg = np.concatenate((data_seg, np.array(Peak_abRatio).reshape(-1, 1)), axis=1)
                    else:
                        data_seg = np.concatenate((data_seg, np.array(Std_abRatio).reshape(-1, 1)), axis=1)

            # calculate time for feature series
            T = [t[i:i + big_window_size].mean() for i in np.arange(0, data_new.shape[0] - big_window_size, nshift)]
            deltaT = (T[5] - T[0]) / 5
            T = np.array(T)

            beta1 = 1 - deltaT / beta1_dur
            beta2 = 1 - deltaT / beta2_dur
            print(beta1, beta2)
            
            mingap = f0 / fs
            print(mingap)

            # calculate stimulus starting time
            stat = data.loc[:, 'Status'].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = t[indx[0][0]]
                print(tst)
            else:
                tst = None

            # decision logic
            integrator, dtmean_array, dtvar_array = calculate_outBound(data_seg, buffer_dur, perc, thr, deltaT, beta1,
                                                                       beta2, tauUp, tauDown)
            integrator = np.array(integrator)
            Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T,
                                                                                    tst=tst,
                                                                                    alarm_threshold=alarm_threshold,
                                                                                    alarm_dur=alarm_dur)
            Delaytime.append(delay_time)
            Fault_predict.append(Fault_indicator)
            Alarm_duration.append(dur_alarm)

            # plot
            fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
            fig.suptitle(figname + analysis_type + ', Fault = {}'.format(Fault_indicator))

            for i in range(data_seg.shape[1]):
                axes[i, 0].plot(T, data_seg[:, i], color='k')
                axes[i, 0].plot(T, dtmean_array[:, i], color='y')
                axes[i, 0].plot(T, dtmean_array[:, i] + dtvar_array[:, i] * thr, '--', color='g')
                axes[i, 0].plot(T, dtmean_array[:, i] - dtvar_array[:, i] * thr, '--', color='g')
                axes[i, 0].set_xlim([0, 10])

                ind = np.argwhere((np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtvar_array[:, i] * thr) & (
                        np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtmean_array[:, i] * perc))

                axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
                axes[i, 1].plot(T, dtvar_array[:, i], color='b')
                axes[i, 1].plot(T, np.abs(dtmean_array[:, i] - data_seg[:, i]), 'r')
                axes[i, 1].set_xlim([0, 10])
                
            axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
            axes[data_seg.shape[1], 0].set_ylim([0, 2])
                
            axes[data_seg.shape[1], 0].plot([0, T[-1]], [0.5, 0.5], '--', color='orange')
            try: 
                axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
                axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
            except:
                print('No breaker status has been detected')

            plt.tight_layout()
            save_path = os.path.join(".\\plot_abratio", analysis_type.replace(' ', '') + figname + '.png')
            plt.savefig(save_path, format='png', dpi=300)
            plt.show()
            
    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Wavelet Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print("threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, beta1:{}, beta2:{}".format(thr,
                                                                                                             alarm_threshold,
                                                                                                             alarm_dur,
                                                                                                             tauUp,
                                                                                                             tauDown,
                                                                                                             beta1,
                                                                                                             beta2))


def run_LSTM():
    """run LSTM with the entire datasets in cloud"""

    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)

    temp_file = os.path.join('.', 'temp_file.h5')
    undersampling = False
    
    target = np.array([0] * 21 + [1] * 68)
    
    # parameters

    # number of types
    n_type = len(np.unique(target))
    # LSTM time sequence length
    window_size_LSTM = 40
    # number of shifts
    nShift_LSTM = 1

    # batch size
    batch_size = 30

    # first 2 seconds of data is deleted
    t0 = 2
    tend = 8

    # list of the feature arrays for different cases 
    feature_list0 = []
    feature_list1 = []

    i = 0
    for blob_name in blob_list:
        
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if i >= len(target):
            break

        if (blob_name.endswith('.h5')) and (not blob_name.__contains__('5p288')):

            tg = target[i]
            data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
            os.remove(temp_file)
            T = data.iloc[:, 0]
            data = data.loc[(T > t0) & (T < tend), :]
            T = data.iloc[:, 0].values - data.iloc[0, 0]
        
            fs = 5 / (T[5] - T[0])
            f0 = 50
            stat = data.iloc[:, -1].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = T[indx[0]]
                # print(tst)
            else:
                tst = None
            Npoints = int(fs / f0)
            window_size = Npoints * 2
            # fstep = fs/window_size/f0
            freq_list = np.arange(1, 15, 1) * f0
            nShift = Npoints // 4
            print(blob_name, tg)

            if np.sum(stat > 0):

                d = data.loc[stat > 0, :].copy()
            else:
                d = data.copy()
                
            d.loc[:, 'Status'] = tg
        
            d = d.iloc[:, 1:8]
            d = d.to_numpy()
            # convert dataframe into tensorflow Dataset    
            dataset = windowed_dataset(d, window_size, nShift)

            feature_list_perfile0 = []
            feature_list_perfile1 = []
            for win, targ in dataset:
                fft_array = np.array([])
                
                for j in range(win.shape[1]):
                    df = win[:, j].numpy()
                    _, fftH = myfft(df, fs, freq_list)
                    
                    fft_array = np.append(fft_array, np.log10(np.abs(fftH[1:]) / np.abs(fftH[0])))
                fft_array = np.append(fft_array, targ)
                if targ == 0:
                    feature_list_perfile0.append(fft_array) 
                else:
                    feature_list_perfile1.append(fft_array) 
            
            # print(len(feature_list))
            if targ == 0:
                feature_list0.append(feature_list_perfile0)
            else:
                feature_list1.append(feature_list_perfile1)
            i += 1
            
    feature_list_new0 = []
    if n_type > 2:
        for FL in feature_list0:
            FL = np.array(FL)
            tg = FL[:, -1]
            tg = to_categorical(tg, n_type + 1)
            FL_ = FL[:, :-1]
            FL = np.concatenate((FL_, tg), axis=1)
            feature_list_new0.append(FL)
    else:
        feature_list_new0 = feature_list0

    feature_list_new1 = []
    if n_type > 2:
        for FL in feature_list1:
            FL = np.array(FL)
            tg = FL[:, -1]
            tg = to_categorical(tg, n_type + 1)
            FL_ = FL[:, :-1]
            FL = np.concatenate((FL_, tg), axis=1)
            feature_list_new1.append(FL)
    else:
        feature_list_new1 = feature_list1
        
    # creat tensorflow datasets
    
    if n_type > 2:
        col = tg.shape[1] * -1
    else:
        col = -1
        tg = tg.reshape(1, -1)

    k = 0
    for FL in feature_list_new0:
        FL = np.array(FL)
        
        ds = windowed_dataset(FL, window_size_LSTM, nShift_LSTM, col)
        # ds.map(lambda win, tg: (tf.cast(win,tf.float32),tf.cast(tg,tf.int32)))
        if k == 0:

            dataset0 = ds
        else:
            dataset0 = dataset0.concatenate(ds)
        k += 1
    
    k = 0
    for FL in feature_list_new1:
        FL = np.array(FL)
        
        ds = windowed_dataset(FL, window_size_LSTM, nShift_LSTM, col)
        print(len(list(ds)))
        if k == 0:

            dataset1 = ds
        else:
            dataset1 = dataset1.concatenate(ds)
        k += 1
 
    # Shuffle the windows
    # shuffle buffer
    shuffle_buffer0 = len(list(dataset0))
    dataset0 = dataset0.shuffle(shuffle_buffer0)
    shuffle_buffer1 = len(list(dataset1))
    dataset1 = dataset1.shuffle(shuffle_buffer1)

    # undersampling
    if undersampling:
        takesize = min(shuffle_buffer0, shuffle_buffer1)
        dataset0_sub = dataset0.take(takesize)
        dataset1_sub = dataset1.take(takesize)
        dataset = dataset0_sub.concatenate(dataset1_sub)
        dataset = dataset.shuffle(takesize * 2)
        dataset_unchanged = dataset0.concatenate(dataset1)
        dataset_size = takesize * 2
    else:
        dataset_size = shuffle_buffer0 + shuffle_buffer1
        dataset = dataset0.concatenate(dataset1)
        dataset = dataset.shuffle(dataset_size)
        dataset_unchanged = dataset

    Label = []
    for win, targ in dataset:
        Label.append(targ.numpy())
    Label = np.array(Label).flatten()
    print(np.sum(Label) / len(Label))
    unique_classes = np.unique(Label)
    # compute weights
    weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=Label)
    weights_ = {c: weights[i] for i, c in enumerate(unique_classes)}
    # train, test, split
    # dataset_size = takesize*2
    train_size = int(0.6 * dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    TG = []
    fftImg = []
    for win, targ in train_dataset:
        fftImg.append(win.numpy())
        TG.append(targ.numpy())
    TG = np.array(TG).flatten()
    print('fault rate in training dataset: {}'.format(np.sum(TG) / len(TG)))
    fftImg = np.array(fftImg)
    scaler = StandardScaler() 
    fftImg_transform = scaler.fit_transform(fftImg.reshape(fftImg.shape[0], -1))
    fftImg_transform = fftImg_transform.reshape(fftImg.shape[0], fftImg.shape[1], fftImg.shape[2])

    fftImg_test = []
    TG_test = []
    for win, targ in test_dataset:
        win_transform = scaler.transform(win.numpy().reshape(1, -1))
        win_transform = win_transform.reshape(fftImg.shape[1], fftImg.shape[2])
        fftImg_test.append(win_transform)
        TG_test.append(targ)
    fftImg_test = np.array(fftImg_test)
    TG_test = np.array(TG_test).flatten()
    print(fftImg_test.shape, TG_test.shape)

    # train_dataset = train_dataset.map(lambda win, targ: (tf.data.Dataset.from_tensor_slices(scaler.transform(win.numpy().reshape(1,-1)).reshape(fftImg.shape[1], fftImg.shape[2])),targ))
    # test_dataset = test_dataset.map(lambda win, targ: (tf.data.Dataset.from_tensor_slices(scaler.transform(win.numpy())),targ))

    # Create batches of windows

    # train_dataset = train_dataset.batch(batch_size,drop_remainder = True).prefetch(1)
    # test_dataset = test_dataset.batch(batch_size,drop_remainder = True).prefetch(1)
        
    feature_size = win.shape[-1]

    # LSTM model
    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(8, input_shape=[window_size_LSTM, feature_size], return_sequences=True)),
                                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),

        tf.keras.layers.Dense(tg.shape[1], activation=tf.keras.activations.sigmoid)
                                    ])

    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 1e-8* 10**(epoch / 16))
    stoplearn = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)

    if n_type == 2:
        loss = tf.keras.losses.BinaryCrossentropy()
        metric = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall()]
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
        metric = [tf.keras.metrics.CategoricalAccuracy()]
        
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)

    history = model.fit(fftImg_transform, TG, batch_size=batch_size, epochs=100,
                        callbacks=[stoplearn, PlotLossesKeras()], validation_data=(fftImg_test, TG_test),
                        class_weight=weights_)
    filename = "completed_model.joblib"
    joblib.dump(model, filename)

    yreal = []
    fftImg_total = []
    for win, targ in dataset_unchanged:
        win_array = win.numpy()
        win_transform = scaler.transform(win_array.reshape(1, -1))
        win_transform = win_transform.reshape(win_array.shape[0], win_array.shape[1])
        fftImg_total.append(win_transform) 
        
        yreal.append(targ)
    fftImg_total = np.array(fftImg_total)
    ypred = model.predict(fftImg_total)
    # ypred = np.array(ypred)
    yreal = np.array(yreal)
    ypred_f = ypred.squeeze().flatten()
    yreal_f = yreal.squeeze().flatten()
    plt.scatter(yreal_f, ypred_f)

    fraction_of_positives, mean_predicted_value = calibration_curve(yreal_f, ypred_f, n_bins=50)
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated", lw=4)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='LSTM', lw=4, color='b')
    ax1.set_xlabel('mean_predicted_value', size=20)
    ax1.legend(bbox_to_anchor=(0.5, 1.0), borderpad=2, fontsize=20)
    ax1.set_ylabel('fraction of positive', size=20)
    ax2.hist(ypred_f, range=(0, 1), bins=50, histtype="step", lw=4)
    ax2.set_xlabel('predicted_value', size=20)
    plt.tight_layout()


def computeMI(x, y):
    """calculate mutrual information of x and y
    """
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([len(x[x == xval]) / float(len(x)) for xval in x_value_list])  # P(x)
    Py = np.array([len(y[y == yval]) / float(len(y)) for yval in y_value_list])  # P(y)
    for i in range(len(x_value_list)):
        if Px[i] == 0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy) == 0:
            continue
        pxy = np.array([len(sy[sy == yval]) / float(len(y)) for yval in y_value_list])  # p(x,y)
        t = pxy[Py > 0.] / Py[Py > 0.] / Px[i]  # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t > 0] * np.log2(t[t > 0]))  # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi


def feature_mulInform(vol, curr, wsize, noverlap, rmvf, bins):
    curr_win = dataRearrange(curr, noverlap, wsize)
    vol_win = dataRearrange(vol, noverlap, wsize)
    mul_info = []

    for j in range(curr_win.shape[0]):
        x = curr_win[j, :]
        x = highPassFilter(x, fs, rmvf)
        x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        y = vol_win[j, :]
        y = highPassFilter(y, fs, rmvf)
        y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
        inds_x = np.digitize(x, bins)
        x = np.array([(bins[inds_x[n] - 1] + bins[inds_x[n]]) / 2 for n in range(x.size)])
        inds_y = np.digitize(y, bins)
        y = np.array([(bins[inds_y[n] - 1] + bins[inds_y[n]]) / 2 for n in range(y.size)])
        mulI = computeMI(x, y)
        mul_info.append(mulI)
    return mul_info


def plot_mulInformation(V, I, T, fs, column_list, nbins, rmvf, thr, filename, tst, wsize, noverlap, tauU_outBound,
                        tauD_outBound, beta1_dur=0.2, beta2_dur=2, buffer_dur=0.5, alarm_threshold=0.5, alarm_dur=0.4,
                        savePlot=False, plot=True):
    # root_dir = ["C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\ph2e100kHz\\hd5", "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\VT526_WithPV_Scaled\\HDFfolder","C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\InductionMotorSoftStart3MVA\\HDFfolder","C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\RegulatorTest\\HDFfolder","C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Capacitor_Switching_10kHz", "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Transformer_Inrush10kHz\\hd5"]
    # nbins = 50
    """plot multrual information with time
    Args:
    V,I: Voltage and current data array that contains voltage and currents: column_list: [Time, 3 phase voltages, 3 phase currents]
    T: time series
    fs: sample rate
    nbins: number of bins for calculate multrual information, digitalize voltage and currents into bins
    rmvf: remove frequency
    filename: file name
    tst: time of starting simulation
    wsize: window size for multrual information
    noverlap: number of overlapping
    tauU_outBound, tauD_outBound: time constant for the decision logic
    alarm_threshold: the threshold for the decision logic 
    alarm_dur: duration for alarm continuously being out of boundary to report fault
    savePlot: boolean. True: save plot; False: not save
    plot: boolean. True: plot time correlation, False: not plot 
    """
    bins = np.linspace(0, 1.01, nbins)
    if not isinstance(V, np.ndarray):
        V = np.array(V).reshape(len(V), -1)
    if not isinstance(I, np.ndarray):
        I = np.array(I).reshape(len(I), -1)

    T_win = dataRearrange(T, noverlap, wsize)
    T_avg = np.mean(T_win, axis=1)

    deltaT = T_avg[1] - T_avg[0]
    buffer_length = int(buffer_dur / deltaT)

    beta1 = 1 - deltaT / beta1_dur
    beta2 = 1 - deltaT / beta2_dur

    # fig,axes = plt.subplots(I.shape[1],2)
    # clr = sns.color_palette(None, I.shape[1])
    # figname = filename +'nbins{}rmvf{}'.format(nbins,rmvf)
    if rmvf > 0:
        saveFolder = './/plot_mutrualInformation'
    else:
        saveFolder = './/plot_mutrualInformation_withfundamental'

    delay_list = []
    for i in range(I.shape[1]):
        curr = I[:, i]
        vol = V[:, i]
        mul_info = feature_mulInform(vol, curr, wsize, noverlap, rmvf, bins)
        mul_info = np.array(mul_info)
        # mul_noiseLevel= np.abs(np.quantile(mul_info[0:Nbuffer],quantile_Level)-np.quantile([mul_info[0:Nbuffer]],1-quantile_Level))
        adapted_mul, std_mul = adaptiveThreshold(mul_info, beta1, beta2, thr, buffer_length)
        delay = plotOutBound(T_avg, mul_info, adapted_mul, std_mul, thr, tauU_outBound=tauU_outBound,
                             tauD_outBound=tauD_outBound,
                             Label='MultualInfo_{},nbins={},rmvf={},{}'.format(filename, nbins, rmvf, column_list[i]),
                             alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot,
                             plot=plot, saveFolder=saveFolder,
                             figname=filename + "nbins{}".format(nbins) + "rmvf{}".format(rmvf) + '{}'.format(
                                 column_list[i]))
        delay_list.append(delay)
    return delay_list
 


def run_timecorr(blob_list, connect_str, container_name, t0, tend, Label, thr, tauU_outBound, tauD_outBound,
                 alarm_threshold, alarm_dur, rmvf=3, savePlot=False, plot=False):

    """run time correlation through the datasets in cloud
    Args:
    blob_list: the list of the files in the blob storage
    connect_str: connect string for the azure blob storage account
    container_name: container name in the blob storage account
    t0: start time
    tend: end time
    Label: a list of Labels for the files in the blob_list. 0: normal 1: fault
    thr: threshold for baseline calculation
    tauU_outBound, tauD_outBound: time constant for the decision logic
    alarm_threshold: the threshold for the decision logic 
    alarm_dur: duration for alarm continuously being out of boundary to report fault
    savePlot: boolean. True: save plot; False: not save
    plot: boolean. True: plot time correlation, False: not plot 
    Output:tuple
    thr, alarm_threshold, alarm_dur,tauU_outBound: as above
    Accur: Accuracy
    F1: F1 score
    False_alarm:false alarm
    mean_respond : mean responding time
    
    """
  
    temp_file = os.path.join('.', 'temp_file.h5')
    delayT = []

    i = 0
    for blob_name in blob_list:
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5') and (not blob_name.__contains__('5p288')):
            data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
            os.remove(temp_file)


            T = data.iloc[:, 0]
            data = data.loc[(T > t0) & (T < tend), :]
            fs = 5 / (T[5] - T[0])

            T = data.iloc[:, 0].values - data.iloc[0, 0]

            f0 = 50
            stat = data.iloc[:, -1].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = T[indx[0]]
                
            else:
                tst = None
            
            sig = data.iloc[:, 4:7]
            
            wsize = int(fs / f0 * 2)
            # wstep = int(fs/f0)
            
            file = os.path.basename(blob_name)
            filename = file.split('.')[0]
            print(filename, Label[i])
            # deltaT =wstep/fs
            # beta1 = 1-deltaT/0.2
            # beta2 = 1-deltaT/2
            # print('beta1={}, beta2={}'.format(beta1,beta2))
            column_list = sig.columns
            # thr = 0.4

            cutoff = rmvf * f0
            delay_time = plotTimeCorrelation(sig, column_list, wsize, fs, thr, rmvf=cutoff,
                                             saveFolder='.\\plot_timecorr', figname=filename, savePlot=savePlot,
                                             tauU_outBound=tauU_outBound, tauD_outBound=tauD_outBound,
                                             alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, plot=plot)

            
            delayT.append(delay_time)
            i += 1

    delayT = np.array(delayT)
    Label = Label[np.arange(len(delayT))]

    delayT = [k[0] if isinstance(k, np.ndarray) else k for k in delayT]
    y = np.array(delayT)
    positive_indx = (y < 1) * 1
    Accur, False_alarm, F1 = calculate_evaluation(positive_indx, Label)


    print("threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauU:{}, rmvf".format(thr, alarm_threshold, alarm_dur,
                                                                                  tauU_outBound, rmvf))
    print('Time correlation Accuracy: {}, F1: {}, False_alarm: {}'.format(Accur, F1, False_alarm))
    mean_respond = np.nanmean(y[y < 1]) + alarm_dur

    print('average detecttime is {}'.format(mean_respond))
    return (thr, alarm_threshold, alarm_dur, tauU_outBound, Accur, F1, False_alarm, mean_respond)


def feature_timecorr(sig, wsize, noverlap, rmvf, fs, NoiseLevel, perc=0.9):
    """ output the time correlation list for the given signal
    Args:
    sig: one deminsion numpy array or array like
    wsize: window size
    noverlap: number of overlaping
    rmvf: removed frequency
    Outputs: list of time correlations between neighboring window
    
    """
    dt_matrix = dataRearrange(sig, noverlap, wsize)
    corr = [pearson_corr(highPassFilter(dt_matrix[j, :], fs, rmvf), highPassFilter(dt_matrix[j + 1, :], fs, rmvf)) if (
                np.quantile(dt_matrix[j, :], perc) > NoiseLevel and np.quantile(dt_matrix[j + 1, :],
                                                                                perc) > NoiseLevel) else 1 for j in
            range(dt_matrix.shape[0] - 1)]

    return corr

def feature_energycorr(sig, bigwsize, nshift_big, smallwsize,nshift_small, harmonicComponents, fs,f0,NoiseLevel, paddingM=None, P=1, perc=0.9):
    """ output the time correlation list for the given signal
    Args:
    sig: one deminsion numpy array or array like
    bigwsize: big window size for consecutive windows
    nshift_big: number of shift for big window
    smallwsize: small window size for calculating fft components
    nshift_small: the number of shift for small window to achieve the engery time sequency
    fs: sample frequency
    f0: fundamental 
    NoiseLevel:noise level
    paddingM: zero padding for calculate fft components. if paddingM is None, then paddingM is equal to the smallwsize 
    harmonicCompoents: energy components to be considered
    


    Outputs: list of time correlations between neighboring window
    
    """
    assert bigwsize >= smallwsize + nshift_small, "bigwsize must be larger than smallwsize + nshift_small"
    assert nshift_small <= smallwsize, "nshift_small should be no larger than smallwsize "
    dt_matrix = [[sig[j:j+bigwsize], sig[j+bigwsize:j+bigwsize*2]] for j in range(0, len(sig)-bigwsize*2+1,nshift_big)]
    noverlap = smallwsize - nshift_small

    corr = [pearson_corr(calculate_signal_HarmonicEnergy(dt_matrix[j][0], smallwsize,noverlap,fs, harmonicComponents,paddingM, P),calculate_signal_HarmonicEnergy(dt_matrix[j][1], smallwsize,noverlap,fs, harmonicComponents,paddingM, P)) if (np.quantile(dt_matrix[j][0], perc) > NoiseLevel and np.quantile(dt_matrix[j][1], perc) > NoiseLevel) else 1 for j in range(len(dt_matrix))]

    return corr

def feature_energycorr(sig, bigwsize, nshift_big, smallwsize, nshift_small, harmonicComponents, fs, f0, NoiseLevel,
                       paddingM=None, P=1, perc=0.9):
    """ output the time correlation list for the given signal
    Args:
    sig: one deminsion numpy array or array like
    bigwsize: big window size for consecutive windows
    nshift_big: number of shift for big window
    smallwsize: small window size for calculating fft components
    nshift_small: the number of shift for small window to achieve the engery time sequency
    fs: sample frequency
    f0: fundamental
    NoiseLevel:noise level
    paddingM: zero padding for calculate fft components. if paddingM is None, then paddingM is equal to the smallwsize
    harmonicCompoents: energy components to be considered

    Outputs: list of time correlations between neighboring window

    """
    assert bigwsize >= smallwsize + nshift_small, "bigwsize must be larger than smallwsize + nshift_small"
    assert nshift_small <= smallwsize, "nshift_small should be no larger than smallwsize "
    dt_matrix = [[sig[j:j + bigwsize], sig[j + bigwsize:j + bigwsize * 2]] for j in
                 range(0, len(sig) - bigwsize * 2 + 1, nshift_big)]
    noverlap = smallwsize - nshift_small

    corr = [pearson_corr(
        calculate_signal_HarmonicEnergy(dt_matrix[j][0], smallwsize, noverlap, fs, harmonicComponents, paddingM, P),
        calculate_signal_HarmonicEnergy(dt_matrix[j][1], smallwsize, noverlap, fs, harmonicComponents, paddingM,
                                        P)) if (
                np.quantile(dt_matrix[j][0], perc) > NoiseLevel and np.quantile(dt_matrix[j][1],
                                                                                perc) > NoiseLevel) else 1 for j in
            range(len(dt_matrix))]

    return corr


def calculate_evaluation(prediction, Label):
    """calculate accuration, false_alarm, F1 score with predicted value and labeled data
    prediction: a numpy array with predictions, 0/ False: normal; 1/True:fault  
    Label: a numpy array with 0s and 1s : 0 : normal 1: fault
    """
    prediction = prediction * 1
    TP = np.sum(Label[prediction > 0])
    FN = np.sum(1 - prediction[Label == 1])
    FP = np.sum(prediction[Label == 0])
    TN = np.sum(1 - prediction[Label == 0])
    F1 = 2 * TP / (2 * TP + FP + FN)
    Accur = (TP + TN) / len(prediction)
    False_alarm = FP / len(prediction)
    return Accur, False_alarm, F1


def save_object(obj, filename):
    """save data into a pickle file
    Args:
    obj: object to be saved
    filename: the name of the file that the object is going to be saved, need to ends with .pickle. For example, data.pickle
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    """load pickle file, return data """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def run_multualInfo_local():
    """run multualInformation through the entire local datasets"""

    chanel_list = ['Test_V', 'Test_I', 'ground_I', 'Voltage', 'Test/Fault Current', 'Leakage Current']
    # root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\\Franksville Test\\20220728\\7-28-22 csv backup"
    root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\\Franksville Test\\20220727\\7-27-22 csv backup"
     
    dir_list = os.listdir(root)
    f0 = 60
    buffer_dur = 0.5

    for i in range(len(dir_list)):
        file_path = os.path.join(root, dir_list[i])
        data,_,_ = data_acquisition(file_path, voltage_channels=['Test_V', 'Voltage'],
                                    current_channels=['Test_I', 'ground_I', 'Test/Fault Current', 'Leakage Current'],
                                    kwds_brkr=['solenoid'])
        filename = dir_list[i].split('.')[0]

        V = data.loc[:, chanel_list[0]]
        I = data.loc[:, chanel_list[1]]

        nbins = 100
        rmvf = 2 * f0

        T = data.iloc[:, 0].to_numpy()
        fs = 5 / (T[5] - T[0])
        f0 = 60
        column_list = [chanel_list[0]]
        wsize = round(fs / f0 * 4)
        noverlap = wsize // 2

        thr = 2
        stat = data.iloc[:, -1].to_numpy()
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = T[indx[0]]
            print(tst)
        else:
            tst = None

        tauU_outBound = 0.2
        tauD_outBound = 0.2
        beta1_dur = 0.2
        beta2_dur = 3
        alarm_dur = 0.4
        alarm_threshold = 0.5

        plot_mulInformation(V, I, T, fs, column_list, nbins, rmvf, thr, filename, tst, wsize, noverlap, tauU_outBound,
                            tauD_outBound, beta1_dur=beta1_dur, beta2_dur=beta2_dur, buffer_dur=buffer_du,
                            alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, savePlot=True)


def plot_amplitude(data, T, chl, maxAmp, wsize, filename, ymax=None, step=1):
    """plot current or voltage max amplitude within a moving window
    Args:
    data: numpy array or array-like with current or voltage or breaker info
    T: time array or array-like
    chl: string, chanel name
    maxAmp: the max amplitude to reach
    wsize: moving window size
    filename: string, file name for the ploted dataset
    ymax: upper limit for the y axis
    step: step for moving window
    Output:
    tuple, (the first time current reaching to maxAmp, max of amplitude of the entire trace)
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not isinstance(T, np.ndarray):
        T = np.array(T)

    if not ymax:
        ymax = np.max(data)
    amplitude = [np.max(np.abs(data[i:i + wsize])) for i in range(0, len(data) - wsize, step)]
    amplitude = np.array(amplitude)
    st = round(wsize / 2)
    ed = len(data) - round(wsize / 2)
    T_step = T[st:ed:step] - T[0]

    plt.plot(T_step, amplitude, label=filename)

    T_ = T_step[amplitude > maxAmp]
    if len(T_):
        plt.plot([T_[0], T_[0]], [0, maxAmp], 'b--')
        plt.xlabel('time (s)')
        plt.ylabel('Max Current Amp, ' + chl)
        plt.ylim([0, ymax])
        t_stop = T_[0]
    else:
        t_stop = None
    
    return (t_stop, np.max(amplitude))


def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Args:
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Outputs:
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n)
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy
    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002 [1]_. details: https://www.aptech.com/blog/permutation-entropy/
    
    References
    ----------
    .. [1] Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
           natural complexity measure for time series." Physical review letters
           88.17 (2002): 174102.
    Examples
    --------
    1. Permutation entropy with order 2
        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(perm_entropy(x, order=2))
            0.918
    2. Normalized permutation entropy with order 3
        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(perm_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(math.factorial(order))
    return pe


def decisionLogic(dt, dt_mean, dt_var, dt_mean_corrected, dt_var_corrected, thr, step, buffer, beta1, beta2, perc,
                  freeze):
    """ decision logic with leaky integrator
    Args:
    dt: current data
    thr: threshold factor for boundary
    buffer: buffer, numpy array
    dt_mean : the mean value of dt
    dt_var: the standard deviance of dt
    dt_mean_corrected: the corrected mean from exponentially moving average
    dt_var_corrected: the corrected std from exponentially moving average
    beta1: time constant for calculating adapative mean
    beta2: time constant for calculating adapative mean
    step: the current step
    perc: it is used for boundary condition.  the percentage changes for identifying out of boundary
    freeze: boolen, 0 : not frozen; 1: frozen. It is used for freezing the update of variance
    """
    n = 0
    dt_mean, dt_mean_corrected = adaptiveMean(dt, beta1, step + 1, dt_mean, biasCorrect=True)

    if (abs(dt - dt_mean_corrected) < dt_var_corrected * thr or step < len(buffer)) and freeze == 0:
        buffer = np.append(buffer, dt)
        buffer = buffer[1:]
        v = np.nanstd(buffer)
        dt_var, dt_var_corrected = adaptiveMean(v, beta2, step + 1, dt_var, biasCorrect=True)

    if (abs(dt - dt_mean_corrected) > dt_var_corrected * thr and abs(dt - dt_mean_corrected) > abs(
            dt_mean_corrected) * perc and step > len(buffer)):
        n = 1
    return n, buffer, dt_mean, dt_var, dt_mean_corrected, dt_var_corrected


def calculate_outBound(data_seg, buffer_dur, perc, thr, deltaT, beta1, beta2, tauUp, tauDown):
    """calculate outBound indicator using leaky integrator
    Args:
    data_seg: list of data segmentations
    buffer_dur (sec): duration of buffer
    perc:  it is used for boundary condition.  the percentage changes for identifying out of boundary
    thr: threshold factor for boundary
    beta1: time constant for calculating adapative mean
    beta2: time constant for calculating adapative mean
    """
    intg = 0

    integrator = []

    dtmean_list = []
    dtvar_list = [] 

    buffer_length = int(buffer_dur / deltaT)
    d = data_seg[0].reshape(1, -1)
    buffer = np.ones((buffer_length, d.shape[1])) * np.nan
    dt_mean = np.array([0.0] * (d.shape[1]))
    dt_var = np.array([0.0] * (d.shape[1]))
    dt_mean_corrected = np.array([0.0] * (d.shape[1]))
    dt_var_corrected = np.array([0.0] * (d.shape[1]))
    freeze = 0

    for k, dt in enumerate(data_seg):

        n = 0
       
        for i, dt_i in enumerate(dt):
            n_, buffer[:, i], dt_mean[i], dt_var[i], dt_mean_corrected[i], dt_var_corrected[i] = decisionLogic(dt_i,
                                                                                                               dt_mean[
                                                                                                                   i],
                                                                                                               dt_var[
                                                                                                                   i],
                                                                                                               dt_mean_corrected[
                                                                                                                   i],
                                                                                                               dt_var_corrected[
                                                                                                                   i],
                                                                                                               thr, k,
                                                                                                               buffer[:,
                                                                                                               i],
                                                                                                               beta1,
                                                                                                               beta2,
                                                                                                               perc,
                                                                                                               freeze)
            n += n_
        
        intg = lowPassFilter_1(tauUp, tauDown, n, deltaT, intg)
            
        if (intg > 0.5 and k > buffer_length):

            freeze = 1
        else:
            freeze = 0

        integrator.append(intg)
        dtmean_list.append(dt_mean_corrected.copy())
        dtvar_list.append(dt_var_corrected.copy())

    dtmean_array = np.array(dtmean_list)
    dtvar_array = np.array(dtvar_list)

    return integrator, dtmean_array, dtvar_array


def calculate_outBound_v2(data_seg, buffer_dur, perc, thr, deltaT, beta1, beta2, tauUp, tauDown, freeze_timelimit=5):
    """calculate outBound indicator using leaky integrator
    Args:
    data_seg: list of data segmentations
    buffer_dur (sec): duration of buffer
    perc:  it is used for boundary condition.  the percentage changes for identifying out of boundary
    thr: threshold factor for boundary
    beta1: time constant for calculating adapative mean
    beta2: time constant for calculating adapative mean
    """

    intg = 0

    integrator = []


    dtmean_list = []
    dtvar_list = []
    ob = []

    buffer_length = int(buffer_dur / deltaT)
    d = data_seg[0].reshape(1, -1)
    buffer = np.ones((buffer_length, d.shape[1])) * np.nan
    dt_mean = np.array([0.0] * (d.shape[1]))
    dt_var = np.array([0.0] * (d.shape[1]))
    dt_mean_corrected = np.array([0.0] * (d.shape[1]))
    dt_var_corrected = np.array([0.0] * (d.shape[1]))
    freeze = 0
    Timer = freeze_timelimit
    for k, dt in enumerate(data_seg):

        n = 0
        if freeze == 1:
            Timer -= deltaT

        for i, dt_i in enumerate(dt):
            n_, buffer[:, i], dt_mean[i], dt_var[i], dt_mean_corrected[i], dt_var_corrected[i] = decisionLogic(dt_i,
                                                                                                               dt_mean[
                                                                                                                   i],
                                                                                                               dt_var[
                                                                                                                   i],
                                                                                                               dt_mean_corrected[
                                                                                                                   i],
                                                                                                               dt_var_corrected[
                                                                                                                   i],
                                                                                                               thr, k,
                                                                                                               buffer[:,
                                                                                                               i],
                                                                                                               beta1,
                                                                                                               beta2,
                                                                                                               perc,
                                                                                                               freeze)
            n += n_

        intg = lowPassFilter_1(tauUp, tauDown, n, deltaT, intg)

        ob.append(n)

        if (intg > 0.5 and k > buffer_length):

            freeze = 1

        else:
            if Timer <= 0:
                freeze = 0
                Timer = freeze_timelimit


        integrator.append(intg)  
        dtmean_list.append(dt_mean_corrected.copy())
        dtvar_list.append(dt_var_corrected.copy())

    dtmean_array = np.array(dtmean_list)
    dtvar_array = np.array(dtvar_list)

    return integrator, dtmean_array, dtvar_array, ob



def calculate_outBound_v2(data_seg, buffer_dur, perc,thr, deltaT, beta1,beta2, tauUp, tauDown, freeze_timelimit =5):
    """calculate outBound indicator using leaky integrator
    Args:
    data_seg: list of data segmentations
    buffer_dur (sec): duration of buffer
    perc:  it is used for boundary condition.  the percentage changes for identifying out of boundary
    thr: threshold factor for boundary
    beta1: time constant for calculating adapative mean
    beta2: time constant for calculating adapative mean
    """

    intg = 0

    integrator=[]

    dtmean_list=[]
    dtvar_list = [] 
    ob=[]

    buffer_length = int(buffer_dur/deltaT)
    d = data_seg[0].reshape(1,-1)
    buffer =np.ones((buffer_length,d.shape[1]))*np.nan
    dt_mean = np.array([0.0]*(d.shape[1]))
    dt_var = np.array([0.0]*(d.shape[1]))
    dt_mean_corrected = np.array([0.0]*(d.shape[1]))
    dt_var_corrected = np.array([0.0]*(d.shape[1]))
    freeze = 0
    Timer = freeze_timelimit
    for k, dt in enumerate(data_seg):

        n=0
        if freeze ==1:
            Timer -=deltaT
        
        for i,dt_i in enumerate(dt):
            

            n_, buffer[:,i], dt_mean[i], dt_var[i], dt_mean_corrected[i], dt_var_corrected[i] = decisionLogic(dt_i,dt_mean[i], dt_var[i], dt_mean_corrected[i], dt_var_corrected[i], thr, k, buffer[:,i], beta1,beta2,perc, freeze)
            n+=n_

        intg = lowPassFilter_1(tauUp,tauDown, n, deltaT, intg)

        ob.append(n)
 
        if (intg > 0.5 and  k>buffer_length) :

            freeze = 1
            
        else:
            if Timer <= 0:
                freeze =0
                Timer = freeze_timelimit

        integrator.append(intg)  
        dtmean_list.append(dt_mean_corrected.copy())
        dtvar_list.append(dt_var_corrected.copy())


    dtmean_array = np.array(dtmean_list)
    dtvar_array = np. array(dtvar_list)

    return integrator, dtmean_array,dtvar_array,ob


def calculate_outBound_timecorr(data_seg, thr, deltaT, tauUp, tauDown):
    """calculate outBound indicator using leaky integrator
    Args:
    data_seg: list of data segmentations
    buffer_dur (sec): duration of buffer
    perc:  it is used for boundary condition.  the percentage changes for identifying out of boundary
    thr: threshold factor for boundary
    beta1: time constant for calculating adapative mean
    beta2: time constant for calculating adapative mean
    """
    if isinstance(data_seg, np.ndarray):
        data_seg = np.array(data_seg)
    
    data_seg = data_seg.reshape(len(data_seg), -1)

    n = np.array([0.0] * data_seg.shape[0])
    for i in range(data_seg.shape[1]):
        dt = data_seg[:, i]

        n_ = dt < thr
        n += n_
        
    intg = lowPassFilter(tauUp, tauDown, n, deltaT, n[0])

    return intg


def calculate_faultIndicator(integrator, deltat, mingap, T, tst=None, alarm_threshold=0.5, alarm_dur=1):
    Alarm_indx = (integrator > alarm_threshold) * 1
    label = measure.label(Alarm_indx)
    for lb in np.unique(label):
        if sum(label == lb) * deltat < mingap and np.sum(Alarm_indx[label == lb]) == 0:
            Alarm_indx[label == lb] = 1
    label = measure.label(Alarm_indx)       

        # unique number of labels
    unique_label = np.unique(label)

    dur_alarm = 0
    Fault_indicator = 0
    alrm = np.zeros(integrator.shape)
    # explore all the regions, if the length of the region is less than minstep, set the sign_sig within that region 0
    for i in unique_label:
        
        if (sum(label == i) * deltat > alarm_dur) and np.sum(Alarm_indx[label == i]) > 1:
            
            alrm[label == i] = 1

            break

    # for i in unique_label:
    #     if np.sum(Alarm_indx[label == i]) > 1:
    #         dur_alarm = sum(label == i) * deltat
    #         break
    dur_alarm = sum(alrm ==1)*deltat

    detection_time = np.argwhere(alrm == 1)
    if len(detection_time) and tst:
  
        delay_time = T[detection_time[0]] - tst
        if delay_time >0:
            Fault_indicator = 1
    else:
        delay_time = np.nan

    return Fault_indicator, delay_time, alrm, dur_alarm

    
def run_deltaEnergy():
    """automatically run deltaEnergy testing using leaky integrator as decision logic through datasets in cloud storage"""

    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)

    temp_file = os.path.join('.', 'temp_file.h5')
    
    Label = np.array([0] * 21 + [1] * 68)

    f0 = 50
    
    # parameter for delta Energy time constant
    tauU_deltaE=0.4
    tauD_deltaE =0.1
    harmonicComponents = np.array([3,5]) * f0

    # parameter for adaptive mean and variance

    buffer_dur = 1
    beta1_dur = 1  # mean time constant
    beta2_dur = 5  # variance time constant

    # parameter for integrator
    tauUp = 0.4
    tauDown = 0.4
    perc = 0.05
    alarm_threshold = 0.3
    alarm_dur = 0.8
    thr = 4
    

    Delaytime = []
    Fault_predict = []
    Alarm_duration = []

    for blob_name in blob_list:
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)


        if blob_name.endswith('.h5'):
            data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
            fs = calculate_sample_freq(data[TIME_COLUMN])

            file = os.path.basename(blob_name)
            figname = file.split('.')[0]

            # delete the first 2 s of data
            t0 = 2
            t = data.loc[:, 'Time']
            data = data.loc[(t > t0) & (t < 25), :]
            I0 = pd.DataFrame(data.iloc[:, 4:7].mean(axis=1).values, columns=['I0'])
            # data_new = pd.concat([d.iloc[:,4:7].reset_index().drop("index",axis='columns'),I0],axis=1)
            data_new = I0
            print(data_new.columns)

            t = t[(t > t0) & (t < 25)].reset_index().drop("index", axis='columns').values
            t = t - t[0]

            # set up parameters for DFT
            Npoints = round(fs / f0)
            window_size = Npoints * 2
            nshift = round(Npoints / 4)
            noverlap = window_size - nshift

            # extract filename 

            data_seg = np.array([])

            for i in range(data_new.shape[1]):
                delta_list = feature_deltaEnergy(data_new.iloc[:, i].values, window_size, noverlap, fs, harmonicComponents,
                                                 tauU_deltaE, tauD_deltaE)

                if i == 0:

                    data_seg = np.array(delta_list).reshape(-1, 1)
                else:
                    data_seg = np.concatenate((data_seg, np.array(delta_list).reshape(-1, 1)), axis=1)

            T = [t[i:i + window_size].mean() for i in np.arange(0, data_new.shape[0] - window_size + 1, nshift)]


            deltaT = (T[5] - T[0]) / 5

            T = np.array(T)

            beta1 = 1 - deltaT / beta1_dur
            beta2 = 1 - deltaT / beta2_dur
            print(beta1, beta2)

            mingap = f0 / fs * 2
           
            # calculate the start time
            stat = data.loc[:, 'Status'].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = t[indx[0][0]]
                print(tst)
            else:
                tst = None

            # calculate decision logic

            integrator, dtmean_array, dtvar_array, N = calculate_outBound_v2(data_seg, buffer_dur, perc, thr, deltaT,
                                                                             beta1, beta2, tauUp, tauDown,
                                                                             freeze_timelimit=5)

            integrator = np.array(integrator)
            Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T,
                                                                                    tst=tst,
                                                                                    alarm_threshold=alarm_threshold,
                                                                                    alarm_dur=alarm_dur)
            Delaytime.append(delay_time)
            Fault_predict.append(Fault_indicator)
            Alarm_duration.append(dur_alarm)
            xmax = 15


            # plots
            fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
            fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))

            for i in range(data_seg.shape[1]):

                axes[i, 0].plot(T, data_seg[:, i], color='k')
                axes[i, 0].plot(T, dtmean_array[:, i], color='y')
                axes[i, 0].plot(T, dtmean_array[:, i] + dtvar_array[:, i] * thr, '--', color='g')
                axes[i, 0].plot(T, dtmean_array[:, i] - dtvar_array[:, i] * thr, '--', color='g')
                axes[i, 0].set_xlim([0, xmax])

                ind = np.argwhere((np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtvar_array[:, i] * thr) & (
                            np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtmean_array[:, i] * perc))
                
                axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
                # axes[i,1].plot(T,dtvar_array[:,i],color ='b')
                # axes[i,1].plot(T,np.abs(dtmean_array[:,i]-data_seg[:,i]),'r')
                axes[i, 1].plot(T, N)
                axes[i, 1].set_xlim([0, xmax])
                # axes[i,1].set_ylim([0,0.5])
            axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
            axes[data_seg.shape[1], 0].set_ylim([0, 2])
            axes[data_seg.shape[1], 0].set_xlim([0, xmax])

            axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
            try: 
                axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
                axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
                axes[data_seg.shape[1], 1].set_xlim([0, xmax])

            except:
                print('No breaker status has been detected')

            plt.tight_layout()

            save_path = os.path.join(".\\plot_deltaEnergy", figname + '.png')
            plt.savefig(save_path, format='png', dpi=300)
            plt.show()

    # prediction & evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label[:len(Fault_predict)])

    print('DeltaEnergy Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, beta1:{}, beta2:{}, tauU_deltaE: {}, tauD_deltaE: {}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, beta1, beta2, tauU_deltaE, tauD_deltaE))


def run_stdDwt():
    """automatically run feature testing using leaky integrator as decision logic through datasets in cloud storage"""

    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)

   
    temp_file = os.path.join('.', 'temp_file.h5')

    Label = np.array([0] * 21 + [1] * 68)
    f0 = 50

    # parameter for adaptive mean and variance
    buffer_dur = 1
    beta1_dur = 1  # mean time constant
    beta2_dur = 5  # variance time constant
    
    # parameter for integrator
    alarm_threshold = 0.5
    alarm_dur = 0.6
    thr = 3
    tauUp = 0.4
    tauDown = 0.4
    perc = 0.05

    Delaytime = []
    Fault_predict = []
    Alarm_duration = []
    
    for blob_name in blob_list:
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5'):
            data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
            fs = calculate_sample_freq(data[TIME_COLUMN])
            file = os.path.basename(blob_name)
            figname = file.split('.')[0]

            # delete the first 2 s of data
            t0 = 2
            t = data.loc[:, 'Time']
            data = data.loc[(t > t0) & (t < 25), :]
            I0 = pd.DataFrame(data.iloc[:, 4:7].mean(axis=1).values, columns=['I0'])
            t = t[(t > t0) & (t < 25)].reset_index().drop("index", axis='columns').values
            t = t - t[0]

            # create new dataset with interested current as columns
            data_new = I0
            print(data_new.columns)

            # set up parameters for DWT 
            Npoints = round(fs / f0)
            window_size = Npoints * 4
            nshift = round(Npoints / 4)

            # extract features
            data_seg = np.array([])

            for i in range(data_new.shape[1]):

                if i == 0:
                    data_seg = np.array(
                        feature_stdDWT(data_new.iloc[:, i].values, window_size, nshift, 'db4', 4, 1)).reshape(-1, 1)
                else:
                    data_seg = np.concatenate((data_seg, np.array(
                        feature_stdDWT(data_new.iloc[:, i].values, window_size, nshift, 'db4', 4, 1)).reshape(-1, 1)),
                                              axis=1)

            # calculate time for feature series
            T = [t[i:i + window_size].mean() for i in np.arange(0, data_new.shape[0] - window_size, nshift)]
            deltaT = (T[5] - T[0]) / 5
            T = np.array(T)

            beta1 = 1 - deltaT / beta1_dur
            beta2 = 1 - deltaT / beta2_dur
            print(beta1, beta2)
            
            mingap = f0 / fs
            print(mingap)

            # calculate stimulus starting time
            stat = data.loc[:, 'Status'].to_numpy()
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = t[indx[0][0]]
                print(tst)
            else:
                tst = None

            # decision logic

            integrator, dtmean_array, dtvar_array = calculate_outBound_v2(data_seg, buffer_dur, perc, thr, deltaT,
                                                                          beta1, beta2, tauUp, tauDown,
                                                                          freeze_timelimit=2)

            integrator = np.array(integrator)
            Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T,
                                                                                    tst=tst,
                                                                                    alarm_threshold=alarm_threshold,
                                                                                    alarm_dur=alarm_dur)
            Delaytime.append(delay_time)
            Fault_predict.append(Fault_indicator)
            Alarm_duration.append(dur_alarm)

            # plot
            fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
            fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))

            for i in range(data_seg.shape[1]):
                axes[i, 0].plot(T, data_seg[:, i], color='k')
                axes[i, 0].plot(T, dtmean_array[:, i], color='y')
                axes[i, 0].plot(T, dtmean_array[:, i] + dtvar_array[:, i] * thr, '--', color='g')
                axes[i, 0].plot(T, dtmean_array[:, i] - dtvar_array[:, i] * thr, '--', color='g')
                axes[i, 0].set_xlim([0, 10])

                ind = np.argwhere((np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtvar_array[:, i] * thr) & (
                            np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtmean_array[:, i] * perc))

                axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
                axes[i, 1].plot(T, dtvar_array[:, i], color='b')
                axes[i, 1].plot(T, np.abs(dtmean_array[:, i] - data_seg[:, i]), 'r')
                axes[i, 1].set_xlim([0, 10])
                
            axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
            axes[data_seg.shape[1], 0].set_ylim([0, 2])
               
            axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
            try: 
                axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
                axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
            except:
                print('No breaker status has been detected')

            plt.tight_layout()
            save_path = os.path.join(".\\plot_dwt", figname + '.png')
            plt.savefig(save_path, format='png', dpi=300)
            plt.show()

    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Wavelet Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print("threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, beta1:{}, beta2:{}".format(thr,
                                                                                                             alarm_threshold,
                                                                                                             alarm_dur,
                                                                                                             tauUp,
                                                                                                             tauDown,
                                                                                                             beta1,
                                                                                                             beta2))


def grounding_channel_extract(data, sensor_list):
    """create a list containing the grounding current series for the sensors in sensor_list
    if the sensor name corresponds to 3-phase currents, grounding current will be calculated by 
    extract the mean of the 3-phase currents. If the sensor name corresponds to a grounding chanel,
    it will return its value as numpy array
    
    Args:
    data: python data frame contains current and voltage time series for different sensors
    sensor_list: list, contains the sensor names of interest
    
    Returns:
    list contains the grounding currents for all the sensors 
    
    """
    # column list
    column_list = data.columns
    grding_list = []
    
    # loop through all the sensor name in the sensor list
    for sns in sensor_list:
        
        # extract the current channel names corresponding to the sensor name
        ph3_curr = [chl for chl in column_list if chl.lower().__contains__(sns.lower())]
        
        # if it is 3-phase currents, calculate grounding current I0 by average
        if len(ph3_curr) == 3:
            I0 = data.loc[:, ph3_curr].mean(axis=1).values
            grding_list.append(I0)
            
        # if it is just a grounding channel, just export it
        elif len(ph3_curr) == 1:
            grding_list.append(data.loc[:, ph3_curr].values)
        
    return grding_list


def generate_data_block(grounding_list, st, ed, nshift, wsize, label):
    """ generate data block for training and testing data from grounding current list
    for each grounding current in the grounding_list, the data block from start time (st) 
    to end time (ed ) with length wsize,shift by nshift will be generated combining label
    as the last element
    
    Args:
    grounding_list: python list contains grounding currents (numpy array) from different 
    current sensors
    
    st: stimulus triggered time
    ed: end of the data collection
    nshift: number of samples shifted for windowed data block
    wsize: window size for a single data block
    label: 0: normal operation, 1: fault
    
    Returns:
    a python list with numpy arrays as single data block
    
    """
    
    # attach label to the single data block with length = wsize, start time and end time are with [st,ed]
    
    data_block = []
    for grd in grounding_list:
        dt = [np.append(grd[j:j + wsize], label) for j in range(st, ed - wsize, nshift)]
        data_block += dt
        
    return data_block


def generate_feature_block(data_block, fs, feature_dict, NoiseLevel=1.e-6):
    """ generate feature blocks 
    Args:
     data_block: list of grounding current blocks with Label at the end
     fs: sample frequency
     feature_dict: dictionary. keys are the features including 'stdDwt', 'timeCorr' and 'deltaEnergy'
     values are also dictionaries containing the parameters needed for feature calculation function
     NoiseLevel: noise level, default = 1.e-6
     Example:
        Npoints = int(fs/f0)
        window_size = Npoints*4
        nshift = int(Npoints//4)
        parmset_dwt={'wsize':window_size , 'wstep': nshift, 'level': 4, 'decompose_level':1, 'waveletname': 'db4'}
        rmvf = 3*f0 
        window_size_corr = Npoints*2
        noverlap = 0
        parmset_timecorr = {'wsize': window_size_corr, 'noverlap': noverlap, 'rmvf':rmvf, 'NoiseLevel': 0.0001}
        window_size_deltaE = Npoints*2
        nshift = int(Npoints//4)
        noverlap = window_size_deltaE-nshift
        tauU = 1
        tauD = 0.2
        parmset_deltaEng= {'wsize':window_size_deltaE,'noverlap': noverlap, 'harmonicComponents':[3*f0, 5*f0], 'tauUp': tauU, 'tauDown': tauD}
        feature_dict = {'stdDwt': parmset_dwt, 'timeCorr': parmset_timecorr, 'deltaEnergy':parmset_deltaEng}
    Returns:
     X: feature matrix (numpy array), each row contains all the features in the feature_dict
     Y: numpy array, includes all the labels for each row
    
    
    """
    X = []
    Y = []
    feature_len = {}

    for dt in data_block:
        dt_ = dt[:-1]

        dt_block = []

        if 'stdDwt' in feature_dict:
            std_comp = feature_stdDWT(dt_, feature_dict['stdDwt']['wsize'], feature_dict['stdDwt']['wstep'],
                                      feature_dict['stdDwt']['waveletname'], feature_dict['stdDwt']['level'],
                                      feature_dict['stdDwt']['decompose_level'], NoiseLevel=NoiseLevel)
            
            dt_block = dt_block + std_comp
            feature_len['stdDwt'] = len(std_comp)

        if 'timeCorr' in feature_dict:
            timecorr = feature_timecorr(dt_, feature_dict['timeCorr']['wsize'], feature_dict['timeCorr']['noverlap'],
                                        feature_dict['timeCorr']['rmvf'], fs,
                                        NoiseLevel=feature_dict['timeCorr']['NoiseLevel'])
            dt_block += timecorr
            feature_len['timeCorr'] = len(timecorr)

        if 'deltaEnergy' in feature_dict:
            deltaE = feature_deltaEnergy(dt_, feature_dict['deltaEnergy']['wsize'],
                                         feature_dict['deltaEnergy']['noverlap'], fs,
                                         feature_dict['deltaEnergy']['harmonicComponents'],
                                         feature_dict['deltaEnergy']['tauUp'], feature_dict['deltaEnergy']['tauDown'])
            dt_block += deltaE
            feature_len['deltaEnergy'] = len(deltaE)
            
        X.append(np.array(dt_block))
        Y.append(int(dt[-1]))
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    print(X.shape, Y.shape)

    print('std_comp length = {}'.format(len(std_comp)), 'time corr length = {}'.format(len(timecorr)),
          'DeltaEnergy length = {}'.format(len(deltaE)))

    return X, Y, feature_len



def generate_labeled_timeseries_dataBlocks_fromCloud(time_series_dur_sec, Ncyc, nshift_factor, type):
    """ generate a list containing labeled time series blocks from data files in cloud storage 
        for IEEE34 bus simulation. 
        Every element in the list is a numpy array [grounding currents blocks with length == Ncyc * N points per cycle, Label]
        Args:
            time_series_dur_sec: the length of the entire time series (unit = second) from when the breaker closes that allow to be seperated as data blocks
                                It could be the entire data traces but to avoid data duplication, shorted length is recommanded 
            Ncyc: number of cycles for each data blocks
            nshift_factor: number of cycles windows of data block shifting towards right
            type: 0: training, 1: testing
        
        Returns:
        a list contains data blocks

    """
    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)
    
    # generate a temperal file for storing data from cloud
    temp_file = os.path.join('.', 'temp_file.h5')

    # Label data sets
    Label = np.array([0] * 21 + [1] * 68)

    indx = np.argwhere(Label == type).flatten()

    f0 = 50
    data_block = []
    
    # Loop datasets
    for idx in indx:
        blob_name = blob_list[idx]
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5'):
            data = read_data(temp_file)
            fs = calculate_sample_freq(data[TIME_COLUMN])
            print(blob_name)
            
            # delete the first 2 s of data
            t0 = 2
            t = data.loc[:, 'Time']
            data = data.loc[(t > t0) & (t < 25), :].reset_index().drop("index", axis='columns')
            t = t[(t > t0) & (t < 25)].reset_index().drop("index", axis='columns').values
            t = t - t[0]

            # calculate grounding currents
            if type == 0:
                sensor_list = ['Icap', 'Ipt7', 'Ipt6', 'Is', 'Ipt3', 'Ipt4', 'Ipt2', 'Ipt1', 'Ipt8', 'Ipt5']
            else:
                sensor_list = ['Ipt4']

            groundings = grounding_channel_extract(data, sensor_list)

            # calculate brk status
            status = estimate_breaker_status(data, kwds_brkr='brk', threshold=0.5, kernel_size=99)
            st = np.where(status > 0)  # when brk switched to close
            if len(st[0]):
                st = int(st[0][0])
            else:
                st = 0

            # generate data blocks
            time_series_length = round(time_series_dur_sec * fs)
            ed = st + time_series_length
            Npoints = round(fs / f0)
            wsize = round(Npoints * Ncyc)
            nshift = round(Npoints * nshift_factor)
            dtb = generate_data_block(groundings, st, ed, nshift, wsize, Label[idx])
            
        data_block += dtb

    return data_block


def weight_generation(Y):
    """"calculate weight for model fitting, especially for unbalanced dataset
    Args:
    Y: label array
    Returns:
    weight: python dictionary 
    """
    if isinstance(Y, np.ndarray):
        Y = np.array(Y)
    Y = Y.reshape(len(Y), )
    unique_classes = np.unique(Y)
    weights = sklearn.utils.compute_class_weight(class_weight='balanced', classes=unique_classes, y=Y)
    weights_ = {c: weights[i] for i, c in enumerate(unique_classes)}

    return weights_


def simple_machineLearning_models(X, Y, weight):
    """ """
    if isinstance(Y, np.ndarray):
        Y = np.array(Y).reshape(len(Y), )
    
    if isinstance(X, np.ndarray):
        X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0, stratify=Y, )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight=weight, random_state=0, solver='lbfgs', max_iter=1500)
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0, class_weight=weight, random_state=0)
    rfc = RandomForestClassifier(class_weight=weight, random_state=0, max_depth=5)

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, name in [(lr, 'Logistic'), (gnb, 'Naive Bayes'), (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        clf.fit(X_train_scaled, y_train)
        if hasattr(clf, 'predict_proba'):
            prob_pos = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            prob_pos = clf.decision_function(X_test_scaled)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positive, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positive, "s-", label="%s" % (name,))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positive")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc='lower right')
    ax1.set_title("Calibration plots (reliability curve)")
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show


def run_timcorr_franksville():
    # load documentation file .xlsx to get the info about which datasets can be used
    pth_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\Franksville test results Second Run\TestPlan.xlsx"
    doc_info_HPL = pd.read_excel(pth_HPL, sheet_name='HPL')
    use_indicator = doc_info_HPL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HPL.loc[:, 'Test Name'].values
    use_test_name_HPL = test_name[use_indicator == 1]
    use_test_name_HPL = list(map(lambda x: x.replace(' ', ''), use_test_name_HPL))

    
    # load path file in HPL to get all the recorded files' paths
    pth_info_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HPL.csv"
    info_pth_HPL = pd.read_csv(pth_info_HPL)
    pth_list_HPL = [pth for pth in info_pth_HPL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HPL = []
    for testName in use_test_name_HPL:
        for pth in pth_list_HPL:
            if pth.__contains__(testName):
                test_pth_HPL.append(pth)
    print(len(test_pth_HPL), len(use_test_name_HPL))

    # HVL
    doc_info_HVL = pd.read_excel(pth_HPL, sheet_name='HVL')
    use_indicator = doc_info_HVL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HVL.loc[:, 'Test Name'].values
    use_test_name_HVL = test_name[use_indicator == 1]
    use_test_name_HVL_update = []
    for name in use_test_name_HVL:
        name_init = '000'
        name = str(name)
        name_init += name
        use_test_name_HVL_update.append(name_init[-3:])
    use_test_name_HVL = use_test_name_HVL_update
    

    # load path file in HVL to get all the recorded files' paths
    pth_info_HVL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HVL.csv"
    info_pth_HVL = pd.read_csv(pth_info_HVL)
    pth_list_HVL = [pth for pth in info_pth_HVL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HVL = []
    for testName in use_test_name_HVL:
        for pth in pth_list_HVL:
            file_name = os.path.basename(pth)
            file_name = file_name.split('.')[0][-3:]
        
            if file_name.__contains__(testName):
                test_pth_HVL.append(pth)
    print(len(test_pth_HVL), len(use_test_name_HVL))

    test_pth = test_pth_HPL + test_pth_HVL

    # parameters 
    thr = 0.7
    alarm_threshold = 0.5
    alarm_dur = 0.128
    NoiseLevel = 0.2
    Delaytime = []
    Fault_predict = []
    Alarm_duration = []
    tauUp = 0.1
    tauDown = 0.1

    Label = np.array([1] * len(test_pth))

    for pth in test_pth:
        data,_,_ = data_acquisition(pth, current_channels=['Fault_I', 'Fault_I_num_2', 'Ground/FaultCurrent'], kwds_brkr=['solenoid', 'Trigger'])
        fs = calculate_sample_freq(data[TIME_COLUMN])
        f0 = 60
        t = data.iloc[:, 0].values
        figname = os.path.basename(pth).split('.')[0]

        # set up parameters for DWT 
        Npoints = round(fs / f0)
        window_size = Npoints * 2
        noverlap = 0
        rmvf = 3 * f0
        nshift = window_size - noverlap

        # extract features
        data_seg = np.array([])
        if 'Fault_I_num_2' in data.columns:
            data_new = data.loc[:, 'Fault_I_num_2'].values.reshape(-1, 1)
        else:
            data_new = data.loc[:, 'Ground/FaultCurrent'].values.reshape(-1, 1)

        for i in range(data_new.shape[1]):
            dt = data_new[:, i]
            dt = dt - dt[:int(fs)].mean()

            if i == 0:
                data_seg = np.array(feature_timecorr(dt, window_size, noverlap, rmvf, fs, NoiseLevel)).reshape(-1, 1)
            else:
                data_seg = np.concatenate((data_seg, np.array(
                    feature_timecorr(data_new[:, i], window_size, noverlap, rmvf, fs, NoiseLevel)).reshape(-1, 1)),
                                          axis=1)

        # calculate time for feature series
        T = [t[i:i + window_size].mean() for i in np.arange(0, data_new.shape[0] - window_size, nshift)]
        T = [(T[i] + T[i + 1]) / 2 for i in np.arange(0, len(T) - 1)]
        deltaT = (T[5] - T[0]) / 5
        T = np.array(T)
        mingap = f0 / fs

        # calculate stimulus starting time
        stat = data.loc[:, 'Status'].to_numpy()
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = t[indx[0][0]]
            print(tst)
        else:
            tst = None

        # decision logic
        adapted_corr = np.ones((len(data_seg),))
        std_corr = np.ones((len(data_seg),))
        integrator = calculate_outBound_timecorr(data_seg, thr, deltaT, tauUp, tauDown)
        integrator = np.array(integrator)
        Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T, tst=tst,
                                                                                alarm_threshold=alarm_threshold,
                                                                                alarm_dur=alarm_dur)
        Delaytime.append(delay_time)
        Fault_predict.append(Fault_indicator)
        Alarm_duration.append(dur_alarm)

        # plot
        fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
        fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))

        for i in range(data_seg.shape[1]):
            axes[i, 0].plot(T, data_seg[:, i], color='k')
            axes[i, 0].plot(T, adapted_corr, color='y')
            # axes[i,0].plot(T,dtmean_array[:,i] + dtvar_array[:,i]*thr,'--', color = 'g')
            axes[i, 0].plot(T, std_corr * thr, '--', color='g')
            axes[i, 0].set_xlim([0, 10])

            ind = np.argwhere(data_seg[:, i] < std_corr * thr)

            axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
            
            axes[i, 1].plot(T, np.abs(adapted_corr - data_seg[:, i]), 'r')
            axes[i, 1].set_xlim([0, 10])
        
        axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
        axes[data_seg.shape[1], 0].set_ylim([0, 2])
            
        axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
        try: 
            axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
            axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
        except:
            print('No breaker status has been detected')

        plt.tight_layout()
        save_path = os.path.join(PLOT_SAVE_PATH, figname + '.png')
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()
 
    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Time corr Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, window_size:{}, rmvf:{}, NoiseLevel:{}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, window_size, rmvf, NoiseLevel))
    
    result = {}
    result['Delaytime'] = Delaytime
    result['Alarm_duration'] = Alarm_duration
    result['Accuracy'] = Accur
    result['False_alarm'] = False_alarm
    result['F1'] = F1
    result['parameters'] = {'threshold': thr, 'alarm_trheshold': alarm_threshold, 'alarm_dur': alarm_dur,
                            'tauUp': tauUp, 'tauDown': tauDown, 'window_size': window_size, 'NoiseLevel': NoiseLevel}

    # hist of Alarm_duration
    num_bins = 100
    plt.figure()
    Alarm_duration = np.array(Alarm_duration)
    Alarm_duration = Alarm_duration[Alarm_duration > 0]
    plt.hist(Alarm_duration, num_bins, density=1, color='green', alpha=0.7)
    plt.xlabel('First Over threshold duration (s)')
    plt.xscale('log')
    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    meanAlarmDur = np.nanmean(Alarm_duration)
    medianAlarmDur = np.nanmedian(Alarm_duration)
    result['meanAlarmDur'] = meanAlarmDur
    result['medianAlarmDur'] = medianAlarmDur
        
    # hist of Delaytime
    num_bins = 100
    Delaytime = [t[0] if isinstance(t, np.ndarray) else t for t in Delaytime]
    plt.figure()
    plt.hist(Delaytime, num_bins, density=1, color='blue', alpha=0.7)
    plt.xlabel('Delaytime (s)')
    plt.title('Delaytime \n\n',
              fontweight="bold")
    plt.xscale('log')
    plt.show()  

    Delaytime = np.array(Delaytime)
    meanDl = np.nanmean(Delaytime)
    medianDl = np.nanmedian(Delaytime)
    result['meanDelaytime'] = meanDl
    result['medianDelaytime'] = medianDl

    save_object(result, 'result_timecorr_Franskville_1.pickle')

    print('mean alarm duration = {}, median alarm duration = {}, mean delaytime ={}, median delaytime={}'.format(
        meanAlarmDur, medianAlarmDur, meanDl, medianDl))


def run_energycorr_franksville():
    # load documentation file .xlsx to get the info about which datasets can be used
    pth_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\Franksville test results Second Run\TestPlan.xlsx"
    doc_info_HPL = pd.read_excel(pth_HPL, sheet_name='HPL')
    use_indicator = doc_info_HPL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HPL.loc[:, 'Test Name'].values
    use_test_name_HPL = test_name[use_indicator == 1]
    use_test_name_HPL = list(map(lambda x: x.replace(' ', ''), use_test_name_HPL))

    # load path file in HPL to get all the recorded files' paths
    pth_info_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HPL.csv"
    info_pth_HPL = pd.read_csv(pth_info_HPL)
    pth_list_HPL = [pth for pth in info_pth_HPL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HPL = []
    for testName in use_test_name_HPL:
        for pth in pth_list_HPL:
            if pth.__contains__(testName):
                test_pth_HPL.append(pth)
    print(len(test_pth_HPL), len(use_test_name_HPL))

    # HVL
    doc_info_HVL = pd.read_excel(pth_HPL, sheet_name='HVL')
    use_indicator = doc_info_HVL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HVL.loc[:, 'Test Name'].values
    use_test_name_HVL = test_name[use_indicator == 1]
    use_test_name_HVL_update = []
    for name in use_test_name_HVL:
        name_init = '000'
        name = str(name)
        name_init += name
        use_test_name_HVL_update.append(name_init[-3:])
    use_test_name_HVL = use_test_name_HVL_update
    
    # load path file in HVL to get all the recorded files' paths
    pth_info_HVL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HVL.csv"
    info_pth_HVL = pd.read_csv(pth_info_HVL)
    pth_list_HVL = [pth for pth in info_pth_HVL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HVL = []
    for testName in use_test_name_HVL:
        for pth in pth_list_HVL:
            file_name = os.path.basename(pth)
            file_name = file_name.split('.')[0][-3:]
        
            if file_name.__contains__(testName):
                test_pth_HVL.append(pth)
    print(len(test_pth_HVL), len(use_test_name_HVL))

    test_pth = test_pth_HPL + test_pth_HVL

    # parameters 
    thr = 0.7
    alarm_threshold = 0.5
    alarm_dur = 0.128
    NoiseLevel = 0.2
    Delaytime = []
    Fault_predict = []
    Alarm_duration = []
    tauUp = 0.1
    tauDown = 0.4

    Label = np.array([1] * len(test_pth))

    for pth in test_pth:
        data,_,_ = data_acquisition(pth, current_channels=['Fault_I', 'Fault_I_num_2', 'Ground/FaultCurrent'], kwds_brkr=['solenoid', 'Trigger'])
        fs = calculate_sample_freq(data[TIME_COLUMN])
        f0 = 60
        t = data.iloc[:, 0].values
        figname = os.path.basename(pth).split('.')[0]

        # set up parameters for DWT 
        Npoints = round(fs / f0)
        smallwsize = Npoints
        nshift_small = round(Npoints / 4)
        bigwsize = Npoints * 2

        nshift_big = nshift_small
        harmonicComponents = np.array([3, 5]) * f0

        # extract features
        data_seg = np.array([])
        if 'Fault_I_num_2' in data.columns:
            data_new = data.loc[:, 'Fault_I_num_2'].values.reshape(-1, 1)
        else:
            data_new = data.loc[:, 'Ground/FaultCurrent'].values.reshape(-1, 1)

        for i in range(data_new.shape[1]):
            dt = data_new[:, i]
            dt = dt - dt[:int(fs)].mean()

            if i == 0:
                data_seg = np.array(feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                                       harmonicComponents, fs, f0, NoiseLevel, paddingM=None, P=1,
                                                       perc=0.9)).reshape(-1, 1)
            else:
                data_seg = np.concatenate((data_seg, np.array(
                    feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                       harmonicComponents, fs, f0, NoiseLevel, paddingM=None, P=1, perc=0.9)).reshape(
                    -1, 1)), axis=1)

        # calculate time for feature series
        T = [t[i:i + bigwsize].mean() for i in np.arange(0, data_new.shape[0] - bigwsize * 2 + 1, nshift_big)]
        T = [(T[i] + T[i + 1]) / 2 for i in np.arange(0, len(T) - 1)]
        
        deltaT = (T[5] - T[0]) / 5
        T = T + [T[-1] + deltaT]
        T = np.array(T)
        mingap = f0 / fs

        # calculate stimulus starting time
        stat = data[:, -1]
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = t[indx[0][0]]
            print(tst)
        else:
            tst = None

        # decision logic
        adapted_corr = np.ones((len(data_seg),))
        std_corr = np.ones((len(data_seg),))
        integrator = calculate_outBound_timecorr(data_seg, thr, deltaT, tauUp, tauDown)
        integrator = np.array(integrator)
        Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T, tst=tst,
                                                                                alarm_threshold=alarm_threshold,
                                                                                alarm_dur=alarm_dur)
        Delaytime.append(delay_time)
        Fault_predict.append(Fault_indicator)
        Alarm_duration.append(dur_alarm)

        # plot
        fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
        fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))

        for i in range(data_seg.shape[1]):
            axes[i, 0].plot(T, data_seg[:, i], color='k')
            axes[i, 0].plot(T, adapted_corr, color='y')
            # axes[i,0].plot(T,dtmean_array[:,i] + dtvar_array[:,i]*thr,'--', color = 'g')
            axes[i, 0].plot(T, std_corr * thr, '--', color='g')
            axes[i, 0].set_xlim([0, 10])

            ind = np.argwhere(data_seg[:, i] < std_corr * thr)

            axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
            
            axes[i, 1].plot(T, np.abs(adapted_corr - data_seg[:, i]), 'r')
            axes[i, 1].set_xlim([0, 10])
        
        axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
        axes[data_seg.shape[1], 0].set_ylim([0, 2])
            
        axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
        try: 
            axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
            axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
        except:
            print('No breaker status has been detected')

        plt.tight_layout()
        save_path = os.path.join(".\\plot_timecorr_franskville", figname + '.png')
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()
 
    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Time corr Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, window_size:{}, rmvf:{}, NoiseLevel:{}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, window_size, rmvf, NoiseLevel))
    
    result = {}
    result['Delaytime'] = Delaytime
    result['Alarm_duration'] = Alarm_duration
    result['Accuracy'] = Accur
    result['False_alarm'] = False_alarm
    result['F1'] = F1
    result['parameters'] = {'threshold': thr, 'alarm_trheshold': alarm_threshold, 'alarm_dur': alarm_dur,
                            'tauUp': tauUp, 'tauDown': tauDown, 'window_size': window_size, 'NoiseLevel': NoiseLevel}

    # hist of Alarm_duration
    num_bins = 100
    plt.figure()
    Alarm_duration = np.array(Alarm_duration)
    Alarm_duration = Alarm_duration[Alarm_duration > 0]
    plt.hist(Alarm_duration, num_bins, density=1, color='green', alpha=0.7)
    plt.xlabel('First Over threshold duration (s)')
    plt.xscale('log')
    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    meanAlarmDur = np.nanmean(Alarm_duration)
    medianAlarmDur = np.nanmedian(Alarm_duration)
    result['meanAlarmDur'] = meanAlarmDur
    result['medianAlarmDur'] = medianAlarmDur
        
    # hist of Delaytime
    num_bins = 100
    Delaytime = [t[0] if isinstance(t, np.ndarray) else t for t in Delaytime]
    plt.figure()
    plt.hist(Delaytime, num_bins, density=1, color='blue', alpha=0.7)
    plt.xlabel('Delaytime (s)')
    plt.title('Delaytime \n\n',
              fontweight="bold")
    plt.xscale('log')
    plt.show()  

    Delaytime = np.array(Delaytime)
    meanDl = np.nanmean(Delaytime)
    medianDl = np.nanmedian(Delaytime)
    result['meanDelaytime'] = meanDl
    result['medianDelaytime'] = medianDl

    save_object(result, 'result_timecorr_Franskville_1.pickle')
    print('mean alarm duration = {}, median alarm duration = {}, mean delaytime ={}, median delaytime={}'.format(
        meanAlarmDur, medianAlarmDur, meanDl, medianDl))


def energycorr_ieee34(blob_list, connect_str, container_name, t0, tend, Label, thr, tauUp, tauDown, NoiseLevel,
                      alarm_threshold, alarm_dur, harmonicComponents, savePlot=False, plot=False, R=1,
                      savefolder='.\\'):
    temp_file = os.path.join('.', 'temp_file.h5')
    Delaytime = []
    Fault_predict = []
    Alarm_duration = []

    caseN = 0
    for blob_name in blob_list:
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5'):
            data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
            os.remove(temp_file)
            fs = calculate_sample_freq(data[TIME_COLUMN])

            t = data.iloc[:, 0].values
            data = data.loc[(t > t0) & (t < tend), :].reset_index().drop("index", axis='columns').values
            
            t = t[(t > t0) & (t < tend)]
            t = t - t[0]
            f0 = 50
            file = os.path.basename(blob_name)
            filename = file.split('.')[0]
            print(filename, Label[caseN])

            # set up parameters for DWT 
            Npoints = round(fs / f0)
            smallwsize = Npoints
            nshift_small = round(Npoints / 4)
            bigwsize = Npoints * 2

            nshift_big = nshift_small
            # harmonicComponents=np.array([3,5])*f0

            # extract features
            data_seg = np.array([])
            data_new = data[:, 4:7].mean(axis=1).reshape(-1, 1)

            for i in range(data_new.shape[1]):
                dt = data_new[:, i]
                dt = dt - dt[:int(fs)].mean()

                if i == 0:
                    data_seg = np.array(
                        feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                           harmonicComponents, fs, f0, NoiseLevel, paddingM=None, P=1,
                                           perc=0.9)).reshape(-1, 1)
                else:
                    data_seg = np.concatenate((data_seg, np.array(
                        feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                           harmonicComponents, fs, f0, NoiseLevel, paddingM=None, P=1,
                                           perc=0.9)).reshape(-1, 1)), axis=1)

            # calculate time for feature series
            T = [t[i:i + bigwsize].mean() for i in np.arange(0, data_new.shape[0] - bigwsize * 2 + 1, nshift_big)]
            T = [(T[i] + T[i + 1]) / 2 for i in np.arange(0, len(T) - 1)]
            deltaT = (T[5] - T[0]) / 5
            T = T + [T[-1] + deltaT]
            T = np.array(T)
            mingap = f0 / fs

            # calculate stimulus starting time
            stat = data[:, -1]
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = t[indx[0][0]]
                print(tst)
            else:
                tst = None

            # decision logic
            adapted_corr = np.ones((len(data_seg),))
            std_corr = np.ones((len(data_seg),))
            integrator = calculate_outBound_timecorr(data_seg, thr, deltaT, tauUp, tauDown)
            integrator = np.array(integrator)
            Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T,
                                                                                    tst=tst,
                                                                                    alarm_threshold=alarm_threshold,
                                                                                    alarm_dur=alarm_dur)
            Delaytime.append(delay_time)
            Fault_predict.append(Fault_indicator)
            Alarm_duration.append(dur_alarm)
            caseN += 1

            # plot
            if plot:
                fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
                fig.suptitle(filename + ', Fault = {}'.format(Fault_indicator))
                xmax = 10

                for i in range(data_seg.shape[1]):
                    axes[i, 0].plot(T, data_seg[:, i], color='k')
                    axes[i, 0].plot(T, adapted_corr, color='y')
                    # axes[i,0].plot(T,dtmean_array[:,i] + dtvar_array[:,i]*thr,'--', color = 'g')
                    axes[i, 0].plot(T, std_corr * thr, '--', color='g')
                    axes[i, 0].set_xlim([0, xmax])

                    ind = np.argwhere(data_seg[:, i] < std_corr * thr)

                    axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
                    
                    axes[i, 1].plot(t, data_new[:, i], 'k')
                    axes[i, 1].plot([0, xmax], [NoiseLevel, NoiseLevel], 'r:')
                    axes[i, 1].set_xlim([0, xmax])
                
                axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
                axes[data_seg.shape[1], 0].set_ylim([0, 2])
                axes[data_seg.shape[1], 0].set_xlim([0, xmax])
                    
                axes[data_seg.shape[1], 0].plot([0, T[-1]], [0.5, 0.5], '--', color='orange')
                try: 
                    axes[data_seg.shape[1], 1].plot(t, data[:, -1], 'g')
                    axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
                    axes[data_seg.shape[1], 1].set_xlim([0, xmax])
                except:
                    print('No breaker status has been detected')

                plt.tight_layout()
                plt.show()
                if savePlot:
                    foldername = 'S{}'.format(R)
                    savefolder_ = os.path.join(savefolder, foldername)
                    if not os.path.exists(savefolder_):
                      os.mkdir(savefolder_)
                    save_path = os.path.join(savefolder_, filename + '.png')
                    plt.savefig(save_path, format='png', dpi=300)

    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Energy corr Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, bigwsize:{},smallwsize:{}, Harmonic:{},NoiseLevel:{}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, bigwsize, smallwsize, harmonicComponents, NoiseLevel))

    result = {}
    result['Delaytime'] = Delaytime
    result['Alarm_duration'] = Alarm_duration
    result['Accuracy'] = Accur
    result['False_alarm'] = False_alarm
    result['F1'] = F1
    result['parameters'] = {'threshold': thr, 'alarm_trheshold': alarm_threshold, 'alarm_dur': alarm_dur,
                            'tauUp': tauUp, 'tauDown': tauDown, 'bigwsize': bigwsize, 'smallwsize': smallwsize,
                            'NoiseLevel': NoiseLevel}

    # hist of Alarm_duration
    num_bins = 10
    plt.figure()
    Alarm_duration = [ad if ad is not None else 0 for ad in Alarm_duration]
    Alarm_duration = np.array(Alarm_duration)
    # Alarm_duration = Alarm_duration[Alarm_duration>0]
    plt.hist(Alarm_duration[Label == 0], num_bins, density=1, color='green', alpha=0.5)
    plt.hist(Alarm_duration[Label == 1], num_bins, density=1, color='', alpha=0.5)
    plt.xlabel('First Over threshold duration (s)')

    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    meanAlarmDur = np.nanmean(Alarm_duration)
    medianAlarmDur = np.nanmedian(Alarm_duration)
    result['meanAlarmDur'] = meanAlarmDur
    result['medianAlarmDur'] = medianAlarmDur

    # hist of Delaytime
    num_bins = 10
    Delaytime = [t[0] if isinstance(t, np.ndarray) else t for t in Delaytime]
    plt.figure()
    plt.hist(Delaytime, num_bins, density=1, color='blue', alpha=0.7)
    plt.xlabel('Delaytime (s)')
    plt.title('Delaytime \n\n',
              fontweight="bold")

    plt.show()

    Delaytime = np.array(Delaytime)
    meanDl = np.nanmean(Delaytime)
    medianDl = np.nanmedian(Delaytime)
    result['meanDelaytime'] = meanDl
    result['medianDelaytime'] = medianDl

    save_object(result, 'result_energycorr_Ieee34_oldversion{}.pickle'.format(R))
    print('mean alarm duration = {}, median alarm duration = {}, mean delaytime ={}, median delaytime={}'.format(
        meanAlarmDur, medianAlarmDur, meanDl, medianDl))
    return (
        thr, alarm_threshold, alarm_dur, tauUp, tauDown, Accur, F1, False_alarm, meanAlarmDur, meanDl, medianAlarmDur,
    medianDl)


def run_energycorr_franksville_v2():
    # load documentation file .xlsx to get the info about which datasets can be used
    pth_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\Franksville test results Second Run\TestPlan.xlsx"
    doc_info_HPL = pd.read_excel(pth_HPL, sheet_name='HPL')
    use_indicator = doc_info_HPL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HPL.loc[:, 'Test Name'].values
    use_test_name_HPL = test_name[use_indicator == 1]
    use_test_name_HPL = list(map(lambda x: x.replace(' ', ''), use_test_name_HPL))

    # load path file in HPL to get all the recorded files' paths
    pth_info_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HPL.csv"
    info_pth_HPL = pd.read_csv(pth_info_HPL)
    pth_list_HPL = [pth for pth in info_pth_HPL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HPL = []
    for testName in use_test_name_HPL:
        for pth in pth_list_HPL:
            if pth.__contains__(testName):
                test_pth_HPL.append(pth)
    print(len(test_pth_HPL), len(use_test_name_HPL))

    # HVL
    doc_info_HVL = pd.read_excel(pth_HPL, sheet_name='HVL')
    use_indicator = doc_info_HVL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HVL.loc[:, 'Test Name'].values
    use_test_name_HVL = test_name[use_indicator == 1]
    use_test_name_HVL_update = []
    for name in use_test_name_HVL:
        name_init = '000'
        name = str(name)
        name_init += name
        use_test_name_HVL_update.append(name_init[-3:])
    use_test_name_HVL = use_test_name_HVL_update

    # load path file in HVL to get all the recorded files' paths
    pth_info_HVL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HVL.csv"
    info_pth_HVL = pd.read_csv(pth_info_HVL)
    pth_list_HVL = [pth for pth in info_pth_HVL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HVL = []
    for testName in use_test_name_HVL:
        for pth in pth_list_HVL:
            file_name = os.path.basename(pth)
            file_name = file_name.split('.')[0][-3:]
        
            if file_name.__contains__(testName):
                test_pth_HVL.append(pth)
    print(len(test_pth_HVL), len(use_test_name_HVL))

    test_pth = test_pth_HPL + test_pth_HVL

    # parameters 
    thr = 0.7
    alarm_threshold = 0.3
    alarm_dur = 0.2
    NoiseLevel = 0
    Delaytime = []
    Fault_predict = []
    Alarm_duration = []
    tauUp = 0.4
    tauDown = 1
    # parameter for adaptive mean and variance
    buffer_dur = 1
    beta1_dur = 1  # mean time constant
    beta2_dur = 5  # variance time constant
    f0 = 60

    # parameter for delta Energy time constant
    tauU_deltaE = 1
    tauD_deltaE = 0.4
    perc = 0.05
    thr_delta = 3
    harmonicComponents = np.array([3, 5]) * f0

    Label = np.array([1] * len(test_pth))

    for pth in test_pth:
        data,_,_ = data_acquisition(pth,current_channels=['Fault_I', 'Fault_I_num_2', 'Ground/FaultCurrent'], kwds_brkr=['solenoid', 'Trigger'])
        fs = calculate_sample_freq(data[TIME_COLUMN])
        
        t = data.iloc[:, 0].values
        figname = os.path.basename(pth).split('.')[0]

        # set up parameters for DWT 
        Npoints = round(fs / f0)
        smallwsize = Npoints
        nshift_small = round(Npoints / 4)
        bigwsize = Npoints * 2
        nshift_big = nshift_small
        
        noverlap = bigwsize - nshift_big
        harmonicComponents = np.array([3, 5]) * f0

        # extract features
        data_seg = np.array([])
        if 'Fault_I_num_2' in data.columns:
            data_new = data.loc[:, 'Fault_I_num_2'].values.reshape(-1, 1)
        else:
            data_new = data.loc[:, 'Ground/FaultCurrent'].values.reshape(-1, 1)

        # calculate deltaEnergy
        data_seg_delta = np.array([])

        for i in range(data_new.shape[1]):
            delta_list = feature_deltaEnergy(data_new[:, i], bigwsize, noverlap, fs, harmonicComponents, tauU_deltaE,
                                             tauD_deltaE)

            if i == 0:

                data_seg_delta = np.array(delta_list).reshape(-1, 1)
            else:
                data_seg_delta = np.concatenate((data_seg_delta, np.array(delta_list).reshape(-1, 1)), axis=1)

        T_delta = [t[i:i + bigwsize].mean() for i in np.arange(0, data_new.shape[0] - bigwsize + 1, nshift_big)]

        deltaT = (T_delta[5] - T_delta[0]) / 5
        T_delta = np.array(T_delta)

        beta1 = 1 - deltaT / beta1_dur
        beta2 = 1 - deltaT / beta2_dur
        
        mingap = f0 / fs * 2

        # calculate integrator for delta energy
        integrator, _, _, _ = calculate_outBound_v2(data_seg_delta, buffer_dur, perc, thr_delta, deltaT, beta1, beta2,
                                                    tauUp, tauDown, freeze_timelimit=2)
        integrator = np.array(integrator)

        Alarm_indx = (integrator > 0.2) * 1
        label = measure.label(Alarm_indx)
        for lb in np.unique(label):
            if sum(label == lb) * deltaT < mingap and np.sum(Alarm_indx[label == lb]) == 0:
                Alarm_indx[label == lb] = 1

        label = measure.label(Alarm_indx)      

        # unique number of labels
        unique_label = np.unique(label)

        # extract features
        data_seg = np.array([])

        for i in range(data_new.shape[1]):
            dt = data_new[:, i]
            dt = dt - dt[:int(fs)].mean()

            if i == 0:
                data_seg = np.array(feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                                       harmonicComponents, fs, f0, NoiseLevel, paddingM=None, P=1,
                                                       perc=0.9)).reshape(-1, 1)
            else:
                data_seg = np.concatenate((data_seg, np.array(
                    feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                       harmonicComponents, fs, f0, NoiseLevel, paddingM=None, P=1, perc=0.9)).reshape(
                    -1, 1)), axis=1)

        # calculate time for feature series
        T = [t[i:i + bigwsize].mean() for i in np.arange(0, data_new.shape[0] - bigwsize * 2 + 1, nshift_big)]
        T = [(T[i] + T[i + 1]) / 2 for i in np.arange(0, len(T) - 1)]
        deltaT = (T[5] - T[0]) / 5
        T = T + [T[-1] + deltaT]
        T = np.array(T)
        mingap = f0 / fs
        for j in range(data_new.shape[1]): 

            dt = np.ones([len(T), ])
            for i in unique_label:
                if np.sum(Alarm_indx[label == i]) > 1:
                    Tmin = T_delta[label == i][0]
                    Tmax = T_delta[label == i][-1]
                    
                    dt[(T >= Tmin) & (T <= Tmax),] = data_seg[(T >= Tmin) & (T <= Tmax), j]
            data_seg[:, j] = dt

        # calculate stimulus starting time
        stat = data.loc[:, 'Status'].to_numpy()
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = t[indx[0][0]]
            print(tst)
        else:
            tst = 5

        # decision logic
        adapted_corr = np.ones((len(data_seg),))
        std_corr = np.ones((len(data_seg),))
        integrator = calculate_outBound_timecorr(data_seg, thr, deltaT, tauUp, tauDown)
        integrator = np.array(integrator)
        Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T, tst=tst,
                                                                                alarm_threshold=alarm_threshold,
                                                                                alarm_dur=alarm_dur)
        Delaytime.append(delay_time)
        Fault_predict.append(Fault_indicator)
        Alarm_duration.append(dur_alarm)

        xmax = min(10, t.max())
        # plot
        fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
        fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))

        for i in range(data_seg.shape[1]):
            axes[i, 0].plot(T, data_seg[:, i], color='k')
            axes[i, 0].plot(T, adapted_corr, color='y')
            # axes[i,0].plot(T,dtmean_array[:,i] + dtvar_array[:,i]*thr,'--', color = 'g')
            axes[i, 0].plot(T, std_corr * thr, '--', color='g')
            axes[i, 0].set_xlim([0, xmax])

            ind = np.argwhere(data_seg[:, i] < std_corr * thr)

            axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
        
            # axes[i,1].plot(T,np.abs(adapted_corr-data_seg[:,i]),'r')
            axes[i, 1].plot(t, data_new[:, i], 'k')

            axes[i, 1].set_xlim([0, xmax])
            
        axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
        axes[data_seg.shape[1], 0].set_ylim([0, 2])
        axes[data_seg.shape[1], 0].set_xlim([0, xmax])

        axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
        try: 
            axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
            axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
            axes[data_seg.shape[1], 1].set_xlim([0, xmax])
        except:
            print('No breaker status has been detected')

        plt.tight_layout()
        save_path = os.path.join(".\\plot_energycorr_franskville", figname + '.png')
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()

    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    
    print('Energy corr Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, bigwsize:{},smallwsize:{}, Harmonic:{}, NoiseLevel:{}, tauU_deltaE:{},tauD_deltaE:{},thr_delta:{}, buffer_dur:{}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, bigwsize, smallwsize, harmonicComponents, NoiseLevel,
            tauU_deltaE, tauD_deltaE, thr_delta,buffer_dur))

    result = {}
    result['Delaytime'] = Delaytime
    result['Alarm_duration'] = Alarm_duration
    result['Accuracy'] = Accur
    result['False_alarm'] = False_alarm
    result['F1'] = F1
    result['parameters'] = {'threshold': thr, 'alarm_trheshold': alarm_threshold, 'alarm_dur': alarm_dur,
                            'tauUp': tauUp, 'tauDown': tauDown, 'bigwsize': bigwsize, 'smallwsize': smallwsize,
                            'NoiseLevel': NoiseLevel, 'tauU_deltaE': tauU_deltaE, 'tauD_deltaE': tauD_deltaE,
                            'thr_delta': thr_delta, 'buffer_dur':buffer_dur}

    # hist of Alarm_duration
    
    num_bins = 10
    plt.figure()
    Alarm_duration = [ad if ad is not None else 0 for ad in Alarm_duration]
    Alarm_duration = np.array(Alarm_duration)
    # Alarm_duration = Alarm_duration[Alarm_duration>0]
    plt.hist(Alarm_duration, num_bins, density=1, color='green', alpha=0.5)
    
    plt.xlabel('First Over threshold duration (s)')
   
    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    Alarm_duration = np.array(Alarm_duration)
    plt.hist(Alarm_duration, num_bins, density=1, color='green', alpha=0.7)
    plt.xlabel('First Over threshold duration (s)')
    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    meanAlarmDur = np.nanmean(Alarm_duration)
    medianAlarmDur = np.nanmedian(Alarm_duration)
    result['meanAlarmDur'] = meanAlarmDur
    result['medianAlarmDur'] = medianAlarmDur
        
    # hist of Delaytime
    num_bins = 10
    Delaytime = [t[0] if isinstance(t, np.ndarray) else t for t in Delaytime]
    plt.figure()
    plt.hist(Delaytime, num_bins, density=1, color='blue', alpha=0.7)
    plt.xlabel('Delaytime (s)')
    plt.title('Delaytime \n\n',
              fontweight="bold")
    plt.xscale('log')
    plt.show()  

    Delaytime = np.array(Delaytime)
    meanDl = np.nanmean(Delaytime)
    medianDl = np.nanmedian(Delaytime)
    result['meanDelaytime'] = meanDl
    result['medianDelaytime'] = medianDl

    save_object(result, 'result_energycorr_Franskville_1.pickle')
    print('mean alarm duration = {}, median alarm duration = {}, mean delaytime ={}, median delaytime={}'.format(
        meanAlarmDur, medianAlarmDur, meanDl, medianDl))


def run_deltaEnergy_franksville():
    # load documentation file .xlsx to get the info about which datasets can be used
    pth_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\Franksville test results Second Run\TestPlan.xlsx"
    doc_info_HPL = pd.read_excel(pth_HPL, sheet_name='HPL')
    use_indicator = doc_info_HPL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HPL.loc[:, 'Test Name'].values
    use_test_name_HPL = test_name[use_indicator == 1]
    use_test_name_HPL = list(map(lambda x: x.replace(' ', ''), use_test_name_HPL))

    # load path file in HPL to get all the recorded files' paths
    pth_info_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HPL.csv"
    info_pth_HPL = pd.read_csv(pth_info_HPL)
    pth_list_HPL = [pth for pth in info_pth_HPL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HPL = []
    for testName in use_test_name_HPL:
        for pth in pth_list_HPL:
            if pth.__contains__(testName):
                test_pth_HPL.append(pth)
    print(len(test_pth_HPL), len(use_test_name_HPL))

    # HVL
    doc_info_HVL = pd.read_excel(pth_HPL, sheet_name='HVL')
    use_indicator = doc_info_HVL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HVL.loc[:, 'Test Name'].values
    use_test_name_HVL = test_name[use_indicator == 1]
    use_test_name_HVL_update = []
    for name in use_test_name_HVL:
        name_init = '000'
        name = str(name)
        name_init += name
        use_test_name_HVL_update.append(name_init[-3:])
    use_test_name_HVL = use_test_name_HVL_update
    
    # load path file in HVL to get all the recorded files' paths
    pth_info_HVL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HVL.csv"
    info_pth_HVL = pd.read_csv(pth_info_HVL)
    pth_list_HVL = [pth for pth in info_pth_HVL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HVL = []
    for testName in use_test_name_HVL:
        for pth in pth_list_HVL:
            file_name = os.path.basename(pth)
            file_name = file_name.split('.')[0][-3:]
        
            if file_name.__contains__(testName):
                test_pth_HVL.append(pth)
    print(len(test_pth_HVL), len(use_test_name_HVL))

    test_pth = test_pth_HPL + test_pth_HVL
    
    Label = np.array([0] * 21 + [1] * 68)
    
    # parameter for delta Energy time constant
    tauU_deltaE = 1
    tauD_deltaE = 0.4

    # parameter for adaptive mean and variance
    buffer_dur = 1
    beta1_dur = 1  # mean time constant
    beta2_dur = 5  # variance time constant

    # parameter for integrator
    tauUp = 0.1
    tauDown = 0.4
    perc = 0.05
    alarm_threshold = 0.3
    alarm_dur = 0.8
    thr = 3
    
    Delaytime = []
    Fault_predict = []
    Alarm_duration = []

    for pth in test_pth:
        data,_,_ = data_acquisition(pth, current_channels=['Fault_I', 'Fault_I_num_2', 'Ground/FaultCurrent'], kwds_brkr=['solenoid', 'Trigger'])
        fs = calculate_sample_freq(data[TIME_COLUMN])
        f0 = 60
        t = data.iloc[:, 0].values
        figname = os.path.basename(pth).split('.')[0]

        # set up parameters for DWT 
        Npoints = round(fs / f0)
        smallwsize = Npoints
        nshift_small = round(Npoints / 4)
        bigwsize = Npoints * 2
        nshift_big = nshift_small
        
        noverlap = bigwsize - nshift_big
        harmonicComponents = np.array([3, 5]) * f0

        # extract features
        data_seg = np.array([])
        if 'Fault_I_num_2' in data.columns:
            data_new = data.loc[:, 'Fault_I_num_2'].values.reshape(-1, 1)
        else:
            data_new = data.loc[:, 'Ground/FaultCurrent'].values.reshape(-1, 1)

        # set up parameters for DFT
        Npoints = round(fs / f0)
        window_size = Npoints * 2
        nshift = round(Npoints / 4)
        noverlap = window_size - nshift

        # extract filename 

        data_seg = np.array([])

        for i in range(data_new.shape[1]):
            delta_list = feature_deltaEnergy(data_new[:, i], window_size, noverlap, fs, harmonicComponents, tauU_deltaE,
                                             tauD_deltaE)

            if i == 0:

                data_seg = np.array(delta_list).reshape(-1, 1)
            else:
                data_seg = np.concatenate((data_seg, np.array(delta_list).reshape(-1, 1)), axis=1)

        T = [t[i:i + window_size].mean() for i in np.arange(0, data_new.shape[0] - window_size + 1, nshift)]

        deltaT = (T[5] - T[0]) / 5
        T = np.array(T)

        beta1 = 1 - deltaT / beta1_dur
        beta2 = 1 - deltaT / beta2_dur
        print(beta1, beta2)

        mingap = f0 / fs * 2
        
        # calculate the start time
        stat = data.loc[:, 'Status'].to_numpy()
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = t[indx[0][0]]
            print(tst)
        else:
            tst = None

        # calculate decision logic
        integrator, dtmean_array, dtvar_array, N = calculate_outBound_v2(data_seg, buffer_dur, perc, thr, deltaT, beta1,
                                                                         beta2, tauUp, tauDown, freeze_timelimit=5)
        integrator = np.array(integrator)
        Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T, tst=tst,
                                                                                alarm_threshold=alarm_threshold,
                                                                                alarm_dur=alarm_dur)
        Delaytime.append(delay_time)
        Fault_predict.append(Fault_indicator)
        Alarm_duration.append(dur_alarm)
        xmax = 10

        # plots
        fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
        fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))

        for i in range(data_seg.shape[1]):
            axes[i, 0].plot(T, data_seg[:, i], color='k')
            axes[i, 0].plot(T, dtmean_array[:, i], color='y')
            axes[i, 0].plot(T, dtmean_array[:, i] + dtvar_array[:, i] * thr, '--', color='g')
            axes[i, 0].plot(T, dtmean_array[:, i] - dtvar_array[:, i] * thr, '--', color='g')
            axes[i, 0].set_xlim([0, xmax])

            ind = np.argwhere((np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtvar_array[:, i] * thr) & (
                        np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtmean_array[:, i] * perc))

            axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
            # axes[i,1].plot(T,dtvar_array[:,i],color ='b')
            # axes[i,1].plot(T,np.abs(dtmean_array[:,i]-data_seg[:,i]),'r')
            axes[i, 1].plot(T, N)
            axes[i, 1].set_xlim([0, xmax])
            # axes[i,1].set_ylim([0,0.5])
        axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
        axes[data_seg.shape[1], 0].set_ylim([0, 2])
        axes[data_seg.shape[1], 0].set_xlim([0, xmax])

        axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
        try: 
            axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
            axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
            axes[data_seg.shape[1], 1].set_xlim([0, xmax])
        except:
            print('No breaker status has been detected')

        plt.tight_layout()
        save_path = os.path.join(".\\plot_deltaEnergy", figname + '.png')
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()

    # prediction & evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label[:len(Fault_predict)])
    print('DeltaEnergy Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, beta1:{}, beta2:{}, tauU_deltaE: {}, tauD_deltaE: {}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, beta1, beta2, tauU_deltaE, tauD_deltaE))


def energycorr_ieee34_v2(blob_list, connect_str, container_name, t0, tend, Label, thr, tauUp, tauDown, alarm_threshold,
                         alarm_dur, savePlot=False, plot=False, R=1, savefolder='.\\'):
    temp_file = os.path.join('.', 'temp_file.h5')
    Delaytime = []
    Fault_predict = []
    Alarm_duration = []

    # parameter for adaptive mean and variance
    buffer_dur = 1
    beta1_dur = 1  # mean time constant
    beta2_dur = 5  # variance time constant

    # parameter for delta Energy time constant
    tauU_deltaE = 1
    tauD_deltaE = 0.4
    perc = 0.05
    thr_delta = 4

    caseN = 0
    for blob_name in blob_list:
        blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name,
                                                        blob_name=blob_name)
        with open(temp_file, "wb") as my_blob:
            blob_data = blob_client.download_blob()
            blob_data.readinto(my_blob)

        if blob_name.endswith('.h5'):
            data,_,_ = data_acquisition(temp_file, nodes=["pt4"])
            os.remove(temp_file)
            fs = calculate_sample_freq(data[TIME_COLUMN])

            t = data.iloc[:, 0].values
            data = data.loc[(t > t0) & (t < tend), :].reset_index().drop("index", axis='columns').values
            
            t = t[(t > t0) & (t < tend)]
            t = t - t[0]
            f0 = 50
            file = os.path.basename(blob_name)
            filename = file.split('.')[0]
            print(filename, Label[caseN])

            # set up parameters for DWT 
            Npoints = round(fs / f0)
            smallwsize = Npoints
            nshift_small = round(Npoints / 4)
            bigwsize = Npoints * 2
            nshift_big = nshift_small
            
            noverlap = bigwsize - nshift_big
            harmonicComponents = np.array([3, 5]) * f0

            # extract features
            data_seg = np.array([])
            data_new = data[:, 4:7].mean(axis=1).reshape(-1, 1)

            # calculate deltaEnergy
            data_seg_delta = np.array([])

            for i in range(data_new.shape[1]):
                delta_list = feature_deltaEnergy(data_new[:, i], bigwsize, noverlap, fs, harmonicComponents, tauU_deltaE,
                                                 tauD_deltaE)

                if i == 0:

                    data_seg_delta = np.array(delta_list).reshape(-1, 1)
                else:
                    data_seg_delta = np.concatenate((data_seg_delta, np.array(delta_list).reshape(-1, 1)), axis=1)

            T_delta = [t[i:i + bigwsize].mean() for i in np.arange(0, data_new.shape[0] - bigwsize + 1, nshift_big)]

            deltaT = (T_delta[5] - T_delta[0]) / 5
            T_delta = np.array(T_delta)

            beta1 = 1 - deltaT / beta1_dur
            beta2 = 1 - deltaT / beta2_dur
            
            mingap = f0 / fs * 4

            # calculate integrator for delta energy
            integrator, _, _, _ = calculate_outBound_v2(data_seg_delta, buffer_dur, perc, thr_delta, deltaT, beta1,
                                                        beta2, tauUp, tauDown, freeze_timelimit=5)
            integrator = np.array(integrator)

            Alarm_indx = (integrator > 0.2) * 1
            label = measure.label(Alarm_indx)
            for lb in np.unique(label):
                if sum(label == lb) * deltaT < mingap and np.sum(Alarm_indx[label == lb]) == 0:
                    Alarm_indx[label == lb] = 1

            label = measure.label(Alarm_indx)      

            # unique number of labels
            unique_label = np.unique(label)

            for i in range(data_new.shape[1]):
                dt = data_new[:, i]
                dt = dt - dt[:int(fs)].mean()

                if i == 0:
                    data_seg = np.array(
                        feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                           harmonicComponents, fs, f0, 0, paddingM=None, P=1, perc=0.9)).reshape(-1, 1)
                else:
                    data_seg = np.concatenate((data_seg, np.array(
                        feature_energycorr(data_new[:, i], bigwsize, nshift_big, smallwsize, nshift_small,
                                           harmonicComponents, fs, f0, 0, paddingM=None, P=1, perc=0.9)).reshape(-1,
                                                                                                                 1)),
                                              axis=1)

            # calculate time for feature series
            T = [t[i:i + bigwsize].mean() for i in np.arange(0, data_new.shape[0] - bigwsize * 2 + 1, nshift_big)]

            T = [(T[i] + T[i + 1]) / 2 for i in np.arange(0, len(T) - 1)]
            deltaT = (T[5] - T[0]) / 5
            T = T + [T[-1] + deltaT]
            T = np.array(T)
            # explore all the regions, if the length of the region is less than minstep, set the sign_sig within that region 0
            for j in range(data_new.shape[1]): 

                dt = np.ones([len(T), ])
                for i in unique_label:
                    if np.sum(Alarm_indx[label == i]) > 1:
                        Tmin = T_delta[label == i][0]
                        Tmax = T_delta[label == i][-1]
                        
                        dt[(T >= Tmin) & (T <= Tmax),] = data_seg[(T >= Tmin) & (T <= Tmax), j]
                data_seg[:, j] = dt
                    
            mingap = f0 / fs * 4

            # calculate stimulus starting time
            stat = data[:, -1]
            indx = np.nonzero(np.diff(stat))
            if len(indx[0]):
                tst = t[indx[0][0]]
                print(tst)
            else:
                tst = None

            # decision logic
            adapted_corr = np.ones((len(data_seg),))
            std_corr = np.ones((len(data_seg),))
            integrator = calculate_outBound_timecorr(data_seg, thr, deltaT, tauUp, tauDown)
            integrator = np.array(integrator)
            Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T,
                                                                                    tst=tst,
                                                                                    alarm_threshold=alarm_threshold,
                                                                                    alarm_dur=alarm_dur)
            Delaytime.append(delay_time)
            Fault_predict.append(Fault_indicator)
            Alarm_duration.append(dur_alarm)
            caseN += 1

            # plot
            if plot:
                fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
                fig.suptitle(filename + ', Fault = {}'.format(Fault_indicator))
                xmax = 10

                for i in range(data_seg.shape[1]):
                    axes[i, 0].plot(T, data_seg[:, i], color='k')
                    axes[i, 0].plot(T, adapted_corr, color='y')
                    # axes[i,0].plot(T,dtmean_array[:,i] + dtvar_array[:,i]*thr,'--', color = 'g')
                    axes[i, 0].plot(T, std_corr * thr, '--', color='g')
                    axes[i, 0].set_xlim([0, xmax])

                    ind = np.argwhere(data_seg[:, i] < std_corr * thr)

                    axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
                    
                    axes[i, 1].plot(t, data_new[:, i], 'k')
                    # axes[i,1].plot([0, xmax],[NoiseLevel, NoiseLevel],'r:')
                    axes[i, 1].set_xlim([0, xmax])
                    
                axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
                axes[data_seg.shape[1], 0].set_ylim([0, 2])
                axes[data_seg.shape[1], 0].set_xlim([0, xmax])

                axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
                try: 
                    axes[data_seg.shape[1], 1].plot(t, data[:, -1], 'g')
                    axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
                    axes[data_seg.shape[1], 1].set_xlim([0, xmax])
                except:
                    print('No breaker status has been detected')

                plt.tight_layout()
                plt.show()
 
                if savePlot:
                    foldername = 'S{}'.format(R)
                    savefolder_ = os.path.join(savefolder, foldername)
                    if not os.path.exists(savefolder_):
                      os.mkdir(savefolder_)
                    save_path = os.path.join(savefolder_, filename + '.png')
                    plt.savefig(save_path, format='png', dpi=300)

    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Energy corr Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print(
        "threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, bigwsize:{},smallwsize:{} Harmonic:{}, threshold_delta: {}".format(
            thr, alarm_threshold, alarm_dur, tauUp, tauDown, bigwsize, smallwsize, harmonicComponents, thr_delta))
    
    result = {}
    result['Delaytime'] = Delaytime
    result['Alarm_duration'] = Alarm_duration
    result['Accuracy'] = Accur
    result['False_alarm'] = False_alarm
    result['F1'] = F1
    result['parameters'] = {'threshold': thr, 'alarm_trheshold': alarm_threshold, 'alarm_dur': alarm_dur,
                            'tauUp': tauUp, 'tauDown': tauDown, 'bigwsize': bigwsize, 'smallwsize': smallwsize,
                            'tauD_deltaE': tauD_deltaE, 'tauU_deltaE': tauU_deltaE, 'thr_Delta': thr_delta}

    # hist of Alarm_duration
    num_bins = 10
    plt.figure()

    Alarm_duration = [ad if ad is not None else 0 for ad in Alarm_duration]
    Alarm_duration = np.array(Alarm_duration)
    # Alarm_duration = Alarm_duration[Alarm_duration>0]
    plt.hist(Alarm_duration[Label == 0], num_bins, density=1, color='green', alpha=0.5)
    plt.hist(Alarm_duration[Label == 1], num_bins, density=1, color='y', alpha=0.5)
 
    plt.xlabel('First Over threshold duration (s)')

    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    meanAlarmDur = np.nanmean(Alarm_duration)
    medianAlarmDur = np.nanmedian(Alarm_duration)
    result['meanAlarmDur'] = meanAlarmDur
    result['medianAlarmDur'] = medianAlarmDur
        
    # hist of Delaytime
    num_bins = 10
    Delaytime = [t[0] if isinstance(t, np.ndarray) else t for t in Delaytime]
    plt.figure()
    plt.hist(Delaytime, num_bins, density=1, color='blue', alpha=0.7)
    plt.xlabel('Delaytime (s)')
    plt.title('Delaytime \n\n',
              fontweight="bold")
    plt.xscale('log')
    plt.show()  

    Delaytime = np.array(Delaytime)
    meanDl = np.nanmean(Delaytime)
    medianDl = np.nanmedian(Delaytime)
    result['meanDelaytime'] = meanDl
    result['medianDelaytime'] = medianDl

    save_object(result, 'result_energycorr_Ieee34_v2_{}.pickle'.format(R))
    print('mean alarm duration = {}, median alarm duration = {}, mean delaytime ={}, median delaytime={}'.format(
        meanAlarmDur, medianAlarmDur, meanDl, medianDl))
    return (
    thr, alarm_threshold, alarm_dur, tauUp, tauDown, Accur, F1, False_alarm, meanAlarmDur, meanDl, medianAlarmDur,
    medianDl)


def run_energycorr_ieee34():
    os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

    # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a unique name for the container

    container_name = "ieee34"
    local_path = "."

    # Create the container
    container_client = blob_service_client.get_container_client(container_name)
    # walk_container(container_client,connect_str, container_name,local_path)
    blob_list = get_blob_list(container_client)

    Label = np.array([0] * 21 + [1] * 68)
    t0 = 2
    tend = 20
    f0 = 50
    Record_array = []
    thr = np.arange(1, 0.5, -0.1)
    alarm_threshold = np.arange(0.1, 0.5, 0.1)
    alarm_dur = np.arange(0.1, 1, 0.1)
    tauU_outBund = np.arange(0.1, 1, 0.1)
    NoiseLevel = np.arange(0.0001, 0.0006, 0.0001)
    harmonicComponents = np.array([3, 5]) * f0
    L = 0
    for tU in tauU_outBund:
        for alm in alarm_threshold:
            for alm_dur in alarm_dur:
                for tr in thr:
                    for nlevel in NoiseLevel:
                        rd = energycorr_ieee34(blob_list, connect_str, container_name, t0, tend, Label, tr, tU, tU,
                                               nlevel, alm, alm_dur, harmonicComponents, R=L)
                        print(rd)
                        Record_array.append(rd)
                        save_object(Record_array, 'energycorr_parameterExplore.pickle')
                        L += 1


def run_stdDwt_franksville():
    """automatically run feature testing using leaky integrator as decision logic through datasets in Franksville"""

    # load documentation file .xlsx to get the info about which datasets can be used
    pth_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\Franksville test results Second Run\TestPlan.xlsx"
    doc_info_HPL = pd.read_excel(pth_HPL, sheet_name='HPL')
    use_indicator = doc_info_HPL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HPL.loc[:, 'Test Name'].values
    use_test_name_HPL = test_name[use_indicator == 1]
    use_test_name_HPL = list(map(lambda x: x.replace(' ', ''), use_test_name_HPL))

    # load path file in HPL to get all the recorded files' paths
    pth_info_HPL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HPL.csv"
    info_pth_HPL = pd.read_csv(pth_info_HPL)
    pth_list_HPL = [pth for pth in info_pth_HPL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HPL = []
    for testName in use_test_name_HPL:
        for pth in pth_list_HPL:
            if pth.__contains__(testName):
                test_pth_HPL.append(pth)
    print(len(test_pth_HPL), len(use_test_name_HPL))

    # HVL
    doc_info_HVL = pd.read_excel(pth_HPL, sheet_name='HVL')
    use_indicator = doc_info_HVL.loc[:, 'Use for Algorithm'].values
    test_name = doc_info_HVL.loc[:, 'Test Name'].values
    use_test_name_HVL = test_name[use_indicator == 1]
    use_test_name_HVL_update = []
    for name in use_test_name_HVL:
        name_init = '000'
        name = str(name)
        name_init += name
        use_test_name_HVL_update.append(name_init[-3:])
    use_test_name_HVL = use_test_name_HVL_update

    # load path file in HVL to get all the recorded files' paths
    pth_info_HVL = r"\\loutcnas002\golden\ESSR\HiZ Fault Data\fileNameList_new_HVL.csv"
    info_pth_HVL = pd.read_csv(pth_info_HVL)
    pth_list_HVL = [pth for pth in info_pth_HVL]

    # extract the file path info for the datasets satisfying our critia for data analysis
    test_pth_HVL = []
    for testName in use_test_name_HVL:
        for pth in pth_list_HVL:
            file_name = os.path.basename(pth)
            file_name = file_name.split('.')[0][-3:]
        
            if file_name.__contains__(testName):
                test_pth_HVL.append(pth)
    print(len(test_pth_HVL), len(use_test_name_HVL))

    test_pth = test_pth_HPL + test_pth_HVL

    # parameter for adaptive mean and variance
    buffer_dur = 1
    beta1_dur = 2  # mean time constant
    beta2_dur = 5  # variance time constant
    
    # parameter for integrator
    alarm_threshold = 0.3
    alarm_dur = 0.8
    thr = 2.5
    tauUp = 0.2
    tauDown = 0.4
    perc = 0
    f0 = 60
    
    Label = np.array([1] * len(test_pth))

    Delaytime = []
    Fault_predict = []
    Alarm_duration = []

    savefolder = os.path.join(".\\plot_dwt", 'Fransville')
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    for pth in test_pth:
        data,_,_ = data_acquisition(pth, current_channels=['Fault_I', 'Fault_I_num_2', 'Ground/FaultCurrent'], kwds_brkr=['solenoid', 'Trigger'])
        fs = calculate_sample_freq(data[TIME_COLUMN])
        t = data.iloc[:, 0].values
        figname = os.path.basename(pth).split('.')[0]

        # set up parameters for DWT 
        Npoints = round(fs / f0)
        window_size = Npoints * 4
        nshift = round(Npoints / 4)

        # extract features
        data_seg = np.array([])
        if 'Fault_I_num_2' in data.columns:
            data_new = data.loc[:, 'Fault_I_num_2'].values.reshape(-1, 1)
        else:
            data_new = data.loc[:, 'Ground/FaultCurrent'].values.reshape(-1, 1)
        for i in range(data_new.shape[1]):

            if i == 0:
                data_seg = np.array(feature_stdDWT(data_new[:, i], window_size, nshift, 'db4', 4, 1)).reshape(-1, 1)
            else:
                data_seg = np.concatenate((data_seg, np.array(
                    feature_stdDWT(data_new[:, i], window_size, nshift, 'db4', 4, 1)).reshape(-1, 1)), axis=1)

        # calculate time for feature series
        T = [t[i:i + window_size].mean() for i in np.arange(0, data_new.shape[0] - window_size, nshift)]
        deltaT = (T[5] - T[0]) / 5
        T = np.array(T)

        beta1 = 1 - deltaT / beta1_dur
        beta2 = 1 - deltaT / beta2_dur
        print(beta1, beta2)
        mingap = f0 / fs * 4
        print(mingap)


        # calculate stimulus starting time
        stat = data.loc[:, 'Status'].to_numpy()
        indx = np.nonzero(np.diff(stat))
        if len(indx[0]):
            tst = t[indx[0][0]]
            print(tst)
        else:
            tst = None

        # decision logic
        integrator, dtmean_array, dtvar_array, _ = calculate_outBound_v2(data_seg, buffer_dur, perc, thr, deltaT, beta1,
                                                                         beta2, tauUp, tauDown, freeze_timelimit=5)
        integrator = np.array(integrator)
        Fault_indicator, delay_time, alrm, dur_alarm = calculate_faultIndicator(integrator, deltaT, mingap, T, tst=tst,
                                                                                alarm_threshold=alarm_threshold,
                                                                                alarm_dur=alarm_dur)
        Delaytime.append(delay_time)
        Fault_predict.append(Fault_indicator)
        Alarm_duration.append(dur_alarm)

        # plot
        fig, axes = plt.subplots(data_seg.shape[1] + 1, 2)
        fig.suptitle(figname + ', Fault = {}'.format(Fault_indicator))
        xmax = min(15, np.min(t.max()))

        for i in range(data_seg.shape[1]):
            axes[i, 0].plot(T, data_seg[:, i], color='k')
            axes[i, 0].plot(T, dtmean_array[:, i], color='y')
            axes[i, 0].plot(T, dtmean_array[:, i] + dtvar_array[:, i] * thr, '--', color='g')
            axes[i, 0].plot(T, dtmean_array[:, i] - dtvar_array[:, i] * thr, '--', color='g')
            axes[i, 0].set_xlim([0, xmax])

            ind = np.argwhere((np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtvar_array[:, i] * thr) & (
                        np.abs(dtmean_array[:, i] - data_seg[:, i]) > dtmean_array[:, i] * perc))

            
            axes[i, 0].plot(T[ind], data_seg[ind, i], '.', color='r')
            axes[i, 1].plot(T, dtvar_array[:, i], color='b')
            axes[i, 1].plot(T, np.abs(dtmean_array[:, i] - data_seg[:, i]), 'r')
            axes[i, 1].set_xlim([0, xmax])
            
        axes[data_seg.shape[1], 0].plot(T, integrator, 'b')
        axes[data_seg.shape[1], 0].set_ylim([0, 2])
        axes[data_seg.shape[1], 0].set_xlim([0, xmax])
        axes[data_seg.shape[1], 0].plot([0, T[-1]], [alarm_threshold, alarm_threshold], '--', color='orange')
        try: 
            axes[data_seg.shape[1], 1].plot(t, data.loc[:, 'Status'], 'g')
            axes[data_seg.shape[1], 1].plot(T, alrm, 'r--')
            axes[data_seg.shape[1], 1].set_xlim([0, xmax])
        except:
            print('No breaker status has been detected')


        plt.tight_layout()

        save_path = os.path.join(savefolder, figname + '.png')
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()

    # prediction and evaluation
    Fault_predict = np.array(Fault_predict)
    Accur, False_alarm, F1 = calculate_evaluation(Fault_predict, Label)
    print('Wavelet Accurary = {}, False_alarm = {}, F1 = {}'.format(Accur, False_alarm, F1))
    print("threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauUp:{}, tauDown:{}, beta1:{}, beta2:{}".format(thr,
                                                                                                             alarm_threshold,
                                                                                                             alarm_dur,
                                                                                                             tauUp,
                                                                                                             tauDown,
                                                                                                             beta1,
                                                                                                             beta2))
    result = {}
    result['Delaytime'] = Delaytime
    result['Alarm_duration'] = Alarm_duration
    result['Accuracy'] = Accur
    result['False_alarm'] = False_alarm
    result['F1'] = F1
    result['parameters'] = {'threshold': thr, 'alarm_trheshold': alarm_threshold, 'alarm_dur': alarm_dur,
                            'tauUp': tauUp, 'tauDown': tauDown, 'window_size': window_size, 'beta1': beta1,
                            'beta2': beta2}

    # hist of Alarm_duration
    num_bins = 10
    plt.figure()

    Alarm_duration = [t if t is not None else 0 for t in Alarm_duration]
    Alarm_duration = np.array(Alarm_duration)
    plt.hist(Alarm_duration, num_bins, density=1, color='green', alpha=0.7)
    plt.xlabel('First Over threshold duration (s)')
    plt.title('First over threshold duration \n\n',
              fontweight="bold")
    plt.show()

    meanAlarmDur = np.nanmean(Alarm_duration)
    medianAlarmDur = np.nanmedian(Alarm_duration)
    result['meanAlarmDur'] = meanAlarmDur
    result['medianAlarmDur'] = medianAlarmDur
        

    # hist of Delaytime
    num_bins = 10
    Delaytime = [t[0] if isinstance(t, np.ndarray) else t for t in Delaytime]
    plt.figure()
    plt.hist(Delaytime, num_bins, density=1, color='blue', alpha=0.7)
    plt.xlabel('Delaytime (s)')
    plt.title('Delaytime \n\n',
              fontweight="bold")
    # plt.xscale('log')
    plt.show()  

    Delaytime = np.array(Delaytime)
    meanDl = np.nanmean(Delaytime)
    medianDl = np.nanmedian(Delaytime)
    result['meanDelaytime'] = meanDl
    result['medianDelaytime'] = medianDl

    save_object(result, 'result_wavelet_Franksville.pickle')
    print('mean alarm duration = {}, median alarm duration = {}, mean delaytime ={}, median delaytime={}'.format(
        meanAlarmDur, medianAlarmDur, meanDl, medianDl))
    return (thr, alarm_threshold, alarm_dur, tauUp, tauDown, NoiseLevel, Accur, F1, False_alarm, meanAlarmDur, meanDl,
            medianAlarmDur, medianDl)

        
def main():
    f0 = 60

    root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Franksville Test\\20220804\\8-04-22 csv backup\\Vegetation-Tree"

    dir_list = os.listdir(root)

    # chanel that being estimated
    chl_monitored = ['Test_I', 'Test/Fault Current', 'Leakage Current']
    fs_target = 128 * 60

    for file in dir_list:
        file_path = os.path.join(root, file)
        # file_path ='C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Transformer_Inrush10kHz\\hd5\\Transformer_Inrush.h5'

        # read voltage and current data for chanel_name from given file_path
        data,_,_ = data_acquisition(file_path, voltage_channels=['Test_V', 'Voltage'],
                                current_channels=['Test_I', 'ground_I', 'Test/Fault Current', 'Leakage Current'],
                                kwds_brkr=['cable', 'brk', 'solenoid'])
        fs = calculate_sample_freq(data[TIME_COLUMN])

        # monitored chanel
        chl_monitored_ = [chl.replace(' ', '') for chl in chl_monitored if chl.replace(' ', '') in data.columns]

        # delete the first 2 s of data
        t0 = 0.5
        t = data.loc[:, 'Time']
        d = data.loc[(t > t0) & (t < 25), chl_monitored_]
        t = t[(t > t0) & (t < 25)].reset_index().drop("index", axis='columns').values
        t = t - t[0]

        # filter data
        
        for j in range(d.shape[1]):
            dl = d.iloc[:, j]
            # bandpass butterworth filter 
            dl_filtered = filter_butterworth(dl, [np.min([5200, int(fs / 2) - 1])], fs, order=2, filter_type='lowpass')

            # resample data
            dl_resample = _resample(dl_filtered, fs, fs_target)

            # # denoise using dwt
            # dl_denoise = denoise_signal(dl_resample,level=2)

            # #medfilter to smooth datat
            # ksize = int(fs_target/60/10)
            # if ksize%2 ==0:
            #     ksize+=1
            # dl_medfil = signal.medfilt(dl_denoise,kernel_size = ksize )

            dl = dl_resample

            if j == 0:
                data_new = dl.reshape(-1, 1)
            else:
                data_new = np.concatenate((data_new, dl.reshape(-1, 1)), axis=1)

        # time

        t_ = np.arange(len(dl)) / fs_target

        # t_ = pd.DataFrame(t_,columns = ["Time"])

        # merge time with data_new
        data_new = pd.DataFrame(data_new, columns=d.columns)
        # data_new = pd.concat((t_,data_new), axis=1)

        # set up parameters for DFT 
        Npoints = round(fs_target / f0)
        window_size = Npoints * 2
        nshift = round(Npoints / 4)

        # extract filename 
        figname = os.path.basename(file_path).split('.')[0]

        data_seg = [data_new.iloc[i:i + window_size, :].to_numpy() for i in
                    np.arange(0, data_new.shape[0] - window_size, nshift)]
        T = [t_[i:i + window_size].mean() for i in np.arange(0, data_new.shape[0] - window_size, nshift)]

        deltaT = (T[5] - T[0]) / 5

        beta1 = 1 - deltaT / 5
        beta2 = 1 - deltaT / 2
        print(beta1, beta2)
        thr = 2

        intg = [0]
        tauU = 1
        tauD = 1
        integrator = []

        NoiseLevel = 1e-2
        rms_list = []
        rmsmean_list = []
        rmsvar_list = [] 

        buffer_length = int(0.5 / deltaT)
        rms_buffer = np.ones((buffer_length, d.shape[1])) * np.nan
        rms_mean = np.array([0.0] * (d.shape[1]))
        rms_var = np.array([0.0] * (d.shape[1]))
        rms_mean_corrected = np.array([0.0] * (d.shape[1]))
        rms_var_corrected = np.array([0.0] * (d.shape[1]))
        freeze = 0
        ksize = int(fs_target / 60 / 10)
        perc = 0.05
        for k, dt in enumerate(data_seg):
            
            rms_array = np.array([])
            n = 0
            
            for i in np.arange(dt.shape[1]):
                dt_i = dt[:, i]
                # plt.figure()
                # plt.plot(dt_i,'b')
                # denoise using dwt
                # dt_i= denoise_signal(dt_i,level=2)
                # medfilter to smooth data

                if ksize % 2 == 0:
                    ksize += 1
                dt_i = signal.medfilt(dt_i, kernel_size=ksize)

                rms_dt = rms(dt_i)

                rms_array = np.append(rms_array, rms_dt)
            
                n, rms_buffer[:, i], rms_mean[i], rms_var[i], rms_mean_corrected[i], rms_var_corrected[
                    i] = decisionLogic(rms_dt, rms_mean[i], rms_var[i], rms_mean_corrected[i], rms_var_corrected[i],
                                       thr, k, rms_buffer[:, i], beta1, beta2, perc, freeze)
            
            intg = lowPassFilter(tauU, tauD, [n], 1 / fs_target * nshift, intg[0])

            if (intg[0] > 0.5 and k > buffer_length):

                freeze = 1
            else:
                freeze = 0

            integrator.append(intg[0])  
            rms_list.append(rms_array.copy())

            rmsmean_list.append(rms_mean_corrected.copy())
            rmsvar_list.append(rms_var_corrected.copy())

        rms_array = np.array(rms_list)
        rmsmean_array = np.array(rmsmean_list)
        rmsvar_array = np.array(rmsvar_list)
        T = np.array(T)

        fig, axes = plt.subplots(d.shape[1] + 1, 2)
        fig.suptitle(figname)

        for i in range(rms_array.shape[1]):
            axes[i, 0].plot(T, rms_array[:, i], color='k')
            axes[i, 0].plot(T, rmsmean_array[:, i], color='y')
            axes[i, 0].plot(T, rmsmean_array[:, i] + rmsvar_array[:, i] * thr, '--', color='g')
            axes[i, 0].plot(T, rmsmean_array[:, i] - rmsvar_array[:, i] * thr, '--', color='g')

            ind = np.argwhere((np.abs(rmsmean_array[:, i] - rms_array[:, i]) > rmsvar_array[:, i] * thr) & (
                        np.abs(rmsmean_array[:, i] - rms_array[:, i]) > rmsmean_array[:, i] * perc))
            
            axes[i, 0].plot(T[ind], rms_array[ind, i], '.', color='r')
            axes[i, 1].plot(T, rmsvar_array[:, i], color='b')
            axes[i, 1].plot(T, np.abs(rmsmean_array[:, i] - rms_array[:, i]), 'r')
            axes[i, 1].set_ylim([0, 0.1])
        axes[d.shape[1], 0].plot(T, integrator, 'b')
        axes[d.shape[1], 0].set_ylim([0, 2])

        axes[d.shape[1], 0].plot([0, T[-1]], [0.5, 0.5], '--', color='orange')
        try: 
            axes[d.shape[1], 1].plot(data.iloc[:, 0], data.loc[:, 'Status'], 'g')
        except:
            print('No breaker status has been detected')

        plt.show()

# #%% run time corr with multi parameters
# os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 


# # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 

# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# print(connect_str)
# # Create the BlobServiceClient object which will be used to create a container client
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# # Create a unique name for the container

# container_name = "ieee34"
# local_path ="."

# # Create the container
# container_client = blob_service_client.get_container_client( container_name)
# # walk_container(container_client,connect_str, container_name,local_path)
# blob_list = get_blob_list(container_client)

# temp_file = os.path.join('.','temp_file.h5')
# channel_name = 'Ipt4'

# Label = np.array([0]*21 + [1]* 68)
# t0=2
# tend =20
# Record_array =[]
# thr = np.arange(0.1,0.6,0.1)
# alarm_threshold = np.arange(0.1,1,0.1)
# alarm_dur = np.arange(0.1,2,0.1)
# tauU_outBund = np.arange(0.1,1,0.1)
# for tU in tauU_outBund:
#     for alm in alarm_threshold:
#         for alm_dur in alarm_dur:
#             for tr in thr:
#                 rd = run_timecorr(blob_list, connect_str,container_name, t0,tend,Label, tr,tU,tU,alm, alm_dur)
#                 print(rd)
#                 Record_array.append(rd)
#                 save_object(Record_array, 'timecorr_parameterExplore.pickle')


# root = "Z:\\NormalLoads"
# dir_list = os.listdir(root)
# dir_list = [dir for dir in dir_list if os.path.isdir(os.path.join(root,dir))]

# for dir in dir_list:
#     foldername = os.path.join(root, dir)
    
#     for file in os.listdir(foldername):
#         if file.endswith('.h5'):
#             file_path = os.path.join(foldername,file)
            
#             data,_,_ = data_acquisition (file_path, nodes=['pt4'])
#             T = data.iloc[:,0]
#             data = data.loc[T>2,:]
#             fs = 5/(T[5]-T[0])
#             f0 = 50
#             stat = data.iloc[:,-1].to_numpy()
#             indx = np.nonzero(np.diff(stat))
#             if indx[0]:
#                 tst = T[indx[0]].values
#                 print(tst)
#             else:
#                 tst = None
            
#             sig = data.iloc[:,4:7]
            
#             wsize = int(fs/f0*4)
#             wstep = int(fs/f0)
#             wavletname = 'db4'
#             decompLevel = 1

    
# root = "Z:\\NormalLoads"
# dir_list = os.listdir(root)
# dir_list = [dir for dir in dir_list if os.path.isdir(os.path.join(root,dir))]

# for dir in dir_list:
#     foldername = os.path.join(root, dir)
    
#     for file in os.listdir(foldername):
#         if file.endswith('.h5'):
#             file_path = os.path.join(foldername,file)
            
#             data,_,_ = data_acquisition (file_path, nodes=['pt4'])
#             T = data.iloc[:,0]
#             data = data.loc[T>2,:]
#             fs = 5/(T[5]-T[0])
#             f0 = 50
#             stat = data.iloc[:,-1].to_numpy()
#             indx = np.nonzero(np.diff(stat))
#             if len(indx[0]):
#                 tst = T[indx[0]].values
#                 print(tst)
#             else:
#                 tst = None
            
#             sig = data.iloc[:,4:7]
            
#             wsize = int(fs/f0*4)
#             wstep = int(fs/f0)
#             wavletname = 'db4'
#             decompLevel = 1


#             filename =file.split('.')[0]
#             wavedec_std(sig,wsize,wstep,fs, wavletname,decompLevel, level=4, plot=True,beta1=0.9,beta2=0.99, quantile_Level=0.999, thr =4,tauU_outBound =0.4, tauD_outBound = 0.4,figname =filename + ', ' +data.columns[4:7],alarm_threshold=0.4,alarm_dur =0.8,tst =tst, savePlot= True, saveFolder ='.\\plot')


# file_path = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\VT097_WithPV_Scaled\\HDFfolder\\VT097_Scaled_25_IEEE34BUS.h5"
# data,_,_ = data_acquisition(file_path, nodes=['pt4'])
# T = data.iloc[:,0]
# fs = 5/(data.iloc[5,0] - data.iloc[0,0])
# f0 = 50


# dt = data.iloc[:,1:7]
# column_list = dt.columns
# wsize_statistic = int(fs/f0/2)
# noverlap_statistic =0
# thr = 2
# plot_skewness_Kurtosis(dt,fs, column_list, wsize_statistic,noverlap_statistic,thr)

# data = data.loc[T>2,:]

# wsize = int(fs/f0) * 2
# freqComp ={'3rd': [3*f0], '5th': [5*f0], 'odd': np.arange(3,16,2)*f0, 'even': np.arange(2,16,2)*f0}
# plot_EnergyRatio(data.iloc[:,1:7],fs,f0,wsize,freqComp )

# sig = data.iloc[:,4]
# shortWsize = int(fs/f0/4)
# bigWsize = int(fs/f0*2)
# noverlap = bigWsize//2
# beta1 =0.99
# beta2 =0.99
# thr =2
# step = shortWsize
# plotABRatio(sig,"VT097_Scaled_25_IEEE34BUS",shortWsize,bigWsize,noverlap,fs,beta1,beta2, thr, step = step, plot=True )   

# column_list = data.columns
# fmin = f0*8
# fmax =f0*15
# plotEnergyCorrelation(data.iloc[:,4:7],column_list[4:7], int(fs/f0)*3,int(fs/f0),int(fs/f0/4),fs,fmin,fmax)

# dt = data.iloc[:,1:7]
# column_list = dt.columns
# wsize_statistic = int(fs/f0/2)
# noverlap_statistic =0
# thr = 2
# plot_skewness_Kurtosis(dt,fs, column_list, wsize_statistic,noverlap_statistic,thr)


# %% wavelet analysis
# os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

# # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# print(connect_str)
# # Create the BlobServiceClient object which will be used to create a container client
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# # Create a unique name for the container

# container_name = "ieee34"
# local_path ="."

# # Create the container
# container_client = blob_service_client.get_container_client( container_name)
# # walk_container(container_client,connect_str, container_name,local_path)
# blob_list = get_blob_list(container_client)

# temp_file = os.path.join('.','temp_file.h5')
# delayT =[]
# Label = np.array([0]*21 + [1]* 68)
# alarm_dur =0.8
# thr =4
# alarm_threshold=0.4
# tauU_outBound =0.4
# for blob_name in blob_list:
#     blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name, blob_name=blob_name)
#     with open(temp_file,"wb") as my_blob:
#         blob_data =blob_client.download_blob()
#         blob_data.readinto(my_blob)


#     if blob_name.endswith('.h5')and (not blob_name.__contains__('5p288')):
#         data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
#         # os.remove(temp_file)

#         T = data.iloc[:,0]
#         data = data.loc[T>2,:]
#         fs = 5/(T[5]-T[0])
#         if fs > 10000:
#             data = data.iloc[::10,:]
#             T = data.iloc[:,0].values-data.iloc[0,0]
#             fs = 5/(T[5]-T[0])
#             print(fs)
#         else:
#             T = data.iloc[:,0].values-data.iloc[0,0]
#         f0 = 50
#         stat = data.iloc[:,-1].to_numpy()
#         indx = np.nonzero(np.diff(stat))
#         if len(indx[0]):
#             tst = T[indx[0]]
#             print(tst)
#         else:
#             tst = None
        
#         sig = data.iloc[:,4:7]
        
#         wsize = int(fs/f0*2)
#         wstep = int(fs/f0)
#         wavletname = 'db4'
#         decompLevel = 1
#         file = os.path.basename(blob_name)
#         filename =file.split('.')[0]
#         deltaT =wstep/fs
#         beta1 = 1-deltaT/0.2
#         beta2 = 1-deltaT/2
#         print('beta1={}, beta2={}'.format(beta1,beta2))
        
#         delay_time = wavedec_std(sig,wsize,wstep,fs, wavletname,decompLevel, level=4, plot=True,savePlot = True, saveFolder = '.\\plot_dwt', beta1=beta1,beta2=beta2, quantile_Level=0.999, thr =thr,tauU_outBound =tauU_outBound , tauD_outBound = tauU_outBound ,figname =filename + ', ' +data.columns[4:7],alarm_threshold=alarm_threshold,alarm_dur =alarm_dur,tst =tst)
      
#         delayT.append(delay_time)


# delayT = np.array(delayT)
# Label = Label[np.arange(len(delayT))]

# delayT =[k[0] if isinstance(k, np.ndarray) else k for k in delayT]
# y = np.array(delayT)
# positive_indx = (y<1)*1
# Accur, False_alarm, F1 = calculate_evaluation(positive_indx, Label)
# print("threshold: {}, alarm_threhold:{}, alarm_dur: {}, tauU:{}".format(thr, alarm_threshold, alarm_dur, tauU_outBound))
# print('Time correlation Accuracy: {}, F1: {}, False_alarm: {}'.format( Accur, F1, False_alarm))
# mean_respond = np.nanmean(y[y<1])+alarm_dur
# print('average detecttime is {}'.format(mean_respond))


# %% ABratio
# os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

# # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# print(connect_str)
# # Create the BlobServiceClient object which will be used to create a container client
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# # Create a unique name for the container

# container_name = "ieee34"
# local_path ="."

# # Create the container
# container_client = blob_service_client.get_container_client( container_name)
# # walk_container(container_client,connect_str, container_name,local_path)
# blob_list = get_blob_list(container_client)

# temp_file = os.path.join('.','temp_file.h5')
# delayT =[]
# Label = np.array([0]*23 + [1]* 68)
# for blob_name in blob_list:
#     blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name, blob_name=blob_name)
#     with open(temp_file,"wb") as my_blob:
#         blob_data =blob_client.download_blob()
#         blob_data.readinto(my_blob)


#     if blob_name.endswith('.h5'):
#         data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
#         os.remove(temp_file)

#         T = data.iloc[:,0]
#         data = data.loc[T>2,:]
#         T = data.iloc[:,0].values-data.iloc[0,0]
       
#         fs = 5/(T[5]-T[0])
#         f0 = 50
#         stat = data.iloc[:,-1].to_numpy()
#         indx = np.nonzero(np.diff(stat))
#         if indx[0]:
#             tst = T[indx[0]]
#             print(tst)
#         else:
#             tst = None
        
#         sig = data.iloc[:,4:7]

#         bigWsize = int(fs/f0*2 ) 
#         shortWsize = int(fs/f0/4 ) 
#         noverlap = bigWsize//2
#         deltaT = (bigWsize - noverlap)/fs
#         beta1 = 1-deltaT/0.2
#         beta2 = 1-deltaT/2
#         print('beta1 = {}, beta2 = {}'.format(beta1,beta2))
#         tauU_outBound = 1-deltaT/0.5
#         tauD_outBound = 1-deltaT/0.4
#         file = os.path.basename(blob_name)
#         filename =file.split('.')[0]
#         thr = 2
#         delayT_3phase =[]

#         for i in range(len(sig.columns)):
#             figname = filename + ', ' +sig.columns[i]

#             delay_time = plotABRatio(sig.iloc[:,i].to_numpy(), shortWsize, bigWsize, noverlap,fs, beta1,beta2,thr,quantile_Level=0.95,step=shortWsize,rmvf=2*f0, plot=True, tauU_outBound =tauU_outBound , tauD_outBound=tauD_outBound , figname = figname,alarm_threshold=0.5,alarm_dur =1.5,tst=tst, savePlot = True, saveFolder = '.\\plot_abratio')
#             delayT_3phase.append(delay_time)
        
#         delayT.append(np.min(np.array(delayT_3phase).reshape(len(sig.columns),-1), axis = 0))
# delayT = np.array(delayT)
# pk_abratio = delayT[:,0]
# pk_abratio =[k[0] if isinstance(k, np.ndarray) else k for k in pk_abratio]
# std_abratio = delayT[:,1]
# std_abratio =[k[0] if isinstance(k, np.ndarray) else k for k in std_abratio]

# y = np.array(pk_abratio)
# diff = 1-np.isnan(y)*1 - Label[:len(y)]

# TP = np.sum(y<1)
# FN = 76 - TP
# FP = np.sum(diff ==1)
# TN = 24-FP
# F1 = 2*TP/(2*TP+FP+FN)
# Accur = (TP+TN)/100
# print('Pk_abratio Accuracy: {}, F1: {}'.format(Accur, F1))


# y = np.array(std_abratio)
# diff = 1-np.isnan(y)*1 - Label[:len(y)]

# TP = np.sum(y<1)
# FN = 76 - TP
# FP = np.sum(diff ==1)
# TN = 24-FP
# F1 = 2*TP/(2*TP+FP+FN)
# Accur = (TP+TN)/100
# print('std_abratio Accuracy: {}, F1: {}'.format(Accur, F1))


# %% skewness, kurtosis, crf
# os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

# # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# print(connect_str)
# # Create the BlobServiceClient object which will be used to create a container client
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# # Create a unique name for the container

# container_name = "ieee34"
# local_path ="."

# # Create the container
# container_client = blob_service_client.get_container_client( container_name)
# # walk_container(container_client,connect_str, container_name,local_path)
# blob_list = get_blob_list(container_client)

# temp_file = os.path.join('.','temp_file.h5')
# delayT_sk =[]
# delayT_kurtosis = []
# delayT_crf =[]

# Label = np.array([0]*23 + [1]* 68)
# for blob_name in blob_list:
#     blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name, blob_name=blob_name)
#     with open(temp_file,"wb") as my_blob:
#         blob_data =blob_client.download_blob()
#         blob_data.readinto(my_blob)


#     if blob_name.endswith('.h5'):
#         data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
#         os.remove(temp_file)

#         T = data.iloc[:,0]
#         data = data.loc[T>2,:]
#         T = data.iloc[:,0].values-data.iloc[0,0]
       
#         fs = 5/(T[5]-T[0])
#         f0 = 50
#         stat = data.iloc[:,-1].to_numpy()
#         indx = np.nonzero(np.diff(stat))
#         if indx[0]:
#             tst = T[indx[0]]
#             print(tst)
#         else:
#             tst = None
        
#         sig = data.iloc[:,4:7]

#         wsize_statistic = int(fs/f0/2)
#         noverlap_statistic =0
#         thr = 2

#         deltaT = ( wsize_statistic - noverlap_statistic)/fs
#         beta1 = 1-deltaT/0.2
#         beta2 = 1-deltaT/0.8
#         print('beta1 = {}, beta2 = {}'.format(beta1,beta2))
#         tauU_outBound = 1-deltaT/0.5
#         tauD_outBound = 1-deltaT/0.4
#         file = os.path.basename(blob_name)
#         filename =file.split('.')[0]
        
        
#         column_list = sig.columns
#         figname = filename 

#         delay_time_sk, delay_time_crf,delay_time_kurtosis = plot_skewness_Kurtosis(sig,fs, column_list, wsize_statistic,noverlap_statistic,thr,quantile_Level = 0.999,saveFolder='.\\plot_skewness_kurtosis_crf',figname=filename , beta1=0.99,beta2=0.999, savePlot=True,tauU_outBound =0.4, tauD_outBound = 0.2,alarm_threshold=0.5,alarm_dur =1,tst=tst)
#         delayT_sk.append(delay_time_sk)
#         delayT_crf.append(delay_time_crf)
#         delayT_kurtosis.append(delay_time_kurtosis)


# delayT_sk = np.array(delayT_sk)
# delayT_crf = np.array(delayT_crf)
# delayT_kurtosis = np.array(delayT_kurtosis)

# delayT_sk =[k[0] if isinstance(k, np.ndarray) else k for k in delayT_sk]
# delayT_crf =[k[0] if isinstance(k, np.ndarray) else k for k in delayT_crf]
# delayT_kurtosis =[k[0] if isinstance(k, np.ndarray) else k for k in delayT_kurtosis]

# feature_list = [delayT_sk,delayT_crf,delayT_kurtosis]
# feature_name =['skewness', 'CRF', 'Kurtosis']
# for i, y in enumerate(feature_list):
#     y = np.array(y)
#     diff = 1-np.isnan(y)*1 - Label[:len(y)]

#     TP = np.sum(y<1)
#     FN = 76 - TP
#     FP = np.sum(diff ==1)
#     TN = 24-FP
#     F1 = 2*TP/(2*TP+FP+FN)
#     Accur = (TP+TN)/100
#     print('{} Accuracy: {}, F1: {}'.format(feature_name[i], Accur, F1))


# %% plot time correlation

# os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

# # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# print(connect_str)
# # Create the BlobServiceClient object which will be used to create a container client
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# # Create a unique name for the container

# container_name = "ieee34"
# local_path ="."

# # Create the container
# container_client = blob_service_client.get_container_client( container_name)
# # walk_container(container_client,connect_str, container_name,local_path)
# blob_list = get_blob_list(container_client)

# temp_file = os.path.join('.','temp_file.h5')
# delayT =[]
# Label = np.array([0]*21 + [1]* 68)
# for blob_name in blob_list:
#     blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name, blob_name=blob_name)
#     with open(temp_file,"wb") as my_blob:
#         blob_data =blob_client.download_blob()
#         blob_data.readinto(my_blob)


#     if blob_name.endswith('.h5'):
#         data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
#         os.remove(temp_file)

#         T = data.iloc[:,0]
#         data = data.loc[T>2,:]
#         fs = 5/(T[5]-T[0])
#         if fs > 10000:
#             data = data.iloc[::10,:]
#             T = data.iloc[:,0].values-data.iloc[0,0]
#             fs = 5/(T[5]-T[0])
#             print(fs)
#         else:
#             T = data.iloc[:,0].values-data.iloc[0,0]
#         f0 = 50
#         stat = data.iloc[:,-1].to_numpy()
#         indx = np.nonzero(np.diff(stat))
#         if indx[0]:
#             tst = T[indx[0]]
#             print(tst)
#         else:
#             tst = None
        
#         sig = data.iloc[:,4:7]
        
#         wsize = int(fs/f0*2)
#         wstep = int(fs/f0)
        
#         file = os.path.basename(blob_name)
#         filename =file.split('.')[0]
#         deltaT =wstep/fs

#         column_list = sig.columns
#         thr = 0.3
#         rmvf = 6 *f0
#         delay_time = plotTimeCorrelation(sig, column_list, wsize, fs, thr, rmvf=rmvf,saveFolder='.\\plot_timecorr',figname=filename,savePlot=True,tauU_outBound =0.1, tauD_outBound = 0.1,alarm_threshold=0.5, alarm_dur =0.1,tst = tst,plot=True)
        
#         delayT.append(delay_time)
   

# delayT = np.array(delayT)
# Label = Label[np.arange(len(delayT))]
# delayT =[k[0] if isinstance(k, np.ndarray) else k for k in delayT]
# y = np.array(delayT)
# positive_indx = (y<1)*1
# Accur, False_alarm, F1 = calculate_evaluation(positive_indx, Label)

# print('Time correlation Accuracy: {}, F1: {}, False_alarm: {}'.format( Accur, F1, False_alarm))
# mean_respond = np.nanmean(y[y<1])+alarm_dur
# print('average detecttime is {}'.format(mean_respond))
# run_LSTM()        


# #%% plot multrual information
# os.environ["HTTP_PROXY"] = "http://zproxy.etn.com:8080" 

# # os.environ["HTTPS_PROXY"] = "http://zproxy.etn.com:8080" 
# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
# print(connect_str)
# # Create the BlobServiceClient object which will be used to create a container client
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)
# # Create a unique name for the container

# container_name = "ieee34"
# local_path ="."

# # Create the container
# container_client = blob_service_client.get_container_client( container_name)
# # walk_container(container_client,connect_str, container_name,local_path)
# blob_list = get_blob_list(container_client)

# temp_file = os.path.join('.','temp_file.h5')

# Label = np.array([0]*21 + [1]* 68)
# for blob_name in blob_list:
#     blob_client = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name, blob_name=blob_name)
#     with open(temp_file,"wb") as my_blob:
#         blob_data =blob_client.download_blob()
#         blob_data.readinto(my_blob)


#         if blob_name.endswith('.h5'):
#             data,_,_ = data_acquisition(temp_file, nodes=['pt4'])
#             # os.remove(temp_file)

#             T = data.iloc[:,0]
#             data = data.loc[T>2,:]
#             fs = 5/(T[5]-T[0])
#             if fs > 10000:
#                 data = data.iloc[::10,:]
#                 T = data.iloc[:,0].values-data.iloc[0,0]
#                 fs = 5/(T[5]-T[0])
#                 print(fs)
#             else:
#                 T = data.iloc[:,0].values-data.iloc[0,0]
#             f0 = 50
#             stat = data.iloc[:,-1].to_numpy()
#             indx = np.nonzero(np.diff(stat))
#             if len(indx[0]):
#                 tst = T[indx[0]]
#                 print(tst)
#             else:
#                 tst = None


            # file = os.path.basename(blob_name)
            # filename =file.split('.')[0]
            # wsize = int(fs/f0*4)
            # noverlap = wsize//2
            # nbins =100
            # rmvf = 0*f0
            # tauU_outBound = 0.2
            # tauD_outBound = 0.2
            # thr = 0.02
            # dt = data.iloc[:,1:7]

            # I0= dt.iloc[:,3:].mean(axis = 1)
            # V0 = dt.iloc[:,:3].mean(axis = 1)
            # I0= I0.to_frame(name = 'I0')
            # I0.reset_index(drop=True, inplace=True)

            # V0 = V0.to_frame(name = 'V0')
            # V0.reset_index(drop=True, inplace=True)
            # d_I = dt.iloc[:,3:].copy() 
            # d_I.reset_index(drop=True, inplace=True)
            # I = pd.concat([d_I,I0], axis = 1)
            # d_V = dt.iloc[:,:3].copy()
            # d_V.reset_index(drop=True, inplace=True)
            # V = pd.concat([d_V,V0],axis = 1)
            # column_list = I.columns
#             plot_mulInformation(V,I,T, fs,column_list, nbins, rmvf,thr, filename, tst,wsize,noverlap, tauU_outBound, tauD_outBound,savePlot=True)


# %% investigate how amplitude change with time
# root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Franksville Test\\20220729\\7-29-22 csv backup\\Grass"
# dir_list = os.listdir(root)
# print(dir_list)
# reaction_time =[]
# for i in range(len(dir_list)):
#     file_path= os.path.join(root,dir_list[i])
#     # chanel_list =[ 'Voltage', 'Test/Fault Current','Leakage Current']
#     data,_,_ = data_acquisition(file_path, voltage_channels=['Test_V', 'Voltage'],
#                             current_channels=['Test_I','ground_I','Test/Fault Current','Leakage Current'],
#                             kwds_brkr= ['solenoid'])
#     print(data.columns)
#     T = data.iloc[:,0].to_numpy()
#     fs = 5/(T[5]-T[0])
#     f0=60
#     wsize = int(fs/f0)
#     step = int(fs/f0/2)
#     chl = 'Test_I'
#     filename = dir_list[i].split('.')[0]
#     T_stop, maxAmp=plot_amplitude(data.loc[:,chl],data.iloc[:,0],chl,1,wsize,filename,step = step,ymax=140)
    
#     reaction_time.append({filename:(T_stop,maxAmp)})
# print(reaction_time)
# plt.legend()
# plt.show()

# reaction_time =[]
# for i in range(len(dir_list)):
#     file_path= os.path.join(root,dir_list[i])
#     data,_,_ = data_acquisition(file_path, voltage_channels=['Test_V', 'Voltage'],
#                             current_channels=['Test_I','ground_I', 'Test/Fault Current','Leakage Current'],
#                             kwds_brkr=['solenoid'])
#     T = data.iloc[:,0].to_numpy()
#     fs = 5/(T[5]-T[0])
#     f0=60
#     wsize = int(fs/f0)
#     step = int(fs/f0/2)
#     chl = 'ground_I'
#     filename = dir_list[i].split('.')[0]
#     T_stop, maxAmp=plot_amplitude(data.loc[:,chl],data.iloc[:,0],chl,1,wsize,filename,step = step, ymax = 2)
    
#     reaction_time.append({filename:(T_stop,maxAmp)})
# print(reaction_time)
# plt.legend()
# plt.show()

# for i in range(len(dir_list)):
#     file_path= os.path.join(root,dir_list[i])
#     data,_,_ = data_acquisition(file_path, voltage_channels=['Voltage'],
#                             current_channels= ['Test/Fault Current','Leakage Current'], kwds_brkr=['solenoid'])
#     T = data.iloc[:,0].to_numpy()
#     plt.plot(T,data.loc[:,'Status'])
# plt.legend()
# plt.show()
# f0=60
# root = "C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Franksville Test\\20220804\\8-04-22 csv backup\\Vegetation-Tree"


# dir_list = os.listdir(root)

# #chanel that being estimated
# chl_monitored = ['Test_I','Test/Fault Current','Leakage Current']
# fs_target = 128*60

# for file in dir_list:
#     file_path = os.path.join(root,file)
#     # file_path ='C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Transformer_Inrush10kHz\\hd5\\Transformer_Inrush.h5'


#     # read voltage and current data for chanel_name from given file_path
#     data,_,_= data_acquisition(file_path, voltage_channels=['Test_V','Voltage'],
#                                current_channels=['Test_I','ground_I','Test/Fault Current','Leakage Current'],
#                                kwds_brkr=['cable'])
#     fs = calculate_sample_freq(data[TIME_COLUMN])

#     # monitored chanel
#     chl_monitored_=[chl for chl in chl_monitored if chl in data.columns]  


#     # delete the first 2 s of data
#     t0=.5
#     t= data.loc[:,'Time']
#     d = data.loc[(t>t0)&(t<25),chl_monitored_]
#     t = t[(t>t0)&(t<25)].reset_index().drop("index",axis='columns').values
#     t = t - t[0]

#     # filter data
    

#     for j in range(d.shape[1]):
#         dl = d.iloc[:,j]
#         # bandpass butterworth filter 
#         dl_filtered = filter_butterworth(dl,[np.min([5200,int(fs/2)-1])],fs,order=2,filter_type='lowpass')

#         # resample data
#         dl_resample = _resample(dl_filtered,fs,fs_target)

#         # # denoise using dwt
#         # dl_denoise = denoise_signal(dl_resample,level=2)

#         # #medfilter to smooth datat
#         # ksize = int(fs_target/60/10)
#         # if ksize%2 ==0:
#         #     ksize+=1
#         # dl_medfil = signal.medfilt(dl_denoise,kernel_size = ksize )

#         dl = dl_resample

#         if j ==0:
#             data_new = dl.reshape(-1,1)
#         else:
#             data_new = np.concatenate((data_new,dl.reshape(-1,1)),axis=1)


#     # time
#     t_ = np.arange(len(dl))/fs_target
#     # t_ = pd.DataFrame(t_,columns = ["Time"])

#     #merge time with data_new
#     data_new = pd.DataFrame(data_new, columns = d.columns)
#     # data_new = pd.concat((t_,data_new), axis=1)


#     # set up parameters for DFT 
#     Npoints = int(fs_target/f0)
#     window_size = Npoints*2
#     nshift = int(Npoints//4)

#     # extract filename 
#     figname = os.path.basename(file_path).split('.')[0]


#     data_seg =[data_new.iloc[i:i+window_size,:].to_numpy() for i in np.arange(0,data_new.shape[0]-window_size, nshift)]
#     T =[t_[i:i+window_size].mean() for i in np.arange(0,data_new.shape[0]-window_size, nshift)]
#     # if 'Status' in data.columns:
#     #     St = np.array([data.loc[i:i+window_size, 'Status'].max() for i in np.arange(0,d.shape[0]-window_size, nshift)])
#     # else:
#     #     St = np.zeros((len(T),))
#     deltaT = (T[5]-T[0])/5

#     beta1 = 1-deltaT/2
#     beta2 = 1-deltaT/2
#     print(beta1,beta2)
#     thr =2

#     intg = [0]
#     tauU=0.2
#     tauD = 0.3
#     integrator=[]

#     NoiseLevel =1e-2
#     rms_list =[]
#     rmsmean_list=[]
#     rmsvar_list = [] 
#     deltT = T[1]-T[0]
#     buffer_length = int(0.5/deltT)
#     rms_buffer =np.ones((buffer_length,d.shape[1]))*np.nan
#     rms_mean = np.array([0.0]*(d.shape[1]))
#     rms_var = np.array([0.0]*(d.shape[1]))
#     rms_mean_corrected = np.array([0.0]*(d.shape[1]))
#     rms_var_corrected = np.array([0.0]*(d.shape[1]))
#     freeze = 0
#     ksize = int(fs_target/60/10)
#     for k, dt in enumerate(data_seg):
        
#         # if St[k] ==0:
#         rms_array = np.array([])
#         n=0
        
#         for i in np.arange(dt.shape[1]):
#             dt_i = dt[:,i]
#             # plt.figure()
#             # plt.plot(dt_i,'b')
#             # denoise using dwt
#             dt_i= denoise_signal(dt_i,level=2)

#             #medfilter to smooth datat

#             if ksize%2 ==0:
#                 ksize+=1
#             dt_i = signal.medfilt(dt_i,kernel_size = ksize )
#             # plt.plot(dt_i,'r-')
            
#             rms_dt = rms(dt_i)

#             rms_array = np.append(rms_array,rms_dt)
#             rms_mean[i],rms_mean_corrected[i] = adaptiveMean(rms_dt,beta1,k+1,rms_mean[i],biasCorrect =True)
            
#             if (abs(rms_dt - rms_mean_corrected[i])<rms_var_corrected[i] or k<buffer_length) and freeze ==0:
#                 bf = rms_buffer[:,i]
#                 bf = np.append(bf,rms_dt)
#                 rms_buffer[:,i] =bf[1:]
#                 v = np.nanstd(rms_buffer[:,i])
#                 rms_var[i],rms_var_corrected[i] = adaptiveMean(v,beta2,k+1,rms_var[i],biasCorrect =True)



#             # v = adaptiveVar(rms_dt,beta2, k,rms_mean[i],rms_var[i],thr, NoiseLevel = NoiseLevel)
#             # rms_var[i] = v
            
#             if (abs(rms_dt - rms_mean_corrected[i])>rms_var_corrected[i]*thr and abs(rms_dt - rms_mean_corrected[i] )>NoiseLevel and k>buffer_length) or rms_dt >100/(np.sqrt(2)):
                
#                 n+=1
            
            
#             """ rms """
        
        
        
        
        
#         intg = lowPassFilter(tauU,tauD, [n], 1/fs_target*nshift, intg[0])
#         if (intg[0] > 0.8 and  k>buffer_length) :
#             freeze = 1
#         else:
#             freeze =0

#         integrator.append(intg[0])  
#         rms_list.append(rms_array.copy())
#         rmsmean_list.append(rms_mean_corrected.copy())
#         rmsvar_list.append(rms_var_corrected.copy())
#         # odd_list.append(oddEnergy_array.copy())
#         # oddmean_list.append(oddmean.copy())
#         # oddvar_list.append(oddvar.copy())


#     rms_array = np.array(rms_list)
#     rmsmean_array = np.array(rmsmean_list)
#     rmsvar_array = np. array(rmsvar_list)
#     T = np.array(T)
#     fig, axes = plt.subplots(d.shape[1]+1,2)
#     fig.suptitle(figname)

#     for i in range(rms_array.shape[1]):
#         axes[i,0].plot(T,rms_array[:,i],color ='k')
#         axes[i,0].plot(T,rmsmean_array[:,i], color = 'y')
#         axes[i,0].plot(T,rmsmean_array[:,i] + rmsvar_array[:,i]*thr,'--', color = 'g')
#         axes[i,0].plot(T, rmsmean_array[:,i]-rmsvar_array[:,i]*thr,'--',color = 'g')
#         ind = np.argwhere((np.abs(rmsmean_array[:,i]-rms_array[:,i])> rmsvar_array[:,i]*thr)&(np.abs(rmsmean_array[:,i]-rms_array[:,i])>NoiseLevel))
        
#         axes[i,0].plot(T[ind],rms_array[ind,i],'.',color ='r')
#         axes[i,1].plot(T,rmsvar_array[:,i],color ='b')
#         axes[i,1].plot(T,np.abs(rmsmean_array[:,i]-rms_array[:,i]),'r')
#         axes[i,1].set_ylim([0,0.1])
#     axes[d.shape[1],0].plot(T,integrator, 'b')
#     axes[d.shape[1],0].set_ylim([0,2])
#     axes[d.shape[1],1].plot(data.iloc[:,0],data.loc[:,'Status'], 'g')

#     plt.show()











