import math
import numpy as np
from scipy.signal import butter
from scipy import signal
from typing import List, Union, Optional

# Internal parameters
BUTTERWORTH_ORDER = 10
BUTTERWORTH_TYPE = "lowpass"
BUTTERWORTH_OP_TYPE = "sos"


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
            y = tempy + (i - tempy) * min(1, dt/tauU)
        else:
            y = tempy + (i - tempy) * min(1, dt/tauD)

        Y.append(y)
        tempy = y

    return Y

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
    freq = np.fft.fftfreq(N, 1/fs)
    freq = freq[0:N//2]

    freq, X_fft = myfft(sig, fs, freq, P=0)

    # find the freq indices that are going to remove
    indx = np.argwhere(freq <= rmvf)

    # add up all the removing components
    temp = np.zeros(N)
    n = np.arange(N)
    for k in indx:
        temp = temp + (X_fft[k] * np.exp(1j*2*np.pi*n*k/N)).real

    # subtract the removing sum from the given sig
    r = sig - temp
    return r

def _calculate_sigma(delta, P):

    sigma_term = 1
    for g in np.arange(1, P+1):
        sigma_term = sigma_term*((g*g) - (delta*delta))
    return sigma_term

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
        C = math.comb(2*P, P)
        DL = C/(2**(2*P))
        win = DL*np.cos(2*np.pi*n*p/N)
        # p > 0 inside loop
        for i in np.arange(1, P+1):
            p = i
            C = math.comb(2*P, P-p)

            DL = ((-1)**p)*(C/(2**(2*P-1)))
            win = win + DL*np.cos(2*np.pi*n*p/N)

    else:
        print('P must be nonnegative')
        win = np.nan

    return win

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
    sig = sig*window

    # calculate frequency components
    fftsig = np.fft.rfft(sig, paddingM)/N
    freqsig = np.fft.rfftfreq(paddingM, 1/fs)
    fftsig = fftsig*2
    fftsig[0] /= 2

    # calculate correct factor for window function
    factor = ((2**(2*P))/math.factorial(2*P)) * _calculate_sigma(0, P)
    fftsig = fftsig*factor

    # find the frequency spectrum for given FreqList using nearist method
    if not len(FreqList):
        FreqList = freqsig
        y2 = fftsig
    else:
        indx = np.argmin(abs(FreqList.reshape(1, -1) -
                         freqsig.reshape(-1, 1)), axis=0)
        y2 = fftsig[indx]
    # return
    return (FreqList, y2)


def filter_butterworth(
    signal_arr: np.ndarray,
    cutoff_freq: Union[List[float], float],
    sample_freq: int,
    order: Optional[int] = BUTTERWORTH_ORDER,
    filter_type: Optional[str] = BUTTERWORTH_TYPE,
) -> np.ndarray:
    """Returns filtered signal
    Note: cutoff_freq is either a float or a list of form [lowerband, upperband]
    """

    # nyquist frequency is half the sample rate
    nyquist_freq = 0.5 * sample_freq
    normalized_cutoff = (np.ndarray(cutoff_freq) / nyquist_freq).reshape(-1)
    second_order_sections = butter(
        order, Wn=normalized_cutoff, btype=filter_type, output=BUTTERWORTH_OP_TYPE
    )
    return signal.sosfilt(second_order_sections, signal_arr)
