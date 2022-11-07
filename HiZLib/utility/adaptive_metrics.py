import numpy as np


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
    if not y0:
        y0 = yt

    y0 = IIRfilter(y0, yt, beta1)

    if biasCorrect:
        corrected_y = y0/(1-beta1**i)

    else:
        corrected_y = y0

    return corrected_y

def IIRfilter(y, yt, beta):
    """
    Args: 
    y: previous y
    yt: current y
    beta: parameter for the integration time constant
    Output: updated y

    """
    return beta*y + (1-beta) * yt

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
    if v <= max(df0*thr, NoiseLevel) or i <= round(1/(1-beta2)):

        # update df0 using iirfilter
        df0 = IIRfilter(df0, v, beta2)

        # correct df0 if biasCorrect is True
        if biasCorrect:
            corrected_df0 = df0/(1-beta2**i)
        else:
            corrected_df0 = df0
    else:
        corrected_df0 = df0

    return corrected_df0

def adaptiveThreshold(sig, beta1, beta2, thr, buffer_length,  y0=None, df0=None, biasCorrect=False):
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

    sig_buffer = np.ones((buffer_length, 1))*np.nan
    sig_var = 0

    # loop through all the data points
    for i, yt in enumerate(sig):

        corrected_y = adaptiveMean(
            yt, beta1, i, corrected_y, biasCorrect=biasCorrect)
        if np.abs(yt-corrected_y) < sig_var * thr or i < buffer_length:
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
