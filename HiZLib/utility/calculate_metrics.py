import numpy as np
from skimage import measure
import scipy
from collections import Counter
import pandas as pd

# Internal Parameters
SAMPLE_PERIOD_ESTIMATION_WINDOW_LEN = 5


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
        crossing_indices = np.nonzero(np.diff(1*sign_sig))[0]
    else:
        crossing_indices = np.nonzero(np.diff(1*sign_sig) > 0)[0]

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
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
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
    fano = var/mean

    # return a list
    return [n5, n25, n75, n95, median, mean, var, fano]


def calculate_sample_freq(timestamps: pd.Series) -> int:
    sample_period_estimate = (
        timestamps[SAMPLE_PERIOD_ESTIMATION_WINDOW_LEN] - timestamps[0]
    ) / SAMPLE_PERIOD_ESTIMATION_WINDOW_LEN
    return int(np.round(1 / sample_period_estimate))
