import pickle
import numpy as np


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

def next_power_of_2(x):
    """
    calculate the nearest power 2 num for given value x
    Args: 
        x: value
    return: nearest power 2
    """
    return 1 if x == 0 else 2**(x - 1).bit_length()

def find_index(freq_array, f_target):
    """find the index in freq_array that freq_array[index] is the closest to f_target


    """
    if not isinstance(freq_array, np.ndarray):
        freq_array = np.array(freq_array)
    if not isinstance(f_target, np.ndarray):
        f_target = np.array(f_target)
    indx = np.argmin(abs(f_target.reshape(1, -1) -
                     freq_array.reshape(-1, 1)), axis=0)
    return indx

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
    step = wsize-noverlap
    wsize = np.int32(wsize)
    if noverlap == 0:
        nRow = np.int32(np.floor(L/wsize))
        sig_ = sig[0:nRow*wsize].reshape(nRow, -1)

        return sig_
    else:
        step = wsize - noverlap
        sig_ = [sig[i:i+wsize] for i in range(0, L-wsize+1, step)]
        sig_ = np.array(sig_)

        return sig_
