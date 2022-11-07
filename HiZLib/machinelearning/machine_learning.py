import tensorflow as tf
import numpy as np


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
    dataset = dataset.map(lambda win, tg: (
        tf.cast(win, tf.float32), tf.cast(tg, tf.int32)))

    return dataset


def calculate_evaluation(prediction, Label):
    """calculate accuration, false_alarm, F1 score with predicted value and labeled data
    prediction: a numpy array with predictions, 0/ False: normal; 1/True:fault  
    Label: a numpy array with 0s and 1s : 0 : normal 1: fault
    """
    prediction = prediction * 1
    TP = np.sum(Label[prediction > 0])
    FN = np.sum(1-prediction[Label == 1])
    FP = np.sum(prediction[Label == 0])
    TN = np.sum(1-prediction[Label == 0])
    F1 = 2*TP/(2*TP+FP+FN)
    Accur = (TP+TN)/len(prediction)
    False_alarm = FP / len(prediction)
    return Accur, False_alarm, F1
