import numpy as np
from ..feature_calibration import *
from ..utility import *


def plot_mulInformation(V, I, T, fs, column_list, nbins, rmvf, thr, filename, tst, wsize, noverlap, tauU_outBound, tauD_outBound, beta1_dur=0.2, beta2_dur=2, buffer_dur=0.5, alarm_threshold=0.5, alarm_dur=0.4, savePlot=False, plot=True):
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
    buffer_length = int(buffer_dur/deltaT)

    beta1 = 1-deltaT/beta1_dur
    beta2 = 1-deltaT/beta2_dur

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
        curr_win = dataRearrange(curr, noverlap, wsize)
        vol_win = dataRearrange(vol, noverlap, wsize)

        mul_info = []

        # buffer_dur = 1
        # Nbuffer = int(np.floor(buffer_dur/deltaT))

        # freeze = 0
        # baseline_array = []
        for j in range(curr_win.shape[0]):
            x = curr_win[j, :]
            x = highPassFilter(x, fs, rmvf)
            x = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
            y = vol_win[j, :]
            y = highPassFilter(y, fs, rmvf)
            y = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))
            inds_x = np.digitize(x, bins)
            x = np.array(
                [(bins[inds_x[n]-1]+bins[inds_x[n]])/2 for n in range(x.size)])
            inds_y = np.digitize(y, bins)
            y = np.array(
                [(bins[inds_y[n]-1]+bins[inds_y[n]])/2 for n in range(y.size)])
            mulI = computeMI(x, y)
            mul_info.append(mulI)

        mul_info = np.array(mul_info)
        # mul_noiseLevel= np.abs(np.quantile(mul_info[0:Nbuffer],quantile_Level)-np.quantile([mul_info[0:Nbuffer]],1-quantile_Level))
        adapted_mul, std_mul = adaptiveThreshold(
            mul_info, beta1, beta2, thr, buffer_length)
        delay = plotOutBound(T_avg, mul_info, adapted_mul, std_mul, thr, tauU_outBound=tauU_outBound, tauD_outBound=tauD_outBound, Label='MultualInfo_{},nbins={},rmvf={},{}'.format(
            filename, nbins, rmvf, column_list[i]), alarm_threshold=alarm_threshold, alarm_dur=alarm_dur, tst=tst, savePlot=savePlot, plot=plot, saveFolder=saveFolder, figname=filename+"nbins{}".format(nbins)+"rmvf{}".format(rmvf)+'{}'.format(column_list[i]))
        delay_list.append(delay)
    return delay_list
