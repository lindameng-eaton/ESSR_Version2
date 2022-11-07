import numpy as np
from ..utility import *
from ..feature_calibration import *


def plot_skewness_Kurtosis(dt,fs, column_list, wsize_statistic,noverlap_statistic,thr,buffer_dur = 0.5,saveFolder='.\\',figname='', beta1=0.99,beta2=0.99, savePlot=False,tauU_outBound =0.4, tauD_outBound = 0.1,alarm_threshold=0.5, alarm_dur =1,tst=None):
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
    deltat = (wsize_statistic-noverlap_statistic) /fs
    buffer_length = int(buffer_dur/deltat)
    
    if not isinstance(dt,np.ndarray):
        dt = np.array(dt)

    L = len(dt)
    dt = dt.reshape(L,-1)
    # column_list = dt.columns
    DTC_sk =[]
    DTC_crf=[]
    DTC_kurtosis =[]
    
    for i in np.arange(dt.shape[1]):
        
       
        sig_modified = dataRearrange(dt[:,i], noverlap_statistic, wsize_statistic)
                
                
        
        sk =[]
        kurtosis =[]
        crf=[]



        for nrow in range(len(sig_modified)):
            
            sig_ = np.abs(sig_modified[nrow])
            mu = np.mean(sig_)
            sigma = np.std(sig_)
            
            s = calculate_skewness(sig_,mu,sigma)
            k = calculate_kurtosis(sig_,mu,sigma)
            c =  calculate_CRF(sig_)
            sk.append(s)
            kurtosis.append(k)
            crf.append(c)
        
        
        
        # sk_noiseLevel = np.abs(np.median(sk[0:10]))/NoiseLevel_f
        # sk_noiseLevel = np.abs(np.quantile(sk[0:50],quantile_Level)-np.quantile([sk[0:50]],1-quantile_Level))
        adapted_sk, std_sk = adaptiveThreshold(sk, beta1,beta2,thr,buffer_length)
        #crf_noiseLevel = np.abs(np.median(crf[0:10]))/NoiseLevel_f
        # crf_noiseLevel = np.abs(np.quantile(crf[0:50],quantile_Level)-np.quantile([crf[0:50]],1-quantile_Level))
        adapted_crf,std_crf = adaptiveThreshold(crf, beta1,beta2,thr, buffer_length)
        #kurtosis_noiseLevel = np.abs(np.median(kurtosis[0:10]))/NoiseLevel_f
        # kurtosis_noiseLevel =  np.abs(np.quantile(kurtosis[0:50],quantile_Level)-np.quantile([kurtosis[0:50]],1-quantile_Level))

        adapted_kurtosis,std_kurtosis =  adaptiveThreshold(kurtosis, beta1,beta2,thr, buffer_length)
        adapted_sk = np.array(adapted_sk)
        std_sk =np.array(std_sk)
        sk = np.array(sk)

        T = np.arange(len(sk))*deltat
        
   
        
        
        adapted_crf = np.array(adapted_crf)
        std_crf =np.array(std_crf)
        crf = np.array(crf)


        
        adapted_kurtosis = np.array(adapted_kurtosis)
        std_kurtosis =np.array(std_kurtosis)
        kurtosis = np.array(kurtosis)

        

       
        delayT_sk = plotOutBound(T, sk, adapted_sk, std_sk,  thr, tauU_outBound = tauU_outBound , tauD_outBound = tauD_outBound, Label ='skewness_{}, {}'.format(column_list[i],figname),alarm_threshold=alarm_threshold,alarm_dur =alarm_dur,tst=tst,savePlot= savePlot, saveFolder = saveFolder, figname = 'skewness_'+figname)
        delayT_crf =plotOutBound(T, crf, adapted_crf, std_crf, thr,tauU_outBound = tauU_outBound , tauD_outBound = tauD_outBound , Label ='CRF_{}, {}'.format(column_list[i],figname),alarm_threshold=alarm_threshold,alarm_dur =alarm_dur,tst=tst,savePlot= savePlot, saveFolder = saveFolder, figname = 'crf_'+figname)
        delayT_kur = plotOutBound(T, kurtosis, adapted_kurtosis, std_kurtosis, thr,tauU_outBound = tauU_outBound , tauD_outBound = tauD_outBound , Label ='kurtosis_{}, {}'.format(column_list[i],figname),alarm_threshold=alarm_threshold,alarm_dur =alarm_dur,tst=tst,savePlot= savePlot, saveFolder = saveFolder, figname = 'kurtosis_'+figname)
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
