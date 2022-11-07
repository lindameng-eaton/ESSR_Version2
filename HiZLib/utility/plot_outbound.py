import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from .filter import lowPassFilter

def plotOutBound(T, sig, adapted_sig, std_sig, thr, tauU_outBound =0.4, tauD_outBound = 0.1, plot=True, ax=None, Label ='',savePlot = False, saveFolder = '.\\', figname = '',alarm_threshold=0.5, alarm_dur =1 ,tst=None, mingap = 0.2):
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
    threshold = std_sig*thr
    # threshold[threshold < NoiseLevel] = NoiseLevel
    upb_sig = adapted_sig + threshold
    Lwb_sig = adapted_sig -threshold
    outBound_ind = np.abs(adapted_sig - sig) > threshold
    sig_outBound = sig[outBound_ind]
    T_outBound = T[outBound_ind]
    deltat = T[1]-T[0]
    outBound_avg = lowPassFilter (tauU_outBound, tauD_outBound, outBound_ind*1, T[1]-T[0],outBound_ind[0])
    outBound_avg = np.array( outBound_avg)
    mean_outBound_avg = np.array([np.sum(outBound_avg[:i])*deltat for i in range(1, len(outBound_avg)+1)])
    Alarm_indx = (outBound_avg > alarm_threshold)*1
    label = measure.label(Alarm_indx)
    for lb in np.unique(label):
        if sum(label ==lb)*deltat < mingap  and np.sum(Alarm_indx[label==lb])==0:
            Alarm_indx[label == lb] =1
    label = measure.label(Alarm_indx)       



        # unique number of labels
    unique_label = np.unique(label)

   
    Fault_indicator =0
    alrm = np.zeros(outBound_avg.shape)
    # explore all the regions, if the length of the region is less than minstep, set the sign_sig within that region 0
    for i in unique_label:
        if (sum(label==i)*deltat >alarm_dur) and np.sum(Alarm_indx[label==i])>1:
            Fault_indicator = 1
            alrm[label==i] =1
   

            break
    
    detection_time = np.argwhere(alrm==1)
    if len(detection_time) and tst:
  
        delay_time =T[detection_time[0]] - tst
    else:
        delay_time = np.nan

    if plot:
        if  ax:
            ax.plot(T,sig,'k',label= Label)
            ax.plot(T,adapted_sig,'b',label = 'adapted')
            ax.plot(T,upb_sig,'b--', label = 'Boundary')
            ax.plot(T,Lwb_sig,'b--')
            ax.plot(T_outBound, sig_outBound, 'g*', label='out of boundary(OB)')
            
            # ax.plot(T,outBound_avg, 'r--', label ='outBound indicator')
        else:
            fig, ax = plt.subplots(3,1)
            ax[0].plot(T,sig,'k',label= Label)
            ax[0].plot(T,adapted_sig,'b')
            ax[0].plot(T,upb_sig,'b--')
            ax[0].plot(T,Lwb_sig,'b--')
            ax[0].plot(T_outBound, sig_outBound, 'g*', label='out of boundary(OB)')
            ax[0].legend()
            ax[1].plot(T,outBound_avg, 'k', label ='outBound indicator, Fault={}'.format(Fault_indicator))
            ax[1].plot(T,alrm,'r--')
            if tst:
                ax[1].plot([tst, tst], [0, 1], 'b--')
            ax[2].plot(T,mean_outBound_avg, 'g', label ='mean of the outBound indicator')
            if tst:
                ax[2].plot([tst, tst], [0, 1], 'b--')
            

            ax[1].legend()
            plt.show()

            if savePlot:
                figname = figname +'.png'
                filename = os.path.join(saveFolder, figname)

                fig.savefig(filename, format ='png',dpi =600)

        

    return delay_time

def reconstruction_waveletrec_plot(waveletrec, xmax,**kwargs):
    """Plot reconstruction from wavelet coeff on x [0,xmax] independently of amount of values it contains."""
    #plt.figure()
    #plt.plot(np.linspace(0, 1, len(yyy)), yyy, **kwargs)
    
    plt.plot(np.linspace(0, 1., num=len(waveletrec))*xmax, waveletrec, **kwargs)

def reconstruction_waveletcoeff_stem(waveletcoeff, xmax, **kwargs):
    """Plot coefficient vector on x [0,xmax] independently of amount of values it contains."""
    # ymax = waveletcoeff.max()
    plt.stem(np.linspace(0, 1., num=len(waveletcoeff))*xmax, waveletcoeff, **kwargs)
