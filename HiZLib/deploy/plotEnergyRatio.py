import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .calculate_metrics import *
from ..utility import *

def plot_EnergyRatio(d,fs, f0, wsize,freqComp,savefolder_energy=".\\", P= 1,plotSave=False, Npad = None,noverlap = None):
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
        noverlap = int(np.fix(wsize*1/2))
    if not Npad:
        Npad = wsize

  
    dt = d.copy()

    column_list = d.columns
    if not isinstance(dt,np.ndarray):
        dt = np.array(dt)
    
    nrow = len(freqComp)
    palette = sns.color_palette(None, nrow)
    deltaT = (wsize - noverlap)/fs

    for i in np.arange(0, dt.shape[1]):
        
        fig,axes = plt.subplots(nrow,1)
        f_0 =[f0]
        Energy0= calculate_signal_HarmonicEnergy(dt[:,i], wsize,noverlap, fs,f_0,Npad,P)
        
        T = np.arange(len(Energy0))*deltaT
        Energy0 = np.array(Energy0)

        for j,comp in enumerate(freqComp):
            
            
            selectedComp = freqComp[comp]
            Energy = calculate_signal_HarmonicEnergy(dt[:,i], wsize,noverlap, fs,selectedComp,Npad,P)
            Energy = np.array(Energy)
            eRatio = np.log10(Energy/Energy0)

            
            axes[j].plot(T,eRatio,color = palette[j],label=comp)
            axes[j].set_xlim([0,T.max()])
            axes[j].legend()

        
        plt.title('EnergyRatio_{}'.format(column_list[i]))
        plt.show()
        if plotSave:
            savefile = os.path.join(savefolder_energy,'EnergyRatio_{}'.format(column_list[i]) )
            fig.savefig(savefile+'.png',format ='png',dpi=600)
