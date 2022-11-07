import numpy as np
import matplotlib.pyplot as plt
from ..utility import *


def plotSpecgram(sig, P, figname, wsize, fs,noverlap,cmap='inferno', Npad=None,plotSave = False):
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
    
    if not isinstance(sig,np.ndarray):
        sig = np.array(sig)

    # rv window    
    window = _rv_win(P,wsize) 


    try:   
        fig,axes= plt.subplots(sig.shape[1],1)
        axes[0].set_title(figname)
        for i in range(sig.shape[1]):
            y = sig[:,i]
            vmin = 20*np.log10(np.max(y))-300
            _, _, _, im = axes[i].specgram(y, window=window, NFFT=wsize, Fs=fs, pad_to=Npad,  noverlap=noverlap,vmin = vmin, cmap=cmap)
            axes[i].set_xlabel('t (s)')
            axes[i].set_ylabel ('Hz')
            plt.colorbar(im, ax = axes[i])
        plt.tight_layout()

    except:
        fig= plt.figure()
        plt.title(figname)
        y = sig
        vmin = 20*np.log10(np.max(y))-300
        _, _, _, im = plt.specgram(y, window=window, NFFT=wsize, Fs=fs, pad_to=Npad,  noverlap=noverlap,vmin = vmin, cmap=cmap)
        plt.colorbar(im)
        plt.xlabel('t (s)')
        plt.ylabel ('Hz')


    
    plt.show()
    if plotSave:
        figname = figname+'_STFT'+'.png'
        fig.savefig(figname,format ='png',dpi =600)
