import numpy as np
import matplotlib.pyplot as plt
from ..utility import *


def plot_VoltageVsCurrent(d, chanel_name,fs,N,ncycle, thrp=0, figname='', savefolder_trace ='',plotSave=False):
    """snapshot of voltage vs current trajectories
    Args:
      d: data matrix
      chanel_name: name of the chanel of voltage and current signals
      fs: sample frequency
      N: number of point per cycle
      ncycle: number of cycle per snapshot
      savefolder_trace: if plotSave == True, the snapshots of the trajectory will be saved into savefolder_trace
      thrp: percentage of the max value of Voltage
      figname: data file name
      plotSave: boolean. True: save figure False: not save
    
    """
    deltaT = 4 #s
    deltaN = int(deltaT * fs)


    if not isinstance(d, np.ndarray):
        d = np.array(d)

    if d.shape[1]>5:
        dt =d
        plt.figure()
        for st in np.arange(0,d.shape[0],deltaN):
            thr = max(dt[:,2])*thrp
            plt.plot([thr, thr], [min(dt[:,5]), max(dt[:,5])])
            # for i, ax in enumerate(axes):
            if st < d.shape[0]-N*ncycle:
                plt.plot(dt[st:st+N*ncycle,2],dt[st:st+N*ncycle,5],label='st={}s'.format(st/fs))

                plt.legend()
                plt.xlabel('Voltage (V)')
                plt.ylabel('Current (A)')

        plt.title(figname)
        plt.show()

        if plotSave:
            savefile = os.path.join(savefolder_trace,'Trace_'+figname+'_{}.png'.format(chanel_name) )
            
            plt.savefig(savefile,format = 'png')

        fig, axes = plt.subplots(3,2)
        T = d[:,0]
        for i in range(3):

            v = d[:,i+1]
            I = d[:,i+4]
            thr = max(v) *thrp
            indx0 = calculate_crossing( v, threshold =thr,nodirection=False)
            T_= T[indx0]
            I0 = I[indx0]
            axes[i,0].plot(T_,I0,'b.')
            axes[i,1].plot(T_[1:],np.diff(I0), 'g.')

            axes[i,1].set_ylim([-0.1,0.1])
        plt.tight_layout()
        plt.show()



 
    else:
        print('this case doesnt contain both voltage and current')
