# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:42:34 2019

@author: E0450443

========================================
SEL arcing sensing technology
========================================
"""

import numpy as np
from ast_lib import *
import matplotlib.pyplot as plt     
import pandas as pd 
from hiz_lib import *
import glob
   
#%% 
# # Australia PBSP data
# pklFile = '.\\data_files\\VT101.pkl'
# #pklFile = '.\\data_files\\capsw.pkl'
# pklData = pklLoad(pklFile)
# t0 = pklData['Time']
# x0 = pklData['Direct current']

filepath_list =['C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Transformer_Inrush10kHz\\hd5\\*.h5','C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Transformer_Inrush_Brkr_On_Second\\*.h5','C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Normal_Operation10kHz\\*.h5', 'C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Capacitor_Switching_10kHz\\*.h5','C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Capacitor_Switching_Back_to_Back_10kHz\\*.h5','C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\Branch_Open_Load_Lost\\HDFfolder\\*.h5','C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\ph2e100kHz\\hd5\\*.h5','C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\grass100kHzat10KHz\\*.h5']
# filepath_list = ['C:\\Users\\E0644597\\Eaton\\ESSR - ERL\\Technology Development\\Australian Bush fire test data\\PSCAD_RESULTS\\\PV_Generic_Simulation\HDFfolder (2)\\*.h5']
chanel_name ='Ipt3'

for fld in filepath_list:
     file_list = [name for name in glob.glob(fld)]
     for file_path in file_list:
        
                  
       
        data= data_acquisition( chanel_name, file_path)
        figname = os.path.basename(file_path).split('.')[0]
        t0 = data.loc[:,'Time']
        x0 = data.iloc[:, 4:7].sum(axis = 1)




        dt0 = (t0[5]-t0[0])/5
        f1 = 50         # fundamental frequency
        n1 = 50         # number of samples per cycle

        d0_scale = 1

#%% 
## PSCAD simulation data
#pklFile = '.\\data_files\\arc_furnace.pkl'
##pklFile = '.\\data_files\\trf_inrush.pkl'
#pklData = pklLoad(pklFile)
#t0 = pklData['time']
#x0 = pklData['In:1']
#
#dt0 = (t0[100]-t0[0])/100
#f1 = 60         # fundamental frequency
#n1 = 32         # number of samples per cycle
#
#d0_scale = 1

#%%
## EPRI testing data
#pklFile = '.\\data_files\\2016-05-18_14_38_29_ev24_sand_01.pkl'
##pklFile = '.\\data_files\\2016-05-18_14_54_56_ev25_grass_10.pkl'
#pklData = pklLoad(pklFile)
#t0 = pklData['t']
#x0 = pklData['x']
#
#dt0 = (t0[100]-t0[0])/100
#f1 = 60         # fundamental frequency
#n1 = 32         # number of samples per cycle
#
#d0_scale = 1

#%% 
## SEL data
#csvFile = '.\\data_files\\SEL_HIF.csv'
##pklFile = '.\\data_files\\capsw.pkl'
#df = pd.read_csv(csvFile)
#t0 = np.array(df['Seconds'])
#x0 = np.array(df[' IF A'])
#
#dt0 = (t0[5]-t0[0])/5
#f1 = 60         # fundamental frequency
#n1 = 32         # number of samples per cycle
#
#d0_scale = 100

#%%
# Select only limited data for fast testing purpose
#ix = findix(t0,[5.0,5.05])
#t0 = t0[ix]
#x0 = x0[ix]

        #%%
        # Interrupter logic
        T0 = dt0
        T1 = 1/f1/n1
        T2 = 2/f1
        T3 = 1

        # Interrupter
        T_list = [T0,T1,T2,T3]
        intrpt = interrupter_logic(T_list)

        # SDI
        n_sub = n1
        n_sum = int(round(T2/T1))
        sdi = sdi_logic(n_sub,n_sum)

        # Freeze logic
        n_mem = n1
        n_sub = int(round(T2/T1))
        thres = 3.0
        n_dly = n1
        f_dly = int(round(0.5/T1))
        freeze = freeze_logic(n_mem,n_sub,thres,n_dly,f_dly)

        # IIR filter
        Ts= T2
        Tf = 1
        s = 2
        f = 1
        iir = iir_filter(Ts,Tf,s,f)

        # Adaptive tuning of d
        n_1sec = int(round(T3/T2))
        thres2 = 0.60*n_1sec
        n_dly2 = 2
        h2 = 0.2
        # init d0 = 0.3 *d0_scale
        d0 = 0.3*d0_scale
        adp = adaptive_tuning(n_1sec,thres2,n_dly2,h2,d0)

        # Trending logic
        n_mem = int(round(T3/T2))
        dt_unit = T2
        trend = trending_logic(n_mem,dt_unit)

        # Decision logic
        n_mem = int(round(5/T3))
        fault_region = {}
        fault_region['dt'] = [0,2,5,5000]
        fault_region['rd'] = [1.5,1.5,3,(3-1.5)/(5-2)*(5000-5)+3]
        q1 = 30
        decision = decision_logic(n_mem,fault_region,q1)


        #%%
        # Main program
        pltData = {}
        data_keys = [
        't1',
        't2',
        'x1',
        'di',
        'freeze',
        'sdi',
        'sdi_ref',
        'adp_d',
        'flag',
        'debug',
        ]

        for key in data_keys:
            pltData[key] = []

        #
        hiz_flag = 0
        for ii in range(len(t0)):
            this_t = t0[ii]
            
            # Interrupter logic
            intrpt.run()
            
            # T1 tasks
            if intrpt.I_list[1]==1:
                # SDI logic
                sdi.run(x0[ii])
                if this_t < t0[0]+n1*T1:
                    sdi.di = 0
                    sdi.x_sum[:] = 0
                    sdi.sdi = 0
                    
                # Freeze logic
                freeze.thres = 2*adp.d
                freeze.run(x0[ii])
                if this_t < t0[0]+2*T2:
                    freeze.i_dly = 0
                    
            # T2 tasks
            if intrpt.I_list[2]==1:
                # IIR filter
                iir.run(sdi.sdi,adp.d,freeze.y)
                if this_t < t0[0]+2*T2:
                    iir.sdi_ref = sdi.sdi
                    
                # Adaptive tuning of d
                mode = 0
                adp.run(sdi.sdi,iir.sdi_ref,mode)
                
                # Trending and memory
                mode = 0
                trend.run(sdi.sdi,iir.sdi_ref,adp.d,this_t,freeze.y,mode)
                
                
            # T3 tasks
            if intrpt.I_list[3]==1:
                # Adaptive tuning of d
                mode = 1
                adp.run(sdi.sdi,iir.sdi_ref,mode)
                
                # Trending and memory
                mode = 1
                trend.run(sdi.sdi,iir.sdi_ref,adp.d,this_t,freeze.y,mode)
                
                # Decision logic
                decision.run(trend.dt,trend.rd)
                if this_t < t0[0]+5:
                    decision.x_mem[:] = 0
                    decision.fault_flag = 0
                
                
            # save for plot
            if intrpt.I_list[1]==1:
                pltData['t1'].append(this_t)
                pltData['x1'].append(x0[ii])
                pltData['di'].append(sdi.di)
                pltData['freeze'].append(freeze.y*0.95)
                pltData['debug'].append((freeze.dx))
                
            if intrpt.I_list[2]==1:
                pltData['t2'].append(this_t)
                pltData['sdi'].append(sdi.sdi)
                pltData['sdi_ref'].append(iir.sdi_ref)
                pltData['adp_d'].append(adp.d)
                pltData['flag'].append(decision.fault_flag)

                
                    
        #%%
        # plot
        fig = plt.figure(1,figsize=[8.5,11.5])
        plt.clf()
        
        legend_loc = 'upper left'
        legend_fnt = 8
        xlim = [t0[0],t0[-1:].values]
        nrow, ncol = 5, 1
        iplt = 0
        #
        iplt = iplt+1
        plt.subplot(nrow,ncol,iplt)
        plt.plot(pltData['t1'],pltData['x1'],label='x1')
        plt.legend(loc=legend_loc,fontsize=legend_fnt)
        plt.xlabel('Time')
        plt.xlim(xlim)
        plt.title(figname)
        #
        iplt = iplt+1
        plt.subplot(nrow,ncol,iplt)
        plt.plot(pltData['t1'],pltData['di'],label='DI')
        plt.legend(loc=legend_loc,fontsize=legend_fnt)
        plt.xlabel('Time')
        plt.xlim(xlim)
        #
        iplt = iplt+1
        plt.subplot(nrow,ncol,iplt)
        plt.semilogy(pltData['t2'],pltData['sdi'],label='SDI')
        plt.semilogy(pltData['t2'],pltData['sdi_ref'],label='SDI_REF')
        plt.legend(loc=legend_loc,fontsize=legend_fnt)
        plt.xlabel('Time')
        plt.grid(True)
        plt.xlim(xlim)
        #plt.ylim([1.e-6,1.e-2])
        #
        iplt = iplt+1
        plt.subplot(nrow,ncol,iplt)
        plt.plot(pltData['t2'],pltData['adp_d'],label='adp_d')
        plt.legend(loc=legend_loc,fontsize=legend_fnt)
        plt.xlabel('Time')
        plt.grid(True)
        plt.xlim(xlim)
        #
        iplt = iplt+1
        plt.subplot(nrow,ncol,iplt)
        plt.plot(pltData['t1'],pltData['freeze'],label='Freeze')
        plt.plot(pltData['t2'],pltData['flag'],label='Fault')
        plt.legend(loc=legend_loc,fontsize=legend_fnt)
        plt.xlabel('Time')
        plt.grid(True)
        plt.xlim(xlim)
        plt.ylim([-0.1,1.1])

       
        #
        #iplt = iplt+1
        #plt.subplot(nrow,ncol,iplt)
        #plt.plot(pltData['t1'],pltData['debug'],label='debug')
        #plt.legend(loc=legend_loc,fontsize=legend_fnt)
        #plt.xlabel('Time')
        #plt.xlim(xlim)
        #
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
     
        

        plt.savefig(f'ast_debug_figure_{figname}.png',format ='png')
        #plt.savefig(pklFile.replace('data_files','png_files').replace('.pkl','_ast.png'),dpi=300)
