# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:42:34 2019

@author: E0450443

========================================
function library for HiZ fault detection 
========================================
"""

import numpy as np
import pickle
    
#%% 
def findix(x,xrange):
    return [i for i in range(len(x)) if x[i]>=xrange[0] and x[i]<xrange[1]] 
    
#%%
def pklLoad(pklFile):    
    f = open(pklFile,'rb')   
    data = pickle.load(f)   
    # Return                        
    return data

#%% csvDump
def pklDump(pklFile,data):
    f = open(pklFile,'wb')
    pickle.dump(data,f)
    
    
#%%
# Interrupter logic
class interrupter_logic(object):
    def __init__(self,T_list):
        self.N_list = [1 for i in range(len(T_list))]
        self.A_list = [0 for i in range(len(T_list))]
        self.I_list = [0 for i in range(len(T_list))]
        for i in range(len(T_list)):
            if i==0:
                # N(0)=1 means every T0
                self.N_list[i] = 1
            else:
                # N(i)=T(i)/T(0)
                self.N_list[i] = int(round(T_list[i]/T_list[0]))
    
    def run(self):
        for i in range(len(self.N_list)):
            if i==0:
                # I(0)=1 means interrupter at every T0
                self.I_list[i] = 1
                self.A_list[i] = 0
            else:
                # accumulate A(i) and issue interrupter when A(i)==N(i)
                self.A_list[i] = self.A_list[i]+1
                if self.A_list[i]==self.N_list[i]:
                    self.I_list[i] = 1
                    self.A_list[i] = 0
                else:
                    self.I_list[i] = 0
    

#%%
# SDI: Sum of Difference Current
class sdi_logic(object):
    """ T1 task 
        SDI is the summation of cycle difference 
        ...
        n_sub = the number of data points for subtraction calculation
        n_sum = the number of data points for summation calculation
    """
    def __init__(self,n_sub,n_sum): 
        self.x_sub = np.zeros(n_sub)
        self.i_sub = 0
        #
        self.x_sum = np.zeros(n_sum)
        self.i_sum = 0

    def run(self,xin):
        # calculate DI by subtraction
        self.di = abs(xin-self.x_sub[self.i_sub])
        # save new measurement in x_sub
        self.x_sub[self.i_sub] = xin
        self.i_sub = self.i_sub+1
        if self.i_sub==len(self.x_sub):
            self.i_sub = 0
        #
        # save new DI in x_sum
        self.x_sum[self.i_sum] = self.di
        self.i_sum = self.i_sum+1
        if self.i_sum==len(self.x_sum):
            self.i_sum = 0
        # calculate SDI
        self.sdi = sum(self.x_sum)


#%%
# IIR filter for generating SDI_REF
class iir_filter(object):
    """ T2 task 
        Find out filtered SDI_REF with given rising limit
        ...
        Ts = sampling time step
        Tf = filter time constant
        s =  integer multiple of d for SDI outlier removal
        f = integer multiple of d for SDI instant delta limit
    """
    def __init__(self,Ts,Tf,s,f):
        self.alpha = np.exp(-Ts/Tf)
        self.s = s
        self.f = f
        self.sdi_ref = 0
        
    def run(self,sdi,d,freeze):
        if freeze != 1:
            if (sdi-self.sdi_ref) > (self.s*d):
                xin = self.sdi_ref+self.f*d
            else:
                xin = sdi
            self.sdi_ref = (1-self.alpha)*xin+self.alpha*self.sdi_ref
        else:
            self.sdi_ref = sdi # this might be wrong 
                
                
#%%
# Freeze logic
class freeze_logic(object):
    """ T1 task
        Freeze the logic if abrupt event happens in the power system
        NOTE 1: only current is used for freeze logic for now
        NOTE 2: RMS is used with my own comprehension
        ...
        n_mem = number of data points in rms calculation
        n_sub = number of data points in subtraction storage
        thres = threshold
        n_dly = number of execution time steps for on delay
        f_dly = number of execution time steps for off delay
    """
    def __init__(self,n_mem,n_sub,thres,n_dly,f_dly):
        self.x_mem = np.zeros(n_mem) 
        self.i_mem = 0 
        #
        self.x_sub = np.zeros(n_sub) 
        self.i_sub = 0  
        #
        self.thres = thres
        self.n_dly = n_dly
        self.i_dly = 0
        #
        self.f_dly = f_dly
        self.j_dly = 0
        # 
        self.y = 0
        
    def run(self,xin): 
        # store raw input in rms calculation
        self.x_mem[self.i_mem] = abs(xin)
        self.i_mem = self.i_mem+1
        if self.i_mem==len(self.x_mem):
            self.i_mem = 0
        # calculate the rms value
        self.rms = np.sqrt(np.mean(self.x_mem**2))        
        #
        # calculate dx by subtraction
        self.dx = abs(self.rms-self.x_sub[self.i_sub])
        # save new input in x_sub
        self.x_sub[self.i_sub] = self.rms
        self.i_sub = self.i_sub+1
        if self.i_sub==len(self.x_sub):
            self.i_sub = 0
        #
        # ON delay
        if self.dx >= self.thres:
            self.i_dly = self.i_dly+1
        else:
            self.i_dly = 0
        #
        if self.i_dly >= self.n_dly:
            self.y = 1
        #
        # OFF delay            
        if self.y == 1:
            self.j_dly = self.j_dly+1
        if (self.j_dly >= self.f_dly) and (self.dx < self.thres):
            self.y = 0
            self.j_dly = 0


#%%
# Adaptive tuning logic
class adaptive_tuning(object):
    """ T2 and T3 task
        Dynamically update margin d 
        If SDI is higher than SDI_REF+d for a certain percentage of the time in 1sec
        and if this condition continues for a delay time, raise d
        NOTE: only 1sec update is considered for the evaluation purpose
        ...
        n_1sec = number of data point in 1sec duration
        thres2 = threshold for the update
        n_dly2 = number of execution time steps (T3) in delay
        h2 =  percentage of increase
        d0 = initial value of d
    """
    def __init__(self,n_1sec,thres2,n_dly2,h2,d0):
        self.x_mem = np.zeros(n_1sec) 
        self.i_mem = 0 
        #
        self.thres2 = thres2
        self.n_dly2 = n_dly2
        self.i_dly2 = 0
        self.h2 = h2
        self.d = d0
        
    def run(self,sdi,sdi_ref,mode):
        if mode == 0:
            # T2 task: check if sdi > sdi_ref+d
            if sdi-sdi_ref > self.d:
                temp = 1 # np.abs(sdi-sdi_ref)
            else: 
                temp = 0
            # save result in storage
            self.x_mem[self.i_mem] = temp
            self.i_mem = self.i_mem+1
            if self.i_mem==len(self.x_mem):
                self.i_mem = 0
        else:
            # T3 task: update d if condition satisfied
            if sum(self.x_mem) >= self.thres2:
                self.i_dly2 = self.i_dly2+1
            else:
                self.i_dly2 = 0
            #
            if self.i_dly2 >= self.n_dly2:
                self.d = self.d+self.h2*sdi_ref
            
            
        

#%%
# Trending logic
class trending_logic(object):
    """ T2 and T3 task 
        Calculate how much and how frequently SDI deviates from SDI_REF by more than d
        Save all such incidents (per second) in dt and rd arrays
        ...
        n_mem = number of data points in storage 
        dt_unit = unit of time (e.g., 2 cycles)
    """
    def __init__(self,n_mem,dt_unit): 
        self.t_mem = np.ones(n_mem)*(-99) 
        self.r_mem = np.ones(n_mem)*(-99)
        self.i_mem = 0 
        #
        self.t_old = 0
        self.dt = []
        self.rd = []
        #
        self.dt_unit = dt_unit

    def run(self,sdi,sdi_ref,d,t_now,freeze,mode):
        if freeze != 1:
            if mode == 0:
                # T2 task
                dSDI = abs(sdi-sdi_ref)
                if dSDI > d:
                    self.t_mem[self.i_mem] = t_now
                    self.r_mem[self.i_mem] = dSDI/d
                    self.i_mem = self.i_mem+1
                    if self.i_mem==len(self.t_mem):
                        self.i_mem = 0
            else:
                # T3 task
                self.dt = [(temp-self.t_old)/self.dt_unit for temp in self.t_mem if temp>=0]
                self.rd = [temp for temp in self.r_mem if temp>=0]
                if len(self.dt) > 0:
                    self.t_old = max(self.t_mem)
                #
                self.t_mem[:] = -99
                self.r_mem[:] = -99
                self.i_mem = 0
        
   
#%%
# Decision logic
class decision_logic(object):
    """ T3 task 
        Count the number of (dt,rd) pairs in fault region
        Issue fault flag is the total count in N windows (e.g., 5sec) exceeds q1
        ...
        n_mem = number of counting time windows
        fault_region = lookup table that defines the fault count region
        q1 = minimum counts in N time windows for generating fault flag
    """
    def __init__(self,n_mem,fault_region,q1): 
        self.x_mem = np.zeros(n_mem)
        self.i_mem = 0
        #
        self.fault_region = fault_region
        self.q1 = q1
        #
        self.fault_flag = 0

    def run(self,dt,rd):
        # count the number of incidents in this time window
        count = 0
        for ii in range(len(dt)):
            rd_thres = np.interp(dt[ii],self.fault_region['dt'],self.fault_region['rd'])
            if rd[ii] >= rd_thres:
                count = count+1
        # save the count in this time window
        self.x_mem[self.i_mem] = count
        self.i_mem = self.i_mem+1
        if self.i_mem==len(self.x_mem):
            self.i_mem = 0
        #
        # check if total count in N time windows exceeds the threshold
        if sum(self.x_mem) >= self.q1:
            self.fault_flag = 1
            
       