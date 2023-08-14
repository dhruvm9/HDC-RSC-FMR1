#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:27:55 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import time 
import matplotlib.pyplot as plt 
import seaborn as sns
from functions import * 
from wrappers import *

#%% 

data_directory = '/media/DataDhruv/Recordings/Edinburgh-FMR1'
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_CueRot_WT.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_CueRot_KO.list'), delimiter = '\n', dtype = str, comments = '#')

meanupdur_hdc = [] 
meandowndur_hdc = []

meanupdur_rsc = [] 
meandowndur_rsc = []

CVup_hdc = []
CVdown_hdc  = []

CVup_rsc = []
CVdown_rsc  = []

allupdur_hdc = [] 
alldowndur_hdc = []

allupdur_rsc = [] 
alldowndur_rsc = []

updist_hdc = pd.DataFrame()
downdist_hdc = pd.DataFrame()

uplogdist_hdc = pd.DataFrame()
downlogdist_hdc = pd.DataFrame()

updist_rsc = pd.DataFrame()
downdist_rsc = pd.DataFrame()

uplogdist_rsc = pd.DataFrame()
downlogdist_rsc = pd.DataFrame()


for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    file = os.path.join(path, name +'.evt.py.hdn')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_hdc = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(path, name +'.evt.py.hup')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        up_hdc = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')    
        
    file = os.path.join(path, name +'.evt.py.rdn')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_rsc = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(path, name +'.evt.py.rup')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        up_rsc = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')   
        
#%% Durations 

    upbins = np.linspace(0,8,60)
    downbins = np.linspace(0,2,60)
    logbins = np.linspace(np.log10(0.02), np.log10(50), 10)

    ##HDC 
    
    updur_hdc = (up_hdc['end'] - up_hdc['start']) 
    meanupdur_hdc.append(np.mean(updur_hdc))
    CVup_hdc.append(np.std(updur_hdc)/np.mean(updur_hdc))
    allupdur_hdc.append([i for i in updur_hdc.values])
    
    downdur_hdc = (down_hdc['end'] - down_hdc['start'])
    meandowndur_hdc.append(np.mean(downdur_hdc))
    CVdown_hdc.append(np.std(downdur_hdc)/np.mean(downdur_hdc))
    alldowndur_hdc.append([i for i in downdur_hdc.values])   
   

    upd_hdc, _ = np.histogram(updur_hdc, upbins)
    upd_hdc = upd_hdc/sum(upd_hdc)
    
    downd_hdc, _  = np.histogram(downdur_hdc, downbins)
    downd_hdc = downd_hdc/sum(downd_hdc)
    
    uplogd_hdc, _ = np.histogram(np.log10(updur_hdc), logbins)
    uplogd_hdc = uplogd_hdc/sum(uplogd_hdc)
    
    downlogd_hdc, _ = np.histogram(np.log10(downdur_hdc), logbins)
    downlogd_hdc = downlogd_hdc/sum(downlogd_hdc)
        
    updist_hdc = pd.concat([updist_hdc, pd.Series(upd_hdc)], axis = 1)
    downdist_hdc = pd.concat([downdist_hdc, pd.Series(downd_hdc)], axis = 1)
            
    uplogdist_hdc = pd.concat([uplogdist_hdc, pd.Series(uplogd_hdc)], axis = 1)
    downlogdist_hdc = pd.concat([downlogdist_hdc, pd.Series(downlogd_hdc)], axis = 1)
    
    ##RSC 
    
    updur_rsc = (up_rsc['end'] - up_rsc['start']) 
    meanupdur_rsc.append(np.mean(updur_rsc))
    CVup_rsc.append(np.std(updur_rsc)/np.mean(updur_rsc))
    allupdur_rsc.append([i for i in updur_rsc.values])
    
    downdur_rsc = (down_rsc['end'] - down_rsc['start'])
    meandowndur_rsc.append(np.mean(downdur_rsc))
    CVdown_rsc.append(np.std(downdur_rsc)/np.mean(downdur_rsc))
    alldowndur_rsc.append([i for i in downdur_rsc.values])   
   

    upd_rsc, _ = np.histogram(updur_rsc, upbins)
    upd_rsc = upd_rsc/sum(upd_rsc)
    
    downd_rsc, _  = np.histogram(downdur_rsc, downbins)
    downd_rsc = downd_rsc/sum(downd_rsc)
    
    uplogd_rsc, _ = np.histogram(np.log10(updur_rsc), logbins)
    uplogd_rsc = uplogd_rsc/sum(uplogd_rsc)
    
    downlogd_rsc, _ = np.histogram(np.log10(downdur_rsc), logbins)
    downlogd_rsc = downlogd_rsc/sum(downlogd_rsc)
        
    updist_rsc = pd.concat([updist_rsc, pd.Series(upd_rsc)], axis = 1)
    downdist_rsc = pd.concat([downdist_rsc, pd.Series(downd_rsc)], axis = 1)
            
    uplogdist_rsc = pd.concat([uplogdist_rsc, pd.Series(uplogd_rsc)], axis = 1)
    downlogdist_rsc = pd.concat([downlogdist_rsc, pd.Series(downlogd_rsc)], axis = 1)
    
#%% Plotting 

upbincenter = 0.5 * (upbins[1:] + upbins[:-1])
downbincenter = 0.5 * (downbins[1:] + downbins[:-1])
logbincenter = 0.5 * (logbins[1:] + logbins[:-1])

##HDC 

uperr = updist_hdc.std(axis=1)
downerr = downdist_hdc.std(axis=1)
uplogerr = uplogdist_hdc.std(axis=1)
downlogerr = downlogdist_hdc.std(axis=1)

plt.figure()
plt.title('HDC')
plt.xlabel('Duration (s)')
plt.ylabel('P (duration)')
plt.plot(upbincenter, updist_hdc.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(upbincenter, updist_hdc.mean(axis = 1) - uperr, updist_hdc.mean(axis = 1) + uperr, color = 'r', alpha = 0.2)
plt.plot(downbincenter, downdist_hdc.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(downbincenter, downdist_hdc.mean(axis = 1) - downerr, 
                 downdist_hdc.mean(axis = 1) + downerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')

plt.figure()
plt.title('HDC')
plt.ylabel('P (duration)')
plt.plot(logbincenter, uplogdist_hdc.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(logbincenter, uplogdist_hdc.mean(axis = 1) - uplogerr, 
                 uplogdist_hdc.mean(axis = 1) + uplogerr, color = 'r', alpha = 0.2)
plt.plot(logbincenter, downlogdist_hdc.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(logbincenter, downlogdist_hdc.mean(axis = 1) - downlogerr, 
                 downlogdist_hdc.mean(axis = 1) + downlogerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')

##RSC 

uperr = updist_rsc.std(axis=1)
downerr = downdist_rsc.std(axis=1)
uplogerr = uplogdist_rsc.std(axis=1)
downlogerr = downlogdist_rsc.std(axis=1)

plt.figure()
plt.title('RSC')
plt.xlabel('Duration (s)')
plt.ylabel('P (duration)')
plt.plot(upbincenter, updist_rsc.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(upbincenter, updist_rsc.mean(axis = 1) - uperr, updist_rsc.mean(axis = 1) + uperr, color = 'r', alpha = 0.2)
plt.plot(downbincenter, downdist_rsc.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(downbincenter, downdist_rsc.mean(axis = 1) - downerr, 
                 downdist_rsc.mean(axis = 1) + downerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')

plt.figure()
plt.title('RSC')
plt.ylabel('P (duration)')
plt.plot(logbincenter, uplogdist_rsc.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(logbincenter, uplogdist_rsc.mean(axis = 1) - uplogerr, 
                 uplogdist_rsc.mean(axis = 1) + uplogerr, color = 'r', alpha = 0.2)
plt.plot(logbincenter, downlogdist_rsc.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(logbincenter, downlogdist_rsc.mean(axis = 1) - downlogerr, 
                 downlogdist_rsc.mean(axis = 1) + downlogerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')
    
    

    
    