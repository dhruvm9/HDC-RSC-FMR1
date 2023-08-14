#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:00:38 2023

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
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

genotype = []

uponsetcorr_hdc = [] 
downonsetcorr_hdc = [] 

uponsetcorr_rsc = []
downonsetcorr_rsc = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(os.path.join(path, 'LFP'))
               
    filepath = os.path.join(path, 'Sleep')
    listdir = os.listdir(filepath)
    file = [f for f in listdir if 'SleepState.states' in f]
    states = scipy.io.loadmat(os.path.join(filepath,file[0])) 
    
    sleepstate = states['SleepState']
    wake_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][0][:,0], end = sleepstate[0][0][0][0][0][0][:,1])
    nrem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][1][:,0], end = sleepstate[0][0][0][0][0][1][:,1])
    rem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][2][:,0], end = sleepstate[0][0][0][0][0][2][:,1])
    
    filepath = os.path.join(path, 'Analysis')
    listdir = os.listdir(filepath)
    file = [f for f in listdir if 'CellTypes' in f]
    celltype = scipy.io.loadmat(os.path.join(filepath,file[0])) 

    hdc = []
    rsc = []
            
    for i in range(len(spikes)):
        if celltype['posub'][i] == 1 and celltype['gd'][i] == 1:
            hdc.append(i)
            
    for i in range(len(spikes)):
        if celltype['rsc'][i] == 1 and celltype['gd'][i] == 1:
            rsc.append(i)
            
    file = [f for f in listdir if 'RecInfo' in f]
    rinfo = scipy.io.loadmat(os.path.join(filepath,file[0])) 
    
    isGranular = rinfo['isGranular'].flatten()
    isWT = rinfo['isWT'].flatten()
    
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
        
    genotype.append(isWT[0])
        
#%%  Cross corrs 

## UP onset of HDC 

    uponset_hdc_corr_hdc = nap.compute_eventcorrelogram(spikes[hdc], nap.Ts(t = up_hdc['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(uponset_hdc_corr_hdc)
    uponset_hdc_corr_hdc = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    uponset_hdc_corr_rsc = nap.compute_eventcorrelogram(spikes[rsc], nap.Ts(t = up_hdc['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(uponset_hdc_corr_rsc)
    uponset_hdc_corr_rsc = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
## DOWN onset of HDC 
    
    dnonset_hdc_corr_hdc = nap.compute_eventcorrelogram(spikes[hdc], nap.Ts(t = down_hdc['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(dnonset_hdc_corr_hdc)
    dnonset_hdc_corr_hdc = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    dnonset_hdc_corr_rsc = nap.compute_eventcorrelogram(spikes[rsc], nap.Ts(t = down_hdc['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(dnonset_hdc_corr_rsc)
    dnonset_hdc_corr_rsc = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
## Append 

    uponsetcorr_hdc.append(uponset_hdc_corr_hdc.mean(axis = 1))
    uponsetcorr_rsc.append(uponset_hdc_corr_rsc.mean(axis = 1))
    
    downonsetcorr_hdc.append(dnonset_hdc_corr_hdc.mean(axis = 1))
    downonsetcorr_rsc.append(dnonset_hdc_corr_rsc.mean(axis = 1))
    
#%% Plotting by genotype 

uponsetcorr_hdc = pd.DataFrame(uponsetcorr_hdc).T
uponsetcorr_rsc = pd.DataFrame(uponsetcorr_rsc).T

downonsetcorr_hdc = pd.DataFrame(downonsetcorr_hdc).T
downonsetcorr_rsc = pd.DataFrame(downonsetcorr_rsc).T

#%% 

WT = np.where(np.array(genotype) == 1)[0]
KO = np.where(np.array(genotype) == 0)[0]

upperlimit = -0.25
lowerlimit = 0.25

plt.figure()
plt.title('HDC UP onset correlogram (WT)')
plt.plot(uponsetcorr_hdc[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(uponsetcorr_rsc[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC UP (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')

plt.figure()
plt.title('HDC UP onset correlogram (KO)')
plt.plot(uponsetcorr_hdc[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(uponsetcorr_rsc[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC UP (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')

plt.figure()
plt.title('HDC DOWN onset correlogram (WT)')
plt.plot(downonsetcorr_hdc[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(downonsetcorr_rsc[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC DOWN (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')

plt.figure()
plt.title('HDC DOWN onset correlogram (KO)')
plt.plot(downonsetcorr_hdc[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(downonsetcorr_rsc[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC DOWN (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')