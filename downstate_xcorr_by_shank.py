#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:06:37 2023

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

uponsetcorr_hdc_s2 = [] 
downonsetcorr_hdc_s2 = [] 

uponsetcorr_rsc_s2 = []
downonsetcorr_rsc_s2 = []

uponsetcorr_hdc_s3 = [] 
downonsetcorr_hdc_s3 = [] 

uponsetcorr_rsc_s3 = []
downonsetcorr_rsc_s3 = []


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
    genotype.append(isWT[0])
    
#%% Detect DOWN states in each shank of HDC 

    bin_size = 0.01 #s
    smoothing_window = 0.02

## Shank 2 
    
    shank2 = np.intersect1d(np.where(shank == 2)[0], hdc)
    spikes_s2 = spikes[shank2]
    
    rates_s2 = spikes_s2.count(bin_size, nrem_ep)
           
    total2_s2 = rates_s2.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2_s2 = total2_s2.sum(axis =1)
    total2_s2 = nap.Tsd(total2_s2)
    idx_s2 = total2_s2.threshold(np.percentile(total2_s2.values,20),'below')
    
    down_s2 = idx_s2.time_support
    
    down_s2 = nap.IntervalSet(start = down_s2['start'], end = down_s2['end'])
    down_s2 = down_s2.drop_short_intervals(bin_size)
    down_s2 = down_s2.merge_close_intervals(bin_size*2)
    down_s2 = down_s2.drop_short_intervals(bin_size*3)
    down_s2 = down_s2.drop_long_intervals(bin_size*50)
    
    up_s2 = nap.IntervalSet(down_s2['end'][0:-1], down_s2['start'][1:])
    down_s2 = nrem_ep.intersect(down_s2)
    
## Shank 3 
    
    shank3 = np.intersect1d(np.where(shank == 3)[0], hdc)
    spikes_s3 = spikes[shank3]
    
    rates_s3 = spikes_s3.count(bin_size, nrem_ep)
           
    total2_s3 = rates_s3.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2_s3 = total2_s3.sum(axis = 1)
    total2_s3 = nap.Tsd(total2_s3)
    idx_s3 = total2_s3.threshold(np.percentile(total2_s3.values,20),'below')
    
    down_s3 = idx_s3.time_support
    
    down_s3 = nap.IntervalSet(start = down_s2['start'], end = down_s2['end'])
    down_s3 = down_s3.drop_short_intervals(bin_size)
    down_s3 = down_s3.merge_close_intervals(bin_size*2)
    down_s3 = down_s3.drop_short_intervals(bin_size*3)
    down_s3 = down_s3.drop_long_intervals(bin_size*50)
    
    up_s3 = nap.IntervalSet(down_s3['end'][0:-1], down_s3['start'][1:])
    down_s3 = nrem_ep.intersect(down_s3)
    
#%% 

##Shank 2

## UP onset of HDC 

    uponset_hdc_s2 = nap.compute_eventcorrelogram(spikes[hdc], nap.Ts(t = up_s2['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(uponset_hdc_s2)
    uponset_hdc_s2 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    uponset_rsc_s2 = nap.compute_eventcorrelogram(spikes[rsc], nap.Ts(t = up_s2['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(uponset_rsc_s2)
    uponset_rsc_s2 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
## DOWN onset of HDC 
    
    dnonset_hdc_s2 = nap.compute_eventcorrelogram(spikes[hdc], nap.Ts(t = down_s2['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(dnonset_hdc_s2)
    dnonset_hdc_s2 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    dnonset_rsc_s2 = nap.compute_eventcorrelogram(spikes[rsc], nap.Ts(t = down_s2['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(dnonset_rsc_s2)
    dnonset_rsc_s2 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
##Shank 3

## UP onset of HDC 

    uponset_hdc_s3 = nap.compute_eventcorrelogram(spikes[hdc], nap.Ts(t = up_s3['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(uponset_hdc_s3)
    uponset_hdc_s3 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    uponset_rsc_s3 = nap.compute_eventcorrelogram(spikes[rsc], nap.Ts(t = up_s3['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(uponset_rsc_s3)
    uponset_rsc_s3 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
## DOWN onset of HDC 
    
    dnonset_hdc_s3 = nap.compute_eventcorrelogram(spikes[hdc], nap.Ts(t = down_s3['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(dnonset_hdc_s3)
    dnonset_hdc_s3 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    dnonset_rsc_s3 = nap.compute_eventcorrelogram(spikes[rsc], nap.Ts(t = down_s3['start'].values), binsize = 0.005, windowsize = 1, ep = nrem_ep)
    tmp = pd.DataFrame(dnonset_rsc_s3)
    dnonset_rsc_s3 = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
  

## Append 

    uponsetcorr_hdc_s2.append(uponset_hdc_s2.mean(axis = 1))
    uponsetcorr_rsc_s2.append(uponset_rsc_s2.mean(axis = 1))
    
    downonsetcorr_hdc_s2.append(dnonset_hdc_s2.mean(axis = 1))
    downonsetcorr_rsc_s2.append(dnonset_rsc_s2.mean(axis = 1))
    
    uponsetcorr_hdc_s3.append(uponset_hdc_s3.mean(axis = 1))
    uponsetcorr_rsc_s3.append(uponset_rsc_s3.mean(axis = 1))
    
    downonsetcorr_hdc_s3.append(dnonset_hdc_s3.mean(axis = 1))
    downonsetcorr_rsc_s3.append(dnonset_rsc_s3.mean(axis = 1))
    
#%% 

uponsetcorr_hdc_s2 = pd.DataFrame(uponsetcorr_hdc_s2).T
uponsetcorr_rsc_s2 = pd.DataFrame(uponsetcorr_rsc_s2).T

downonsetcorr_hdc_s2 = pd.DataFrame(downonsetcorr_hdc_s2).T
downonsetcorr_rsc_s2 = pd.DataFrame(downonsetcorr_rsc_s2).T

uponsetcorr_hdc_s3 = pd.DataFrame(uponsetcorr_hdc_s3).T
uponsetcorr_rsc_s3 = pd.DataFrame(uponsetcorr_rsc_s3).T

downonsetcorr_hdc_s3 = pd.DataFrame(downonsetcorr_hdc_s3).T
downonsetcorr_rsc_s3 = pd.DataFrame(downonsetcorr_rsc_s3).T
    
WT = np.where(np.array(genotype) == 1)[0]
KO = np.where(np.array(genotype) == 0)[0]

upperlimit = -0.25
lowerlimit = 0.25
#%% 


plt.figure()
plt.suptitle('HDC UP onset correlogram (WT)')
plt.subplot(121)
plt.title('Shank 2')
plt.plot(uponsetcorr_hdc_s2[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(uponsetcorr_rsc_s2[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC UP (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')
plt.subplot(122)
plt.title('Shank 3')
plt.plot(uponsetcorr_hdc_s3[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(uponsetcorr_rsc_s3[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC UP (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')

plt.figure()
plt.suptitle('HDC UP onset correlogram (KO)')
plt.subplot(121)
plt.title('Shank 2')
plt.plot(uponsetcorr_hdc_s2[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(uponsetcorr_rsc_s2[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC UP (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')
plt.subplot(122)
plt.title('Shank 3')
plt.plot(uponsetcorr_hdc_s3[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(uponsetcorr_rsc_s3[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC UP (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')

plt.figure()
plt.suptitle('HDC DOWN onset correlogram (WT)')
plt.subplot(121)
plt.title('Shank 2')
plt.plot(downonsetcorr_hdc_s2[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(downonsetcorr_rsc_s2[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC DOWN (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')
plt.subplot(122)
plt.title('Shank 3')
plt.plot(downonsetcorr_hdc_s3[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(downonsetcorr_rsc_s3[WT].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC DOWN (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')

plt.figure()
plt.suptitle('HDC DOWN onset correlogram (KO)')
plt.subplot(121)
plt.title('Shank 2')
plt.plot(downonsetcorr_hdc_s2[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(downonsetcorr_rsc_s2[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC DOWN (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')
plt.subplot(122)
plt.title('Shank 3')
plt.plot(downonsetcorr_hdc_s3[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'cornflowerblue', label = 'HDC')
plt.plot(downonsetcorr_rsc_s3[KO].mean(axis = 1)[upperlimit:lowerlimit], color = 'indianred', label = 'RSC')
plt.axvline(0, color = 'k', linestyle ='--')
plt.xlabel('Time from HDC DOWN (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')



