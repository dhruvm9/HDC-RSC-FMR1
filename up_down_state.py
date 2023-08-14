#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:54:29 2023

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
            
    spikes_hdc = spikes[hdc]
    spikes_rsc = spikes[rsc]
    
#%% Detect DOWN states 

    bin_size = 0.01 #s
    smoothing_window = 0.02

    rates_hdc = spikes_hdc.count(bin_size, nrem_ep)
    rates_rsc = spikes_rsc.count(bin_size, nrem_ep)
    
    ##HDC
    
    total2_hdc = rates_hdc.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2_hdc = total2_hdc.sum(axis =1)
    total2_hdc = nap.Tsd(total2_hdc)
    idx_hdc = total2_hdc.threshold(np.percentile(total2_hdc.values,20),'below')
    
    down_hdc = idx_hdc.time_support
    
    down_hdc = nap.IntervalSet(start = down_hdc['start'], end = down_hdc['end'])
    down_hdc = down_hdc.drop_short_intervals(bin_size)
    down_hdc = down_hdc.merge_close_intervals(bin_size*2)
    down_hdc = down_hdc.drop_short_intervals(bin_size*3)
    down_hdc = down_hdc.drop_long_intervals(bin_size*50)
    
    up_hdc = nap.IntervalSet(down_hdc['end'][0:-1], down_hdc['start'][1:])
    down_hdc = nrem_ep.intersect(down_hdc)
    
    ##RSC 
    
    total2_rsc = rates_rsc.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2_rsc = total2_rsc.sum(axis =1)
    total2_rsc = nap.Tsd(total2_rsc)
    idx_rsc = total2_rsc.threshold(np.percentile(total2_rsc.values,20),'below')
    
    down_rsc = idx_rsc.time_support
    
    down_rsc = nap.IntervalSet(start = down_rsc['start'], end = down_rsc['end'])
    down_rsc = down_rsc.drop_short_intervals(bin_size)
    down_rsc = down_rsc.merge_close_intervals(bin_size*2)
    down_rsc = down_rsc.drop_short_intervals(bin_size*3)
    down_rsc = down_rsc.drop_long_intervals(bin_size*50)
    
    up_rsc = nap.IntervalSet(down_rsc['end'][0:-1], down_rsc['start'][1:])
    down_rsc = nrem_ep.intersect(down_rsc)
    
    ##Validate 
    # plt.figure()
    # plt.plot(total2_hdc) 
    # plt.plot(idx_hdc, 'o') 
    # plt.axhline(np.percentile(total2_hdc.values,20), color = 'k')
    
    # plt.figure()
    # plt.plot(total2_rsc) 
    # plt.plot(idx_rsc, 'o') 
    # plt.axhline(np.percentile(total2_rsc.values,20), color = 'k')
    

#%% Write to Neuroscope 

    ##HDC 
    
    start = down_hdc.as_units('ms')['start'].values
    ends = down_hdc.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(down_hdc)

    texttowrite = np.vstack(((np.repeat(np.array(['PyDownHDC start 1']), n)), 
                              (np.repeat(np.array(['PyDownHDC stop 1']), n))
                              )).T.flatten()

    evt_file = path + '/' + name + '.evt.py.hdn'
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()        

    start = up_hdc.as_units('ms')['start'].values
    ends = up_hdc.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(up_hdc)

    texttowrite = np.vstack(((np.repeat(np.array(['PyUpHDC start 1']), n)), 
                              (np.repeat(np.array(['PyUpHDC stop 1']), n))
                              )).T.flatten()

    evt_file = path + '/' + name + '.evt.py.hup'
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()
    
    ##RSC 
    
    start = down_rsc.as_units('ms')['start'].values
    ends = down_rsc.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(down_rsc)

    texttowrite = np.vstack(((np.repeat(np.array(['PyDownRSC start 1']), n)), 
                              (np.repeat(np.array(['PyDownRSC stop 1']), n))
                              )).T.flatten()

    evt_file = path + '/' + name + '.evt.py.rdn'
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()        

    start = up_rsc.as_units('ms')['start'].values
    ends = up_rsc.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(up_rsc)

    texttowrite = np.vstack(((np.repeat(np.array(['PyUpRSC start 1']), n)), 
                              (np.repeat(np.array(['PyUpRSC stop 1']), n))
                              )).T.flatten()

    evt_file = path + '/' + name + '.evt.py.rup'
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()

    
    
    