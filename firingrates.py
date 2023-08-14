#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:42:03 2023

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

hdc_wake_rates_wt = []
hdc_nrem_rates_wt = []
hdc_rem_rates_wt = [] 

rsc_wake_rates_wt = []
rsc_nrem_rates_wt = []
rsc_rem_rates_wt = [] 

hdc_wake_rates_ko = []
hdc_nrem_rates_ko = []
hdc_rem_rates_ko = [] 

rsc_wake_rates_ko = []
rsc_nrem_rates_ko = []
rsc_rem_rates_ko = [] 


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
    
    filepath = os.path.join(path, 'Analysis')
    listdir = os.listdir(filepath)
    file = [f for f in listdir if 'RecInfo' in f]
    rinfo = scipy.io.loadmat(os.path.join(filepath,file[0])) 
    isWT = rinfo['isWT'][0][0]
        
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
    
    if isWT == 1:
        hdc_wake_rates_wt.extend(spikes_hdc.restrict(wake_ep)._metadata['rate'].values)
        hdc_nrem_rates_wt.extend(spikes_hdc.restrict(nrem_ep)._metadata['rate'].values)
        hdc_rem_rates_wt.extend(spikes_hdc.restrict(rem_ep)._metadata['rate'].values)
    
        rsc_wake_rates_wt.extend(spikes_rsc.restrict(wake_ep)._metadata['rate'].values)
        rsc_nrem_rates_wt.extend(spikes_rsc.restrict(nrem_ep)._metadata['rate'].values)
        rsc_rem_rates_wt.extend(spikes_rsc.restrict(rem_ep)._metadata['rate'].values)
         
        
    else: 
        hdc_wake_rates_ko.extend(spikes_hdc.restrict(wake_ep)._metadata['rate'].values)
        hdc_nrem_rates_ko.extend(spikes_hdc.restrict(nrem_ep)._metadata['rate'].values)
        hdc_rem_rates_ko.extend(spikes_hdc.restrict(rem_ep)._metadata['rate'].values)
    
        rsc_wake_rates_ko.extend(spikes_rsc.restrict(wake_ep)._metadata['rate'].values)
        rsc_nrem_rates_ko.extend(spikes_rsc.restrict(nrem_ep)._metadata['rate'].values)
        rsc_rem_rates_ko.extend(spikes_rsc.restrict(rem_ep)._metadata['rate'].values)
               


#%% 

WT_hdc = np.array(['WT_hdc' for x in range(len(hdc_wake_rates_wt))])
KO_hdc = np.array(['KO_hdc' for x in range(len(hdc_wake_rates_ko))])
WT_rsc = np.array(['WT_rsc' for x in range(len(rsc_wake_rates_wt))])
KO_rsc = np.array(['KO_rsc' for x in range(len(rsc_wake_rates_ko))])

types = np.hstack([WT_hdc, KO_hdc, WT_rsc, KO_rsc])

wakerates = []
wakerates.extend(hdc_wake_rates_wt)
wakerates.extend(hdc_wake_rates_ko)
wakerates.extend(rsc_wake_rates_wt)
wakerates.extend(rsc_wake_rates_ko)

nremrates = []
nremrates.extend(hdc_nrem_rates_wt)
nremrates.extend(hdc_nrem_rates_ko)
nremrates.extend(rsc_nrem_rates_wt)
nremrates.extend(rsc_nrem_rates_ko)

remrates = [] 
remrates.extend(hdc_rem_rates_wt)
remrates.extend(hdc_rem_rates_ko)
remrates.extend(rsc_rem_rates_wt)
remrates.extend(rsc_rem_rates_ko)


wakedf = pd.DataFrame(data = [wakerates, types], index = ['rate', 'type']).T
nremdf = pd.DataFrame(data = [nremrates, types], index = ['rate', 'type']).T
remdf = pd.DataFrame(data = [remrates, types], index = ['rate', 'type']).T

#%% 

plt.figure()
plt.title('Wake')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = wakedf['type'], y=wakedf['rate'].astype(float) , data = wakedf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = wakedf['type'], y=wakedf['rate'].astype(float) , data = wakedf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax, alpha = 0.05)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)


plt.figure()
plt.title('NREM')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = nremdf['type'], y=nremdf['rate'].astype(float) , data = nremdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = nremdf['type'], y=nremdf['rate'].astype(float) , data = nremdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = nremdf['type'], y = nremdf['rate'].astype(float), data = nremdf, color = 'k', dodge=False, ax=ax, alpha = 0.05)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)


plt.figure()
plt.title('REM')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = remdf['type'], y=remdf['rate'].astype(float) , data = remdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = remdf['type'], y=remdf['rate'].astype(float) , data = remdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = remdf['type'], y = remdf['rate'].astype(float), data = remdf, color = 'k', dodge=False, ax=ax, alpha = 0.05)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)