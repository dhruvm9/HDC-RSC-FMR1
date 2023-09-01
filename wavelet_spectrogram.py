#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:23:29 2023

@author: adrien
"""

import numpy as np 
import pandas as pd 
import scipy.io
import os, sys
import time 
import matplotlib.pyplot as plt 
import pynapple as nap
import seaborn as sns
from functions import *
from wrappers import *
from scipy.fft import fft, ifft
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import hilbert, fftconvolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import math 
import pickle

#%% 

def MorletWavelet(f, ncyc, si):
    
    #Parameters
    s = ncyc/(2*np.pi*f)    #SD of the gaussian
    tbound = (4*s);   #time bounds - at least 4SD on each side, 0 in center
    tbound = si*np.floor(tbound/si)
    t = np.arange(-tbound,tbound,si) #time
    
    #Wavelet
    sinusoid = np.exp(2*np.pi*f*t*-1j)
    gauss = np.exp(-(t**2)/(2*(s**2)))
    
    A = 1
    wavelet = A * sinusoid * gauss
    wavelet = wavelet / np.linalg.norm(wavelet)
    return wavelet 

#%% 

data_directory = '/media/adrien/LaCie/Edinburgh-FMR1'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM_test.list'), delimiter = '\n', dtype = str, comments = '#')

all_pspec_z_hdc = pd.DataFrame()
all_pspec_median_hdc = pd.DataFrame()

all_pspec_z_rsc = pd.DataFrame()
all_pspec_median_rsc = pd.DataFrame()

genotype = []

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
    
#%%
      
    lfp_rsc = nap.load_eeg(os.path.join(path, 'LFP') + '/' + name + '.lfp', channel = int(shank_to_channel[0][0]), n_channels = n_channels, frequency = 1250, precision ='int16', bytes_size = 2) 
    lfp_hdc = nap.load_eeg(os.path.join(path, 'LFP') + '/' + name + '.lfp', channel = int(shank_to_channel[2][0]), n_channels = n_channels, frequency = 1250, precision ='int16', bytes_size = 2) 
    
#%% 

    fmin = 0.5
    fmax = 150
    nfreqs = 100
    ncyc = 3 #5
    si = 1/fs
    
    downsample = 10
    
    freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)
    
    nfreqs = len(freqs)
    
###HDC 
    
    wavespec_hdc = nap.TsdFrame(t = lfp_hdc.index.values[::downsample], columns = freqs)
    powerspec_hdc = nap.TsdFrame(t = lfp_hdc.index.values[::downsample], columns = freqs)
    
    for f in range(len(freqs)):
         wavelet = MorletWavelet(freqs[f],ncyc,si)
         tmpspec = fftconvolve(lfp_hdc.values, wavelet, mode = 'same')
         wavespec_hdc[freqs[f]] = tmpspec [::downsample]
         temppower = abs(wavespec_hdc[freqs[f]]) #**2
         powerspec_hdc[freqs[f]] =  temppower #
         
###RSC 
    
    wavespec_rsc = nap.TsdFrame(t = lfp_rsc.index.values[::downsample], columns = freqs)
    powerspec_rsc = nap.TsdFrame(t = lfp_rsc.index.values[::downsample], columns = freqs)
    
    for f in range(len(freqs)):
         wavelet = MorletWavelet(freqs[f],ncyc,si)
         tmpspec = fftconvolve(lfp_rsc.values, wavelet, mode = 'same')
         wavespec_rsc[freqs[f]] = tmpspec [::downsample]
         temppower = abs(wavespec_rsc[freqs[f]]) #**2
         powerspec_rsc[freqs[f]] =  temppower #
         
#%% 

###HDC 

    DU_hdc = nap.Tsd(up_hdc['start'].values)
      
    realigned = powerspec_hdc.index[powerspec_hdc.index.get_indexer(DU_hdc.index.values, method='nearest')]
    
    # pspec_median_hdc = pd.DataFrame()
    pspec_z_hdc = pd.DataFrame()
    
    for i in range(len(powerspec_hdc.columns)):
        tmp = nap.compute_perievent(powerspec_hdc[powerspec_hdc.columns[i]], nap.Ts(realigned.values) , minmax = (-1,1), time_unit = 's')
           
        peth_all = []
        for j in range(len(tmp)):
            peth_all.append(tmp[j].as_series())
            
        trials = pd.concat(peth_all, axis = 1, join = 'outer')
        
        z = ((trials - trials.mean()) / trials.std()).mean(axis = 1)    
        pspec_z_hdc[freqs[i]] = z
        
        # mdn = (trials/trials.median()).mean(axis = 1)
        # pspec_median_hdc[freqs[i]] = mdn
        
        
    pspec_z_hdc['label'] = np.array([isWT[0] for x in range(len(pspec_z_hdc.index))])
    # pspec_median_hdc['label'] = np.array([isWT[0] for x in range(len(pspec_median_hdc.index))])
    
    # all_pspec_median_hdc = pd.concat((pspec_median_hdc, all_pspec_median_hdc))
    all_pspec_z_hdc = pd.concat((pspec_z_hdc, all_pspec_z_hdc))
    
    
###RSC 

    DU_rsc = nap.Tsd(up_rsc['start'].values)
      
    realigned = powerspec_rsc.index[powerspec_rsc.index.get_indexer(DU_rsc.index.values, method='nearest')]
    
    # pspec_median_hdc = pd.DataFrame()
    pspec_z_rsc = pd.DataFrame()
    
    for i in range(len(powerspec_rsc.columns)):
        tmp = nap.compute_perievent(powerspec_rsc[powerspec_rsc.columns[i]], nap.Ts(realigned.values) , minmax = (-1,1), time_unit = 's')
           
        peth_all = []
        for j in range(len(tmp)):
            peth_all.append(tmp[j].as_series())
            
        trials = pd.concat(peth_all, axis = 1, join = 'outer')
        
        z = ((trials - trials.mean()) / trials.std()).mean(axis = 1)    
        pspec_z_rsc[freqs[i]] = z
        
        # mdn = (trials/trials.median()).mean(axis = 1)
        # pspec_median_hdc[freqs[i]] = mdn
        
        
    pspec_z_rsc['label'] = np.array([isWT[0] for x in range(len(pspec_z_rsc.index))])
    # pspec_median_hdc['label'] = np.array([isWT[0] for x in range(len(pspec_median_hdc.index))])
    
    # all_pspec_median_hdc = pd.concat((pspec_median_hdc, all_pspec_median_hdc))
    all_pspec_z_rsc = pd.concat((pspec_z_rsc, all_pspec_z_rsc))
    


#%% 

###HDC

specgram_z_hdc_wt = all_pspec_z_hdc[all_pspec_z_hdc['label'] == 1].iloc[:,:-1]
specgram_z_hdc_ko = all_pspec_z_hdc[all_pspec_z_hdc['label'] == 0].iloc[:,:-1]

specgram_z_hdc_wt2 = specgram_z_hdc_wt.groupby(specgram_z_hdc_wt.index).mean()
specgram_z_hdc_ko2 = specgram_z_hdc_ko.groupby(specgram_z_hdc_ko.index).mean()

###RSC

specgram_z_rsc_wt = all_pspec_z_rsc[all_pspec_z_rsc['label'] == 1].iloc[:,:-1]
specgram_z_rsc_ko = all_pspec_z_rsc[all_pspec_z_rsc['label'] == 0].iloc[:,:-1]

specgram_z_rsc_wt2 = specgram_z_rsc_wt.groupby(specgram_z_rsc_wt.index).mean()
specgram_z_rsc_ko2 = specgram_z_rsc_ko.groupby(specgram_z_rsc_ko.index).mean()

 
#%% 
   
###Plotting 

## Z-scored 

# labels = 2**np.arange(8)[3:]
# norm = colors.TwoSlopeNorm(vmin=specgram_z_hdc_wt2[freqs[38:]][-0.1:0.5].values.min(),
#                             vcenter=0, vmax = specgram_z_hdc_wt2[freqs[38:]][-0.1:0.5].values.max())
       
# fig, ax = plt.subplots()
# plt.title('WT')
# cax = ax.imshow(specgram_z_hdc_wt2[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
#             origin = 'lower',
#             extent = [specgram_z_hdc_wt2[freqs[38:]][-0.1:0.5].index.values[0], 
#                       specgram_z_hdc_wt2[freqs[38:]][-0.1:0.5].index.values[-1],
#                       np.log10(specgram_z_hdc_wt2[freqs[38:]].columns[0]),
#                       np.log10(specgram_z_hdc_wt2[freqs[38:]].columns[-1])], 
#             norm = norm)
# plt.xlabel('Time from DU (s)')
# plt.xticks([0, 0.25, 0.5])
# plt.ylabel('Freq (Hz)')
# plt.yticks(np.log10(labels), labels = labels)
# cbar = fig.colorbar(cax, label = 'Power (z)')
# plt.axvline(0, color = 'k',linestyle = '--')
# plt.gca().set_box_aspect(1)

# fig, ax = plt.subplots()
# plt.title('KO')
# cax = ax.imshow(specgram_z_hdc_ko2[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
#             origin = 'lower',
#             extent = [specgram_z_hdc_ko2[freqs[38:]][-0.1:0.5].index.values[0], 
#                       specgram_z_hdc_ko2[freqs[38:]][-0.1:0.5].index.values[-1],
#                       np.log10(specgram_z_hdc_ko2[freqs[38:]].columns[0]),
#                       np.log10(specgram_z_hdc_ko2[freqs[38:]].columns[-1])], 
#             norm = norm)
# plt.xlabel('Time from DU (s)')
# plt.xticks([0, 0.25, 0.5])
# plt.ylabel('Freq (Hz)')
# plt.yticks(np.log10(labels), labels = labels)
# cbar = fig.colorbar(cax, label = 'Power (z)')
# plt.axvline(0, color = 'k',linestyle = '--')
# plt.gca().set_box_aspect(1)



# ## Median Normalized

# norm = colors.TwoSlopeNorm(vmin=specgram_m_hdc[freqs[38:]][-0.1:0.5].values.min(),
#                            vcenter=1, vmax = specgram_m_hdc[freqs[38:]][-0.1:0.5].values.max())
       
# plt.figure()
# plt.title('Median-normalized spectrogram (HDC)')
# plt.imshow(specgram_m_hdc[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
#            origin = 'lower',
#            extent = [specgram_m_hdc[freqs[38:]][-0.1:0.5].index.values[0],
#                      specgram_m_hdc[freqs[38:]][-0.1:0.5].index.values[-1],
#                      np.log10(specgram_z_hdc[freqs[38:]].columns[0]),
#                      np.log10(specgram_z_hdc[freqs[38:]].columns[-1])], 
#            norm = norm)
# plt.xlabel('Time from DU (s)')
# plt.ylabel('Freq (Hz)')  
# plt.yticks(np.log10(labels), labels = labels)
# plt.colorbar()
# plt.axvline(0, color = 'k', linestyle ='--')
# plt.gca().set_box_aspect(1)

#%% 

###HDC 

specgram_z_hdc_wt2.to_pickle(data_directory + '/specgram_z_hdc_wt.pkl')
specgram_z_hdc_ko2.to_pickle(data_directory + '/specgram_z_hdc_ko.pkl')

###RSC 

specgram_z_rsc_wt2.to_pickle(data_directory + '/specgram_z_rsc_wt.pkl')
specgram_z_rsc_ko2.to_pickle(data_directory + '/specgram_z_rsc_ko.pkl')
