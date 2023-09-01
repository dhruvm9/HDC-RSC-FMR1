#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:18:50 2023

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

data_directory = '/media/adrien/LaCie/Edinburgh-FMR1'

specgram_z_hdc_wt = pd.read_pickle(data_directory + '/specgram_z_hdc_wt.pkl')
specgram_z_hdc_ko = pd.read_pickle(data_directory + '/specgram_z_hdc_ko.pkl')
    
specgram_z_rsc_wt = pd.read_pickle(data_directory + '/specgram_z_rsc_wt.pkl')
specgram_z_rsc_ko = pd.read_pickle(data_directory + '/specgram_z_rsc_ko.pkl')

fmin = 0.5
fmax = 150
nfreqs = 100
 
freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)

#%%

###HDC

labels = 2**np.arange(8)[3:]
norm = colors.TwoSlopeNorm(vmin=specgram_z_hdc_wt[freqs[38:]][-0.1:0.5].values.min(),
                            vcenter=0, vmax = specgram_z_hdc_wt[freqs[38:]][-0.1:0.5].values.max())
       
fig, ax = plt.subplots()
plt.title('HDC WT')
cax = ax.imshow(specgram_z_hdc_wt[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_z_hdc_wt[freqs[38:]][-0.1:0.5].index.values[0], 
                      specgram_z_hdc_wt[freqs[38:]][-0.1:0.5].index.values[-1],
                      np.log10(specgram_z_hdc_wt[freqs[38:]].columns[0]),
                      np.log10(specgram_z_hdc_wt[freqs[38:]].columns[-1])], 
            norm = norm)
plt.xlabel('Time from DU (s)')
plt.xticks([0, 0.25, 0.5])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)

fig, ax = plt.subplots()
plt.title('HDC KO')
cax = ax.imshow(specgram_z_hdc_ko[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_z_hdc_ko[freqs[38:]][-0.1:0.5].index.values[0], 
                      specgram_z_hdc_ko[freqs[38:]][-0.1:0.5].index.values[-1],
                      np.log10(specgram_z_hdc_ko[freqs[38:]].columns[0]),
                      np.log10(specgram_z_hdc_ko[freqs[38:]].columns[-1])], 
            norm = norm)
plt.xlabel('Time from DU (s)')
plt.xticks([0, 0.25, 0.5])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)

#%% 

###HDC

labels = 2**np.arange(8)[3:]
norm = colors.TwoSlopeNorm(vmin=specgram_z_rsc_wt[freqs[38:]][-0.1:0.5].values.min(),
                            vcenter=0, vmax = specgram_z_rsc_wt[freqs[38:]][-0.1:0.5].values.max())
       
fig, ax = plt.subplots()
plt.title('RSC WT')
cax = ax.imshow(specgram_z_rsc_wt[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_z_rsc_wt[freqs[38:]][-0.1:0.5].index.values[0], 
                      specgram_z_rsc_wt[freqs[38:]][-0.1:0.5].index.values[-1],
                      np.log10(specgram_z_rsc_wt[freqs[38:]].columns[0]),
                      np.log10(specgram_z_rsc_wt[freqs[38:]].columns[-1])], 
            norm = norm)
plt.xlabel('Time from DU (s)')
plt.xticks([0, 0.25, 0.5])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)

fig, ax = plt.subplots()
plt.title('RSC KO')
cax = ax.imshow(specgram_z_rsc_ko[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_z_rsc_ko[freqs[38:]][-0.1:0.5].index.values[0], 
                      specgram_z_rsc_ko[freqs[38:]][-0.1:0.5].index.values[-1],
                      np.log10(specgram_z_rsc_ko[freqs[38:]].columns[0]),
                      np.log10(specgram_z_rsc_ko[freqs[38:]].columns[-1])], 
            norm = norm)
plt.xlabel('Time from DU (s)')
plt.xticks([0, 0.25, 0.5])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)
