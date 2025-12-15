#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:03:12 2023

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
percent = []

downdist = pd.DataFrame() 
downlogdist = pd.DataFrame() 

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
    
    filepath = os.path.join(path, 'Analysis')
    listdir = os.listdir(filepath)
    file = [f for f in listdir if 'RecInfo' in f]
    rinfo = scipy.io.loadmat(os.path.join(filepath,file[0])) 
    
    isGranular = rinfo['isGranular'].flatten()
    isWT = rinfo['isWT'].flatten()
    genotype.append(isWT[0])
    
#%% Co-occurring DOWN states

    downbins = np.linspace(0,2,60)
    logbins = np.linspace(np.log10(0.02), np.log10(50), 10)

    cooccur = down_hdc.intersect(down_rsc)
    percent.append(len(cooccur)/max(len(down_hdc), len(down_rsc)))
    
    dur = cooccur['end'] - cooccur['start']    
    
    downd, _ = np.histogram(dur, downbins)
    downlogd, _ = np.histogram(dur, logbins)
    
    downdist = pd.concat([downdist, pd.Series(downd)], axis = 1)
    downlogdist = pd.concat([downlogdist, pd.Series(downlogd)], axis = 1)
    
