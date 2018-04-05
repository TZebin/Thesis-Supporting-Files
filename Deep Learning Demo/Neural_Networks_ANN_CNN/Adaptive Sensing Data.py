# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:12:52 2018
http://ieeexplore.ieee.org/document/7320218/
@author: mchijtz4
"""


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import pandas as pd
dataset=pd.read_csv('Series_with Label_20.csv')

A_data=dataset.iloc[:, 0:3].values

G_data=dataset.iloc[:, 3:6].values
L=dataset.iloc[:, 6].values

A1=np.sqrt(np.sum(np.square(A_data), axis=1))
G1=np.sqrt(np.sum(np.square(G_data), axis=1))

# Creating windows of specific size with 1 stride

#def subsequences(ts, window):
#    shape = (ts.size - window + 1, window)
#    strides = ts.strides * 2
#    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)
#window=6
#A1_new=subsequences(A1,window)
#

# Creating a non-overlapping Array of 128 samples
samples = list()
samples2=list()
samples3=list()
length = 128
n=(419264-128)
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
	# grab from i to i + 200
	sample = A1[i:i+length]
	sample2 = G1[i:i+length]
	sample3 = L[i:i+length]    
	samples.append(sample)
	samples2.append(sample2)
	samples3.append(sample3)
print(len(samples))
A_windowed = np.array(samples)
G_windowed = np.array(samples2)
Label=np.array(samples3)
L1=Label[:,0]

#Function for splitting with overlap
#def split_with_overlap(seq, length, overlap):
#    return [seq[i:i+length] for i in range(0, len(seq), length - overlap)]
#A2=split_with_overlap(A1, 128, 20)
#print(len(A2))

#def hist_per_row(data, bins):
#
#    data = np.asarray(data)
#    assert np.all(bins[:-1] <= bins[1:])
#    r, c = data.shape
#
#    nbins = len(bins)-1
#    data = data/bins[-1]
#    idx = array(data*nbins, dtype=int)+1
#
#    step = len(bins) + 1
#    last = step * r
#    idx += np.arange(0, last, step).reshape((r, 1))
#    res = np.bincount(idx.ravel(), minlength=last)
#    res = res.reshape((r, step))
#    return res[:, 1:-1]
#
#x1=hist_per_row(A_windowed,8)
bins=2
A_histogram=np.apply_along_axis(lambda x: np.histogram(x, bins)[0], 1, A_windowed)
G_histogram=np.apply_along_axis(lambda x: np.histogram(x, bins)[0], 1, G_windowed)

#Calculating Probabilities
A_Prob=A_histogram/128;
G_Prob=G_histogram/128;


# Entropy Calculation from histogram
A_Prob.sum(axis=1)
from scipy.special import entr
A_entropy = entr(A_Prob).sum(axis=1)
G_entropy = entr(G_Prob).sum(axis=1)
#Simple Calculation
#A1_entropy=(-A_Prob*np.log2(A_Prob)).sum(axis=1)

import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(20, 10), dpi=120, facecolor='w', edgecolor='k')
t = np.linspace(0, 200, 200)
plt.plot(t, A_entropy[0:200], 'g', linewidth=2.0) # plotting t, a separately 
plt.plot(t, G_entropy[0:200], 'r',linewidth=1.7) # plotting t, b separately 
plt.plot(t,Label[0:200]/10, 'b') # plotting t, c separately 
plt.title('Entropy change with Activity Classes',fontsize=18)
plt.ylabel('Entyopy Values[Arbitrary Unit]',fontsize=18)
plt.xlabel('Sampling Points',fontsize=18)
plt.legend(['Acc', 'Gyro','Label'], loc='lower left',fontsize=14)
plt.show()
plt.savefig('Entropy_2_bins.jpg')

