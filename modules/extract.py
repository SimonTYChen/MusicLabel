# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:03:52 2021

@author: Simon
"""

import numpy as np
import scipy.stats

# summarize features sampled (time-series format) so it can be used in models properly
class NewClass(object): pass

def summarize(freqs, prefix):

    # turn into dict and create pd.DataFrame from it (add prefix)
    mydict = NewClass()

    mydict.mean = np.mean(freqs, axis=0)
    mydict.std = np.std(freqs, axis=0)
    mydict.p10 = np.quantile(freqs, 0.1, axis=0)
    mydict.q1 = np.quantile(freqs, 0.25, axis=0)
    mydict.median = np.median(freqs, axis=0)
    mydict.q3 = np.quantile(freqs, 0.75, axis=0)
    mydict.p90 = np.quantile(freqs, 0.9, axis=0)
    mydict.maxv = np.amax(freqs, axis=0)
    mydict.iqr = scipy.stats.iqr(freqs, axis=0)
    mydict.skew = scipy.stats.skew(freqs, axis=0)
    mydict.kurt = scipy.stats.kurtosis(freqs, axis=0)
    mydict.mode = scipy.stats.mode(np.round(freqs,0), axis=0)[0][0]


    mydict = {prefix + str(key): val for key, val in mydict.__dict__.items()}

    return mydict

def summarize_series(series, prefix):

    mydict = {}
    freqs = series.T

    mean = np.mean(freqs, axis=0)
    for i in range(len(mean)):
        mydict[prefix + 'mean_' + str(i)] = mean[i]

    std = np.std(freqs, axis=0)
    for i in range(len(std)):
        mydict[prefix + 'std_' + str(i)] = std[i]

    p10 = np.quantile(freqs, 0.1, axis=0)
    for i in range(len(p10)):
        mydict[prefix + 'p10_' + str(i)] = p10[i]

    q1 = np.quantile(freqs, 0.25, axis=0)
    for i in range(len(q1)):
        mydict[prefix + 'q1_' + str(i)] = q1[i]

    median = np.median(freqs, axis=0)
    for i in range(len(median)):
        mydict[prefix + 'median_' + str(i)] = median[i]

    q3 = np.quantile(freqs, 0.75, axis=0)
    for i in range(len(q3)):
        mydict[prefix + 'q3_' + str(i)] = q3[i]

    p90 = np.quantile(freqs, 0.9, axis=0)
    for i in range(len(p90)):
        mydict[prefix + 'p90_' + str(i)] = p90[i]

    maxv = np.amax(freqs, axis=0)
    for i in range(len(maxv)):
        mydict[prefix + 'maxv_' + str(i)] = maxv[i]

    iqr = scipy.stats.iqr(freqs, axis=0)
    for i in range(len(iqr)):
        mydict[prefix + 'iqr_' + str(i)] = iqr[i]

    skew = scipy.stats.skew(freqs, axis=0)
    for i in range(len(skew)):
        mydict[prefix + 'skew_' + str(i)] = skew[i]

    kurt = scipy.stats.kurtosis(freqs, axis=0)
    for i in range(len(kurt)):
        mydict[prefix + 'kurt_' + str(i)] = kurt[i]

    mode = scipy.stats.mode(np.round(freqs,0), axis=0)[0][0]
    for i in range(len(mode)):
        mydict[prefix + 'mode_' + str(i)] = mode[i]

    return mydict