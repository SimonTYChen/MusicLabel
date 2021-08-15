# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 01:12:38 2021

@author: Simon
"""

# Prepare the prediction dataset
# Turn the musics into csv/pandas

# Run read in time with it, used saved model to predict,
# and output (move music to respective folder) and create a CSV with name + flag?

# Can os read in sub folders too?  test

from modules.read_in_mfcc import read_in_mfcc

path = 'C:/Users/Simon/Documents/projects/MusicLabel/New Songs'
df = read_in_mfcc(path=path)

# save a csv file in case for future
df.to_json('C:/Users/Simon/Documents/projects/MusicLabel/data/pred_time_series.json')
