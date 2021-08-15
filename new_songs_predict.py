# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 22:25:56 2021

@author: Simon
"""

from tensorflow import keras
import pandas as pd
import numpy as np
import os
from pathlib import Path
import music_tag

# read in the data from csv
df_crude = pd.read_json("C:/Users/Simon/Documents/projects/MusicLabel/data/pred_time_series.json")

# feature is in list of list format
# turn each list (mfcc_1, mfcc_2...etc.)
mfcc = df_crude['mfcc']
mfcc_d1 = df_crude['mfcc_d1']
mfcc_d2 = df_crude['mfcc_d2']

df = pd.concat([pd.DataFrame.from_records(mfcc, columns=["mfcc_" + str(i) for i in range(13)]),
               pd.DataFrame.from_records(mfcc_d1, columns=["mfcc_d1_" + str(i) for i in range(13)]),
               pd.DataFrame.from_records(mfcc_d2, columns=["mfcc_d2_" + str(i) for i in range(13)]),],
               axis=1)

# get the input shape from saved model (min length of songs in my playlist)
model = keras.models.load_model("C:/Users/Simon/Documents/projects/MusicLabel/models/cut_from_start_conv.h5")

min_length = model.input_shape[1]

df_start = df.applymap(lambda x: x[0:min_length])

# Convert the data into (nobs, each time series feature, n feature)
l = []
for col in df_start.columns:
    t = df_start.loc[:, col].to_numpy()
    l.append(np.concatenate(t).reshape(df_start.shape[0], len(df_start.iloc[0, 0])))

np_start = np.dstack(l)

pred = model.predict(np_start)
df_label = np.argmax(pred, axis=1)
df_crude['label'] = df_label

# write song title/artist and label to a csv file
df_crude.loc[:,['title', 'artist', 'album', 'label']].to_csv(
    'C:/Users/Simon/Documents/projects/MusicLabel/data/pred_output.csv')

# move file to respective subfolder for easier import into my phone/organize
path = "C:/Users/Simon/Documents/projects/MusicLabel/data/New Songs"
Path(path+"/S/").mkdir(parents=True, exist_ok=True)
Path(path+"/H/").mkdir(parents=True, exist_ok=True)

for file_name in os.listdir(path):
    # double-checking all file under this dir is music/only move the relavent files
    if any(substring in file_name for substring in ['m4a', 'mp3', 'wav', 'flac', 'alac', 'wma']):
        audio_path = path+'\\'+file_name
        tag = music_tag.load_file(audio_path)
        if (df_crude[df_crude['title']==str(tag['title'])]['label'] == 0).bool():
            Path(audio_path).rename(path+"/S/"+file_name)
        elif (df_crude[df_crude['title']==str(tag['title'])]['label'] == 1).bool():
            Path(audio_path).rename(path+"/H/"+file_name)

df_crude.groupby('label')['title'].count()