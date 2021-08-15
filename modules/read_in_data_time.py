# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 02:02:19 2021

@author: Simon
"""

import os
import librosa
import pandas as pd
import music_tag

def read_in_data_time(path, label):

    data = pd.DataFrame()

    for file_name in os.listdir(path):

        audio_path = path+'\\'+file_name
        tag = music_tag.load_file(audio_path)
        audio , sr = librosa.load(audio_path, sr=int(tag['#samplerate']))

        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        mfcc_d1 = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)

        mydict = dict(title=str(tag['title']), artist=str(tag['artist']),
                      album=str(tag['album']), sr=sr, mfcc=mfcc, mfcc_d1=mfcc_d1,
                      mfcc_d2=mfcc_d2, label=label)

        data = data.append(pd.DataFrame([mydict]), ignore_index=True)

    return data

#h_path = 'C:/Users/Simon/Documents/projects/MusicLabel/musicdata/H'
#df_h = read_in_data_time(path=h_path,label=1)
#df_h.to_json('C:/Users/Simon/Documents/projects/MusicLabel/musicdata/h_time_series.json')

#s_path = 'C:/Users/Simon/Documents/projects/MusicLabel/musicdata/S'
#df_s = read_in_data_time(path=s_path,label=0)
#df_s.to_json('C:/Users/Simon/Documents/projects/MusicLabel/musicdata/s_time_series.json')

# to handle dynamic length - pad by largest file in dsn.
# pad along centre perhaps
# can also experiment with cutting length (along centre probably best)


