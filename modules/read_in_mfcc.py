# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 02:52:48 2021

@author: Simon
"""

import os
import librosa
import pandas as pd
import music_tag

def read_in_mfcc(path):

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
                      mfcc_d2=mfcc_d2)

        data = data.append(pd.DataFrame([mydict]), ignore_index=True)

    return data