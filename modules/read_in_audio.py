# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 01:13:29 2021

@author: Simon
"""

import os
import librosa
import pandas as pd
import music_tag
import extract

def read_in_data(path, label):

    data = pd.DataFrame()

    for file_name in os.listdir(path):

        audio_path = path+'\\'+file_name
        tag = music_tag.load_file(audio_path)
        audio , sr = librosa.load(audio_path, sr=int(tag['#samplerate']))

        mydict = dict(title=str(tag['title']), artist=str(tag['artist']),
                      album=str(tag['album']), sr=sr, label=label)

        # calculate features using librosa and summarize them
        # get mean/median...etc. from the time-series format
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_dict = extract.summarize_series(mfcc, 'mfcc_')

        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_d_dict = extract.summarize_series(mfcc_d, 'mfcc_d_')

        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        mfcc_d2_dict = extract.summarize_series(mfcc_d2, 'mfcc_d2_')

        rms = librosa.feature.rms(y=audio)
        rms_dict = extract.summarize_series(rms, 'rms_')

        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        cent_dict = extract.summarize_series(cent, 'cent_')

        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        bandwidth_dict = extract.summarize_series(bandwidth, 'bandwidth_')

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_dict = extract.summarize_series(contrast, 'contrast_')

        flatness = librosa.feature.spectral_flatness(y=audio)
        flatness_dict = extract.summarize_series(flatness, 'flatness_')

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_dict = extract.summarize_series(rolloff, 'rolloff_')

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_dict = extract.summarize_series(tonnetz, 'tonnetz_')

        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_dict = extract.summarize_series(zcr, 'zcr_')

        ons = librosa.onset.onset_strength(y=audio, sr=sr)
        ons_dict = extract.summarize(ons, 'ons_')

        dtempo = librosa.beat.tempo(y=audio, sr=sr, aggregate=None)
        dtempo_dict = extract.summarize(dtempo, 'dtempo_')

        features = [mfcc_dict, mfcc_d_dict, mfcc_d2_dict, rms_dict, cent_dict,
                    bandwidth_dict, contrast_dict, flatness_dict, rolloff_dict,
                    tonnetz_dict, zcr_dict, ons_dict, dtempo_dict]

        for d in features:
            mydict.update(d)

        data = data.append(pd.DataFrame([mydict]), ignore_index=True)

    return data

'''
h_path = 'C:/Users/Simon/Documents/projects/MusicLabel/musicdata/H'
df_h = read_in_data(path=h_path,label=1)
df_h.to_csv('C:/Users/Simon/Documents/projects/MusicLabel/musicdata/h_list.csv',
            encoding='utf-8')

s_path = 'C:/Users/Simon/Documents/projects/MusicLabel/musicdata/S'
df_s = read_in_data(path=s_path,label=0)
df_s.to_csv('C:/Users/Simon/Documents/projects/MusicLabel/musicdata/s_list.csv',
            encoding='utf-8')
'''

