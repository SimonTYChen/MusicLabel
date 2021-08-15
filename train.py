# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 04:36:50 2021

@author: Simon
"""

# Using all my available data in training - no splitting
# Train/test split test was done in exploratory work

from tensorflow import keras
import numpy as np
import pandas as pd

# read in data
h = pd.read_json('C:/Users/Simon/Documents/projects/MusicLabel/data/h_time_series.json')
s = pd.read_json('C:/Users/Simon/Documents/projects/MusicLabel/data/s_time_series.json')
df_raw = pd.concat([h, s], ignore_index=True)

# turn list of list into multiple time series list
# so that each mfcc/d1/d2 (13*3) is a column
mfcc = df_raw['mfcc']
mfcc_d1 = df_raw['mfcc_d1']
mfcc_d2 = df_raw['mfcc_d2']

df = pd.concat([pd.DataFrame.from_records(mfcc, columns=["mfcc_" + str(i) for i in range(13)]),
               pd.DataFrame.from_records(mfcc_d1, columns=["mfcc_d1_" + str(i) for i in range(13)]),
               pd.DataFrame.from_records(mfcc_d2, columns=["mfcc_d2_" + str(i) for i in range(13)]),
               df_raw['label']], axis=1)

# need to standardize length of input - need minimum length of music
min_length = df.loc[:, 'mfcc_0'].str.len().min()

# separate out the features and labels - feature is to be cut
df_x = df.loc[:, df.columns != 'label']
df_y = df.loc[:, df.columns == 'label'].values

# from exploratory work - use data from start and trim the end is better than
# cutting the songs from the middle to get them into the same length
df_start_x = df_x.applymap(lambda x: x[0:min_length])

# Convert the data into (nobs, each time series feature, n feature)
l = []
for col in df_start_x.columns:
    t = df_start_x.loc[:, col].to_numpy()
    l.append(np.concatenate(t).reshape(df_start_x.shape[0], len(df_start_x.iloc[0, 0])))

np_start_x = np.dstack(l)
np_start_x.shape

# random sort the data for validation
idx = np.random.RandomState(seed=792).permutation(len(np_start_x))
np_start_x = np_start_x[idx]
df_y = df_y[idx]

# define number of labels
num_classes = len(np.unique(df_y))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=np_start_x.shape[1:])

epochs = 200
batch_size = 32
model_path = "C:/Users/Simon/Documents/projects/MusicLabel/models/cut_from_start_conv.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        model_path,
        save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    np_start_x,
    df_y,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)