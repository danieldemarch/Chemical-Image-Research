import pickle
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import json
import time
import numpy as np
import random

from dl_util import *
from ml_util import *
import pickle
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import json
import time
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn
import sklearn.metrics

from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs

from keras.layers import Input, Conv2D, Dense, concatenate

import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

from keras import preprocessing

trx1 = np.load("f:/Users/Montague/Desktop/DStuff/trx1.npy")
try1 = np.load("f:/Users/Montague/Desktop/DStuff/try1.npy")
tex1 = np.load("f:/Users/Montague/Desktop/DStuff/tex1.npy")
tey1 = np.load("f:/Users/Montague/Desktop/DStuff/tey1.npy")

trx2 = np.load("f:/Users/Montague/Desktop/DStuff/trx2.npy")
try2 = np.load("f:/Users/Montague/Desktop/DStuff/try2.npy")
tex2 = np.load("f:/Users/Montague/Desktop/DStuff/tex2.npy")
tey2 = np.load("f:/Users/Montague/Desktop/DStuff/tey2.npy")

trx3 = np.load("f:/Users/Montague/Desktop/DStuff/trx3.npy")
try3 = np.load("f:/Users/Montague/Desktop/DStuff/try3.npy")
tex3 = np.load("f:/Users/Montague/Desktop/DStuff/tex3.npy")
tey3 = np.load("f:/Users/Montague/Desktop/DStuff/tey3.npy")

trx4 = np.load("f:/Users/Montague/Desktop/DStuff/trx4.npy")
try4 = np.load("f:/Users/Montague/Desktop/DStuff/try4.npy")
tex4 = np.load("f:/Users/Montague/Desktop/DStuff/tex4.npy")
tey4 = np.load("f:/Users/Montague/Desktop/DStuff/tey4.npy")

trx5 = np.load("f:/Users/Montague/Desktop/DStuff/trx5.npy")
try5 = np.load("f:/Users/Montague/Desktop/DStuff/try5.npy")
tex5 = np.load("f:/Users/Montague/Desktop/DStuff/tex5.npy")
tey5 = np.load("f:/Users/Montague/Desktop/DStuff/tey5.npy")

trainx = [trx1,trx2,trx3,trx4,trx5]
trainy = [try1,try2,try3,try4,try5]
tex = [tex1,tex2,tex3,tex4,tex5]
tey = [tey1,tey2,tey3,tey4,tey5]

print("DONEZO")

raw_cock_list = []

input_shape = tex1.shape[1:]
input_img = Input(shape=input_shape)

from keras.callbacks import Callback

class roc_callback(Callback):
    def __init__(self):
        self.x = trx1
        self.y = try1
        self.x_val = tex1
        self.y_val = tey1

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        raw_cock_list.append(roc_val)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def BlockA(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    
    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output

def ReductionA(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)
    
    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output

def BlockB(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (1, 7), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(16, (7, 1), padding='same', activation='relu')(tower_2)

    output = keras.layers.concatenate([tower_1, tower_2], axis=-1)
    return output 

def ReductionB(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)
    
    tower_4 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_4 = Conv2D(16, (3, 1), padding='same', activation='relu')(tower_4)
    tower_4 = Conv2D(16, (3, 1), padding='same', activation='relu')(tower_4)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=-1)
    return output

def BlockC(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (1, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(16, (3, 1), padding='same', activation='relu')(tower_2)

    
    output = keras.layers.concatenate([tower_1, tower_2], axis=-1)
    return output 

for i in range(2):
    xa = BlockA(input_img)
    xa = BlockA(xa)
    xa = BlockA(xa)
    xa = BlockA(xa)
    xa = ReductionA(xa)
    
    xb = BlockB(input_img)
    xb = BlockB(xb)
    xb = BlockB(xb)
    xb = BlockB(xb)
    xb = ReductionB(xb)
    
    xc = BlockC(input_img)
    xc = BlockC(xc)
    xc = BlockC(xc)
    xc = BlockC(xc)
        
    x = keras.layers.concatenate([xa, xb, xc], axis=-1)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[input_img], outputs=[out])
    
    model.summary()
    
    concat = 1
    epochs = 100
    learning_rate = .001
    batch_size = 32
    
    steps_per_epoch = 10000/batch_size

    optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    
    model.compile(loss='mse',optimizer=optimizer)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=100, min_lr=1e-20, verbose=1)
    rocauccalc = roc_callback()
    
    generator = ImageDataGenerator(rotation_range=360,
                                   fill_mode="constant",cval = 0,
                                   horizontal_flip=True, vertical_flip=True,data_format='channels_last',
                                   )
    
    g = generator.flow(trainx[i], trainy[i], batch_size=batch_size, shuffle=True)
    
    history = model.fit_generator(g,
                                      steps_per_epoch=len(trx1)//batch_size,
                                      epochs=epochs,
                                      validation_data=(tex[i],tey[i]),
                                      callbacks=[reduce_lr, rocauccalc])





