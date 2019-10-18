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

posfingerx = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posfingers.np.npy")
negfingerx = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negfingers.np.npy")
posmaccx = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posmaccs.np.npy")
negmaccx = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negmaccs.np.npy")
posimgx = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posimgs.np.npy")
negimgx = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negimgs.np.npy")
posy = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posy.np.npy")
negy = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negy.np.npy")


from sklearn.model_selection import KFold

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

[shuffposmacc, shuffposimg] = unison_shuffled_copies(posmaccx, posimgx)
[shuffnegmacc, shuffnegimg] = unison_shuffled_copies(negmaccx, negimgx)

[posmacc1, posmacc2, posmacc3, posmacc4, posmacc5] = np.array_split(shuffposmacc, 5)
[negmacc1, negmacc2, negmacc3, negmacc4, negmacc5] = np.array_split(shuffnegmacc, 5)

[shuffposfinger, shuffposimg] = unison_shuffled_copies(posfingerx, posimgx)
[shuffnegfinger, shuffnegimg] = unison_shuffled_copies(negfingerx, negimgx)

[posfinger1, posfinger2, posfinger3, posfinger4, posfinger5] = np.array_split(shuffposfinger, 5)
[negfinger1, negfinger2, negfinger3, negfinger4, negfinger5] = np.array_split(shuffnegfinger, 5)

[posimg1a, posimg2a, posimg3a, posimg4a, posimg5a] = np.array_split(shuffposimg, 5)
[negimg1, negimg2, negimg3, negimg4, negimg5] = np.array_split(shuffnegimg, 5)

[posy1, posy2, posy3, posy4, posy5] = np.array_split(posy, 5)
[negy1, negy2, negy3, negy4, negy5] = np.array_split(negy, 5)

posy1 = np.tile(posy1, 31)
posy2 = np.tile(posy2, 31)
posy3 = np.tile(posy3, 31)
posy4 = np.tile(posy4, 31)
posy5 = np.tile(posy5, 31)

posimg1 = posimg1a
posimg2 = posimg2a
posimg3 = posimg3a
posimg4 = posimg4a
posimg5 = posimg5a
for i in range(30):
    posimg1 = np.concatenate((posimg1, posimg1a))
    posimg2 = np.concatenate((posimg2, posimg2a))
    posimg3 = np.concatenate((posimg3, posimg3a))
    posimg4 = np.concatenate((posimg4, posimg4a))
    posimg5 = np.concatenate((posimg5, posimg5a))

posmacc1 = np.tile(posmacc1, (31, 1))
posmacc2 = np.tile(posmacc2, (31, 1))
posmacc3 = np.tile(posmacc3, (31, 1))
posmacc4 = np.tile(posmacc4, (31, 1))
posmacc5 = np.tile(posmacc5, (31, 1))

posfinger1 = np.tile(posfinger1, (31, 1))
posfinger2 = np.tile(posfinger2, (31, 1))
posfinger3 = np.tile(posfinger3, (31, 1))
posfinger4 = np.tile(posfinger4, (31, 1))
posfinger5 = np.tile(posfinger5, (31, 1))

img1 = np.concatenate((posimg1, negimg1))
img2 = np.concatenate((posimg2, negimg2))
img3 = np.concatenate((posimg3, negimg3))
img4 = np.concatenate((posimg4, negimg4))
img5 = np.concatenate((posimg5, negimg5))

macc1 = np.concatenate((posmacc1, negmacc1))
macc2 = np.concatenate((posmacc2, negmacc2))
macc3 = np.concatenate((posmacc3, negmacc3))
macc4 = np.concatenate((posmacc4, negmacc4))
macc5 = np.concatenate((posmacc5, negmacc5))

finger1 = np.concatenate((posfinger1, negfinger1))
finger2 = np.concatenate((posfinger2, negfinger2))
finger3 = np.concatenate((posfinger3, negfinger3))
finger4 = np.concatenate((posfinger4, negfinger4))
finger5 = np.concatenate((posfinger5, negfinger5))

y1 = np.concatenate((posy1, negy1))
y2 = np.concatenate((posy2, negy2))
y3 = np.concatenate((posy3, negy3))
y4 = np.concatenate((posy4, negy4))
y5 = np.concatenate((posy5, negy5))

trainimg1 = np.concatenate((img2, img3, img4, img5))
trainimg2 = np.concatenate((img1, img3, img4, img5))
trainimg3 = np.concatenate((img1, img2, img4, img5))
trainimg4 = np.concatenate((img1, img2, img3, img5))
trainimg5 = np.concatenate((img1, img2, img3, img4))

trainmacc1 = np.concatenate((macc2, macc3, macc4, macc5))
trainmacc2 = np.concatenate((macc1, macc3, macc4, macc5))
trainmacc3 = np.concatenate((macc1, macc2, macc4, macc5))
trainmacc4 = np.concatenate((macc1, macc2, macc3, macc5))
trainmacc5 = np.concatenate((macc1, macc2, macc3, macc4))

trainfinger1 = np.concatenate((finger2, finger3, finger4, finger5))
trainfinger2 = np.concatenate((finger1, finger3, finger4, finger5))
trainfinger3 = np.concatenate((finger1, finger2, finger4, finger5))
trainfinger4 = np.concatenate((finger1, finger2, finger3, finger5))
trainfinger5 = np.concatenate((finger1, finger2, finger3, finger4))

trainy1 = np.concatenate((y2, y3, y4, y5))
trainy2 = np.concatenate((y1, y3, y4, y5))
trainy3 = np.concatenate((y1, y2, y4, y5))
trainy4 = np.concatenate((y1, y2, y3, y5))
trainy5 = np.concatenate((y1, y2, y3, y4))

def unison_shuffled_copies2(a, b, c, d):
    assert len(a) == len(b)
    assert len(b) == len(c)
    assert len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

[strainimg1, strainmacc1, strainfinger1, strainy1] = unison_shuffled_copies2(trainimg1, trainmacc1, trainfinger1, trainy1)
[strainimg2, strainmacc2, strainfinger2, strainy2] = unison_shuffled_copies2(trainimg2, trainmacc2, trainfinger2, trainy2)
[strainimg3, strainmacc3, strainfinger3, strainy3] = unison_shuffled_copies2(trainimg3, trainmacc3, trainfinger3, trainy3)
[strainimg4, strainmacc4, strainfinger4, strainy4] = unison_shuffled_copies2(trainimg4, trainmacc4, trainfinger4, trainy4)
[strainimg5, strainmacc5, strainfinger5, strainy5] = unison_shuffled_copies2(trainimg5, trainmacc5, trainfinger5, trainy5)

imgs = [strainimg1, strainimg2, strainimg3, strainimg4, strainimg5]
maccs = [strainmacc1, strainmacc2, strainmacc3, strainmacc4, strainmacc5]
fingers = [strainfinger1, strainfinger2, strainfinger3, strainfinger4, strainfinger5]
ys = [strainy1, strainy2, strainy3, strainy4, strainy5]

testimgs = [img1, img2, img3, img4, img5]
testmaccs = [macc1, macc2, macc3, macc4, macc5]
testfingers = [finger1, finger2, finger3, finger4, finger5]
testys = [y1, y2, y3, y4, y5]

input_shape = imgs[0].shape[1:]

test_raw_cock_list = []
train_raw_cock_list = []

input_img = Input(shape=input_shape)

for i in range(5):
    maccs[i] = maccs[i].reshape(len(maccs[i]), 167, 1, 1)
    testmaccs[i] = testmaccs[i].reshape(len(testmaccs[i]), 167, 1, 1)
    fingers[i] = fingers[i].reshape(len(fingers[i]), 2048, 1, 1)
    testfingers[i] = testfingers[i].reshape(len(testfingers[i]), 2048, 1, 1)

from keras.callbacks import Callback

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

for i in range(5):
    class roc_callback_noval(Callback):
        def __init__(self):
            self.x = imgs[i]
            self.y = ys[i]
            self.x_tes = testimgs[i]
            self.y_tes = testys[i]
    
        def on_train_begin(self, logs={}):
            return
    
        def on_train_end(self, logs={}):
            return
    
        def on_epoch_begin(self, epoch, logs={}):
            return
    
        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict(self.x)
            roc = roc_auc_score(self.y, y_pred)
            y_pred_tes = self.model.predict(self.x_tes)
            roc_tes = roc_auc_score(self.y_tes, y_pred_tes)
            print('\rroc-auc: %s - roc-auc_test: %s' % (str(round(roc,4)),str(round(roc_tes,4))),end=100*' '+'\n')
            test_raw_cock_list.append(roc_tes)
            train_raw_cock_list.append(roc)
            return
    
        def on_batch_begin(self, batch, logs={}):
            return
    
        def on_batch_end(self, batch, logs={}):
            return


    xa = BlockA(input_img)
    xa = BlockA(xa)
    xa = BlockA(xa)
    xa = BlockA(xa)
    xa = ReductionA(xa)
    
    xb = BlockB(xa)
    xb = BlockB(xb)
    xb = BlockB(xb)
    xb = BlockB(xb)
    xb = ReductionB(xb)
    
    xc = BlockC(xb)
    xc = BlockC(xc)
    xc = BlockC(xc)
    xc = BlockC(xc)
        
    x = keras.layers.concatenate([xa, xb, xc], axis=-1)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[input_img], outputs=[out])
    
    model.summary()
    
    concat = 1
    epochs = 50
    learning_rate = .001
    batch_size = 32
    
    steps_per_epoch = 10000/batch_size

    optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    
    model.compile(loss='mse',optimizer=optimizer)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, min_lr=1e-20, verbose=1)
    rocauccalc = roc_callback_noval()
    
    generator = ImageDataGenerator(rotation_range=360,
                                   fill_mode="constant",cval = 0,
                                   horizontal_flip=True, vertical_flip=True,data_format='channels_last',
                                   )
    
    g = generator.flow(imgs[i], ys[i], batch_size=batch_size, shuffle=True)
    
    history = model.fit_generator(g,
                                      steps_per_epoch=len(ys)//batch_size,
                                      epochs=epochs,
                                      validation_data=(testimgs[i],testys[i]),
                                      callbacks=[reduce_lr, rocauccalc])





