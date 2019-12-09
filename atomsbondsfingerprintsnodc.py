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
from keras.preprocessing.image import ImageDataGenerator
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



print("Keras: %s"%keras.__version__)
dX_train = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_train.npy")
dX_val = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_val.npy")
dX_test = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_test.npy")

dy_train_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dy_train_s.npy")
dy_val_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dy_val_s.npy")
dy_test_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dy_test_s.npy")

mX_train = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/mX_train.npy")
mX_val = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/mX_val.npy")
mX_test = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/mX_test.npy")

my_train_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_train_s.npy")
my_val_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_val_s.npy")
my_test_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_test_s.npy")

fX_train = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_train.npy")
fX_val = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_val.npy")
fX_test = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_test.npy")

fy_train_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fy_train_s.npy")
fy_val_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fy_val_s.npy")
fy_test_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fy_test_s.npy")

input_shape = dX_train.shape[1:]
input_img = Input(shape=input_shape)

def Inception0(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output


def Inception(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output

xd = Inception0(input_img)
od=int(xd.shape[1])
xd = MaxPooling2D(pool_size=(od,od), strides=(1,1))(xd)
xd = Flatten()(xd)
xd = keras.layers.Dropout(rate = 0.6)(xd)
outd = Dense(1)(xd)


inf = Input(shape = (2048,))
xf = Dense(512, activation = 'relu')(inf)
xf = keras.layers.Dropout(rate = 0.5)(xf)
xf = Dense(64, activation = 'linear')(xf)
outf = Dense(1)(xf)

x = concatenate([outf, outd])
out = Dense(1, activation = 'linear')(x)

model = Model(inputs=[inf, input_img], outputs=[out])

concat = 1
epochs = 10
learning_rate = .001
batch_size = 128

#Concatenate for longer epochs
fXt = np.concatenate([fX_train]*concat, axis=0)
fyt = np.concatenate([fy_train_s]*concat, axis=0)  

#Concatenate for longer epochs
dXt = np.concatenate([dX_train]*concat, axis=0)
dyt = np.concatenate([dy_train_s]*concat, axis=0)    

#Concatenate for longer epochs
mXt = np.concatenate([mX_train]*concat, axis=0)
myt = np.concatenate([my_train_s]*concat, axis=0)

steps_per_epoch = 10000/batch_size

optimizer = Adam(lr=learning_rate)
model.summary()
print(len(mX_train))
print(len(mX_val))
print(len(mX_test))

from keras.callbacks import Callback

class roc_callback(Callback):
    def __init__(self):
        self.x = [fXt, dXt]
        self.y = myt
        self.x_val = [fX_val, dX_val]
        self.y_val = my_val_s
        self.x_tes = [fX_test, dX_test]
        self.y_tes = my_test_s

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
        y_pred_tes = self.model.predict(self.x_tes)
        roc_test = roc_auc_score(self.y_tes, y_pred_tes)
        print('\rroc-auc: %s - roc-auc_val: % s- roc-auc_test: %s' % (str(round(roc,4)),str(round(roc_val,4)),str(round(roc_test,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


model.compile(loss='mse',optimizer=optimizer, metrics = ['acc'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=100, min_lr=1e-20, verbose=1)
rocauccalc = roc_callback()


start = time.time()
history = model.fit(x = [fXt, dXt], y = myt,
                                  epochs=epochs,
                                  validation_data=([fX_val, dX_val],my_val_s),
                                  callbacks=[reduce_lr, rocauccalc])
stop = time.time()
time_elapsed = stop - start

print("########################")
my_predict = model.predict([fX_test, dX_test])
my_predict = my_predict.reshape(len(my_predict))
print("ROC_AUC SCORE:", sklearn.metrics.roc_auc_score(my_test_s, my_predict))
print("MEAN ABSOLUTE ERROR:", np.mean(np.absolute(np.subtract(my_test_s, my_predict))))

