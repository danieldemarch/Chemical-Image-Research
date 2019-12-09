# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:41:19 2019

@author: Montague
"""
from __future__ import print_function
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

print("Keras: %s"%keras.__version__)
train_atoms_bonds = np.load("f:/Users/Montague/Desktop/DStuff/atombondstrainx.npy")
train_y = np.load("f:/Users/Montague/Desktop/DStuff/atombondstrainy.npy")
test_atoms_bonds = np.load("f:/Users/Montague/Desktop/DStuff/atombondstestx.npy")
test_y = np.load("f:/Users/Montague/Desktop/DStuff/atombondstesty.npy")
print("done")

postest = []
negtest = []
posty = []
negty = []

for i in range(len(test_atoms_bonds)):
    if (test_y[i] == 1):
        postest.append(np.reshape(test_atoms_bonds[i], (60, 120, 1)))
        posty.append(1)
    else:
        negtest.append(np.reshape(test_atoms_bonds[i], (60, 120, 1)))
        negty.append(0)
        
postrain = []
negtrain = []
posy = []
negy = []

for i in range(len(train_atoms_bonds)):
    if (train_y[i] == 1):
        postrain.append(np.reshape(train_atoms_bonds[i], (60, 120, 1)))
        posy.append(1)
    else:
        negtrain.append(np.reshape(train_atoms_bonds[i], (60, 120, 1)))
        negy.append(0)
negtrain = random.sample(negtrain, len(postrain))
negy = random.sample(negy, len(postrain))
Xt = negtrain+postrain
yt = posy+negy
yt = np.array(yt)
Xt = np.array(Xt)
    
negtest = random.sample(negtest, 99)
negty = random.sample(negty, 99)

x_test = postest+negtest
testy = posty+negty
x_test = np.array(x_test)
testy = np.array(testy)

model = keras.models.load_model('chemception_.h5')


y_predict = model.predict(x_test)
y_predict = np.rint(y_predict)
accuracy = accuracy_score(testy,y_predict)
f1 = f1_score(testy,y_predict)
precision = precision_score(testy,y_predict)
recall = recall_score(testy,y_predict)
roc_auc = roc_auc_score(testy,y_predict)
stats = { "accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "auc":roc_auc, "time":time_elapsed}
print("Test AUC:", auc)
print(stats)

y_predict = model.predict(Xt)
y_predict = np.rint(y_predict)

roc_auc = roc_auc_score(yt,y_predict)
print(roc_auc)