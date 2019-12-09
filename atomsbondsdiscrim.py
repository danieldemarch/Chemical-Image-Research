from __future__ import print_function
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
print("RDKit: %s"%rdkit.__version__)
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

print("Keras: %s"%keras.__version__)
train_atoms_bonds = np.load("f:/Users/Montague/Desktop/DStuff/atombondstrainx.npy")
train_y = np.load("f:/Users/Montague/Desktop/DStuff/atombondstrainy.npy")
test_atoms_bonds = np.load("f:/Users/Montague/Desktop/DStuff/atombondstestx.npy")
test_y = np.load("f:/Users/Montague/Desktop/DStuff/atombondstesty.npy")
print("done")

X_train_val, X_test, y_train_val, y_test = sklearn.model_selection.train_test_split(train_atoms_bonds, train_y, test_size=0.2, random_state=1024)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=1024)

y_train_s, y_val_s, y_test_s = y_train, y_val, y_test


task = "classification"

input_shape = X_train.shape[1:]
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


generator = ImageDataGenerator(rotation_range=0,
                               width_shift_range=0.1,height_shift_range=0.1,
                               fill_mode="constant",cval = 0,
                               horizontal_flip=True, vertical_flip=True,data_format='channels_last',)

x = Inception0(input_img)
x = Inception(x)
x = Inception(x)
od=int(x.shape[1])
x = MaxPooling2D(pool_size=(od,od), strides=(1,1))(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_img, outputs=output)
    #Concatenate for longer epochs
Xt = np.concatenate([X_train]*concat, axis=0)
yt = np.concatenate([y_train_s]*concat, axis=0)

g = generator.flow(Xt, yt, batch_size=batch_size, shuffle=True)
steps_per_epoch = 10000/batch_size

optimizer = Adam(lr=learning_rate)
model.summary()
print(len(X_train))
print(len(X_val))
print(len(X_test))
model.compile(loss='mae',optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=6, min_lr=1e-6, verbose=1)

start = time.time()
history = model.fit_generator(g,
                                  steps_per_epoch=len(Xt)//batch_size,
                                  epochs=epochs,
                                  validation_data=(X_val,y_val_s),
                                  callbacks=[reduce_lr])
stop = time.time()
time_elapsed = stop - start

#name = "chemception_"+dataset+"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_learning_rate_"+str(learning_rate)
#model.save("%s.h5"%name)
#hist = history.history
#pickle.dump(hist, file("%s_history.pickle"%name,"w"))
print("########################")
#print("model and history saved",name)
print("########################")
y_predict = model.predict(X_test)
y_predict = y_predict.reshape(len(y_predict))
print(np.count_nonzero(y_predict))
print(sklearn.metrics.roc_auc_score(y_test_s, y_predict))
print(np.mean(np.absolute(np.subtract(y_test_s, y_predict))))
print(len(np.intersect1d(X_train, X_test)))
print(len(np.intersect1d(X_train, X_val)))
print(len(np.intersect1d(X_val, X_test)))

