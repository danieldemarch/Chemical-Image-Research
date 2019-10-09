# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:41:49 2019

@author: Montague
"""
        
print("Keras: %s"%keras.__version__)
x = np.load("f:/Users/Montague/Desktop/DStuff/testingx.npy")
y = np.load("f:/Users/Montague/Desktop/DStuff/testingy.npy")

posx = []
negx = []

posy = []
negy = []

for i in range(len(x)):
    if (y[i] == 1):
        posx.append(x[i])
        posy.append(1)
    else:
        negx.append(x[i])
        negy.append(0)
    
def cmol(mol, embed=15.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,4))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            if (idx >= dims):
                return 0
            if (idy >= dims):
                return 0
            if (idx < 0):
                return 0
            if (idy < 0):
                return 0
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 1] = atom.GetAtomicNum()
            #Gasteiger Charges
            charge = atom.GetProp("_GasteigerCharge")
            vect[ idx , idy, 3] = charge
            #Hybridization
            hyptype = atom.GetHybridization().real
            vect[ idx , idy, 2] = hyptype
    return vect

def cmol2(mol, embed=15.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            if (idx >= dims):
                return 0
            if (idy >= dims):
                return 0
            if (idx < 0):
                return 0
            if (idy < 0):
                return 0
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 0] = atom.GetAtomicNum()
    return vect

epochs = 40
batch_size = 64
learning_rate = .0002
concat = 1

print(learning_rate, batch_size, epochs)

dposx = []
dnegx = []

dposy = []
dnegy = []

datax = []
datay = []

mposx = []
mnegx = []

mposy = []
mnegy = []

matax = []
matay = []

fposx = []
fnegx = []

fposy = []
fnegy = []

fatax = []
fatay = []

for i in range(len(posx)):
    mol = cmol2(posx[i])
    mac = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(posx[i]), mac)
    fin = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(posx[i], radius=2), fin)

    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        dposx.append(mol)
        dposy.append(1)
        mposx.append(mac)
        mposy.append(1)
        fposx.append(fin)
        fposy.append(1)
print(len(dposx))

for i in range(len(negx)):
    mol = cmol2(negx[i])
    mac = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(negx[i]), mac)
    fin = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(negx[i], radius=2), fin)
    if isinstance(mol, int):
        print(i)
    else:
        dnegx.append(mol)
        dnegy.append(0)
        mnegx.append(mac)
        mnegy.append(0)
        fnegx.append(fin)
        fnegy.append(0)

dpostrx = []
dposvax = []
dpostex = []

dnegtrx = []
dnegvax = []
dnegtex = []

dpostry = []
dposvay = []
dpostey = []

dnegtry = []
dnegvay = []
dnegtey = []

mpostrx = []
mposvax = []
mpostex = []

mnegtrx = []
mnegvax = []
mnegtex = []

mpostry = []
mposvay = []
mpostey = []

mnegtry = []
mnegvay = []
mnegtey = []

fpostrx = []
fposvax = []
fpostex = []

fnegtrx = []
fnegvax = []
fnegtex = []

fpostry = []
fposvay = []
fpostey = []

fnegtry = []
fnegvay = []
fnegtey = []

for i in range(len(mposx)):
    if (i < len(mposx)/5):
        mpostex.append(mposx[i])
        mpostey.append(1)
        fpostex.append(fposx[i])
        fpostey.append(1)
        dpostex.append(dposx[i])
        dpostey.append(1)
    elif (i < len(mposx)*.3):
        mposvax.append(mposx[i])
        mposvay.append(1)
        fposvax.append(fposx[i])
        fposvay.append(1)
        dposvax.append(dposx[i])
        dposvay.append(1)
    else:
        mpostrx.append(mposx[i])
        mpostry.append(1)
        fpostrx.append(fposx[i])
        fpostry.append(1)
        dpostrx.append(dposx[i])
        dpostry.append(1)
    print(i)
        
for i in range(len(mnegx)):
    if (i < len(mnegx)/5):
        mnegtex.append(mnegx[i])
        mnegtey.append(0)
        fnegtex.append(fnegx[i])
        fnegtey.append(0)
        dnegtex.append(dnegx[i])
        dnegtey.append(0)
    elif (i < len(mnegx)*.3):
        mnegvax.append(mnegx[i])
        mnegvay.append(0)
        fnegvax.append(fnegx[i])
        fnegvay.append(0)
        dnegvax.append(dnegx[i])
        dnegvay.append(0)
    else:
        mnegtrx.append(mnegx[i])
        mnegtry.append(0)
        fnegtrx.append(fnegx[i])
        fnegtry.append(0)
        dnegtrx.append(dnegx[i])
        dnegtry.append(0)
    print(i)


dX_train = np.array(dnegtrx+dpostrx*31)
dX_val = np.array(dnegvax+dposvax)
dX_test = np.array(dnegtex+dpostex)

dy_train_s = np.array(dnegtry+dpostry*31)
dy_val_s = np.array(dnegvay+dposvay)
dy_test_s = np.array(dnegtey+dpostey)

mX_train = np.array(mnegtrx+mpostrx*31)
mX_val = np.array(mnegvax+mposvax)
mX_test = np.array(mnegtex+mpostex)

my_train_s = np.array(mnegtry+mpostry*31)
my_val_s = np.array(mnegvay+mposvay)
my_test_s = np.array(mnegtey+mpostey)

fX_train = np.array(fnegtrx+fpostrx*31)
fX_val = np.array(fnegvax+fposvax)
fX_test = np.array(fnegtex+fpostex)

fy_train_s = np.array(fnegtry+fpostry*31)
fy_val_s = np.array(fnegvay+fposvay)
fy_test_s = np.array(fnegtey+fpostey)

np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_train.npy", dX_train)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_val.npy", dX_val)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_test.npy", dX_test)

np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dy_train_s.npy", dy_train_s)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dy_val_s.npy", dy_val_s)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dy_test_s.npy", dy_test_s)

np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/mX_train.npy", mX_train)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/mX_val.npy", mX_val)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/mX_test.npy", mX_test)

np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_train_s.npy", my_train_s)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_val_s.npy", my_val_s)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_test_s.npy", my_test_s)

np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_train.npy", fX_train)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_val.npy", fX_val)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_test.npy", fX_test)

np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fy_train_s.npy", fy_train_s)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fy_val_s.npy", fy_val_s)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fy_test_s.npy", fy_test_s)


    
dX_train = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_train.npy")
dX_val = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_val.npy")
dX_test = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/dX_test.npy")

fX_train = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_train.npy")
fX_val = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_val.npy")
fX_test = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/fX_test.npy")

my_train_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_train_s.npy")
my_val_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_val_s.npy")
my_test_s = np.load("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/my_test_s.npy")

fX_train = fX_train.reshape(len(fX_train), 2048, 1, 1)
fX_val = fX_val.reshape(len(fX_val), 2048, 1, 1)
fX_test = fX_test.reshape(len(fX_test), 2048, 1, 1)

dx = np.concatenate((dX_train,dX_val, dX_test))
fx = np.concatenate((fX_train,fX_val, fX_test))

y = np.concatenate((my_train_s,my_val_s,my_test_s))

kf = sklearn.model_selection.KFold(n_splits = 5)

dx5fold = kf.split(dx)
fx5fold = kf.split(fx)
y5fold = kf.split(y)

dtrain1 = np.concatenate((dx5fold[1],dx5fold[2],dx5fold[3],dx5fold[4]))
dtrain2 = np.concatenate((dx5fold[0],dx5fold[2],dx5fold[3],dx5fold[4]))
dtrain3 = np.concatenate((dx5fold[0],dx5fold[1],dx5fold[3],dx5fold[4]))
dtrain4 = np.concatenate((dx5fold[0],dx5fold[1],dx5fold[2],dx5fold[4]))
dtrain5 = np.concatenate((dx5fold[0],dx5fold[1],dx5fold[2],dx5fold[3]))

ftrain1 = np.concatenate((fx5fold[1],fx5fold[2],fx5fold[3],fx5fold[4]))
ftrain2 = np.concatenate((fx5fold[0],fx5fold[2],fx5fold[3],fx5fold[4]))
ftrain3 = np.concatenate((fx5fold[0],fx5fold[1],fx5fold[3],fx5fold[4]))
ftrain4 = np.concatenate((fx5fold[0],fx5fold[1],fx5fold[2],fx5fold[4]))
ftrain5 = np.concatenate((fx5fold[0],fx5fold[1],fx5fold[2],fx5fold[3]))

ytrain1 = np.concatenate((y5fold[1],y5fold[2],y5fold[3],y5fold[4]))
ytrain2 = np.concatenate((y5fold[0],y5fold[2],y5fold[3],y5fold[4]))
ytrain3 = np.concatenate((y5fold[0],y5fold[1],y5fold[3],y5fold[4]))
ytrain4 = np.concatenate((y5fold[0],y5fold[1],y5fold[2],y5fold[4]))
ytrain5 = np.concatenate((y5fold[0],y5fold[1],y5fold[2],y5fold[3]))

ftrain = [ftrain1,ftrain2,ftrain3,ftrain4,ftrain5]
dtrain = [dtrain1,dtrain2,dtrain3,dtrain4,dtrain5]
ytrain = [ytrain1,ytrain2,ytrain3,ytrain4,ytrain5]


input_shape = dX_train.shape[1:]

raw_cock_list = []

input_img = Input(shape=input_shape)

from keras.callbacks import Callback

class roc_callback(Callback):
    def __init__(self):
        self.x = [dX_train, fX_train]
        self.y = my_train_s
        self.x_val = [dX_val, fX_val]
        self.y_val = my_val_s
        self.x_tes = [dX_test, fX_test]
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
        roc_tes = roc_auc_score(self.y_tes, y_pred_tes)
        print('\rroc-auc: %s - roc-auc_val: %s - roc-auc_test: %s' % (str(round(roc,4)),str(round(roc_val,4)),str(round(roc_tes,4))),end=100*' '+'\n')
        raw_cock_list.append(roc_val)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
class roc_callback_noval(Callback):
    def __init__(self):
        self.x = [dtrain[i], ftrain[i]]
        self.y = ytrain[i]
        self.x_tes = [dx5fold[i], fx5fold[i]]
        self.y_tes = y5fold[i]

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
        raw_cock_list.append(roc_tes)
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


for i in range(5):
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
    
    inf = Input(shape = (2048,1, 1))
    xf = keras.layers.Reshape((2048,))(inf)
    outf = Dense(1, activation = 'relu')(xf)
    
    x = keras.layers.concatenate([xa, xb, xc], axis=-1)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.concatenate([x, xf])
    out = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[input_img, inf], outputs=[out])
    
    model.summary()
    
    concat = 1
    epochs = 10
    learning_rate = .0001
    batch_size = 32
        
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    
    seed = random.randint(0, 100)
    
    generator = ImageDataGenerator(rotation_range=360,
                                   fill_mode="constant",cval = 0,
                                   horizontal_flip=True, vertical_flip=True,data_format='channels_last',
                                   )
    stringgenerator = ImageDataGenerator(rotation_range=0,
                                   fill_mode="constant",cval = 0,
                                   horizontal_flip=False, vertical_flip=False,data_format='channels_last')
    ygenerator = ImageDataGenerator(rotation_range=0,
                                   fill_mode="constant",cval = 0,
                                   horizontal_flip=False, vertical_flip=False,data_format='channels_last')
        
    g = generator.flow(dtrain[i], ytrain[i], batch_size=batch_size, shuffle=True, seed=seed)
    gg = stringgenerator.flow(ftrain[i], ytrain[i], batch_size=batch_size, shuffle=True, seed=seed)
    
    def combinegenerator(gen1, gen2):
        while True:
            x1 = next(gen1)
            x2 = next(gen2)
            yield([x1[0], x2[0]], x1[1])
    
    ggg = combinegenerator(g, gg)
    model.compile(loss='mse',optimizer=optimizer)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05,patience=100, min_lr=1e-20, verbose=1)
    rocauccalc = roc_callback_noval()
    
    
    history = model.fit_generator(ggg,
                                  steps_per_epoch=len(dX_train)//batch_size,
                                  epochs=epochs,
                                  validation_data=([dx5fold[i], fx5fold[i]],y5fold[i]),
                                   callbacks=[reduce_lr, rocauccalc])




import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau
import numpy as np

from sklearn.metrics import roc_auc_score
import random

from keras.preprocessing.image import ImageDataGenerator
import sklearn


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

x = np.load("f:/Users/Montague/Desktop/DStuff/testingx.npy")
y = np.load("f:/Users/Montague/Desktop/DStuff/testingy.npy")

posx = []
negx = []

posy = []
negy = []

for i in range(len(x)):
    if (y[i] == 1):
        posx.append(x[i])
        posy.append(1)
    else:
        negx.append(x[i])
        negy.append(0)


def cmol2(mol, embed=15.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,1))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            if (idx >= dims):
                return 0
            if (idy >= dims):
                return 0
            if (idx < 0):
                return 0
            if (idy < 0):
                return 0
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 0] = atom.GetAtomicNum()
    return vect

posmaccx = []
negmaccx = []


posimgx = []
negimgx = []

posy = []
negy = []
for i in range(len(posx)):
    macc = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(posx[i]), macc)
    mol = cmol2(posx[i])
    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        posimgx.append(mol)
        posmaccx.append(macc)
        posy.append(1)
    
for i in range(len(negx)):
    macc = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(neg[i]), macc)
    mol = cmol2(negx[i])
    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        negimgx.append(mol)
        negmaccx.append(macc)
        negy.append(0)
    
    
posmaccx = np.array(posmaccx)
negmaccx = np.array(negmaccx)
posimgx = np.array(posimgx)
negimgx = np.array(negimgx)
posy = np.array(posy)
negy = np.array(negy)



print(len(negx))
print(len(posx))
print(len(negy))
print(len(posy))
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posmaccs.np", posmaccx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negmaccs.np", negmaccx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posimgs.np", posimgx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negimgs.np", negimgx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posy.np", posy)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negy.np", negy)