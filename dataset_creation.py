
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

posfingerx = []
negfingerx = []

posimgx = []
negimgx = []

posy = []
negy = []
for i in range(len(posx)):
    macc = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(posx[i]), macc)
    mol = cmol2(posx[i])
    fin = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(posx[i], radius=2), fin)
    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        posimgx.append(mol)
        posmaccx.append(macc)
        posfingerx.append(fin)
        posy.append(1)
    
for i in range(len(negx)):
    macc = np.zeros(167)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(negx[i]), macc)
    mol = cmol2(negx[i])
    fin = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(negx[i], radius=2), fin)
    if isinstance(mol, int):
        print("dropped")
        print(i)
    else:
        negimgx.append(mol)
        negmaccx.append(macc)
        negfingerx.append(fin)
        negy.append(0)
    
    
posfingerx = np.array(posfingerx)
negfingerx = np.array(negfingerx)
posmaccx = np.array(posmaccx)
negmaccx = np.array(negmaccx)
posimgx = np.array(posimgx)
negimgx = np.array(negimgx)
posy = np.array(posy)
negy = np.array(negy)



print(len(negx))
print(len(posx))
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posmaccs.np", posfingerx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negmaccs.np", negfingerx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posmaccs.np", posmaccx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negmaccs.np", negmaccx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posimgs.np", posimgx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negimgs.np", negimgx)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/posy.np", posy)
np.save("f:/Users/Montague/Desktop/DStuff/atomsbondsdatasets/negy.np", negy)