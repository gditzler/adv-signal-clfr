#!/usr/bin/env python 

###############################################################################
# standard imports 
import argparse
import numpy as np
import os
import pickle
import sys

# keras imports 
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D
from keras.optimizers import adam
import tensorflow as tf

# models
from models import gvn_00

import matplotlib.pyplot as plt
import seaborn as sns
import keras

from sklearn.model_selection import train_test_split


###############################################################################
# build up the parser
parser = argparse.ArgumentParser(
    description = ('....\n'))
parser.add_argument('-i', '--input',
    help = 'input file',
    required = True)
parser.add_argument('-o', '--output',
    help = 'location of the output directory.',
    required = True)
parser.add_argument('-m', '--model',
    help = 'model: gvn00.',
    required = True)
ALGS = set(['gvn00', 'gvn01'])
parser.add_argument('-s', '--seed',
    help = 'random seed.',
    type = int,
    default = 1,
    required = False)
parser.add_argument('-e', '--epochs',
    help = 'number of epochs.',
    type = int,
    default = 100,
    required = False)
parser.add_argument('-b', '--batch',
    help = 'batch size.',
    type = int,
    default = 128,
    required = False)
args = parser.parse_args()


SEED = args.seed
TEST_SPLIT = 0.25
EPOCHS = args.epochs
BATCH_SIZE = args.batch

np.random.seed(args.seed)


###############################################################################
# make sure the file is there 
if not os.path.isfile(args.input):
    parser.error('Input file not found.')
# make sure the model specified is there
if args.model not in ALGS:
    parser.error('The model you specified does not exist.') 

# read in the pickle file that needs to be in the python3 format
# TODO - check this import against other more generic formats; however, we need 
#        to have the raw data and labelings. 
with open("data/gnu_data.pkl", 'rb') as f:
    loaded = pickle.load(f, encoding='latin1') 
x = loaded['X']      # data labels
lbl = loaded['lbl']  # labels - a list with tuples ('ModScheme', -dB)


###############################################################################
# DATA PREPROCESSING 
# 1) generate lists with the dBs and mod schemes 
# 2) get the unique mod schemes 
# 
y_db = []
y_type = []

for c_type, c_db in lbl:
    y_db.append(c_db)
    y_type.append(c_type)

y_db = np.array(y_db)
y_type = np.array(y_type)


# get the classes in the data set
mods = np.unique(y_type)  

# build up a labels vector with classes represented by indices
y = np.zeros((len(y_db),))
for n, m in zip(range(len(mods)), mods):
    y[np.where(np.array(y_type) == m)] = n

# split up the data into training and testing + encode the outputs  
# THESE SEEDS MUST REMAIIN THE SAME!!!!!
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=SEED)
db_train, db_test, type_train, type_test = train_test_split(y_db, y_type, test_size=0.5, random_state=SEED)

y_train = keras.utils.to_categorical(y_train)
y_test_hot = keras.utils.to_categorical(y_test)

in_shp = list(x_train.shape[1:])


###############################################################################
if args.model == 'gvn00':
    model = gvn_00(len(mods), in_shp)


model.fit(x_train, y_train,
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          verbose = 1,
          validation_data = (x_test, y_test_hot))


###############################################################################
snrs = np.unique(y_db)
acc = {}
acc_mat = np.zeros((len(mods), len(snrs)))
for i, snr in zip(range(len(snrs)), snrs):
    for j,mod in zip(range(len(mods)), mods): 
        mod_types = set(np.where(type_test == mod)[0])
        snr_types = set(np.where(db_test == snr)[0])
        idx = list(snr_types.intersection(mod_types))
        
        x_test_rnd = x_test[idx].copy()
        y_test_rnd = y_test[idx].copy()
        y_i_hat = model.predict(x_test_rnd)
        
        y_classes = y_i_hat.argmax(axis=-1)
        acc_mat[j, i] = np.sum((y_classes == y_test_rnd)*1.0)/len(y_classes)

sty = ['rx','ro','rs','ko','kx','ks','co','cx','cs','bo','bx']
h = plt.figure()
for n in range(len(sty)): 
    plt.plot(snrs, acc_mat[n,:], c=sty[n][0], marker=sty[n][1], label=mods[n])

plt.legend(mods)


save_opts = {
    'acc_mat': acc_mat
}

