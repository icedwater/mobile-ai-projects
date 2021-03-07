#! /usr/bin/env python3

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# read in housing.csv using pandas (github.com/PacktPublishing/Mobile-Artificial-Intelligence-Projects/)
dataframe = pd.read_csv("housing.csv", sep=',', header=0)
dataset = dataframe.values

# print(dataframe.head()) ## uncomment to show top of dataframe

# book uses a 70/30 split
features = dataset[:, 0:7]
target = dataset[:, 7]

## ANN bit starts here ##

def simple_shallow_seq_net():
    """
    Create a sequential ANN.
    """
    model = Sequential()
    model.add(Dense(7, input_dim=7, kernel_initializer="normal", activation="sigmoid"))
    model.add(Dense(1, kernel_initializer="normal"))
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss="mean_squared_error", optimizer=sgd)
    return model

## validation settings
seed = 5    ## fixed for reproducibility, do change it
numpy.random.seed(seed)
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
estimator = KerasRegressor(build_fn=simple_shallow_seq_net, epochs=100, batch_size=50, verbose=0)
results = cross_val_score(estimator, features, target, cv=kfold)
print(f"simple_shallow_seq_net model: {results.std():5.3} MSE")
