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
