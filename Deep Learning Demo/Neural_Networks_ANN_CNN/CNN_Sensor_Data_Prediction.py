# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:50:17 2016

@author: mchijtz4
"""
from __future__ import print_function, division
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential

import pandas as pd

dataset=pd.read_csv('Series_with Label_20.csv')
timeseries = dataset.iloc[0:20000, 1:2].values
#dataset=pd.read_csv('Series_with Label_20.csv')
#total_set = dataset.iloc[0:20000, 1:2].values
#Labels= dataset.iloc[0:20000, 6:7].values
#
#training_set=total_set[0:16000,:]
#test_set=total_set[16000:20000,:]
#L_train=Labels[0:16000,:]
#L_test=Labels[0:16000,:]
#
##normalize/scale the data
#
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(training_set)

filter_length = 5
nb_filter = 4
window_size = 50

def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(nb_outputs, activation='linear'),     
        # For binary classification, change the activation to 'sigmoid'
        #For multi-class Classification change the activation function to 'sigmoid'
    ))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model

def make_timeseries_instances(timeseries, window_size):
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q

def evaluate_timeseries(timeseries, window_size):
   
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T     # Convert 1D vectors to 2D column vectors
nb_samples, nb_series = timeseries.shape
print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)
print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
model.summary()

X, y, q = make_timeseries_instances(timeseries, window_size)
print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q, sep='\n')
test_size = int(0.01 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
model.fit(X_train, y_train, nb_epoch=25, batch_size=2, validation_data=(X_test, y_test))

pred = model.predict(X_test)
print('\n\nactual', 'predicted', sep='\t')
for actual, predicted in zip(y_test, pred.squeeze()):
    print(actual.squeeze(), predicted, sep='\t')
print('next', model.predict(q).squeeze(), sep='\t')
    
def main():
    """Prepare input data, build model, evaluate."""
    np.set_printoptions(threshold=25)
    ts_length = 20000
    window_size = 50

    print('\nSimple single timeseries vector prediction')
    timeseries = np.arange(ts_length)                   # The timeseries f(t) = t
    evaluate_timeseries(timeseries, window_size)

    print('\nMultiple-input, multiple-output prediction')
    timeseries = np.array([np.arange(ts_length), -np.arange(ts_length)]).T      # The timeseries f(t) = [t, -t]
    evaluate_timeseries(timeseries, window_size)


if __name__ == '__main__':
    main()