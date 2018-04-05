# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:17:43 2017

@author: mchijtz4
"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import pandas as pd
dataset=pd.read_csv('Series_with Label_20.csv')

A_data=dataset.iloc[:, 0:3].values

G_data=dataset.iloc[:, 3:6].values
L=dataset.iloc[:, 6].values

A1=np.sqrt(np.sum(np.square(A_data), axis=1))
G1=np.sqrt(np.sum(np.square(G_data), axis=1))




total_set = dataset.iloc[0:20000, 1:2].values
Labels= dataset.iloc[0:20000, 6:7].values

total_set = dataset.iloc[0:20000, 1:2].values
total_set = dataset.iloc[0:20000, 1:2].values
#from sklearn.model_selection import train_test_split
#training_set, test_set,L_train, L_test = train_test_split( total_set,Labels, test_size = 0.2, random_state = 0)

training_set=total_set[0:16000,:]
test_set=total_set[16000:20000,:]
L_train=Labels[0:16000,:]
L_test=Labels[0:16000,:]


#normalize/scale the data

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

## Encoding categorical data
#from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)
#dummy_y = np_utils.to_categorical(encoded_Y)

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#Prepare dataset for LSTM that takes 64 previous measurements into consideration while making a decision
X_train = []
y_train = []
for i in range(64, 16000):
    X_train.append(training_set_scaled[i-64:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#Dropout regularization
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
#regressor.fit(X_train, y_train, epochs = 10, batch_size = 50)
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1, batch_size = 50)


inputs =total_set[len(total_set) - len(test_set) - 64:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(64, 4064):
    X_test.append(inputs[i-64:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_test_set = regressor.predict(X_test)
predicted_test_set = sc.inverse_transform(predicted_test_set)

# Visualising the results
import matplotlib.pyplot as plt
plt.plot(test_set, color = 'red', label = 'Real sensor data_Values')
plt.plot(predicted_test_set, color = 'blue', label = 'Predicted sensor data Values')
plt.title('Sensor data Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('ax m/s2')
plt.legend()
plt.show()

