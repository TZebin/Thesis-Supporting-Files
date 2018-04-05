# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:00:34 2017
Comparative CNN, LSTM, LSTM CNN
https://github.com/bhimmetoglu/time-series-medicine/tree/master/HAR
"""

'''@author: tahmina.zebin@manchester.ac.uk
"""
'''
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
import time
#Load Signals and Labels
X_train=np.load('F:\\Supporting files\\Deep Learning Demo\\Neural_Networks_ANN_CNN\\Train.npy')
label_Tr=np.load('F:\\Supporting files\\Deep Learning Demo\\Neural_Networks_ANN_CNN\\label_Train.npy')
label_Test=np.load('F:\\Supporting files\\Deep Learning Demo\\Neural_Networks_ANN_CNN\\label_Test.npy')
X_test=np.load('F:\\Supporting files\\Deep Learning Demo\\Neural_Networks_ANN_CNN\\Test.npy')

#Exploratory data Analysis

print ("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_train.shape[0],
                                                                             X_train.shape[1],
                                                                             X_train.shape[2]))
print ("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_test.shape[0],
                                                                         X_test.shape[1],
                                                                         X_test.shape[2]))


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Mean value for each channel at each step
all_data = np.concatenate((X_train,X_test), axis = 0)
means_ = np.zeros((all_data.shape[1],all_data.shape[2]))
stds_ = np.zeros((all_data.shape[1],all_data.shape[2]))

for ch in range(X_train.shape[2]):
    means_[:,ch] = np.mean(all_data[:,:,ch], axis=0)
    stds_[:,ch] = np.std(all_data[:,:,ch], axis=0)
    
df_mean = pd.DataFrame(data = means_)
df_std = pd.DataFrame(data = stds_)
df_mean.hist()
plt.show()
df_std.hist()
plt.show()
import os

def standardize(train, test):
	""" Standardize data """

	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test

#def one_hot(labels, n_class = 6):
#	""" One-hot encoding """
#	expansion = np.eye(n_class)
#	y = expansion[:, labels-1].T
#	assert y.shape[1] == n_class, "Wrong number of labels!"
#
#	return y
##
#def get_batches(X, y, batch_size = 100):
#	""" Return a generator for batches """
#	n_batches = len(X) // batch_size
#	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
#
#	# Loop over batches and yield
#	for b in range(0, len(X), batch_size):
#		yield X[b:b+batch_size], y[b:b+batch_size]

#from sklearn import preprocessing
#Some channels have mean values near 1, most close to 0. Let's standardize them all
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
##from sklearn.model_selection import StratifiedShuffleSplit
#scaler = StandardScaler()
X_train, X_test = standardize(X_train, X_test)


# Check Mean value for each channel at each step
all_data = np.concatenate((X_train,X_test), axis = 0)
means_ = np.zeros((all_data.shape[1],all_data.shape[2]))
stds_ = np.zeros((all_data.shape[1],all_data.shape[2]))

for ch in range(X_train.shape[2]):
    means_[:,ch] = np.mean(all_data[:,:,ch], axis=0)
    stds_[:,ch] = np.std(all_data[:,:,ch], axis=0)
    
df_mean = pd.DataFrame(data = means_)
df_std = pd.DataFrame(data = stds_)
df_mean.hist()
plt.show()
#from sklearn.model_selection import train_test_split
#X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, label_Tr, 
#                                                stratify = label_Tr, random_state = 123)


#y_tr = one_hot(lab_tr)
#y_vld = one_hot(lab_vld)
#y_test = one_hot(label_Test)
#Hyper-parameters
#batch_size = 600       # Batch size
#seq_len = 128          # Number of steps
#learning_rate = 0.0001
#epochs = 120
#
#n_classes = 6
#n_channels = 6

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, BatchNormalization
#from keras.optimizers import SGD,Adam
from keras.utils import np_utils
#from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model = Sequential()
# (batch, 128, 6) --> (batch, 64, 12)
model.add(Convolution1D(filters=12, kernel_size=2, strides=1, padding='same', activation='relu',input_shape = (X_train.shape[1], 6)))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
#model.add(Flatten())
#model.add(Dropout(0.2))
# (batch, 64, 12) --> (batch, 32, 24)
model.add(Convolution1D(filters=24, kernel_size=2, strides=1, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
# (batch, 64, 24) --> (batch, 32, 48)
model.add(Convolution1D(filters=48, kernel_size=2, strides=1, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Convolution1D(filters=96, kernel_size=2, strides=1, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 50,kernel_initializer = 'uniform', activation = 'relu'))

#model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units = 6,kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the RNN
from keras.optimizers import adam
opt=adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer =opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
Tic=time.time()
history=model.fit(X_train, label_Tr,validation_split=0.2, epochs = 120, batch_size = 200)

toc=time.time()-Tic
import matplotlib.pyplot as plt
#import itertools
# list all data in history
print(history.history.keys())
# summarize history for accuracy aand Loss

fig1 =plt.figure(figsize=(7, 6), dpi=80)

print(history.history.keys())
acc_Batch2=[ history.history['acc'], history.history['val_acc'],history.history['loss'],history.history['val_loss']]


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], '--')
plt.title('Model accuracy CNN',fontsize=16)
plt.ylabel('Average Accuracy [0-1]',fontsize=16)
plt.xlabel('No. of Epoch',fontsize=16)# summarize history for loss
plt.rcParams['font.family'] = 'Times new Roman'


plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'large'
        
#font = {'family' : 'Times New Roman',
#        'size'   : 14
#        }
#plt.rc('font', **font)

plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='lower right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6), dpi=80)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],'--', linewidth=2.5)

plt.title('Model loss( Categorical-Cross-Entropy)',fontsize=16)
plt.ylabel('Loss[Arbitrary Unit]',fontsize=16)
plt.xlabel('No. of Epoch',fontsize=16)
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix,classification_report
 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments

y_out=y_pred.argmax(1)
y_label=label_Test.argmax(1)

cm = confusion_matrix(y_label, y_out)
print(classification_report(y_label, y_out))

##Save and load the model
#from keras.models import model_from_json
##Serialize model to json
#saved_model1=classifier.to_json()
#with open("classifier.json","w") as json_file:
#    json_file.write(saved_model1)
##Serialize weights to HDF5
#    classifier.save_weights('saved_weights2.h5')
#import itertools   
#
##Load json and create model
#json_file=open('classifier.json','r')
#Loaded_model=json_file.read()
#json_file.close()
#Loaded=model_from_json(Loaded_model)
##Load weights into new model
#Loaded.load_weights('saved_weights2.h5')

#loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
import itertools
"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`."""
def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Greens):
                      if normalize:
                          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                          print("Normalized confusion matrix")
                      else:
                          print('Confusion matrix')

                      print(cm)
                      plt.imshow(cm, interpolation='nearest', cmap=cmap)
                      plt.title(title)
                      plt.colorbar()
                      
#                      plt.plot(history.history['loss'])
#                      plt.plot(history.history['val_loss'],'--', linewidth=2.5)
                      tick_marks = np.arange(len(classes))
                      plt.xticks(tick_marks, classes, rotation=45)
                      plt.yticks(tick_marks, classes)           
                      fmt = '.2f' if normalize else 'd'
                      thresh = cm.max() / 2.
                      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                            plt.text(j, i, format(cm[i, j], fmt),
                                     horizontalalignment="center",
                                     color="white" if cm[i, j] > thresh else "black")
                        
                      plt.tight_layout()
                      plt.ylabel('True Class',fontsize=14)
                      plt.xlabel('Predicted Class', fontsize=14)

# Compute confusion matrix

# Plot non-normalized confusion matrix
plt.figure(figsize=(6, 6), dpi=80)
#font = {'family' : 'Times new Roman',
#        'size'   : 12}
#plt.rc('font', **font)

class_names =np.array(['Walk_Level', 'Walk_up', 'Walk_Down','Sit','Stand', 'Lying flat'],dtype='<U10')
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix(No. Of Test Data Frames)')
