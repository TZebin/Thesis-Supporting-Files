# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:15:08 2016

@author: tahmina.zebin@manchester.ac.uk
"""

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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#from keras.layers import BatchNormalization
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D

classifier = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 6)))
# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 30,dropout=0.2, recurrent_dropout=0.2))
# Adding the output layer
classifier.add(Dense(units = 15,kernel_initializer = 'uniform', activation = 'relu'))

#classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the RNN
from keras.optimizers import adam
opt=adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
classifier.compile(optimizer =opt, loss = 'binary_crossentropy', metrics=['accuracy'])

print(classifier.summary())
#LSTM+CNN

#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(BatchNormalization())

# Fitting the RNN to the Training set
#regressor.fit(X_train, y_train, epochs = 10, batch_size = 50)
# Fitting the RNN to the Training set

# Fitting the RNN to the Training set
#regressor.fit(X_train, y_train, epochs = 10, batch_size = 50) 
# Fitting the RNN to the Training set
Tic=time.time()
history=classifier.fit(X_train, label_Tr,validation_split=0.2, epochs = 120, batch_size = 50)

toc=time.time()-Tic
import matplotlib.pyplot as plt

print(history.history.keys())
acc_Batch2=[ history.history['acc'], history.history['val_acc'],history.history['loss'],history.history['val_loss']]


# list all data in history
#print(history.history.keys())
#import pickle
#with open('/trainHistoryDict', 'wb') as file_pi:
#        pickle.dump(history.history, file_pi)
# summarize history for accuracy

import itertools
plt.figure(figsize=(8, 6), dpi=600)

plt.rcParams['font.family'] = 'Times new Roman'

plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'
        
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], '--')


plt.title('Model accuracy Generic LSTM')
plt.ylabel('Multi-Class Average Accuracy [0-1]')
plt.xlabel('No. of Epoch')# summarize history for loss
plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='lower right')
plt.tight_layout()
plt.show()

plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


plt.rcParams['font.family'] = 'Times new Roman'

plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss(Cross-Entropy)')
plt.ylabel('Loss[arbitrary Unit]')
plt.xlabel('No. of Epoch')
plt.legend(['Training Acuracy', 'validation Accuracy'], loc='lower right')
plt.show()
from sklearn.metrics import confusion_matrix,classification_report
 
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments

y_out=y_pred.argmax(1)
y_label=label_Test.argmax(1)

cm = confusion_matrix(y_label, y_out)
print(classification_report(y_label, y_out))

#Save and load the model
from keras.models import model_from_json
#Serialize model to json
saved_model=classifier.to_json()
with open("classifier2.json","w") as json_file:
    json_file.write(saved_model)
#Serialize weights to HDF5
    classifier.save_weights('saved_weights.h5')
    

#Load json and create model
json_file=open('classifier2.json','r')
Loaded_model=json_file.read()
json_file.close()
Loaded=model_from_json(Loaded_model)
#Load weights into new model
Loaded.load_weights('saved_weights.h5')
import itertools
#loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
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
                      plt.ylabel('True Class')
                      plt.xlabel('Predicted Class')

# Compute confusion matrix

# Plot non-normalized confusion matrix
plt.figure(figsize=(8, 6), dpi=1200)
font = {'family' : 'Times new Roman',
        'size'   : 14}
plt.rc('font', **font)
class_names =np.array(['Walk_Level', 'Walk_up', 'Walk_Down','Sit','Stand', 'Lying flat'],dtype='<U10')
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix(No. Of Test Data Frames)')

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6), dpi=1200)
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




