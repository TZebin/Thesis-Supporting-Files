# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Feat66.csv')
X = dataset.iloc[:, 1:67].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train,batch_size=128,epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments

y_out=y_pred.argmax(1)
y_label=y_test.argmax(1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_label, y_out)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#
#def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 128, epochs = 100)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, n_jobs = -1)
#mean = accuracies.mean()
#variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN

#from sklearn.model_selection import GridSearchCV
#
#def build_classifier(optimizer):
#    classifier = Sequential()
#    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
#    
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [25, 32],
#              'epochs': [100, 500],
#              'optimizer': ['adam', 'rmsprop']}
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#grid_search = grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_