# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:55:26 2018
https://github.com/smellslikeml/deepIDS/
@author: mchijtz4:tahmina.zebin@manchester.ac.uk
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#Data preparation Stage 
kdd_cols =np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","class","difficulty"])
kdd =pd.read_csv('G:\\NSL_KDD-master\\KDDTrain+.txt',sep=',',encoding= 'utf-8-sig',names =kdd_cols)
#train20_df=pd.read_csv('G:\\NSL_KDD-master\\KDDTrain+_20Percent.txt',sep=',',encoding= 'utf-8-sig',names =col_names)

kdd_t=pd.read_csv('G:\\NSL_KDD-master\\KDDTest+.txt',sep=',',encoding= 'utf-8-sig',names =kdd_cols)

# Here we opt for the 5-class problem
attack_map = {
    'normal': 'normal',
    
    'back': 'DoS','land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS','apache2': 'DoS', 'processtable': 'DoS','udpstorm': 'DoS',
    
    'ipsweep': 'Probe', 'nmap': 'Probe','portsweep': 'Probe', 'satan': 'Probe','mscan': 'Probe','saint': 'Probe',

    'ftp_write': 'R2L','guess_passwd': 'R2L','imap': 'R2L','multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L','warezclient': 'R2L','warezmaster': 'R2L','sendmail': 'R2L', 'named': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L','xlock': 'R2L','xsnoop': 'R2L', 'worm': 'R2L',
    
    'buffer_overflow': 'U2R','loadmodule': 'U2R','perl': 'U2R','rootkit': 'U2R','httptunnel': 'U2R','ps': 'U2R',    
    'sqlattack': 'U2R','xterm': 'U2R'
}
kdd['class'] = kdd['class'].replace(attack_map)
kdd_t['class'] = kdd_t['class'].replace(attack_map)

#Define functions cone-hot encoding and log scaling for variables
def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)

def log_trns(df, col):
    return df[col].apply(np.log1p)
# Convert protocol, service and flag to label encoded/dummy variables
cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = cat_encode(kdd, col)
    kdd_t = cat_encode(kdd_t, col)
# Apply log scale on duratiom, source byte and destination byte Variables    
log_lst = ['duration', 'src_bytes', 'dst_bytes']
for col in log_lst:
    kdd[col] = log_trns(kdd, col)
    kdd_t[col] = log_trns(kdd_t, col)
kdd_cols2= list(kdd.columns.values)   
kdd = kdd[kdd_cols2]
for col in kdd_cols2:
    if col not in kdd_t.columns:
        kdd_t[col] = 0
kdd_t = kdd_t[kdd_cols2]

kdd.head()

difficulty = kdd.pop('difficulty')
target = kdd.pop('class')
y_diff = kdd_t.pop('difficulty')
y_test = kdd_t.pop('class')

target = pd.get_dummies(target)
y_test = pd.get_dummies(y_test)
#Data Ready and Prepared for classification Algorithms
#Select training set and test set variables and features
target = target.values
train = kdd.values
test = kdd_t.values
y_test = y_test.values


# We rescale features to [0, 1]

min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

for idx, col in enumerate(list(kdd.columns)):
    print(idx, col)
 #Bluild the model with keras   
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout
from keras.layers.embeddings import Embedding

def build_network():

    model = Sequential()
    model.add(Dense(122, input_dim=122,kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(64,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32,activation = 'relu'))  
    model.add(Dropout(0.2))
    model.add(Dense(16,activation = 'relu'))
    model.add(Dense(5,kernel_initializer = 'uniform', activation = 'softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
NN = build_network()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
#Train  The model
history=NN.fit(x=train, y=target, epochs=100, validation_split=0.2, batch_size=128)
#NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])
#plot accuracy graph for different iterations
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 6), dpi=80)

print(history.history.keys())
accuracy=[ history.history['acc'], history.history['val_acc'],history.history['loss'],history.history['val_loss']]
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], '--')
plt.title('Model accuracy with Dense ANN',fontsize=16)
plt.ylabel('Average Accuracy [0-1]',fontsize=16)
plt.xlabel('No. of Epoch',fontsize=16)
plt.ylim((.97,1))
plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='lower right')
plt.tight_layout()
plt.show()

#plot confusion matrix

from sklearn.metrics import confusion_matrix,classification_report,roc_curve

# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model
preds = NN.predict(test)
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)

cm= confusion_matrix(true_lbls, pred_lbls)
NN.evaluate(test, y_test)
print(classification_report(true_lbls, pred_lbls))