from __future__ import print_function
# --------------------- importing libraries ---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- importing class for label encoding ------------

from sklearn.preprocessing import LabelEncoder

# --------- importing method to split the dataset -------------------

from sklearn.model_selection import train_test_split

# ------------- importing class for feature scaling --------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# ------------ importing class for cross validation --------------

from sklearn.model_selection import StratifiedKFold

# ------------ importing class for machine learning model ----------

"""###################################################""" 

# ----------- importing methods for performance evaluation ------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve

#---------------------- other imports --------------------------


import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# --------------------- importing dataset -----------------------

train_dataset = pd.read_csv('NSL-KDD__Multiclass_Classification_Dataset.csv')
test_dataset = pd.read_csv('KDDTest+.csv')



# -------------------- taking care of inconsistent data -----------------

train_dataset = train_dataset.replace([np.inf, -np.inf], np.nan) 
train_dataset = train_dataset.dropna()
train_dataset = train_dataset.drop_duplicates()
print('No of null values : ', train_dataset.isnull().sum().sum())


test_dataset = test_dataset.replace([np.inf, -np.inf], np.nan) 
test_dataset = test_dataset.dropna()
test_dataset = test_dataset.drop_duplicates()
print('No of null values : ', test_dataset.isnull().sum().sum())



# --------------- generalising different types of attacks in test set --------------------------

DoS = ['apache2','mailbomb','neptune','teardrop','smurf','pod','back','land','processtable']
test_dataset = test_dataset.replace(to_replace = DoS, value = 'DoS')

U2R = ['httptunnel','ps','xterm','sqlattack','rootkit','buffer_overflow','loadmodule','perl']
test_dataset = test_dataset.replace(to_replace = U2R, value = 'U2R')

R2L = ['udpstorm','worm','snmpgetattack','sendmail','named','snmpguess','xsnoop','xlock','warezclient','guess_passwd','ftp_write','multihop','imap','phf','warezmaster','spy']
test_dataset = test_dataset.replace(to_replace = R2L, value = 'R2L')

Probe = ['mscan','saint','ipsweep','portsweep','nmap','satan']
test_dataset = test_dataset.replace(to_replace = Probe, value = 'Probe')



# --------------- creating dependent variable vector ----------------

y_train = train_dataset.iloc[:, -2].values
y_test = test_dataset.iloc[:, -2].values

print(np.asarray(np.unique(y_train, return_counts=True)))
print(np.asarray(np.unique(y_test, return_counts=True)))



# --------------- onehotencoding the categorical variables with dummy variables ----------------

train_dataset.drop(train_dataset.iloc[:, [41, 42]], inplace = True, axis = 1) 
train_dataset = pd.get_dummies(train_dataset)
test_dataset.drop(test_dataset.iloc[:, [41, 42]], inplace = True, axis = 1) 
test_dataset = pd.get_dummies(test_dataset)



# ----- inserting columns of zeroes for categorical variables that are not common in train & test set -------

train_dataset.insert(78, "service_nnsp", 0)
train_dataset.insert(83, "service_pop_2", 0)
train_dataset.insert(85, "service_printer", 0)
train_dataset.insert(87, "service_remote_job", 0)
train_dataset.insert(88, "service_rje", 0)
train_dataset.insert(89, "service_shell", 0)
train_dataset.insert(97, "service_tftp_u", 0)



# ----------------- creating matrix of features ----------------

X_train = train_dataset.iloc[:, :].values
X_test = test_dataset.iloc[:, :].values



# -------------- encoding categorical (dependent) variable -----------------

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(np.asarray(np.unique(y_train, return_counts=True)))
print(np.asarray(np.unique(y_test, return_counts=True)))



# --------------------------- feature scaling --------------------------------

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#-----------------reshape input to be [samples, time steps, features]------------

X_train = np.reshape(X_train, (X_train.shape[0], 1,X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#---------------------------------RNN-----------------------------------------

model = Sequential()
model.add(SimpleRNN(4,input_dim=116))  
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Activation('softmax'))
'''
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)
np.savetxt('kddresults/lstm1layer/lstm1predicted.txt', np.transpose([y_test1,y_pred]), fmt='%01d')
'''
model.load_weights("results/lstm1layer_model.hdf5")

y_pred = model.predict_classes(X_test)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# --------------- making confusion matrix -----------------

cm = multilabel_confusion_matrix(y_test, y_pred)


# ---------------- performance evaluation using confusion matrix -----------------

for i in range(0, 5):    
    tn = cm[i][0][0]
    fn = cm[i][1][0]
    tp = cm[i][1][1]
    fp = cm[i][0][1]
    
    
    TNR = []
    FPR = []
    FNR = []
    FAR = []
    
    
    # ----------- calculating values for each class ------------------
    TNR.append(tn / (fp + tn))
    FPR.append(fp / (fp + tn))
    FNR.append(fn / (fn + tp))
    FAR.append((fp + fn) / (fp + fn + tp + tn))



# --------------- printing results of performance evaluation --------------------

print('================ Test Set ================')

print('Accuracy Score : ', accuracy_score(y_test, y_pred))

print('Precision Score : ', np.mean(precision_score(y_test, y_pred, average = None)))

print('Recall Score : ', np.mean(recall_score(y_test, y_pred, average = None)))

print('F1 Score : ', np.mean(f1_score(y_test, y_pred, average = None)))

print('Specificity or True Negative Rate : ', np.mean(TNR))

print('False Positive Rate : ', np.mean(FPR))

print('False Negative Rate : ', np.mean(FNR))

print('False Alarm Rate : ', np.mean(FAR))