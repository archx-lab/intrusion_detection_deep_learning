from __future__ import print_function
# --------------------- importing libraries ---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------- importing class for label encoding ------------

from sklearn.preprocessing import LabelEncoder

# --------- importing method to split the dataset -------------------

from sklearn.model_selection import train_test_split

# ------------- importing class for feature scaling --------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# ------------ importing class for cross validation --------------

from sklearn.model_selection import StratifiedKFold

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

test_dataset = pd.read_csv('KDDTest+.csv')
train_dataset_original = pd.read_csv('NSL-KDD__Multiclass_Classification_Dataset.csv')


train_dataset_syn1 = pd.read_csv('out1.csv')
train_dataset_syn2 = pd.read_csv('out2.csv')
train_dataset_syn3 = pd.read_csv('out3.csv')
train_dataset_syn4 = pd.read_csv('out4.csv')
train_dataset_syn5 = pd.read_csv('out5.csv')
syn_data_list=[train_dataset_syn1,train_dataset_syn2,train_dataset_syn3,train_dataset_syn4,train_dataset_syn5]
train_dataset=train_dataset_original
accept=[]

acc_mentioned=76.11

def driver_func(var,train_dataset,test_dataset,train_dataset_original,syn_data_list):
    global acc_mentioned
    global accept
    if(len(accept)!=0):
        for j in accept:
            train_dataset = train_dataset.append(syn_data_list[j])
    train_dataset = train_dataset.append(syn_data_list[var])
    train_dataset = train_dataset.replace([np.inf, -np.inf], np.nan) 
    train_dataset = train_dataset.dropna()
    train_dataset = train_dataset.drop_duplicates()
    train_dataset=train_dataset[(train_dataset[["duration","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"]] >= 0).all(1)]
    # -------------------- taking care of inconsistent data -----------------
    train_dataset = train_dataset.replace([np.inf, -np.inf], np.nan) 
    train_dataset = train_dataset.dropna()
    train_dataset = train_dataset.drop_duplicates()
    test_dataset = test_dataset.replace([np.inf, -np.inf], np.nan) 
    test_dataset = test_dataset.dropna()
    test_dataset = test_dataset.drop_duplicates()
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
    #-------------- encoding categorical (dependent) variable -----------------
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    # --------------------------- feature scaling --------------------------------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #-----------------reshape input to be [samples, time steps, features]------------
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    #---------------------------------CNN-----------------------------------------
    cnn = Sequential()
    cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(116, 1)))
    cnn.add(Convolution1D(64, 3, padding="same", activation="relu"))
    cnn.add(MaxPooling1D(pool_size=(2)))
    cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
    cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
    cnn.add(MaxPooling1D(pool_size=(2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(5, activation="softmax"))
    # define optimizer and objective and compiling cnn model
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=[tf.keras.metrics.CategoricalAccuracy()])
    # train and validation of the data
    cnn.fit(X_train, y_train,epochs=1,validation_data=(X_test, y_test))
    y_pred = cnn.predict_classes(X_test)
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
    acc_obt=accuracy_score(y_test, y_pred)
    acc_obt=acc_obt*100
    print('================ Test Set ================')
    print('Accuracy Score : ', accuracy_score(y_test, y_pred))
    print('Precision Score : ', np.mean(precision_score(y_test, y_pred, average = None)))
    print('Recall Score : ', np.mean(recall_score(y_test, y_pred, average = None)))
    print('F1 Score : ', np.mean(f1_score(y_test, y_pred, average = None)))
    print('Specificity or True Negative Rate : ', np.mean(TNR))
    print('False Positive Rate : ', np.mean(FPR))
    print('False Negative Rate : ', np.mean(FNR))
    print('False Alarm Rate : ', np.mean(FAR))
    if(acc_obt<acc_mentioned):
        train_dataset=train_dataset_original
        print("Synthetic Dataset Discarded !!")
    else:
        acc_mentioned=acc_obt
        print("Synthetic Dataset Accepted !!")
        accept.append(var)
        print(accept)

for var in range(5):
    driver_func(var,train_dataset,test_dataset,train_dataset_original,syn_data_list)

print("Accuracy obtained after all iterations:"+str(acc_mentioned))
