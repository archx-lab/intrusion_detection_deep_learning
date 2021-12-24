from ctgan import CTGANSynthesizer 
import pandas as pd
import numpy as np

X_train = pd.read_csv('NSL-KDD__Multiclass_Classification_Dataset.csv')
discrete_columns = ["protocol_type","service","flag","class_name"]
invalid_columns = set(discrete_columns) - set(X_train.columns)
if invalid_columns:
    raise ValueError('Invalid columns found: {}'.format(invalid_columns))


ctgan = CTGANSynthesizer()
ctgan.fit(X_train,discrete_columns,epochs=2)

#synthetic = [1] * 1000
#samples['Synthetic'] = synthetic
#file_out="out.csv"

samples = ctgan.sample(100000)
samples.to_csv("out1.csv",encoding="utf-8",index=False)
samples = ctgan.sample(100000)
samples.to_csv("out2.csv",encoding="utf-8",index=False)
samples = ctgan.sample(100000)
samples.to_csv("out3.csv",encoding="utf-8",index=False)
samples = ctgan.sample(100000)
samples.to_csv("out4.csv",encoding="utf-8",index=False)



