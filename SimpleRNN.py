import pandas as pd
col_names = ['Recency', 'Frequency', 'Monetary', 'Time', 'BloodinMarch2017']
# load dataset
pima = pd.read_csv("transfusion.data", header=None, names=col_names)
#split dataset in features and target variable
feature_cols = ['Recency', 'Frequency', 'Monetary', 'Time']
X = pima[feature_cols] # Features
y = pima.BloodinMarch2017 # Target variable
print(pima)

from sklearn.preprocessing import StandardScaler as SS
scaler = SS()
X = scaler.fit_transform(X)
pd.DataFrame(X).sample(5)
pd.DataFrame(y).sample(5)

from sklearn import preprocessing 
from keras.utils import to_categorical as category
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = category(y)
pd.DataFrame(y).sample(5)
pd.DataFrame(y).sample(5)
import numpy as np
X = np.reshape(X, (X.shape[0], X.shape[1],1))
print('X : ' + str(X.shape))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test


from keras.models import Sequential as Seq
model = Seq()


from keras.layers import SimpleRNN as SRNN
model.add(SRNN(units = 100, activation = "relu", use_bias = True, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(SRNN(units = 100, activation = "relu", use_bias = True, return_sequences = True))
model.add(SRNN(units = 100, activation = "relu", use_bias = True))

from keras.layers import Dense
from keras.layers import Activation
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

modelfit = model.fit(X_train,y_train,batch_size=64,epochs=10,verbose=1,validation_data=(X_test, y_test))

import matplotlib.pyplot as plot
f, newp = plot.subplots(2, 1, figsize=(12,5))
newp[0].plot(modelfit.modelfit['loss'], color='b', label="Training loss")
newp[0].plot(modelfit.modelfit['val_loss'], color='r', label="Validation loss",axes =newp[0])
newp[0].grid(color='black', linestyle='-', linewidth=0.25)
legend = newp[0].legend(loc='best', shadow=True)

newp[1].plot(modelfit.modelfit['acc'], color='b', label="Training accuracy")
newp[1].plot(modelfit.modelfit['val_acc'], color='r',label="Validation accuracy")
newp[1].grid(color='black', linestyle='-', linewidth=0.25)
legend = newp[1].legend(loc='best', shadow=True)