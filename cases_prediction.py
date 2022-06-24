# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:14:22 2022

@author: ANAS
"""

#%% Imports

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from library_cp import ModelSaving
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input

#%% Statics

TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'datasets',
                                  'cases_malaysia_train.csv')

TEST_DATASET_PATH = os.path.join(os.getcwd(), 'datasets',
                                 'cases_malaysia_test.csv')

SCALER_PATH = os.path.join(os.getcwd(), 'saved_models', 'mm_scaler.pkl')

LOG_PATH = os.path.join(os.getcwd(), 'logs')

PLOT_MODEL_PATH = os.path.join(os.getcwd(), 'statics', 'model-architecture.png')

MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')

#%% EDA

#%% Step 1: Data Loading

df_train = pd.read_csv(TRAIN_DATASET_PATH, index_col = 'date')
df_test = pd.read_csv(TEST_DATASET_PATH, index_col = 'date')


#%% Step 2: Data Inspection

df_train.info()

# Cases_new in object type > Convert to numeric
# Cluster only available in 2021

df_test.info()

df_test.isna().sum() # Got 1 NaNs in Cases_new
df_test.duplicated().sum() # No duplicates



#%% Step 3) Data Cleaning

clean_train = df_train.copy()
clean_test = df_test.copy()

# Dummies for comparison

# Convert object type to numerical

clean_train['cases_new'] = pd.to_numeric(clean_train['cases_new'],
                                         errors='coerce')

 
clean_train.info()
# check conversion

# We inspect after change to numeric

clean_train.isna().sum() # 12 Nans in cases_new



# Impute using 
clean_train['cases_new'].interpolate(method='linear', inplace=True)
clean_test['cases_new'].interpolate(method='linear', inplace=True)

clean_train.duplicated().sum() # 10 duplicates

# Even though there are duplicates we decide not to drop as we are dealing TS

# Get the target data
train_data = clean_train['cases_new'].values
test_data = clean_test['cases_new'].values

# Visualize continous data using line graph
for i in [train_data, test_data]:
    plt.figure()
    plt.plot(i)
    plt.show()

#%% Preprocessing

mm_scaler = MinMaxScaler()
scaled_x_train = mm_scaler.fit_transform(train_data.reshape(-1, 1))
scaled_x_test = mm_scaler.transform(train_data.reshape(-1, 1))

# save model for deployment
ms = ModelSaving()
ms.save_model( SCALER_PATH, mm_scaler)

X_train = []
y_train = []

win_size = 30 # shape

for i in range(win_size,np.shape(scaled_x_train)[0]):
    X_train.append(scaled_x_train[i-win_size:i,0])
    y_train.append(scaled_x_train[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []

for i in range(win_size,np.shape(scaled_x_test)[0]):
    X_test.append(scaled_x_test[i-win_size:i,0])
    y_test.append(scaled_x_test[i,0])

X_test = np.array(X_test)
y_test = np.array(y_test)

#%% 


model = Sequential ()
model.add(Input((np.shape(X_train)[1],1))) # input_length, #features
model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.summary()

model.compile(optimizer='adam',
              loss = 'mse',
              metrics='mape')

# %%% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

# %%% Model Training

X_train = np.expand_dims(X_train, -1)

hist = model.fit(X_train, y_train, epochs=30, batch_size=128,
                 callbacks=[tensorboard_callback, early_stopping_callback])

# %%% Save model and plot
plot_model(model, to_file=PLOT_MODEL_PATH)
model.save(MODEL_PATH)
#%%
hist.history.keys()

plt.figure()
plt.plot(hist.history['mape'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()

#%%

con_test = np.concatenate((scaled_x_train, scaled_x_test), axis=0)
con_test = con_test[-710:]

predicted = model.predict(np.expand_dims(X_test,axis=-1))


#%%
plt.figure()
plt.plot(y_test,'b',label='actual new cases')
plt.plot(predicted,'r', label ='predicted new cases')
plt.legend()
plt.show()

inversed_y_pred = mm_scaler.inverse_transform(predicted)
inversed_y_true = mm_scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure()
plt.plot((inversed_y_true),'b',label='new cases')
plt.plot((inversed_y_pred),'r', label ='new cases')
plt.legend()
plt.show()

test = mean_absolute_error(y_test, predicted)/sum(abs(y_test))*100
print("The mape for this model is:",test[0],"%")



