#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model, save_model
from scipy.stats import pearsonr
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.backend import slice
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# ## Functions

# In[2]:


def load_data(station): ## Station in String
    flow = np.load('../usgsflow_'+station+'.npy')
    precip = np.load('../NLDAS_precip_'+station+'.npy')
    srad = np.load('../NLDAS_srad_'+station+'.npy')
    tmax = np.load('../NLDAS_tmax_'+station+'.npy')
    y = np.array(flow).reshape(-1, 1)
    indx = np.where(y>=0)[0]
    # print(precip.shape)
    date = np.load('../usgsdate_'+station+'.npy', allow_pickle=True)
    x = np.concatenate((precip, srad, tmax), axis=1)
    # x = np.concatenate((precip, srad), axis=1)
    return x, y
def nse(y_pred, y_true):
    nse = 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)
    return nse
def dataset_ld(x,y,W,L):
    obs = x.shape[0]
    features = x.shape[1]
    a = np.zeros([obs-W-L+1, W, features])
    b = np.zeros([obs-W-L+1, 1])
    for i in range(obs-W-L+1):
        a[i,:,:] = x[i:i+W,:]
        b[i,:] = y[i+W+L-1,0]    
    return a, b
def train_test_pre(x, y):
    xtrain = x[:10000]; xtest = x[10000:]
    ytrain = y[:10000]; ytest = y[10000:]
    xscale = StandardScaler().fit(xtrain)
    yscale = StandardScaler().fit(ytrain)
    Xtrain = xscale.transform(xtrain); Xtest = xscale.transform(xtest)
    Ytrain = yscale.transform(ytrain); Ytest = yscale.transform(ytest)
    return Xtrain, Xtest, Ytrain, Ytest, xscale, yscale
def custom_loss(y_true, y_pred):
    s1 = K.sum((y_pred-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)
    return s1


# In[3]:


def build_model(W,L):
    x_in = keras.layers.Input(shape=(W,3)) # Batch, Length, Dimension
    ## Block 1
    x_tp = keras.layers.Conv1D(kernel_size=7, filters=40, dilation_rate=1, padding='causal')(x_in)
    # x_tp = keras.layers.BatchNormalization()(x_tp)
    x_tp = keras.layers.Activation('relu')(x_tp)
    x_tp = keras.layers.Dropout(0.6)(x_tp)
    x_tp = keras.layers.Conv1D(kernel_size=7, filters=40, dilation_rate=1, padding='causal')(x_tp)
    # x_tp = keras.layers.BatchNormalization()(x_tp)
    x_tp = keras.layers.Activation('relu')(x_tp)
    x_tp = keras.layers.Dropout(0.6)(x_tp)
    ## add res for block 1
    x_res = keras.layers.Conv1D(kernel_size=1, filters=40, dilation_rate=1, padding='causal')(x_in)
    x_tp = keras.layers.Add()([x_tp, x_res])
    x_tp = keras.layers.Activation('relu')(x_tp)
    ## Block 2
    x_block1 = x_tp
    x_tp = keras.layers.Conv1D(kernel_size=7, filters=20, dilation_rate=6, padding='causal')(x_tp)
    # x_tp = keras.layers.BatchNormalization()(x_tp)
    x_tp = keras.layers.Activation('relu')(x_tp)
    x_tp = keras.layers.Dropout(0.6)(x_tp)
    x_tp = keras.layers.Conv1D(kernel_size=7, filters=20, dilation_rate=6, padding='causal')(x_tp)
    # x_tp = keras.layers.BatchNormalization()(x_tp)
    x_tp = keras.layers.Activation('relu')(x_tp)
    x_tp = keras.layers.Dropout(0.6)(x_tp)
    ## add res for block 2
    x_res = keras.layers.Conv1D(kernel_size=1, filters=20, dilation_rate=1, padding='causal')(x_block1)
    x_tp = keras.layers.Add()([x_tp, x_res])
    # x_tp = keras.layers.Add()([x_tp, x_block1])
    x_tp = keras.layers.Activation('relu')(x_tp)    
    ## Block 3
    x_block2 = x_tp
    x_tp = keras.layers.Conv1D(kernel_size=7, filters=20, dilation_rate=12, padding='causal')(x_tp)
    # x_tp = keras.layers.BatchNormalization()(x_tp)
    x_tp = keras.layers.Activation('relu')(x_tp)
    x_tp = keras.layers.Dropout(0.6)(x_tp)
    x_tp = keras.layers.Conv1D(kernel_size=7, filters=20, dilation_rate=12, padding='causal')(x_tp)
    # x_tp = keras.layers.BatchNormalization()(x_tp)
    x_tp = keras.layers.Activation('relu')(x_tp)
    x_tp = keras.layers.Dropout(0.6)(x_tp)
    ## add res for block 3
    # x_res = keras.layers.Conv1D(kernel_size=1, filters=20, dilation_rate=1, padding='causal')(x_block2)
    # x_tp = keras.layers.Add()([x_tp, x_res])
    x_tp = keras.layers.Add()([x_tp, x_block2])
    x_tp = keras.layers.Activation('relu')(x_tp)
    ## SLICE
    x_tp = keras.layers.Lambda(lambda x:slice(x,(0,80,0),(-1,-1,-1)))(x_tp) # batch, length, channels 
    x_tp = keras.layers.Flatten()(x_tp)
    x_tp = keras.layers.Dropout(0.5)(x_tp)
    x_tp = keras.layers.Dense(100, activation='relu')(x_tp)
    x_tp = keras.layers.Dropout(0.5)(x_tp)
    x_out = keras.layers.Dense(1)(x_tp)
    model = Model(inputs=x_in, outputs=x_out)
    return model


# ## Constants

# In[4]:


lr = 0.0005; W=365; L=0;
f = open('../../StationArea.pkl','rb')
areas = pickle.load(f); f.close()


# In[ ]:


stations = np.load('../first-stations.npy')
d_nse = np.zeros((15,20)); d_mse = np.zeros((15,20)); d_mae = np.zeros((15,20)); d_r = np.zeros((15,20));
high_mse = np.zeros((15,20))
low_mse = np.zeros((15,20))
i=0
for station in stations:
    x, y = load_data(str(station))
    area = areas[str(station)]
    ## Transform to Runoff
    y = y*86400*1000/(area*1000*1000)
    Xtrain, Xtest, Ytrain, Ytest, xscale, yscale = train_test_pre(x, y)
    X_train, Y_train = dataset_ld(Xtrain, Ytrain, W, L)
    X_test, Y_test = dataset_ld(Xtest, Ytest, W, L)
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=7)
    a_nse = []; a_mse = []; a_r = []; a_mae = []; total_time=0
    a_high_mse=[]; a_low_mse=[]
    best_nse = -100; model_name='CNN/'+str(station)+'_DCNN.h5' ## Save the best nse and best model. 
    for training_id in range(15):
        ensemble_name = 'CNN/'+str(station)+'_CNN_'+str(training_id)+'.h5'
        model = build_model(W,L)
        adam = keras.optimizers.Adam(lr=lr)
        # model.compile(loss='mse', optimizer=adam)
        model.compile(loss=custom_loss, optimizer=adam)
        
        # training
        start = time.time()
        history = model.fit(X_train, Y_train, epochs=150, batch_size=512, 
                            verbose=0, shuffle=True)
        run_time = time.time()-start
        total_time+=run_time
        # testing:
        Y_pred = model.predict(X_test)
        y_pred = yscale.inverse_transform((Y_pred).reshape(-1, 1))
        y_true = yscale.inverse_transform((Y_test).reshape(-1, 1))
        NSE = nse(y_pred, y_true); 
        R = pearsonr(y_pred.flatten(), y_true.flatten())[0]
        MSE = mean_squared_error(y_pred, y_true); MAE = mean_absolute_error(y_pred, y_true) 
        a_nse+=[NSE]; a_mse+=[MSE]; a_r+=[R]; a_mae+=[MAE]
        ## Save the best nse and best model. 
        if (NSE>best_nse):
            best_nse=NSE; best_model=model; print('better')
        ## High flow and low flow
        ind = np.argwhere(y_true<=np.percentile(y_true, 5))
        low_pred = y_pred[ind]
        low_true = y_true[ind]
        ind = np.argwhere(y_true>=np.percentile(y_true, 95))
        high_pred = y_pred[ind]
        high_true = y_true[ind]
        
        e = high_pred-high_true; mse = np.mean(np.square(e)); a_high_mse+=[mse]
        e = low_pred-low_true; mse = np.mean(np.square(e)); a_low_mse+=[mse]
        
        del model
        del adam

    print(station,': run time is ', total_time/15, 's')
    print(best_nse)
    # print('NSE: ', a_nse, ' R: ', a_r)
    # print('MSE: ', a_mse, ' MAE: ', a_mae)
    d_nse[:,i]=a_nse; d_mse[:,i]=a_mse; d_mae[:,i]=a_mae; d_r[:,i]=a_r
    high_mse[:,i]=a_high_mse; low_mse[:,i]=a_low_mse;
    i+=1
    # model.save(model_name)
    best_model.save_weights(model_name)


# In[ ]:


np.save('CNN_NSE', d_nse);
np.save('CNN_MSE', d_mse);
np.save('CNN_high_mse', high_mse)
np.save('CNN_low_mse', low_mse)
print('done')

