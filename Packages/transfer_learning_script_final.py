import os
import numpy as np
import pandas as pd
import scipy
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split, KFold
from mllib.model import KerasSurrogate
from tensorflow.keras import layers
from tensorflow import keras
from helper_functions.scan_helper_functions_configs_MP import *
from helper_functions.MIE_NN.MieTFNew import PINNLossFunctionP11
#from helper_functions.ml_helper_functions import RSquaredSeparated, AdjustedRSquaredSeparated
#from helper_functions.invertible_neural_network import InvertibleNetworkSurrogate
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score,mean_absolute_percentage_error

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir')
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--replace_output', type=int)
args = parser.parse_args()
resultdirectory = args.result_dir
BatchSize = args.batch_size
Epochs = args.epochs
seedvalue = 42
model_name = 'forward_model'
model_dir_improved =  resultdirectory
model_dir = '/data/project/general/aerosolretriev/aerosol_results_mp/Reproducing10k/Set01/models'

# +
train_datafile = '/data/user/ponts_m/DataSets/GODataSet/DS10000_532nm_GO_scaled.h5'
dvar_max = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/dvar_max_10k.npy'))
dvar_min = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/dvar_min_10k.npy'))
qoi_max = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/qoi_max_10k.npy'))
qoi_min = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/qoi_min_10k.npy'))


dvar_train_wo_scaling, dvar_val_wo_scaling, qoi_train_wo_scaling, qoi_val_wo_scaling = load_dataset_wo_split(train_datafile)

dvar_train = dvar_train_wo_scaling
dvar_val = dvar_val_wo_scaling

qoi_train = qoi_train_wo_scaling
qoi_val = qoi_val_wo_scaling


# -

def inverse_scaling(data, xmax, xmin):
    return data*(xmax-xmin)+xmin


# +
custom_objects = {
    'AdjustedRSquared': AdjustedRSquared,
    'AdjustedRSquared_custom': AdjustedRSquared_custom
}

kwargs = {
    'custom_objects': custom_objects,
    'compile': False,
}

surr = KerasSurrogate.load(model_dir, model_name, model_kwargs=kwargs)


# +
surr.model.summary()
surr.model.trainable = False

n_in = dvar_train.shape[1]
n_out = qoi_train.shape[1]
opt = keras.optimizers.Adam(learning_rate=3.3e-05)

# +
if args.replace_output == 1:
    
    surr.model.layers[1]

    inp_layer_nn = keras.layers.Input(shape = (n_in,))
    first_layer_nn = surr.model.get_layer(index=0)(inp_layer_nn)

    for i in range((len(surr.model.layers)-2)):
        first_layer_nn = surr.model.get_layer(index=i+1)(first_layer_nn)

    last_layer_nn = keras.layers.Dense(units = n_out, name = 'last_layer', activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seedvalue))(first_layer_nn)
    output_layer_nn = last_layer_nn

    final_model_nn = keras.models.Model(inputs=[inp_layer_nn], outputs=[output_layer_nn])

    final_model_nn.compile(loss = 'MSE',optimizer=opt, metrics=['MAE', 'MAPE'])

    final_model_nn.summary()

    final_model_nn_history = final_model_nn.fit(dvar_train, qoi_train, validation_data = (dvar_val,qoi_val), batch_size = BatchSize, epochs = Epochs)
    

if args.replace_output == 0:
    
    surr.model.layers[1]

    inp_layer_nn = keras.layers.Input(shape = (n_in,))
    first_layer_nn = surr.model.get_layer(index=0)(inp_layer_nn)

    for i in range((len(surr.model.layers)-1)):
        first_layer_nn = surr.model.get_layer(index=i+1)(first_layer_nn)

    last_layer_nn = keras.layers.Dense(units = n_out, name = 'last_layer', activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seedvalue))(first_layer_nn)
    output_layer_nn = last_layer_nn

    final_model_nn = keras.models.Model(inputs=[inp_layer_nn], outputs=[output_layer_nn])

    final_model_nn.compile(loss = 'MSE',optimizer=opt, metrics=['MAE', 'MAPE'])

    final_model_nn.summary()

    final_model_nn_history = final_model_nn.fit(dvar_train, qoi_train, validation_data = (dvar_val,qoi_val), batch_size = BatchSize, epochs = Epochs)


# +
if args.replace_output == 1:
    
    surr.model.layers[1]

    inp_layer_pinn = keras.layers.Input(shape = (n_in,))
    first_layer_pinn = surr.model.get_layer(index=0)(inp_layer_pinn)

    for i in range((len(surr.model.layers)-2)):
        first_layer_pinn = surr.model.get_layer(index=i+1)(first_layer_pinn)

    last_layer_pinn = keras.layers.Dense(units = n_out, name = 'last_layer', activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seedvalue))(first_layer_pinn)
    output_layer_pinn = keras.layers.concatenate([inp_layer_pinn, last_layer_pinn])

    final_model_pinn = keras.models.Model(inputs=[inp_layer_pinn], outputs=[output_layer_pinn])

    final_model_pinn.compile(loss=custom_loss_wrapper(batchsize=BatchSize),optimizer=opt,\
                        metrics=[AdjustedRSquared_custom(BatchSize, n_in)])

    final_model_pinn.summary()
    
    final_model_pinn_history = final_model_pinn.fit(dvar_train, qoi_train, validation_data = (dvar_val,qoi_val), batch_size = BatchSize, epochs = Epochs)
    
    
if args.replace_output == 0:
    
    surr.model.layers[1]

    inp_layer_pinn = keras.layers.Input(shape = (n_in,))
    first_layer_pinn = surr.model.get_layer(index=0)(inp_layer_pinn)

    for i in range((len(surr.model.layers)-1)):
        first_layer_pinn = surr.model.get_layer(index=i+1)(first_layer_pinn)

    last_layer_pinn = keras.layers.Dense(units = n_out, name = 'last_layer', activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seedvalue))(first_layer_pinn)
    output_layer_pinn = keras.layers.concatenate([inp_layer_pinn, last_layer_pinn])

    final_model_pinn = keras.models.Model(inputs=[inp_layer_pinn], outputs=[output_layer_pinn])

    final_model_pinn.compile(loss=custom_loss_wrapper(batchsize=BatchSize),optimizer=opt,\
                        metrics=[AdjustedRSquared_custom(BatchSize, n_in)])

    final_model_pinn.summary()
    
    final_model_pinn_history = final_model_pinn.fit(dvar_train, qoi_train, validation_data = (dvar_val,qoi_val), batch_size = BatchSize, epochs = Epochs)


# +
final_model_nn.save(f'{model_dir_improved}/forward_model_nn')
final_model_pinn.save(f'{model_dir_improved}/forward_model_pinn')

#save training history
with open(f'{model_dir_improved}/nn_history.pkl', 'wb') as f:
    pickle.dump(final_model_nn_history, f)
    
with open(f'{model_dir_improved}/pinn_history.pkl', 'wb') as f:
    pickle.dump(final_model_pinn_history, f)
    
with open(f'{model_dir_improved}/nn_history_history.pkl', 'wb') as f:
    pickle.dump(final_model_nn_history.history, f)
    
with open(f'{model_dir_improved}/pinn_history_history.pkl', 'wb') as f:
    pickle.dump(final_model_pinn_history.history, f)

# +
#predicting
qoi_pred_train_nn = final_model_nn.predict(dvar_train)
qoi_pred_val_nn = final_model_nn.predict(dvar_val)

qoi_pred_train_pinn = final_model_pinn.predict(dvar_train)[:,-5:]
qoi_pred_val_pinn = final_model_pinn.predict(dvar_val)[:,-5:]


#manually inverse preprocessing everything
qoi_pred_train_nn = inverse_scaling(qoi_pred_train_nn, qoi_max, qoi_min)
qoi_pred_val_nn = inverse_scaling(qoi_pred_val_nn, qoi_max, qoi_min)

qoi_pred_train_pinn = inverse_scaling(qoi_pred_train_pinn, qoi_max, qoi_min)
qoi_pred_val_pinn = inverse_scaling(qoi_pred_val_pinn, qoi_max, qoi_min)

qoi_train = inverse_scaling(qoi_train, qoi_max, qoi_min)
qoi_val = inverse_scaling(qoi_val, qoi_max, qoi_min)

#calculating metrics from inverse-preprocessed quantities
metrics_train_nn = calculate_metrics_custom(qoi_train, qoi_pred_train_nn, n_in)
metrics_val_nn = calculate_metrics_custom(qoi_val, qoi_pred_val_nn, n_in)

metrics_train_pinn = calculate_metrics_custom(qoi_train, qoi_pred_train_pinn, n_in)
metrics_train_abs_pinn = calculate_metrics_custom(qoi_train, tf.abs(qoi_pred_train_pinn), n_in)
        
metrics_val_pinn = calculate_metrics_custom(qoi_val, qoi_pred_val_pinn, n_in)
metrics_val_abs_pinn = calculate_metrics_custom(qoi_val, tf.abs(qoi_pred_val_pinn), n_in)

#saving results
with open(f'{model_dir_improved}/metrics_train_nn.pkl', 'wb') as f:
    pickle.dump(metrics_train_nn, f)
    
with open(f'{model_dir_improved}/metrics_val_nn.pkl', 'wb') as f:
    pickle.dump(metrics_val_nn, f)


with open(f'{model_dir_improved}/metrics_train_pinn.pkl', 'wb') as f:
    pickle.dump(metrics_train_pinn, f)
    
with open(f'{model_dir_improved}/metrics_val_pinn.pkl', 'wb') as f:
    pickle.dump(metrics_val_pinn, f)
    
with open(f'{model_dir_improved}/metrics_train_abs_pinn.pkl', 'wb') as f:
    pickle.dump(metrics_train_pinn, f)
    
with open(f'{model_dir_improved}/metrics_val_abs_pinn.pkl', 'wb') as f:
    pickle.dump(metrics_val_pinn, f)

