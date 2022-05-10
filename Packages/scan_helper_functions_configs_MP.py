# -*- coding: utf-8 -*-
# + {}
import os
from helper_functions.MIE_NN.MieTFNew import PINNLossFunctionP11
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import numpy as np
from tensorflow import keras

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation

from keras.engine import data_adapter
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from ray import tune
from mllib.model import KerasSurrogate, AdaptiveMinMaxScaler, DummyPreprocessor, LogarithmTransform
from helper_functions.invertible_neural_network import InvertibleNetworkSurrogate


# -

class RSquared(tf.keras.losses.Loss):
    '''
    For more details, see:
    https://www.analyticsvidhya.com/blog/2020/07/difference-between-r-squared-and-adjusted-r-squared/
    '''
    def __init__(self):
        super().__init__(name='r2')

    def call(self, y_true, y_pred):
        mean_true = tf.math.reduce_mean(y_true, axis=0)

        total_sum_of_squares = tf.math.reduce_sum(tf.math.squared_difference(y_true, mean_true),
                                                  axis=0)

        

        residual_sum_of_squares = tf.math.reduce_sum(tf.math.squared_difference(y_true, y_pred),
                                                     axis=0)
        r2 = 1. - residual_sum_of_squares / (total_sum_of_squares)

        return tf.reduce_mean(r2)


class AdjustedRSquared_custom(tf.keras.losses.Loss):
    '''
    For more details, see:
    https://www.analyticsvidhya.com/blog/2020/07/difference-between-r-squared-and-adjusted-r-squared/
    '''
    def __init__(self, batch_size, number_of_input):
        '''
        Parameters
        ==========
        batch_size: int
            Number of samples in a batch,
            i. e. number of rows of a batch of the X matrix.
        number_of_input: int
            Number of independent variables (=columns) in the problem,
            i. e. number of columns of the X matrix.
        '''
        super().__init__(name='adjusted_r2')
        self._n = batch_size
        self._n_in = number_of_input

    def call(self, y_true, y_pred):
        y_pred = tf.math.abs(y_pred[:,-5:])
        #y_pred[:,-3] = tf.math.abs(y_pred[:,-3])
        #y_pred[:,-2] = tf.math.abs(y_pred[:,-2])
        
        r2 = RSquared().call(y_true, y_pred)

        adjusted_r2 = 1. - (1. - r2) * (self._n - 1.) / (self._n - self._n_in - 1.)
        adjusted_r2 = tf.math.reduce_mean(adjusted_r2)

        return adjusted_r2

    @classmethod
    def from_config(cls, config):
        return AdjustedRSquared_custom(config['n'], config['n_in'])

    def get_config(self):
        return {
            'n': self._n,
            'n_in': self._n_in,
        }


class AdjustedRSquared(tf.keras.losses.Loss):
    '''
    For more details, see:
    https://www.analyticsvidhya.com/blog/2020/07/difference-between-r-squared-and-adjusted-r-squared/
    '''
    def __init__(self, batch_size, number_of_input):
        '''
        Parameters
        ==========
        batch_size: int
            Number of samples in a batch,
            i. e. number of rows of a batch of the X matrix.
        number_of_input: int
            Number of independent variables (=columns) in the problem,
            i. e. number of columns of the X matrix.
        '''
        super().__init__(name='adjusted_r2')
        self._n = batch_size
        self._n_in = number_of_input

    def call(self, y_true, y_pred):
        r2 = RSquared().call(y_true, y_pred)

        adjusted_r2 = 1. - (1. - r2) * (self._n - 1.) / (self._n - self._n_in - 1.)
        adjusted_r2 = tf.math.reduce_mean(adjusted_r2)

        return adjusted_r2

    @classmethod
    def from_config(cls, config):
        return AdjustedRSquared(config['n'], config['n_in'])

    def get_config(self):
        return {
            'n': self._n,
            'n_in': self._n_in,
        }


def calculate_metrics_custom(qoi_true, qoi_pred, n_in):
    '''
    Parameters
    ==========
    qoi_true:
        True target vector / matrix.
    qoi_pred:
        Predicted target vector / matrix.
    n_in:
        Number of features in the X matrix.
    '''
    abs_residuals = np.abs(qoi_true - qoi_pred)
    rel_residuals = np.abs(abs_residuals / qoi_true) * 100.
    
    abs_residuals2 = abs_residuals*abs_residuals
    
    
    maximum_percentage_error = np.max(rel_residuals, axis=1)
    median = np.quantile(maximum_percentage_error, 0.5)
    ninety_percentile = np.quantile(maximum_percentage_error, 0.9)

    r2 = RSquared().call(qoi_true, qoi_pred)
    r2_adj = AdjustedRSquared_custom(qoi_true.shape[0], n_in).call(qoi_true, qoi_pred)
    
    wmape = np.sum(np.abs(qoi_true-qoi_pred))/np.sum(np.abs(qoi_true))*100


    return {
        'MAE': np.mean(abs_residuals),
        'MAPE': np.mean(rel_residuals),
        'MSE': np.mean(abs_residuals2),
        'wmape': wmape,
        'median_percentile_max_error': median,
        '90_percentile_max_error': ninety_percentile,
        'r2': r2.numpy(),
        'r2_adj': r2_adj.numpy(),
    }


def calculate_metrics(qoi_true, qoi_pred, n_in):
    '''
    Parameters
    ==========
    qoi_true:
        True target vector / matrix.
    qoi_pred:
        Predicted target vector / matrix.
    n_in:
        Number of features in the X matrix.
    '''
    abs_residuals = np.abs(qoi_true - qoi_pred)
    rel_residuals = np.abs(abs_residuals / qoi_true) * 100.
    
    abs_residuals2 = abs_residuals*abs_residuals
    
    
    maximum_percentage_error = np.max(rel_residuals, axis=1)
    median = np.quantile(maximum_percentage_error, 0.5)
    ninety_percentile = np.quantile(maximum_percentage_error, 0.9)

    r2 = RSquared().call(qoi_true, qoi_pred)
    r2_adj = AdjustedRSquared(qoi_true.shape[0], n_in).call(qoi_true, qoi_pred)
    
    wmape = np.sum(np.abs(qoi_true-qoi_pred))/np.sum(np.abs(qoi_true))*100


    return {
        'MAE': np.mean(abs_residuals),
        'MAPE': np.mean(rel_residuals),
        'MSE': np.mean(abs_residuals2),
        'median_percentile_max_error': median,
        '90_percentile_max_error': ninety_percentile,
        'r2': r2.numpy(),
        'r2_adj': r2_adj.numpy(),
        'wmape': wmape,
    }


def custom_loss_wrapper(batchsize):
    def custom_loss(y_true, y_pred):
        loss = PINNLossFunctionP11(y_pred, y_true, batchsize)
        return loss
    return custom_loss


class KerasSurrogate_custom(KerasSurrogate):
    def predict(self, X, **kwargs):
        X = self.preprocessor_x.transform(X)
        prediction = self._predict_model(X, **kwargs)
        prediction = prediction[:,-5:]
        return self.preprocessor_y.inverse_transform(prediction)



def build_forward_surrogate_custom(n_in, n_out, config): #dvar_train in die klammer hinzufügen
    # build the surrogate model
    n_units = [config['width']] * config['depth']
    input_layer = Input(shape = (n_in, ))
    hidden_layer = tf.keras.layers.GaussianNoise(config['x_noise'], input_shape=(n_in, ))(input_layer)
    
    for size in n_units:
        hidden_layer = Dense(units = size, activation=config['activation_function'])(hidden_layer)
        
    output_layer_temp = Dense(units = n_out)(hidden_layer)
    output_layer =  keras.layers.concatenate([input_layer, output_layer_temp])

    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(
        loss=custom_loss_wrapper(batchsize=config['batch_size']), #write here for custom LOSS FUNCTION
        optimizer=config['optimizer'](learning_rate=config['learning_rate']),
        metrics=[AdjustedRSquared_custom(config['batch_size'], n_in)],
    )
    # Potentially : Add constraints on properties
    
    #tf.keras.backend.set_value(model.optimizer.learning_rate, config['learning_rate'])

    print(model.metrics)
      
    return KerasSurrogate_custom(model,
                          preprocessor_x=config['preprocessor_x'],
                          preprocessor_y=config['preprocessor_y'],
                          name='surrogate_model',
                          version=f'TensorFlow version: {tf.__version__}')    


def build_forward_surrogate(n_in, n_out, config):
    # build the surrogate model
    n_units = [config['width']] * config['depth']
    layers = [tf.keras.layers.InputLayer(input_shape=(n_in, ))]
    #===== Add Gaussian Noise layer =====#
    layers.append(tf.keras.layers.GaussianNoise(config['x_noise'], input_shape=(n_in, )))
    #====================================#
    for size in n_units:
        layers.append(tf.keras.layers.Dense(units=size, activation=config['activation_function']))
        
    layers.append(tf.keras.layers.Dense(units=n_out))

    model = tf.keras.models.Sequential(layers)

    model.compile(
        loss=config['loss'],
        optimizer=config['optimizer'](learning_rate=config['learning_rate']),
        metrics=['MAE', 'MAPE', AdjustedRSquared(config['batch_size'], n_in)],
    )
    # Potentially : Add constraints on properties
    
    #tf.keras.backend.set_value(model.optimizer.learning_rate, config['learning_rate'])

    print(model.metrics)
      
    return KerasSurrogate(model,
                          preprocessor_x=config['preprocessor_x'],
                          preprocessor_y=config['preprocessor_y'],
                          name='surrogate_model',
                          version=f'TensorFlow version: {tf.__version__}')    


def load_dataset(datafile):
    dvar = pd.read_hdf(datafile, key='dvar')
    qoi = pd.read_hdf(datafile, key='qoi')

    dvar_trainval, dvar_test, qoi_trainval, qoi_test = train_test_split(dvar,
                                                                        qoi,
                                                                        test_size=0.2)
    dvar_trainval.reset_index(drop=True, inplace=True)
    qoi_trainval.reset_index(drop=True, inplace=True)

    kf = KFold(n_splits=5, random_state=76, shuffle=True)

    train_ind, val_ind = next(kf.split(dvar_trainval))

    dvar_train = dvar_trainval.values[train_ind]
    dvar_val = dvar_trainval.values[val_ind]
    qoi_train = qoi_trainval.values[train_ind]
    qoi_val = qoi_trainval.values[val_ind]

    return dvar_train, dvar_val, dvar_test, qoi_train, qoi_val, qoi_test


def load_dataset_wo_split(datafile):
    dvar_trainval = pd.read_hdf(datafile, key='dvar')
    qoi_trainval = pd.read_hdf(datafile, key='qoi')

    dvar_trainval.reset_index(drop=True, inplace=True)
    qoi_trainval.reset_index(drop=True, inplace=True)

    kf = KFold(n_splits=5, random_state=76, shuffle=True)

    train_ind, val_ind = next(kf.split(dvar_trainval))

    dvar_train = dvar_trainval.values[train_ind]
    dvar_val = dvar_trainval.values[val_ind]
    qoi_train = qoi_trainval.values[train_ind]
    qoi_val = qoi_trainval.values[val_ind]

    return dvar_train, dvar_val, qoi_train, qoi_val


def train_forward_model_custom(config):
    seed = config.get('seed', 49857)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load the training and validation data.
    datafile = config['datafile']

    dvar_train, dvar_val, qoi_train, qoi_val = load_dataset_wo_split(datafile)
    n_in = dvar_train.shape[1]
    

    surr = build_forward_surrogate_custom(dvar_train.shape[1],
                                   qoi_train.shape[1],
                                   config)
    
    
    # Train.
    for _ in range(config['training_repetitions']):
        surr.fit(dvar_train,
                 qoi_train,
                 X_val=dvar_val,
                 y_val=qoi_val,
                 batch_size=config['batch_size'],
                 epochs=config['epochs']) #---

        # Save the model.
        surr.save(tune.get_trial_dir())

        # Calculate the performance metrics
        qoi_pred_train = surr.predict(dvar_train)
        
        qoi_pred_val = surr.predict(dvar_val)

        metrics_train = calculate_metrics_custom(qoi_train, qoi_pred_train, n_in)
        metrics_train_abs = calculate_metrics_custom(qoi_train, tf.abs(qoi_pred_train), n_in)
        
        metrics_val = calculate_metrics_custom(qoi_val, qoi_pred_val, n_in)
        metrics_val_abs = calculate_metrics_custom(qoi_val, tf.abs(qoi_pred_val), n_in)

        metrics = {}

        for key in metrics_train.keys():
            metrics[f'{key}_train'] = metrics_train[key]
            metrics[f'{key}_val'] = metrics_val[key]
            metrics[f'{key}_train_abs'] = metrics_train_abs[key]
            metrics[f'{key}_val_abs'] = metrics_val_abs[key]

        tune.report(**metrics)


def train_forward_model(config):
    seed = config.get('seed', 49857)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load the training and validation data.
    datafile = config['datafile']

    dvar_train, dvar_val, qoi_train, qoi_val = load_dataset_wo_split(datafile)
    n_in = dvar_train.shape[1]

    surr = build_forward_surrogate(dvar_train.shape[1],
                                   qoi_train.shape[1],
                                   config)

    # Train.
    for _ in range(config['training_repetitions']):
        surr.fit(dvar_train,
                 qoi_train,
                 X_val=dvar_val,
                 y_val=qoi_val,
                 batch_size=config['batch_size'],
                 epochs=config['epochs'])

        # Save the model.
        surr.save(tune.get_trial_dir())

        # Calculate the performance metrics
        qoi_pred_train = surr.predict(dvar_train)
        
        qoi_pred_val = surr.predict(dvar_val)

        metrics_train = calculate_metrics(qoi_train, qoi_pred_train, n_in)
        metrics_val = calculate_metrics(qoi_val, qoi_pred_val, n_in)

        metrics = {}

        for key in metrics_train.keys():
            metrics[f'{key}_train'] = metrics_train[key]
            metrics[f'{key}_val'] = metrics_val[key]

        tune.report(**metrics)


def train_invertible_model(config):
    '''
    Parameters
    config: dict
        Must contain the following entries:

        - n_blocks
        - n_depth
        - n_width
        - learning_rate
        - batch_size
        - weight_artificial
        - weight_reconstruction
        - weight_x
        - weight_y
        - weight_z
        - y_noise
    '''
    seed = config.get('seed', 49857)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load the training and validation data.
    datafile = config['datafile']

    dvar_train, dvar_val, qoi_train, qoi_val = load_dataset_wo_split(datafile)
    n_in = dvar_train.shape[1]

    # hardcoded parameters
    n_blocks = config['n_blocks']
    n_depth = config['n_depth']
    n_width = config['n_width']
    weight_artificial = config['weight_artificial']
    weight_reconstruction = config['weight_reconstruction']
    weight_x = config['weight_x']
    weight_y = config['weight_y']
    weight_z = config['weight_z']
    y_noise = config['y_noise']

    nominal_dim = config['nominal_dimension']
    epochs = config['epochs']

#    surr = InvertibleNetworkSurrogate.from_config(
#        x_dim=dvar_train.shape[1],
#        y_dim=qoi_train.shape[1],
#        z_dim=1,
#        nominal_dim=nominal_dim,
#        number_of_blocks=n_blocks,
#        coefficient_network_units=[n_width for i in range(n_depth)] + [nominal_dim // 2],
#        coefficient_network_activations=['relu' for i in range(n_depth)] + ['linear'],
#        share_s_and_t=True,
#        preprocessor_x=AdaptiveMinMaxScaler(),
#        preprocessor_y=AdaptiveMinMaxScaler(),
#        name='surrogate_model',
#        sampling_distribution='gaussian',
#        version=f'TensorFlow  version: {tf.__version__}'
#    )
#    preprocessor_y = PreprocessorPipeline([LogarithmTransform(),AdaptiveMinMaxScaler()])

    surr = InvertibleNetworkSurrogate.from_config(
        x_dim=dvar_train.shape[1],
        y_dim=qoi_train.shape[1],
        z_dim=1,
        nominal_dim=nominal_dim,
        number_of_blocks=n_blocks,
        coefficient_network_units=[n_width for i in range(n_depth)] + [nominal_dim // 2],
        coefficient_network_activations=[config['activation_functions_inbetween'] for i in range(n_depth)] + [config['activation_function_last_layer']],
        share_s_and_t=True,
        preprocessor_x=config['preprocessor_x'],
        preprocessor_y=config['preprocessor_y'],
        name='surrogate_model',
        sampling_distribution='gaussian',
        version=f'TensorFlow  version: {tf.__version__}'
    )

    custom_objects = {
        'AdjustedRSquared': AdjustedRSquared,
    }

    kwargs = {
        'custom_objects': custom_objects,
        'compile': False,
    }


    # Train.
    tensorboard_dir = f'{tune.get_trial_dir()}/tensorboard'
    os.makedirs(tensorboard_dir)

    params = {
        'tensorboard_dir': tensorboard_dir,
        'optimizer': config['optimizer'](learning_rate=config['learning_rate']),
        'sampling_distribution': 'gaussian',
        'loss_weight_artificial': weight_artificial,
        'loss_weight_reconstruction': weight_reconstruction,
        'y_noise': y_noise,
        'loss_weight_x': weight_x,
        'loss_weight_y': weight_y,
        'loss_weight_z': weight_z,
        'learning_rate': config['learning_rate'], # For tensorboard?
    }

    for _ in range(epochs):
        surr.fit(dvar_train,
                 qoi_train,
                 X_val=dvar_val,
                 y_val=qoi_val,
                 batch_size=config['batch_size'],
                 epochs=1,
                 **params)

        # Save the model.
        surr.save(tune.get_trial_dir())

        # Calculate the performance metrics
        dvar_pred_train = surr.sample(qoi_train, batch_size=config['batch_size'])
        dvar_pred_val = surr.sample(qoi_val, batch_size=config['batch_size'])

        qoi_pred_train = surr.predict(dvar_pred_train)
        qoi_pred_val = surr.predict(dvar_pred_val)

        metrics_train_dvar = calculate_metrics(dvar_train, dvar_pred_train,n_in)
        metrics_val_dvar = calculate_metrics(dvar_val, dvar_pred_val, n_in)
        metrics_train = calculate_metrics(qoi_train, qoi_pred_train, n_in)
        metrics_val = calculate_metrics(qoi_val, qoi_pred_val, n_in)

        metrics = {}

        for key in metrics_train.keys():
            metrics[f'{key}_train'] = metrics_train[key]
            metrics[f'{key}_val'] = metrics_val[key]

        for key in metrics_train_dvar.keys():
            metrics[f'{key}_train_dvar'] = metrics_train_dvar[key]
            metrics[f'{key}_val_dvar'] = metrics_val_dvar[key]

        tune.report(**metrics)


