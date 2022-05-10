import os
import time
import argparse
import ray
from ray import tune
import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
from helper_functions.scan_helper_functions_configs_MP import train_forward_model_custom, train_invertible_model, train_forward_model
#from helper_functions.ScatMat_SpecificScalers import ScatMat_SpecificScalers
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from ray.tune.schedulers import AsyncHyperBandScheduler
from mllib.model import  AdaptiveMinMaxScaler, DummyPreprocessor, LogarithmTransform, StandardScaler,QuantileTransformerPreprocessor


if __name__ == '__main__':
    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir',
                        help='Directory in which to save the results.')
    parser.add_argument('--model',
                        help='For which model to scan the parameters; can be "forward"'
                        'or "invertible" (without the quotation marks)')
    parser.add_argument('--datafile',
                        help='Path to the HDF5 file containing the dataset.')
    args = parser.parse_args()

    result_dir = args.result_dir
    history_dir = f'{result_dir}/histories'

    for directory in [result_dir, history_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

            
    optimizer_list = {
        'Adam': tf.keras.optimizers.Adam, #(learning_rate=learn['learn_rate'],
                                         #beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
        'SGD': tf.keras.optimizers.SGD, #(learning_rate=learn['learn_rate'],
                                        #momentum=0.0, nesterov=False),
        # Those requireing addons
        #'AdamW': tfa.optimizers.AdamW
        #'LAMB': tfa.optimizers.LAMB
    }            
    train_func = None
    # Forward models
    if args.model == 'forward':
        train_func = train_forward_model
        learn = {'learn_rate' : tune.grid_search([0.000098])}
        config = {
            'width': tune.grid_search([118]),   #20,40,80
            'depth': tune.grid_search([3]),      # 40
            'learning_rate': learn['learn_rate'],
            'batch_size': tune.grid_search([40]),  # 8,16,32,128,256
            'datafile': args.datafile,
            'loss' : 'MSE',
            'optimizer': optimizer_list['Adam'],
            'preprocessor_x': AdaptiveMinMaxScaler(),
            'preprocessor_y': AdaptiveMinMaxScaler(),
            'epochs': tune.grid_search([100]),
            'training_repetitions': tune.grid_search([200]),
            'activation_function' : tune.grid_search(['relu']),
            'x_noise' : tune.grid_search([0.001])
        }
        
    elif args.model == 'forward_custom':
        train_func = train_forward_model_custom
        learn = {'learn_rate' : tune.grid_search([0.000033])}
        config = {
            'width': tune.grid_search([139]),   #20,40,80
            'depth': tune.grid_search([3]),      # 40
            'learning_rate': learn['learn_rate'],
            'batch_size': 50,  # 8,16,32,128,256  #batchsize has to be a divisor of nbr of train data AND nbr of val data
            'datafile': args.datafile,
            'optimizer': optimizer_list['Adam'],
            'preprocessor_x': AdaptiveMinMaxScaler(),
            'preprocessor_y': AdaptiveMinMaxScaler(),
            'epochs': tune.grid_search([100]),
            'training_repetitions': tune.grid_search([50]),
            'activation_function' : tune.grid_search(['relu']),
            'x_noise' : tune.grid_search([0.001])
        }
        
    elif args.model == 'ASHA_forward':
        train_func = train_forward_model
        config = {
                'depth': tune.randint(2, 210),
                'width': tune.randint(5, 500),
                'learning_rate' : tune.uniform(0.01, 0.001),
                'batch_size': tune.choice([2,8,16,32,128,256,500]),
                'datafile': args.datafile,
                'loss' : 'MSE',
                'optimizer': optimizer_list['Adam'],
                'preprocessor_x': AdaptiveMinMaxScaler(),
                'preprocessor_y': AdaptiveMinMaxScaler(),
                'epochs': tune.grid_search([50]),
                'training_repetitions': tune.grid_search([50]),
                'activation_function' : tune.grid_search(['relu']),
                'x_noise' : tune.choice([0.00, 0.01])
        }
        # AsyncHyperBand enables aggressive early stopping of bad trials.
        """ Note: During training of the forward model, it reports to ray every
            100 epochs. Therefore in the case of the maximum number of
            iterations, which is 50, 5000 epochs have beeen gone through."""
        scheduler = AsyncHyperBandScheduler(
            # 'training_iteration' is incremented every time `trainable.step` is called
            time_attr='training_iteration',
            # The training result objective value attribute. Stopping procedures will use this attribute.
            metric='MAE_val',
            # mode: {min, max}. Determines whether objective is minimizing or maximizing the metric attribute.
            mode='min',
            # max time units per trial. Trials will be stopped after max_t time units (determined by time_attr).
            max_t=50,
            # Only stop trials at least this old in time. The units are the same as the attribute named by time_attr.
            grace_period=10,
            # Used to set halving rate and amount. This is simply a unit-less scalar.
            reduction_factor=4,
            # Number of brackets. Each bracket has a different halving rate, specified by the reduction factor.
            brackets=1
            )
        # 'training_iteration' is incremented every time `trainable.step` is called
        # For each trial, stop when trial has reached 50 iterations
        stopping_criteria = {"training_iteration": 50}
        stopper = TrialPlateauStopper(metric='MAE_val', std=0.001,
                                      num_results=5, grace_period=60)

    # Invertible models
    elif args.model == 'invertible':
        train_func = train_invertible_model
        learn = {'learn_rate' : tune.grid_search([0.001])}
        config = {
                'n_blocks': tune.grid_search([3,4]),
                'n_depth': tune.grid_search([2,5]),
                'n_width': tune.grid_search([10,20]),
                'learning_rate': learn['learn_rate'],
                'batch_size': tune.grid_search([32]),
                'datafile': args.datafile,
                'weight_artificial': tune.grid_search([1]),
                'weight_reconstruction': tune.grid_search([100]),#0.05,100
                'weight_x': tune.grid_search([400]),#1,0.3,100,400
                'weight_y': tune.grid_search([400]),#1,0.3,100,400
                'weight_z': tune.grid_search([400]),#1,0.3,100,400
                'optimizer': optimizer_list['Adam'],
                'nominal_dimension' : 330,   #must be an even number and greater than the number of qoi
                'epochs' : 50,
                'preprocessor_x' : tune.grid_search([AdaptiveMinMaxScaler(),StandardScaler()]),
                'preprocessor_y' : tune.grid_search([AdaptiveMinMaxScaler(),StandardScaler()]),
                'activation_functions_inbetween' : 'relu',
                'activation_function_last_layer' : tune.choice(['linear','relu'])
                #'optimizer' : 'Adam', # 'Adam', 'SGD', ['AdamW', 'LAMB']
        }
    elif args.model == 'ASHA_invertible':
        train_func = train_invertible_model
        #pre_stst = ScatMat_SpecificScalers(Scaler_P11=StandardScaler(),
        #                                  Scaler_PPF=StandardScaler(), LogP11=True)
        #pre_mmqt = ScatMat_SpecificScalers(Scaler_P11=AdaptiveMinMaxScaler(),
                                                           #Scaler_PPF=QuantileTransformerPreprocessor(), LogP11=True)
 #       pre_qtqt = ScatMat_SpecificScalers(Scaler_P11=AdaptiveMinMaxScaler(),
                                           #Scaler_PPF=QuantileTransformerPreprocessor(), LogP11=False)
        config = {
                'n_blocks': tune.randint(3,5),
                'n_depth': tune.randint(2,4),
                'n_width': tune.randint(80, 150),
                #'learning_rate': tune.uniform(0.001, 0.0001),
                'batch_size': tune.choice([8]),
                'learning_rate': 9e-5,
                'datafile': args.datafile,
                'weight_artificial': tune.randint(0.05,1),
                'weight_reconstruction': tune.randint(220,350),
                'weight_x': tune.randint(140,200),
                'weight_y': 350, #tune.randint(0.3,400),
                'weight_z': tune.randint(180,350),
                'y_noise': tune.uniform(0.01, 0.10),
                'optimizer': optimizer_list['Adam'],
                'nominal_dimension' : 962,  #must be an even number and greater than the number of qoi
                'epochs' : 100,
                'preprocessor_x' : AdaptiveMinMaxScaler(),
                'preprocessor_y' : StandardScaler(),
                'activation_functions_inbetween' : 'relu',
                'activation_function_last_layer' : 'linear',
        }
        # AsyncHyperBand enables aggressive early stopping of bad trials.
        """ Note: During training of the inverse model, it reports to ray every
            epoch. Therefore in the case of the maximum number of
            iterations, which is 30, 30 epochs have beeen gone through.
            (@ Romana: I am unsure, wether this is intended,
            but I have adapted the code below acordingly)
        """
        scheduler = AsyncHyperBandScheduler(
            # 'training_iteration' is incremented every time `trainable.step` is called
            time_attr='training_iteration',
            # The training result objective value attribute. Stopping procedures will use this attribute.
            metric='MAE_val',
            # mode: {min, max}. Determines whether objective is minimizing or maximizing the metric attribute.
            mode='min',
            # max time units per trial. Trials will be stopped after max_t time units (determined by time_attr).
            max_t=30,
            # Only stop trials at least this old in time. The units are the same as the attribute named by time_attr.
            grace_period=5,
            # Used to set halving rate and amount. This is simply a unit-less scalar.
            reduction_factor=4,
            # Number of brackets. Each bracket has a different halving rate, specified by the reduction factor.
            brackets=1
            )
        # 'training_iteration' is incremented every time `trainable.step` is called
        # For each trial, stop when trial has reached 30 iterations
        stopping_criteria = {"training_iteration": 30}
        stopper = TrialPlateauStopper(metric='MAE_val', std=0.001,
                                      num_results=5, grace_period=60)

    else:
        raise ValueError(f'Unknown model: {args.model}')

    ray.init(_temp_dir='/scratch/ray')
    # Use Async Successive Halving, https://arxiv.org/abs/1810.05934
    if 'ASHA' in args.model:
        start = time.time()
        analysis = tune.run(train_func,
                            # num_samples are number of trials.
                            # Each trial is one instance of a Trainable.
                            num_samples=60,
                            name=args.model + '_id',
                            config=config,
                            local_dir=result_dir,
                            scheduler=scheduler,
                            stop=stopper,
                            resources_per_trial={'cpu': 1, 
                                                 'gpu': 0})#,
                            #queue_trials=False)
    else:  # No ASHA, plain vanilla
        start = time.time()
        analysis = tune.run(train_func,
                            # num_samples are number of trials.
                            # Each trial is one instance of a Trainable.
                            num_samples=1,
                            name=args.model + '_id',
                            config=config,
                            local_dir=result_dir,
                            resources_per_trial={'cpu': 1,
                                                 'gpu': 0})#,
                            #queue_trials=False)

    # Save the training histories and the results.
    for key, df in analysis.trial_dataframes.items():
        ID = df['trial_id'].unique()[0]
        df.to_csv(f'{history_dir}/{ID}.csv')

    analysis.dataframe().to_csv(f'{result_dir}/results.csv')

    end = time.time()
    with open(f'{result_dir}/time_for_FW_scan.txt', 'w') as file:
        file.write(f'{end - start} s')
    with open(f'{result_dir}/path_to_datafile.txt', 'w') as file:
        file.write(args.datafile)


