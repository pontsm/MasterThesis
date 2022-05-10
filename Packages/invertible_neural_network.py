import math
import json
import logging
from timeit import default_timer # needed to tell user how long a single epoch takes to train
import numpy as np
import tensorflow as tf
from mllib.model import KerasSurrogate
from helper_functions.affine_coupling_block import AffineCouplingBlock
from helper_functions.permutation_layer import PermutationLayer
from helper_functions.ardizzone_et_al_helpers import prepare_X, distribution_code_map, prepare_y, apply_zero_padding, mmd2

_type = tf.float64

def build_invertible_neural_network(x_dim, y_dim, z_dim, nominal_dim,
                                    number_of_blocks,
                                    coefficient_net_config_s,
                                    coefficient_net_config_t,
                                    share_s_and_t=False):
        # input layer
        input_layer = tf.keras.layers.Input(shape=nominal_dim)
        l = input_layer

        # blocks consisting of an AffineCouplingBlock
        # followed by a PermutationLayer
        for i in range(number_of_blocks):
            # build an affine coupling block
            s1_config = coefficient_net_config_s
            s2_config = coefficient_net_config_s
            t1_config = coefficient_net_config_t
            t2_config = coefficient_net_config_t
            l = AffineCouplingBlock(s1_config, s2_config, t1_config, t2_config, share_s_and_t)(l)

            # add a permutation layer between the affine coupling layers
            if i != (number_of_blocks - 1):
                l = PermutationLayer()(l)

        return tf.keras.Model(input_layer, l)


def inverse_pass(model, dimensions, inputs):
    num_layers = len(model.layers)
    x_dim = dimensions['x_dim']

    out = inputs
    for i in range(num_layers):
        l = model.get_layer(index=num_layers - 1 - i)
        if type(l) is not tf.keras.layers.InputLayer:
            out = l.inverse(out)

    return out


def calculate_losses(model,
                    dimensions,
                    batch_size,
                    dtype,
                    X_train_batch, label_train_batch, y_noise_scale,
                    loss_x, loss_y, loss_z, ignore_dims_loss, loss_reconstruction):
    # scale of noise to add before reconstruction pass
    # y_noise_scale = 0.1

    x_dim = dimensions['x_dim']
    y_dim = dimensions['y_dim']
    z_dim = dimensions['z_dim']
    nominal_dim = dimensions['nominal_dim']
    artificial_yz_dim = nominal_dim - y_dim - z_dim
    artificial_x_dim = nominal_dim - x_dim

    # shape of the padding (= artificial "features")
    artificial_yz_batch_shape = (batch_size, artificial_yz_dim)
    artificial_x_batch_shape = (batch_size, artificial_x_dim)

    ################
    # Forward pass
    ################
    # get predictions for y and z
    prediction = model(X_train_batch)

    y_pred = prediction[:, :y_dim]
    z_pred = prediction[:, y_dim:y_dim+z_dim]

    # get true values of y and z
    y_true = label_train_batch[:, :y_dim]
    z_true = label_train_batch[:, y_dim:y_dim+z_dim]

    # losses from forward pass
    ly = loss_y(y_true, y_pred)
    lz = loss_z(label_train_batch[:, :y_dim+z_dim], prediction[:, :y_dim+z_dim])

    ################
    # Backward pass
    ################
    # perform backward pass
    X_pred = inverse_pass(model, dimensions, label_train_batch)
    lx = loss_x(X_train_batch[:, :x_dim], X_pred[:, :x_dim])

    ################################################
    # make sure artificial dimensions are not used
    ################################################
    l_artificial_dims = ignore_dims_loss(tf.zeros(artificial_yz_batch_shape),
                                         label_train_batch[:, y_dim+z_dim:])

    l_artificial_dims += ignore_dims_loss(tf.zeros(artificial_x_batch_shape),
                                         X_train_batch[:, x_dim:])

    ##########################
    # reconstruction loss
    ##########################
    input_for_reconstruction = prediction

    # add noise to the y and z components
    perturbed_yz = prediction[:, :y_dim + z_dim] + y_noise_scale * tf.random.normal((input_for_reconstruction.shape[0], y_dim + z_dim),
                                                                                   mean=0.,
                                                                                   stddev=1.0,
                                                                                   dtype=_type)
    zero_padding = tf.zeros((prediction.shape[0], nominal_dim - y_dim - z_dim), dtype=_type)
    input_for_reconstruction = tf.concat([perturbed_yz, zero_padding], axis=1)

    X_reconstructed = inverse_pass(model, dimensions, input_for_reconstruction)
    l_reconstruction = loss_reconstruction(X_train_batch, X_reconstructed)

    return lx, ly, lz, l_artificial_dims, l_reconstruction


# We need a class here because if we didn't, the computation graph would
# define the same variable twice if we want to fit two models in the same
# script. That would raise an error. We need to train two models for
# unit testing.
# For details, see:
# https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/#handling-states-breaking-the-function-scope
# Another useful resource:
# https://www.tensorflow.org/guide/function
# --> Tracing Semantics --> Variables
class Train:

    @tf.function
    def __call__(self,
                 model,
                 dimensions,
                 dataset_train, dataset_val,
                 optimizer,
                 batch_size,
                 number_of_samples,
                 y_noise_scale,
                 loss_x, loss_y, loss_z, loss_reconstruction, loss_weights,
                 dtype,
                 writer_train,
                 writer_val,
                 epoch):
        '''
        :param loss_weights: factors to multiply lx, ly, lz by
        :param writer: file_writer for logging to tensorboard
        '''
        ignore_dims_loss = tf.keras.losses.MeanSquaredError()

        total_loss = tf.constant(0., dtype=dtype)
        num_processed_batches = tf.constant(0, dtype='int64')
        num_batches = number_of_samples // batch_size

        tf.print('Total number of batches:', num_batches)

        l_tot_sum = tf.constant(0., dtype=dtype)
        lx_sum = tf.constant(0., dtype=dtype)
        ly_sum = tf.constant(0., dtype=dtype)
        lz_sum = tf.constant(0., dtype=dtype)
        l_artificial_dims_sum = tf.constant(0., dtype=dtype)
        l_reconstruction_sum = tf.constant(0., dtype=dtype)

        # drop_remainder=True needed if batch_size does not divide dataset size evenly
        for X_train_batch, label_train_batch in dataset_train.shuffle(number_of_samples).batch(batch_size, drop_remainder=True):
            num_processed_batches += 1

            with tf.GradientTape() as tape:
                # calculate the loss on the validation set
                lx, ly, lz, l_artificial_dims, l_reconstruction = calculate_losses(model,
                                                                                   dimensions,
                                                                                   batch_size,
                                                                                   dtype,
                                                                                   X_train_batch,
                                                                                   label_train_batch,
                                                                                   y_noise_scale,
                                                                                   loss_x, loss_y, loss_z, ignore_dims_loss, loss_reconstruction)

                # calculate total loss
                total_loss = (loss_weights[0] * lx
                              + loss_weights[1] * ly
                              + loss_weights[2] * lz
                              + loss_weights[3] * l_artificial_dims
                              + loss_weights[4] * l_reconstruction)

            # calculate the gradients
            grads = tape.gradient(total_loss, model.trainable_variables)

            # perform optimization step
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # average over the batches
            l_tot_sum += total_loss
            lx_sum += lx
            ly_sum += ly
            lz_sum += lz
            l_artificial_dims_sum += l_artificial_dims
            l_reconstruction_sum += l_reconstruction

            # log the train and validation losses
            if num_processed_batches % 1000 == 0:
                tf.print('Processed batch', num_processed_batches)

        with writer_train.as_default():
            divisor = tf.dtypes.cast(num_processed_batches, dtype)

            tf.summary.scalar('lx', data=lx_sum / divisor, step=epoch)
            tf.summary.scalar('ly', data=ly_sum / divisor, step=epoch)
            tf.summary.scalar('lz', data=lz_sum / divisor, step=epoch)
            tf.summary.scalar('lartificial', data=l_artificial_dims_sum / divisor, step=epoch)
            tf.summary.scalar('l_reconstruction', data=l_reconstruction_sum / divisor, step=epoch)
            tf.summary.scalar('l_tot', data=l_tot_sum / divisor, step=epoch)

        # calculate the validation losses
        if dataset_val is not None:
            l_tot_sum = tf.constant(0., dtype=dtype)
            lx_sum = tf.constant(0., dtype=dtype)
            ly_sum = tf.constant(0., dtype=dtype)
            lz_sum = tf.constant(0., dtype=dtype)
            l_artificial_dims_sum = tf.constant(0., dtype=dtype)
            l_reconstruction_sum = tf.constant(0., dtype=dtype)

            num_processed_batches = 0
            for X_train_batch, label_train_batch in dataset_val.shuffle(number_of_samples).batch(batch_size, drop_remainder=True):
                num_processed_batches += 1

                lx, ly, lz, l_artificial_dims, l_reconstruction = calculate_losses(model,
                                                                                   dimensions,
                                                                                   batch_size,
                                                                                   dtype,
                                                                                   X_train_batch,
                                                                                   label_train_batch,
                                                                                   y_noise_scale,
                                                                                   loss_x, loss_y, loss_z, ignore_dims_loss, loss_reconstruction)
                # calculate total loss
                total_loss = (loss_weights[0] * lx
                              + loss_weights[1] * ly
                              + loss_weights[2] * lz
                              + loss_weights[3] * l_artificial_dims
                              + loss_weights[4] * l_reconstruction)
                l_tot_sum += total_loss
                lx_sum += lx
                ly_sum += ly
                lz_sum += lz
                l_artificial_dims_sum += l_artificial_dims
                l_reconstruction_sum += l_reconstruction

            with writer_val.as_default():
                divisor = tf.dtypes.cast(num_processed_batches, dtype)

                tf.summary.scalar('lx', data=lx_sum / divisor, step=epoch)
                tf.summary.scalar('ly', data=ly_sum / divisor, step=epoch)
                tf.summary.scalar('lz', data=lz_sum / divisor, step=epoch)
                tf.summary.scalar('lartificial', data=l_artificial_dims_sum / divisor, step=epoch)
                tf.summary.scalar('l_reconstruction', data=l_reconstruction_sum / divisor, step=epoch)
                tf.summary.scalar('l_tot', data=l_tot_sum / divisor, step=epoch)


class InvertibleNetworkSurrogate(KerasSurrogate):

    @staticmethod
    def from_config(x_dim, y_dim, z_dim, nominal_dim,
                 number_of_blocks,
                 coefficient_network_units,
                 coefficient_network_activations,
                 share_s_and_t,
                 preprocessor_x,
                 preprocessor_y,
                 name,
                 sampling_distribution,
                 version):
        '''
        :param x_dim: dimension of the x space
        :param y_dim: dimension of the y space
        :param z_dim: dimension of the z space
        :param nominal_dim: nominal dimension
        :param number_of_blocks: how many blocks consisting of
                                 AffineCouplingLayers and PermutationLayers
                                 to use
        :param coefficient_network_units: list(int) number of units in each
                                          layer of the coefficient networks
        :param coefficient_network_activations: list(str) activations after
                                                each layer of the
                                                coefficient networks
        :param share_s_and_t: whether to use weight sharing for the s and t
                              coefficient networks
        :param preprocessor_x: preprocessor for the x values
        :param preprocessor_y: preprocessor for the y values
        :param name: name of the surrogate model
        :param sampling_distribution: distribution of the z space
                                      ('uniform' or 'gaussian')
        :param version: version of the surrogate model
        '''
        coefficient_config = {
            'subnet_type': 'dense',
            'units': coefficient_network_units,
            'activations': coefficient_network_activations
        }

        model = build_invertible_neural_network(x_dim,
                                    y_dim,
                                    z_dim,
                                    nominal_dim,
                                    number_of_blocks,
                                    coefficient_config,
                                    coefficient_config,
                                    share_s_and_t)

        surr = InvertibleNetworkSurrogate(model, preprocessor_x, preprocessor_y, name, version)

        surr._x_dim = x_dim
        surr._y_dim = y_dim
        surr._z_dim = z_dim
        surr._nominal_dim = nominal_dim

        surr._sampling_distribution = sampling_distribution

        return surr

    def _fit_model(self, X_train, y_train, X_val, y_val, **kwargs):
        '''
        parameters = {
            'optimizer': Tensorflow optimizer object,
            'batch_size': int,
            'y_noise': scale of noise added during reconstruction
            'loss_weight_x': float,
            'loss_weight_y': float,
            'loss_weight_z': float,
            'loss_weight_artificial': float, # weight for loss to make
                                             # artificial dimensions
                                             # close to zero
            'loss_weight_reconstruction': float,
            'tensorboard_dir': str,          # tensorboard logdir
            'epochs': int,
        }
        '''
        # build datastructures for training
        dimensions = {
            'x_dim': self._x_dim,
            'y_dim': self._y_dim,
            'z_dim':self._z_dim,
            'nominal_dim': self._nominal_dim,
        }

        loss_weights = np.array([
            kwargs['loss_weight_x'],
            kwargs['loss_weight_y'],
            kwargs['loss_weight_z'],
            kwargs['loss_weight_artificial'],
            kwargs['loss_weight_reconstruction'],
        ])
        # normalise loss weights
        loss_weights = loss_weights / np.sum(loss_weights)

        # losses
        def loss_x(x_true, x_pred):
            return mmd2(x_true, x_pred)

        loss_y = tf.keras.losses.MeanSquaredError()

        def loss_z(z_true, z):
            return mmd2(z_true, z)

        loss_reconstruction = tf.keras.losses.MeanSquaredError()

        logging.info('X_train min:')
        logging.info(np.min(X_train, axis=0))
        logging.info('X_train max:')
        logging.info(np.max(X_train, axis=0))
        logging.info('y_train min:')
        logging.info(np.min(y_train, axis=0))
        logging.info('y_train max:')
        logging.info(np.max(y_train, axis=0))

        # build the training dataset
        X_train_padded, y_train_padded = apply_zero_padding(X_train, y_train,
                                                            self._x_dim,
                                                            self._y_dim,
                                                            self._z_dim,
                                                            self._nominal_dim,
                                                            distribution_code_map[self._sampling_distribution])
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train_padded))

        if X_val is None:
            dataset_val = None
        else:
            X_val_padded, y_val_padded = apply_zero_padding(X_val, y_val,
                                                            self._x_dim,
                                                            self._y_dim,
                                                            self._z_dim,
                                                            self._nominal_dim,
                                                            distribution_code_map[self._sampling_distribution])
            dataset_val = tf.data.Dataset.from_tensor_slices((X_val_padded, y_val_padded))


        # organise logging to tensorboard
        writer_train = tf.summary.create_file_writer(f'{kwargs["tensorboard_dir"]}/train', name='train')
        writer_val = tf.summary.create_file_writer(f'{kwargs["tensorboard_dir"]}/val', name='val')

        train_one_epoch = Train()

        # train
        for i in range(kwargs['epochs']):
            logging.info(f'Start epoch {i}...')

            start = default_timer()

            train_one_epoch(self.model,
                            dimensions,
                            dataset_train,
                            dataset_val,
                            kwargs['optimizer'],
                            kwargs['batch_size'],
                            X_train.shape[0], # number of samples
                            kwargs['y_noise'],
                            loss_x, loss_y, loss_z, loss_reconstruction,
                            loss_weights,
                            tf.float64,
                            writer_train,
                            writer_val,
                            i)

            end = default_timer()

            # get duration [s]
            duration = end - start
            logging.info(f"Epoch {i+1}/{kwargs['epochs']} needed {int(duration//60)}:{int(math.ceil(duration) % 60)}.")

    def _predict_model(self, X, **kwargs):
        X_padded = prepare_X(X, self._x_dim, self._nominal_dim)
        prediction = self.model.predict(X_padded, **kwargs)

        # remove latent space and padding dimensions
        return prediction[:, :self._y_dim]

    def sample(self, y, batch_size):
        '''Generate samples of the X space corresponding to the given y space configurations.'''
        dimensions = {
            'x_dim': self._x_dim,
            'y_dim': self._y_dim,
            'z_dim':self._z_dim,
            'nominal_dim': self._nominal_dim,
        }

        y = self.preprocessor_y.transform(y)

        X_samples = []

        num_batches = y.shape[0] // batch_size

        for i in range(num_batches):
            y_batch = y[i*batch_size:(i+1)*batch_size, :]
            yz_padded = prepare_y(y_batch, self._y_dim, self._z_dim, self._nominal_dim, distribution_code_map[self._sampling_distribution])
            sampled = inverse_pass(self.model, dimensions, yz_padded)
            X_samples.append(sampled)

        if (y.shape[0] % batch_size) != 0:
            y_batch = y[num_batches*batch_size:, :]
            yz_padded = prepare_y(y_batch, self._y_dim, self._z_dim, self._nominal_dim, distribution_code_map[self._sampling_distribution])
            sampled = inverse_pass(self.model, dimensions, yz_padded)
            X_samples.append(sampled)

        return self.preprocessor_x.inverse_transform(np.vstack(X_samples)[:, :self._x_dim])

    def sample_n_tries(self, y, batch_size, n_tries=1):
        '''
        Generate samples of the X space corresponding to the given y space configurations.

        Returns the DVAR configs that are closest to each individual target QOI
        config (best-of-n_tries).

        Please see `doc/INN_best_of_n_algorithm/best_of_n.pdf` for mathematical
        explanations.

        :param n_tries: How many DVAR configs to sample for each desired QOI config
        :returns: The DVAR configs that leads to the QOI configs that are closest to `y`
        '''
        # get n_tries DVAR candidate configs
        X_tried = []
        for i in range(n_tries):
            X_i = self.sample(y, batch_size)
            X_tried.append(X_i)
        X_s = np.stack(X_tried, axis=2)

        # evaluate the sampled DVAR configs
        R = []
        for X in X_tried:
            Y_i = self.predict(X, batch_size=batch_size)
            rel_error = np.abs((y - Y_i) / (y + 1e-12)) # avoid division by zero
            # remove path length
            rel_error = rel_error[:, :-1]
            rel_error = np.max(rel_error, axis=1)
            rel_error = rel_error.reshape((rel_error.shape[0], 1))
            R.append(rel_error)
        R = np.hstack(R)

        # determine minimum DVAR config for each sample
        m_r = np.min(R, axis=1)
        m_r = np.reshape(m_r, (y.shape[0], 1))
        M_r = np.tile(m_r, (1, n_tries))

        m = (R == M_r).astype('float64')

        DVAR = np.einsum('ijk,ki->ij', X_s, m.T)

        return DVAR


    def _save_model(self, model_dir):
        # save neural network
        model_path = '{}/model.hdf5'.format(model_dir)
        self.model.save(model_path)

        # save the dimensions
        dimensions = {
            'x_dim': self._x_dim,
            'y_dim': self._y_dim,
            'z_dim':self._z_dim,
            'nominal_dim': self._nominal_dim,
        }
        with open('{}/dimensions.json'.format(model_dir), 'w') as file:
            json.dump(dimensions, file, indent=4)

        with open(f'{model_dir}/sampling_distribution.txt', 'w') as file:
            file.write(self._sampling_distribution)

    @classmethod
    def _load_model(cls, model_dir, model_kwargs={}):
        '''
        :param identifier: [models_dir, model_name]
        '''
        custom_objects = {
            'AffineCouplingBlock': AffineCouplingBlock,
            'PermutationLayer': PermutationLayer
        }
        model_path = '{}/model.hdf5'.format(model_dir)
        model = tf.keras.models.load_model(model_path,
                                           custom_objects=custom_objects,
                                           compile=False)

        return model

    @classmethod
    def _build_surrogate(cls, model, preprocessor_x, preprocessor_y, name, version, model_dir):
        surr = InvertibleNetworkSurrogate(model,
                                          preprocessor_x,
                                          preprocessor_y,
                                          name,
                                          version)
        # load the dimensions
        json_path = '{}/dimensions.json'.format(model_dir)

        with open(json_path, 'r') as file:
            dimensions = json.load(file)

        surr._x_dim = dimensions['x_dim']
        surr._y_dim = dimensions['y_dim']
        surr._z_dim = dimensions['z_dim']
        surr._nominal_dim = dimensions['nominal_dim']

        with open(f'{model_dir}/sampling_distribution.txt', 'r') as file:
            surr._sampling_distribution = file.read().strip()

        return surr
