import numpy as np
from helper_functions import *

"""
Throughout:
data (ndarray): time x trial x ... x neuron
movement (ndarray): time x ... x trial x XY 
"""

def decoding(x, y):
    """
    Linearly decodes (with bias) y from x (least square)
    in:
        x (ndarray): z_1, z_2, ..., X
        y (ndarray): z_1, z_2, ..., Y
    out:
        decoder_weight (ndarray): dim(Y) x dim(X) decoder weight
        decoder_bias (ndarray): dim(Y) decoder bias
    """

    x_shape = x.shape

    x_with_ones = np.concatenate([x, np.ones((list(x_shape[:-1])+[1]))], axis=-1)

    x_with_ones_flat = x_with_ones.reshape(-1, x_shape[-1]+1)
    y_flat = y.reshape(-1, y.shape[-1])

    W_b = np.linalg.lstsq(x_with_ones_flat, y_flat, rcond=None)[0].T # dim(Y) x dim(X)+1

    W, b = W_b[:,:x_shape[-1]], W_b[:,-1]

    return W, b

def velocity_decoding(data, movement):
    """
    Linearly decodes (with bias) the velocity from the data and integrates it from true starting point
    in:
        data, movement
    out:
        decoded_movement (ndarray): time x trial x ... x XY decoded movement
        decoding_function (python function): a function which takes data as input and returns movement
    """

    velocity = movement[1:]-movement[:-1]
    data_cut = data[:-1]

    W, b = decoding(data_cut, velocity)

    def decoder(x):

        velocity_hat = x[:-1] @ W.T + b

        position_zero_centered = np.cumsum(velocity_hat, axis=0)
        position_no_init = position_zero_centered + movement[0][np.newaxis]
        position = np.concatenate([movement[0][np.newaxis], position_no_init], axis=0)

        return position

    return decoder(data), decoder


def position_decoding(data, movement):
    """
    Linearly decodes (with bias) the position from the data
    in:
        data, movement
    out:
        decoded_movement (ndarray): time x trial x ... x x XY decoded movement
        decoding_function (python function): a function which takes data as input and returns movement
    """

    W, b = decoding(data, movement)

    def decoder(x): return x @ W.T + b

    return decoder(data), decoder

def trial_wise_r2(y, y_hat):
    """
    Returns trial-wise R^2
    in:
        y (ndarray): trial x time x neuron
        y_hat (ndarray): trial x time x neuron
    out:
        r2 (float): R^2
    """
    trial_r2 = 1 - np.sum((y - y_hat) ** 2, axis=(1, 2)) / np.sum((y - y.mean(axis=1)[:, np.newaxis]) ** 2,
                                                                  axis=(1, 2))
    r2 = trial_r2.mean(axis=0)
    return r2

def k_fold_cross_validated_r2(data, movement, decoding_function, folds=5, sample_size=1):
    """
    Performs k-fold linear decoding of the data
    in:
        data, movement
        decoding function (python function): a function which takes data and movement as an argument and returns a decoder
        folds (int): the number of splits of the data
        sample_size (int): how many times the whole procedure is done
    out:
        r2_mean (float): mean R^2 over trials, folds, permutations (see trial_wise_r2)
        r2_std (float): same but std
    """
    #decoded test movement on all folds?

    data_shape, data_trial_time_neuron, data_trial_time_neuron_shape = flatten_mid_dims(data)
    nb_trial, nb_time, nb_neuron = data_trial_time_neuron_shape

    movement_shape, movement_trial_time_x, movement_trial_time_x_shape = flatten_mid_dims(movement)

    size_of_folds_samples = round((nb_trial-nb_trial % folds)/folds)

    all_r2 = []
    for sample in range(sample_size):
        permutation = np.random.permutation(nb_trial)
        permuted_data, permuted_movement = data_trial_time_neuron[permutation], movement_trial_time_x[permutation]
        n = 0
        r2 = []
        for fold in range(folds):
            train_data, test_data = split(permuted_data, n, size_of_folds_samples)
            train_movement, test_movement = split(permuted_movement, n, size_of_folds_samples)

            train_movement_hat, decoder = decoding_function(train_data, train_movement)

            test_movement_hat = decoder(test_data)

            current_r2 = trial_wise_r2(test_movement, test_movement_hat)

            r2 += [current_r2]

            n += size_of_folds_samples

        all_r2 += [r2]

    all_r2 = np.array(all_r2)

    return all_r2.mean(), all_r2.std()


if __name__=='__main__':

    x = np.random.randn(10,11,13)
    y = x @ np.random.randn(13,3) + np.random.randn(*(list(x.shape)[:-1]+[3]))/10

    y_hat, decoder = position_decoding(x, y)

    print(mse(y, decoder(x)))

    y_hat, decoder = velocity_decoding(x, y)

    print(mse(y, decoder(x)))

    r2, r2_std = k_fold_cross_validated_r2(x, y, velocity_decoding, folds=5, sample_size=3)

    print(r2, r2_std)