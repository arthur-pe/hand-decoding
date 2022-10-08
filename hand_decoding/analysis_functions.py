import numpy as np
from sklearn import discriminant_analysis as da
from .functions import *

def lda_pos(initial_state, condition, target_position):
    """
    Returns a decoder of the neural activity that decodes condition on the first axis and target position on the other two
    in:
        initial_state (ndarray): trial x neuron. The initial state from which to decode
        condition (ndarray, int): trial. The trial condition
        target_position (ndarray): trial x XY. The target position
    out:
        decoder (ndarray): neuron x 3.
    """

    W_lda = da.LinearDiscriminantAnalysis().fit(initial_state, condition).coef_
    W_target = np.linalg.lstsq(initial_state, target_position, rcond=None)[0].T

    W = np.concatenate([W_lda, W_target], axis=0)

    return W


if __name__=='__main__':

    initial_state = np.random.randn(20,10)
    condition = (np.random.rand(20)<0.5).astype(int)
    target_position = np.random.randn(20,2)

    W = lda_pos(initial_state, condition, target_position)

    decoded_initial_state = initial_state @ W.T

    print(decoded_initial_state.shape)