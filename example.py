import hand_decoding
import numpy as np

data = np.random.rand(10,11,12)*2-1
movement = data @ np.random.randn(12,2) + np.random.randn(10,11,2)/10

decoded_movement, decoder = hand_decoding.functions.position_decoding(data, movement)

r2 = hand_decoding.functions.trial_wise_r2(movement, decoded_movement)

print('mse of movement:', np.mean((decoded_movement-movement)**2), '\tR^2:', r2)