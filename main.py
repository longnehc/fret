import numpy as np

ideal_data = np.load('ideal_efficiency.npy')
noisy_data = np.load('noisy_efficiency.npy')

print(ideal_data.shape)  # (100, frames)
print(noisy_data.shape)  # (100, frames, 2)