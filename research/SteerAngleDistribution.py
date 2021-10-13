import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '../../test_dataset/SeqTrain'
h5s = []
steers = []

for root, _, files in os.walk(data_path):
    for file in files:
        h5s.append(os.path.join(root, file))

for h5_path in h5s:
    with h5py.File(h5_path, 'r') as f:
        for i in range(200):
            steers.append(f['steer'][i])

# num_bins = 20
# fig, ax = plt.subplots()
# n, bins_limits, patches = ax.hist(steers, num_bins, density=1)
# plt.show()

sns.distplot(steers)

plt.tight_layout()
plt.show()
