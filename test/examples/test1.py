

import numpy as np
import matplotlib.pyplot as plt

nr_of_clusters = 4
nr_of_points_x_cluster = 50

close_threshold = 0.2
distant_threshold = 1.0

margin = .2

mean = [np.random.uniform([-1 + margin, -1 + margin], [0 - margin, 0 - margin], 2),
        np.random.uniform([0 + margin, -1 + margin], [1 - margin, 0 - margin], 2),
        np.random.uniform([-1 + margin, 0 + margin], [0 - margin, 1 - margin], 2),
        np.random.uniform([0 + margin, 0 + margin], [1 - margin, 1 - margin], 2)]
print(mean)

tt=np.random.uniform([-1 + margin, -1 + margin], [0 - margin, 0 - margin], (4,2))
print(tt)