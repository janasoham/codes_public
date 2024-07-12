#### This code plots the hockey-data goals from the year 2018 and
#### their corresponding minimum-distance based EB predictions for the year 2019 ####


import sys
import numpy as np
import warnings
from matplotlib.ticker import MaxNLocator;
import pandas as pd;
import time
import random


# Disable all warnings
warnings.filterwarnings("ignore")

# Add parent directory to the Python module search path
sys.path.append("../main_mixture_codes")

from main_poisson_worst_prior import *;

from roots_and_preds import *;

my_rand = random.random()

print(my_rand)

######################### Grids ####################

Xs_max = 30
# all_grid_sizes = list(range(2, 10, 1))

all_grid_sizes = [400]

theta_max = 10


################ Gradient descent ##################

all_target_vals = []

all_poisson_ebs = []

all_supports = []

all_probs = []

all_grids = []

for grid_size in all_grid_sizes:

    grid = np.linspace(0, theta_max, grid_size)

    target_val, theta, mu = worst_prior(Xs_max, grid);

    all_supports.append(theta)

    all_probs.append(mu)

    all_target_vals.append(target_val)\

ell_star = np.argmax(all_target_vals)

print(all_target_vals)

target_val = all_target_vals[ell_star]

plt.clf()

plt.plot(figsize=(8, 5))

plt.hist(all_supports[ell_star], bins=50, weights=all_probs[ell_star], color='blue', edgecolor='black')
plt.xlabel('support')
plt.ylabel('probs')
plt.title('worst case distribution')

plt.savefig("numerical_results/least_favorable_prior_ID_%g.png" % my_rand)

