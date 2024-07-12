#### This code plots the worst prior supported on [0,theta_max]
# that maximizes the posterior variance

from pathlib import Path
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

Xs_max = 80

all_grid_sizes = [400]

theta_max = 50


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


# Define the folder name
folder_name = 'numerical_results'

# Create a Path object for the folder
folder_path = Path(folder_name)

# Create the folder/directory if it doesn't exist
folder_path.mkdir(parents=True, exist_ok=True)

file_path = "numerical_results/least_favorable_prior_ID_%g.png" % my_rand

plt.savefig(file_path)

