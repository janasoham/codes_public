
from pathlib import Path
import numpy as np;
from functions_worst_prior import *;
from roots_and_preds import *;
from tqdm import tqdm
import matplotlib.pyplot as plt

# Computes the worst case prior with a given range of prior support and
# Xs_max denotes truncation for poisson expectation, var, etc., approximations


my_rand = np.random.rand(1)

# Define the folder name
folder_name = 'numerical_results'

# Create a Path object for the folder
folder_path = Path(folder_name)

# Create the folder/directory if it doesn't exist
folder_path.mkdir(parents=True, exist_ok=True)

def worst_prior(Xs_max, grid):

    # Initialize support of the worst prior
    theta = np.copy(grid);
    theta_supp = np.copy(theta);
    # plt.hist(theta_worst_prior, density=True)

    # Initialize mass atoms of the worst prior
    mu0 = np.random.rand(len(grid));

    # mu0 = np.ones(grid_size)
    mu0 = mu0/np.sum(mu0);
    mu_curr = np.copy(mu0);

    mu_next = mu_gradient_step_worst_prior(theta_supp, mu_curr, 1e-4, Xs_max)
    mu_next = mu_next/sum(mu_next)

    tot_iter = 2000

    all_target_vals = []

    for iter_temp in tqdm(range(tot_iter)):

        mu_curr_temp = np.copy(mu_next)

        theta_supp = theta_supp[mu_curr_temp > 1e-20]

        mu_curr = mu_curr_temp[mu_curr_temp > 1e-20]

        mu_curr = mu_curr/np.sum(mu_curr)

        step = 0.001

        mu_next = mu_gradient_step_worst_prior(theta_supp, mu_curr, step, Xs_max)

        mu_next = mu_next / sum(mu_next)

        temp_target_val = target_error(theta_supp, mu_next, Xs_max)

        all_target_vals.append(temp_target_val)

        if iter_temp % 5 == 0:

        #     ####### Plotting starts here ################
        #
            iter_temp_vec = list(range(1, iter_temp + 2))

            iter_temp_ticks = [int(xi) for xi in list(range(1, iter_temp + 2, 3))]

            plt.plot(figsize=(8, 5))

            plt.plot(iter_temp_vec, all_target_vals)

            plt.xticks(iter_temp_ticks)

            plt.xlabel('iteration')
            plt.ylabel('objective function value')
            plt.title('objective value till iteration %g ID: %g' % (iter_temp+1, my_rand))

            plt.savefig("numerical_results/target_plot_ID_%g.png" % my_rand)

            plt.clf()

            plt.hist(theta_supp, bins=50, weights=mu_next, color='blue', edgecolor='black')
            plt.xlabel('support')
            plt.ylabel('probs')
            plt.title('worst case distribution at iteration %g ID_%g' % (iter_temp+1, my_rand))

            plt.savefig("numerical_results/least_favorable_prior_ID_%g.png" % my_rand)

            plt.clf()


            ############### Saving priors #######################

            columns = ['theta', 'mu']
            df = pd.DataFrame(columns=columns)
            df['theta'] = theta_supp
            df['mu'] = mu_next

            # Open a text file in write mode
            file_path = "numerical_results/least_fav_prior_%g.csv" % my_rand
            df.to_csv(file_path, index=False)

            ############### Saving priors #######################

            columns = ['iteration', 'target']
            df_target = pd.DataFrame(columns=columns)
            df_target['iteration'] = range(iter_temp+1)
            df_target['target'] = all_target_vals

            # Open a text file in write mode
            file_path = "numerical_results/target_values_%g.csv" % my_rand
            df_target.to_csv(file_path, index=False)

    support = np.copy(theta_supp);

    probs = np.copy(mu_next);

    final_target = temp_target_val.copy()

    return final_target, support, probs;



