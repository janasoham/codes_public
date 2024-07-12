import numpy as np;
from functions_worst_prior import *;
from roots_and_preds import *;
from tqdm import tqdm
import matplotlib.pyplot as plt

# Computes the worst case prior with a given range of prior support and
# Xs_max denotes truncation for poisson expectation, var, etc., approximations


my_rand = np.random.rand(1)


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

    tot_iter = 500

    change_val = []

    grad_max = []

    grad =[]

    change_mu = []

    for iter_temp in tqdm(range(tot_iter)):

        mu_prev = np.copy(mu_curr)

        mu_curr = np.copy(mu_next)

        # nab_curr = mu_grad_worst_prior(theta_supp, mu_curr, Xs_max)
        # nab_prev = mu_grad_worst_prior(theta_supp, mu_prev, Xs_max)
        #
        # denom = np.sum(nab_curr - nab_prev) ** 2
        #
        # numer = abs(np.sum((mu_curr - mu_prev) * (nab_curr - nab_prev)))
        #
        # step_factor = 1
        #
        # step = step_factor * (numer/(denom + 1e-30));

        step = 0.01

        mu_next = mu_gradient_step_worst_prior(theta_supp, mu_curr, step, Xs_max)
        # (theta_worst_prior, mu0_worst_prior) = prune_roots(theta_worst_prior, mu0_worst_prior);

        mu_next = mu_next / sum(mu_next)

        change_val.append(target_error(theta_supp, mu_next, Xs_max))

        grad_max.append(np.max(abs(mu_grad_worst_prior(theta_supp, mu_next, Xs_max))))

        grad.append(mu_grad_worst_prior(theta_supp, mu_next, Xs_max))

        change_mu.append(np.linalg.norm(mu_next - mu_curr))

        if iter_temp % 5 == 0:
        #
        #     ####### Plotting starts here ################
        #
            iter_temp_vec = list(range(1, iter_temp + 2))
        #
            iter_temp_ticks = [int(xi) for xi in list(range(1, iter_temp + 2, 3))]
        #
            plt.plot(figsize=(8,5))
        #
            plt.plot(iter_temp_vec, change_val)

            plt.xticks(iter_temp_ticks)

            plt.xlabel('iteration')
            plt.ylabel('objective function value')
            plt.title('objective value till iteration %g ID: %g' % (iter_temp+1, my_rand))

            plt.savefig("target_plot_ID_%g.png" % my_rand)
        #
        #     plt.clf()
        #
        #     plt.hist(theta_supp, bins=50, weights=mu_next, color='blue', edgecolor='black')
        #     plt.xlabel('support')
        #     plt.ylabel('probs')
        #     plt.title('worst case distribution at iteration %g ID_%g' % (iter_temp+1, my_rand))
        #
        #     plt.savefig("least_favorable_prior_ID_%g.png" % my_rand)
        #
        #     plt.clf()
        #
        #     plt.plot(iter_temp_vec, change_mu)
        #
        #     plt.xticks(iter_temp_ticks)
        #
        #     plt.xlabel('iteration')
        #     plt.ylabel('norm of change')
        #     plt.title('Normed difference for mu in subsequent iterations ID_%g' % my_rand)
        #
        #     plt.savefig("mu_change_plot_ID_%g.png" % my_rand)
        #
        #     plt.clf()
        #
        #     plt.plot(iter_temp_vec, grad_max)
        #
        #     plt.xticks(iter_temp_ticks)
        #
        #     plt.xlabel('iteration')
        #     plt.ylabel('max absolute grad')
        #     plt.title('Maximum of absolute gradient vector ID_%g' % my_rand)
        #
        #     plt.savefig("max_grad_abs_plot_ID_%g.png" % my_rand)


            # ############### Saving priors #######################
            #
            # columns = ['theta', 'mu']
            # df = pd.DataFrame(columns=columns)
            # df['theta'] = theta_supp
            # df['mu'] = mu_next
            #
            # # Open a text file in write mode
            # file_path = "least_fav_prior_%g.csv" % my_rand
            # df.to_csv(file_path, index=False)
            # # The file is automatically closed when the 'with' block exits

            # ############### Saving gradient max #######################
            #
            # df_grad_max = pd.DataFrame()
            # df_grad_max['grad_max'] = grad_max
            #
            # # Open a text file in write mode
            # file_path = "grad_max_ID_%g.csv" % my_rand
            # df_grad_max.to_csv(file_path, index=False)
            # # The file is automatically closed when the 'with' block exits

            # ############### Saving gradient #######################
            #
            # df_grad = pd.DataFrame()
            # df_grad['grad'] = grad
            #
            # # Open a text file in write mode
            # file_path = "grad_ID_%g.csv" % my_rand
            # df_grad.to_csv(file_path, index=False)
            # # The file is automatically closed when the 'with' block exits
            #
            # ############### Saving mu change #######################
            #
            # columns_change_mu = ['change_mu']
            # df_change_mu = pd.DataFrame(columns=columns_change_mu)
            # df_change_mu['change_mu'] = change_mu
            #
            # # Open a text file in write mode
            # file_path = "change_mu_ID_%g.csv" % my_rand
            # df_change_mu.to_csv(file_path, index=False)
            # # The file is automatically closed when the 'with' block exits

    support = np.copy(theta_supp);

    probs = np.copy(mu_next);

    # print("final grad max value to check for convergence")
    #
    # print(grad_max[tot_iter-1])
    #
    # print("Changes in consequent steps")
    #
    # print(np.max(abs(mu_next-mu_curr)))
    #
    # print("final target value")
    # print(change_val[tot_iter-1])

    final_target = change_val[tot_iter-1]

    return final_target, support, probs;



