#### The code below stores the necessary functions for calculating the minimum-Chi square distance estimates #####

import numpy as np;
import scipy as sp;
import scipy.optimize as optimize;
from scipy.stats import poisson
from scipy.optimize import minimize;
from scipy.optimize import LinearConstraint;
import cvxpy as cp;
import matplotlib.pyplot as plt


################### Simple gradient descent ######################


def target_error(theta, mu, Xs_max):

    m = len(theta);

    logfact_temp = np.zeros(Xs_max + 1)

    for i in range(Xs_max + 1):
        logfact_temp[i] = sp.special.gammaln(i + 1)

    ret_bayes_part = np.zeros(Xs_max);

    logtheta = np.zeros(m);
    logtheta[theta > 0] = np.log(theta[theta > 0]);
    logtheta[theta == 0] = -1e100;

    for x in range(Xs_max):
        # density at x and x+1
        fdens_x = np.sum(mu * (np.exp(-theta + x * logtheta - logfact_temp[x])));
        fdens_x1 = np.sum(mu * (np.exp(-theta + (x+1) * logtheta - logfact_temp[x+1])));

        ret_bayes_part[x] = ((x + 1)**2) * (fdens_x1**2) / fdens_x;

    # So final "ret" computes E[{E(theta|X)}^2] using X ~ fdens_x and  E(theta|X) = (x + 1) * fdens_x1 / fdens_x \
    ret_bayes = np.sum(ret_bayes_part)

    # simplify E[{theta - E(theta | X)}^2] = E[theta^2] - E[{E(theta|X)}^2]
    ret_target = (np.sum(mu * (theta ** 2)) - ret_bayes)

    return ret_target;

# Jacobian computation wrt mu


def mu_grad_worst_prior(theta, mu, Xs_max):
    m = len(theta);

    logfact_temp = np.zeros(Xs_max + 1)

    for i in range(Xs_max + 1):
        logfact_temp[i] = sp.special.gammaln(i + 1)

    logtheta = np.zeros(m);
    logtheta[theta > 0] = np.log(theta[theta > 0]);
    logtheta[theta == 0] = -1e100;

    ret_bayes = np.zeros(m);

    # Marginal dist of X computation

    fdens = np.zeros(Xs_max+1);
    for x in range(Xs_max+1):
        # density at x
        fdens[x] = np.sum(mu * (np.exp(-theta + x * logtheta - logfact_temp[x])));

    # Jacobian of E[theta^2] = sum_j mu_j theta_j^2 wrt mu
    comp_e_theta2 = theta ** 2

    # Jacobian of E[{E(theta|X)}^2] = sum_x (x + 1)^2 * (fdens_x1)^2 / fdens_x wrt mu
    for j in range(m):
        comp_bayes_j = np.zeros(Xs_max);
        for y in range(Xs_max):

            fdens_prime_j_y = np.exp(-theta[j] + y * logtheta[j] - logfact_temp[y])

            fdens_prime_j_y1 = np.exp(-theta[j] + (y + 1) * logtheta[j] - logfact_temp[y + 1])

            comp_bayes_j[y] = (y + 1) ** 2 * (2 * fdens[y + 1] * fdens_prime_j_y1 * fdens[y] -
                                              fdens_prime_j_y * (fdens[y + 1]) ** 2) / ((fdens[y]) ** 2);

        ret_bayes[j] = np.sum(comp_bayes_j);

    ret_jacob = (comp_e_theta2 - ret_bayes)\
                # + 2 * 99999 * (np.sum(mu) - 1)

    return ret_jacob;


def mu_gradient_step_worst_prior(theta, mu, step, Xs_max):

    temp_grad = mu_grad_worst_prior(theta, mu, Xs_max)

    mu_new = mu + step * temp_grad/np.linalg.norm(temp_grad);
    mu_new = np.maximum(mu_new, 0);
    # mu_new = mu_new/np.sum(mu_new)

    return mu_new;


########################  merging roots and plotting worst distribution ###############


